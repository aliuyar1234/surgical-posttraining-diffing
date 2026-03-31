from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.modeling import load_causal_model, load_tokenizer
from src.common.runmeta import collect_runtime_facts
from src.eval.common import (
    answer_token_kl,
    build_full_delta_interventions,
    build_mean_diff_interventions,
    forward_with_interventions,
    load_mean_diff_vectors,
    load_completion_rows,
    load_sparse_checkpoint,
    teacher_forced_inputs_from_row,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate FullDelta scalar gates on select_tune answer-token KL.")
    parser.add_argument("--config", required=True, help="Path to interventions config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_gate_calibration(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_gate_calibration(config: dict[str, Any]) -> dict[str, Any]:
    data_config = load_yaml_config(config["data_config_path"])
    model_config = load_yaml_config(config["model_config_path"])
    rows = load_completion_rows(
        completion_dir=model_config["paths"]["completion_dir"],
        completion_run_id=config["completion_run_id"],
        splits=[config["gate_search"]["split"]],
        slices=list(config["gate_search"].get("slices", data_config["slices"])),
    )
    checkpoints = [load_sparse_checkpoint(path) for path in config["checkpoint_paths"]]
    checkpoints.sort(key=lambda item: item["layer_index"])
    intervention_kind = str(config.get("intervention_kind", "full_delta"))
    mean_deltas = (
        load_mean_diff_vectors(config["mean_diff_cache_summary_path"]) if intervention_kind == "mean_diff" else None
    )
    run_payload = {
        "config": config,
        "checkpoint_paths": config["checkpoint_paths"],
        "completion_run_id": config["completion_run_id"],
        "record_count": len(rows),
    }
    run_id = build_run_id(config["stage_name"], run_payload)
    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")
    save_resolved_config_snapshot({"config": config, "data_config": data_config, "model_config": model_config}, snapshot_path)

    tokenizer = load_tokenizer(model_config["model_pair"]["it_path"])
    pt_model = load_causal_model(
        model_config["model_pair"]["pt_path"],
        device=config["runtime"]["device"],
        dtype_name=config["runtime"]["dtype"],
    )
    it_model = load_causal_model(
        model_config["model_pair"]["it_path"],
        device=config["runtime"]["device"],
        dtype_name=config["runtime"]["dtype"],
    )

    search_log: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    alphas = {checkpoint["layer_index"]: 1.0 for checkpoint in checkpoints}
    current_score = evaluate_gate_pair(
        rows=rows,
        tokenizer=tokenizer,
        pt_model=pt_model,
        it_model=it_model,
        checkpoints=checkpoints,
        alphas=alphas,
        device=config["runtime"]["device"],
        intervention_kind=intervention_kind,
        mean_deltas=mean_deltas,
    )
    search_log.append({"step": "initial", "alphas": dict(alphas), "kl": current_score})

    base_grid = [float(value) for value in config["gate_search"]["grid"]]
    boundary_expansion = [float(value) for value in config["gate_search"]["boundary_expansion"]]
    for checkpoint in checkpoints:
        layer_index = checkpoint["layer_index"]
        best_alpha = alphas[layer_index]
        best_score = current_score
        for candidate in base_grid:
            trial_alphas = dict(alphas)
            trial_alphas[layer_index] = candidate
            score = evaluate_gate_pair(
                rows=rows,
                tokenizer=tokenizer,
                pt_model=pt_model,
                it_model=it_model,
                checkpoints=checkpoints,
                alphas=trial_alphas,
                device=config["runtime"]["device"],
                intervention_kind=intervention_kind,
                mean_deltas=mean_deltas,
            )
            search_log.append({"step": f"grid_layer_{layer_index}", "alphas": trial_alphas, "kl": score})
            if score < best_score:
                best_score = score
                best_alpha = candidate

        expansion_candidates: list[float] = []
        if best_alpha == min(base_grid):
            expansion_candidates = [min(boundary_expansion)]
        elif best_alpha == max(base_grid):
            expansion_candidates = [max(boundary_expansion)]
        for candidate in expansion_candidates:
            trial_alphas = dict(alphas)
            trial_alphas[layer_index] = candidate
            score = evaluate_gate_pair(
                rows=rows,
                tokenizer=tokenizer,
                pt_model=pt_model,
                it_model=it_model,
                checkpoints=checkpoints,
                alphas=trial_alphas,
                device=config["runtime"]["device"],
                intervention_kind=intervention_kind,
                mean_deltas=mean_deltas,
            )
            search_log.append({"step": f"boundary_layer_{layer_index}", "alphas": trial_alphas, "kl": score})
            if score < best_score:
                best_score = score
                best_alpha = candidate

        alphas[layer_index] = best_alpha
        current_score = best_score

    gate_dir = Path(config["paths"]["gate_dir"])
    gate_dir.mkdir(parents=True, exist_ok=True)
    gates_path = gate_dir / f"gates_{run_id}.json"
    gates_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "objective": config["gate_search"]["objective"],
        "intervention_kind": intervention_kind,
        "split": config["gate_search"]["split"],
        "checkpoint_paths": config["checkpoint_paths"],
        "alphas": {str(layer): alpha for layer, alpha in alphas.items()},
        "best_kl": current_score,
        "search_log": search_log,
        "snapshot_path": snapshot_path.as_posix(),
    }
    gates_path.write_text(json.dumps(gates_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")
    runtime_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "split": config["gate_search"]["split"],
        "record_count": len(rows),
        "device": config["runtime"]["device"],
        "wall_clock_seconds": time.perf_counter() - start_time,
        "snapshot_path": snapshot_path.as_posix(),
        "gates_path": gates_path.as_posix(),
        "environment": collect_runtime_facts(),
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "alphas": alphas,
        "best_kl": current_score,
        "gates_path": gates_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
    }


def evaluate_gate_pair(
    *,
    rows: list[dict[str, Any]],
    tokenizer,
    pt_model,
    it_model,
    checkpoints: list[dict[str, Any]],
    alphas: dict[int, float],
    device: str,
    intervention_kind: str,
    mean_deltas: dict[int, torch.Tensor] | None,
) -> float:
    kl_sum = 0.0
    token_count = 0
    for row in rows:
        input_ids, attention_mask, answer_start = teacher_forced_inputs_from_row(tokenizer, row)
        model_inputs = {
            "input_ids": input_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
            "use_cache": False,
        }
        with torch.no_grad():
            it_logits = it_model(**model_inputs).logits

        position_mask = torch.zeros(int(input_ids.shape[0]), dtype=torch.float32)
        position_mask[answer_start:] = 1.0
        interventions = build_interventions_for_kind(
            intervention_kind=intervention_kind,
            checkpoints=checkpoints,
            alphas=alphas,
            position_mask=position_mask,
            incremental_only=False,
            mean_deltas=mean_deltas,
        )
        variant_logits = forward_with_interventions(pt_model, interventions, model_inputs).logits
        example_kl, example_tokens = answer_token_kl(it_logits, variant_logits, answer_start=answer_start)
        kl_sum += example_kl
        token_count += example_tokens
    if token_count == 0:
        return 0.0
    return kl_sum / token_count


def build_interventions_for_kind(
    *,
    intervention_kind: str,
    checkpoints: list[dict[str, Any]],
    alphas: dict[int, float],
    position_mask: torch.Tensor | None,
    incremental_only: bool,
    mean_deltas: dict[int, torch.Tensor] | None,
) -> dict[int, Any]:
    if intervention_kind == "full_delta":
        return build_full_delta_interventions(
            checkpoints=checkpoints,
            alphas=alphas,
            position_mask=position_mask,
            incremental_only=incremental_only,
        )
    if intervention_kind == "mean_diff":
        if mean_deltas is None:
            raise ValueError("mean_diff calibration requires mean_deltas to be loaded from cache summary")
        return build_mean_diff_interventions(
            mean_deltas=mean_deltas,
            alphas=alphas,
            position_mask=position_mask,
            incremental_only=incremental_only,
        )
    raise ValueError(f"Unsupported intervention_kind: {intervention_kind}")


if __name__ == "__main__":
    raise SystemExit(main())
