from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.analysis.common import FEATURE_SUMMARY_COLUMNS, build_feature_table_run_payload, candidate_feature_key
from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.modeling import load_causal_model, load_tokenizer
from src.common.runmeta import collect_runtime_facts
from src.eval.common import (
    add_recovery_metrics,
    aggregate_variant_metrics,
    build_full_delta_interventions,
    greedy_generate_batch,
    load_completion_rows,
    load_prompt_map,
    load_sparse_checkpoint,
    max_new_tokens_by_slice,
    score_generation,
    teacher_forced_inputs_from_row,
)
from src.train.intervention import replace_hidden_in_layer_output
from src.train.sparse_delta import standardize_hidden


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the M5 feature table and candidate pool.")
    parser.add_argument("--config", required=True, help="Path to selectors config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_build_feature_table(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_build_feature_table(config: dict[str, Any]) -> dict[str, Any]:
    data_config = load_yaml_config(config["data_config_path"])
    model_config = load_yaml_config(config["model_config_path"])
    rows = load_completion_rows(
        completion_dir=model_config["paths"]["completion_dir"],
        completion_run_id=config["completion_run_id"],
        splits=list(config["feature_splits"]),
        slices=list(config.get("eval_slices", data_config["slices"])),
    )
    prompt_map = load_prompt_map(split_manifest_dir=data_config["paths"]["split_manifest_dir"], slices=data_config["slices"])
    checkpoints = [load_sparse_checkpoint(path) for path in config["checkpoint_paths"]]
    checkpoints.sort(key=lambda item: item["layer_index"])
    gates_payload = json.loads(Path(config["gates_path"]).read_text(encoding="utf-8"))
    alphas = {int(layer): float(alpha) for layer, alpha in gates_payload["alphas"].items()}

    run_payload = build_feature_table_run_payload(config, record_count=len(rows))
    run_id = build_run_id("build_feature_table", run_payload)
    summary_path = build_artifact_path("selector", "feature_table_summary", run_id, ".json")
    candidate_table_path = build_artifact_path("selector", "candidate_feature_table", run_id, ".parquet")
    candidate_rows_path = build_artifact_path("selector", "candidate_feature_rows", run_id, ".parquet")
    generation_examples_path = build_artifact_path("selector", "selection_generation_examples", run_id, ".jsonl")
    generation_metrics_path = build_artifact_path("selector", "selection_generation_metrics", run_id, ".json")
    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")
    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")

    save_resolved_config_snapshot(
        {"config": config, "data_config": data_config, "model_config": model_config, "gates": gates_payload},
        snapshot_path,
    )

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

    for checkpoint in checkpoints:
        checkpoint["module"] = checkpoint["module"].to(pt_model.device)
        checkpoint["module"].eval()

    start_time = time.perf_counter()
    prompt_summaries = extract_prompt_feature_summaries(
        rows=rows,
        tokenizer=tokenizer,
        pt_model=pt_model,
        checkpoints=checkpoints,
        alphas=alphas,
        device=config["runtime"]["device"],
    )
    candidate_rows = build_candidate_table(
        prompt_summaries=prompt_summaries,
        candidate_pool_size=int(config["candidate_pool_size"]),
        slices=list(config.get("eval_slices", data_config["slices"])),
    )
    candidate_keys = {candidate_feature_key(row["layer"], row["feature_id"]) for row in candidate_rows}
    candidate_feature_rows = materialize_candidate_feature_rows(prompt_summaries, candidate_keys)
    generation_examples, generation_metrics = generate_selection_examples(
        rows=rows,
        prompt_map=prompt_map,
        tokenizer=tokenizer,
        pt_model=pt_model,
        it_model=it_model,
        checkpoints=checkpoints,
        alphas=alphas,
        model_config=model_config,
        variant_names=list(config["variants"]),
        generation_batch_size=int(config["generation"]["batch_size"]),
        slices=list(config.get("eval_slices", data_config["slices"])),
    )

    write_parquet_rows(candidate_table_path, candidate_rows)
    write_parquet_rows(candidate_rows_path, candidate_feature_rows)
    write_jsonl_rows(generation_examples_path, generation_examples)
    generation_metrics_path.write_text(json.dumps(generation_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_payload = {
        "run_id": run_id,
        "stage_name": "build_feature_table",
        "completion_run_id": config["completion_run_id"],
        "feature_splits": list(config["feature_splits"]),
        "candidate_scoring_split": "select_train",
        "candidate_pool_size": int(config["candidate_pool_size"]),
        "feature_summary_columns": list(FEATURE_SUMMARY_COLUMNS),
        "checkpoint_paths": config["checkpoint_paths"],
        "gates_path": config["gates_path"],
        "top_candidate_preview": candidate_rows[:12],
        "feature_table_split_usage": {
            "feature_summary_splits": list(config["feature_splits"]),
            "candidate_scoring_splits": ["select_train"],
            "test_split_touched": any(row["split"] == "test" for row in rows),
        },
        "summary_path": summary_path.as_posix(),
        "candidate_table_path": candidate_table_path.as_posix(),
        "candidate_rows_path": candidate_rows_path.as_posix(),
        "generation_examples_path": generation_examples_path.as_posix(),
        "generation_metrics_path": generation_metrics_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_payload = {
        "run_id": run_id,
        "stage_name": "build_feature_table",
        "record_count": len(rows),
        "candidate_pool_size": int(config["candidate_pool_size"]),
        "device": config["runtime"]["device"],
        "wall_clock_seconds": time.perf_counter() - start_time,
        "summary_path": summary_path.as_posix(),
        "candidate_table_path": candidate_table_path.as_posix(),
        "candidate_rows_path": candidate_rows_path.as_posix(),
        "generation_examples_path": generation_examples_path.as_posix(),
        "generation_metrics_path": generation_metrics_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
        "environment": collect_runtime_facts(),
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "candidate_table_path": candidate_table_path.as_posix(),
        "candidate_rows_path": candidate_rows_path.as_posix(),
        "generation_examples_path": generation_examples_path.as_posix(),
        "generation_metrics_path": generation_metrics_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "summary_path": summary_path.as_posix(),
    }


def extract_prompt_feature_summaries(
    *,
    rows: list[dict[str, Any]],
    tokenizer,
    pt_model,
    checkpoints: list[dict[str, Any]],
    alphas: dict[int, float],
    device: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows:
        input_ids, attention_mask, answer_start = teacher_forced_inputs_from_row(tokenizer, row)
        model_inputs = {
            "input_ids": input_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
            "use_cache": False,
        }
        by_layer = capture_full_delta_feature_summaries(
            model=pt_model,
            checkpoints=checkpoints,
            alphas=alphas,
            model_inputs=model_inputs,
            answer_start=answer_start,
        )
        summaries.append(
            {
                "prompt_id": row["prompt_id"],
                "split": row["split"],
                "slice": row["slice"],
                "layers": by_layer,
            }
        )
    return summaries


def capture_full_delta_feature_summaries(
    *,
    model,
    checkpoints: list[dict[str, Any]],
    alphas: dict[int, float],
    model_inputs: dict[str, Any],
    answer_start: int,
) -> dict[int, dict[str, torch.Tensor]]:
    layers = model.language_model.layers if hasattr(model, "language_model") else model.model.layers
    by_layer: dict[int, dict[str, torch.Tensor]] = {}

    def make_hook(checkpoint: dict[str, Any]):
        layer_index = int(checkpoint["layer_index"])
        module = checkpoint["module"]
        input_mean = checkpoint["input_mean"].to(model_inputs["input_ids"].device)
        input_std = checkpoint["input_std"].to(model_inputs["input_ids"].device)
        eps_std = float(checkpoint["eps_std"])
        alpha = float(alphas[layer_index])
        decoder_norms = checkpoint["decoder_column_norms"].to(model_inputs["input_ids"].device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            standardized = standardize_hidden(hidden.to(torch.float32), input_mean, input_std, eps=eps_std)
            standardized = standardized.to(device=module.encoder.weight.device, dtype=module.encoder.weight.dtype)
            outputs = module(standardized)
            dense_features = outputs.features.to(torch.float32)
            dense_mass = dense_features * decoder_norms.view(1, 1, -1) * alpha
            by_layer[layer_index] = summarize_dense_feature_tensors(
                dense_features[0, answer_start:, :],
                dense_mass[0, answer_start:, :],
            )

            delta = (alpha * outputs.delta_hat).to(device=hidden.device, dtype=hidden.dtype)
            if answer_start > 0:
                delta = delta.clone()
                delta[:, :answer_start, :] = 0
            updated_hidden = hidden + delta
            return replace_hidden_in_layer_output(output, updated_hidden)

        return hook

    with ExitStack() as stack:
        for checkpoint in checkpoints:
            handle = layers[int(checkpoint["layer_index"])].register_forward_hook(make_hook(checkpoint))
            stack.callback(handle.remove)
        with torch.no_grad():
            model(**model_inputs)
    return by_layer


def summarize_dense_feature_tensors(feature_values: torch.Tensor, contribution_values: torch.Tensor) -> dict[str, torch.Tensor]:
    if feature_values.ndim != 2 or contribution_values.ndim != 2:
        raise ValueError("Expected 2D [answer_tokens, width] feature tensors")
    if feature_values.shape != contribution_values.shape:
        raise ValueError("Feature and contribution tensors must match exactly")
    width = int(feature_values.shape[1])
    if feature_values.shape[0] == 0:
        zero = torch.zeros(width, dtype=torch.float32)
        return {name: zero.clone() for name in FEATURE_SUMMARY_COLUMNS}
    return {
        "max_answer": feature_values.max(dim=0).values.detach().cpu().to(torch.float32),
        "mean_answer": feature_values.mean(dim=0).detach().cpu().to(torch.float32),
        "last_answer": feature_values[-1].detach().cpu().to(torch.float32),
        "mean_contribution_norm": contribution_values.mean(dim=0).detach().cpu().to(torch.float32),
    }


def build_candidate_table(
    *,
    prompt_summaries: list[dict[str, Any]],
    candidate_pool_size: int,
    slices: list[str],
) -> list[dict[str, Any]]:
    select_rows = [row for row in prompt_summaries if row["split"] == "select_train"]
    if not select_rows:
        raise ValueError("Feature table requires select_train rows")

    layer_widths = {
        int(layer): int(next(iter(select_rows[0]["layers"][layer].values())).shape[0]) for layer in select_rows[0]["layers"]
    }
    mass_sums = {layer: torch.zeros(width, dtype=torch.float32) for layer, width in layer_widths.items()}
    slice_mean_sums = {
        slice_name: {layer: torch.zeros(width, dtype=torch.float32) for layer, width in layer_widths.items()}
        for slice_name in slices
    }
    slice_counts = {slice_name: 0 for slice_name in slices}

    for row in select_rows:
        slice_name = row["slice"]
        slice_counts[slice_name] += 1
        for layer, summaries in row["layers"].items():
            mass_sums[int(layer)] += summaries["mean_contribution_norm"]
            slice_mean_sums[slice_name][int(layer)] += summaries["mean_answer"]

    total_prompts = len(select_rows)
    candidate_rows: list[dict[str, Any]] = []
    for layer in sorted(layer_widths):
        mean_mass = mass_sums[layer] / total_prompts
        per_slice_means = []
        for slice_name in slices:
            divisor = max(slice_counts[slice_name], 1)
            per_slice_means.append(slice_mean_sums[slice_name][layer] / divisor)
        slice_mean_tensor = torch.stack(per_slice_means, dim=0)
        slice_variance = slice_mean_tensor.var(dim=0, unbiased=False)
        candidate_score = mean_mass * slice_variance

        for feature_id in range(layer_widths[layer]):
            row = {
                "candidate_key": candidate_feature_key(layer, feature_id),
                "layer": layer,
                "feature_id": feature_id,
                "mass": float(mean_mass[feature_id].item()),
                "slice_variance": float(slice_variance[feature_id].item()),
                "candidate_score": float(candidate_score[feature_id].item()),
            }
            for slice_index, slice_name in enumerate(slices):
                row[f"slice_mean_answer_{slice_name}"] = float(slice_mean_tensor[slice_index, feature_id].item())
            candidate_rows.append(row)

    candidate_rows.sort(key=lambda item: (-item["candidate_score"], item["layer"], item["feature_id"]))
    return candidate_rows[: min(candidate_pool_size, len(candidate_rows))]


def materialize_candidate_feature_rows(
    prompt_summaries: list[dict[str, Any]],
    candidate_keys: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_row in prompt_summaries:
        for layer, summaries in prompt_row["layers"].items():
            width = int(next(iter(summaries.values())).shape[0])
            for feature_id in range(width):
                candidate_key = candidate_feature_key(int(layer), feature_id)
                if candidate_key not in candidate_keys:
                    continue
                row = {
                    "candidate_key": candidate_key,
                    "prompt_id": prompt_row["prompt_id"],
                    "split": prompt_row["split"],
                    "slice": prompt_row["slice"],
                    "layer": int(layer),
                    "feature_id": int(feature_id),
                }
                for summary_name in FEATURE_SUMMARY_COLUMNS:
                    row[summary_name] = float(summaries[summary_name][feature_id].item())
                rows.append(row)
    rows.sort(key=lambda item: (item["split"], item["prompt_id"], item["layer"], item["feature_id"]))
    return rows


def generate_selection_examples(
    *,
    rows: list[dict[str, Any]],
    prompt_map: dict[str, dict[str, Any]],
    tokenizer,
    pt_model,
    it_model,
    checkpoints: list[dict[str, Any]],
    alphas: dict[int, float],
    model_config: dict[str, Any],
    variant_names: list[str],
    generation_batch_size: int,
    slices: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    metrics_by_split: dict[str, dict[str, dict[str, float]]] = {}
    max_new_tokens = max_new_tokens_by_slice(model_config)
    full_delta_interventions = build_full_delta_interventions(
        checkpoints=checkpoints,
        alphas=alphas,
        position_mask=None,
        incremental_only=True,
    )

    for split in sorted({row["split"] for row in rows}):
        split_examples: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variant_names}
        split_rows = [row for row in rows if row["split"] == split]
        for variant_name in variant_names:
            for slice_name in slices:
                slice_rows = [row for row in split_rows if row["slice"] == slice_name]
                for batch_start in range(0, len(slice_rows), generation_batch_size):
                    batch_rows = slice_rows[batch_start : batch_start + generation_batch_size]
                    rendered_prefixes = [row["rendered_prefix"] for row in batch_rows]
                    if variant_name == "PT":
                        generations = greedy_generate_batch(
                            model=pt_model,
                            tokenizer=tokenizer,
                            rendered_prefixes=rendered_prefixes,
                            max_new_tokens=max_new_tokens[slice_name],
                            interventions=None,
                        )
                    elif variant_name == "IT_neutral":
                        generations = greedy_generate_batch(
                            model=it_model,
                            tokenizer=tokenizer,
                            rendered_prefixes=rendered_prefixes,
                            max_new_tokens=max_new_tokens[slice_name],
                            interventions=None,
                        )
                    elif variant_name == "PT_plus_FullDelta":
                        generations = greedy_generate_batch(
                            model=pt_model,
                            tokenizer=tokenizer,
                            rendered_prefixes=rendered_prefixes,
                            max_new_tokens=max_new_tokens[slice_name],
                            interventions=full_delta_interventions,
                        )
                    else:
                        raise ValueError(f"Unsupported selection variant: {variant_name}")

                    for row, generation in zip(batch_rows, generations, strict=True):
                        prompt_row = prompt_map[row["prompt_id"]]
                        scored = score_generation(prompt_row, generation["completion_text"], generation["completion_token_ids"])
                        example = {
                            "variant": variant_name,
                            "prompt_id": row["prompt_id"],
                            "split": split,
                            "slice": row["slice"],
                            "completion_text": generation["completion_text"],
                            "completion_token_ids": generation["completion_token_ids"],
                            "stop_reason": generation["stop_reason"],
                            "eos_reached": generation["eos_reached"],
                            **scored,
                        }
                        split_examples[variant_name].append(example)
                        examples.append(example)

        split_metrics = {variant: aggregate_variant_metrics(split_examples[variant]) for variant in variant_names}
        add_recovery_metrics(split_metrics)
        metrics_by_split[split] = split_metrics

    return examples, metrics_by_split


def write_parquet_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)


def write_jsonl_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
