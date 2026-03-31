from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.modeling import load_causal_model, load_tokenizer
from src.common.runmeta import collect_runtime_facts
from src.eval.bootstrap import build_bootstrap_summary
from src.eval.common import (
    add_recovery_metrics,
    aggregate_variant_metrics,
    answer_token_kl,
    build_full_delta_minus_mask_interventions,
    build_full_delta_interventions,
    build_masked_interventions,
    build_mean_diff_interventions,
    capture_hidden_states,
    forward_with_interventions,
    greedy_generate_batch,
    load_gate_alphas,
    load_mask_payload,
    load_mean_diff_vectors,
    load_completion_rows,
    load_prompt_map,
    load_sparse_checkpoint,
    max_new_tokens_by_slice,
    score_generation,
    teacher_forced_inputs_from_row,
)
from src.train.sparse_delta import compute_r2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M4 fidelity evaluation.")
    parser.add_argument("--config", required=True, help="Path to eval fidelity config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_fidelity_eval(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_fidelity_eval(config: dict[str, Any]) -> dict[str, Any]:
    data_config = load_yaml_config(config["data_config_path"])
    model_config = load_yaml_config(config["model_config_path"])
    completion_rows = load_completion_rows(
        completion_dir=model_config["paths"]["completion_dir"],
        completion_run_id=config["completion_run_id"],
        splits=[config["fidelity_split"]],
        slices=list(config.get("eval_slices", data_config["slices"])),
    )
    prompt_map = load_prompt_map(split_manifest_dir=data_config["paths"]["split_manifest_dir"], slices=data_config["slices"])
    checkpoints = [load_sparse_checkpoint(path) for path in config["checkpoint_paths"]]
    checkpoints.sort(key=lambda item: item["layer_index"])
    gates_payload = json.loads(Path(config["gates_path"]).read_text(encoding="utf-8"))
    alphas = load_gate_alphas(config["gates_path"])
    variant_specs = normalize_variant_specs(config["variants"])
    variant_resources = prepare_variant_resources(
        variant_specs,
        checkpoints=checkpoints,
        full_delta_alphas=alphas,
        default_gates_path=config["gates_path"],
    )

    run_payload = {
        "config": config,
        "completion_run_id": config["completion_run_id"],
        "checkpoint_paths": config["checkpoint_paths"],
        "gates_path": config["gates_path"],
        "record_count": len(completion_rows),
    }
    run_id = build_run_id(config["stage_name"], run_payload)
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

    start_time = time.perf_counter()
    teacher_forced_metrics, teacher_forced_example_rows = evaluate_teacher_forced_fidelity(
        rows=completion_rows,
        tokenizer=tokenizer,
        pt_model=pt_model,
        it_model=it_model,
        checkpoints=checkpoints,
        variant_resources=variant_resources,
        device=config["runtime"]["device"],
    )
    generation_metrics, generation_example_rows = evaluate_free_generation(
        rows=completion_rows,
        prompt_map=prompt_map,
        tokenizer=tokenizer,
        pt_model=pt_model,
        it_model=it_model,
        checkpoints=checkpoints,
        variant_resources=variant_resources,
        generation_batch_size=int(config["generation"]["batch_size"]),
        model_config=model_config,
    )
    add_recovery_metrics(generation_metrics)
    c1_assessment = assess_c1_support(teacher_forced_metrics, generation_metrics)
    bootstrap_payload = None
    if "bootstrap" in config:
        bootstrap_payload = build_bootstrap_summary(
            generation_rows=generation_example_rows,
            teacher_forced_rows=teacher_forced_example_rows,
            comparisons=config["bootstrap"].get("comparisons"),
            resamples=int(config["bootstrap"]["resamples"]),
            seed=int(config["bootstrap"]["seed"]),
        )

    metric_path = build_artifact_path("metric", "fidelity", run_id, ".json")
    bootstrap_path = build_artifact_path("metric", "bootstrap_cis", run_id, ".json")
    table_path = build_artifact_path("table", "fidelity_table", run_id, ".md")
    figure_path = build_artifact_path("figure", "fidelity_summary", run_id, ".svg")
    examples_path = build_artifact_path("example", "generation_examples", run_id, ".jsonl")
    teacher_forced_examples_path = build_artifact_path("example", "teacher_forced_prompt_metrics", run_id, ".jsonl")
    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")

    metric_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "fidelity_split": config["fidelity_split"],
        "checkpoint_paths": config["checkpoint_paths"],
        "gates_path": config["gates_path"],
        "teacher_forced": teacher_forced_metrics,
        "generation_metrics": generation_metrics,
        "c1_assessment": c1_assessment,
        "snapshot_path": snapshot_path.as_posix(),
        "metric_path": metric_path.as_posix(),
        "bootstrap_path": bootstrap_path.as_posix() if bootstrap_payload is not None else None,
        "table_path": table_path.as_posix(),
        "figure_path": figure_path.as_posix(),
        "examples_path": examples_path.as_posix(),
        "teacher_forced_examples_path": teacher_forced_examples_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
    }
    metric_path.write_text(json.dumps(metric_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if bootstrap_payload is not None:
        bootstrap_path.write_text(json.dumps(bootstrap_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_path.write_text(build_markdown_table(teacher_forced_metrics, generation_metrics, c1_assessment), encoding="utf-8")
    figure_path.write_text(build_svg_summary(teacher_forced_metrics, generation_metrics), encoding="utf-8")
    write_jsonl_like(examples_path, generation_example_rows)
    write_jsonl_like(teacher_forced_examples_path, teacher_forced_example_rows)

    runtime_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "fidelity_split": config["fidelity_split"],
        "record_count": len(completion_rows),
        "device": config["runtime"]["device"],
        "wall_clock_seconds": time.perf_counter() - start_time,
        "snapshot_path": snapshot_path.as_posix(),
        "metric_path": metric_path.as_posix(),
        "bootstrap_path": bootstrap_path.as_posix() if bootstrap_payload is not None else None,
        "table_path": table_path.as_posix(),
        "figure_path": figure_path.as_posix(),
        "examples_path": examples_path.as_posix(),
        "teacher_forced_examples_path": teacher_forced_examples_path.as_posix(),
        "environment": collect_runtime_facts(),
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "metric_path": metric_path.as_posix(),
        "bootstrap_path": bootstrap_path.as_posix() if bootstrap_payload is not None else None,
        "table_path": table_path.as_posix(),
        "figure_path": figure_path.as_posix(),
        "examples_path": examples_path.as_posix(),
        "teacher_forced_examples_path": teacher_forced_examples_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "c1_assessment": c1_assessment,
    }


def evaluate_teacher_forced_fidelity(
    *,
    rows: list[dict[str, Any]],
    tokenizer,
    pt_model,
    it_model,
    checkpoints: list[dict[str, Any]],
    variant_resources: list[dict[str, Any]],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layer_indices = [checkpoint["layer_index"] for checkpoint in checkpoints]
    kl_sums = {resource["name"]: 0.0 for resource in variant_resources}
    kl_token_counts = {resource["name"]: 0 for resource in variant_resources}
    delta_targets: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_indices}
    delta_predictions: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_indices}
    per_example_rows: list[dict[str, Any]] = []

    for row in rows:
        input_ids, attention_mask, answer_start = teacher_forced_inputs_from_row(tokenizer, row)
        model_inputs = {
            "input_ids": input_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
            "use_cache": False,
        }
        with torch.no_grad():
            pt_outputs = pt_model(**model_inputs)
            it_outputs = it_model(**model_inputs)
        pt_logits = pt_outputs.logits
        it_logits = it_outputs.logits

        position_mask = torch.zeros(int(input_ids.shape[0]), dtype=torch.float32)
        position_mask[answer_start:] = 1.0
        for resource in variant_resources:
            plan = build_variant_plan(resource, checkpoints=checkpoints, position_mask=position_mask, incremental_only=False)
            if plan["model_kind"] == "it":
                variant_logits = it_logits
            elif plan["interventions"] is None:
                variant_logits = pt_logits
            else:
                variant_logits = forward_with_interventions(pt_model, plan["interventions"], model_inputs).logits
            example_kl, example_count = answer_token_kl(it_logits, variant_logits, answer_start=answer_start)
            kl_sums[resource["name"]] += example_kl
            kl_token_counts[resource["name"]] += example_count
            per_example_rows.append(
                {
                    "variant": resource["name"],
                    "prompt_id": row["prompt_id"],
                    "split": row["split"],
                    "slice": row["slice"],
                    "KL_ans_to_IT": float(example_kl),
                    "answer_token_count": int(example_count),
                }
            )

        pt_hidden = capture_hidden_states(pt_model, layer_indices, model_inputs)
        it_hidden = capture_hidden_states(it_model, layer_indices, model_inputs)
        for checkpoint in checkpoints:
            layer_index = checkpoint["layer_index"]
            pt_answer = pt_hidden[layer_index][0, answer_start:, :].to(torch.float32).cpu()
            it_answer = it_hidden[layer_index][0, answer_start:, :].to(torch.float32).cpu()
            actual_delta = it_answer - pt_answer
            standardized = (pt_answer - checkpoint["input_mean"]) / (checkpoint["input_std"] + checkpoint["eps_std"])
            with torch.no_grad():
                predicted = checkpoint["module"](standardized).delta_hat.cpu()
            delta_targets[layer_index].append(actual_delta)
            delta_predictions[layer_index].append(predicted)

    r2_by_layer = {}
    for checkpoint in checkpoints:
        layer_index = checkpoint["layer_index"]
        target = torch.cat(delta_targets[layer_index], dim=0)
        prediction = torch.cat(delta_predictions[layer_index], dim=0)
        r2_by_layer[layer_index] = compute_r2(prediction, target)

    teacher_forced_metrics = {
        resource["name"]: {"KL_ans_to_IT": kl_sums[resource["name"]] / max(kl_token_counts[resource["name"]], 1)}
        for resource in variant_resources
    }
    for resource in variant_resources:
        if resource["kind"] == "full_delta":
            teacher_forced_metrics[resource["name"]].update({f"R2_layer_{layer}": value for layer, value in r2_by_layer.items()})
    teacher_forced_metrics["kl_token_count"] = dict(kl_token_counts)
    return teacher_forced_metrics, per_example_rows


def evaluate_free_generation(
    *,
    rows: list[dict[str, Any]],
    prompt_map: dict[str, dict[str, Any]],
    tokenizer,
    pt_model,
    it_model,
    checkpoints: list[dict[str, Any]],
    variant_resources: list[dict[str, Any]],
    generation_batch_size: int,
    model_config: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    per_example_rows: list[dict[str, Any]] = []
    variant_names = [resource["name"] for resource in variant_resources]
    variant_plans = {
        resource["name"]: build_variant_plan(
            resource,
            checkpoints=checkpoints,
            position_mask=None,
            incremental_only=True,
        )
        for resource in variant_resources
    }
    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variant_names}
    max_new_tokens = max_new_tokens_by_slice(model_config)

    for variant_name in variant_names:
        variant_plan = variant_plans[variant_name]
        model = it_model if variant_plan["model_kind"] == "it" else pt_model
        for slice_name in ("QA", "Math", "Format", "Brevity", "Harmful", "BenignAdjacent"):
            slice_rows = [row for row in rows if row["slice"] == slice_name]
            if not slice_rows:
                continue
            for batch_start in range(0, len(slice_rows), generation_batch_size):
                batch_rows = slice_rows[batch_start : batch_start + generation_batch_size]
                rendered_prefixes = [row["rendered_prefix"] for row in batch_rows]
                generations = greedy_generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    rendered_prefixes=rendered_prefixes,
                    max_new_tokens=max_new_tokens[slice_name],
                    interventions=variant_plan["interventions"],
                )

                for row, generation in zip(batch_rows, generations, strict=True):
                    prompt_row = prompt_map[row["prompt_id"]]
                    scored = score_generation(prompt_row, generation["completion_text"], generation["completion_token_ids"])
                    example = {
                        "variant": variant_name,
                        "prompt_id": row["prompt_id"],
                        "split": row["split"],
                        "slice": row["slice"],
                        "completion_text": generation["completion_text"],
                        "completion_token_ids": generation["completion_token_ids"],
                        "stop_reason": generation["stop_reason"],
                        "eos_reached": generation["eos_reached"],
                        **scored,
                    }
                    by_variant[variant_name].append(example)
                    per_example_rows.append(example)

    metrics = {variant: aggregate_variant_metrics(by_variant[variant]) for variant in variant_names}
    return metrics, per_example_rows


def normalize_variant_specs(raw_variants: list[Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for raw in raw_variants:
        if isinstance(raw, str):
            specs.append({"name": raw, "kind": legacy_variant_kind(raw)})
            continue
        if not isinstance(raw, dict):
            raise TypeError(f"Unsupported variant spec type: {type(raw).__name__}")
        if "name" not in raw or "kind" not in raw:
            raise ValueError(f"Variant specs must include name and kind, got keys={sorted(raw)}")
        specs.append(dict(raw))
    return specs


def legacy_variant_kind(variant_name: str) -> str:
    mapping = {
        "PT": "pt",
        "IT_neutral": "it",
        "PT_plus_FullDelta": "full_delta",
    }
    if variant_name not in mapping:
        raise ValueError(
            f"Legacy string variant {variant_name!r} is no longer supported for non-M4 rows; use a dict spec with name/kind"
        )
    return mapping[variant_name]


def build_variant_plan(
    variant_resource: dict[str, Any],
    *,
    checkpoints: list[dict[str, Any]],
    position_mask: torch.Tensor | None,
    incremental_only: bool,
) -> dict[str, Any]:
    kind = str(variant_resource["kind"])
    if kind == "pt":
        return {"model_kind": "pt", "interventions": None}
    if kind == "it":
        return {"model_kind": "it", "interventions": None}
    if kind == "full_delta":
        return {
            "model_kind": "pt",
            "interventions": build_full_delta_interventions(
                checkpoints=checkpoints,
                alphas=variant_resource["alphas"],
                position_mask=position_mask,
                incremental_only=incremental_only,
            ),
        }
    if kind == "sparse_mask":
        return {
            "model_kind": "pt",
            "interventions": build_masked_interventions(
                checkpoints=checkpoints,
                alphas=variant_resource["alphas"],
                mask_payload=variant_resource["mask_payload"],
                position_mask=position_mask,
                incremental_only=incremental_only,
                alpha_scale=float(variant_resource.get("alpha_scale", 1.0)),
            ),
        }
    if kind == "full_delta_minus_mask":
        return {
            "model_kind": "pt",
            "interventions": build_full_delta_minus_mask_interventions(
                checkpoints=checkpoints,
                alphas=variant_resource["alphas"],
                mask_payload=variant_resource["mask_payload"],
                position_mask=position_mask,
                incremental_only=incremental_only,
            ),
        }
    if kind == "mean_diff":
        return {
            "model_kind": "pt",
            "interventions": build_mean_diff_interventions(
                mean_deltas=variant_resource["mean_deltas"],
                alphas=variant_resource["alphas"],
                position_mask=position_mask,
                incremental_only=incremental_only,
            ),
        }
    raise ValueError(f"Unsupported variant kind: {kind}")


def prepare_variant_resources(
    variant_specs: list[dict[str, Any]],
    *,
    checkpoints: list[dict[str, Any]],
    full_delta_alphas: dict[int, float],
    default_gates_path: str,
) -> list[dict[str, Any]]:
    _ = checkpoints
    resources: list[dict[str, Any]] = []
    for spec in variant_specs:
        resource = {"name": str(spec["name"]), "kind": str(spec["kind"])}
        if resource["kind"] == "pt":
            resources.append(resource)
            continue
        if resource["kind"] == "it":
            resources.append(resource)
            continue
        if resource["kind"] == "full_delta":
            resource["alphas"] = dict(full_delta_alphas)
            resources.append(resource)
            continue
        if resource["kind"] in {"sparse_mask", "full_delta_minus_mask"}:
            gates_path = str(spec.get("gates_path", default_gates_path))
            resource["alphas"] = load_gate_alphas(gates_path)
            resource["mask_payload"] = load_mask_payload(spec["mask_path"])
            if "alpha_scale" in spec:
                resource["alpha_scale"] = float(spec["alpha_scale"])
            resources.append(resource)
            continue
        if resource["kind"] == "mean_diff":
            resource["alphas"] = load_gate_alphas(spec["gates_path"])
            resource["mean_deltas"] = load_mean_diff_vectors(spec["cache_summary_path"])
            resources.append(resource)
            continue
        raise ValueError(f"Unsupported variant kind: {resource['kind']}")
    return resources


def assess_c1_support(teacher_forced_metrics: dict[str, Any], generation_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    pt_kl = teacher_forced_metrics["PT"]["KL_ans_to_IT"]
    full_kl = teacher_forced_metrics["PT_plus_FullDelta"]["KL_ans_to_IT"]
    pt_cap = generation_metrics["PT"]["Cap"]
    full_cap = generation_metrics["PT_plus_FullDelta"]["Cap"]
    it_cap = generation_metrics["IT_neutral"]["Cap"]
    cap_recovery = generation_metrics["PT_plus_FullDelta"]["Cap_Recovery"]
    r2_values = [value for key, value in teacher_forced_metrics["PT_plus_FullDelta"].items() if key.startswith("R2_layer_")]
    return {
        "kl_reduction_fraction": 1.0 - (full_kl / pt_kl if pt_kl > 0 else math.nan),
        "cap_recovery": cap_recovery,
        "cap_gain_absolute": full_cap - pt_cap,
        "it_cap": it_cap,
        "positive_r2_all_layers": all(value > 0.0 for value in r2_values),
        "meets_c1_threshold": (
            (pt_kl > 0 and full_kl <= 0.70 * pt_kl)
            and (cap_recovery >= 0.35 or (full_cap - pt_cap) >= 0.10)
            and all(value > 0.0 for value in r2_values)
        ),
    }


def build_markdown_table(
    teacher_forced_metrics: dict[str, Any],
    generation_metrics: dict[str, dict[str, float]],
    c1_assessment: dict[str, Any],
) -> str:
    teacher_forced_lookup = {
        name: payload["KL_ans_to_IT"]
        for name, payload in teacher_forced_metrics.items()
        if isinstance(payload, dict) and "KL_ans_to_IT" in payload
    }
    lines = [
        "# Fidelity Summary",
        "",
        "| Variant | KL_ans_to_IT | Cap | Cap_Recovery | Len | BrevEx | Policy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_name, metrics in generation_metrics.items():
        kl_value = teacher_forced_lookup.get(variant_name)
        kl_cell = f"{kl_value:.6f}" if kl_value is not None else "NA"
        lines.append(
            f"| {variant_name} | {kl_cell} | {metrics['Cap']:.4f} | {metrics['Cap_Recovery']:.4f} | {metrics['Len']:.2f} | {metrics['BrevEx']:.2f} | {metrics['Policy']:.4f} |"
        )

    full = teacher_forced_metrics["PT_plus_FullDelta"]
    lines.extend(
        [
            "",
            f"- R2 values: {', '.join(f'{key}={value:.4f}' for key, value in full.items() if key.startswith('R2_layer_'))}",
            f"- Meets C1 threshold: `{c1_assessment['meets_c1_threshold']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_svg_summary(teacher_forced_metrics: dict[str, Any], generation_metrics: dict[str, dict[str, float]]) -> str:
    teacher_forced_lookup = {
        name: payload["KL_ans_to_IT"]
        for name, payload in teacher_forced_metrics.items()
        if isinstance(payload, dict) and "KL_ans_to_IT" in payload
    }
    variant_names = list(generation_metrics)
    cap_max = max((metrics["Cap"] for metrics in generation_metrics.values()), default=1e-9)
    kl_max = max([1e-9] + list(teacher_forced_lookup.values()))
    row_height = 24
    bar_width = 180
    width = 760
    height = 90 + row_height * len(variant_names)
    rows: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <rect width="{width}" height="{height}" fill="white"/>',
        '  <text x="20" y="24" font-family="Arial" font-size="18">Evaluation Summary</text>',
        '  <text x="20" y="52" font-family="Arial" font-size="12">Capability</text>',
        '  <text x="330" y="52" font-family="Arial" font-size="12">Answer-token KL to IT</text>',
    ]
    for index, variant_name in enumerate(variant_names):
        y = 76 + index * row_height
        cap_value = generation_metrics[variant_name]["Cap"]
        kl_value = teacher_forced_lookup.get(variant_name)
        cap_width = bar_width * (cap_value / cap_max if cap_max > 0 else 0.0)
        rows.append(f'  <text x="20" y="{y}" font-family="Arial" font-size="11">{variant_name}</text>')
        rows.append(f'  <rect x="120" y="{y - 10}" width="{cap_width:.2f}" height="12" fill="#2f6f9f"/>')
        rows.append(f'  <text x="305" y="{y}" font-family="Arial" font-size="11">{cap_value:.4f}</text>')
        if kl_value is None:
            rows.append(f'  <text x="330" y="{y}" font-family="Arial" font-size="11">NA</text>')
        else:
            kl_width = bar_width * (kl_value / kl_max if kl_max > 0 else 0.0)
            rows.append(f'  <rect x="400" y="{y - 10}" width="{kl_width:.2f}" height="12" fill="#9c3d3d"/>')
            rows.append(f'  <text x="585" y="{y}" font-family="Arial" font-size="11">{kl_value:.4f}</text>')
    rows.append("</svg>")
    return "\n".join(rows) + "\n"


def write_jsonl_like(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
