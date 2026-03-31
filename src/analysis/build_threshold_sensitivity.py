from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.common import (
    build_feature_table_run_id,
    build_mask_artifact_path,
    build_split_audit,
    ensure_no_test_leakage,
    matched_mask_size_cap,
    mask_selection_run_id,
    sorted_mask_members,
    layer_count_map,
)
from src.analysis.select_feature_masks import (
    build_capability_core,
    build_prompt_frame,
    build_refusal_core,
    build_tune_context,
    build_verbosity_core,
    capability_objective,
    fit_selector_models,
    forward_select_mask,
    precompute_target_contributions,
    refusal_objective,
    verbosity_objective,
)
from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.jsonl import read_jsonl
from src.common.runmeta import collect_runtime_facts


@dataclass
class ThresholdSensitivityInputs:
    feature_table_run_id: str
    selection_run_id: str
    split_audit: dict[str, Any]
    score_rows: list[dict[str, Any]]
    score_lookup: dict[str, dict[str, Any]]
    target_contributions: dict[str, dict[str, np.ndarray]]
    tune_context: dict[str, Any]
    mask_caps: dict[str, Any]
    min_gain: float
    source_masks: dict[str, dict[str, Any]]
    source_mask_paths: dict[str, str]
    refusal_signal_nontrivial: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CPU-only threshold-sensitivity masks from frozen M5 selectors.")
    parser.add_argument("--config", required=True, help="Path to the threshold-sensitivity config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_threshold_sensitivity(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_threshold_sensitivity(config: dict[str, Any]) -> dict[str, Any]:
    stage_name = str(config["stage_name"])
    variants = normalize_threshold_variants(config["variants"])
    run_payload = {
        "config": {
            "stage_name": stage_name,
            "selectors_config_path": str(config["selectors_config_path"]),
            "variants": variants,
        }
    }
    run_id = build_run_id(stage_name, run_payload)

    snapshot_path = build_output_path(config["paths"]["snapshot_dir"], f"resolved_config_{run_id}.json")
    save_resolved_config_snapshot({"config": config, "run_id": run_id}, snapshot_path)

    selectors_config = load_yaml_config(config["selectors_config_path"])
    inputs = load_threshold_sensitivity_inputs(selectors_config)

    start_time = time.perf_counter()
    generated_masks: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    for variant in variants:
        payload = build_threshold_variant_payload(variant=variant, inputs=inputs, run_id=run_id, stage_name=stage_name)
        output_path = build_output_path(config["paths"]["mask_dir"], f"{payload['mask_name']}_{run_id}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        generated_masks.append(
            {
                "mask_name": payload["mask_name"],
                "target": payload["target"],
                "thresholds": payload["thresholds"],
                "output_path": output_path.as_posix(),
            }
        )
        variant_summaries.append(
            {
                "mask_name": payload["mask_name"],
                "target": payload["target"],
                "candidate_count": payload["candidate_count"],
                "size": payload["size"],
                "predicted_objective": payload["predicted_objective"],
                "matches_locked_mask": payload["matches_locked_mask"],
                "thresholds": payload["thresholds"],
                "output_path": output_path.as_posix(),
            }
        )

    summary_path = build_output_path(config["paths"]["summary_dir"], f"threshold_sensitivity_{run_id}.json")
    runtime_path = build_output_path(config["paths"]["runtime_dir"], f"runtime_report_{run_id}.json")
    summary_payload = {
        "run_id": run_id,
        "stage_name": stage_name,
        "selectors_config_path": str(config["selectors_config_path"]),
        "feature_table_run_id": inputs.feature_table_run_id,
        "selection_run_id": inputs.selection_run_id,
        "snapshot_path": snapshot_path.as_posix(),
        "variant_count": len(variant_summaries),
        "variants": variant_summaries,
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_payload = {
        "run_id": run_id,
        "stage_name": stage_name,
        "wall_clock_seconds": time.perf_counter() - start_time,
        "generated_mask_count": len(generated_masks),
        "snapshot_path": snapshot_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "environment": collect_runtime_facts(),
    }
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "stage_name": stage_name,
        "snapshot_path": snapshot_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "generated_mask_count": len(generated_masks),
        "generated_masks": generated_masks,
    }


def load_threshold_sensitivity_inputs(selectors_config: dict[str, Any]) -> ThresholdSensitivityInputs:
    feature_table_run_id = build_feature_table_run_id(selectors_config)
    selection_run_id = mask_selection_run_id(selectors_config, feature_table_run_id=feature_table_run_id)

    candidate_rows_path = build_artifact_path("selector", "candidate_feature_rows", feature_table_run_id, ".parquet")
    generation_examples_path = build_artifact_path(
        "selector",
        "selection_generation_examples",
        feature_table_run_id,
        ".jsonl",
    )
    generation_metrics_path = build_artifact_path(
        "selector",
        "selection_generation_metrics",
        feature_table_run_id,
        ".json",
    )
    score_table_path = build_artifact_path("selector", "feature_selector_scores", selection_run_id, ".parquet")
    selection_summary_path = build_artifact_path("selector", "mask_selection_summary", selection_run_id, ".json")

    candidate_rows = pd.read_parquet(candidate_rows_path)
    generation_examples = read_jsonl(generation_examples_path)
    generation_metrics = json.loads(generation_metrics_path.read_text(encoding="utf-8"))
    score_frame = pd.read_parquet(score_table_path).sort_values(
        ["rank_candidate_score", "layer", "feature_id"],
        kind="stable",
    )
    selection_summary = json.loads(selection_summary_path.read_text(encoding="utf-8"))

    prompt_frame = build_prompt_frame(candidate_rows, generation_examples)
    feature_columns = sorted(column for column in prompt_frame.columns if "::" in column)
    split_audit = build_split_audit(
        feature_summary_splits=selectors_config["feature_splits"],
        candidate_scoring_splits=["select_train"],
        selector_fit_splits=["select_train"],
        forward_selection_splits=["select_tune"],
    )
    ensure_no_test_leakage(split_audit)

    selector_models = fit_selector_models(
        prompt_frame=prompt_frame,
        feature_columns=feature_columns,
        selector_model_config=selectors_config["selector_model"],
    )
    tune_frame = prompt_frame[prompt_frame["split"] == "select_tune"].reset_index(drop=True)
    tune_context = build_tune_context(tune_frame=tune_frame, generation_metrics=generation_metrics["select_tune"])
    target_contributions = precompute_target_contributions(
        prompt_frame=tune_frame,
        selector_models=selector_models,
        candidate_keys=score_frame["candidate_key"].tolist(),
    )

    source_masks: dict[str, dict[str, Any]] = {}
    source_mask_paths: dict[str, str] = {}
    for target, mask_name in (
        ("capability", "capability_mask"),
        ("verbosity", "verbosity_mask"),
        ("refusal", "refusal_mask"),
    ):
        mask_path = build_mask_artifact_path(mask_name, selection_run_id=selection_run_id)
        if not mask_path.exists():
            continue
        source_mask_paths[target] = mask_path.as_posix()
        source_masks[target] = json.loads(mask_path.read_text(encoding="utf-8"))

    score_rows = score_frame.to_dict(orient="records")
    score_lookup = {str(row["candidate_key"]): row for row in score_rows}
    return ThresholdSensitivityInputs(
        feature_table_run_id=feature_table_run_id,
        selection_run_id=selection_run_id,
        split_audit=split_audit,
        score_rows=score_rows,
        score_lookup=score_lookup,
        target_contributions=target_contributions,
        tune_context=tune_context,
        mask_caps=dict(selectors_config["mask_caps"]),
        min_gain=float(selectors_config["forward_selection"]["min_gain"]),
        source_masks=source_masks,
        source_mask_paths=source_mask_paths,
        refusal_signal_nontrivial=bool(selection_summary.get("refusal_signal_nontrivial", False)),
    )


def normalize_threshold_variants(raw_variants: list[Any]) -> list[dict[str, Any]]:
    if not raw_variants:
        raise ValueError("variants must not be empty")
    variants: list[dict[str, Any]] = []
    for raw in raw_variants:
        if not isinstance(raw, dict):
            raise TypeError(f"Unsupported threshold variant type: {type(raw).__name__}")
        missing = [key for key in ("name", "target", "thresholds") if key not in raw]
        if missing:
            raise ValueError(f"Threshold variants must include {missing}, got keys={sorted(raw)}")
        target = str(raw["target"])
        thresholds = normalize_thresholds(target=target, raw_thresholds=raw["thresholds"])
        variants.append(
            {
                "name": str(raw["name"]),
                "target": target,
                "thresholds": thresholds,
            }
        )
    return variants


def normalize_thresholds(*, target: str, raw_thresholds: dict[str, Any]) -> dict[str, float]:
    if not isinstance(raw_thresholds, dict):
        raise TypeError(f"thresholds must be a mapping, got {type(raw_thresholds).__name__}")
    if target == "capability":
        return {
            "core_fraction": normalize_fraction(raw_thresholds["core_fraction"], name="core_fraction"),
            "verbosity_exclusion_fraction": normalize_fraction(
                raw_thresholds["verbosity_exclusion_fraction"],
                name="verbosity_exclusion_fraction",
            ),
            "refusal_exclusion_fraction": normalize_fraction(
                raw_thresholds["refusal_exclusion_fraction"],
                name="refusal_exclusion_fraction",
            ),
        }
    if target == "verbosity":
        return {
            "core_fraction": normalize_fraction(raw_thresholds["core_fraction"], name="core_fraction"),
            "capability_exclusion_fraction": normalize_fraction(
                raw_thresholds["capability_exclusion_fraction"],
                name="capability_exclusion_fraction",
            ),
        }
    if target == "refusal":
        return {
            "core_fraction": normalize_fraction(raw_thresholds["core_fraction"], name="core_fraction"),
            "capability_exclusion_fraction": normalize_fraction(
                raw_thresholds["capability_exclusion_fraction"],
                name="capability_exclusion_fraction",
            ),
        }
    raise ValueError(f"Unsupported target for threshold sensitivity: {target}")


def normalize_fraction(value: Any, *, name: str) -> float:
    normalized = float(value)
    if normalized <= 0.0 or normalized > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {normalized}")
    return normalized


def build_threshold_variant_payload(
    *,
    variant: dict[str, Any],
    inputs: ThresholdSensitivityInputs,
    run_id: str,
    stage_name: str,
) -> dict[str, Any]:
    target = str(variant["target"])
    thresholds = dict(variant["thresholds"])
    candidate_keys, objective_name, objective_fn, size_cap = variant_execution_plan(
        target=target,
        thresholds=thresholds,
        inputs=inputs,
    )
    result = forward_select_mask(
        mask_name=str(variant["name"]),
        candidate_keys=candidate_keys,
        score_lookup=inputs.score_lookup,
        target_contributions=inputs.target_contributions,
        objective=objective_fn,
        max_size=size_cap,
        min_gain=inputs.min_gain,
        objective_name=objective_name,
    )

    source_mask = inputs.source_masks.get(target)
    sorted_members = sorted_mask_members(result["members"])
    return {
        "run_id": run_id,
        "stage_name": stage_name,
        "mask_name": str(variant["name"]),
        "target": target,
        "reference_type": "threshold_sensitivity",
        "reference_mask_name": source_mask.get("mask_name") if source_mask else None,
        "source_selection_run_id": inputs.selection_run_id,
        "source_mask_path": inputs.source_mask_paths.get(target),
        "feature_table_run_id": inputs.feature_table_run_id,
        "selection_split": "select_tune",
        "selector_fit_split": "select_train",
        "objective_name": objective_name,
        "predicted_objective": result["objective_score"],
        "candidate_count": result["candidate_count"],
        "size": len(sorted_members),
        "members": sorted_members,
        "layer_counts": {str(layer): count for layer, count in layer_count_map(sorted_members).items()},
        "construction_log": result["log"],
        "split_audit": inputs.split_audit,
        "thresholds": thresholds,
        "matches_locked_mask": (
            sorted_members == sorted_mask_members(source_mask["members"]) if source_mask is not None else None
        ),
    }


def variant_execution_plan(
    *,
    target: str,
    thresholds: dict[str, float],
    inputs: ThresholdSensitivityInputs,
) -> tuple[list[str], str, Any, int]:
    if target == "capability":
        return (
            build_capability_core(
                inputs.score_rows,
                capability_fraction=thresholds["core_fraction"],
                verbosity_exclusion_fraction=thresholds["verbosity_exclusion_fraction"],
                refusal_exclusion_fraction=thresholds["refusal_exclusion_fraction"],
            ),
            "J_cap",
            lambda current: capability_objective(current, inputs.tune_context),
            matched_mask_size_cap("capability", inputs.mask_caps),
        )
    if target == "verbosity":
        return (
            build_verbosity_core(
                inputs.score_rows,
                verbosity_fraction=thresholds["core_fraction"],
                capability_exclusion_fraction=thresholds["capability_exclusion_fraction"],
            ),
            "J_vsub",
            lambda current: verbosity_objective(current, inputs.tune_context),
            matched_mask_size_cap("verbosity", inputs.mask_caps),
        )
    if target == "refusal":
        if not inputs.refusal_signal_nontrivial:
            raise ValueError("Refusal threshold sensitivity requested, but no nontrivial refusal signal was frozen in M5")
        return (
            build_refusal_core(
                inputs.score_rows,
                refusal_fraction=thresholds["core_fraction"],
                capability_exclusion_fraction=thresholds["capability_exclusion_fraction"],
            ),
            "J_ref",
            lambda current: refusal_objective(current, inputs.tune_context),
            matched_mask_size_cap("refusal", inputs.mask_caps),
        )
    raise ValueError(f"Unsupported threshold sensitivity target: {target}")


def build_output_path(output_dir: str | Path, filename: str) -> Path:
    return Path(output_dir) / filename


if __name__ == "__main__":
    raise SystemExit(main())
