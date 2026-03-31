from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from src.common.configs import build_artifact_path, build_run_id, load_yaml_config
from src.common.jsonl import read_jsonl

FEATURE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "max_answer",
    "mean_answer",
    "last_answer",
    "mean_contribution_norm",
)


def candidate_feature_key(layer: int, feature_id: int) -> str:
    return f"layer_{int(layer)}::feature_{int(feature_id)}"


def parse_candidate_feature_key(value: str) -> tuple[int, int]:
    layer_token, feature_token = value.split("::", maxsplit=1)
    return int(layer_token.split("_", maxsplit=1)[1]), int(feature_token.split("_", maxsplit=1)[1])


def feature_column_name(layer: int, feature_id: int, summary_name: str) -> str:
    if summary_name not in FEATURE_SUMMARY_COLUMNS:
        raise ValueError(f"Unsupported feature summary column: {summary_name}")
    return f"{candidate_feature_key(layer, feature_id)}::{summary_name}"


def build_split_audit(
    *,
    feature_summary_splits: Iterable[str],
    candidate_scoring_splits: Iterable[str],
    selector_fit_splits: Iterable[str],
    forward_selection_splits: Iterable[str],
) -> dict[str, Any]:
    payload = {
        "feature_summary_splits": sorted({str(split) for split in feature_summary_splits}),
        "candidate_scoring_splits": sorted({str(split) for split in candidate_scoring_splits}),
        "selector_fit_splits": sorted({str(split) for split in selector_fit_splits}),
        "forward_selection_splits": sorted({str(split) for split in forward_selection_splits}),
    }
    touched = set().union(
        payload["feature_summary_splits"],
        payload["candidate_scoring_splits"],
        payload["selector_fit_splits"],
        payload["forward_selection_splits"],
    )
    payload["touched_splits"] = sorted(touched)
    payload["test_split_touched"] = "test" in touched
    payload["candidate_scoring_uses_only_select_train"] = payload["candidate_scoring_splits"] == ["select_train"]
    payload["selector_fit_uses_only_select_train"] = payload["selector_fit_splits"] == ["select_train"]
    payload["forward_selection_uses_only_select_tune"] = payload["forward_selection_splits"] == ["select_tune"]
    payload["no_test_leakage"] = (
        not payload["test_split_touched"]
        and payload["candidate_scoring_uses_only_select_train"]
        and payload["selector_fit_uses_only_select_train"]
        and payload["forward_selection_uses_only_select_tune"]
    )
    return payload


def ensure_no_test_leakage(audit: dict[str, Any]) -> None:
    if audit.get("test_split_touched"):
        raise ValueError(f"Test split leakage detected: touched_splits={audit.get('touched_splits')}")
    if not audit.get("candidate_scoring_uses_only_select_train"):
        raise ValueError(f"Candidate scoring must use only select_train, got {audit.get('candidate_scoring_splits')}")
    if not audit.get("selector_fit_uses_only_select_train"):
        raise ValueError(f"Selector fitting must use only select_train, got {audit.get('selector_fit_splits')}")
    if not audit.get("forward_selection_uses_only_select_tune"):
        raise ValueError(f"Forward selection must use only select_tune, got {audit.get('forward_selection_splits')}")


def layer_count_map(members: Iterable[dict[str, int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for member in members:
        layer = int(member["layer"])
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def sorted_mask_members(members: Iterable[dict[str, int]]) -> list[dict[str, int]]:
    return sorted(
        [{"layer": int(member["layer"]), "feature_id": int(member["feature_id"])} for member in members],
        key=lambda item: (item["layer"], item["feature_id"]),
    )


def matched_mask_size_cap(mask_name: str, mask_caps: dict[str, Any]) -> int:
    if mask_name not in {"capability", "verbosity", "refusal"}:
        raise KeyError(f"Unsupported mask name: {mask_name}")
    return int(mask_caps[mask_name])


def top_fraction_count(total: int, fraction: float) -> int:
    if total <= 0:
        return 0
    return max(1, int(math.ceil(total * fraction)))


def build_feature_table_run_payload(config: dict[str, Any], *, record_count: int) -> dict[str, Any]:
    return {
        "config": {
            "completion_run_id": config["completion_run_id"],
            "checkpoint_paths": config["checkpoint_paths"],
            "gates_path": config["gates_path"],
            "candidate_pool_size": int(config["candidate_pool_size"]),
            "feature_splits": list(config["feature_splits"]),
            "feature_summary_columns": list(config.get("feature_summary_columns", FEATURE_SUMMARY_COLUMNS)),
        },
        "record_count": int(record_count),
    }


def build_feature_table_run_id(config: dict[str, Any]) -> str:
    data_config = load_yaml_config(config["data_config_path"])
    model_config = load_yaml_config(config["model_config_path"])
    rows = load_completion_rows(
        completion_dir=model_config["paths"]["completion_dir"],
        completion_run_id=config["completion_run_id"],
        splits=list(config["feature_splits"]),
        slices=list(config.get("eval_slices", data_config["slices"])),
    )
    return build_run_id("build_feature_table", build_feature_table_run_payload(config, record_count=len(rows)))


def load_completion_rows(
    *,
    completion_dir: str | Path,
    completion_run_id: str,
    splits: list[str],
    slices: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    completion_root = Path(completion_dir)
    for split in splits:
        for slice_name in slices:
            rows.extend(read_jsonl(completion_root / f"{split}_{slice_name}_{completion_run_id}.jsonl"))
    return rows


def feature_table_artifact_paths(config: dict[str, Any]) -> dict[str, str]:
    run_id = build_feature_table_run_id(config)
    return {
        "run_id": run_id,
        "summary_path": build_artifact_path("selector", "feature_table_summary", run_id, ".json").as_posix(),
        "candidate_table_path": build_artifact_path("selector", "candidate_feature_table", run_id, ".parquet").as_posix(),
        "candidate_rows_path": build_artifact_path("selector", "candidate_feature_rows", run_id, ".parquet").as_posix(),
        "generation_examples_path": build_artifact_path("selector", "selection_generation_examples", run_id, ".jsonl").as_posix(),
        "generation_metrics_path": build_artifact_path("selector", "selection_generation_metrics", run_id, ".json").as_posix(),
        "runtime_path": build_artifact_path("runtime", "runtime_report", run_id, ".json").as_posix(),
        "snapshot_path": build_artifact_path("config_snapshot", "resolved_config", run_id, ".json").as_posix(),
    }


def build_mask_selection_run_payload(config: dict[str, Any], *, feature_table_run_id: str) -> dict[str, Any]:
    return {
        "config": {
            "candidate_pool_size": int(config["candidate_pool_size"]),
            "mask_caps": dict(config["mask_caps"]),
            "selector_model": dict(config["selector_model"]),
            "forward_selection": dict(config["forward_selection"]),
            "random_baseline_seed": int(config["random_baseline_seed"]),
        },
        "feature_table_run_id": feature_table_run_id,
    }


def mask_selection_run_id(config: dict[str, Any], *, feature_table_run_id: str | None = None) -> str:
    resolved_feature_table_run_id = feature_table_run_id or build_feature_table_run_id(config)
    return build_run_id(
        "select_feature_masks",
        build_mask_selection_run_payload(config, feature_table_run_id=resolved_feature_table_run_id),
    )


def mask_selection_artifact_paths(config: dict[str, Any]) -> dict[str, str]:
    feature_table_run_id = build_feature_table_run_id(config)
    run_id = mask_selection_run_id(config, feature_table_run_id=feature_table_run_id)
    return {
        "run_id": run_id,
        "summary_path": build_artifact_path("selector", "mask_selection_summary", run_id, ".json").as_posix(),
        "runtime_path": build_artifact_path("runtime", "runtime_report", run_id, ".json").as_posix(),
        "snapshot_path": build_artifact_path("config_snapshot", "resolved_config", run_id, ".json").as_posix(),
    }


def build_mask_artifact_path(mask_name: str, *, selection_run_id: str) -> Path:
    return build_artifact_path("mask", mask_name, selection_run_id, ".json")
