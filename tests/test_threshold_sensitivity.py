from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.analysis.build_threshold_sensitivity as threshold_sensitivity
from src.analysis.build_threshold_sensitivity import ThresholdSensitivityInputs, run_threshold_sensitivity
from src.analysis.select_feature_masks import build_capability_core, build_verbosity_core


def test_build_capability_core_respects_custom_thresholds() -> None:
    score_rows = [
        {"candidate_key": "a", "layer": 16, "feature_id": 1, "C_j": 0.90, "V_j": 0.10, "P_j": 0.10},
        {"candidate_key": "b", "layer": 16, "feature_id": 2, "C_j": 0.80, "V_j": 0.20, "P_j": 0.10},
        {"candidate_key": "c", "layer": 28, "feature_id": 3, "C_j": 0.70, "V_j": 0.90, "P_j": 0.10},
        {"candidate_key": "d", "layer": 28, "feature_id": 4, "C_j": 0.60, "V_j": 0.10, "P_j": 0.90},
    ]

    narrow_core = build_capability_core(
        score_rows,
        capability_fraction=0.25,
        verbosity_exclusion_fraction=0.25,
        refusal_exclusion_fraction=0.25,
    )
    wider_core = build_capability_core(
        score_rows,
        capability_fraction=0.50,
        verbosity_exclusion_fraction=0.25,
        refusal_exclusion_fraction=0.25,
    )

    assert narrow_core == ["a"]
    assert wider_core == ["a", "b"]


def test_build_verbosity_core_respects_capability_exclusion_threshold() -> None:
    score_rows = [
        {"candidate_key": "a", "layer": 16, "feature_id": 1, "C_j": 0.90, "V_j": 0.10, "P_j": 0.10},
        {"candidate_key": "b", "layer": 16, "feature_id": 2, "C_j": 0.20, "V_j": 0.80, "P_j": 0.10},
        {"candidate_key": "c", "layer": 28, "feature_id": 3, "C_j": 0.70, "V_j": 0.70, "P_j": 0.10},
        {"candidate_key": "d", "layer": 28, "feature_id": 4, "C_j": 0.10, "V_j": 0.60, "P_j": 0.10},
    ]

    looser_exclusion = build_verbosity_core(
        score_rows,
        verbosity_fraction=0.50,
        capability_exclusion_fraction=0.25,
    )
    tighter_exclusion = build_verbosity_core(
        score_rows,
        verbosity_fraction=0.50,
        capability_exclusion_fraction=0.50,
    )

    assert looser_exclusion == ["b", "c"]
    assert tighter_exclusion == ["b"]


def test_run_threshold_sensitivity_writes_mask_variants(tmp_path: Path, monkeypatch) -> None:
    selectors_config_path = tmp_path / "selectors.yaml"
    selectors_config_path.write_text("stage_name: select_feature_masks\n", encoding="utf-8")

    zero = np.zeros(2, dtype=np.float64)
    fake_inputs = ThresholdSensitivityInputs(
        feature_table_run_id="build_feature_table-abc123",
        selection_run_id="select_feature_masks-abc123",
        split_audit={"no_test_leakage": True},
        score_rows=[
            {"candidate_key": "a", "layer": 16, "feature_id": 1, "C_j": 0.90, "V_j": 0.10, "P_j": 0.10},
            {"candidate_key": "b", "layer": 28, "feature_id": 2, "C_j": 0.80, "V_j": 0.95, "P_j": 0.10},
            {"candidate_key": "c", "layer": 28, "feature_id": 3, "C_j": 0.10, "V_j": 0.80, "P_j": 0.10},
            {"candidate_key": "d", "layer": 16, "feature_id": 4, "C_j": 0.05, "V_j": 0.20, "P_j": 0.90},
        ],
        score_lookup={
            "a": {"candidate_key": "a", "layer": 16, "feature_id": 1},
            "b": {"candidate_key": "b", "layer": 28, "feature_id": 2},
            "c": {"candidate_key": "c", "layer": 28, "feature_id": 3},
            "d": {"candidate_key": "d", "layer": 16, "feature_id": 4},
        },
        target_contributions={
            "qa_correct_delta": {
                "a": np.array([1.0, 0.0], dtype=np.float64),
                "b": np.array([0.0, 0.0], dtype=np.float64),
                "c": np.array([0.5, 0.0], dtype=np.float64),
                "d": zero.copy(),
            },
            "math_correct_delta": {key: zero.copy() for key in ("a", "b", "c", "d")},
            "format_pass_delta": {key: zero.copy() for key in ("a", "b", "c", "d")},
            "harmful_refused_delta": {key: zero.copy() for key in ("a", "b", "c", "d")},
            "benign_refused_delta": {key: zero.copy() for key in ("a", "b", "c", "d")},
            "brevity_excess_tokens_delta": {
                "a": zero.copy(),
                "b": np.array([0.0, -1.0], dtype=np.float64),
                "c": np.array([0.0, -0.5], dtype=np.float64),
                "d": zero.copy(),
            },
            "token_len_delta": {
                "a": zero.copy(),
                "b": np.array([-1.0, -1.0], dtype=np.float64),
                "c": np.array([-3.0, -3.0], dtype=np.float64),
                "d": zero.copy(),
            },
        },
        tune_context={
            "pt_metrics": {"Len": 10.0, "BrevEx": 6.0, "BenignRefusal": 0.0, "Cap": 0.0},
            "it_metrics": {"Len": 5.0, "BrevEx": 0.0},
            "full_metrics": {"Len": 7.0, "BrevEx": 1.0, "Cap": 1.0},
            "qa_mask": pd.Series([True, False]),
            "math_mask": pd.Series([False, False]),
            "format_mask": pd.Series([False, False]),
            "brevity_mask": pd.Series([False, True]),
            "harmful_mask": pd.Series([False, False]),
            "benign_mask": pd.Series([False, False]),
            "length_mask": pd.Series([True, True]),
            "pt_passed": np.array([0.0, 0.0], dtype=np.float64),
            "full_passed": np.array([1.0, 0.0], dtype=np.float64),
            "pt_token_len": np.array([10.0, 10.0], dtype=np.float64),
            "full_token_len": np.array([7.0, 7.0], dtype=np.float64),
            "pt_brevity_excess_tokens": np.array([0.0, 6.0], dtype=np.float64),
            "full_brevity_excess_tokens": np.array([0.0, 1.0], dtype=np.float64),
            "pt_refused": np.array([0.0, 0.0], dtype=np.float64),
            "full_refused": np.array([0.0, 0.0], dtype=np.float64),
        },
        mask_caps={"capability": 24, "verbosity": 16, "refusal": 12},
        min_gain=0.002,
        source_masks={
            "capability": {"mask_name": "capability_mask", "members": [{"layer": 16, "feature_id": 1}]},
            "verbosity": {"mask_name": "verbosity_mask", "members": [{"layer": 28, "feature_id": 2}]},
        },
        source_mask_paths={
            "capability": "C:/tmp/capability_mask.json",
            "verbosity": "C:/tmp/verbosity_mask.json",
        },
        refusal_signal_nontrivial=False,
    )
    monkeypatch.setattr(threshold_sensitivity, "load_threshold_sensitivity_inputs", lambda _: fake_inputs)

    config = {
        "stage_name": "build_threshold_sensitivity",
        "selectors_config_path": selectors_config_path.as_posix(),
        "variants": [
            {
                "name": "capability_mask_core25pct",
                "target": "capability",
                "thresholds": {
                    "core_fraction": 0.25,
                    "verbosity_exclusion_fraction": 0.25,
                    "refusal_exclusion_fraction": 0.25,
                },
            },
            {
                "name": "verbosity_mask_core25pct",
                "target": "verbosity",
                "thresholds": {
                    "core_fraction": 0.25,
                    "capability_exclusion_fraction": 0.25,
                },
            },
        ],
        "paths": {
            "mask_dir": (tmp_path / "masks").as_posix(),
            "summary_dir": (tmp_path / "metrics").as_posix(),
            "runtime_dir": (tmp_path / "runtime").as_posix(),
            "snapshot_dir": (tmp_path / "snapshots").as_posix(),
        },
    }

    result = run_threshold_sensitivity(config)

    assert result["generated_mask_count"] == 2
    assert Path(result["summary_path"]).exists()
    assert Path(result["runtime_path"]).exists()
    assert Path(result["snapshot_path"]).exists()

    generated_files = sorted((tmp_path / "masks").glob("*.json"))
    assert len(generated_files) == 2
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in generated_files]
    loaded_by_name = {item["mask_name"]: item for item in loaded}
    assert loaded_by_name["capability_mask_core25pct"]["members"] == [{"layer": 16, "feature_id": 1}]
    assert loaded_by_name["capability_mask_core25pct"]["matches_locked_mask"] is True
    assert loaded_by_name["verbosity_mask_core25pct"]["members"] == [{"layer": 28, "feature_id": 2}]
    assert loaded_by_name["verbosity_mask_core25pct"]["matches_locked_mask"] is True
