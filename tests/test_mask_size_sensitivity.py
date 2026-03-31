from __future__ import annotations

import json
from pathlib import Path

from src.analysis.build_mask_size_sensitivity import (
    build_sensitivity_mask_payload,
    run_mask_size_sensitivity,
    write_sensitivity_variants,
)


def _source_payload() -> dict[str, object]:
    return {
        "run_id": "select_feature_masks-abc123",
        "stage_name": "select_feature_masks",
        "mask_name": "capability_mask",
        "reference_type": "primary",
        "reference_mask_name": None,
        "feature_table_run_id": "build_feature_table-abc123",
        "selection_split": "select_tune",
        "selector_fit_split": "select_train",
        "objective_name": "J_cap",
        "predicted_objective": 0.75,
        "members": [
            {"layer": 16, "feature_id": 2},
            {"layer": 28, "feature_id": 4},
            {"layer": 28, "feature_id": 9},
        ],
        "layer_counts": {"16": 1, "28": 2},
        "construction_log": [
            {"step": 1, "layer": 16, "feature_id": 2, "objective_after": 0.25},
            {"step": 2, "layer": 28, "feature_id": 4, "objective_after": 0.50},
            {"step": 3, "layer": 28, "feature_id": 9, "objective_after": 0.75},
        ],
        "split_audit": {"no_test_leakage": True},
    }


def test_build_sensitivity_mask_payload_uses_construction_log_prefix() -> None:
    payload = build_sensitivity_mask_payload(
        source_payload=_source_payload(),
        source_mask_path="C:/tmp/capability_mask.json",
        source_mask_name="capability_mask",
        target_size=2,
        run_id="build_mask_size_sensitivity-123",
    )

    assert payload["mask_name"] == "capability_mask_k2"
    assert payload["size"] == 2
    assert payload["members"] == [{"layer": 16, "feature_id": 2}, {"layer": 28, "feature_id": 4}]
    assert payload["construction_log"][-1]["step"] == 2
    assert payload["source_mask_path"] == "C:/tmp/capability_mask.json"
    assert payload["subset_policy"] == "construction_log_prefix"


def test_run_mask_size_sensitivity_writes_nested_artifacts(tmp_path: Path) -> None:
    source_path = tmp_path / "capability_mask.json"
    source_path.write_text(json.dumps(_source_payload(), indent=2) + "\n", encoding="utf-8")
    config = {
        "stage_name": "build_mask_size_sensitivity",
        "seed": 20260326,
        "source_masks": [
            {
                "name": "capability_mask",
                "source_mask_path": source_path.as_posix(),
                "sizes": [1, 3],
            }
        ],
        "paths": {
            "mask_dir": (tmp_path / "masks").as_posix(),
            "summary_dir": (tmp_path / "metrics").as_posix(),
            "runtime_dir": (tmp_path / "runtime").as_posix(),
            "snapshot_dir": (tmp_path / "snapshots").as_posix(),
        },
    }

    result = run_mask_size_sensitivity(config)

    assert result["generated_mask_count"] == 2
    assert Path(result["summary_path"]).exists()
    assert Path(result["runtime_path"]).exists()
    assert Path(result["snapshot_path"]).exists()

    generated_files = sorted((tmp_path / "masks").glob("*.json"))
    assert len(generated_files) == 2
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in generated_files]
    assert [item["size"] for item in loaded] == [1, 3]
    assert all(item["stage_name"] == "build_mask_size_sensitivity" for item in loaded)


def test_write_sensitivity_variants_rejects_sizes_larger_than_source() -> None:
    source_payload = _source_payload()
    try:
        write_sensitivity_variants(
            source_payload=source_payload,
            source_mask_path="C:/tmp/capability_mask.json",
            source_mask_name="capability_mask",
            sizes=[4],
            output_dir="C:/tmp/masks",
            run_id="build_mask_size_sensitivity-123",
        )
    except ValueError as exc:
        assert "exceeds source mask size" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for oversize sensitivity mask")
