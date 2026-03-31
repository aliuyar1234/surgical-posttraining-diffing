from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.analysis.common import layer_count_map, sorted_mask_members
from src.common.configs import build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.runmeta import collect_runtime_facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CPU-only mask-size sensitivity artifacts from frozen M5 masks.")
    parser.add_argument("--config", required=True, help="Path to the mask-size sensitivity config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_mask_size_sensitivity(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_mask_size_sensitivity(config: dict[str, Any]) -> dict[str, Any]:
    source_specs = normalize_source_specs(config["source_masks"])
    stage_name = str(config["stage_name"])
    run_payload = {
        "config": {
            "stage_name": stage_name,
            "seed": int(config.get("seed", 0)),
            "source_masks": [
                {
                    "name": spec["name"],
                    "source_mask_path": spec["source_mask_path"],
                    "sizes": list(spec["sizes"]),
                }
                for spec in source_specs
            ],
        }
    }
    run_id = build_run_id(stage_name, run_payload)

    snapshot_path = build_output_path(config["paths"]["snapshot_dir"], f"resolved_config_{run_id}.json")
    save_resolved_config_snapshot({"config": config, "run_id": run_id}, snapshot_path)

    start_time = time.perf_counter()
    generated_masks: list[dict[str, Any]] = []
    for spec in source_specs:
        source_payload = load_mask_payload(spec["source_mask_path"])
        generated_masks.extend(
            write_sensitivity_variants(
                source_payload=source_payload,
                source_mask_path=spec["source_mask_path"],
                source_mask_name=spec["name"],
                sizes=spec["sizes"],
                output_dir=config["paths"]["mask_dir"],
                run_id=run_id,
            )
        )

    summary_path = build_output_path(config["paths"]["summary_dir"], f"mask_size_sensitivity_{run_id}.json")
    runtime_path = build_output_path(config["paths"]["runtime_dir"], f"runtime_report_{run_id}.json")
    summary_payload = {
        "run_id": run_id,
        "stage_name": stage_name,
        "snapshot_path": snapshot_path.as_posix(),
        "generated_mask_count": len(generated_masks),
        "generated_masks": generated_masks,
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_payload = {
        "run_id": run_id,
        "stage_name": stage_name,
        "wall_clock_seconds": time.perf_counter() - start_time,
        "source_mask_count": len(source_specs),
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


def normalize_source_specs(raw_source_masks: list[Any]) -> list[dict[str, Any]]:
    if not raw_source_masks:
        raise ValueError("source_masks must not be empty")
    specs: list[dict[str, Any]] = []
    for raw in raw_source_masks:
        if not isinstance(raw, dict):
            raise TypeError(f"Unsupported source mask spec type: {type(raw).__name__}")
        missing = [key for key in ("name", "source_mask_path", "sizes") if key not in raw]
        if missing:
            raise ValueError(f"Source mask specs must include {missing}, got keys={sorted(raw)}")
        sizes = normalize_sizes(raw["sizes"])
        specs.append(
            {
                "name": str(raw["name"]),
                "source_mask_path": str(raw["source_mask_path"]),
                "sizes": sizes,
            }
        )
    return specs


def normalize_sizes(raw_sizes: list[Any]) -> list[int]:
    sizes = sorted({int(size) for size in raw_sizes})
    if not sizes:
        raise ValueError("sizes must not be empty")
    if sizes[0] <= 0:
        raise ValueError(f"Mask sizes must be positive, got {sizes}")
    return sizes


def load_mask_payload(mask_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(mask_path).read_text(encoding="utf-8"))


def build_sensitivity_mask_payload(
    *,
    source_payload: dict[str, Any],
    source_mask_path: str | Path,
    source_mask_name: str,
    target_size: int,
    run_id: str,
) -> dict[str, Any]:
    construction_log = list(source_payload.get("construction_log") or [])
    if not construction_log:
        raise ValueError(f"Source mask {source_mask_name!r} does not include a construction_log")
    if target_size > len(construction_log):
        raise ValueError(
            f"Requested size {target_size} exceeds the available construction log length {len(construction_log)}"
        )

    selected_log = [dict(entry) for entry in construction_log[:target_size]]
    selected_members = sorted_mask_members({"layer": entry["layer"], "feature_id": entry["feature_id"]} for entry in selected_log)
    output_mask_name = f"{source_mask_name}_k{target_size}"
    layer_counts = layer_count_map(selected_members)
    final_objective = selected_log[-1].get("objective_after") if selected_log else None
    return {
        "run_id": run_id,
        "stage_name": "build_mask_size_sensitivity",
        "mask_name": output_mask_name,
        "reference_type": "sensitivity_subset",
        "reference_mask_name": source_payload.get("mask_name"),
        "source_mask_name": source_payload.get("mask_name", source_mask_name),
        "source_mask_path": Path(source_mask_path).as_posix(),
        "source_selection_run_id": source_payload.get("run_id"),
        "feature_table_run_id": source_payload.get("feature_table_run_id"),
        "selection_split": source_payload.get("selection_split"),
        "selector_fit_split": source_payload.get("selector_fit_split"),
        "objective_name": source_payload.get("objective_name"),
        "predicted_objective": final_objective,
        "size": len(selected_members),
        "target_size": int(target_size),
        "source_size": len(source_payload.get("members", [])),
        "members": selected_members,
        "layer_counts": {str(layer): count for layer, count in layer_counts.items()},
        "construction_log": selected_log,
        "split_audit": source_payload.get("split_audit"),
        "subset_policy": "construction_log_prefix",
        "parent_mask_name": source_payload.get("mask_name"),
    }


def write_sensitivity_variants(
    *,
    source_payload: dict[str, Any],
    source_mask_path: str | Path,
    source_mask_name: str,
    sizes: list[int],
    output_dir: str | Path,
    run_id: str,
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    source_size = len(source_payload.get("members", []))
    for size in sizes:
        if size > source_size:
            raise ValueError(f"Requested sensitivity size {size} exceeds source mask size {source_size}")
        payload = build_sensitivity_mask_payload(
            source_payload=source_payload,
            source_mask_path=source_mask_path,
            source_mask_name=source_mask_name,
            target_size=size,
            run_id=run_id,
        )
        output_path = build_output_path(output_dir, f"{payload['mask_name']}_{run_id}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated.append(
            {
                "mask_name": payload["mask_name"],
                "target_size": size,
                "output_path": output_path.as_posix(),
                "source_mask_name": payload["source_mask_name"],
                "source_mask_path": payload["source_mask_path"],
            }
        )
    return generated


def build_output_path(output_dir: str | Path, filename: str) -> Path:
    return Path(output_dir) / filename


if __name__ == "__main__":
    raise SystemExit(main())
