from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.runmeta import collect_runtime_facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate stage runtimes for the current evaluation path.")
    parser.add_argument("--config", required=True, help="Path to eval fidelity config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_runtime_report(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_runtime_report(config: dict[str, Any]) -> dict[str, Any]:
    runtime_inputs = [Path(path) for path in config["runtime_inputs"]]
    stage_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in runtime_inputs]
    run_payload = {
        "config": config,
        "runtime_inputs": [path.as_posix() for path in runtime_inputs],
    }
    run_id = build_run_id("runtime_total", run_payload)
    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")
    save_resolved_config_snapshot({"config": config, "runtime_inputs": [path.as_posix() for path in runtime_inputs]}, snapshot_path)

    total_seconds = sum(float(payload["wall_clock_seconds"]) for payload in stage_payloads)
    report_path = build_artifact_path("runtime", "runtime_total", run_id, ".json")
    report_payload = {
        "run_id": run_id,
        "stage_name": "runtime_total",
        "runtime_inputs": [path.as_posix() for path in runtime_inputs],
        "total_wall_clock_seconds": total_seconds,
        "total_wall_clock_hours": total_seconds / 3600.0,
        "gpu_budget_fraction": total_seconds / (24.0 * 3600.0),
        "environment": collect_runtime_facts(),
        "snapshot_path": snapshot_path.as_posix(),
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "report_path": report_path.as_posix(),
        "total_wall_clock_seconds": total_seconds,
        "gpu_budget_fraction": report_payload["gpu_budget_fraction"],
        "snapshot_path": snapshot_path.as_posix(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
