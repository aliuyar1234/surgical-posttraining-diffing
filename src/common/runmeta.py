from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from typing import Any

from .configs import (
    REPO_ROOT,
    build_artifact_path,
    build_run_id,
    load_config_bundle,
    save_resolved_config_snapshot,
)


def collect_runtime_facts() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repo_root": REPO_ROOT.as_posix(),
        "git_state": _detect_git_state(),
    }


def run_smoke() -> dict[str, str]:
    config_bundle = load_config_bundle()
    run_payload = {
        "milestone": "M0",
        "active_requirements": ["R1", "R12"],
        "configs": config_bundle,
        "environment": collect_runtime_facts(),
    }
    run_id = build_run_id("m0-smoke", run_payload)

    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")
    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")

    save_resolved_config_snapshot(run_payload, snapshot_path)
    save_resolved_config_snapshot(
        {
            "run_id": run_id,
            "stage_name": "m0-smoke",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "requirements": ["R1", "R12"],
            "snapshot_path": snapshot_path.as_posix(),
            "environment": collect_runtime_facts(),
        },
        runtime_path,
    )

    return {
        "run_id": run_id,
        "snapshot_path": snapshot_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run metadata smoke utility.")
    parser.add_argument("--smoke", action="store_true", help="Load canonical configs and emit a deterministic run id.")
    args = parser.parse_args()

    if not args.smoke:
        parser.error("No action requested. Use --smoke.")

    payload = run_smoke()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _detect_git_state() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "git-unavailable"
    if proc.returncode != 0:
        return "no-git"
    return proc.stdout.strip() or "unknown-git-state"


if __name__ == "__main__":
    raise SystemExit(main())
