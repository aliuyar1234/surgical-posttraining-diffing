from __future__ import annotations

import json
from pathlib import Path

from src.common.configs import (
    ARTIFACT_ROOTS,
    CANONICAL_CONFIGS,
    CONFIGS_DIR,
    REPO_ROOT,
    build_artifact_path,
    build_run_id,
    load_config_bundle,
    load_yaml_config,
    save_resolved_config_snapshot,
    validate_config_name,
)
from src.common.runmeta import run_smoke


def test_canonical_configs_exist_and_parse() -> None:
    bundle = load_config_bundle()
    assert set(bundle) == set(CANONICAL_CONFIGS)

    for config_name in CANONICAL_CONFIGS:
        config_path = CONFIGS_DIR / config_name
        assert config_path.exists()
        payload = load_yaml_config(config_path)
        validate_config_name(config_name, payload)


def test_run_id_is_deterministic() -> None:
    payload = load_yaml_config(CONFIGS_DIR / "model_pair.yaml")
    first = build_run_id("m0-smoke", payload)
    second = build_run_id("m0-smoke", payload)
    assert first == second


def test_resolved_config_snapshot_saves_json(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "resolved.json"
    payload = load_config_bundle()
    save_resolved_config_snapshot(payload, snapshot_path)

    assert snapshot_path.exists()
    restored = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert restored["data.yaml"]["max_prompt_tokens"] == 192


def test_artifact_paths_land_under_expected_roots() -> None:
    for kind, relative_root in ARTIFACT_ROOTS.items():
        built = build_artifact_path(kind, "example", "run-123", ".json")
        assert built.parent == REPO_ROOT / relative_root


def test_smoke_run_writes_snapshot_and_runtime_report() -> None:
    output = run_smoke()

    snapshot_path = Path(output["snapshot_path"])
    runtime_path = Path(output["runtime_path"])

    assert output["run_id"].startswith("m0-smoke-")
    assert snapshot_path.exists()
    assert runtime_path.exists()
