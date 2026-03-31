from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

CANONICAL_CONFIGS: tuple[str, ...] = (
    "data.yaml",
    "model_pair.yaml",
    "cache.yaml",
    "delta_module.yaml",
    "interventions.yaml",
    "selectors.yaml",
    "eval.yaml",
)

REQUIRED_TOP_LEVEL_KEYS: dict[str, tuple[str, ...]] = {
    "data.yaml": (
        "stage_name",
        "seed",
        "prompt_template",
        "prompt_template_hash",
        "tokenizer_path",
        "max_prompt_tokens",
        "splits",
        "slices",
        "qa_source",
        "harm_policy",
        "long_prompt_policy",
        "paths",
    ),
    "model_pair.yaml": (
        "stage_name",
        "seed",
        "model_pair",
        "prompt_template_hash",
        "generation",
        "paths",
    ),
    "cache.yaml": (
        "stage_name",
        "seed",
        "cache_backend",
        "metadata_backend",
        "layers",
        "max_cache_vectors_per_layer",
        "paths",
    ),
    "delta_module.yaml": (
        "stage_name",
        "seed",
        "training",
        "sparse_module",
        "paths",
    ),
    "interventions.yaml": (
        "stage_name",
        "seed",
        "gate_search",
        "variants",
        "paths",
    ),
    "selectors.yaml": (
        "stage_name",
        "seed",
        "candidate_pool_size",
        "mask_caps",
        "selector_scores",
        "verbosity_subtraction",
        "paths",
    ),
    "eval.yaml": (
        "stage_name",
        "seed",
        "fidelity_split",
        "variants",
        "metrics",
        "paths",
    ),
}

ARTIFACT_ROOTS: dict[str, str] = {
    "completion": "artifacts/completions",
    "cache": "artifacts/cache",
    "checkpoint": "artifacts/checkpoints",
    "selector": "artifacts/selectors",
    "mask": "artifacts/masks",
    "config_snapshot": "artifacts/config_snapshots",
    "metric": "results/metrics",
    "table": "results/tables",
    "figure": "results/figures",
    "example": "results/examples",
    "runtime": "results/runtime",
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping at {config_path}, got {type(payload).__name__}")
    return _resolve_paths(payload, base_dir=config_path.parent)


def validate_config_name(config_name: str, payload: Mapping[str, Any]) -> None:
    required = REQUIRED_TOP_LEVEL_KEYS[config_name]
    missing = [key for key in required if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{config_name} is missing required keys: {joined}")


def load_config_bundle(config_dir: str | Path = CONFIGS_DIR) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for config_name in CANONICAL_CONFIGS:
        config_path = Path(config_dir) / config_name
        payload = load_yaml_config(config_path)
        validate_config_name(config_name, payload)
        bundle[config_name] = payload
    return bundle


def build_run_id(stage_name: str, payload: Mapping[str, Any]) -> str:
    canonical = _canonicalize(payload)
    encoded = json.dumps({"stage_name": stage_name, "payload": canonical}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    return f"{stage_name}-{digest}"


def build_artifact_path(kind: str, stem: str, run_id: str, suffix: str) -> Path:
    if kind not in ARTIFACT_ROOTS:
        raise KeyError(f"Unknown artifact kind: {kind}")
    target_root = REPO_ROOT / ARTIFACT_ROOTS[kind]
    return target_root / f"{stem}_{run_id}{suffix}"


def save_resolved_config_snapshot(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_canonicalize(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _resolve_paths(value: Any, *, base_dir: Path, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_paths(item, base_dir=base_dir, parent_key=key) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir=base_dir, parent_key=parent_key) for item in value]
    if isinstance(value, str) and _looks_like_path_key(parent_key):
        if _should_preserve_model_identifier(value=value, parent_key=parent_key):
            return value
        maybe_path = Path(value)
        if maybe_path.is_absolute():
            return maybe_path.as_posix()
        return (base_dir / maybe_path).resolve().as_posix()
    return value


def _looks_like_path_key(parent_key: str | None) -> bool:
    if parent_key is None:
        return False
    normalized = parent_key.lower()
    return (
        normalized == "path"
        or normalized == "runtime_inputs"
        or normalized.endswith("_path")
        or normalized.endswith("_paths")
        or normalized.endswith("_dir")
    )


def _should_preserve_model_identifier(*, value: str, parent_key: str | None) -> bool:
    if parent_key not in {"pt_path", "it_path", "tokenizer_path"}:
        return False
    normalized = value.strip()
    if not normalized:
        return False
    drive, _ = os.path.splitdrive(normalized)
    if drive:
        return False
    if normalized.startswith(("/", "\\", "./", "../", "~")):
        return False
    return "/" in normalized and "\\" not in normalized


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value
