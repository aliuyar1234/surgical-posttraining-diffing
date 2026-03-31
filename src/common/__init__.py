"""Shared config and run metadata utilities."""

from importlib import import_module

from .configs import (
    CANONICAL_CONFIGS,
    REPO_ROOT,
    build_artifact_path,
    build_run_id,
    load_config_bundle,
    load_yaml_config,
    save_resolved_config_snapshot,
    validate_config_name,
)
from .jsonl import read_jsonl, write_jsonl

_MODELING_EXPORTS = {
    "assert_tokenizer_compatibility",
    "get_decoder_layers",
    "get_text_model",
    "locked_layer_indices",
    "late_layer_index",
    "load_causal_model",
    "load_tokenizer",
    "mid_layer_index",
}

__all__ = [
    "CANONICAL_CONFIGS",
    "REPO_ROOT",
    "assert_tokenizer_compatibility",
    "build_artifact_path",
    "build_run_id",
    "get_decoder_layers",
    "get_text_model",
    "locked_layer_indices",
    "late_layer_index",
    "load_config_bundle",
    "load_causal_model",
    "load_tokenizer",
    "load_yaml_config",
    "mid_layer_index",
    "read_jsonl",
    "save_resolved_config_snapshot",
    "validate_config_name",
    "write_jsonl",
]


def __getattr__(name: str):
    if name not in _MODELING_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    modeling = import_module("src.common.modeling")
    value = getattr(modeling, name)
    globals()[name] = value
    return value
