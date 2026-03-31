from __future__ import annotations

from src.common.configs import load_yaml_config
from src.data.rendering import compute_template_hash, render_full_sequence, render_neutral_prefix


def test_template_hash_matches_config() -> None:
    config = load_yaml_config("configs/data.yaml")
    assert compute_template_hash(config["prompt_template"]) == config["prompt_template_hash"]


def test_rendering_contract_uses_neutral_template() -> None:
    config = load_yaml_config("configs/data.yaml")
    prefix = render_neutral_prefix("Name the capital of France.", config["prompt_template"])
    full = render_full_sequence("Name the capital of France.", "Paris", config["prompt_template"])
    assert prefix == "Instruction:\nName the capital of France.\n\nResponse:\n"
    assert full.endswith("Response:\nParis")
