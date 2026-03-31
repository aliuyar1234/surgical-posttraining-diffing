from __future__ import annotations

from transformers import AutoTokenizer

from src.common.configs import load_yaml_config
from src.data.prompt_suite import PromptRecord, validate_prompt_suite


def test_prompt_suite_validator_accepts_balanced_records() -> None:
    config = load_yaml_config("configs/data.yaml")
    config["splits"] = {
        "train_unlabeled": 6,
        "select_train": 6,
        "select_tune": 6,
        "test": 6,
    }
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    records = []
    for split in ("train_unlabeled", "select_train", "select_tune", "test"):
        for slice_name in config["slices"]:
            checker = "alias_exact_match"
            if slice_name == "Format":
                checker = "exact_string_match"
            if slice_name in {"Harmful", "BenignAdjacent"}:
                checker = "rule_based_refusal"
            records.append(
                PromptRecord(
                    id=f"{slice_name.lower()}_{split}",
                    split=split,
                    slice=slice_name,
                    prompt=f"Simple prompt for {slice_name} {split}.",
                    gold="yes",
                    aliases=["yes"],
                    checker=checker,
                    target_len=1,
                    meta={"test": True},
                )
            )
    validate_prompt_suite(records, config, tokenizer)


def test_prompt_suite_validator_rejects_duplicate_ids() -> None:
    config = load_yaml_config("configs/data.yaml")
    config["splits"] = {
        "train_unlabeled": 6,
        "select_train": 6,
        "select_tune": 6,
        "test": 6,
    }
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    records = [
        PromptRecord(
            id="dup",
            split="train_unlabeled",
            slice="QA",
            prompt="Question one?",
            gold="yes",
            aliases=["yes"],
            checker="alias_exact_match",
            target_len=1,
            meta={},
        ),
        PromptRecord(
            id="dup",
            split="train_unlabeled",
            slice="Math",
            prompt="Question two?",
            gold="2",
            aliases=["2"],
            checker="numeric_exact_match",
            target_len=1,
            meta={},
        ),
    ]
    try:
        validate_prompt_suite(records, config, tokenizer)
    except ValueError as exc:
        assert "Duplicate prompt id" in str(exc)
    else:
        raise AssertionError("Expected duplicate-id validation failure")
