from __future__ import annotations

import pytest
import torch

from src.eval.common import add_recovery_metrics, aggregate_variant_metrics, answer_token_kl


def test_answer_token_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor(
        [[[1.0, 0.0], [0.5, -0.5], [0.2, -0.2]]],
        dtype=torch.float32,
    )
    kl, count = answer_token_kl(logits, logits.clone(), answer_start=1)
    assert kl == 0.0
    assert count == 2


def test_add_recovery_metrics_uses_pt_and_it_baselines() -> None:
    pt_examples = [
        {"slice": "QA", "passed": True, "token_len": 4, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Math", "passed": False, "token_len": 4, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Format", "passed": False, "token_len": 4, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Brevity", "passed": True, "token_len": 6, "brevity_excess_tokens": 4, "refused": False},
    ]
    it_examples = [
        {"slice": "QA", "passed": True, "token_len": 3, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Math", "passed": True, "token_len": 3, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Format", "passed": True, "token_len": 3, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Brevity", "passed": True, "token_len": 2, "brevity_excess_tokens": 0, "refused": False},
    ]
    full_examples = [
        {"slice": "QA", "passed": True, "token_len": 3, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Math", "passed": True, "token_len": 3, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Format", "passed": False, "token_len": 3, "brevity_excess_tokens": 0, "refused": False},
        {"slice": "Brevity", "passed": True, "token_len": 4, "brevity_excess_tokens": 2, "refused": False},
    ]

    metrics = {
        "PT": aggregate_variant_metrics(pt_examples),
        "IT_neutral": aggregate_variant_metrics(it_examples),
        "PT_plus_FullDelta": aggregate_variant_metrics(full_examples),
    }
    add_recovery_metrics(metrics)

    assert metrics["PT"]["Cap_Recovery"] == 0.0
    assert metrics["IT_neutral"]["Cap_Recovery"] == pytest.approx(1.0)
    assert 0.0 < metrics["PT_plus_FullDelta"]["Cap_Recovery"] < 1.0
