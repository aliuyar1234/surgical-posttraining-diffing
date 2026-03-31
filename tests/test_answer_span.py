from __future__ import annotations

import torch

from src.cache.cache_utils import answer_span_slice, build_teacher_forced_inputs


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True):
        base = [11, 12, 13] if add_special_tokens else [12, 13]
        return {"input_ids": base}


def test_build_teacher_forced_inputs_keeps_exact_frozen_completion_ids() -> None:
    tokenizer = _FakeTokenizer()
    full_ids, answer_start = build_teacher_forced_inputs(tokenizer, "prefix", [99, 100, 101, 1, 1])
    assert answer_start == 3
    assert torch.equal(full_ids, torch.tensor([11, 12, 13, 99, 100, 101, 1, 1], dtype=torch.long))


def test_answer_span_slice_uses_full_frozen_answer_length() -> None:
    span = answer_span_slice(answer_start=3, full_sequence_length=8)
    assert span.start == 3
    assert span.stop == 8
