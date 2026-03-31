from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def select_smoke_completion_rows(rows: list[dict[str, Any]], *, records_per_split_slice: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["split"], item["slice"], item["prompt_id"])):
        grouped[(row["split"], row["slice"])].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        selected.extend(grouped[key][:records_per_split_slice])
    return selected


def build_teacher_forced_inputs(tokenizer, rendered_prefix: str, completion_token_ids: list[int]) -> tuple[torch.Tensor, int]:
    prefix_ids = tokenizer(rendered_prefix, add_special_tokens=True)["input_ids"]
    full_ids = prefix_ids + list(completion_token_ids)
    return torch.tensor(full_ids, dtype=torch.long), len(prefix_ids)


def answer_span_slice(*, answer_start: int, full_sequence_length: int) -> slice:
    if answer_start < 0 or answer_start > full_sequence_length:
        raise ValueError(f"Invalid answer_start={answer_start} for full_sequence_length={full_sequence_length}")
    return slice(answer_start, full_sequence_length)


def extract_hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        candidate = output[0]
    else:
        candidate = output
    if not isinstance(candidate, torch.Tensor):
        raise TypeError(f"Expected tensor layer output, got {type(candidate).__name__}")
    return candidate


def cache_metadata_row(
    *,
    completion_row: dict[str, Any],
    layer_index: int,
    token_index: int,
    answer_offset: int,
    seq_len_effective: int,
) -> dict[str, Any]:
    return {
        "prompt_id": completion_row["prompt_id"],
        "split": completion_row["split"],
        "slice": completion_row["slice"],
        "layer": layer_index,
        "token_index": token_index,
        "answer_offset": answer_offset,
        "seq_len_effective": seq_len_effective,
        "truncated": completion_row["stop_reason"] == "max_new_tokens",
        "eos_reached": completion_row["eos_reached"],
    }
