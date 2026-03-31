from __future__ import annotations

from pathlib import Path

import torch

from src.cache.cache_io import load_cache_shard, write_cache_shard
from src.cache.cache_utils import select_smoke_completion_rows


def test_select_smoke_completion_rows_is_stable_per_split_slice() -> None:
    rows = [
        {"prompt_id": "b", "split": "test", "slice": "QA"},
        {"prompt_id": "a", "split": "test", "slice": "QA"},
        {"prompt_id": "c", "split": "test", "slice": "Math"},
        {"prompt_id": "d", "split": "test", "slice": "Math"},
    ]
    selected = select_smoke_completion_rows(rows, records_per_split_slice=1)
    assert [row["prompt_id"] for row in selected] == ["c", "a"]


def test_cache_shard_round_trip(tmp_path: Path) -> None:
    h_pt = torch.randn(4, 8, dtype=torch.bfloat16)
    delta = torch.randn(4, 8, dtype=torch.bfloat16)
    metadata = [
        {
            "prompt_id": "qa_test_0001",
            "split": "test",
            "slice": "QA",
            "layer": 28,
            "token_index": 12,
            "answer_offset": 0,
            "seq_len_effective": 20,
            "truncated": False,
            "eos_reached": True,
        }
    ]
    paths = write_cache_shard(
        cache_dir=tmp_path,
        layer_index=28,
        run_id="run123",
        shard_name="late_smoke",
        h_pt=h_pt,
        delta=delta,
        metadata_rows=metadata,
    )
    restored = load_cache_shard(
        h_pt_path=paths["h_pt_path"],
        delta_path=paths["delta_path"],
        meta_path=paths["meta_path"],
    )
    assert restored["h_pt"].shape == h_pt.shape
    assert restored["delta"].shape == delta.shape
    assert torch.equal(restored["h_pt"], h_pt)
    assert torch.equal(restored["delta"], delta)
    assert restored["metadata"] == metadata
