from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch


def write_cache_shard(
    *,
    cache_dir: str | Path,
    layer_index: int,
    run_id: str,
    shard_name: str,
    h_pt: torch.Tensor,
    delta: torch.Tensor,
    metadata_rows: list[dict[str, Any]],
) -> dict[str, str]:
    layer_dir = Path(cache_dir) / f"layer_{layer_index}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    h_pt_path = layer_dir / f"h_pt_{shard_name}_{run_id}.pt"
    delta_path = layer_dir / f"delta_{shard_name}_{run_id}.pt"
    meta_path = layer_dir / f"meta_{shard_name}_{run_id}.parquet"

    torch.save(h_pt, h_pt_path)
    torch.save(delta, delta_path)
    pq.write_table(pa.Table.from_pylist(metadata_rows), meta_path)

    return {
        "h_pt_path": h_pt_path.as_posix(),
        "delta_path": delta_path.as_posix(),
        "meta_path": meta_path.as_posix(),
    }


def load_cache_shard(*, h_pt_path: str | Path, delta_path: str | Path, meta_path: str | Path) -> dict[str, Any]:
    metadata = pq.read_table(meta_path).to_pylist()
    return {
        "h_pt": torch.load(h_pt_path, map_location="cpu"),
        "delta": torch.load(delta_path, map_location="cpu"),
        "metadata": metadata,
    }
