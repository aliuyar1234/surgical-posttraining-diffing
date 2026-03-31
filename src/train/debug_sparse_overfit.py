from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.cache.cache_io import load_cache_shard
from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.runmeta import collect_runtime_facts
from src.train.sparse_delta import SparseDeltaModule, compute_r2
from src.train.train_sparse_delta import file_sha256, peak_memory_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny real-data sparse overfit diagnostic against one cache shard.")
    parser.add_argument("--config", required=True, help="Path to debug sparse-overfit config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_debug_sparse_overfit(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_debug_sparse_overfit(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))

    cache_paths = resolve_cache_paths(config)
    shard = load_cache_shard(**cache_paths)
    probe = select_probe_subset(
        h_pt=shard["h_pt"],
        delta=shard["delta"],
        metadata=shard["metadata"],
        source_split=config["probe"]["split"],
        prompt_id=config["probe"]["prompt_id"],
        max_vectors=int(config["probe"]["max_vectors"]),
    )
    run_payload = {
        "config": config,
        "cache_paths": cache_paths,
        "layer_index": probe["layer_index"],
        "prompt_id": probe["prompt_id"],
        "vector_count": probe["vector_count"],
    }
    run_id = build_run_id(config["stage_name"], run_payload)
    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")
    save_resolved_config_snapshot({"config": config, "run_payload": run_payload}, snapshot_path)

    device = choose_device(config["training"]["device"])
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    result = run_overfit_probe(
        h_pt=probe["h_pt"],
        delta=probe["delta"],
        sparse_config=config["sparse_module"],
        training_config=config["training"],
        seed=int(config["seed"]),
        device=device,
    )

    checkpoint_path = save_debug_checkpoint(
        checkpoint_dir=Path(config["paths"]["checkpoint_dir"]),
        run_id=run_id,
        layer_index=probe["layer_index"],
        payload={
            "run_id": run_id,
            "stage_name": config["stage_name"],
            "layer_index": probe["layer_index"],
            "prompt_id": probe["prompt_id"],
            "vector_count": probe["vector_count"],
            "module_state": result["module"].state_dict(),
            "input_mean": result["input_mean"].cpu(),
            "input_std": result["input_std"].cpu(),
            "eps_std": float(result["eps_std"]),
            "mean_diff": result["mean_diff"].cpu(),
            "sparse_module": config["sparse_module"],
            "training": config["training"],
        },
    )

    summary_path = build_artifact_path("metric", "debug_sparse_overfit", run_id, ".json")
    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")
    summary_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "layer_index": probe["layer_index"],
        "source_cache_run_id": config["probe"]["cache_run_id"],
        "prompt_id": probe["prompt_id"],
        "vector_count": probe["vector_count"],
        "width": int(config["sparse_module"]["width"]),
        "top_k": int(config["sparse_module"]["top_k"]),
        "max_steps": int(config["training"]["max_steps"]),
        "batch_size": int(config["training"]["batch_size"]),
        "initial_train_mse": result["initial_train_mse"],
        "best_train_mse": result["best_train_mse"],
        "final_train_mse": result["final_train_mse"],
        "best_train_r2": result["best_train_r2"],
        "mean_diff_mse": result["mean_diff_mse"],
        "mean_diff_r2": result["mean_diff_r2"],
        "best_vs_mean_diff_mse_gap": result["mean_diff_mse"] - result["best_train_mse"],
        "best_beats_mean_diff": result["best_train_mse"] < result["mean_diff_mse"],
        "history_head": result["history"][:5],
        "history_tail": result["history"][-5:],
        "sanity": result["sanity"],
        "mean_diff_sanity": result["mean_diff_sanity"],
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "snapshot_path": snapshot_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "layer_index": probe["layer_index"],
        "prompt_id": probe["prompt_id"],
        "vector_count": probe["vector_count"],
        "device": device.type,
        "batch_size": int(config["training"]["batch_size"]),
        "max_steps": int(config["training"]["max_steps"]),
        "peak_memory_bytes": peak_memory_bytes(device),
        "wall_clock_seconds": time.perf_counter() - start_time,
        "snapshot_path": snapshot_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "environment": collect_runtime_facts(),
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "prompt_id": probe["prompt_id"],
        "vector_count": probe["vector_count"],
        "best_train_mse": result["best_train_mse"],
        "mean_diff_mse": result["mean_diff_mse"],
        "best_beats_mean_diff": result["best_train_mse"] < result["mean_diff_mse"],
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
    }


def resolve_cache_paths(config: dict[str, Any]) -> dict[str, str]:
    layer_index = int(config["probe"]["layer_index"])
    cache_dir = Path(config["paths"]["cache_dir"]) / f"layer_{layer_index}"
    shard_name = config["probe"]["shard_name"]
    cache_run_id = config["probe"]["cache_run_id"]
    return {
        "h_pt_path": (cache_dir / f"h_pt_{shard_name}_{cache_run_id}.pt").as_posix(),
        "delta_path": (cache_dir / f"delta_{shard_name}_{cache_run_id}.pt").as_posix(),
        "meta_path": (cache_dir / f"meta_{shard_name}_{cache_run_id}.parquet").as_posix(),
    }


def select_probe_subset(
    *,
    h_pt: torch.Tensor,
    delta: torch.Tensor,
    metadata: list[dict[str, Any]],
    source_split: str,
    prompt_id: str,
    max_vectors: int,
) -> dict[str, Any]:
    matching = [
        (idx, row)
        for idx, row in enumerate(metadata)
        if row["split"] == source_split and row["prompt_id"] == prompt_id
    ]
    if not matching:
        raise ValueError(f"No cache rows found for prompt_id={prompt_id!r} in split={source_split!r}")
    matching.sort(key=lambda item: (int(item[1]["answer_offset"]), int(item[1]["token_index"])))
    selected_indices = [idx for idx, _row in matching[:max_vectors]]
    layer_values = {int(metadata[idx]["layer"]) for idx in selected_indices}
    if len(layer_values) != 1:
        raise ValueError(f"Expected one layer in the selected subset, found {sorted(layer_values)}")
    layer_index = next(iter(layer_values))
    return {
        "prompt_id": prompt_id,
        "layer_index": layer_index,
        "vector_count": len(selected_indices),
        "h_pt": h_pt[selected_indices].to(torch.float32),
        "delta": delta[selected_indices].to(torch.float32),
    }


def run_overfit_probe(
    *,
    h_pt: torch.Tensor,
    delta: torch.Tensor,
    sparse_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    if isinstance(device, str):
        device = choose_device(device)
    torch.manual_seed(seed)

    eps_std = float(training_config.get("eps_std", 1e-6))
    input_mean = h_pt.mean(dim=0)
    input_std = h_pt.std(dim=0, unbiased=False).clamp_min(eps_std)
    standardized = ((h_pt - input_mean) / (input_std + eps_std)).to(torch.float32)
    mean_diff = delta.mean(dim=0)

    module = SparseDeltaModule(
        d_model=int(h_pt.shape[1]),
        width=int(sparse_config["width"]),
        top_k=int(sparse_config["top_k"]),
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(training_config["lr"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    train_x = standardized.to(device=device)
    train_y = delta.to(device=device)
    batch_size = int(training_config["batch_size"])
    max_steps = int(training_config["max_steps"])

    with torch.no_grad():
        initial_train_mse = nn.functional.mse_loss(module(train_x).delta_hat, train_y).item()

    best_train_mse = initial_train_mse
    best_state = {key: value.detach().clone() for key, value in module.state_dict().items()}
    history: list[dict[str, float]] = []

    for step in range(1, max_steps + 1):
        module.train()
        batch_indices = torch.randint(0, train_x.shape[0], size=(batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = module(train_x[batch_indices]).delta_hat
        loss = nn.functional.mse_loss(prediction, train_y[batch_indices])
        loss.backward()
        optimizer.step()

        if step == 1 or step % int(training_config["log_every_steps"]) == 0 or step == max_steps:
            module.eval()
            with torch.no_grad():
                train_mse = nn.functional.mse_loss(module(train_x).delta_hat, train_y).item()
            history.append({"step": float(step), "train_mse": train_mse})
            if train_mse < best_train_mse:
                best_train_mse = train_mse
                best_state = {key: value.detach().clone() for key, value in module.state_dict().items()}

    module.load_state_dict(best_state)
    module.eval()
    with torch.no_grad():
        best_prediction = module(train_x).delta_hat.cpu()
        final_train_mse = nn.functional.mse_loss(best_prediction, delta).item()

    mean_diff_prediction = mean_diff.unsqueeze(0).expand_as(delta)
    sanity = compute_sanity_panel(
        h_pt=h_pt,
        delta=delta,
        variant_delta=best_prediction,
    )
    mean_diff_sanity = compute_sanity_panel(
        h_pt=h_pt,
        delta=delta,
        variant_delta=mean_diff_prediction,
    )

    return {
        "module": module.cpu(),
        "input_mean": input_mean.cpu(),
        "input_std": input_std.cpu(),
        "eps_std": eps_std,
        "mean_diff": mean_diff.cpu(),
        "initial_train_mse": initial_train_mse,
        "best_train_mse": best_train_mse,
        "final_train_mse": final_train_mse,
        "best_train_r2": compute_r2(best_prediction, delta),
        "mean_diff_mse": nn.functional.mse_loss(mean_diff_prediction, delta).item(),
        "mean_diff_r2": compute_r2(mean_diff_prediction, delta),
        "history": history,
        "sanity": sanity,
        "mean_diff_sanity": mean_diff_sanity,
    }


def compute_sanity_panel(*, h_pt: torch.Tensor, delta: torch.Tensor, variant_delta: torch.Tensor) -> dict[str, float]:
    it_hidden = h_pt + delta
    variant_hidden = h_pt + variant_delta
    before = torch.linalg.vector_norm(it_hidden - h_pt, dim=-1)
    after = torch.linalg.vector_norm(it_hidden - variant_hidden, dim=-1)
    return {
        "mean_distance_before": before.mean().item(),
        "mean_distance_after": after.mean().item(),
        "mean_distance_reduction": (before.mean() - after.mean()).item(),
    }


def choose_device(requested_device: str) -> torch.device:
    requested = requested_device.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested cuda but no CUDA device is available")
    return torch.device(requested)


def save_debug_checkpoint(*, checkpoint_dir: Path, run_id: str, layer_index: int, payload: dict[str, Any]) -> Path:
    layer_dir = checkpoint_dir / f"layer_{layer_index}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = layer_dir / f"debug_sparse_overfit_{run_id}.pt"
    torch.save(payload, checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    raise SystemExit(main())
