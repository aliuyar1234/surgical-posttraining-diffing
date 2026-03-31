from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.cache.cache_io import load_cache_shard
from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.runmeta import collect_runtime_facts
from src.train.intervention import SparseDeltaIntervention
from src.train.sparse_delta import SparseDeltaModule, compute_r2


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a one-layer sparse delta module on cached answer-phase vectors.")
    parser.add_argument("--config", required=True, help="Path to sparse-delta training config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if "layers" in config:
        payload = run_sparse_delta_training_bundle(config)
    else:
        payload = run_sparse_delta_training(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_sparse_delta_training_bundle(config: dict[str, Any]) -> dict[str, Any]:
    bundle_payload = {
        "config": config,
        "layer_indices": [int(layer["layer_index"]) for layer in config["layers"]],
    }
    bundle_run_id = build_run_id(config["stage_name"], bundle_payload)
    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", bundle_run_id, ".json")
    save_resolved_config_snapshot({"config": config, "bundle_payload": bundle_payload}, snapshot_path)

    layer_results: list[dict[str, Any]] = []
    wall_clock_seconds = 0.0
    for layer_config in config["layers"]:
        single_config = {
            "stage_name": config["stage_name"],
            "seed": config["seed"],
            "source_cache": {
                "cache_run_id": layer_config["cache_run_id"],
                "layer_index": layer_config["layer_index"],
                "shard_name": layer_config["shard_name"],
                "train_split": layer_config["train_split"],
                "holdout_prompt_count": layer_config["holdout_prompt_count"],
            },
            "training": config["training"],
            "sparse_module": config["sparse_module"],
            "paths": config["paths"],
        }
        result = run_sparse_delta_training(single_config)
        runtime_payload = json.loads(Path(result["runtime_path"]).read_text(encoding="utf-8"))
        wall_clock_seconds += float(runtime_payload["wall_clock_seconds"])
        layer_results.append(
            {
                "layer_index": int(result["layer_index"]),
                "run_id": result["run_id"],
                "checkpoint_path": result["checkpoint_path"],
                "summary_path": result["summary_path"],
                "runtime_path": result["runtime_path"],
                "best_val_mse": float(result["best_val_mse"]),
                "initial_val_mse": float(result["initial_val_mse"]),
                "sanity_distance_reduction": float(result["sanity_distance_reduction"]),
            }
        )

    summary_path = build_artifact_path("metric", "sparse_delta_bundle", bundle_run_id, ".json")
    runtime_path = build_artifact_path("runtime", "runtime_report", bundle_run_id, ".json")
    summary_payload = {
        "run_id": bundle_run_id,
        "stage_name": config["stage_name"],
        "bundle_type": "multi_layer_sparse_delta",
        "layer_results": sorted(layer_results, key=lambda item: item["layer_index"]),
        "snapshot_path": snapshot_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime_payload = {
        "run_id": bundle_run_id,
        "stage_name": config["stage_name"],
        "bundle_type": "multi_layer_sparse_delta",
        "layer_run_ids": [item["run_id"] for item in layer_results],
        "wall_clock_seconds": wall_clock_seconds,
        "snapshot_path": snapshot_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "environment": collect_runtime_facts(),
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": bundle_run_id,
        "layer_results": summary_payload["layer_results"],
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
    }


def run_sparse_delta_training(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))

    cache_paths = resolve_cache_paths(config)
    shard = load_cache_shard(**cache_paths)
    split_payload = build_dataset_split(
        h_pt=shard["h_pt"],
        delta=shard["delta"],
        metadata=shard["metadata"],
        source_split=config["source_cache"]["train_split"],
        holdout_prompt_count=int(config["source_cache"]["holdout_prompt_count"]),
        seed=int(config["seed"]),
    )
    run_payload = {
        "config": config,
        "cache_paths": cache_paths,
        "layer_index": split_payload["layer_index"],
        "train_prompt_ids": split_payload["train_prompt_ids"],
        "val_prompt_ids": split_payload["val_prompt_ids"],
        "source_split": config["source_cache"]["train_split"],
    }
    run_id = build_run_id(config["stage_name"], run_payload)

    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")
    save_resolved_config_snapshot({"config": config, "run_payload": run_payload}, snapshot_path)

    device = choose_device(config["training"]["device"])
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    training_result = train_sparse_delta_model(
        train_h_pt=split_payload["train_h_pt"],
        train_delta=split_payload["train_delta"],
        val_h_pt=split_payload["val_h_pt"],
        val_delta=split_payload["val_delta"],
        sparse_config=config["sparse_module"],
        training_config=config["training"],
        seed=int(config["seed"]),
        device=device,
    )

    checkpoint_path = save_checkpoint(
        checkpoint_dir=Path(config["paths"]["checkpoint_dir"]),
        run_id=run_id,
        layer_index=split_payload["layer_index"],
        checkpoint_payload={
            "run_id": run_id,
            "stage_name": config["stage_name"],
            "layer_index": split_payload["layer_index"],
            "source_cache": config["source_cache"],
            "module_state": training_result["module"].state_dict(),
            "input_mean": training_result["input_mean"].cpu(),
            "input_std": training_result["input_std"].cpu(),
            "eps_std": float(training_result["eps_std"]),
            "decoder_column_norms": training_result["decoder_column_norms"].cpu(),
            "sparse_module": config["sparse_module"],
            "training": config["training"],
            "train_prompt_ids": split_payload["train_prompt_ids"],
            "val_prompt_ids": split_payload["val_prompt_ids"],
        },
    )

    summary_path = build_artifact_path("metric", "sparse_delta_training", run_id, ".json")
    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")

    summary_payload = build_summary_payload(
        run_id=run_id,
        config=config,
        split_payload=split_payload,
        training_result=training_result,
        checkpoint_path=checkpoint_path,
        snapshot_path=snapshot_path,
        summary_path=summary_path,
        runtime_path=runtime_path,
    )
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "layer_index": split_payload["layer_index"],
        "tokens_processed": int(split_payload["train_h_pt"].shape[0] * training_result["epochs_ran"]),
        "train_vectors": int(split_payload["train_h_pt"].shape[0]),
        "val_vectors": int(split_payload["val_h_pt"].shape[0]),
        "batch_size": int(config["training"]["batch_size"]),
        "device": device.type,
        "source_cache_run_id": config["source_cache"]["cache_run_id"],
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
        "layer_index": split_payload["layer_index"],
        "checkpoint_path": checkpoint_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
        "best_val_mse": training_result["best_val_mse"],
        "initial_val_mse": training_result["initial_val_mse"],
        "sanity_distance_reduction": training_result["sanity"]["mean_distance_reduction"],
    }


def resolve_cache_paths(config: dict[str, Any]) -> dict[str, str]:
    layer_index = int(config["source_cache"]["layer_index"])
    cache_dir = Path(config["paths"]["cache_dir"]) / f"layer_{layer_index}"
    shard_name = config["source_cache"]["shard_name"]
    cache_run_id = config["source_cache"]["cache_run_id"]
    return {
        "h_pt_path": (cache_dir / f"h_pt_{shard_name}_{cache_run_id}.pt").as_posix(),
        "delta_path": (cache_dir / f"delta_{shard_name}_{cache_run_id}.pt").as_posix(),
        "meta_path": (cache_dir / f"meta_{shard_name}_{cache_run_id}.parquet").as_posix(),
    }


def build_dataset_split(
    *,
    h_pt: torch.Tensor,
    delta: torch.Tensor,
    metadata: list[dict[str, Any]],
    source_split: str,
    holdout_prompt_count: int,
    seed: int,
) -> dict[str, Any]:
    if not (len(metadata) == h_pt.shape[0] == delta.shape[0]):
        raise ValueError("Cache tensor rows and metadata rows must match exactly")

    train_indices = [idx for idx, row in enumerate(metadata) if row["split"] == source_split]
    if not train_indices:
        raise ValueError(f"No cache rows found for split {source_split}")

    prompt_ids = sorted({metadata[idx]["prompt_id"] for idx in train_indices})
    if holdout_prompt_count <= 0 or holdout_prompt_count >= len(prompt_ids):
        raise ValueError("holdout_prompt_count must be between 1 and the number of source-split prompts - 1")

    rng = random.Random(seed)
    shuffled_prompt_ids = prompt_ids[:]
    rng.shuffle(shuffled_prompt_ids)
    val_prompt_ids = sorted(shuffled_prompt_ids[:holdout_prompt_count])
    val_prompt_set = set(val_prompt_ids)
    train_prompt_ids = sorted(prompt_id for prompt_id in prompt_ids if prompt_id not in val_prompt_set)

    train_rows = [idx for idx in train_indices if metadata[idx]["prompt_id"] in train_prompt_ids]
    val_rows = [idx for idx in train_indices if metadata[idx]["prompt_id"] in val_prompt_set]
    if not train_rows or not val_rows:
        raise ValueError("Both train and validation token splits must be non-empty")

    layer_values = {int(metadata[idx]["layer"]) for idx in train_indices}
    if len(layer_values) != 1:
        raise ValueError(f"Expected one layer in the shard, found {sorted(layer_values)}")
    layer_index = next(iter(layer_values))

    return {
        "layer_index": layer_index,
        "train_prompt_ids": train_prompt_ids,
        "val_prompt_ids": val_prompt_ids,
        "train_h_pt": h_pt[train_rows].to(torch.float32),
        "train_delta": delta[train_rows].to(torch.float32),
        "val_h_pt": h_pt[val_rows].to(torch.float32),
        "val_delta": delta[val_rows].to(torch.float32),
        "train_vector_count": len(train_rows),
        "val_vector_count": len(val_rows),
    }


def train_sparse_delta_model(
    *,
    train_h_pt: torch.Tensor,
    train_delta: torch.Tensor,
    val_h_pt: torch.Tensor,
    val_delta: torch.Tensor,
    sparse_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    if isinstance(device, str):
        device = choose_device(device)
    torch.manual_seed(seed)

    eps_std = float(training_config.get("eps_std", 1e-6))
    input_mean = train_h_pt.mean(dim=0)
    input_std = train_h_pt.std(dim=0, unbiased=False).clamp_min(eps_std)
    standardized_train = ((train_h_pt - input_mean) / (input_std + eps_std)).to(torch.float32)
    standardized_val = ((val_h_pt - input_mean) / (input_std + eps_std)).to(torch.float32)

    module = SparseDeltaModule(
        d_model=int(train_h_pt.shape[1]),
        width=int(sparse_config["width"]),
        top_k=int(sparse_config["top_k"]),
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(training_config["lr"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    batch_size = int(training_config["batch_size"])
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["early_stopping_patience"])

    train_x = standardized_train.to(device=device)
    train_y = train_delta.to(device=device)
    val_x = standardized_val.to(device=device)
    val_y = val_delta.to(device=device)

    history: list[dict[str, float]] = []
    with torch.no_grad():
        initial_train_mse = evaluate_mse(module, train_x, train_y)
        initial_val_mse = evaluate_mse(module, val_x, val_y)

    best_val_mse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        module.train()
        permutation = torch.randperm(train_x.shape[0], device=device)
        for start in range(0, train_x.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(training_config, device):
                prediction = module(train_x[batch_indices]).delta_hat
                loss = nn.functional.mse_loss(prediction, train_y[batch_indices])
            loss.backward()
            optimizer.step()

        module.eval()
        with torch.no_grad():
            train_mse = evaluate_mse(module, train_x, train_y)
            val_mse = evaluate_mse(module, val_x, val_y)
        history.append({"epoch": float(epoch), "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < (best_val_mse - 1e-12):
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = deepcopy(module.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                break

    if best_state is None:
        raise RuntimeError("Sparse delta training did not record a best checkpoint")
    module.load_state_dict(best_state)
    module.eval()

    with torch.no_grad():
        train_prediction = module(train_x).delta_hat
        val_prediction = module(val_x).delta_hat
    sanity = compute_sanity_panel(
        module=module,
        input_mean=input_mean,
        input_std=input_std,
        eps_std=eps_std,
        h_pt=val_h_pt,
        delta=val_delta,
        panel_size=int(training_config.get("sanity_panel_size", min(32, val_h_pt.shape[0]))),
    )

    return {
        "module": module.cpu(),
        "input_mean": input_mean.cpu(),
        "input_std": input_std.cpu(),
        "eps_std": eps_std,
        "decoder_column_norms": module.decoder_column_norms().cpu(),
        "history": history,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "initial_train_mse": initial_train_mse,
        "initial_val_mse": initial_val_mse,
        "best_train_mse": nn.functional.mse_loss(train_prediction.cpu(), train_delta).item(),
        "best_val_mse": nn.functional.mse_loss(val_prediction.cpu(), val_delta).item(),
        "best_val_r2": compute_r2(val_prediction.cpu(), val_delta),
        "sanity": sanity,
    }


def evaluate_mse(module: SparseDeltaModule, standardized_hidden: torch.Tensor, target_delta: torch.Tensor) -> float:
    prediction = module(standardized_hidden).delta_hat
    return nn.functional.mse_loss(prediction, target_delta).item()


def compute_sanity_panel(
    *,
    module: SparseDeltaModule,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    eps_std: float,
    h_pt: torch.Tensor,
    delta: torch.Tensor,
    panel_size: int,
) -> dict[str, float]:
    if h_pt.shape[0] == 0:
        raise ValueError("Sanity panel requires at least one validation vector")
    panel_size = min(int(panel_size), int(h_pt.shape[0]))
    pt_panel = h_pt[:panel_size]
    it_panel = pt_panel + delta[:panel_size]
    intervention = SparseDeltaIntervention(
        module=module,
        input_mean=input_mean,
        input_std=input_std,
        alpha=1.0,
        mask=None,
        eps_std=eps_std,
    )
    predicted_panel = intervention.apply(pt_panel)
    before = torch.linalg.vector_norm(it_panel - pt_panel, dim=-1)
    after = torch.linalg.vector_norm(it_panel - predicted_panel, dim=-1)
    return {
        "panel_size": panel_size,
        "mean_distance_before": before.mean().item(),
        "mean_distance_after": after.mean().item(),
        "mean_distance_reduction": (before.mean() - after.mean()).item(),
    }


def maybe_autocast(training_config: dict[str, Any], device: torch.device):
    dtype_name = str(training_config.get("dtype", "float32")).lower()
    if device.type == "cuda" and "bf16" in dtype_name:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def choose_device(requested_device: str) -> torch.device:
    requested = requested_device.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested cuda but no CUDA device is available")
    return torch.device(requested)


def save_checkpoint(*, checkpoint_dir: Path, run_id: str, layer_index: int, checkpoint_payload: dict[str, Any]) -> Path:
    layer_dir = checkpoint_dir / f"layer_{layer_index}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = layer_dir / f"sparse_delta_one_layer_{run_id}.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    return checkpoint_path


def build_summary_payload(
    *,
    run_id: str,
    config: dict[str, Any],
    split_payload: dict[str, Any],
    training_result: dict[str, Any],
    checkpoint_path: Path,
    snapshot_path: Path,
    summary_path: Path,
    runtime_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "layer_index": split_payload["layer_index"],
        "source_cache_run_id": config["source_cache"]["cache_run_id"],
        "source_split": config["source_cache"]["train_split"],
        "train_prompt_ids": split_payload["train_prompt_ids"],
        "val_prompt_ids": split_payload["val_prompt_ids"],
        "train_vectors": split_payload["train_vector_count"],
        "val_vectors": split_payload["val_vector_count"],
        "width": int(config["sparse_module"]["width"]),
        "top_k": int(config["sparse_module"]["top_k"]),
        "history": training_result["history"],
        "initial_train_mse": training_result["initial_train_mse"],
        "initial_val_mse": training_result["initial_val_mse"],
        "best_train_mse": training_result["best_train_mse"],
        "best_val_mse": training_result["best_val_mse"],
        "best_val_r2": training_result["best_val_r2"],
        "loss_fell": training_result["best_val_mse"] < training_result["initial_val_mse"],
        "sanity": training_result["sanity"],
        "decoder_column_norm_checksum": tensor_checksum(training_result["decoder_column_norms"]),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "snapshot_path": snapshot_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
    }


def tensor_checksum(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


if __name__ == "__main__":
    raise SystemExit(main())
