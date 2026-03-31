from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from src.cache.cache_io import load_cache_shard, write_cache_shard
from src.cache.cache_utils import (
    answer_span_slice,
    build_teacher_forced_inputs,
    cache_metadata_row,
    extract_hidden_from_layer_output,
    select_smoke_completion_rows,
)
from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.jsonl import read_jsonl
from src.common.modeling import (
    assert_tokenizer_compatibility,
    get_decoder_layers,
    late_layer_index,
    load_causal_model,
    load_tokenizer,
    locked_layer_indices,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paired answer-phase activation caches.")
    parser.add_argument("--config", required=True, help="Path to cache config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_cache_build(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_cache_build(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("selected_layer", {}).get("mode") == "late_only":
        return run_smoke_cache_build(config)
    return run_locked_layer_cache_build(config)


def run_smoke_cache_build(config: dict[str, Any]) -> dict[str, Any]:
    data_config = load_yaml_config(config["data_config_path"])
    model_config = load_yaml_config(config["model_config_path"])

    rows = _load_completion_rows(config, data_config["slices"], splits=None)
    smoke_rows = select_smoke_completion_rows(rows, records_per_split_slice=int(config["smoke"]["records_per_split_slice"]))
    run_payload = {
        "config": config,
        "completion_run_id": config["completion_run_id"],
        "smoke_prompt_ids": [row["prompt_id"] for row in smoke_rows],
    }
    run_id = build_run_id(config["stage_name"], run_payload)
    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")
    save_resolved_config_snapshot({"config": config, "data_config": data_config, "model_config": model_config}, snapshot_path)

    tokenizer, pt_model, it_model = _load_models_and_tokenizer(config, model_config)
    layer_index = late_layer_index(it_model)
    if config["selected_layer"]["mode"] != "late_only":
        raise ValueError("M2 smoke config must use selected_layer.mode = late_only")

    payload = _collect_layer_cache(
        rows=smoke_rows,
        tokenizer=tokenizer,
        pt_model=pt_model,
        it_model=it_model,
        layer_indices=[layer_index],
        cache_dir=config["paths"]["cache_dir"],
        shard_name_by_layer={layer_index: config["shard"]["name"]},
        run_id=run_id,
        max_cache_vectors_per_layer=None,
        seed=int(config.get("seed", 0)),
    )

    verification_summary = {
        "all_teacher_forced_input_ids_identical": True,
        "all_answer_token_counts_match_completion_counts": True,
        "reload_verification_passed": True,
        "records_cached": len(smoke_rows),
        "answer_vectors_cached": payload["vector_counts"][layer_index],
        "d_model": payload["d_model"],
        "layer_index": layer_index,
        "verification_examples": payload["verification_examples"][:5],
    }
    verification_path = Path(config["paths"]["cache_dir"]) / f"cache_one_layer_summary_{run_id}.json"
    verification_path.write_text(json.dumps(verification_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "records_cached": len(smoke_rows),
        "answer_vectors_cached": payload["vector_counts"][layer_index],
        "layer_index": layer_index,
        "completion_run_id": config["completion_run_id"],
        "wall_clock_seconds": payload["wall_clock_seconds"],
        "device": config["runtime"]["device"],
        "snapshot_path": snapshot_path.as_posix(),
    }
    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "layer_index": layer_index,
        "records_cached": len(smoke_rows),
        "answer_vectors_cached": payload["vector_counts"][layer_index],
        "shard_paths": payload["shard_paths_by_layer"][layer_index],
        "verification_path": verification_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
    }


def run_locked_layer_cache_build(config: dict[str, Any]) -> dict[str, Any]:
    data_config = load_yaml_config(config["data_config_path"])
    model_config = load_yaml_config(config["model_config_path"])
    included_splits = tuple(config["include_splits"])
    rows = _load_completion_rows(config, data_config["slices"], splits=included_splits)
    run_payload = {
        "config": config,
        "completion_run_id": config["completion_run_id"],
        "included_splits": list(included_splits),
        "record_count": len(rows),
    }
    run_id = build_run_id(config["stage_name"], run_payload)
    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")
    save_resolved_config_snapshot({"config": config, "data_config": data_config, "model_config": model_config}, snapshot_path)

    tokenizer, pt_model, it_model = _load_models_and_tokenizer(config, model_config)
    if config["layers"]["selection_rule"] != "locked_mid_and_late_resid_post":
        raise ValueError("M4 cache config must use locked_mid_and_late_resid_post")
    mid_layer, late_layer = locked_layer_indices(it_model)
    shard_name_by_layer = {
        mid_layer: f"{config['shard']['stem']}_mid",
        late_layer: f"{config['shard']['stem']}_late",
    }
    payload = _collect_layer_cache(
        rows=rows,
        tokenizer=tokenizer,
        pt_model=pt_model,
        it_model=it_model,
        layer_indices=[mid_layer, late_layer],
        cache_dir=config["paths"]["cache_dir"],
        shard_name_by_layer=shard_name_by_layer,
        run_id=run_id,
        max_cache_vectors_per_layer=int(config["max_cache_vectors_per_layer"]),
        seed=int(config["seed"]),
    )

    verification_summary = {
        "all_teacher_forced_input_ids_identical": True,
        "all_answer_token_counts_match_completion_counts": True,
        "reload_verification_passed": True,
        "completion_run_id": config["completion_run_id"],
        "records_cached": len(rows),
        "d_model": payload["d_model"],
        "included_splits": list(included_splits),
        "layer_indices": [mid_layer, late_layer],
        "vector_counts_by_layer": {str(layer): count for layer, count in payload["vector_counts"].items()},
        "verification_examples": payload["verification_examples"][:8],
        "shard_paths_by_layer": {str(layer): paths for layer, paths in payload["shard_paths_by_layer"].items()},
    }
    verification_path = Path(config["paths"]["cache_dir"]) / f"cache_summary_{run_id}.json"
    verification_path.write_text(json.dumps(verification_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_payload = {
        "run_id": run_id,
        "stage_name": config["stage_name"],
        "completion_run_id": config["completion_run_id"],
        "included_splits": list(included_splits),
        "records_cached": len(rows),
        "vector_counts_by_layer": {str(layer): count for layer, count in payload["vector_counts"].items()},
        "wall_clock_seconds": payload["wall_clock_seconds"],
        "device": config["runtime"]["device"],
        "snapshot_path": snapshot_path.as_posix(),
        "verification_path": verification_path.as_posix(),
    }
    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "layer_indices": [mid_layer, late_layer],
        "records_cached": len(rows),
        "vector_counts_by_layer": payload["vector_counts"],
        "shard_paths_by_layer": payload["shard_paths_by_layer"],
        "verification_path": verification_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
    }


def _load_models_and_tokenizer(config: dict[str, Any], model_config: dict[str, Any]):
    assert_tokenizer_compatibility(model_config["model_pair"]["pt_path"], model_config["model_pair"]["it_path"])
    tokenizer = load_tokenizer(model_config["model_pair"]["it_path"])
    pt_model = load_causal_model(
        model_config["model_pair"]["pt_path"],
        device=config["runtime"]["device"],
        dtype_name=config["runtime"]["dtype"],
    )
    it_model = load_causal_model(
        model_config["model_pair"]["it_path"],
        device=config["runtime"]["device"],
        dtype_name=config["runtime"]["dtype"],
    )
    return tokenizer, pt_model, it_model


def _collect_layer_cache(
    *,
    rows: list[dict[str, Any]],
    tokenizer,
    pt_model,
    it_model,
    layer_indices: list[int],
    cache_dir: str,
    shard_name_by_layer: dict[int, str],
    run_id: str,
    max_cache_vectors_per_layer: int | None,
    seed: int,
) -> dict[str, Any]:
    pt_layers = get_decoder_layers(pt_model)
    it_layers = get_decoder_layers(it_model)
    pt_capture: dict[int, torch.Tensor] = {}
    it_capture: dict[int, torch.Tensor] = {}
    h_pt_rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_indices}
    delta_rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_indices}
    metadata_rows: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layer_indices}
    verification_examples: list[dict[str, Any]] = []

    def make_hook(bucket: dict[int, torch.Tensor], layer_index: int):
        def hook(_module, _inputs, output):
            bucket[layer_index] = extract_hidden_from_layer_output(output).detach()

        return hook

    handles = []
    for layer_index in layer_indices:
        handles.append(pt_layers[layer_index].register_forward_hook(make_hook(pt_capture, layer_index)))
        handles.append(it_layers[layer_index].register_forward_hook(make_hook(it_capture, layer_index)))

    start_time = time.perf_counter()
    try:
        for row in rows:
            input_ids, answer_start = build_teacher_forced_inputs(tokenizer, row["rendered_prefix"], row["completion_token_ids"])
            attention_mask = torch.ones_like(input_ids)
            if answer_start != int(row["prompt_token_count"]):
                raise ValueError(
                    f"Prompt token count mismatch for {row['prompt_id']}: answer_start={answer_start}, stored={row['prompt_token_count']}"
                )
            answer_slice = answer_span_slice(answer_start=answer_start, full_sequence_length=int(input_ids.shape[0]))
            if (answer_slice.stop - answer_slice.start) != int(row["answer_token_count"]):
                raise ValueError(
                    f"Answer-token count mismatch for {row['prompt_id']}: slice={answer_slice.stop - answer_slice.start}, stored={row['answer_token_count']}"
                )

            model_inputs = {
                "input_ids": input_ids.unsqueeze(0).to(pt_model.device),
                "attention_mask": attention_mask.unsqueeze(0).to(pt_model.device),
                "use_cache": False,
            }
            with torch.no_grad():
                pt_model(**model_inputs)
                it_model(**model_inputs)

            for layer_index in layer_indices:
                pt_hidden = pt_capture[layer_index][0].to(torch.float32)
                it_hidden = it_capture[layer_index][0].to(torch.float32)
                answer_h_pt = pt_hidden[answer_slice].cpu().to(torch.bfloat16)
                answer_delta = (it_hidden[answer_slice] - pt_hidden[answer_slice]).cpu().to(torch.bfloat16)
                h_pt_rows[layer_index].append(answer_h_pt)
                delta_rows[layer_index].append(answer_delta)
                for answer_offset in range(answer_h_pt.shape[0]):
                    metadata_rows[layer_index].append(
                        cache_metadata_row(
                            completion_row=row,
                            layer_index=layer_index,
                            token_index=answer_start + answer_offset,
                            answer_offset=answer_offset,
                            seq_len_effective=int(input_ids.shape[0]),
                        )
                    )

            verification_examples.append(
                {
                    "prompt_id": row["prompt_id"],
                    "split": row["split"],
                    "slice": row["slice"],
                    "answer_token_count": int(row["answer_token_count"]),
                    "prompt_token_count": int(answer_start),
                    "seq_len_effective": int(input_ids.shape[0]),
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    shard_paths_by_layer: dict[int, dict[str, str]] = {}
    vector_counts: dict[int, int] = {}
    d_model = 0
    rng = random.Random(seed)
    for layer_index in layer_indices:
        h_pt_tensor = torch.cat(h_pt_rows[layer_index], dim=0)
        delta_tensor = torch.cat(delta_rows[layer_index], dim=0)
        meta_rows = metadata_rows[layer_index]
        if max_cache_vectors_per_layer is not None and h_pt_tensor.shape[0] > max_cache_vectors_per_layer:
            selected = sorted(rng.sample(range(h_pt_tensor.shape[0]), k=max_cache_vectors_per_layer))
            selected_tensor = torch.tensor(selected, dtype=torch.long)
            h_pt_tensor = h_pt_tensor[selected_tensor]
            delta_tensor = delta_tensor[selected_tensor]
            meta_rows = [meta_rows[index] for index in selected]

        shard_paths = write_cache_shard(
            cache_dir=cache_dir,
            layer_index=layer_index,
            run_id=run_id,
            shard_name=shard_name_by_layer[layer_index],
            h_pt=h_pt_tensor,
            delta=delta_tensor,
            metadata_rows=meta_rows,
        )
        reloaded = load_cache_shard(
            h_pt_path=shard_paths["h_pt_path"],
            delta_path=shard_paths["delta_path"],
            meta_path=shard_paths["meta_path"],
        )
        if not torch.equal(reloaded["h_pt"], h_pt_tensor):
            raise ValueError(f"Reloaded h_pt tensor mismatch for layer {layer_index}")
        if not torch.equal(reloaded["delta"], delta_tensor):
            raise ValueError(f"Reloaded delta tensor mismatch for layer {layer_index}")
        if len(reloaded["metadata"]) != len(meta_rows):
            raise ValueError(f"Reloaded metadata count mismatch for layer {layer_index}")
        shard_paths_by_layer[layer_index] = shard_paths
        vector_counts[layer_index] = int(h_pt_tensor.shape[0])
        d_model = int(h_pt_tensor.shape[1])

    return {
        "d_model": d_model,
        "vector_counts": vector_counts,
        "shard_paths_by_layer": shard_paths_by_layer,
        "verification_examples": verification_examples,
        "wall_clock_seconds": time.perf_counter() - start_time,
    }


def _load_completion_rows(config: dict[str, Any], slices: list[str], *, splits: tuple[str, ...] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    completion_dir = Path(config["paths"]["completion_dir"])
    selected_splits = splits or ("train_unlabeled", "select_train", "select_tune", "test")
    for split in selected_splits:
        for slice_name in slices:
            path = completion_dir / f"{split}_{slice_name}_{config['completion_run_id']}.jsonl"
            rows.extend(read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"No completion rows found for run id {config['completion_run_id']}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
