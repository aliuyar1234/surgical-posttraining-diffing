from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.configs import build_artifact_path, build_run_id, load_yaml_config, save_resolved_config_snapshot
from src.common.jsonl import write_jsonl
from src.common.modeling import generation_stop_token_ids, strip_trailing_stop_tokens
from src.data.prompt_suite import SPLIT_ORDER, load_prompt_records
from src.data.rendering import compute_template_hash, render_neutral_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic IT completions for the M1 prompt suite.")
    parser.add_argument("--config", required=True, help="Path to configs/model_pair.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to configs/data.yaml")
    args = parser.parse_args()

    model_config = load_yaml_config(args.config)
    data_config = load_yaml_config(args.data_config)
    payload = run_generation(model_config, data_config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_generation(model_config: dict[str, Any], data_config: dict[str, Any]) -> dict[str, Any]:
    records = load_prompt_records(data_config["paths"]["split_manifest_dir"], slices=data_config["slices"])
    if not records:
        raise FileNotFoundError("No prompt manifests found. Run build_prompt_suite first.")

    template_hash = compute_template_hash(data_config["prompt_template"])
    if template_hash != model_config["prompt_template_hash"]:
        raise ValueError(
            f"Model config template hash {model_config['prompt_template_hash']} does not match computed template hash {template_hash}"
        )

    run_payload = {
        "model_config": model_config,
        "data_prompt_template_hash": template_hash,
        "record_count": len(records),
    }
    run_id = build_run_id(model_config["stage_name"], run_payload)

    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", run_id, ".json")
    save_resolved_config_snapshot({"model_config": model_config, "data_config": data_config}, snapshot_path)

    completion_paths = {
        f"{split}:{slice_name}": (Path(model_config["paths"]["completion_dir"]) / f"{split}_{slice_name}_{run_id}.jsonl")
        for split in SPLIT_ORDER
        for slice_name in data_config["slices"]
    }
    dropped_path = Path(model_config["paths"]["completion_dir"]) / f"dropped_{run_id}.jsonl"
    smoke_path = Path(model_config["paths"]["completion_dir"]) / f"determinism_smoke_{run_id}.json"
    runtime_path = build_artifact_path("runtime", "runtime_report", run_id, ".json")

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_pair"]["it_path"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = generation_stop_token_ids(tokenizer)

    _assert_tokenizer_compatibility(model_config)

    device = model_config["generation"]["device"]
    dtype = torch.bfloat16 if model_config["generation"]["dtype"] == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_config["model_pair"]["it_path"], dtype=dtype).to(device)
    model.eval()

    start_time = time.perf_counter()
    reuse_existing = all(path.exists() for path in completion_paths.values()) and _existing_completion_artifacts_are_current(
        completion_paths.values(),
        stop_token_ids=stop_token_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion_rows: list[dict[str, Any]] = []
    dropped_records: list[dict[str, Any]] = []
    total_generated_tokens = 0
    batch_size = int(model_config["generation"]["batch_size"])

    if reuse_existing:
        for path in completion_paths.values():
            completion_rows.extend(_read_completion_rows(path))
        if dropped_path.exists():
            dropped_records = _read_completion_rows(dropped_path)
        total_generated_tokens = sum(int(row["answer_token_count"]) for row in completion_rows)
    else:
        grouped = defaultdict(list)
        for record in records:
            grouped[(record["split"], record["slice"])].append(record)

        per_slice_max_new_tokens = _max_new_tokens_by_slice(model_config)
        for split in SPLIT_ORDER:
            for slice_name in data_config["slices"]:
                rows = grouped[(split, slice_name)]
                outputs: list[dict[str, Any]] = []
                for batch_start in range(0, len(rows), batch_size):
                    batch = rows[batch_start : batch_start + batch_size]
                    prefixes: list[str] = []
                    valid_batch: list[dict[str, Any]] = []
                    for record in batch:
                        prefix = render_neutral_prefix(record["prompt"], data_config["prompt_template"])
                        prompt_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
                        if len(prompt_ids) > int(model_config["generation"]["max_prompt_tokens"]):
                            dropped_records.append(
                                {
                                    "prompt_id": record["id"],
                                    "split": split,
                                    "slice": slice_name,
                                    "reason": "prompt_too_long",
                                    "prompt_token_count": len(prompt_ids),
                                }
                            )
                            continue
                        prefixes.append(prefix)
                        valid_batch.append(record)
                    if not valid_batch:
                        continue

                    encoded = tokenizer(prefixes, padding=True, return_tensors="pt").to(device)
                    batch_input_width = int(encoded["input_ids"].shape[1])
                    with torch.no_grad():
                        generated = model.generate(
                            **encoded,
                            max_new_tokens=per_slice_max_new_tokens[slice_name],
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=stop_token_ids,
                        )

                    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                    for row_index, record in enumerate(valid_batch):
                        prompt_length = int(prompt_lengths[row_index])
                        raw_generated_ids = generated[row_index, batch_input_width:].tolist()
                        generated_ids, stop_reached = strip_trailing_stop_tokens(
                            raw_generated_ids,
                            stop_token_ids,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        total_generated_tokens += len(generated_ids)
                        outputs.append(
                            {
                                "prompt_id": record["id"],
                                "split": split,
                                "slice": slice_name,
                                "prompt": record["prompt"],
                                "rendered_prefix": prefixes[row_index],
                                "template_hash": template_hash,
                                "completion_text": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
                                "raw_completion_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
                                "completion_token_ids": generated_ids,
                                "answer_token_count": len(generated_ids),
                                "prompt_token_count": prompt_length,
                                "stop_reason": "eos" if stop_reached else "max_new_tokens",
                                "eos_reached": stop_reached,
                                "model_path": model_config["model_pair"]["it_path"],
                            }
                        )

                output_path = completion_paths[f"{split}:{slice_name}"]
                write_jsonl(output_path, outputs)
                completion_rows.extend(outputs)

        write_jsonl(dropped_path, dropped_records)

    smoke_report = _run_determinism_smoke(
        model=model,
        tokenizer=tokenizer,
        completion_rows=completion_rows,
        data_config=data_config,
        model_config=model_config,
    )
    smoke_path.write_text(json.dumps(smoke_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_seconds = time.perf_counter() - start_time
    prior_runtime: dict[str, Any] | None = None
    if runtime_path.exists():
        prior_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_payload = {
        "run_id": run_id,
        "stage_name": model_config["stage_name"],
        "model_path": model_config["model_pair"]["it_path"],
        "device": device,
        "record_count": len(records),
        "completion_record_count": len(completion_rows),
        "dropped_record_count": len(dropped_records),
        "generated_answer_tokens": total_generated_tokens,
        "wall_clock_seconds": prior_runtime["wall_clock_seconds"] if prior_runtime and reuse_existing else runtime_seconds,
        "validation_wall_clock_seconds": runtime_seconds,
        "reused_existing_completions": reuse_existing,
        "snapshot_path": snapshot_path.as_posix(),
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "completion_paths": {key: value.as_posix() for key, value in completion_paths.items()},
        "dropped_path": dropped_path.as_posix(),
        "determinism_smoke_path": smoke_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
        "completion_record_count": len(completion_rows),
        "dropped_record_count": len(dropped_records),
        "smoke_match": smoke_report["all_match"],
        "reused_existing_completions": reuse_existing,
    }


def _assert_tokenizer_compatibility(model_config: dict[str, Any]) -> None:
    pt_tokenizer = AutoTokenizer.from_pretrained(model_config["model_pair"]["pt_path"])
    it_tokenizer = AutoTokenizer.from_pretrained(model_config["model_pair"]["it_path"])
    probe_strings = (
        "Instruction:\nName the capital of France.\n\nResponse:\n",
        "Instruction:\nReply with only digits: convert forty-two to numerals.\n\nResponse:\n",
        "Instruction:\nExplain how to report a stolen bike in a safe and lawful way.\n\nResponse:\n",
    )
    if type(pt_tokenizer) is not type(it_tokenizer):
        raise ValueError("PT and IT tokenizers use different classes")
    if pt_tokenizer.vocab_size != it_tokenizer.vocab_size:
        raise ValueError("PT and IT tokenizers use different vocab sizes")
    if pt_tokenizer.eos_token_id != it_tokenizer.eos_token_id or pt_tokenizer.bos_token_id != it_tokenizer.bos_token_id:
        raise ValueError("PT and IT tokenizer special token ids differ")
    for probe in probe_strings:
        if pt_tokenizer(probe)["input_ids"] != it_tokenizer(probe)["input_ids"]:
            raise ValueError("PT and IT tokenizer encodings differ on probe strings")


def _max_new_tokens_by_slice(model_config: dict[str, Any]) -> dict[str, int]:
    default = int(model_config["generation"]["max_new_tokens"]["default"])
    mapping = {slice_name: default for slice_name in ("QA", "Math", "Format", "Brevity", "Harmful", "BenignAdjacent")}
    mapping["Harmful"] = int(model_config["generation"]["max_new_tokens"]["harmful"])
    mapping["BenignAdjacent"] = int(model_config["generation"]["max_new_tokens"]["benign_adjacent"])
    return mapping


def _run_determinism_smoke(
    *,
    model: Any,
    tokenizer: Any,
    completion_rows: list[dict[str, Any]],
    data_config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    smoke_size = int(model_config["generation"]["determinism_smoke_size"])
    selected = completion_rows[:smoke_size]
    matches: list[dict[str, Any]] = []
    if not selected:
        return {"all_match": True, "checked": 0, "records": matches}

    per_slice_max = _max_new_tokens_by_slice(model_config)
    batch_size = int(model_config["generation"]["batch_size"])
    stop_token_ids = generation_stop_token_ids(tokenizer)
    for batch_start in range(0, len(selected), batch_size):
        batch = selected[batch_start : batch_start + batch_size]
        prefixes = [render_neutral_prefix(row["prompt"], data_config["prompt_template"]) for row in batch]
        encoded = tokenizer(prefixes, padding=True, return_tensors="pt").to(model_config["generation"]["device"])
        batch_input_width = int(encoded["input_ids"].shape[1])
        max_new_tokens = max(per_slice_max[row["slice"]] for row in batch)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )
        for row_index, row in enumerate(batch):
            rerun_ids, _ = strip_trailing_stop_tokens(
                generated[row_index, batch_input_width:].tolist(),
                stop_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
            matched = rerun_ids == row["completion_token_ids"]
            matches.append({"prompt_id": row["prompt_id"], "match": matched})
    return {"all_match": all(item["match"] for item in matches), "checked": len(selected), "records": matches}


def _read_completion_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _existing_completion_artifacts_are_current(
    paths: Any,
    *,
    stop_token_ids: list[int],
    pad_token_id: int | None,
) -> bool:
    stop_id_set = set(stop_token_ids)
    for path in paths:
        for row in _read_completion_rows(Path(path)):
            completion_ids = row.get("completion_token_ids", [])
            if completion_ids and (int(completion_ids[-1]) in stop_id_set or int(completion_ids[-1]) == int(pad_token_id)):
                return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
