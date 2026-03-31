from __future__ import annotations

import json
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl


def test_completion_storage_round_trip(tmp_path: Path) -> None:
    rows = [
        {
            "prompt_id": "qa_test_0001",
            "split": "test",
            "slice": "QA",
            "completion_text": "Paris",
            "completion_token_ids": [123, 456],
            "template_hash": "e062d19358cd8326",
            "stop_reason": "eos",
            "eos_reached": True,
        }
    ]
    path = tmp_path / "completions.jsonl"
    write_jsonl(path, rows)
    restored = read_jsonl(path)
    assert restored == rows


def test_determinism_report_json_contract(tmp_path: Path) -> None:
    payload = {"all_match": True, "checked": 3, "records": [{"prompt_id": "a", "match": True}]}
    path = tmp_path / "determinism.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["all_match"] is True
    assert restored["checked"] == 3
