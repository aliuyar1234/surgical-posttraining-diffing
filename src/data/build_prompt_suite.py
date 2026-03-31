from __future__ import annotations

import argparse
import json

from src.common.configs import load_yaml_config
from src.data.prompt_suite import build_prompt_suite, prompt_suite_run_id, write_prompt_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the M1 prompt suite manifests.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    parser.add_argument(
        "--model-config",
        default="configs/model_pair.yaml",
        help="Optional model config used for tokenizer-path context in metadata.",
    )
    args = parser.parse_args()

    data_config = load_yaml_config(args.config)
    model_config = load_yaml_config(args.model_config)
    records, summary = build_prompt_suite(data_config, model_config=model_config)
    run_id = prompt_suite_run_id(data_config, summary)
    written = write_prompt_suite(records, data_config, run_id=run_id, summary=summary)

    payload = {
        "run_id": run_id,
        "summary": summary,
        "written": written,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
