from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def generation_stop_token_ids(tokenizer) -> list[int]:
    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id >= 0 and end_of_turn_id not in stop_ids:
        stop_ids.append(end_of_turn_id)
    return stop_ids


def strip_trailing_stop_tokens(
    token_ids: list[int],
    stop_token_ids: Iterable[int],
    *,
    pad_token_id: int | None = None,
) -> tuple[list[int], bool]:
    trimmed = list(token_ids)
    stop_id_set = {int(token_id) for token_id in stop_token_ids}
    reached_stop = False
    if pad_token_id is not None:
        while trimmed and trimmed[-1] == int(pad_token_id):
            trimmed.pop()
    while trimmed and trimmed[-1] in stop_id_set:
        trimmed.pop()
        reached_stop = True
    return trimmed, reached_stop


def assert_tokenizer_compatibility(pt_path: str, it_path: str, probe_strings: Iterable[str] | None = None) -> None:
    pt_tokenizer = AutoTokenizer.from_pretrained(pt_path)
    it_tokenizer = AutoTokenizer.from_pretrained(it_path)
    probes = tuple(
        probe_strings
        or (
            "Instruction:\nName the capital of France.\n\nResponse:\n",
            "Instruction:\nReply with only digits: convert forty-two to numerals.\n\nResponse:\n",
            "Instruction:\nExplain how to report a stolen bike in a safe and lawful way.\n\nResponse:\n",
        )
    )
    if type(pt_tokenizer) is not type(it_tokenizer):
        raise ValueError("PT and IT tokenizers use different classes")
    if pt_tokenizer.vocab_size != it_tokenizer.vocab_size:
        raise ValueError("PT and IT tokenizers use different vocab sizes")
    if pt_tokenizer.eos_token_id != it_tokenizer.eos_token_id or pt_tokenizer.bos_token_id != it_tokenizer.bos_token_id:
        raise ValueError("PT and IT tokenizer special token ids differ")
    for probe in probes:
        if pt_tokenizer(probe)["input_ids"] != it_tokenizer(probe)["input_ids"]:
            raise ValueError("PT and IT tokenizer encodings differ on probe strings")


def load_causal_model(model_path: str, *, device: str, dtype_name: str):
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    model.eval()
    return model


def get_text_model(model):
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model"):
        return model.model
    return model


def get_decoder_layers(model):
    text_model = get_text_model(model)
    if not hasattr(text_model, "layers"):
        raise AttributeError(f"{type(text_model).__name__} does not expose .layers")
    return text_model.layers


def mid_layer_index(model) -> int:
    num_layers = len(get_decoder_layers(model))
    return round(0.50 * (num_layers - 1))


def late_layer_index(model) -> int:
    num_layers = len(get_decoder_layers(model))
    return round(0.85 * (num_layers - 1))


def locked_layer_indices(model) -> tuple[int, int]:
    return (mid_layer_index(model), late_layer_index(model))
