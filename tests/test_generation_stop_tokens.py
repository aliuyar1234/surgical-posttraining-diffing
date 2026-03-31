from __future__ import annotations

from src.common.modeling import generation_stop_token_ids, strip_trailing_stop_tokens


class _DummyTokenizer:
    eos_token_id = 1

    def convert_tokens_to_ids(self, token: str) -> int:
        mapping = {
            "<end_of_turn>": 106,
        }
        return mapping.get(token, -1)


def test_generation_stop_token_ids_include_end_of_turn() -> None:
    tokenizer = _DummyTokenizer()
    assert generation_stop_token_ids(tokenizer) == [1, 106]


def test_strip_trailing_stop_tokens_removes_terminal_end_of_turns() -> None:
    trimmed, reached_stop = strip_trailing_stop_tokens([72, 107, 106, 106], [1, 106])
    assert trimmed == [72, 107]
    assert reached_stop is True


def test_strip_trailing_stop_tokens_removes_pad_after_stop_token() -> None:
    trimmed, reached_stop = strip_trailing_stop_tokens([72, 107, 106, 0, 0], [1, 106], pad_token_id=0)
    assert trimmed == [72, 107]
    assert reached_stop is True


def test_strip_trailing_stop_tokens_keeps_nonterminal_tokens() -> None:
    trimmed, reached_stop = strip_trailing_stop_tokens([72, 107, 42], [1, 106])
    assert trimmed == [72, 107, 42]
    assert reached_stop is False
