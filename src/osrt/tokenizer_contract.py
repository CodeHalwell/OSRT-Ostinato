"""Fail-closed validation for the tokenizer baked into model checkpoints."""

from __future__ import annotations

from typing import Protocol


class _Tokenizer(Protocol):
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None
    unk_token_id: int | None

    def __len__(self) -> int: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


OSTINATO_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<|begin_of_text|>": 49152,
    "<|end_of_text|>": 49153,
    "<|padding|>": 49154,
    "<|unknown|>": 49155,
    "<|think|>": 49159,
    "<|/think|>": 49160,
    "<|answer|>": 49161,
    "<|/answer|>": 49162,
    "<|user|>": 49163,
    "<|assistant|>": 49164,
    "<|system|>": 49165,
    "<|end_turn|>": 49166,
    "<|tool_call|>": 49167,
    "<|/tool_call|>": 49168,
    "<|tool_result|>": 49169,
    "<|/tool_result|>": 49170,
}

OSTINATO_ROLE_TOKEN_IDS = {
    "bos_token_id": 49152,
    "eos_token_id": 49153,
    "pad_token_id": 49154,
}


def validate_tokenizer_contract(
    tokenizer: _Tokenizer,
    *,
    expected_vocab_size: int = 49_184,
) -> None:
    """Raise when vocab size or any structural-token ID has drifted."""
    errors: list[str] = []
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != expected_vocab_size:
        errors.append(
            f"vocab size is {actual_vocab_size}, expected {expected_vocab_size}"
        )

    for token, expected_id in OSTINATO_SPECIAL_TOKEN_IDS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            errors.append(f"{token} has id {actual_id}, expected {expected_id}")

    for attribute, expected_id in OSTINATO_ROLE_TOKEN_IDS.items():
        actual_id = getattr(tokenizer, attribute, None)
        if actual_id != expected_id:
            errors.append(f"{attribute} is {actual_id}, expected {expected_id}")

    if errors:
        details = "; ".join(errors)
        raise ValueError(
            "Tokenizer contract mismatch; refusing to construct a model with "
            f"incompatible embeddings: {details}"
        )
