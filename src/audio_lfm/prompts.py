from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from audio_lfm.utils.hashing import canonical_sha256, sha256_text

CAPTION_EXPANSION_SYSTEM_PROMPT = (
    "You are an AI assistant directly hearing this audio. "
    "Respond as if you heard it yourself."
)


class PromptCompilationError(ValueError):
    """A rendered prompt violated an explicit audio-alignment contract."""


def build_caption_expansion_messages(caption: str) -> list[dict[str, str]]:
    if not isinstance(caption, str):
        raise TypeError("caption must be a string")
    if not caption.strip():
        raise ValueError("caption must not be empty")
    return [
        {"role": "system", "content": CAPTION_EXPANSION_SYSTEM_PROMPT},
        {"role": "user", "content": caption},
    ]


def build_audio_prompt_messages(audio_sentinel: str) -> list[dict[str, str]]:
    if not audio_sentinel:
        raise ValueError("audio_sentinel must not be empty")
    return [{"role": "user", "content": audio_sentinel}]


def build_audio_training_messages(
    *, audio_sentinel: str, expanded_response: str
) -> list[dict[str, str]]:
    if not audio_sentinel:
        raise ValueError("audio_sentinel must not be empty")
    if not expanded_response.strip():
        raise ValueError("expanded_response must not be empty")
    return [
        {"role": "user", "content": audio_sentinel},
        {"role": "assistant", "content": expanded_response},
    ]


def caption_expansion_prompt_template_sha256() -> str:
    return canonical_sha256(
        {
            "messages": [
                {"role": "system", "content": CAPTION_EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": "{official_caption}"},
            ]
        }
    )


def audio_training_prompt_template_sha256(audio_sentinel: str) -> str:
    return canonical_sha256({"messages": build_audio_prompt_messages(audio_sentinel)})


def validate_no_expansion_prompt_leakage(rendered: str, audio_token: str) -> None:
    if CAPTION_EXPANSION_SYSTEM_PROMPT in rendered:
        raise PromptCompilationError(
            "Caption-expansion instruction leaked into audio prompt"
        )
    if rendered.count(audio_token) != 1:
        raise PromptCompilationError("Audio placeholder must occur exactly once")


def build_vllm_audio_request_prompt(
    *, tokenizer: Any, audio_token: str
) -> tuple[str, list[int]]:
    messages = build_audio_prompt_messages(audio_token)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    validate_no_expansion_prompt_leakage(rendered, audio_token)
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return rendered, [int(value) for value in tokenized]


EXPANSION_SYSTEM_PROMPT_SHA256 = sha256_text(CAPTION_EXPANSION_SYSTEM_PROMPT)
