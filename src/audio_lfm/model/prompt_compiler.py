from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from audio_lfm.data.types import PreparedText, TargetRecord
from audio_lfm.prompts import (
    CAPTION_EXPANSION_SYSTEM_PROMPT,
    PromptCompilationError,
    audio_training_prompt_template_sha256,
    build_audio_prompt_messages,
    build_audio_training_messages,
)
from audio_lfm.utils.hashing import sha256_text


class TokenizerLike(Protocol):
    chat_template: str | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> Any: ...


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_only: str
    full: str
    redacted_example: str


class PromptCompiler:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        *,
        prompt_file: str | Path | None,
        audio_sentinel: str,
        mode: str = "direct_caption_alignment",
        system_message: str | None = None,
        supervise_assistant_termination: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.mode = mode
        self.prompt_file = Path(prompt_file) if prompt_file is not None else None
        if mode == "caption_expansion_alignment":
            if system_message is not None:
                raise PromptCompilationError(
                    "System messages are forbidden during audio training"
                )
            if prompt_file is not None:
                raise PromptCompilationError(
                    "Caption-expansion alignment must not use a prompt file"
                )
            self.prompt_text = audio_sentinel
        elif mode == "direct_caption_alignment":
            if self.prompt_file is None:
                raise ValueError("Direct-caption alignment requires a prompt file")
            self.prompt_text = self.prompt_file.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unknown prompt mode: {mode}")
        self.audio_sentinel = audio_sentinel
        self.system_message = system_message
        self.supervise_assistant_termination = supervise_assistant_termination
        if self.prompt_text.count(audio_sentinel) != 1:
            raise ValueError("Prompt file must contain the audio sentinel exactly once")
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template")
        self.prompt_sha256 = (
            audio_training_prompt_template_sha256(audio_sentinel)
            if mode == "caption_expansion_alignment"
            else sha256_text(self.prompt_text)
        )
        self.chat_template_sha256 = sha256_text(tokenizer.chat_template)

    def _messages(self) -> list[dict[str, str]]:
        if self.mode == "caption_expansion_alignment":
            return build_audio_prompt_messages(self.audio_sentinel)
        messages: list[dict[str, str]] = []
        if self.system_message is not None:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": self.prompt_text})
        return messages

    def render(self, target_text: str) -> RenderedPrompt:
        prompt_messages = self._messages()
        prompt_only = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_messages = (
            build_audio_training_messages(
                audio_sentinel=self.audio_sentinel,
                expanded_response=target_text,
            )
            if self.mode == "caption_expansion_alignment"
            else [*prompt_messages, {"role": "assistant", "content": target_text}]
        )
        full = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if self.mode == "caption_expansion_alignment":
            if any(message["role"] == "system" for message in prompt_messages):
                raise PromptCompilationError(
                    "System messages are forbidden during audio training"
                )
            if any(message["role"] == "system" for message in full_messages):
                raise PromptCompilationError(
                    "System messages are forbidden during audio training"
                )
            if CAPTION_EXPANSION_SYSTEM_PROMPT in prompt_only:
                raise PromptCompilationError(
                    "Expansion system prompt leaked into training prompt"
                )
            if CAPTION_EXPANSION_SYSTEM_PROMPT in full:
                raise PromptCompilationError(
                    "Expansion system prompt leaked into supervised training text"
                )
        if prompt_only.count(self.audio_sentinel) != 1:
            raise ValueError("Rendered prompt must contain the sentinel exactly once")
        if full.count(self.audio_sentinel) != 1:
            raise ValueError(
                "Rendered full conversation must contain the sentinel exactly once"
            )
        return RenderedPrompt(
            prompt_only=prompt_only,
            full=full,
            redacted_example=full.replace(target_text, "<target-redacted>"),
        )

    def compile(self, target: TargetRecord) -> PreparedText:
        rendered = self.render(target.text)
        before_prompt, after_prompt = rendered.prompt_only.split(self.audio_sentinel)
        before_full, after_full = rendered.full.split(self.audio_sentinel)
        if before_prompt != before_full:
            raise ValueError("Chat template changed content before the audio sentinel")
        if not after_full.startswith(after_prompt):
            raise ValueError("Assistant generation prefix differs in full rendering")
        target_suffix = after_full[len(after_prompt) :]
        before_ids = self._ids(before_prompt)
        after_ids = self._ids(after_prompt)
        target_ids = self._ids(target_suffix)
        combined_ids = self._ids(after_prompt + target_suffix)
        if combined_ids != after_ids + target_ids:
            after_ids, target_ids = self._split_with_offsets(
                after_prompt, target_suffix
            )
        if not target_ids:
            raise ValueError("Assistant target suffix tokenized to zero tokens")
        if not self.supervise_assistant_termination:
            eos = self.tokenizer.eos_token_id
            if eos is not None and target_ids and target_ids[-1] == eos:
                target_ids = target_ids[:-1]
            if not target_ids:
                raise ValueError("Target contains only assistant termination")
        return PreparedText(
            before_audio_ids=tuple(before_ids),
            after_audio_prompt_ids=tuple(after_ids),
            target_suffix_ids=tuple(target_ids),
            target_id=target.target_id,
            prompt_sha256=self.prompt_sha256,
        )

    def _ids(self, text: str) -> list[int]:
        result = self.tokenizer(text, add_special_tokens=False)
        return [int(value) for value in result.input_ids]

    def _split_with_offsets(
        self, prefix: str, suffix: str
    ) -> tuple[list[int], list[int]]:
        try:
            result = self.tokenizer(
                prefix + suffix,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = result.offset_mapping
        except (TypeError, AttributeError) as error:
            raise ValueError(
                "Tokenizer boundary merge requires a fast tokenizer with offsets"
            ) from error
        boundary = len(prefix)
        crossing = [pair for pair in offsets if pair[0] < boundary < pair[1]]
        if crossing:
            raise ValueError("A tokenizer token crosses the prompt/target boundary")
        ids = [int(value) for value in result.input_ids]
        split = next(
            (index for index, pair in enumerate(offsets) if pair[0] >= boundary),
            len(ids),
        )
        return ids[:split], ids[split:]
