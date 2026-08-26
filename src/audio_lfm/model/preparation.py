from __future__ import annotations

from typing import Any, Protocol

from audio_lfm.data.types import PrecomputedAudioExample, PreparedExample


class PromptCompilerLike(Protocol):
    def compile(self, target: Any) -> Any: ...


class ProjectorLike(Protocol):
    def projected_length(self, frontend_length: int) -> int: ...


def prepare_example(
    raw: PrecomputedAudioExample,
    *,
    prompt_compiler: PromptCompilerLike,
    projector: ProjectorLike,
) -> PreparedExample:
    text = prompt_compiler.compile(raw.selected_target)
    frontend_length = sum(raw.effective_encoder_lengths)
    projected_length = int(projector.projected_length(frontend_length))
    total = (
        len(text.before_audio_ids)
        + 1
        + projected_length
        + 1
        + len(text.after_audio_prompt_ids)
        + len(text.target_suffix_ids)
    )
    return PreparedExample(
        raw=raw,
        text=text,
        estimated_audio_embedding_length=projected_length,
        estimated_total_lfm_length=total,
    )
