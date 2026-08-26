from __future__ import annotations

from collections.abc import Iterable, Iterator

from audio_lfm.data.types import PrecomputedAudioExample


def filter_committed(
    stream: Iterable[PrecomputedAudioExample], committed_audio_ids: set[str]
) -> Iterator[PrecomputedAudioExample]:
    seen: set[str] = set()
    for example in stream:
        if example.audio_id in seen:
            raise ValueError(f"Duplicate streamed audio_id: {example.audio_id}")
        seen.add(example.audio_id)
        if example.audio_id not in committed_audio_ids:
            yield example
