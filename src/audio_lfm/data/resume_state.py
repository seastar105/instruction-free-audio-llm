from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import zstandard as zstd


@dataclass
class DataResumeState:
    epoch: int = 0
    committed_audio_ids: set[str] = field(default_factory=set)

    def commit(self, audio_ids: list[str]) -> None:
        duplicates = self.committed_audio_ids.intersection(audio_ids)
        if duplicates:
            raise ValueError(f"Audio IDs already committed: {sorted(duplicates)}")
        self.committed_audio_ids.update(audio_ids)

    def advance_epoch(self) -> None:
        self.epoch += 1
        self.committed_audio_ids.clear()

    def save_ids(self, path: str | Path) -> None:
        payload = "".join(
            f"{audio_id}\n" for audio_id in sorted(self.committed_audio_ids)
        )
        Path(path).write_bytes(zstd.ZstdCompressor(level=9).compress(payload.encode()))

    def load_ids(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            self.committed_audio_ids = set()
            return
        payload = zstd.ZstdDecompressor().decompress(source.read_bytes()).decode()
        self.committed_audio_ids = set(payload.splitlines())
