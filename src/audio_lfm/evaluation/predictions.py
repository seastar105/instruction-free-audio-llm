from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class PredictionWriter:
    def __init__(self, output_dir: str | Path, *, part_size: int = 256) -> None:
        self.output_dir = Path(output_dir)
        self.parts_dir = self.output_dir / "predictions"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.part_size = part_size
        self.pending: list[dict[str, Any]] = []
        self.next_part = len(list(self.parts_dir.glob("part-*.parquet")))

    def add(self, record: dict[str, Any]) -> None:
        self.pending.append(record)
        if len(self.pending) >= self.part_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        final = self.parts_dir / f"part-{self.next_part:06d}.parquet"
        temporary = final.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(self.pending), temporary)
        os.replace(temporary, final)
        with (self.output_dir / "predictions.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            for record in self.pending:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self.pending.clear()
        self.next_part += 1

    def completed_audio_ids(self) -> set[str]:
        completed: set[str] = set()
        for part in sorted(self.parts_dir.glob("part-*.parquet")):
            for audio_id in pq.read_table(part, columns=["audio_id"])[
                "audio_id"
            ].to_pylist():
                if audio_id in completed:
                    raise RuntimeError(f"Duplicate prediction audio_id: {audio_id}")
                completed.add(str(audio_id))
        return completed
