from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class PredictionStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "predictions.jsonl"

    def completed_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        completed = set()
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if "error" not in record:
                        completed.add(str(record["id"]))
        return completed

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected JSON objects in {self.path}")
                    records.append(value)
        return records

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def write_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(output_dir) / "generation_manifest.json"
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Incompatible resume manifest at {path}")
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)
