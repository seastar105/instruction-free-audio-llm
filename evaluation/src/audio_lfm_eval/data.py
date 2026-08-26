from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    question: str
    choices: tuple[str, ...]
    audio_values: tuple[object, ...]
    source_record: dict[str, Any]


def _json_value(value: object) -> Any:
    if isinstance(value, bytes):
        return "<embedded-audio-bytes>"
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "as_py"):
        return _json_value(value.as_py())
    return value


def _first_text(row: Mapping[str, object], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _choices(row: Mapping[str, object]) -> tuple[str, ...]:
    for name in ("choices", "options", "answer_choices"):
        value = row.get(name)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value if str(item).strip())
        if isinstance(value, Mapping):
            return tuple(str(item) for item in value.values() if str(item).strip())
    ordered = []
    for letter in "abcdef":
        value = row.get(f"choice_{letter}")
        if value is not None and str(value).strip():
            ordered.append(str(value))
    return tuple(ordered)


def _audio_values(row: Mapping[str, object]) -> tuple[object, ...]:
    for name in (
        "audios",
        "audio_paths",
        "audio",
        "context",
        "audio_path",
        "audio_file",
        "path",
        "audio_id",
    ):
        value = row.get(name)
        if value is None:
            continue
        # MMAU-Pro uses ``audio_path`` (singular) for an ordered list of one
        # to three clips. Other datasets use the same field names for scalars,
        # so accept sequences consistently rather than keying on the spelling.
        if isinstance(value, (list, tuple)):
            if value:
                return tuple(value)
            continue
        if (
            isinstance(value, str)
            and name == "audio_id"
            and Path(value).suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg"}
        ):
            continue
        return (value,)
    raise ValueError("Benchmark row has no supported audio field")


def normalize_row(row: Mapping[str, object]) -> EvalSample:
    source = {str(key): _json_value(value) for key, value in row.items()}
    sample_id = _first_text(
        row, ("id", "key", "audio_id", "question_id", "uid", "index")
    )
    if not sample_id:
        sample_id = hashlib.sha256(
            json.dumps(source, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    question = _first_text(row, ("question", "instruction", "prompt", "text"))
    return EvalSample(
        sample_id=sample_id,
        question=question,
        choices=_choices(row),
        audio_values=_audio_values(row),
        source_record=source,
    )


def _candidate_files(root: Path, subset: str) -> list[Path]:
    if subset != "default":
        files = sorted((root / subset).glob("*.parquet"))
        if files:
            return files
    parquet_files = sorted(root.glob("*.parquet"))
    if parquet_files:
        return parquet_files
    # Standalone MMSU follows the Hub repository layout and stores its shards
    # under ``data/``. Keep the fallback recursive so equivalent pinned Hub
    # layouts do not require moving multi-gigabyte files after download.
    nested_parquet_files = sorted(root.glob("**/*.parquet"))
    if nested_parquet_files:
        return nested_parquet_files
    json_files = [
        path
        for name in ("MMAR-meta.json", "MMAR-meta.jsonl", "mmau-test-mini.json")
        if (path := root / name).exists()
    ]
    if json_files:
        return json_files
    raise FileNotFoundError(f"No benchmark rows found under {root}")


def iter_rows(root: str | Path, subset: str) -> Iterator[dict[str, Any]]:
    root_path = Path(root)
    for path in _candidate_files(root_path, subset):
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=32):
                yield from batch.to_pylist()
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
        else:
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON list in {path}")
            yield from payload


def iter_samples(
    root: str | Path, subset: str, *, limit: int | None = None
) -> Iterator[EvalSample]:
    for index, row in enumerate(iter_rows(root, subset)):
        if limit is not None and index >= limit:
            break
        yield normalize_row(row)
