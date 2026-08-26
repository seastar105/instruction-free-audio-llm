from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    )


def stable_reference_index(
    *, seed: int, epoch: int, audio_id: str, num_references: int
) -> int:
    if num_references <= 0:
        raise ValueError("num_references must be positive")
    payload = f"{seed}\0{epoch}\0{audio_id}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % num_references


def deterministic_int(
    *, seed: int, epoch: int, audio_id: str, upper_exclusive: int
) -> int:
    return stable_reference_index(
        seed=seed,
        epoch=epoch,
        audio_id=audio_id,
        num_references=upper_exclusive,
    )
