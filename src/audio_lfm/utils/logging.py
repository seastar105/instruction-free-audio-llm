from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|KEY", re.IGNORECASE)
REDACTED = "<redacted>"


def is_secret_name(name: str) -> bool:
    return SECRET_NAME.search(name) is not None


def redact(value: Any, *, parent_key: str = "") -> Any:
    if is_secret_name(parent_key):
        return REDACTED
    if isinstance(value, Mapping):
        return {str(k): redact(v, parent_key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item, parent_key=parent_key) for item in value)
    return value


def append_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    redact_secrets: bool = True,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = redact(record) if redact_secrets else record
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
