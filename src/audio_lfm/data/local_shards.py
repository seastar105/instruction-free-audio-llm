from __future__ import annotations

import json
from pathlib import Path

from audio_lfm.data.catalog import CatalogIndex

LOCAL_MIRROR_MARKER = ".audio_lfm_complete.json"


def complete_local_shards(
    root: str | Path, catalog: CatalogIndex
) -> tuple[str, ...] | None:
    """Return local shard paths only for a complete, size-verified mirror."""
    root_path = Path(root).resolve()
    shard_dir = root_path / f"_webdataset/{catalog.dataset}/16k-flac/shards"
    marker_path = shard_dir / LOCAL_MIRROR_MARKER
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if marker.get("format_version") != 1 or marker.get("dataset") != catalog.dataset:
        return None
    files = marker.get("files")
    if not isinstance(files, dict):
        return None
    local_paths: list[str] = []
    for relative in catalog.selected_shards:
        expected_size = files.get(relative)
        if not isinstance(expected_size, int) or expected_size <= 0:
            return None
        path = root_path / relative
        try:
            if not path.is_file() or path.stat().st_size != expected_size:
                return None
        except OSError:
            return None
        local_paths.append(str(path))
    return tuple(local_paths)
