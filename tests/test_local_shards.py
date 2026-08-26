from __future__ import annotations

import json
from pathlib import Path

from audio_lfm.data.catalog import CatalogIndex
from audio_lfm.data.local_shards import LOCAL_MIRROR_MARKER, complete_local_shards


def _catalog(root: Path) -> CatalogIndex:
    del root
    return CatalogIndex(
        dataset="WavCaps",
        logical_split="AudioSet_SL",
        allowed_audio_ids=frozenset(),
        audio_by_id={},
        style_captions_by_id={},
        transcript_by_id={},
        selected_shards=(
            "_webdataset/WavCaps/16k-flac/shards/one.tar",
            "_webdataset/WavCaps/16k-flac/shards/two.tar",
        ),
        review_status_distribution={},
        target_count_distribution={},
        split_overlap_report={},
        fingerprint="test",
    )


def test_local_shards_require_complete_size_verified_marker(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    shard_dir = tmp_path / "_webdataset/WavCaps/16k-flac/shards"
    shard_dir.mkdir(parents=True)
    one = shard_dir / "one.tar"
    two = shard_dir / "two.tar"
    one.write_bytes(b"one")
    two.write_bytes(b"two-two")
    files = {
        catalog.selected_shards[0]: one.stat().st_size,
        catalog.selected_shards[1]: two.stat().st_size,
    }
    marker = {"format_version": 1, "dataset": "WavCaps", "files": files}
    (shard_dir / LOCAL_MIRROR_MARKER).write_text(json.dumps(marker))

    assert complete_local_shards(tmp_path, catalog) == (str(one), str(two))
    two.write_bytes(b"tampered")
    assert complete_local_shards(tmp_path, catalog) is None
