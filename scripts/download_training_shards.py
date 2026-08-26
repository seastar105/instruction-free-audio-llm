from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.dataset as pads

from audio_lfm.data.local_shards import LOCAL_MIRROR_MARKER

DATASETS = ("ParaSpeechCaps-Base", "WavCaps")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror CaptionStew audio TAR shards with the hf CLI."
    )
    parser.add_argument(
        "--captionstew-root",
        type=Path,
        default=os.environ.get("CAPTIONSTEW_ROOT"),
        required="CAPTIONSTEW_ROOT" not in os.environ,
    )
    parser.add_argument("--dataset", action="append", choices=DATASETS, dest="datasets")
    parser.add_argument("--bucket", default="seastar105/caption-stew")
    parser.add_argument("--hf-executable", default="hf")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _catalog_shards(root: Path, dataset: str) -> tuple[str, ...]:
    audio_dir = root / f"_webdataset/{dataset}/16k-flac/parquet/audio"
    table = pads.dataset(audio_dir, format="parquet").to_table(columns=["wds_shard"])
    return tuple(sorted({str(value) for value in table["wds_shard"].to_pylist()}))


def _verify_and_mark(
    root: Path, dataset: str, bucket: str, shards: tuple[str, ...]
) -> dict[str, Any]:
    files: dict[str, int] = {}
    missing: list[str] = []
    for relative in shards:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(relative)
        else:
            files[relative] = path.stat().st_size
    if missing:
        raise RuntimeError(
            f"Local mirror for {dataset} is incomplete: {len(missing)} of "
            f"{len(shards)} shards are missing or empty; first={missing[0]!r}"
        )
    marker = {
        "format_version": 1,
        "dataset": dataset,
        "bucket": bucket,
        "created_at": datetime.now(UTC).isoformat(),
        "shard_count": len(files),
        "total_tar_bytes": sum(files.values()),
        "files": files,
    }
    marker_path = root / f"_webdataset/{dataset}/16k-flac/shards" / LOCAL_MIRROR_MARKER
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(marker_path)
    return marker


def main() -> None:
    args = _arguments()
    root = args.captionstew_root.resolve()
    executable = shutil.which(args.hf_executable)
    if executable is None:
        raise RuntimeError(f"hf CLI executable not found: {args.hf_executable!r}")
    datasets = tuple(args.datasets or DATASETS)
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        shards = _catalog_shards(root, dataset)
        local_dir = root / f"_webdataset/{dataset}/16k-flac/shards"
        local_dir.mkdir(parents=True, exist_ok=True)
        remote = (
            f"hf://buckets/{args.bucket}/CaptionStew/_webdataset/"
            f"{dataset}/16k-flac/shards"
        )
        if not args.verify_only:
            command = [
                executable,
                "buckets",
                "sync",
                remote,
                str(local_dir),
                "--include",
                "*.tar",
                "--ignore-times",
            ]
            if args.dry_run:
                command.append("--dry-run")
            subprocess.run(command, check=True)
        if args.dry_run:
            results.append({"dataset": dataset, "shards": len(shards), "dry_run": True})
            continue
        marker = _verify_and_mark(root, dataset, args.bucket, shards)
        results.append(
            {
                "dataset": dataset,
                "shards": marker["shard_count"],
                "total_tar_bytes": marker["total_tar_bytes"],
                "marker": str(local_dir / LOCAL_MIRROR_MARKER),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
