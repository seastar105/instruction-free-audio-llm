from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import soundfile as sf


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact num_samples metadata from local FLAC STREAMINFO."
    )
    parser.add_argument("--captionstew-root", type=Path, required=True)
    parser.add_argument("--input-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets")
    return parser.parse_args()


def _streaminfo(header: bytes, *, audio_id: str) -> tuple[int, int, int]:
    if len(header) < 42 or header[:4] != b"fLaC":
        raise ValueError(f"Invalid FLAC header for {audio_id!r}")
    block_type = header[4] & 0x7F
    block_length = int.from_bytes(header[5:8], "big")
    if block_type != 0 or block_length != 34:
        raise ValueError(f"Missing FLAC STREAMINFO for {audio_id!r}")
    packed = int.from_bytes(header[18:26], "big")
    sample_rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    total_samples = packed & ((1 << 36) - 1)
    if sample_rate <= 0 or channels <= 0:
        raise ValueError(f"Invalid STREAMINFO values for {audio_id!r}")
    return sample_rate, channels, total_samples


def _audio_rows(root: Path, dataset: str) -> dict[str, dict[str, Any]]:
    candidates = (
        root / f"_webdataset/{dataset}/16k-flac/parquet/audio",
        root / f"{dataset}/16k-flac/parquet/audio",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No audio catalog found for {dataset!r} under {root}")
    table = pads.dataset(path, format="parquet").to_table(
        columns=["audio_id", "wds_key", "wds_shard", "flac_size", "flac_sha256"]
    )
    return {str(row["audio_id"]): row for row in table.to_pylist()}


def _exact_samples(
    root: Path, dataset: str, audio_ids: set[str]
) -> dict[str, dict[str, int | str]]:
    metadata = _audio_rows(root, dataset)
    by_shard: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_metadata = audio_ids.difference(metadata)
    if missing_metadata:
        raise KeyError(
            f"{dataset} sidecar IDs absent from audio catalog: "
            f"{next(iter(missing_metadata))!r}"
        )
    for audio_id in audio_ids:
        by_shard[str(metadata[audio_id]["wds_shard"])].append(metadata[audio_id])
    result: dict[str, dict[str, int | str]] = {}
    for index, (relative_shard, rows) in enumerate(sorted(by_shard.items()), start=1):
        path = root / relative_shard
        if not path.is_file():
            raise FileNotFoundError(f"Local shard is missing: {path}")
        expected = {
            f"{row['wds_key']}.{suffix}": (row, suffix)
            for row in rows
            for suffix in ("flac", "json")
        }
        partial: defaultdict[str, dict[str, int | str]] = defaultdict(dict)
        with tarfile.open(path, mode="r:") as archive:
            found: set[str] = set()
            for member in archive:
                match = expected.get(member.name)
                if match is None:
                    continue
                row, suffix = match
                audio_id = str(row["audio_id"])
                partial[audio_id][f"{suffix}_offset"] = member.offset_data
                partial[audio_id][f"{suffix}_size"] = member.size
                if suffix == "flac":
                    if member.size != int(row["flac_size"]):
                        raise ValueError(
                            f"FLAC TAR size mismatch for {audio_id!r}: "
                            f"{member.size} != {row['flac_size']}"
                        )
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Cannot read FLAC member for {audio_id!r}")
                    header = stream.read(42)
                    sample_rate, channels, total_samples = _streaminfo(
                        header, audio_id=audio_id
                    )
                    if sample_rate != 16_000 or channels != 1:
                        raise ValueError(
                            f"Unexpected audio format for {audio_id!r}: "
                            f"{sample_rate} Hz, {channels} channels"
                        )
                    sample_source = "flac_streaminfo"
                    if total_samples == 0:
                        with sf.SoundFile(io.BytesIO(header + stream.read())) as audio:
                            total_samples = int(audio.frames)
                            if (
                                audio.samplerate != sample_rate
                                or audio.channels != channels
                            ):
                                raise ValueError(
                                    f"FLAC header disagreement for {audio_id!r}"
                                )
                        if total_samples <= 0 or total_samples >= 2**62:
                            total_samples = 0
                            sample_source = "flac_empty"
                        else:
                            sample_source = "libsndfile_header"
                    partial[audio_id]["num_samples"] = total_samples
                    partial[audio_id]["num_samples_source"] = sample_source
                    partial[audio_id]["wds_shard"] = relative_shard
                    partial[audio_id]["wds_key"] = str(row["wds_key"])
                found.add(member.name)
            missing = expected.keys() - found
            if missing:
                raise KeyError(
                    f"Shard {relative_shard!r} lacks member {next(iter(missing))!r}"
                )
        for row in rows:
            audio_id = str(row["audio_id"])
            reference = partial[audio_id]
            required = {
                "num_samples",
                "num_samples_source",
                "wds_shard",
                "wds_key",
                "flac_offset",
                "flac_size",
                "json_offset",
                "json_size",
            }
            if not required.issubset(reference):
                raise RuntimeError(f"Incomplete TAR reference for {audio_id!r}")
            result[audio_id] = reference
        if index % 10 == 0 or index == len(by_shard):
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "shards_scanned": index,
                        "shards_total": len(by_shard),
                        "audio_headers": len(result),
                    }
                ),
                flush=True,
            )
    return result


def main() -> None:
    args = _arguments()
    root = args.captionstew_root.resolve()
    source = pq.read_table(args.input_sidecar)
    required = {
        "audio_id",
        "dataset",
        "source_id",
        "num_samples",
        "duration_seconds",
        "flac_sha256",
    }
    if not required.issubset(source.column_names):
        raise ValueError("Input duration sidecar schema is invalid")
    selected = set(args.datasets or source["dataset"].unique().to_pylist())
    rows = [row for row in source.to_pylist() if str(row["dataset"]) in selected]
    exact: dict[str, dict[str, int | str]] = {}
    for dataset in sorted(selected):
        audio_ids = {
            str(row["audio_id"]) for row in rows if str(row["dataset"]) == dataset
        }
        exact.update(_exact_samples(root, dataset, audio_ids))
    output_rows = [
        {
            **row,
            **exact[str(row["audio_id"])],
            "duration_seconds": (
                int(exact[str(row["audio_id"])]["num_samples"]) / 16_000
            ),
        }
        for row in rows
    ]
    if len(output_rows) != len(exact):
        raise RuntimeError("Exact sidecar output cardinality is inconsistent")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    pq.write_table(pa.Table.from_pylist(output_rows), temporary, compression="zstd")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "datasets": sorted(selected),
                "rows": len(output_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
