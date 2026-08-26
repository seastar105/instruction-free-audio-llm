from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from audio_lfm.model.prompt_compiler import PromptCompiler
from audio_lfm.utils.hashing import stable_reference_index

PACK_LENGTHS = (4096, 8192, 12_288, 16_384, 24_576, 32_768)
PLANNING_BUFFERS = (256, 512, 1024, 2048, 4096)
WAVCAPS_SPLITS = (
    "AudioSet_SL",
    "BBC_Sound_Effects",
    "FreeSound",
    "SoundBible",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captionstew-root", type=Path, required=True)
    parser.add_argument("--paraspeech-metadata", type=Path, required=True)
    parser.add_argument("--wavcaps-metadata", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-sidecar-output", type=Path)
    parser.add_argument(
        "--exact-duration-sidecar",
        type=Path,
        help="Override upstream durations with exact FLAC STREAMINFO num_samples.",
    )
    parser.add_argument(
        "--sidecar-only",
        action="store_true",
        help="Write metadata-derived durations and exit before tokenization/packing.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--runtime-num-workers", type=int, default=2)
    parser.add_argument("--runtime-shard-shuffle", type=int, default=53)
    parser.add_argument("--runtime-sample-shuffle", type=int, default=1024)
    parser.add_argument("--runtime-planning-buffer", type=int, default=2048)
    return parser.parse_args()


def _wavcaps_durations(root: Path) -> dict[str, dict[str, float]]:
    files = {
        "AudioSet_SL": root / "json_files/AudioSet_SL/as_final.json",
        "BBC_Sound_Effects": root / "json_files/BBC_Sound_Effects/bbc_final.json",
        "FreeSound": root / "json_files/FreeSound/fsd_final.json",
        "SoundBible": root / "json_files/SoundBible/sb_final.json",
    }
    result: dict[str, dict[str, float]] = {}
    for split, path in files.items():
        values: dict[str, float] = {}
        with path.open("rb") as stream:
            for row in ijson.items(stream, "data.item"):
                duration = row.get("duration")
                if not isinstance(duration, (int, float, Decimal)):
                    raise TypeError(f"Invalid duration in {path}: {duration!r}")
                values[str(row["id"])] = float(duration)
        result[split] = values
    return result


def _paraspeech_durations(path: Path) -> dict[str, float]:
    table = pq.read_table(path, columns=["relative_audio_path", "duration"])
    return {
        str(source): float(duration)
        for source, duration in zip(
            table["relative_audio_path"].to_pylist(),
            table["duration"].to_pylist(),
            strict=True,
        )
    }


def _audio_rows(root: Path, dataset: str) -> list[dict[str, Any]]:
    path = root / "_webdataset" / dataset / "16k-flac/parquet/audio"
    columns = [
        "audio_id",
        "source_id",
        "splits",
        "wds_shard",
        "wds_key",
        "flac_sha256",
    ]
    return pads.dataset(path, format="parquet").to_table(columns=columns).to_pylist()


def _write_duration_sidecar(
    path: Path, records: list[tuple[str, str, float, str, str, str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "audio_id": [record[1] for record in records],
            "dataset": [record[0] for record in records],
            "source_id": [record[5] for record in records],
            "num_samples": [round(record[2] * 16_000) for record in records],
            "duration_seconds": [record[2] for record in records],
            "flac_sha256": [record[6] for record in records],
        }
    )
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _selected_texts(
    root: Path,
    dataset: str,
    audio_ids: set[str],
    *,
    seed: int,
    epoch: int,
) -> dict[str, str]:
    path = root / "_webdataset" / dataset / "16k-flac/parquet/overlays/kind=response"
    table = pads.dataset(path, format="parquet").to_table(
        columns=["audio_id", "target_id", "text"]
    )
    references: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for audio_id, target_id, text in zip(
        table["audio_id"].to_pylist(),
        table["target_id"].to_pylist(),
        table["text"].to_pylist(),
        strict=True,
    ):
        key = str(audio_id)
        if key in audio_ids:
            references[key].append((str(target_id), str(text)))
    selected: dict[str, str] = {}
    for audio_id in audio_ids:
        ordered = sorted(references[audio_id])
        if not ordered:
            raise ValueError(f"No expanded response for {dataset}:{audio_id}")
        index = stable_reference_index(
            seed=seed,
            epoch=epoch,
            audio_id=audio_id,
            num_references=len(ordered),
        )
        selected[audio_id] = ordered[index][1]
    return selected


def _projected_audio_tokens(duration_seconds: float) -> int:
    num_samples = round(duration_seconds * 16_000)
    full_chunks, remainder = divmod(num_samples, 480_000)

    def encoder_length(samples: int) -> int:
        mel_length = (samples + 159) // 160
        return (mel_length + 1) // 2

    encoder_frames = full_chunks * 1500
    if remainder:
        encoder_frames += encoder_length(remainder)
    return (encoder_frames + 3) // 4


def _text_lengths(texts: list[str], *, hf_cache: Path) -> list[int]:
    revision = "0f604ada3f766f9f257460c4c9f0b5d6f69d431b"
    tokenizer = AutoTokenizer.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct",
        revision=revision,
        cache_dir=hf_cache,
        local_files_only=True,
    )
    sentinel = "<<__AUDIO_EMBEDDINGS_08E8F7E7__>>"
    compiler = PromptCompiler(
        tokenizer,
        prompt_file=None,
        audio_sentinel=sentinel,
        mode="caption_expansion_alignment",
        system_message=None,
        supervise_assistant_termination=True,
    )
    marker = "__AUDIO_LFM_TARGET_MARKER_5C68__"
    rendered = compiler.render(marker)
    before_prompt, after_prompt = rendered.prompt_only.split(sentinel)
    _, after_full = rendered.full.split(sentinel)
    marker_and_tail = after_full[len(after_prompt) :]
    if not marker_and_tail.startswith(marker):
        raise RuntimeError("Could not isolate assistant termination text")
    termination = marker_and_tail[len(marker) :]
    before_length = len(tokenizer(before_prompt, add_special_tokens=False).input_ids)
    lengths: list[int] = []
    for start in range(0, len(texts), 256):
        batch = [
            after_prompt + text + termination for text in texts[start : start + 256]
        ]
        encoded = tokenizer(batch, add_special_tokens=False)
        lengths.extend(before_length + len(ids) for ids in encoded.input_ids)
    return lengths


def _pack(
    lengths: list[int], capacity: int, planning_buffer: int
) -> dict[str, int | float]:
    valid = [length for length in lengths if length <= capacity]
    oversized = len(lengths) - len(valid)
    packs = 0
    packed_tokens = 0
    packed_examples = 0
    for start in range(0, len(valid), planning_buffer):
        window = sorted(valid[start : start + planning_buffer], reverse=True)
        bins: list[tuple[int, int]] = []
        for size in window:
            candidates = [
                (capacity - used - size, index)
                for index, (used, count) in enumerate(bins)
                if used + size <= capacity
            ]
            if candidates:
                _, index = min(candidates)
                used, count = bins[index]
                bins[index] = (used + size, count + 1)
            else:
                bins.append((size, 1))
        packs += len(bins)
        packed_tokens += sum(used for used, _ in bins)
        packed_examples += sum(count for _, count in bins)
    return {
        "packs_or_optimizer_steps": packs,
        "usable_examples": len(valid),
        "oversized_examples": oversized,
        "utilization": packed_tokens / (packs * capacity),
        "mean_examples_per_pack": packed_examples / packs,
        "ideal_token_lower_bound": math.ceil(sum(valid) / capacity),
        "packed_tokens": packed_tokens,
    }


def _wds_shuffle(items: list[Any], *, bufsize: int, seed: int) -> list[Any]:
    if not bufsize:
        return items
    import webdataset as wds

    return list(wds.shuffle(bufsize=bufsize, rng=random.Random(seed))(iter(items)))


def _runtime_worker_lengths(
    records: list[tuple[str, str, float, str, str, str, str]],
    lengths_by_audio_id: dict[str, int],
    *,
    seed: int,
    epoch: int,
    num_workers: int,
    shard_shuffle: int,
    sample_shuffle: int,
) -> list[list[int]]:
    if num_workers <= 0:
        raise ValueError("Runtime worker simulation requires num_workers > 0")
    by_dataset_shard: defaultdict[str, defaultdict[str, list[tuple[str, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for dataset, audio_id, _, shard, key, *_ in records:
        by_dataset_shard[dataset][shard].append((key, audio_id))
    result: list[list[int]] = []
    for worker_id in range(num_workers):
        dataset_streams: list[Iterator[str]] = []
        for dataset in sorted(by_dataset_shard):
            shard_rows = by_dataset_shard[dataset]
            shards = _wds_shuffle(
                sorted(shard_rows),
                bufsize=shard_shuffle,
                seed=seed + epoch,
            )[worker_id::num_workers]
            ordered = [
                audio_id
                for shard in shards
                for _, audio_id in sorted(shard_rows[shard])
            ]
            shuffled = _wds_shuffle(
                ordered,
                bufsize=sample_shuffle,
                seed=seed + epoch,
            )
            dataset_streams.append(iter(shuffled))
        rng = random.Random(seed + epoch)
        mixed: list[int] = []
        while dataset_streams:
            index = rng.randrange(len(dataset_streams))
            try:
                audio_id = next(dataset_streams[index])
            except StopIteration:
                dataset_streams.pop(index)
                continue
            mixed.append(lengths_by_audio_id[audio_id])
        result.append(mixed)
    return result


def _combine_worker_packing(
    worker_lengths: list[list[int]], capacity: int, planning_buffer: int
) -> dict[str, int | float]:
    parts = [_pack(lengths, capacity, planning_buffer) for lengths in worker_lengths]
    packs = sum(int(part["packs_or_optimizer_steps"]) for part in parts)
    packed_tokens = sum(int(part["packed_tokens"]) for part in parts)
    usable = sum(int(part["usable_examples"]) for part in parts)
    oversized = sum(int(part["oversized_examples"]) for part in parts)
    return {
        "packs_or_optimizer_steps": packs,
        "usable_examples": usable,
        "oversized_examples": oversized,
        "utilization": packed_tokens / (packs * capacity),
        "mean_examples_per_pack": usable / packs,
        "ideal_token_lower_bound_per_worker": sum(
            int(part["ideal_token_lower_bound"]) for part in parts
        ),
        "packed_tokens": packed_tokens,
    }


def main() -> None:
    args = _arguments()
    para_durations = _paraspeech_durations(args.paraspeech_metadata)
    wav_durations = _wavcaps_durations(args.wavcaps_metadata)
    records: list[tuple[str, str, float, str, str, str, str]] = []

    para_rows = [
        row
        for row in _audio_rows(args.captionstew_root, "ParaSpeechCaps-Base")
        if "train_base" in row["splits"]
    ]
    for row in para_rows:
        records.append(
            (
                "ParaSpeechCaps-Base",
                str(row["audio_id"]),
                para_durations[str(row["source_id"])],
                str(row["wds_shard"]),
                str(row["wds_key"]),
                str(row["source_id"]),
                str(row["flac_sha256"]),
            )
        )

    wav_rows = _audio_rows(args.captionstew_root, "WavCaps")
    for row in wav_rows:
        split = str(row["splits"][0])
        if split not in WAVCAPS_SPLITS:
            continue
        records.append(
            (
                "WavCaps",
                str(row["audio_id"]),
                wav_durations[split][str(row["source_id"])],
                str(row["wds_shard"]),
                str(row["wds_key"]),
                str(row["source_id"]),
                str(row["flac_sha256"]),
            )
        )

    selected_by_dataset = {
        dataset: _selected_texts(
            args.captionstew_root,
            dataset,
            {audio_id for name, audio_id, *_ in records if name == dataset},
            seed=args.seed,
            epoch=args.epoch,
        )
        for dataset in ("ParaSpeechCaps-Base", "WavCaps")
    }
    records.sort(key=lambda item: (item[0], item[3], item[4]))
    if args.duration_sidecar_output is not None:
        _write_duration_sidecar(args.duration_sidecar_output, records)
    if args.sidecar_only:
        if args.duration_sidecar_output is None:
            raise ValueError("--sidecar-only requires --duration-sidecar-output")
        print(
            json.dumps(
                {
                    "format_version": 1,
                    "rows": len(records),
                    "duration_sidecar": str(args.duration_sidecar_output.resolve()),
                    "audio_decode_required": False,
                    "num_samples_rule": "round(duration_seconds * 16000)",
                },
                indent=2,
            )
        )
        return
    duration_source = "upstream duration metadata"
    if args.exact_duration_sidecar is not None:
        exact_table = pq.read_table(
            args.exact_duration_sidecar,
            columns=["audio_id", "num_samples", "num_samples_source"],
        )
        sources = set(exact_table["num_samples_source"].to_pylist())
        exact_sources = {
            "flac_streaminfo",
            "libsndfile_header",
            "flac_empty",
        }
        if not sources or not sources.issubset(exact_sources):
            raise ValueError(f"Exact sidecar has non-exact sources: {sources}")
        exact_samples = {
            str(audio_id): (int(num_samples), str(source))
            for audio_id, num_samples, source in zip(
                exact_table["audio_id"].to_pylist(),
                exact_table["num_samples"].to_pylist(),
                exact_table["num_samples_source"].to_pylist(),
                strict=True,
            )
        }
        missing = [
            audio_id for _, audio_id, *_ in records if audio_id not in exact_samples
        ]
        if missing:
            raise ValueError(
                f"Exact duration sidecar lacks {len(missing)} selected rows; "
                f"first={missing[0]!r}"
            )
        invalid_audio_ids = {
            audio_id
            for audio_id, (num_samples, source) in exact_samples.items()
            if num_samples <= 0 or source == "flac_empty"
        }
        records = [
            (dataset, audio_id, exact_samples[audio_id][0] / 16_000, *tail)
            for dataset, audio_id, _, *tail in records
            if audio_id not in invalid_audio_ids
        ]
        duration_source = "exact FLAC STREAMINFO num_samples"
    else:
        invalid_audio_ids = set()
    rng = random.Random(args.seed + args.epoch)
    rng.shuffle(records)
    text_lengths = _text_lengths(
        [selected_by_dataset[name][audio_id] for name, audio_id, *_ in records],
        hf_cache=args.hf_cache,
    )
    lengths = [
        text_length + 2 + _projected_audio_tokens(duration)
        for (_, _, duration, *_), text_length in zip(records, text_lengths, strict=True)
    ]
    lengths_by_audio_id = {
        audio_id: length
        for (_, audio_id, *_), length in zip(records, lengths, strict=True)
    }
    runtime_worker_lengths = _runtime_worker_lengths(
        records,
        lengths_by_audio_id,
        seed=args.seed,
        epoch=args.epoch,
        num_workers=args.runtime_num_workers,
        shard_shuffle=args.runtime_shard_shuffle,
        sample_shuffle=args.runtime_sample_shuffle,
    )
    sorted_lengths = sorted(lengths)

    def quantile(fraction: float) -> int:
        return sorted_lengths[round((len(sorted_lengths) - 1) * fraction)]

    result = {
        "format_version": 2,
        "seed": args.seed,
        "epoch": args.epoch,
        "logical_examples": len(records),
        "invalid_audio_examples_excluded": len(invalid_audio_ids),
        "dataset_examples": {
            dataset: sum(name == dataset for name, *_ in records)
            for dataset in ("ParaSpeechCaps-Base", "WavCaps")
        },
        "duration_hours": sum(item[2] for item in records) / 3600,
        "total_lfm_input_tokens": sum(lengths),
        "mean_lfm_tokens_per_example": sum(lengths) / len(lengths),
        "length_quantiles": {
            "p50": quantile(0.50),
            "p90": quantile(0.90),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
            "max": max(lengths),
        },
        "planner_by_buffer": {
            str(planning_buffer): {
                str(capacity): _pack(lengths, capacity, planning_buffer)
                for capacity in PACK_LENGTHS
            }
            for planning_buffer in PLANNING_BUFFERS
        },
        "planner_settings": {
            "planning_buffer_examples_tested": list(PLANNING_BUFFERS),
            "max_examples_per_pack": None,
            "best_fit_decreasing": True,
            "one_pack_per_optimizer_step": True,
        },
        "runtime_worker_planner": {
            "num_workers": args.runtime_num_workers,
            "shard_shuffle": args.runtime_shard_shuffle,
            "sample_shuffle": args.runtime_sample_shuffle,
            "planning_buffer_examples": args.runtime_planning_buffer,
            "worker_examples": [len(values) for values in runtime_worker_lengths],
            "by_capacity": {
                str(capacity): _combine_worker_packing(
                    runtime_worker_lengths,
                    capacity,
                    args.runtime_planning_buffer,
                )
                for capacity in PACK_LENGTHS
            },
        },
        "metadata_only_audio_lengths": {
            "audio_decode_required": False,
            "source": duration_source,
            "num_samples_rule": (
                "FLAC STREAMINFO total_samples"
                if args.exact_duration_sidecar is not None
                else "round(duration_seconds * 16000)"
            ),
            "exact_sidecar_path": (
                str(args.exact_duration_sidecar.resolve())
                if args.exact_duration_sidecar is not None
                else None
            ),
            "sidecar_path": (
                str(args.duration_sidecar_output.resolve())
                if args.duration_sidecar_output is not None
                else None
            ),
            "sidecar_rows": len(records) + len(invalid_audio_ids),
            "usable_sidecar_rows": len(records),
        },
        "runtime_worker_pipeline": {
            "planning_window_payload": "catalog/TAR byte-range references only",
            "flac_and_log_mel_materialization": "selected pack only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
