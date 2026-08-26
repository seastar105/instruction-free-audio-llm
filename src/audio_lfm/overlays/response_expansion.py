from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from audio_lfm.prompts import (
    EXPANSION_SYSTEM_PROMPT_SHA256,
    build_caption_expansion_messages,
    caption_expansion_prompt_template_sha256,
)
from audio_lfm.utils.hashing import canonical_sha256, sha256_text

RESPONSE_TYPE = "audio_assistant_response"
OVERLAY_SOURCE = "caption_expansion_v1"
VLLM_VERSION = "0.27.1"
MODEL_CARD_TEMPERATURE = 0.1
MODEL_CARD_TOP_K = 50
MODEL_CARD_REPETITION_PENALTY = 1.05
COMBINED_PARASPEECH_SOURCE_TYPE = "style_caption+transcription"
COMBINED_PARASPEECH_TEMPLATE = (
    "Style caption: {style_caption}\nTranscription: {transcription}"
)
OVERLAY_SCHEMA = pa.schema(
    [
        pa.field("audio_id", pa.string(), nullable=False),
        pa.field("target_id", pa.string(), nullable=False),
        pa.field("target_type", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("generator_model", pa.string(), nullable=False),
        pa.field("generator_revision", pa.string(), nullable=False),
        pa.field("prompt_sha256", pa.string(), nullable=False),
        pa.field("review_status", pa.string(), nullable=False),
        pa.field("official_target_id", pa.string(), nullable=False),
        pa.field("official_style_caption_target_id", pa.string(), nullable=True),
        pa.field("official_transcription_target_id", pa.string(), nullable=True),
        pa.field("source_target_type", pa.string(), nullable=False),
        pa.field("response_type", pa.string(), nullable=False),
        pa.field("overlay_kind", pa.string(), nullable=False),
        pa.field("overlay_source", pa.string(), nullable=False),
        pa.field("expansion_system_prompt_sha256", pa.string(), nullable=False),
        pa.field("expansion_prompt_template_sha256", pa.string(), nullable=False),
        pa.field("rendered_prompt_sha256", pa.string(), nullable=False),
        pa.field("generation_config_sha256", pa.string(), nullable=False),
        pa.field("expansion_recipe_sha256", pa.string(), nullable=False),
        pa.field("chat_template_sha256", pa.string(), nullable=False),
        pa.field("finish_reason", pa.string(), nullable=False),
        pa.field("stop_reason", pa.string(), nullable=True),
        pa.field("truncated", pa.bool_(), nullable=False),
        pa.field("generated_token_count", pa.int64(), nullable=False),
    ]
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        pq.write_table(
            pa.Table.from_pylist([dict(row) for row in rows], schema=OVERLAY_SCHEMA),
            temporary,
            compression="zstd",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _immutable_revision(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("model_revision must be an immutable 40-character SHA")
    return normalized


def _load_source_targets(
    catalog_dir: str | Path, source_target_types: Iterable[str]
) -> list[dict[str, Any]]:
    allowed = tuple(sorted(set(source_target_types)))
    if not allowed:
        raise ValueError("At least one source target type is required")
    dataset = pads.dataset(catalog_dir, format="parquet")
    required = {"audio_id", "target_id", "target_type", "text", "split"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Official target catalog lacks columns: {sorted(missing)}")
    table = dataset.to_table(columns=sorted(required))
    mask = pc.is_in(table["target_type"], value_set=pa.array(allowed))
    rows = cast(list[dict[str, Any]], table.filter(mask).to_pylist())
    rows.sort(key=lambda row: str(row["target_id"]))
    seen: set[str] = set()
    for row in rows:
        target_id = str(row["target_id"])
        if target_id in seen:
            raise ValueError(f"Duplicate official target_id: {target_id}")
        seen.add(target_id)
        if not str(row["text"]).strip():
            raise ValueError(f"Official target {target_id!r} has empty text")
    if not rows:
        raise ValueError(f"No targets of types {allowed!r} found beneath {catalog_dir}")
    return rows


def _combine_paraspeech_sources(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    transcripts: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row["target_type"] != "transcription":
            continue
        audio_id = str(row["audio_id"])
        if audio_id in transcripts:
            raise ValueError(f"Multiple transcripts found for audio {audio_id!r}")
        transcripts[audio_id] = row
    combined: list[dict[str, Any]] = []
    for style in rows:
        if style["target_type"] != "style_caption":
            continue
        audio_id = str(style["audio_id"])
        if audio_id not in transcripts:
            raise ValueError(f"No transcript found for audio {audio_id!r}")
        transcript = transcripts[audio_id]
        style_id = str(style["target_id"])
        transcript_id = str(transcript["target_id"])
        combined.append(
            {
                "audio_id": audio_id,
                "target_id": style_id,
                "target_type": COMBINED_PARASPEECH_SOURCE_TYPE,
                "text": COMBINED_PARASPEECH_TEMPLATE.format(
                    style_caption=str(style["text"]),
                    transcription=str(transcript["text"]),
                ),
                "split": str(style["split"]),
                "official_style_caption_target_id": style_id,
                "official_transcription_target_id": transcript_id,
            }
        )
    if not combined:
        raise ValueError("No ParaSpeech style-caption/transcription pairs found")
    combined.sort(key=lambda row: str(row["target_id"]))
    return combined


def _completed_output_stats(output_dir: Path) -> tuple[set[str], int]:
    parts = sorted(output_dir.glob("part-*.parquet"))
    if not parts:
        return set(), 0
    table = pads.dataset(parts, format="parquet").to_table(
        columns=["official_target_id", "truncated"]
    )
    values = [str(value) for value in table["official_target_id"].to_pylist()]
    if len(values) != len(set(values)):
        raise RuntimeError("Existing expansion parts contain duplicate source targets")
    return set(values), sum(bool(value) for value in table["truncated"].to_pylist())


def _render_prompt(tokenizer: Any, caption: str) -> str:
    rendered = tokenizer.apply_chat_template(
        build_caption_expansion_messages(caption),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("Tokenizer returned a non-string rendered prompt")
    return rendered


def _recipe(
    *,
    model_id: str,
    model_revision: str,
    chat_template_sha256: str,
    source_target_types: Sequence[str],
    max_tokens: int,
    combine_style_caption_and_transcription: bool,
    request_batch_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
) -> tuple[dict[str, Any], str, str]:
    generation = {
        "backend": "vllm",
        "vllm_version": VLLM_VERSION,
        "parameter_source": (
            "LiquidAI/LFM2.5-1.2B-Instruct model card generation parameters"
        ),
        "sampling_params": {
            "temperature": MODEL_CARD_TEMPERATURE,
            "top_k": MODEL_CARD_TOP_K,
            "repetition_penalty": MODEL_CARD_REPETITION_PENALTY,
            "max_tokens": max_tokens,
        },
        "do_sample": True,
        "all_other_sampling_params": "vllm_defaults",
        "engine_batching": {
            "request_batch_size": request_batch_size,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": max_model_len,
        },
    }
    generation_hash = canonical_sha256(generation)
    recipe = {
        "format_version": 2,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": model_id,
        "tokenizer_revision": model_revision,
        "chat_template_sha256": chat_template_sha256,
        "expansion_system_prompt_sha256": EXPANSION_SYSTEM_PROMPT_SHA256,
        "expansion_prompt_template_sha256": (
            caption_expansion_prompt_template_sha256()
        ),
        "generation_config_sha256": generation_hash,
        "generation": generation,
        "source_target_types": list(source_target_types),
        "combine_style_caption_and_transcription": (
            combine_style_caption_and_transcription
        ),
        "combined_source_template": (
            COMBINED_PARASPEECH_TEMPLATE
            if combine_style_caption_and_transcription
            else None
        ),
        "response_type": RESPONSE_TYPE,
        "overlay_source": OVERLAY_SOURCE,
    }
    return recipe, canonical_sha256(recipe), generation_hash


def expand_catalog_with_vllm(
    *,
    catalog_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    source_target_types: Sequence[str],
    model_path: str | Path,
    model_id: str,
    model_revision: str,
    max_tokens: int = 1024,
    request_batch_size: int = 8192,
    max_num_seqs: int = 512,
    max_num_batched_tokens: int = 32768,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.90,
    max_targets: int | None = None,
    tokenizer: Any | None = None,
    llm: Any | None = None,
    combine_style_caption_and_transcription: bool = False,
) -> dict[str, Any]:
    """Expand official caption surrogates into a resumable Parquet overlay."""
    if max_tokens <= 0 or request_batch_size <= 0 or max_num_seqs <= 0:
        raise ValueError("Generation and batching limits must be positive")
    if max_model_len <= max_tokens:
        raise ValueError("max_model_len must leave room for an input prompt")
    revision = _immutable_revision(model_revision)
    target_types = tuple(sorted(set(source_target_types)))
    source_rows = _load_source_targets(catalog_dir, target_types)
    if combine_style_caption_and_transcription:
        if dataset_name != "ParaSpeechCaps-Base" or set(target_types) != {
            "style_caption",
            "transcription",
        }:
            raise ValueError(
                "Combined style-caption/transcription expansion requires "
                "ParaSpeechCaps-Base and both source target types"
            )
        source_rows = _combine_paraspeech_sources(source_rows)
    if max_targets is not None:
        source_rows = source_rows[:max_targets]

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=Path(model_path).exists(),
            trust_remote_code=False,
        )
    chat_template = tokenizer.chat_template
    if not chat_template:
        raise ValueError("Expansion tokenizer has no chat template")
    chat_template_hash = sha256_text(chat_template)
    recipe, recipe_hash, generation_hash = _recipe(
        model_id=model_id,
        model_revision=revision,
        chat_template_sha256=chat_template_hash,
        source_target_types=target_types,
        max_tokens=max_tokens,
        combine_style_caption_and_transcription=(
            combine_style_caption_and_transcription
        ),
        request_batch_size=request_batch_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "expansion_manifest.json"
    lock_path = output / "decoder_lock.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("expansion_recipe_sha256") != recipe_hash:
            raise RuntimeError("Existing expansion output uses a different recipe")
    decoder_lock = {
        "format_version": 2,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_id": model_id,
        "tokenizer_revision": revision,
        "chat_template_sha256": chat_template_hash,
        "expansion_system_prompt_sha256": EXPANSION_SYSTEM_PROMPT_SHA256,
        "expansion_prompt_template_sha256": (
            caption_expansion_prompt_template_sha256()
        ),
        "generation_config_sha256": generation_hash,
        "expansion_recipe_sha256": recipe_hash,
    }
    _atomic_json(lock_path, decoder_lock)

    completed, previously_truncated = _completed_output_stats(output)
    source_target_ids = {str(row["target_id"]) for row in source_rows}
    unexpected = completed - source_target_ids
    if unexpected:
        raise RuntimeError(
            "Existing expansion output contains source targets outside this run: "
            f"{sorted(unexpected)[:3]}"
        )
    pending = [row for row in source_rows if str(row["target_id"]) not in completed]
    initial_completed = len(completed)
    manifest: dict[str, Any] = {
        "format_version": 2,
        "dataset": dataset_name,
        "source_target_types": list(target_types),
        "combine_style_caption_and_transcription": (
            combine_style_caption_and_transcription
        ),
        "source_target_count": len(source_rows),
        "completed_target_count": initial_completed,
        "truncated_target_count": previously_truncated,
        "response_type": RESPONSE_TYPE,
        "overlay_kind": "response",
        "overlay_source": OVERLAY_SOURCE,
        "expansion_recipe_sha256": recipe_hash,
        "expansion_system_prompt_sha256": EXPANSION_SYSTEM_PROMPT_SHA256,
        "expansion_prompt_template_sha256": (
            caption_expansion_prompt_template_sha256()
        ),
        "generation_config_sha256": generation_hash,
        "decoder_model_id": model_id,
        "decoder_revision": revision,
        "tokenizer_revision": revision,
        "chat_template_sha256": chat_template_hash,
        "recipe": recipe,
        "complete": not pending,
    }
    _atomic_json(manifest_path, manifest)
    if not pending:
        return manifest

    if importlib.metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError(f"Caption expansion requires vLLM {VLLM_VERSION}")
    if llm is None:
        from vllm import LLM

        llm = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            dtype="bfloat16",
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=False,
            disable_log_stats=True,
        )
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=MODEL_CARD_TEMPERATURE,
        top_k=MODEL_CARD_TOP_K,
        repetition_penalty=MODEL_CARD_REPETITION_PENALTY,
        max_tokens=max_tokens,
    )
    part_numbers = [
        int(path.stem.removeprefix("part-")) for path in output.glob("part-*.parquet")
    ]
    next_part = max(part_numbers, default=-1) + 1
    truncated_total = previously_truncated
    generated_this_run = 0
    started = time.perf_counter()
    for offset in range(0, len(pending), request_batch_size):
        batch = pending[offset : offset + request_batch_size]
        prompts = [_render_prompt(tokenizer, str(row["text"])) for row in batch]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM output count differs from request count")
        overlay_rows: list[dict[str, Any]] = []
        for source, rendered, request_output in zip(
            batch, prompts, outputs, strict=True
        ):
            completion = request_output.outputs[0]
            response = str(completion.text).strip()
            if not response:
                raise RuntimeError(
                    f"Expansion produced empty text for {source['target_id']}"
                )
            truncated = completion.finish_reason == "length"
            truncated_total += int(truncated)
            identity = canonical_sha256(
                {
                    "dataset": dataset_name,
                    "audio_id": str(source["audio_id"]),
                    "official_target_id": str(source["target_id"]),
                    "official_style_caption_target_id": source.get(
                        "official_style_caption_target_id"
                    ),
                    "official_transcription_target_id": source.get(
                        "official_transcription_target_id"
                    ),
                    "expansion_recipe_sha256": recipe_hash,
                }
            )
            overlay_rows.append(
                {
                    "audio_id": str(source["audio_id"]),
                    "target_id": f"response-{identity[:32]}",
                    "target_type": RESPONSE_TYPE,
                    "text": response,
                    "split": str(source["split"]),
                    "source": OVERLAY_SOURCE,
                    "generator_model": model_id,
                    "generator_revision": revision,
                    "prompt_sha256": caption_expansion_prompt_template_sha256(),
                    "review_status": "unreviewed",
                    "official_target_id": str(source["target_id"]),
                    "official_style_caption_target_id": source.get(
                        "official_style_caption_target_id"
                    ),
                    "official_transcription_target_id": source.get(
                        "official_transcription_target_id"
                    ),
                    "source_target_type": str(source["target_type"]),
                    "response_type": RESPONSE_TYPE,
                    "overlay_kind": "response",
                    "overlay_source": OVERLAY_SOURCE,
                    "expansion_system_prompt_sha256": (EXPANSION_SYSTEM_PROMPT_SHA256),
                    "expansion_prompt_template_sha256": (
                        caption_expansion_prompt_template_sha256()
                    ),
                    "rendered_prompt_sha256": sha256_text(rendered),
                    "generation_config_sha256": generation_hash,
                    "expansion_recipe_sha256": recipe_hash,
                    "chat_template_sha256": chat_template_hash,
                    "finish_reason": str(completion.finish_reason),
                    "stop_reason": (
                        str(completion.stop_reason)
                        if completion.stop_reason is not None
                        else None
                    ),
                    "truncated": truncated,
                    "generated_token_count": len(completion.token_ids),
                }
            )
        destination = output / f"part-{next_part:05d}.parquet"
        if destination.exists():
            raise FileExistsError(f"Expansion part already exists: {destination}")
        _atomic_parquet(destination, overlay_rows)
        next_part += 1
        generated_this_run += len(overlay_rows)
        elapsed = time.perf_counter() - started
        completed_count = initial_completed + generated_this_run
        manifest.update(
            {
                "completed_target_count": completed_count,
                "truncated_target_count": truncated_total,
                "complete": completed_count == len(source_rows),
                "last_part": destination.name,
            }
        )
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "completed": completed_count,
                    "total": len(source_rows),
                    "samples_per_second": generated_this_run / elapsed,
                    "truncated_total": truncated_total,
                    "part": destination.name,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return manifest
