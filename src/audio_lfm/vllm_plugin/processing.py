from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

MAX_AUDIO_ITEMS_PER_PROMPT = 3
MAX_TOTAL_AUDIO_SECONDS = 720.0


def replacement_token_ids(audio_token_index: int, audio_token_length: int) -> list[int]:
    if audio_token_length <= 2:
        raise ValueError("Invalid audio token length")
    return [audio_token_index] * audio_token_length


def split_audio_waveform(waveform: object, *, chunk_samples: int) -> list[np.ndarray]:
    """Split one logical audio item without dropping or padding source samples."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio waveform must be a nonempty mono array")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    if not np.isfinite(audio).all():
        raise ValueError("Audio waveform contains non-finite samples")
    return [
        np.ascontiguousarray(audio[start : start + chunk_samples])
        for start in range(0, audio.size, chunk_samples)
    ]


def max_audio_tokens_per_item(
    *, seq_len: int, item_count: int, stack_factor: int
) -> int:
    """Return a scheduler-safe per-item budget for the configured context."""
    if seq_len <= 0 or item_count <= 0 or stack_factor <= 0:
        raise ValueError(
            "Sequence length, item count, and stack factor must be positive"
        )
    duration_limit = round(MAX_TOTAL_AUDIO_SECONDS * 50 / stack_factor) + 2
    return min(duration_limit, seq_len // item_count)


def dummy_waveform_num_samples(*, token_budget: int, stack_factor: int) -> int:
    """Choose a waveform length whose projected token count meets the budget."""
    if token_budget <= 2:
        return 1
    projected_frames = token_budget - 2
    encoder_frames = (projected_frames - 1) * stack_factor + 1
    mel_frames = 2 * encoder_frames - 1
    return mel_frames * 160


def extract_chunked_audio_features(
    feature_extractor: Any,
    waveform: object,
    *,
    sample_rate: int,
    chunk_samples: int,
    stack_factor: int,
) -> tuple[Any, int, Any, int]:
    """Extract every chunk and retain boundaries for independent encoding."""
    import torch

    from audio_lfm.model.frontends.whisper_math import whisper_encoder_lengths

    chunks = split_audio_waveform(waveform, chunk_samples=chunk_samples)
    batch = feature_extractor(
        chunks,
        sampling_rate=sample_rate,
        padding="longest",
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    lengths = batch.attention_mask.sum(dim=-1).to(torch.long)
    trimmed = [
        value[..., : int(length.item())]
        for value, length in zip(batch.input_features, lengths, strict=True)
    ]
    features = torch.cat(trimmed, dim=-1).float().contiguous()
    feature_length = int(lengths.sum().item())
    encoder_length = int(whisper_encoder_lengths(lengths).sum().item())
    token_length = (encoder_length + stack_factor - 1) // stack_factor + 2
    return features, feature_length, lengths, token_length


class AudioLfm2Processor:
    def __init__(self, *, tokenizer: Any, feature_extractor: Any, config: Any) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.config = config


def build_processing_classes() -> tuple[type[Any], type[Any], type[Any]]:
    """Build against vLLM 0.27.1 lazily so training never imports vLLM."""
    from vllm.multimodal.processing import (
        BaseDummyInputsBuilder,
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
    )

    from audio_lfm.vllm_plugin.data_parser import build_vllm_data_parser_class

    parser_class = build_vllm_data_parser_class()

    class AudioLfm2ProcessingInfo(BaseProcessingInfo):
        def get_supported_mm_limits(self) -> dict[str, int]:
            return {"audio": MAX_AUDIO_ITEMS_PER_PROMPT}

        def get_mm_max_tokens_per_item(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int] | None = None,
        ) -> Mapping[str, int]:
            config = self.get_hf_config()
            counts = mm_counts or {}
            count = max(1, int(counts.get("audio", 1)))
            return {
                "audio": max_audio_tokens_per_item(
                    seq_len=seq_len,
                    item_count=count,
                    stack_factor=int(config.projector_config["stack_factor"]),
                )
            }

        def get_data_parser(self) -> Any:
            config = self.get_hf_config()
            return parser_class(
                target_sr=float(config.audio_sample_rate),
                target_channels=1,
            )

        def get_hf_processor(self, **kwargs: Any) -> AudioLfm2Processor:
            from transformers import WhisperFeatureExtractor

            config = self.get_hf_config()
            extractor = WhisperFeatureExtractor.from_pretrained(
                config.audio_model_id, revision=config.audio_model_revision
            )
            return AudioLfm2Processor(
                tokenizer=self.get_tokenizer(),
                feature_extractor=extractor,
                config=config,
            )

    class AudioLfm2DummyInputsBuilder(BaseDummyInputsBuilder):
        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            config = self.info.get_hf_config()
            return config.audio_token * int(mm_counts.get("audio", 0))

        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, Any],
        ) -> dict[str, Any]:
            config = self.info.get_hf_config()
            count = int(mm_counts.get("audio", 0))
            token_budget = max_audio_tokens_per_item(
                seq_len=seq_len,
                item_count=max(1, count),
                stack_factor=int(config.projector_config["stack_factor"]),
            )
            num_samples = dummy_waveform_num_samples(
                token_budget=token_budget,
                stack_factor=int(config.projector_config["stack_factor"]),
            )
            waveform = np.zeros(num_samples, dtype=np.float32)
            return {"audio": [waveform] * count}

    class AudioLfm2MultiModalProcessor(BaseMultiModalProcessor):
        def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
            tok_kwargs: Mapping[str, object],
        ) -> Any:
            import torch
            from transformers import BatchFeature

            processor = self.info.get_hf_processor(**mm_kwargs)
            audios = list(mm_data.get("audios", []))  # type: ignore[arg-type]
            input_ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
            if not audios:
                return BatchFeature({"input_ids": [input_ids]})
            config = processor.config
            chunk_samples = int(config.max_audio_seconds * config.audio_sample_rate)
            stack_factor = int(config.projector_config["stack_factor"])
            features: list[torch.Tensor] = []
            feature_lengths: list[int] = []
            chunk_lengths: list[torch.Tensor] = []
            token_lengths: list[int] = []
            for audio in audios:
                feature, feature_length, lengths, token_length = (
                    extract_chunked_audio_features(
                        processor.feature_extractor,
                        audio,
                        sample_rate=int(config.audio_sample_rate),
                        chunk_samples=chunk_samples,
                        stack_factor=stack_factor,
                    )
                )
                features.append(feature)
                feature_lengths.append(feature_length)
                chunk_lengths.append(lengths)
                token_lengths.append(token_length)
            return BatchFeature(
                {
                    "input_ids": [input_ids],
                    "audio_features": features,
                    "audio_feature_length": torch.tensor(feature_lengths),
                    "audio_chunk_length": chunk_lengths,
                    "audio_token_length": torch.tensor(token_lengths),
                }
            )

        def _hf_processor_applies_updates(self, *args: Any, **kwargs: Any) -> bool:
            return False

        def _get_mm_fields_config(
            self,
            hf_inputs: Any,
            hf_processor_mm_kwargs: Mapping[str, object],
        ) -> Mapping[str, Any]:
            from vllm.multimodal.inputs import MultiModalFieldConfig

            return {
                name: MultiModalFieldConfig.batched("audio")
                for name in (
                    "audio_features",
                    "audio_feature_length",
                    "audio_chunk_length",
                    "audio_token_length",
                    "audio_embeds",
                )
                if name in hf_inputs
            }

        def _get_prompt_updates(
            self,
            mm_items: Any,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: Any,
        ) -> list[Any]:
            config = self.info.get_hf_config()
            out_mm_data = out_mm_kwargs.get_data()

            def replacement(index: int) -> list[int]:
                length = int(out_mm_data["audio_token_length"][index])
                return replacement_token_ids(config.audio_token_index, length)

            return [
                PromptReplacement(
                    modality="audio",
                    target=[config.audio_token_index],
                    replacement=replacement,
                )
            ]

    return (
        AudioLfm2ProcessingInfo,
        AudioLfm2DummyInputsBuilder,
        AudioLfm2MultiModalProcessor,
    )
