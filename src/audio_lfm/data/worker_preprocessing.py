from __future__ import annotations

from dataclasses import dataclass

import torch

from audio_lfm.data.types import PrecomputedAudioExample, RawAudioExample
from audio_lfm.model.frontends.whisper_math import (
    split_waveform_chunks,
    whisper_encoder_lengths,
)


@dataclass(frozen=True)
class AudioPreprocessingConfig:
    model_id: str | None
    revision: str | None
    sample_rate: int = 16_000
    block_seconds: float = 30.0

    @property
    def block_samples(self) -> int:
        return round(self.sample_rate * self.block_seconds)


class WhisperWorkerPreprocessor:
    """One CPU feature extractor instance owned by one DataLoader worker."""

    def __init__(self, config: AudioPreprocessingConfig) -> None:
        from transformers import WhisperFeatureExtractor

        self.config = config
        if config.model_id is None:
            self.feature_extractor = WhisperFeatureExtractor()  # type: ignore[no-untyped-call]
        elif config.revision is None:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                config.model_id
            )
        else:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                config.model_id, revision=config.revision
            )

    def __call__(self, raw: RawAudioExample) -> PrecomputedAudioExample:
        if raw.sample_rate != self.config.sample_rate:
            raise ValueError(
                f"Expected {self.config.sample_rate} Hz, received {raw.sample_rate}"
            )
        chunks = split_waveform_chunks(
            raw.waveform, chunk_samples=self.config.block_samples
        )
        arrays = [chunk.detach().cpu().float().numpy() for chunk in chunks]
        batch = self.feature_extractor(
            arrays,
            sampling_rate=self.config.sample_rate,
            padding="max_length",
            max_length=self.config.block_samples,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = batch.input_features.float().contiguous()
        mel_lengths = batch.attention_mask.sum(dim=-1).to(dtype=torch.long)
        encoder_lengths = whisper_encoder_lengths(mel_lengths)
        encoder_positions = torch.arange(1500)
        projected_positions = torch.arange(375)
        encoder_frame_mask = encoder_positions[None] < encoder_lengths[:, None]
        projected_lengths = torch.div(encoder_lengths + 3, 4, rounding_mode="floor")
        projected_frame_mask = projected_positions[None] < projected_lengths[:, None]
        metadata = dict(raw.metadata)
        audio_contract = dict(metadata.get("audio_contract", {}))
        audio_contract.update(
            {
                "sample_rate": raw.sample_rate,
                "original_num_samples": raw.original_num_samples,
                "evaluated_num_samples": raw.waveform.numel(),
                "whisper_block_samples": self.config.block_samples,
                "whisper_block_count": len(chunks),
                "effective_encoder_lengths": encoder_lengths.tolist(),
            }
        )
        metadata["audio_contract"] = audio_contract
        return PrecomputedAudioExample(
            audio_id=raw.audio_id,
            input_features=input_features,
            effective_encoder_lengths=tuple(int(value) for value in encoder_lengths),
            evaluated_num_samples=raw.waveform.numel(),
            sample_rate=raw.sample_rate,
            source_id=raw.source_id,
            splits=raw.splits,
            style_captions=raw.style_captions,
            transcript=raw.transcript,
            selected_target=raw.selected_target,
            metadata=metadata,
            crop_start_sample=raw.crop_start_sample,
            original_num_samples=raw.original_num_samples,
            encoder_frame_mask=encoder_frame_mask.contiguous(),
            projected_frame_mask=projected_frame_mask.contiguous(),
            projected_frame_indices=torch.nonzero(
                projected_frame_mask.flatten(), as_tuple=False
            )
            .flatten()
            .contiguous(),
        )
