from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import torch


@dataclass(frozen=True)
class TargetRecord:
    audio_id: str
    target_id: str
    target_type: str
    text: str
    split: str
    source: str
    review_status: str
    generator_model: str | None = None
    generator_revision: str | None = None
    prompt_sha256: str | None = None


@dataclass(frozen=True)
class CatalogAudioRecord:
    audio_id: str
    dataset: str
    source_id: str
    splits: tuple[str, ...]
    wds_key: str
    wds_shard: str
    flac_sha256: str
    flac_size: int
    target_count: int


@dataclass
class RawAudioExample:
    audio_id: str
    waveform: torch.Tensor
    sample_rate: int
    source_id: str
    splits: tuple[str, ...]
    style_captions: tuple[TargetRecord, ...]
    transcript: TargetRecord | None
    selected_target: TargetRecord
    metadata: dict[str, Any]
    crop_start_sample: int | None
    original_num_samples: int


@dataclass
class LocalSampleReference:
    """Byte ranges for one sample in an uncompressed local TAR shard."""

    wds_shard: str
    wds_key: str
    flac_offset: int
    flac_size: int
    json_offset: int
    json_size: int

    def __post_init__(self) -> None:
        if not self.wds_shard or not self.wds_key:
            raise ValueError("Local TAR reference requires a shard and key")
        if self.flac_offset < 0 or self.json_offset < 0:
            raise ValueError("Local TAR offsets must be nonnegative")
        if self.flac_size <= 0 or self.json_size <= 0:
            raise ValueError("Local TAR member sizes must be positive")


@dataclass
class DeferredAudioExample:
    """A lightweight planned sample whose FLAC is loaded only after packing."""

    audio_id: str
    sample: dict[str, Any] | None
    local_reference: LocalSampleReference | None
    catalog_record: CatalogAudioRecord
    style_captions: tuple[TargetRecord, ...]
    transcript: TargetRecord | None
    selected_target: TargetRecord
    planned_num_samples: int


@dataclass
class PrecomputedAudioExample:
    audio_id: str
    input_features: torch.Tensor
    effective_encoder_lengths: tuple[int, ...]
    evaluated_num_samples: int
    sample_rate: int
    source_id: str
    splits: tuple[str, ...]
    style_captions: tuple[TargetRecord, ...]
    transcript: TargetRecord | None
    selected_target: TargetRecord
    metadata: dict[str, Any]
    crop_start_sample: int | None
    original_num_samples: int
    encoder_frame_mask: torch.Tensor | None = None
    projected_frame_mask: torch.Tensor | None = None
    projected_frame_indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.input_features.ndim != 3:
            raise ValueError("input_features must be [blocks, mel_bins, mel_frames]")
        if self.input_features.shape[0] != len(self.effective_encoder_lengths):
            raise ValueError("one effective encoder length is required per block")
        if self.input_features.shape[0] == 0:
            raise ValueError("at least one audio block is required")
        if any(length <= 0 for length in self.effective_encoder_lengths):
            raise ValueError("effective encoder lengths must be positive")
        if self.evaluated_num_samples <= 0:
            raise ValueError("evaluated_num_samples must be positive")
        blocks = self.input_features.shape[0]
        if self.encoder_frame_mask is not None:
            if self.encoder_frame_mask.shape != (blocks, 1500):
                raise ValueError("encoder_frame_mask must be [blocks, 1500]")
            if self.encoder_frame_mask.dtype is not torch.bool:
                raise TypeError("encoder_frame_mask must be boolean")
        if self.projected_frame_mask is not None:
            if self.projected_frame_mask.shape != (blocks, 375):
                raise ValueError("projected_frame_mask must be [blocks, 375]")
            if self.projected_frame_mask.dtype is not torch.bool:
                raise TypeError("projected_frame_mask must be boolean")
        if self.projected_frame_indices is not None:
            if self.projected_frame_indices.ndim != 1:
                raise ValueError("projected_frame_indices must be one-dimensional")
            if self.projected_frame_indices.dtype is not torch.long:
                raise TypeError("projected_frame_indices must be torch.long")


@dataclass(frozen=True)
class PreparedText:
    before_audio_ids: tuple[int, ...]
    after_audio_prompt_ids: tuple[int, ...]
    target_suffix_ids: tuple[int, ...]
    target_id: str
    prompt_sha256: str


@dataclass
class PreparedExample:
    raw: PrecomputedAudioExample
    text: PreparedText
    estimated_audio_embedding_length: int
    estimated_total_lfm_length: int

    @property
    def audio_id(self) -> str:
        return self.raw.audio_id


@dataclass
class DeferredPreparedExample:
    raw: DeferredAudioExample
    text: PreparedText
    estimated_audio_embedding_length: int
    estimated_total_lfm_length: int

    @property
    def audio_id(self) -> str:
        return self.raw.audio_id


class PackableExample(Protocol):
    estimated_total_lfm_length: int

    @property
    def audio_id(self) -> str: ...


PackableT = TypeVar("PackableT", bound=PackableExample)


@dataclass
class PackPlan(Generic[PackableT]):
    examples: list[PackableT]
    estimated_total_lfm_length: int


@dataclass
class PackedLayout:
    input_ids: torch.Tensor
    audio_slot_mask: torch.Tensor
    audio_payload_frame_mask: torch.Tensor
    audio_payload_start_mask: torch.Tensor
    audio_payload_end_mask: torch.Tensor
    audio_payload_frame_indices: torch.Tensor
    audio_slot_indices: torch.Tensor
    labels: torch.Tensor
    supervised_token_indices: torch.Tensor
    position_ids: torch.Tensor
    seq_idx: torch.Tensor
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    max_length_q: int
    max_length_k: int
    logical_lengths: list[int]
    audio_ids: list[str]
    target_ids: list[str]
    input_token_count: int
    supervised_token_count: int
    audio_frame_count: int

    def to(self, device: torch.device) -> PackedLayout:
        return PackedLayout(
            input_ids=self.input_ids.to(device, non_blocking=True),
            audio_slot_mask=self.audio_slot_mask.to(device, non_blocking=True),
            audio_payload_frame_mask=self.audio_payload_frame_mask.to(
                device, non_blocking=True
            ),
            audio_payload_start_mask=self.audio_payload_start_mask.to(
                device, non_blocking=True
            ),
            audio_payload_end_mask=self.audio_payload_end_mask.to(
                device, non_blocking=True
            ),
            audio_payload_frame_indices=(
                self.audio_payload_frame_indices.to(device, non_blocking=True)
            ),
            audio_slot_indices=self.audio_slot_indices.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            supervised_token_indices=self.supervised_token_indices.to(
                device, non_blocking=True
            ),
            position_ids=self.position_ids.to(device, non_blocking=True),
            seq_idx=self.seq_idx.to(device, non_blocking=True),
            cu_seq_lens_q=self.cu_seq_lens_q.to(device, non_blocking=True),
            cu_seq_lens_k=self.cu_seq_lens_k.to(device, non_blocking=True),
            max_length_q=self.max_length_q,
            max_length_k=self.max_length_k,
            logical_lengths=self.logical_lengths,
            audio_ids=self.audio_ids,
            target_ids=self.target_ids,
            input_token_count=self.input_token_count,
            supervised_token_count=self.supervised_token_count,
            audio_frame_count=self.audio_frame_count,
        )


@dataclass
class HostAudioBatch:
    input_features: torch.Tensor
    encoder_frame_mask: torch.Tensor
    projected_frame_mask: torch.Tensor
    projected_frame_indices: torch.Tensor
    layout: PackedLayout
    audio_seconds: float

    def to(self, device: torch.device) -> HostAudioBatch:
        return HostAudioBatch(
            input_features=self.input_features.to(device, non_blocking=True),
            encoder_frame_mask=self.encoder_frame_mask.to(device, non_blocking=True),
            projected_frame_mask=self.projected_frame_mask.to(
                device, non_blocking=True
            ),
            projected_frame_indices=self.projected_frame_indices.to(
                device, non_blocking=True
            ),
            layout=self.layout.to(device),
            audio_seconds=self.audio_seconds,
        )


@dataclass
class PackedHostItem:
    batch: HostAudioBatch
    oversized_examples_skipped: int = 0
    decode_failures_skipped: int = 0


@dataclass
class PackedBatch:
    inputs_embeds: torch.Tensor
    labels: torch.Tensor
    supervised_token_indices: torch.Tensor
    position_ids: torch.Tensor
    seq_idx: torch.Tensor
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    max_length_q: int
    max_length_k: int
    logical_lengths: list[int]
    audio_ids: list[str]
    target_ids: list[str]
    input_token_count: int
    supervised_token_count: int
    validate_on_init: bool = True

    def __post_init__(self) -> None:
        if self.validate_on_init:
            self.validate()

    def validate(self) -> None:
        if self.inputs_embeds.ndim != 3 or self.inputs_embeds.shape[0] != 1:
            raise ValueError("inputs_embeds must be [1, total_tokens, hidden_size]")
        total = self.inputs_embeds.shape[1]
        for name, tensor in {
            "labels": self.labels,
            "position_ids": self.position_ids,
            "seq_idx": self.seq_idx,
        }.items():
            if tensor.shape != (1, total):
                raise ValueError(f"{name} must have shape [1, total_tokens]")
        if self.labels.dtype != torch.long or self.position_ids.dtype != torch.long:
            raise TypeError("labels and position_ids must be torch.long")
        if (
            self.supervised_token_indices.ndim != 1
            or self.supervised_token_indices.dtype != torch.long
        ):
            raise TypeError("supervised_token_indices must be one-dimensional long")
        if self.supervised_token_indices.numel() != self.supervised_token_count:
            raise ValueError("supervised token index count mismatch")
        if self.seq_idx.dtype != torch.int32:
            raise TypeError("seq_idx must be torch.int32")
        for name, cu in {
            "cu_seq_lens_q": self.cu_seq_lens_q,
            "cu_seq_lens_k": self.cu_seq_lens_k,
        }.items():
            if cu.dtype != torch.int32:
                raise TypeError(f"{name} must be torch.int32")
            if cu.ndim != 1 or cu.numel() != len(self.logical_lengths) + 1:
                raise ValueError(f"{name} has invalid shape")
            if cu[0].item() != 0 or cu[-1].item() != total:
                raise ValueError(f"{name} does not delimit all tokens")
        if not self.logical_lengths or any(
            length <= 0 for length in self.logical_lengths
        ):
            raise ValueError("logical sequences must be nonempty")
        if sum(self.logical_lengths) != total or self.input_token_count != total:
            raise ValueError("logical lengths do not sum to input token count")
        if len(self.audio_ids) != len(self.logical_lengths):
            raise ValueError("one audio ID is required per logical sequence")
        if len(self.target_ids) != len(self.logical_lengths):
            raise ValueError("one target ID is required per logical sequence")
        offset = 0
        for index, length in enumerate(self.logical_lengths):
            positions = self.position_ids[0, offset : offset + length]
            expected = torch.arange(length, device=positions.device)
            if not torch.equal(positions, expected):
                raise ValueError("position_ids must reset for every sequence")
            if not torch.all(self.seq_idx[0, offset : offset + length] == index):
                raise ValueError("seq_idx is not constant within a sequence")
            if not self.labels[0, offset : offset + length].ne(-100).any():
                raise ValueError("every sequence needs a supervised token")
            offset += length
        supervised = int(self.labels.ne(-100).sum().item())
        if supervised != self.supervised_token_count:
            raise ValueError("supervised token count mismatch")
