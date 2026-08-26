from __future__ import annotations

import itertools
from collections.abc import Callable

import torch

from audio_lfm.data.types import (
    HostAudioBatch,
    PackedBatch,
    PackedLayout,
    PreparedExample,
)


def build_host_audio_batch(
    *,
    examples: list[PreparedExample],
    vocabulary_size: int,
    max_lfm_tokens: int,
) -> HostAudioBatch:
    """Build a complete CPU batch; safe to call inside a DataLoader worker."""
    encoder_masks = [example.raw.encoder_frame_mask for example in examples]
    projected_masks = [example.raw.projected_frame_mask for example in examples]
    projected_indices = [example.raw.projected_frame_indices for example in examples]
    if any(mask is None for mask in encoder_masks):
        raise ValueError("DataLoader did not provide encoder-frame masks")
    if any(mask is None for mask in projected_masks):
        raise ValueError("DataLoader did not provide projected-frame masks")
    if any(indices is None for indices in projected_indices):
        raise ValueError("DataLoader did not provide projected-frame indices")
    adjusted_indices: list[torch.Tensor] = []
    projected_offset = 0
    for example, indices in zip(examples, projected_indices, strict=True):
        if indices is not None:
            adjusted_indices.append(indices + projected_offset)
        projected_offset += example.raw.input_features.shape[0] * 375
    return HostAudioBatch(
        input_features=torch.cat(
            [example.raw.input_features.detach() for example in examples], dim=0
        ),
        encoder_frame_mask=torch.cat(
            [mask for mask in encoder_masks if mask is not None], dim=0
        ),
        projected_frame_mask=torch.cat(
            [mask for mask in projected_masks if mask is not None], dim=0
        ),
        projected_frame_indices=torch.cat(adjusted_indices),
        layout=build_packed_layout(
            examples=examples,
            vocabulary_size=vocabulary_size,
            max_lfm_tokens=max_lfm_tokens,
        ),
        audio_seconds=sum(
            example.raw.evaluated_num_samples / example.raw.sample_rate
            for example in examples
        ),
    )


def build_packed_layout(
    *,
    examples: list[PreparedExample],
    vocabulary_size: int,
    max_lfm_tokens: int,
) -> PackedLayout:
    logical_ids: list[torch.Tensor] = []
    logical_audio_masks: list[torch.Tensor] = []
    logical_labels: list[torch.Tensor] = []
    payload_frame_masks: list[torch.Tensor] = []
    payload_start_masks: list[torch.Tensor] = []
    payload_end_masks: list[torch.Tensor] = []
    lengths: list[int] = []
    audio_ids: list[str] = []
    target_ids: list[str] = []
    supervised_tokens = 0
    audio_frames = 0
    for example in examples:
        text = example.text
        id_groups = (
            text.before_audio_ids,
            text.after_audio_prompt_ids,
            text.target_suffix_ids,
        )
        if any(
            token < 0 or token >= vocabulary_size for ids in id_groups for token in ids
        ):
            raise ValueError("Text token ID is outside the frozen LFM vocabulary")
        frame_count = example.estimated_audio_embedding_length
        audio_count = frame_count + 2
        before = torch.tensor(text.before_audio_ids, dtype=torch.long)
        after = torch.tensor(text.after_audio_prompt_ids, dtype=torch.long)
        target = torch.tensor(text.target_suffix_ids, dtype=torch.long)
        placeholder = torch.zeros(audio_count, dtype=torch.long)
        ids = torch.cat([before, placeholder, after, target])
        audio_mask = torch.zeros(ids.shape[0], dtype=torch.bool)
        audio_mask[before.numel() : before.numel() + audio_count] = True
        labels = torch.full_like(ids, -100)
        labels[-target.numel() :] = target
        payload_frame = torch.zeros(audio_count, dtype=torch.bool)
        payload_frame[1:-1] = True
        payload_start = torch.zeros(audio_count, dtype=torch.bool)
        payload_start[0] = True
        payload_end = torch.zeros(audio_count, dtype=torch.bool)
        payload_end[-1] = True
        logical_ids.append(ids)
        logical_audio_masks.append(audio_mask)
        logical_labels.append(labels)
        payload_frame_masks.append(payload_frame)
        payload_start_masks.append(payload_start)
        payload_end_masks.append(payload_end)
        lengths.append(ids.numel())
        audio_ids.append(example.raw.audio_id)
        target_ids.append(text.target_id)
        supervised_tokens += target.numel()
        audio_frames += frame_count
    total = sum(lengths)
    if total > max_lfm_tokens:
        raise RuntimeError(
            f"Packed layout length {total} exceeds {max_lfm_tokens}; "
            f"audio_ids={audio_ids}, lengths={lengths}"
        )
    cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)
    flat_audio_mask = torch.cat(logical_audio_masks)
    flat_payload_frame_mask = torch.cat(payload_frame_masks)
    flat_labels = torch.cat(logical_labels).unsqueeze(0)

    return PackedLayout(
        input_ids=torch.cat(logical_ids),
        audio_slot_mask=flat_audio_mask,
        audio_payload_frame_mask=flat_payload_frame_mask,
        audio_payload_start_mask=torch.cat(payload_start_masks),
        audio_payload_end_mask=torch.cat(payload_end_masks),
        audio_payload_frame_indices=torch.nonzero(
            flat_payload_frame_mask, as_tuple=False
        ).flatten(),
        audio_slot_indices=torch.nonzero(flat_audio_mask, as_tuple=False).flatten(),
        labels=flat_labels,
        supervised_token_indices=torch.nonzero(
            flat_labels[:, 1:].reshape(-1).ne(-100), as_tuple=False
        ).flatten(),
        position_ids=torch.cat([torch.arange(length) for length in lengths]).unsqueeze(
            0
        ),
        seq_idx=torch.cat(
            [
                torch.full((length,), index, dtype=torch.int32)
                for index, length in enumerate(lengths)
            ]
        ).unsqueeze(0),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu.clone(),
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        logical_lengths=lengths,
        audio_ids=audio_ids,
        target_ids=target_ids,
        input_token_count=total,
        supervised_token_count=supervised_tokens,
        audio_frame_count=audio_frames,
    )


def build_vectorized_packed_batch(
    *,
    layout: PackedLayout,
    projected_audio: torch.Tensor,
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    audio_start: torch.Tensor,
    audio_end: torch.Tensor,
) -> PackedBatch:
    if projected_audio.ndim != 2:
        raise ValueError("Projected audio must be [frames, hidden_size]")
    if projected_audio.shape[0] != layout.audio_frame_count:
        raise ValueError("Projected audio length differs from packed layout")
    hidden_size = projected_audio.shape[1]
    payload_slots = layout.audio_payload_frame_mask.shape[0]
    payload = projected_audio.new_zeros((payload_slots, hidden_size))
    payload = static_masked_scatter(
        payload,
        layout.audio_payload_frame_mask[:, None].expand(-1, hidden_size),
        projected_audio,
        layout.audio_payload_frame_indices,
    )
    payload = torch.where(
        layout.audio_payload_start_mask[:, None],
        audio_start.to(dtype=payload.dtype)[None],
        payload,
    )
    payload = torch.where(
        layout.audio_payload_end_mask[:, None],
        audio_end.to(dtype=payload.dtype)[None],
        payload,
    )
    with torch.no_grad():
        text_embeddings = embed_tokens(layout.input_ids).detach()
    inputs_embeds = static_masked_scatter(
        text_embeddings,
        layout.audio_slot_mask[:, None].expand(-1, hidden_size),
        payload,
        layout.audio_slot_indices,
    )
    return PackedBatch(
        inputs_embeds=inputs_embeds.unsqueeze(0),
        labels=layout.labels,
        supervised_token_indices=layout.supervised_token_indices,
        position_ids=layout.position_ids,
        seq_idx=layout.seq_idx,
        cu_seq_lens_q=layout.cu_seq_lens_q,
        cu_seq_lens_k=layout.cu_seq_lens_k,
        max_length_q=layout.max_length_q,
        max_length_k=layout.max_length_k,
        logical_lengths=layout.logical_lengths,
        audio_ids=layout.audio_ids,
        target_ids=layout.target_ids,
        input_token_count=layout.input_token_count,
        supervised_token_count=layout.supervised_token_count,
        validate_on_init=False,
    )


class _StaticMaskedScatter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        destination: torch.Tensor,
        mask: torch.Tensor,
        source: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(row_indices)
        return destination.masked_scatter(mask, source)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, torch.Tensor | None, None]:
        (row_indices,) = ctx.saved_tensors
        grad_destination = None
        if ctx.needs_input_grad[0]:
            grad_destination = grad_output.clone()
            grad_destination.index_fill_(0, row_indices, 0)
        grad_source = None
        if ctx.needs_input_grad[2]:
            grad_source = grad_output.index_select(0, row_indices)
        return grad_destination, None, grad_source, None


def static_masked_scatter(
    destination: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Masked-scatter forward with an asynchronous, fixed-size backward."""
    return _StaticMaskedScatter.apply(destination, mask, source, row_indices)


def build_packed_batch(
    *,
    examples: list[PreparedExample],
    projected_audio: list[torch.Tensor],
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    audio_start: torch.Tensor,
    audio_end: torch.Tensor,
    device: torch.device,
    vocabulary_size: int,
    max_lfm_tokens: int,
) -> PackedBatch:
    if len(examples) != len(projected_audio):
        raise ValueError("Projected audio count differs from example count")
    logical_embeds: list[torch.Tensor] = []
    logical_labels: list[torch.Tensor] = []
    lengths: list[int] = []
    audio_ids: list[str] = []
    target_ids: list[str] = []
    for example, audio in zip(examples, projected_audio, strict=True):
        text = example.text
        id_groups = (
            text.before_audio_ids,
            text.after_audio_prompt_ids,
            text.target_suffix_ids,
        )
        if any(
            token < 0 or token >= vocabulary_size for ids in id_groups for token in ids
        ):
            raise ValueError("Text token ID is outside the frozen LFM vocabulary")
        before_ids = torch.tensor(
            text.before_audio_ids, device=device, dtype=torch.long
        )
        after_ids = torch.tensor(
            text.after_audio_prompt_ids, device=device, dtype=torch.long
        )
        target = torch.tensor(text.target_suffix_ids, device=device, dtype=torch.long)
        with torch.no_grad():
            before_embed = embed_tokens(before_ids).detach()
            after_embed = embed_tokens(after_ids).detach()
            target_embed = embed_tokens(target).detach()
        # The concatenation remains differentiable with respect to the projector.
        sequence = torch.cat(
            [
                before_embed,
                audio_start[None].to(audio.dtype),
                audio,
                audio_end[None].to(audio.dtype),
                after_embed,
                target_embed,
            ]
        )
        suffix_start = sequence.shape[0] - target.numel()
        labels = torch.full((sequence.shape[0],), -100, device=device, dtype=torch.long)
        labels[suffix_start:] = target
        logical_embeds.append(sequence)
        logical_labels.append(labels)
        lengths.append(sequence.shape[0])
        audio_ids.append(example.raw.audio_id)
        target_ids.append(text.target_id)
    total = sum(lengths)
    if total > max_lfm_tokens:
        estimates = [example.estimated_total_lfm_length for example in examples]
        raise RuntimeError(
            f"Actual packed length {total} exceeds {max_lfm_tokens}; "
            f"audio_ids={audio_ids}, estimates={estimates}, actual={lengths}"
        )
    inputs_embeds = torch.cat(logical_embeds).unsqueeze(0)
    labels = torch.cat(logical_labels).unsqueeze(0)
    supervised_token_indices = torch.nonzero(
        labels[:, 1:].reshape(-1).ne(-100), as_tuple=False
    ).flatten()
    position_ids = torch.cat(
        [torch.arange(length, device=device) for length in lengths]
    ).unsqueeze(0)
    seq_idx = torch.cat(
        [
            torch.full((length,), index, dtype=torch.int32, device=device)
            for index, length in enumerate(lengths)
        ]
    ).unsqueeze(0)
    cu = torch.tensor(
        [0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device
    )
    return PackedBatch(
        inputs_embeds=inputs_embeds,
        labels=labels,
        supervised_token_indices=supervised_token_indices,
        position_ids=position_ids,
        seq_idx=seq_idx,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu.clone(),
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        logical_lengths=lengths,
        audio_ids=audio_ids,
        target_ids=target_ids,
        input_token_count=total,
        supervised_token_count=supervised_token_indices.numel(),
    )
