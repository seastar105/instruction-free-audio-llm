from __future__ import annotations

from collections.abc import Callable

import torch

from audio_lfm.model.frontends.base import AudioFrontend
from audio_lfm.model.frontends.whisper_encoder import VariableLengthWhisperEncoder
from audio_lfm.model.frontends.whisper_math import (
    chunked_whisper_encoder_lengths,
    slice_valid_outputs,
    waveform_to_mel_lengths,
    whisper_encoder_lengths,
)


class WhisperFrontend(AudioFrontend):
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: torch.device,
        max_seconds: float = 30.0,
        mode: str = "variable_length_masked",
        chunk_long_audio: bool = False,
        encoder_microbatch_max_padded_samples: int = 960_000,
    ) -> None:
        super().__init__()
        from transformers import WhisperModel

        model = WhisperModel.from_pretrained(model_id, revision=revision)
        official_encoder = model.encoder
        if mode == "variable_length_masked":
            self.encoder = VariableLengthWhisperEncoder.from_encoder(official_encoder)
        elif mode == "official_fixed_30s":
            self.encoder = official_encoder
        else:
            raise ValueError(f"Unknown Whisper frontend mode: {mode}")
        self.encoder.eval().requires_grad_(False)
        self.encoder.to(
            device=device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        self.output_dim = int(model.config.d_model)
        self.device = device
        self.max_samples = int(max_seconds * 16_000)
        self.mode = mode
        self.chunk_long_audio = chunk_long_audio
        self.hop_length = 160
        self.encoder_microbatch_max_padded_samples = (
            encoder_microbatch_max_padded_samples
        )
        self._compiled_encoder_forward: (
            Callable[[torch.Tensor], torch.Tensor] | None
        ) = None
        del model

    def estimate_output_lengths(self, num_samples: torch.Tensor) -> torch.Tensor:
        if self.chunk_long_audio:
            return chunked_whisper_encoder_lengths(
                num_samples,
                chunk_samples=self.max_samples,
                hop_length=self.hop_length,
            )
        mel = waveform_to_mel_lengths(num_samples, hop_length=self.hop_length)
        return whisper_encoder_lengths(mel)

    def enable_torch_compile(
        self,
        *,
        backend: str,
        mode: str,
        dynamic: bool,
    ) -> None:
        if self.mode != "official_fixed_30s":
            raise ValueError(
                "Whisper compilation requires official_fixed_30s for fixed time shapes"
            )
        self._compiled_encoder_forward = torch.compile(
            self._official_encoder_forward,
            backend=backend,
            mode=mode,
            # The time axis is always exactly 30 seconds. The batch axis is
            # padded to the configured block microbatch below, so keeping this
            # graph static avoids Inductor re-specialization/CantSplit failures
            # as a pack's final microbatch changes size.
            dynamic=False,
            fullgraph=False,
        )

    def _official_encoder_forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features).last_hidden_state

    def encode_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """Encode fixed 30-second blocks without per-audio slicing."""
        if blocks.ndim != 3 or blocks.shape[-1] != 3000:
            raise ValueError("Whisper blocks must be [blocks, mel_bins, 3000]")
        blocks_per_microbatch = max(
            1, self.encoder_microbatch_max_padded_samples // self.max_samples
        )
        encoder_forward = (
            self._compiled_encoder_forward or self._official_encoder_forward
        )
        encoded_chunks: list[torch.Tensor] = []
        for start in range(0, blocks.shape[0], blocks_per_microbatch):
            stop = min(blocks.shape[0], start + blocks_per_microbatch)
            real_count = stop - start
            encoder_input = blocks[start:stop]
            if (
                self._compiled_encoder_forward is not None
                and real_count < blocks_per_microbatch
            ):
                encoder_input = torch.cat(
                    [
                        encoder_input,
                        torch.zeros(
                            (
                                blocks_per_microbatch - real_count,
                                *encoder_input.shape[1:],
                            ),
                            device=encoder_input.device,
                            dtype=encoder_input.dtype,
                        ),
                    ],
                    dim=0,
                )
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ),
            ):
                hidden = encoder_forward(encoder_input)[:real_count]
            encoded_chunks.append(hidden)
        return torch.cat(encoded_chunks, dim=0)

    def encode_precomputed(
        self,
        input_features: list[torch.Tensor],
        effective_encoder_lengths: list[tuple[int, ...]],
    ) -> list[torch.Tensor]:
        if not input_features:
            return []
        if len(input_features) != len(effective_encoder_lengths):
            raise ValueError("Feature and effective-length example counts differ")
        block_counts = [value.shape[0] for value in input_features]
        if any(
            count != len(lengths)
            for count, lengths in zip(
                block_counts, effective_encoder_lengths, strict=True
            )
        ):
            raise ValueError("Feature block count differs from effective lengths")
        blocks = torch.cat(input_features, dim=0)
        lengths = torch.tensor(
            [length for values in effective_encoder_lengths for length in values],
            device=blocks.device,
            dtype=torch.long,
        )
        hidden = self.encode_blocks(blocks)
        encoded_chunks = slice_valid_outputs(hidden, lengths)
        outputs: list[torch.Tensor] = []
        offset = 0
        for count in block_counts:
            outputs.append(torch.cat(encoded_chunks[offset : offset + count], dim=0))
            offset += count
        if any(output.requires_grad for output in outputs):
            raise RuntimeError("Frozen Whisper outputs must be detached")
        return outputs
