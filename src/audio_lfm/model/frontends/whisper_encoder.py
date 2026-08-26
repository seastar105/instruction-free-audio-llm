from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VariableLengthWhisperEncoder(nn.Module):
    """Pinned Transformers 5.15.1 Whisper encoder with variable Mel lengths."""

    def __init__(self, config: object) -> None:
        super().__init__()
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        encoder = WhisperEncoder(config)
        self._take_modules(encoder)

    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> VariableLengthWhisperEncoder:
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance._take_modules(encoder)
        return instance

    def _take_modules(self, encoder: nn.Module) -> None:
        self.conv1 = encoder.conv1
        self.conv2 = encoder.conv2
        self.embed_positions = encoder.embed_positions
        self.layers = encoder.layers
        self.layer_norm = encoder.layer_norm

    def forward(
        self, input_features: torch.Tensor, mel_lengths: torch.Tensor
    ) -> list[torch.Tensor]:
        if input_features.ndim != 3:
            raise ValueError("Whisper features must be [batch, mel_bins, time]")
        if mel_lengths.shape != (input_features.shape[0],):
            raise ValueError("Whisper Mel lengths must contain one value per item")
        if bool(torch.any(mel_lengths <= 0)):
            raise ValueError("Whisper Mel lengths must be positive")
        if bool(torch.any(mel_lengths > input_features.shape[-1])):
            raise ValueError("Whisper Mel length exceeds the padded feature width")

        def forward_bucket(bucket: torch.Tensor) -> torch.Tensor:
            bucket = F.gelu(self.conv1(bucket))
            bucket = F.gelu(self.conv2(bucket)).transpose(1, 2)
            time = bucket.shape[1]
            if time > self.embed_positions.weight.shape[0]:
                raise ValueError("Whisper input exceeds learned positional embeddings")
            bucket = bucket + self.embed_positions.weight[:time]
            for layer in self.layers:
                # Transformers 5.15.1 WhisperEncoderLayer returns the tensor
                # directly. Running without a padding mask also matches the
                # official Whisper attention path.
                bucket = layer(bucket, attention_mask=None)
            return self.layer_norm(bucket)

        # Bucket before convolution as well as attention. Conv layers have
        # biases, so merely zeroing a padded tail is not equivalent to ending
        # the tensor at the effective boundary.
        outputs: list[torch.Tensor | None] = [None] * input_features.shape[0]
        for length_tensor in torch.unique(mel_lengths, sorted=True):
            length = int(length_tensor.item())
            indices = torch.nonzero(
                mel_lengths == length_tensor, as_tuple=False
            ).flatten()
            bucket = forward_bucket(
                input_features.index_select(0, indices)[..., :length]
            )
            for bucket_index, original_index in enumerate(indices.tolist()):
                outputs[original_index] = bucket[bucket_index].detach()
        if any(output is None for output in outputs):
            raise RuntimeError("Whisper length bucketing lost an output")
        return [output for output in outputs if output is not None]
