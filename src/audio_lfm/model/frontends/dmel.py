from __future__ import annotations

import torch

from audio_lfm.model.frontends.base import AudioFrontend


class DmelFrontend(AudioFrontend):
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_stride_ms: float = 12.5,
        frame_size_ms: int = 50,
        n_fft: int = 1024,
        n_bits: int = 4,
        num_mel_channels: int = 80,
    ) -> None:
        super().__init__()
        if sample_rate != 16_000:
            raise ValueError("dMel baseline requires 16 kHz")
        try:
            import dmel
        except ImportError as error:
            raise RuntimeError(
                "Install the optional dmel dependency with uv"
            ) from error
        log_mel = dmel.LogMelFbank(
            sampling_freq=sample_rate,
            n_fft=n_fft,
            frame_size_ms=frame_size_ms,
            frame_stride_ms=frame_stride_ms,
            n_filterbank=num_mel_channels,
        )
        self.extractor = dmel.DiscretizedLogMelFbank(
            logmelfbank=log_mel,
            n_bits=n_bits,
            quantize_min_value=-7,
            quantize_max_value=2,
        )
        self.extractor.eval().requires_grad_(False)
        self.output_dim = num_mel_channels
        self.num_channels = num_mel_channels
        self.num_bins = 2**n_bits
        self.hop_length = round(sample_rate * frame_stride_ms / 1000)

    def estimate_output_lengths(self, num_samples: torch.Tensor) -> torch.Tensor:
        return torch.div(num_samples, self.hop_length, rounding_mode="floor") + 1

    def encode(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        with torch.no_grad():
            outputs: list[torch.Tensor] = []
            for waveform in waveforms:
                length = torch.tensor([waveform.numel()], dtype=torch.long)
                codes, lengths = self.extractor(waveform.float()[None], length)
                valid_length = int(lengths[0].item())
                # The official package adds its own BOS/EOS frames. This model
                # uses trainable continuous boundary vectors instead.
                output = codes[0, 1 : valid_length - 1].detach()
                expected = int(self.estimate_output_lengths(length)[0].item())
                if output.shape != (expected, self.num_channels):
                    raise RuntimeError(
                        "Official dMel output-length contract changed: "
                        f"expected {(expected, self.num_channels)}, got {output.shape}"
                    )
                if output.numel() and int(output.max()) >= self.num_bins:
                    raise RuntimeError(
                        "dMel frame codes include an unexpected special ID"
                    )
                outputs.append(output)
        return outputs
