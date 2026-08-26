from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class IsolationResult:
    name: str
    max_forward_difference: float
    max_gradient_difference: float


def run_direct_causal_conv_boundary_test(
    *,
    device: torch.device | None = None,
    tolerance: float = 2e-3,
    conv_function: Callable[..., torch.Tensor] | None = None,
) -> IsolationResult:
    if conv_function is None:
        try:
            from causal_conv1d import causal_conv1d_fn
        except ImportError as error:
            raise RuntimeError("causal-conv1d is not installed") from error
        conv_function = causal_conv1d_fn
    device = device or torch.device("cuda")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(123)
    channels, width, a_length, b_length = 8, 3, 11, 9
    # causal-conv1d accepts ``seq_idx`` only when the logical [B, C, L]
    # tensor has channel-last storage (stride(1) == 1).  Construct through
    # [B, L, C] instead of making a contiguous channel-first tensor so this
    # preflight exercises the same layout used by LFM's convolution blocks.
    a = torch.randn(1, a_length, channels, device=device, dtype=dtype).transpose(1, 2)
    b = torch.randn(1, b_length, channels, device=device, dtype=dtype).transpose(1, 2)
    weight = torch.randn(channels, width, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype)

    def convolve(
        value: torch.Tensor, seq_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"activation": None}
        if seq_idx is not None:
            kwargs["seq_idx"] = seq_idx
        assert conv_function is not None
        return conv_function(value, weight, bias, **kwargs)

    separate = torch.cat([convolve(a), convolve(b)], dim=-1)
    packed_input = torch.cat([a.transpose(1, 2), b.transpose(1, 2)], dim=1).transpose(
        1, 2
    )
    seq_idx = torch.cat(
        [
            torch.zeros(a_length, dtype=torch.int32, device=device),
            torch.ones(b_length, dtype=torch.int32, device=device),
        ]
    ).unsqueeze(0)
    packed = convolve(packed_input, seq_idx)
    forward_difference = float((packed - separate).abs().max().item())
    if forward_difference > tolerance:
        raise RuntimeError(
            f"causal-conv boundary forward isolation failed: {forward_difference}"
        )

    separate_leaf = packed_input.detach().clone().requires_grad_(True)
    separate_output = torch.cat(
        [
            convolve(separate_leaf[..., :a_length]),
            convolve(separate_leaf[..., a_length:]),
        ],
        dim=-1,
    )
    separate_output.square().sum().backward()
    separate_gradient = separate_leaf.grad.detach().clone()
    packed_leaf = packed_input.detach().clone().requires_grad_(True)
    convolve(packed_leaf, seq_idx).square().sum().backward()
    gradient_difference = float(
        (packed_leaf.grad - separate_gradient).abs().max().item()
    )
    if gradient_difference > tolerance:
        raise RuntimeError(
            f"causal-conv boundary backward isolation failed: {gradient_difference}"
        )
    perturbed = packed_input.clone()
    perturbed[..., :a_length] += 7
    b_difference = float(
        (convolve(perturbed, seq_idx)[..., a_length:] - packed[..., a_length:])
        .abs()
        .max()
        .item()
    )
    if b_difference > tolerance:
        raise RuntimeError(f"Sequence A perturbs sequence B: {b_difference}")
    return IsolationResult(
        name="causal_conv1d",
        max_forward_difference=forward_difference,
        max_gradient_difference=gradient_difference,
    )


def run_lfm_packing_isolation_test(
    llm: torch.nn.Module,
    *,
    device: torch.device,
    tolerance: float = 3e-2,
) -> IsolationResult:
    hidden_size = int(llm.config.hidden_size)
    a_length, b_length = 7, 6
    a = torch.randn(1, a_length, hidden_size, device=device, dtype=torch.bfloat16)
    b = torch.randn(1, b_length, hidden_size, device=device, dtype=torch.bfloat16)

    def call(
        value: torch.Tensor,
        lengths: list[int],
    ) -> torch.Tensor:
        positions = torch.cat(
            [torch.arange(length, device=device) for length in lengths]
        ).unsqueeze(0)
        seq_idx = torch.cat(
            [
                torch.full((length,), index, device=device, dtype=torch.int32)
                for index, length in enumerate(lengths)
            ]
        ).unsqueeze(0)
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        )
        return llm.model(
            inputs_embeds=value,
            attention_mask=None,
            position_ids=positions,
            seq_idx=seq_idx,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    with torch.no_grad():
        separate = torch.cat([call(a, [a_length]), call(b, [b_length])], dim=1)
        packed = call(torch.cat([a, b], dim=1), [a_length, b_length])
    forward_difference = float((packed - separate).abs().max().item())
    if forward_difference > tolerance:
        raise RuntimeError(f"LFM packed-forward isolation failed: {forward_difference}")
    a_leaf = a.detach().clone().requires_grad_(True)
    b_leaf = b.detach().clone().requires_grad_(True)
    output = call(torch.cat([a_leaf, b_leaf], dim=1), [a_length, b_length])
    output[:, a_length : a_length + 2].float().square().sum().backward()
    a_gradient = float(a_leaf.grad.abs().max().item())
    b_gradient = float(b_leaf.grad.abs().max().item())
    if a_gradient > tolerance or b_gradient == 0:
        raise RuntimeError(
            f"LFM backward isolation failed: A={a_gradient}, B={b_gradient}"
        )
    return IsolationResult(
        name="lfm2",
        max_forward_difference=forward_difference,
        max_gradient_difference=a_gradient,
    )
