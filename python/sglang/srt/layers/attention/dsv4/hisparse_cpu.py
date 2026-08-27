"""CPU partial attention for DeepSeek-V4 HiSparse host misses.

This is the correctness/reference backend.  Its matrix multiplications go
through PyTorch's CPU dispatcher (oneDNN/MKL where available), which selects
AVX2/AVX-512/AMX kernels for the host CPU at runtime.  The byte gather and FP8
E4M3FN + UE8M0 dequantization are kept explicit so the page-padded host layout
is identical to FlashMLA's C4 layout.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import torch


_NOPE_DIM = 448
_ROPE_DIM = 64
_DIM = _NOPE_DIM + _ROPE_DIM
_DATA_BYTES = 576
_SCALE_BYTES = 8
_PAGE_SIZE = 64
_NUM_TILES = 7
_TILE_SIZE = 64


class CpuAttentionTiming(NamedTuple):
    total_ms: float
    dequant_ms: float
    qk_softmax_ms: float
    pv_ms: float
    miss_tokens: int


def _fp8_e4m3fn_to_float(raw: torch.Tensor) -> torch.Tensor:
    bits = raw.to(torch.int32)
    sign = torch.where((bits & 0x80) != 0, -1.0, 1.0)
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    normal = torch.ldexp(
        1.0 + mantissa.to(torch.float32) / 8.0,
        exponent - 7,
    )
    subnormal = mantissa.to(torch.float32) * (2.0**-9)
    value = torch.where(exponent == 0, subnormal, normal)
    # E4M3FN reserves only the two sign variants of 0x7f as NaN.
    value = torch.where(
        (exponent == 0x0F) & (mantissa == 0x07),
        torch.full_like(value, float("nan")),
        value,
    )
    return value * sign


def _ue8m0_to_float(raw: torch.Tensor) -> torch.Tensor:
    exponent = raw.to(torch.int32)
    value = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent - 127)
    return torch.where(
        exponent == 0xFF,
        torch.full_like(value, float("nan")),
        value,
    )


def gather_and_dequant_host_c4(
    host_cache: torch.Tensor,
    token_locs: torch.Tensor,
) -> torch.Tensor:
    """Gather token-level C4 rows from the page-padded pinned host buffer."""
    if host_cache.device.type != "cpu" or host_cache.dtype != torch.uint8:
        raise ValueError("DeepSeek-V4 host C4 cache must be a CPU uint8 tensor")
    token_locs = token_locs.to(device="cpu", dtype=torch.int64).reshape(-1)
    if token_locs.numel() == 0:
        return torch.empty((0, _DIM), dtype=torch.float32)

    page_bytes = host_cache.stride(0)
    pages = torch.div(token_locs, _PAGE_SIZE, rounding_mode="floor")
    offsets = token_locs.remainder(_PAGE_SIZE)
    raw_pages = host_cache.as_strided(
        (host_cache.shape[0], page_bytes), (page_bytes, 1)
    )

    data_base = offsets * _DATA_BYTES
    nope_offsets = data_base[:, None] + torch.arange(_NOPE_DIM, dtype=torch.int64)
    rope_offsets = data_base[:, None] + _NOPE_DIM + torch.arange(
        _ROPE_DIM * 2, dtype=torch.int64
    )
    scale_offsets = (
        _PAGE_SIZE * _DATA_BYTES
        + offsets[:, None] * _SCALE_BYTES
        + torch.arange(_NUM_TILES, dtype=torch.int64)
    )

    nope_bytes = raw_pages[pages[:, None], nope_offsets]
    rope_bytes = raw_pages[pages[:, None], rope_offsets]
    scale_bytes = raw_pages[pages[:, None], scale_offsets]

    nope = _fp8_e4m3fn_to_float(nope_bytes).view(-1, _NUM_TILES, _TILE_SIZE)
    scales = _ue8m0_to_float(scale_bytes).view(-1, _NUM_TILES, 1)
    rope = rope_bytes.contiguous().view(torch.bfloat16).to(torch.float32)
    return torch.cat([(nope * scales).view(-1, _NOPE_DIM), rope], dim=-1)


@torch.no_grad()
def cpu_miss_attention(
    *,
    query: torch.Tensor,
    miss_host_locs: torch.Tensor,
    host_cache: torch.Tensor,
    softmax_scale: float,
    head_dim_v: int,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> CpuAttentionTiming:
    """Compute one exact partial attention over non-resident C4 entries."""
    begin_ns = time.perf_counter_ns()
    dequant_ns = qk_ns = pv_ns = 0
    miss_tokens = 0
    output.zero_()
    lse.fill_(float("-inf"))

    query_f = query.to(torch.float32)
    for batch_idx in range(query.shape[0]):
        valid_locs = miss_host_locs[batch_idx]
        valid_locs = valid_locs[valid_locs >= 0]
        if valid_locs.numel() == 0:
            continue
        miss_tokens += int(valid_locs.numel())

        start_ns = time.perf_counter_ns()
        kv = gather_and_dequant_host_c4(host_cache, valid_locs)
        dequant_ns += time.perf_counter_ns() - start_ns

        start_ns = time.perf_counter_ns()
        scores = torch.matmul(query_f[batch_idx], kv.transpose(0, 1))
        scores.mul_(softmax_scale)
        req_lse = torch.logsumexp(scores, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        qk_ns += time.perf_counter_ns() - start_ns

        start_ns = time.perf_counter_ns()
        req_out = torch.matmul(weights, kv[:, :head_dim_v])
        pv_ns += time.perf_counter_ns() - start_ns
        output[batch_idx].copy_(req_out.to(output.dtype))
        lse[batch_idx].copy_(req_lse)

    total_ns = time.perf_counter_ns() - begin_ns
    ns_to_ms = 1.0e-6
    return CpuAttentionTiming(
        total_ms=total_ns * ns_to_ms,
        dequant_ms=dequant_ns * ns_to_ms,
        qk_softmax_ms=qk_ns * ns_to_ms,
        pv_ms=pv_ns * ns_to_ms,
        miss_tokens=miss_tokens,
    )


def merge_cpu_gpu_attention(
    *,
    gpu_output: torch.Tensor,
    gpu_lse: torch.Tensor,
    cpu_output: torch.Tensor,
    cpu_lse: torch.Tensor,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    """Stable merge of FlashMLA resident output and CPU miss output."""
    gpu_partition_lse = gpu_lse.to(dtype=torch.float32)

    # FlashMLA uses an infinite LSE as the sentinel for an empty KV partition.
    # A normal GPU LSE already includes the sink contribution, but the empty
    # sentinel replaces it, so restore the sink only for those empty rows.
    gpu_empty = torch.isinf(gpu_partition_lse)
    if attn_sink is None:
        empty_gpu_lse = torch.full_like(gpu_partition_lse, float("-inf"))
    else:
        empty_gpu_lse = (
            attn_sink.to(device=gpu_lse.device, dtype=torch.float32)
            .view(1, -1, 1)
            .expand_as(gpu_partition_lse)
        )
    gpu_partition_lse = torch.where(
        gpu_empty, empty_gpu_lse, gpu_partition_lse
    )

    cpu_lse = cpu_lse.to(
        device=gpu_lse.device, dtype=torch.float32, non_blocking=True
    ).unsqueeze(-1)
    joint_lse = torch.logaddexp(gpu_partition_lse, cpu_lse)
    joint_empty = torch.isneginf(joint_lse)
    safe_joint_lse = torch.where(
        joint_empty, torch.zeros_like(joint_lse), joint_lse
    )
    gpu_weight = torch.where(
        joint_empty,
        torch.zeros_like(joint_lse),
        torch.exp(gpu_partition_lse - safe_joint_lse),
    ).transpose(1, 2).unsqueeze(-1)
    cpu_weight = torch.where(
        joint_empty,
        torch.zeros_like(joint_lse),
        torch.exp(cpu_lse - safe_joint_lse),
    ).transpose(1, 2).unsqueeze(-1)
    cpu_output = cpu_output.to(
        device=gpu_output.device, dtype=gpu_output.dtype, non_blocking=True
    )
    if cpu_output.ndim == 3:
        cpu_output = cpu_output.unsqueeze(1)
    return gpu_output * gpu_weight.to(gpu_output.dtype) + cpu_output * cpu_weight.to(
        gpu_output.dtype
    )

