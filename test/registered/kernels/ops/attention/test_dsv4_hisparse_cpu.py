import math

import torch

from sglang.srt.layers.attention.dsv4.hisparse_cpu import (
    _fp8_e4m3fn_to_float,
    cpu_miss_attention,
    gather_and_dequant_host_c4,
    merge_cpu_gpu_attention,
)


_PAGE_SIZE = 64
_DATA_BYTES = 576
_SCALE_BYTES = 8
_PAGE_BYTES = math.ceil((_DATA_BYTES + _SCALE_BYTES) * _PAGE_SIZE / 576) * 576


def _write_unit_c4_token(host_cache: torch.Tensor, token: int) -> None:
    page, offset = divmod(token, _PAGE_SIZE)
    data_start = offset * _DATA_BYTES
    host_cache[page, data_start : data_start + 448] = 0x38  # E4M3FN 1.0
    rope = torch.full((64,), 2.0, dtype=torch.bfloat16).view(torch.uint8)
    host_cache[page, data_start + 448 : data_start + 576] = rope
    scale_start = _PAGE_SIZE * _DATA_BYTES + offset * _SCALE_BYTES
    host_cache[page, scale_start : scale_start + 7] = 127  # UE8M0 1.0


def test_fp8_e4m3fn_reference_decoder():
    raw = torch.tensor(
        [0x00, 0x01, 0x38, 0x40, 0x7E, 0xB8], dtype=torch.uint8
    )
    expected = torch.tensor([0.0, 2**-9, 1.0, 2.0, 448.0, -1.0])
    torch.testing.assert_close(_fp8_e4m3fn_to_float(raw), expected)


def test_gather_and_dequant_host_c4_page_layout():
    host_cache = torch.zeros((1, _PAGE_BYTES), dtype=torch.uint8)
    _write_unit_c4_token(host_cache, 0)
    _write_unit_c4_token(host_cache, 63)

    result = gather_and_dequant_host_c4(
        host_cache, torch.tensor([0, 63], dtype=torch.int64)
    )
    torch.testing.assert_close(result[:, :448], torch.ones((2, 448)))
    torch.testing.assert_close(result[:, 448:], torch.full((2, 64), 2.0))


def test_cpu_miss_attention_returns_partial_output_and_natural_lse():
    host_cache = torch.zeros((1, _PAGE_BYTES), dtype=torch.uint8)
    _write_unit_c4_token(host_cache, 0)
    query = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
    miss_locs = torch.tensor([[0, -1]], dtype=torch.int64)
    output = torch.empty((1, 2, 512), dtype=torch.bfloat16)
    lse = torch.empty((1, 2), dtype=torch.float32)

    timing = cpu_miss_attention(
        query=query,
        miss_host_locs=miss_locs,
        host_cache=host_cache,
        softmax_scale=512**-0.5,
        head_dim_v=512,
        output=output,
        lse=lse,
    )

    expected = torch.cat([torch.ones(448), torch.full((64,), 2.0)])
    torch.testing.assert_close(output[0, 0].float(), expected)
    torch.testing.assert_close(output[0, 1].float(), expected)
    torch.testing.assert_close(lse, torch.zeros_like(lse))
    assert timing.miss_tokens == 1


def test_merge_cpu_gpu_attention_uses_disjoint_partition_lse():
    gpu_output = torch.ones((1, 1, 2, 4), dtype=torch.bfloat16)
    cpu_output = torch.full((1, 2, 4), 3.0, dtype=torch.bfloat16)
    gpu_lse = torch.zeros((1, 2, 1), dtype=torch.float32)
    cpu_lse = torch.zeros((1, 2), dtype=torch.float32)

    merged = merge_cpu_gpu_attention(
        gpu_output=gpu_output,
        gpu_lse=gpu_lse,
        cpu_output=cpu_output,
        cpu_lse=cpu_lse,
        attn_sink=torch.zeros(2),
    )
    torch.testing.assert_close(merged.float(), torch.full_like(merged.float(), 2.0))


def test_merge_does_not_add_sink_twice_to_flashmla_lse():
    gpu_output = torch.ones((1, 1, 1, 2), dtype=torch.bfloat16)
    cpu_output = torch.full((1, 1, 2), 3.0, dtype=torch.bfloat16)
    # The GPU partition has total denominator 2, already including its sink;
    # the CPU miss partition has denominator 1.
    gpu_lse = torch.full((1, 1, 1), math.log(2.0), dtype=torch.float32)
    cpu_lse = torch.zeros((1, 1), dtype=torch.float32)

    merged = merge_cpu_gpu_attention(
        gpu_output=gpu_output,
        gpu_lse=gpu_lse,
        cpu_output=cpu_output,
        cpu_lse=cpu_lse,
        attn_sink=torch.full((1,), 100.0),
    )
    torch.testing.assert_close(
        merged.float(), torch.full_like(merged.float(), 5.0 / 3.0), atol=0.01, rtol=0
    )

