"""Benchmark the native and Torch DSV4 host-miss attention backends."""

import argparse
import math
import time

import torch

from sglang.srt.layers.attention.dsv4.hisparse_cpu import (
    cpu_miss_attention,
    cpu_miss_attention_native,
)

PAGE_SIZE = 64
PAGE_BYTES = math.ceil((576 + 8) * PAGE_SIZE / 576) * 576


def run(fn, query, misses, cache, iterations):
    output = torch.empty((*query.shape[:2], 512), dtype=torch.bfloat16)
    lse = torch.empty(query.shape[:2], dtype=torch.float32)
    kwargs = dict(
        query=query,
        miss_host_locs=misses,
        host_cache=cache,
        softmax_scale=512**-0.5,
        head_dim_v=512,
        output=output,
        lse=lse,
    )
    for _ in range(3):
        fn(**kwargs)
    begin = time.perf_counter()
    for _ in range(iterations):
        fn(**kwargs)
    return (time.perf_counter() - begin) * 1e3 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--misses", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512]
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not hasattr(torch.ops.sgl_kernel, "dsv4_hisparse_cpu_attention"):
        raise RuntimeError("native DSV4 CPU attention operator is not loaded")
    for batch in args.batches:
        for heads in args.heads:
            for count in args.misses:
                pages = math.ceil(count / PAGE_SIZE)
                cache = torch.randint(0, 126, (pages, PAGE_BYTES), dtype=torch.uint8)
                for page in range(pages):
                    scale_base = PAGE_SIZE * 576
                    cache[page, scale_base : scale_base + PAGE_SIZE * 8] = 127
                query = torch.randn((batch, heads, 512), dtype=torch.bfloat16)
                misses = (
                    torch.arange(count, dtype=torch.int64)
                    .expand(batch, -1)
                    .contiguous()
                )
                ref = run(cpu_miss_attention, query, misses, cache, args.iterations)
                native = run(
                    cpu_miss_attention_native, query, misses, cache, args.iterations
                )
                print(
                    f"batch={batch:2d} heads={heads:2d} misses={count:3d} "
                    f"torch={ref:8.3f}ms native={native:8.3f}ms speedup={ref / native:6.2f}x"
                )


if __name__ == "__main__":
    main()
