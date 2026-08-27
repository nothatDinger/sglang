import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    BaseSWAKVPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4SparseRuntimePoolSelection(unittest.TestCase):
    @staticmethod
    def _selected_c4_pool(*, runtime_enabled: bool):
        selected_pool_classes = []

        def capture_pool(**kwargs):
            selected_pool_classes.append(
                kwargs.get("cls", DeepSeekV4SingleKVPool)
            )
            return SimpleNamespace()

        with (
            patch.object(BaseSWAKVPool, "__init__", return_value=None),
            patch.object(
                DeepSeekV4TokenToKVPool,
                "_make_kv_pool",
                side_effect=capture_pool,
            ),
            patch.object(
                DeepSeekV4TokenToKVPool,
                "_make_indexer_pool",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                DeepSeekV4TokenToKVPool,
                "_init_compressed_layer_mapping",
            ),
            patch.object(
                DeepSeekV4TokenToKVPool,
                "_init_paged_compress_states",
            ),
            patch.object(
                DeepSeekV4TokenToKVPool,
                "get_ring_size",
                return_value=1,
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels."
                "env_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            DeepSeekV4TokenToKVPool(
                max_num_reqs=1,
                swa_size=256,
                c4_size=64,
                c128_size=2,
                c4_state_pool_size=1,
                c128_state_pool_size=1,
                page_size=256,
                swa_page_size=128,
                dtype=torch.float8_e4m3fn,
                c4_state_dtype=torch.float32,
                c128_state_dtype=torch.float32,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                indexer_head_dim=128,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
                compression_ratios=[4],
                enable_sparse_runtime=runtime_enabled,
            )

        # Calls are SWA, C4 and C128 in order.
        return selected_pool_classes[1]

    def test_baseline_dsv4_keeps_regular_c4_pool(self):
        self.assertIs(
            self._selected_c4_pool(runtime_enabled=False),
            DeepSeekV4SingleKVPool,
        )

    def test_standalone_and_legacy_runtime_share_host_backed_c4_pool(self):
        self.assertIs(
            self._selected_c4_pool(runtime_enabled=True),
            HiSparseC4DevicePool,
        )


if __name__ == "__main__":
    unittest.main()
