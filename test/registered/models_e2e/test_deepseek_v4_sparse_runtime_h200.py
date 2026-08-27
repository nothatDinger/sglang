"""DeepSeek-V4 standalone ScoutAttention/InfiniGen E2E coverage on H200."""

import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="8-gpu-h200")

MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
MODEL_LOADER_CONFIG = '{"enable_multithread_load": true, "num_threads": 64}'
SERVER_LAUNCH_TIMEOUT = 3600
SERVER_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
}


class _DSV4StandaloneSparseRuntimeBase(unittest.TestCase):
    dsv4_prefetch_mode = None

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        sparse_config = (
            '{"top_k":512,"device_buffer_size":4096,'
            '"host_to_device_ratio":2,'
            f'"dsv4_prefetch_mode":"{cls.dsv4_prefetch_mode}"}}'
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--page-size",
                "256",
                "--max-running-requests",
                "16",
                "--mem-fraction-static",
                "0.9",
                "--disable-radix-cache",
                "--disable-decode-cuda-graph",
                "--model-loader-extra-config",
                MODEL_LOADER_CONFIG,
                "--hisparse-config",
                sparse_config,
                "--watchdog-timeout",
                "900",
            ],
            env=SERVER_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _generate(self, text):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        self.assertTrue(payload.get("text"))
        self.assertNotIn("nan", payload["text"].lower())
        return payload

    def test_concurrent_short_and_long_context_decode(self):
        # The repeated four-token phrase exceeds the 4096-token device buffer,
        # exercising host misses while another request decodes concurrently.
        prompts = [
            "The capital city of France is",
            "alpha beta gamma delta " * 2048,
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(self._generate, prompts))
        self.assertEqual(len(results), 2)
        self.assertIn("Paris", results[0]["text"])


class TestDSV4StandaloneScoutAttention(_DSV4StandaloneSparseRuntimeBase):
    dsv4_prefetch_mode = "scout"


class TestDSV4StandaloneInfiniGen(_DSV4StandaloneSparseRuntimeBase):
    dsv4_prefetch_mode = "infinigen"


if __name__ == "__main__":
    unittest.main()
