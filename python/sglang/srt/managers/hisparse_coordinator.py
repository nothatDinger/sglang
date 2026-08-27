# to be combined with the sparse coordinator class and sparse algorithm family

import logging
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    classify_cache_residency_mla,
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


def resolve_shared_index_layers(
    *,
    hf_text_config,
    pp_size: int,
    is_speculative: bool,
) -> Optional[List[bool]]:
    """Per-layer "reuses the previous layer's top-k index" pattern, or None.

    Mirrors DeepseekV2AttentionMLA's skip_topk derivation (index_topk_pattern /
    index_topk_freq / cli_factor); None when the model has no sharing or the
    prefetch cannot run (PP, speculative decoding, kill-switch).
    """
    if not is_deepseek_dsa(hf_text_config):
        return None
    num_layers = hf_text_config.num_hidden_layers
    cli_factor = getattr(hf_text_config, "cli_factor", 1) or 1
    if cli_factor > 1:
        pattern = [i % cli_factor != 0 for i in range(num_layers)]
    else:
        pattern = [dsa_layer_skips_topk(hf_text_config, i) for i in range(num_layers)]
    if not any(pattern):
        return None
    if pp_size != 1 or is_speculative:
        logger.warning(
            "HiSparse shared-index prefetch is unsupported under pipeline "
            "parallelism / speculative decoding; falling back to synchronous "
            "swap-in."
        )
        return None
    if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        logger.info(
            "HiSparse shared-index prefetch disabled via "
            "SGLANG_DISABLE_HISPARSE_PREFETCH; using synchronous swap-in."
        )
        return None
    return pattern


def _build_prefetch_groups(
    is_shared_index_layer: List[bool],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group consecutive shared-index (skip) layers under their anchor layer.

    Returns (groups, slot): anchor layer_id -> ordered skip layers, and each
    skip layer's position in its group (indexes the per-slot prefetch events).
    """
    groups: Dict[int, List[int]] = {}
    slot = [0] * len(is_shared_index_layer)
    anchor = None
    for i, is_shared in enumerate(is_shared_index_layer):
        if not is_shared:
            anchor = i  # compute layer; anchors the skip layers after it
            continue
        assert anchor is not None, (
            f"shared-index (skip) layer {i} has no preceding compute layer; "
            "the model's index-topk pattern is invalid"
        )
        group = groups.setdefault(anchor, [])
        slot[i] = len(group)
        group.append(i)
    return groups, slot


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        shared_index_layers: Optional[List[bool]] = None,
        dsv4_prefetch_mode: str = "cpu",
        dsv4_recall_interval: int = 8,
        dsv4_cpu_attention_backend: str = "auto",
        dsv4_cpu_threads: int = 0,
        dsv4_profile: bool = False,
        dsv4_profile_log_interval: int = 100,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        # Timing probe: skip the host->device KV bytes to measure the "IO is
        # free" floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        self.dsv4_prefetch_mode = (
            dsv4_prefetch_mode if self.is_dsv4_hisparse else None
        )
        self.dsv4_recall_interval = dsv4_recall_interval
        self.dsv4_cpu_attention_backend = dsv4_cpu_attention_backend
        self.dsv4_cpu_threads = dsv4_cpu_threads
        self.dsv4_profile = dsv4_profile
        self.dsv4_profile_log_interval = dsv4_profile_log_interval
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
            # C4 stores 576 data bytes plus 8 UE8M0 scale bytes per token.
            # The swap kernel ignores item_size_bytes for the page-padded DSV4
            # layout, but profiling needs the real transfer payload.
            self.dsv4_item_size_bytes = self.item_size_bytes + 8
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

        self._init_shared_index_prefetch(
            shared_index_layers=shared_index_layers,
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
        )
        self._init_dsv4_prefetch(
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
        )

    def _init_shared_index_prefetch(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        """Set up the plan-then-IO prefetch for shared-index (IndexShare) models:
        the anchor's kernel records its miss plan and skip layers replay it on
        `prefetch_stream`, overlapping their IO with the intervening compute."""
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            # Attention-layer count differs from num_hidden_layers (e.g. Longcat
            # doubles it): pattern would be misindexed, fall back to synchronous.
            logger.warning(
                "HiSparse shared-index prefetch disabled: pattern length %d != "
                "KV pool layer_num %d; using synchronous swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = 4
        max_group_size = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = device_module.Stream()
        self._prefetch_events = [device_module.Event() for _ in range(max_group_size)]
        # Plan recorded by the current anchor, replayed by its skip layers. One
        # buffer set suffices: the last skip layer's event wait orders the next
        # anchor's writes after this group's copies.
        self._miss_src = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int64, device=self.device
        )
        self._miss_dst = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int32, device=self.device
        )
        self._miss_count = torch.zeros(
            (max_num_req_slots,), dtype=torch.int32, device=self.device
        )
        logger.info(
            "HiSparse: shared-index prefetch (plan-then-IO) enabled; %d anchor "
            "group(s), %d skip layer(s) of %d total.",
            len(self._prefetch_groups),
            sum(self._is_shared_index_layer),
            layer_num,
        )

    def _init_dsv4_prefetch(
        self,
        *,
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        self._dsv4_cpu_executor = None
        self._dsv4_registered_layers = {}
        self._dsv4_next_csa_layer = {}
        self._dsv4_compressed_to_physical = {}
        self._dsv4_prediction_physical_layer = None
        self._dsv4_prediction_compressed_layer = None
        self._dsv4_decode_step = 0
        self._dsv4_periodic_due = False
        self._dsv4_prediction_allowed = None
        self._dsv4_num_real_reqs_cpu = 0
        self._dsv4_profile_data = defaultdict(lambda: defaultdict(list))
        if not self.is_dsv4_hisparse:
            return

        self.dsv4_prefetch_stream = device_module.Stream()
        self.dsv4_d2h_stream = device_module.Stream()
        self.dsv4_periodic_stream = device_module.Stream()
        event_kwargs = {"enable_timing": True} if self.dsv4_profile else {}
        self._dsv4_ready_events = [
            device_module.Event(**event_kwargs) for _ in range(layer_num)
        ]
        self._dsv4_prefetch_start_events = [
            device_module.Event(**event_kwargs) for _ in range(layer_num)
        ]
        self._dsv4_index_end_events = [
            device_module.Event(**event_kwargs) for _ in range(layer_num)
        ]
        self._dsv4_periodic_start_events = [
            device_module.Event(**event_kwargs) for _ in range(layer_num)
        ]
        self._dsv4_periodic_end_events = [
            device_module.Event(**event_kwargs) for _ in range(layer_num)
        ]
        self._dsv4_ready = [False] * layer_num
        self._dsv4_periodic_pending = [False] * layer_num
        self._dsv4_batch_num_reqs_cpu = [0] * layer_num
        self._dsv4_periodic_profile_step = [None] * layer_num
        self._dsv4_periodic_profile_num_reqs = [0] * layer_num
        self._dsv4_periodic_profile_by_step = defaultdict(dict)

        shape = (layer_num, max_num_req_slots, self.top_k)
        self.dsv4_predicted_raw_indices = torch.full(
            shape, -1, dtype=torch.int32, device=self.device
        )
        self.dsv4_predicted_page_indices = torch.full(
            shape, -1, dtype=torch.int32, device=self.device
        )
        self.dsv4_predicted_device_locs = torch.full(
            shape, -1, dtype=torch.int32, device=self.device
        )
        self.dsv4_batch_req_indices = torch.zeros(
            (layer_num, max_num_req_slots),
            dtype=torch.int64,
            device=self.device,
        )
        self.dsv4_batch_seq_lens = torch.zeros(
            (layer_num, max_num_req_slots),
            dtype=torch.int64,
            device=self.device,
        )
        self.dsv4_batch_num_reqs = torch.zeros(
            (layer_num, 1), dtype=torch.int32, device=self.device
        )
        self.dsv4_miss_host_locs: Optional[torch.Tensor] = None
        self.dsv4_miss_count: Optional[torch.Tensor] = None
        self._dsv4_cpu_miss_locs: List[torch.Tensor] = []
        self._dsv4_periodic_miss_count_cpu: List[torch.Tensor] = []
        if self.dsv4_prefetch_mode == "cpu":
            self.dsv4_miss_host_locs = torch.full(
                shape, -1, dtype=torch.int64, device=self.device
            )
            self.dsv4_miss_count = torch.zeros(
                (layer_num, max_num_req_slots),
                dtype=torch.int32,
                device=self.device,
            )
            self._dsv4_cpu_miss_locs = [
                torch.empty(
                    (max_num_req_slots, self.top_k),
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(layer_num)
            ]
            if self.dsv4_profile:
                self._dsv4_periodic_miss_count_cpu = [
                    torch.empty(
                        max_num_req_slots,
                        dtype=torch.int32,
                        device="cpu",
                        pin_memory=True,
                    )
                    for _ in range(layer_num)
                ]
        self._dsv4_cpu_jobs: Dict[
            int, Tuple[Future, torch.Tensor, torch.Tensor, int]
        ] = {}
        self._dsv4_active_cpu_layers: Dict[int, int] = {}

        if self.dsv4_prefetch_mode == "cpu":
            if self.dsv4_cpu_threads > 0:
                torch.set_num_threads(self.dsv4_cpu_threads)
            self._dsv4_cpu_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="sglang-dsv4-attn"
            )
            cpu_capability = (
                torch.backends.cpu.get_cpu_capability()
                if hasattr(torch.backends.cpu, "get_cpu_capability")
                else "unknown"
            )
            logger.info(
                "DeepSeek-V4 HiSparse CSA prefetch enabled: mode=cpu, "
                "recall_interval=%d, cpu_backend=%s, cpu_capability=%s, "
                "torch_threads=%d",
                self.dsv4_recall_interval,
                self.dsv4_cpu_attention_backend,
                cpu_capability,
                torch.get_num_threads(),
            )
        else:
            logger.info(
                "DeepSeek-V4 HiSparse CSA prefetch enabled: mode=h2d."
            )

    def register_dsv4_csa_layers(self, layers) -> None:
        """Register local CSA modules and derive the next-CSA relation."""
        if not self.is_dsv4_hisparse or self._dsv4_registered_layers:
            return
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if (
                attn is not None
                and getattr(attn, "compress_ratio", None) == 4
                and getattr(attn, "indexer", None) is not None
            ):
                self._dsv4_registered_layers[attn.layer_id] = attn
                self._dsv4_compressed_to_physical[
                    self._dsv4_compressed_layer_id(attn.layer_id)
                ] = attn.layer_id
        csa_layers = sorted(self._dsv4_registered_layers)
        self._dsv4_next_csa_layer = dict(zip(csa_layers, csa_layers[1:]))
        logger.info(
            "DeepSeek-V4 HiSparse registered CSA layers=%s; next-CSA prefetch "
            "uses each source CSA layer's normalized input hidden state.",
            csa_layers,
        )

    def begin_decode_step(self, *, num_real_reqs: int) -> None:
        if not self.is_dsv4_hisparse:
            return
        if num_real_reqs < 0:
            raise ValueError(
                "DeepSeek-V4 HiSparse real request count must be non-negative, "
                f"got {num_real_reqs}."
            )
        self._dsv4_num_real_reqs_cpu = int(num_real_reqs)
        self._dsv4_decode_step += 1
        self._dsv4_prediction_allowed = None
        self._dsv4_periodic_due = (
            self.dsv4_prefetch_mode == "cpu"
            and self.dsv4_recall_interval > 0
            and self._dsv4_decode_step % self.dsv4_recall_interval == 0
        )
        if (
            self.dsv4_profile
            and self._dsv4_decode_step % self.dsv4_profile_log_interval == 0
        ):
            self._log_dsv4_profile()

    def _dsv4_compressed_layer_id(self, physical_layer_id: int) -> int:
        return self.token_to_kv_pool_allocator._kvcache.layer_mapping[
            physical_layer_id
        ].compress_layer_id

    def is_dsv4_prediction(self, physical_layer_id: int) -> bool:
        return (
            self.is_dsv4_hisparse
            and self._dsv4_prediction_physical_layer == physical_layer_id
        )

    def dsv4_prediction_buffers(
        self, physical_layer_id: int, num_queries: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        compressed_layer = self._dsv4_compressed_layer_id(physical_layer_id)
        assert compressed_layer == self._dsv4_prediction_compressed_layer
        return (
            self.dsv4_predicted_raw_indices[compressed_layer, :num_queries],
            self.dsv4_predicted_page_indices[compressed_layer, :num_queries],
        )

    def launch_dsv4_next_csa_prefetch(
        self,
        *,
        source_layer_id: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        attn_backend,
    ) -> None:
        if (
            not self.is_dsv4_hisparse
            or not forward_batch.forward_mode.is_decode()
            or source_layer_id not in self._dsv4_next_csa_layer
        ):
            return

        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if self._dsv4_prediction_allowed is None:
            if seq_lens_cpu is None:
                raise RuntimeError(
                    "DeepSeek-V4 HiSparse CSA prefetch requires the decode "
                    "seq_lens_cpu mirror."
                )
            num_real_reqs = self._dsv4_num_real_reqs_cpu
            if num_real_reqs > len(seq_lens_cpu):
                raise RuntimeError(
                    "DeepSeek-V4 HiSparse real request count exceeds the "
                    "padded sequence-length buffer: "
                    f"real={num_real_reqs}, padded={len(seq_lens_cpu)}."
                )
            self._dsv4_prediction_allowed = num_real_reqs > 0 and all(
                int(seq_len) > self.compress_ratio
                for seq_len in seq_lens_cpu[:num_real_reqs]
            )
        if not self._dsv4_prediction_allowed:
            # No committed target-layer C4 index exists yet. Let the target CSA
            # take the same-layer fallback; all of these tokens are resident/SWA.
            return

        target_layer_id = self._dsv4_next_csa_layer[source_layer_id]
        target_attn = self._dsv4_registered_layers[target_layer_id]
        compressed_layer = self._dsv4_compressed_layer_id(target_layer_id)
        if self._dsv4_ready[compressed_layer]:
            raise RuntimeError(
                "DeepSeek-V4 HiSparse predicted index was not consumed before "
                f"the next decode step (layer={target_layer_id})."
            )

        current_stream = device_module.current_stream()
        self.dsv4_prefetch_stream.wait_stream(current_stream)
        with device_module.stream(self.dsv4_prefetch_stream):
            self._wait_dsv4_periodic(compressed_layer)
            if self.dsv4_profile:
                self._dsv4_prefetch_start_events[compressed_layer].record(
                    self.dsv4_prefetch_stream
                )
            self._dsv4_prediction_physical_layer = target_layer_id
            self._dsv4_prediction_compressed_layer = compressed_layer
            try:
                with torch.profiler.record_function(
                    f"dsv4_hisparse/predict_index/layer_{target_layer_id}"
                ):
                    q_lora = target_attn._compute_q_a(x)
                    target_attn.indexer(
                        x=x,
                        q_lora=q_lora,
                        forward_batch=forward_batch,
                        attn_backend=attn_backend,
                        skip_compressor=True,
                    )
            finally:
                self._dsv4_prediction_physical_layer = None
                self._dsv4_prediction_compressed_layer = None
        x.record_stream(self.dsv4_prefetch_stream)
        positions.record_stream(self.dsv4_prefetch_stream)

    def _prepare_dsv4_selection(
        self,
        *,
        compressed_layer: int,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
    ) -> int:
        num_reqs = req_pool_indices.size(0)
        num_real_reqs = self._dsv4_num_real_reqs_cpu
        if num_real_reqs > num_reqs:
            raise RuntimeError(
                "DeepSeek-V4 HiSparse real request count exceeds the padded "
                f"selection batch: real={num_real_reqs}, padded={num_reqs}."
            )
        self.dsv4_predicted_raw_indices[compressed_layer, :num_reqs].copy_(
            top_k_result[:num_reqs]
        )
        self.dsv4_batch_req_indices[compressed_layer, :num_reqs].copy_(
            req_pool_indices.to(torch.int64)
        )
        self.dsv4_batch_seq_lens[compressed_layer, :num_reqs].copy_(
            compressed_seq_lens[:num_reqs].to(torch.int64)
        )
        self.dsv4_batch_num_reqs[compressed_layer].fill_(num_real_reqs)
        self._dsv4_batch_num_reqs_cpu[compressed_layer] = num_real_reqs
        return num_reqs

    def _run_dsv4_selection(
        self,
        *,
        compressed_layer: int,
        num_reqs: int,
    ) -> torch.Tensor:
        raw_indices = self.dsv4_predicted_raw_indices[
            compressed_layer, :num_reqs
        ]
        req_indices = self.dsv4_batch_req_indices[
            compressed_layer, :num_reqs
        ]
        seq_lens = self.dsv4_batch_seq_lens[compressed_layer, :num_reqs]
        num_real_reqs = self.dsv4_batch_num_reqs[compressed_layer]
        output = self.dsv4_predicted_device_locs[
            compressed_layer, :num_reqs
        ]
        num_real_reqs_cpu = self._dsv4_batch_num_reqs_cpu[compressed_layer]
        if num_real_reqs_cpu < num_reqs:
            output[num_real_reqs_cpu:].fill_(-1)

        if self.dsv4_prefetch_mode == "h2d":
            return self._run_swap_in_kernel(
                req_indices,
                seq_lens,
                raw_indices,
                compressed_layer,
                output_buffer=output,
                num_real_reqs=num_real_reqs,
            )

        assert self.dsv4_miss_host_locs is not None
        assert self.dsv4_miss_count is not None
        classify_cache_residency_mla(
            top_k_tokens=raw_indices,
            device_buffer_tokens=self.req_device_buffer_tokens[compressed_layer],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[compressed_layer],
            hit_device_locs=output,
            miss_host_locs=self.dsv4_miss_host_locs[
                compressed_layer, :num_reqs
            ],
            miss_count=self.dsv4_miss_count[compressed_layer, :num_reqs],
            req_pool_indices=req_indices,
            seq_lens=seq_lens,
            hot_buffer_size=self.device_buffer_size,
            num_real_reqs=num_real_reqs,
            block_size=256,
        )
        return output

    def complete_dsv4_prediction(
        self,
        *,
        physical_layer_id: int,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
    ) -> None:
        compressed_layer = self._dsv4_compressed_layer_id(physical_layer_id)
        assert compressed_layer == self._dsv4_prediction_compressed_layer
        if self.dsv4_profile:
            self._dsv4_index_end_events[compressed_layer].record(
                device_module.current_stream()
            )
        num_reqs = self._prepare_dsv4_selection(
            compressed_layer=compressed_layer,
            req_pool_indices=req_pool_indices,
            compressed_seq_lens=compressed_seq_lens,
            top_k_result=top_k_result,
        )
        with torch.profiler.record_function(
            f"dsv4_hisparse/prefetch_{self.dsv4_prefetch_mode}/"
            f"layer_{physical_layer_id}"
        ):
            self._run_dsv4_selection(
                compressed_layer=compressed_layer,
                num_reqs=num_reqs,
            )
        self._dsv4_ready_events[compressed_layer].record(
            device_module.current_stream()
        )
        self._dsv4_ready[compressed_layer] = True

    def process_dsv4_current_index(
        self,
        *,
        physical_layer_id: int,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback for the first CSA layer, which has no previous CSA input."""
        compressed_layer = self._dsv4_compressed_layer_id(physical_layer_id)
        self._wait_dsv4_periodic(compressed_layer)
        num_reqs = self._prepare_dsv4_selection(
            compressed_layer=compressed_layer,
            req_pool_indices=req_pool_indices,
            compressed_seq_lens=compressed_seq_lens,
            top_k_result=top_k_result,
        )
        result = self._run_dsv4_selection(
            compressed_layer=compressed_layer,
            num_reqs=num_reqs,
        )
        if self.dsv4_prefetch_mode == "cpu":
            self._dsv4_active_cpu_layers[physical_layer_id] = compressed_layer
        return result

    def try_consume_dsv4_prefetch(
        self,
        *,
        physical_layer_id: int,
        num_reqs: int,
    ) -> Optional[torch.Tensor]:
        if not self.is_dsv4_hisparse:
            return None
        compressed_layer = self._dsv4_compressed_layer_id(physical_layer_id)
        if not self._dsv4_ready[compressed_layer]:
            return None

        current_stream = device_module.current_stream()
        was_ready = self._dsv4_ready_events[compressed_layer].query()
        wait_start = wait_end = None
        if self.dsv4_profile:
            wait_start = device_module.Event(enable_timing=True)
            wait_end = device_module.Event(enable_timing=True)
            wait_start.record(current_stream)
        self._dsv4_ready_events[compressed_layer].wait(current_stream)
        if self.dsv4_profile:
            wait_end.record(current_stream)
            wait_end.synchronize()
            self._profile_add(
                physical_layer_id,
                "target_wait_ms",
                wait_start.elapsed_time(wait_end),
            )
            self._profile_add(
                physical_layer_id, "target_blocked", float(not was_ready)
            )
            self._profile_add(
                physical_layer_id,
                "index_ms",
                self._dsv4_prefetch_start_events[compressed_layer].elapsed_time(
                    self._dsv4_index_end_events[compressed_layer]
                ),
            )
            self._profile_add(
                physical_layer_id,
                f"{self.dsv4_prefetch_mode}_prefetch_ms",
                self._dsv4_index_end_events[compressed_layer].elapsed_time(
                    self._dsv4_ready_events[compressed_layer]
                ),
            )

        self._dsv4_ready[compressed_layer] = False
        if self.dsv4_prefetch_mode == "cpu":
            self._dsv4_active_cpu_layers[physical_layer_id] = compressed_layer
        return self.dsv4_predicted_device_locs[
            compressed_layer, :num_reqs
        ]

    def launch_dsv4_cpu_attention(
        self,
        *,
        physical_layer_id: int,
        q: torch.Tensor,
        softmax_scale: float,
        head_dim_v: int,
    ) -> bool:
        compressed_layer = self._dsv4_active_cpu_layers.get(physical_layer_id)
        if compressed_layer is None or self.dsv4_prefetch_mode != "cpu":
            return False
        if physical_layer_id in self._dsv4_cpu_jobs:
            raise RuntimeError(
                "DeepSeek-V4 CPU attention job already exists for layer "
                f"{physical_layer_id}"
            )

        from sglang.srt.layers.attention.dsv4.hisparse_cpu import (
            cpu_miss_attention,
        )

        query = q.squeeze(1)
        num_reqs, num_heads, _ = query.shape
        num_real_reqs = self._dsv4_batch_num_reqs_cpu[compressed_layer]
        if num_real_reqs > num_reqs:
            raise RuntimeError(
                "DeepSeek-V4 HiSparse real request count exceeds the CPU "
                f"attention batch: real={num_real_reqs}, padded={num_reqs}."
            )
        assert self.dsv4_miss_host_locs is not None
        assert self._dsv4_cpu_miss_locs
        miss_gpu = self.dsv4_miss_host_locs[
            compressed_layer, :num_real_reqs
        ]
        query_cpu = torch.empty(
            (num_real_reqs, num_heads, query.shape[-1]),
            dtype=query.dtype,
            device="cpu",
            pin_memory=True,
        )
        miss_cpu = self._dsv4_cpu_miss_locs[compressed_layer][:num_real_reqs]
        output_cpu = torch.zeros(
            (num_reqs, num_heads, head_dim_v),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        lse_cpu = torch.full(
            (num_reqs, num_heads),
            float("-inf"),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )

        current_stream = device_module.current_stream()
        self.dsv4_d2h_stream.wait_stream(current_stream)
        copy_done = device_module.Event()
        with device_module.stream(self.dsv4_d2h_stream):
            query_cpu.copy_(query[:num_real_reqs], non_blocking=True)
            miss_cpu.copy_(miss_gpu, non_blocking=True)
            copy_done.record(self.dsv4_d2h_stream)

        host_cache = self.mem_pool_host.kv_buffer[compressed_layer]

        def run_cpu_attention():
            copy_done.synchronize()
            with torch.profiler.record_function(
                f"dsv4_hisparse/cpu_miss_attention/layer_{physical_layer_id}"
            ):
                return cpu_miss_attention(
                    query=query_cpu,
                    miss_host_locs=miss_cpu,
                    host_cache=host_cache,
                    softmax_scale=softmax_scale,
                    head_dim_v=head_dim_v,
                    output=output_cpu[:num_real_reqs],
                    lse=lse_cpu[:num_real_reqs],
                )

        assert self._dsv4_cpu_executor is not None
        future = self._dsv4_cpu_executor.submit(run_cpu_attention)
        self._dsv4_cpu_jobs[physical_layer_id] = (
            future,
            output_cpu,
            lse_cpu,
            time.perf_counter_ns(),
        )
        return True

    def finish_dsv4_cpu_attention(
        self,
        *,
        physical_layer_id: int,
        gpu_output: torch.Tensor,
        gpu_lse: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
        gpu_timing_events=None,
    ) -> torch.Tensor:
        job = self._dsv4_cpu_jobs.pop(physical_layer_id, None)
        if job is None:
            return gpu_output
        future, output_cpu, lse_cpu, launch_ns = job
        wait_begin_ns = time.perf_counter_ns()
        timing = future.result()
        cpu_wait_ms = (time.perf_counter_ns() - wait_begin_ns) * 1.0e-6
        cpu_critical_ms = (time.perf_counter_ns() - launch_ns) * 1.0e-6

        from sglang.srt.layers.attention.dsv4.hisparse_cpu import (
            merge_cpu_gpu_attention,
        )

        with torch.profiler.record_function(
            f"dsv4_hisparse/lse_merge/layer_{physical_layer_id}"
        ):
            merged = merge_cpu_gpu_attention(
                gpu_output=gpu_output,
                gpu_lse=gpu_lse,
                cpu_output=output_cpu,
                cpu_lse=lse_cpu,
                attn_sink=attn_sink,
            )

        if self.dsv4_profile:
            self._profile_add(physical_layer_id, "cpu_total_ms", timing.total_ms)
            self._profile_add(
                physical_layer_id, "cpu_dequant_ms", timing.dequant_ms
            )
            self._profile_add(
                physical_layer_id, "cpu_qk_softmax_ms", timing.qk_softmax_ms
            )
            self._profile_add(physical_layer_id, "cpu_pv_ms", timing.pv_ms)
            # ScoutAttention returns this host-miss partition's attention
            # output/LSE to the GPU; it does not recall these KV rows.
            self._profile_add(
                physical_layer_id,
                "cpu_attention_miss_tokens",
                float(timing.miss_tokens),
            )
            self._profile_add(physical_layer_id, "cpu_blocked_ms", cpu_wait_ms)
            self._profile_add(
                physical_layer_id,
                "cpu_critical_ms",
                cpu_critical_ms,
            )
            if gpu_timing_events is not None:
                gpu_start, gpu_end = gpu_timing_events
                gpu_end.synchronize()
                gpu_attention_ms = gpu_start.elapsed_time(gpu_end)
                self._profile_add(
                    physical_layer_id,
                    "gpu_hit_attention_ms",
                    gpu_attention_ms,
                )
                # Both intervals start immediately before the GPU hit kernel
                # (the CPU interval also includes its small D2H prologue).
                # Their difference estimates the CPU tail exposed on the
                # decode critical path after GPU attention can no longer hide it.
                self._profile_add(
                    physical_layer_id,
                    "cpu_unhidden_ms",
                    max(0.0, cpu_critical_ms - gpu_attention_ms),
                )

        compressed_layer = self._dsv4_active_cpu_layers.pop(physical_layer_id)
        self._schedule_dsv4_periodic_recall(
            physical_layer_id=physical_layer_id,
            compressed_layer=compressed_layer,
        )
        return merged

    def _schedule_dsv4_periodic_recall(
        self,
        *,
        physical_layer_id: int,
        compressed_layer: int,
    ) -> None:
        if not self._dsv4_periodic_due:
            return
        assert self.dsv4_prefetch_mode == "cpu"
        if self._dsv4_periodic_pending[compressed_layer]:
            raise RuntimeError(
                "DeepSeek-V4 periodic recall from the previous interval is "
                "still pending"
            )

        current_stream = device_module.current_stream()
        self.dsv4_periodic_stream.wait_stream(current_stream)
        with device_module.stream(self.dsv4_periodic_stream):
            if self.dsv4_profile:
                assert self.dsv4_miss_host_locs is not None
                assert self.dsv4_miss_count is not None
                assert self._dsv4_periodic_miss_count_cpu
                self._dsv4_periodic_start_events[compressed_layer].record(
                    self.dsv4_periodic_stream
                )
            num_reqs = self.dsv4_batch_num_reqs[compressed_layer]
            # num_reqs is a device scalar; buffers are fixed-capacity and the
            # kernel uses num_real_reqs to ignore the padded rows.
            self._run_swap_in_kernel(
                self.dsv4_batch_req_indices[compressed_layer],
                self.dsv4_batch_seq_lens[compressed_layer],
                self.dsv4_predicted_raw_indices[compressed_layer],
                compressed_layer,
                record_plan=self.dsv4_profile,
                output_buffer=self.dsv4_predicted_device_locs[compressed_layer],
                num_real_reqs=num_reqs,
                miss_src_buffer=(
                    self.dsv4_miss_host_locs[compressed_layer]
                    if self.dsv4_profile
                    else None
                ),
                miss_dst_buffer=(
                    self.dsv4_predicted_page_indices[compressed_layer]
                    if self.dsv4_profile
                    else None
                ),
                miss_count_buffer=(
                    self.dsv4_miss_count[compressed_layer]
                    if self.dsv4_profile
                    else None
                ),
            )
            if self.dsv4_profile:
                num_reqs_cpu = self._dsv4_batch_num_reqs_cpu[compressed_layer]
                self._dsv4_periodic_miss_count_cpu[compressed_layer][
                    :num_reqs_cpu
                ].copy_(
                    self.dsv4_miss_count[compressed_layer, :num_reqs_cpu],
                    non_blocking=True,
                )
            self._dsv4_periodic_end_events[compressed_layer].record(
                self.dsv4_periodic_stream
            )
        self._dsv4_periodic_pending[compressed_layer] = True
        if self.dsv4_profile:
            self._dsv4_periodic_profile_step[
                compressed_layer
            ] = self._dsv4_decode_step
            self._dsv4_periodic_profile_num_reqs[
                compressed_layer
            ] = self._dsv4_batch_num_reqs_cpu[compressed_layer]

    def _wait_dsv4_periodic(self, compressed_layer: int) -> None:
        if not self._dsv4_periodic_pending[compressed_layer]:
            return
        current_stream = device_module.current_stream()
        was_ready = self._dsv4_periodic_end_events[compressed_layer].query()
        if self.dsv4_profile:
            wait_start = device_module.Event(enable_timing=True)
            wait_end = device_module.Event(enable_timing=True)
            wait_start.record(current_stream)
            self._dsv4_periodic_end_events[compressed_layer].wait(current_stream)
            wait_end.record(current_stream)
            wait_end.synchronize()
            recall_ms = self._dsv4_periodic_start_events[
                compressed_layer
            ].elapsed_time(self._dsv4_periodic_end_events[compressed_layer])
            wait_ms = wait_start.elapsed_time(wait_end)
            physical_layer_id = self._dsv4_compressed_to_physical[compressed_layer]
            self._profile_add(
                physical_layer_id,
                "periodic_kv_recall_ms",
                recall_ms,
            )
            self._profile_add(
                physical_layer_id,
                "periodic_kv_recall_wait_ms",
                wait_ms,
            )
            self._profile_add(
                physical_layer_id,
                "periodic_kv_recall_blocked",
                float(not was_ready),
            )
            self._collect_dsv4_periodic_profile(
                compressed_layer=compressed_layer,
                recall_ms=recall_ms,
                wait_ms=wait_ms,
                blocked=not was_ready,
            )
        else:
            self._dsv4_periodic_end_events[compressed_layer].wait(current_stream)
        self._dsv4_periodic_pending[compressed_layer] = False

    def _collect_dsv4_periodic_profile(
        self,
        *,
        compressed_layer: int,
        recall_ms: float,
        wait_ms: float,
        blocked: bool,
    ) -> None:
        step = self._dsv4_periodic_profile_step[compressed_layer]
        num_reqs = self._dsv4_periodic_profile_num_reqs[compressed_layer]
        if step is None:
            raise RuntimeError(
                "DeepSeek-V4 periodic recall completed without profile metadata"
            )

        per_req_misses = self._dsv4_periodic_miss_count_cpu[compressed_layer][
            :num_reqs
        ].tolist()
        miss_tokens = sum(per_req_misses)
        recall_bytes = miss_tokens * self.dsv4_item_size_bytes
        physical_layer_id = self._dsv4_compressed_to_physical[compressed_layer]
        miss_mean = miss_tokens / num_reqs if num_reqs else 0.0
        miss_p50 = self._profile_percentile(per_req_misses, 0.50)
        miss_p95 = self._profile_percentile(per_req_misses, 0.95)
        miss_max = max(per_req_misses, default=0)

        self._profile_add(
            physical_layer_id,
            "periodic_kv_recall_miss_tokens",
            float(miss_tokens),
        )
        self._profile_add(
            physical_layer_id,
            "periodic_kv_recall_bytes",
            float(recall_bytes),
        )
        logger.info(
            "DeepSeek-V4 HiSparse periodic KV recall step=%d layer=%d "
            "num_reqs=%d periodic_kv_recall_miss_tokens=%d "
            "kv_miss_per_req_mean=%.3f kv_miss_per_req_p50=%d "
            "kv_miss_per_req_p95=%d kv_miss_per_req_max=%d "
            "kv_recall_bytes=%d kv_recall_ms=%.3f "
            "kv_recall_wait_ms=%.3f kv_recall_blocked=%s",
            step,
            physical_layer_id,
            num_reqs,
            miss_tokens,
            miss_mean,
            miss_p50,
            miss_p95,
            miss_max,
            recall_bytes,
            recall_ms,
            wait_ms,
            blocked,
        )

        step_layers = self._dsv4_periodic_profile_by_step[step]
        if physical_layer_id in step_layers:
            raise RuntimeError(
                "DeepSeek-V4 periodic recall profile was collected twice for "
                f"step={step}, layer={physical_layer_id}"
            )
        step_layers[physical_layer_id] = miss_tokens
        expected_layers = set(self._dsv4_registered_layers)
        if expected_layers and expected_layers.issubset(step_layers):
            misses_by_layer = {
                layer_id: step_layers[layer_id]
                for layer_id in sorted(expected_layers)
            }
            layer_misses = list(misses_by_layer.values())
            minimum = min(layer_misses)
            maximum = max(layer_misses)
            logger.info(
                "DeepSeek-V4 HiSparse periodic KV recall layer comparison "
                "step=%d kv_recall_miss_tokens_by_layer=%s min=%d max=%d "
                "mean=%.3f spread=%d",
                step,
                misses_by_layer,
                minimum,
                maximum,
                sum(layer_misses) / len(layer_misses),
                maximum - minimum,
            )
            del self._dsv4_periodic_profile_by_step[step]

        self._dsv4_periodic_profile_step[compressed_layer] = None
        self._dsv4_periodic_profile_num_reqs[compressed_layer] = 0

    def _flush_dsv4_periodic_recalls(self) -> None:
        for compressed_layer, pending in enumerate(self._dsv4_periodic_pending):
            if pending:
                self._wait_dsv4_periodic(compressed_layer)
        # Without profiling, _wait_dsv4_periodic only inserts stream waits.
        # The host-side synchronization is still required before request or
        # coordinator resources can be released.
        self.dsv4_periodic_stream.synchronize()

    def _profile_add(self, layer_id: int, metric: str, value: float) -> None:
        if self.dsv4_profile:
            self._dsv4_profile_data[layer_id][metric].append(float(value))

    @staticmethod
    def _profile_percentile(values: List[float], fraction: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
        return ordered[index]

    def _log_dsv4_profile(self) -> None:
        for layer_id, metrics in self._dsv4_profile_data.items():
            summary = {}
            for name, values in metrics.items():
                if not values:
                    continue
                summary[name] = {
                    "mean": sum(values) / len(values),
                    "p50": self._profile_percentile(values, 0.50),
                    "p95": self._profile_percentile(values, 0.95),
                    "max": max(values),
                }
            prefetch_name = f"{self.dsv4_prefetch_mode}_prefetch_ms"
            if prefetch_name in metrics and "target_wait_ms" in metrics:
                prefetch_mean = sum(metrics[prefetch_name]) / len(
                    metrics[prefetch_name]
                )
                wait_mean = sum(metrics["target_wait_ms"]) / len(
                    metrics["target_wait_ms"]
                )
                summary["prefetch_overlap_coverage"] = (
                    max(0.0, 1.0 - wait_mean / prefetch_mean)
                    if prefetch_mean > 0
                    else 1.0
                )
            if "cpu_critical_ms" in metrics and "cpu_unhidden_ms" in metrics:
                cpu_critical_mean = sum(metrics["cpu_critical_ms"]) / len(
                    metrics["cpu_critical_ms"]
                )
                cpu_unhidden_mean = sum(metrics["cpu_unhidden_ms"]) / len(
                    metrics["cpu_unhidden_ms"]
                )
                summary["attention_overlap_coverage"] = (
                    max(0.0, 1.0 - cpu_unhidden_mean / cpu_critical_mean)
                    if cpu_critical_mean > 0
                    else 1.0
                )
            logger.info(
                "DeepSeek-V4 HiSparse profile step=%d layer=%d %s",
                self._dsv4_decode_step,
                layer_id,
                summary,
            )
        self._dsv4_profile_data.clear()

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self.enable_prefetch:
            # Skip-layer copies read the pinned host pool on the prefetch stream.
            self.prefetch_stream.synchronize()
        if self.is_dsv4_hisparse:
            self.dsv4_prefetch_stream.synchronize()
            self.dsv4_d2h_stream.synchronize()
            self._flush_dsv4_periodic_recalls()
            if self.dsv4_profile and self._dsv4_profile_data:
                self._log_dsv4_profile()
        if self._dsv4_cpu_executor is not None:
            self._dsv4_cpu_executor.shutdown(wait=True)
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        if self.is_dsv4_hisparse:
            # A periodic recall may still read this request's pinned host rows
            # and write its device-buffer slots after the model forward returns.
            # Drain it and collect its profile before either allocation can be
            # released or reused.
            self._flush_dsv4_periodic_recalls()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
        output_buffer: Optional[torch.Tensor] = None,
        num_real_reqs: Optional[torch.Tensor] = None,
        miss_src_buffer: Optional[torch.Tensor] = None,
        miss_dst_buffer: Optional[torch.Tensor] = None,
        miss_count_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full plan+IO swap-in kernel for one layer; return its slot table.

        record_plan also records the miss plan. Shared-index prefetch uses the
        coordinator-wide default buffers; callers may provide explicit buffers
        when the plan is only needed for profiling.
        """
        num_reqs = req_pool_indices.size(0)
        top_k_indices = (
            self.top_k_device_locs_buffer[:num_reqs]
            if output_buffer is None
            else output_buffer
        )
        num_real_reqs = (
            num_real_reqs
            if num_real_reqs is not None
            else self.num_real_reqs
        )

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        if record_plan:
            explicit_plan = (
                miss_src_buffer,
                miss_dst_buffer,
                miss_count_buffer,
            )
            if any(buffer is not None for buffer in explicit_plan) and not all(
                buffer is not None for buffer in explicit_plan
            ):
                raise ValueError(
                    "miss_src_buffer, miss_dst_buffer, and miss_count_buffer "
                    "must be provided together"
                )
            if miss_src_buffer is None:
                miss_src_buffer = self._miss_src
                miss_dst_buffer = self._miss_dst
                miss_count_buffer = self._miss_count
            plan = dict(
                miss_src=miss_src_buffer[:num_reqs],
                miss_dst=miss_dst_buffer[:num_reqs],
                miss_count=miss_count_buffer[:num_reqs],
            )
        else:
            if any(
                buffer is not None
                for buffer in (
                    miss_src_buffer,
                    miss_dst_buffer,
                    miss_count_buffer,
                )
            ):
                raise ValueError("miss-plan buffers require record_plan=True")
            plan = {}
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=num_real_reqs,
            skip_io=self.skip_io,
            **plan,
        )
        return top_k_indices

    def _run_copy_only_kernel(self, num_reqs: int, skip_layer: int) -> None:
        """Replay the anchor's recorded miss plan into a skip layer's buffers
        (IO-only; the anchor's slot table stays valid -- lockstep layout)."""
        copy_cache_planned_mla(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[skip_layer],
            device_buffer=self.mem_pool_device.kv_buffer[skip_layer],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._prefetch_copy_blocks,
            is_dsv4_layout=self.is_dsv4_hisparse,
            skip_io=self.skip_io,
        )

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        With prefetch enabled, anchors swap in synchronously (recording the miss
        plan) and prefetch their skip layers' copies; skip layers just wait.
        """
        if not self.enable_prefetch:
            return self._run_swap_in_kernel(
                req_pool_indices, compressed_seq_lens, top_k_result, layer_id
            )

        num_reqs = req_pool_indices.size(0)
        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            return self.top_k_device_locs_buffer[:num_reqs]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
        )
        if group:
            # Fork: the prefetch stream must observe the anchor's plan (produced
            # on the current stream) before replaying it.
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    self._run_copy_only_kernel(num_reqs, skip_layer)
                    self._prefetch_events[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs
