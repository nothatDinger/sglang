import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_dsa import DeepSeekDSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    DSABackendAdaptor,
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)
from sglang.srt.mem_cache.sparsity.runtime import (
    load_hisparse_extra_config,
    resolve_dsv4_prefetch_mode,
)

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_dsa": lambda config, device, **kw: DeepSeekDSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekDSAAlgorithm):
        return DSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2), swap_in_block_size (960).
    DeepSeek-V4 prefetch fields are parsed explicitly. All remaining fields go
    to sparse_extra_config.
    """
    extra_config = load_hisparse_extra_config(server_args)

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)
    swap_in_block_size = extra_config.pop("swap_in_block_size", 960)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )
    if not isinstance(swap_in_block_size, int) or isinstance(swap_in_block_size, bool):
        raise ValueError(
            f"swap_in_block_size must be an integer, got {swap_in_block_size!r}"
        )
    if swap_in_block_size <= 0 or swap_in_block_size > 1024:
        raise ValueError(
            f"swap_in_block_size ({swap_in_block_size}) must be in the range [1, 1024]"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)

    dsv4_prefetch_mode_selection = resolve_dsv4_prefetch_mode(extra_config)
    if dsv4_prefetch_mode_selection.explicit:
        extra_config.pop("dsv4_prefetch_mode")
    dsv4_prefetch_correction = extra_config.pop("dsv4_prefetch_correction", False)
    dsv4_recall_interval = extra_config.pop("dsv4_recall_interval", 8)
    dsv4_cpu_attention_backend = extra_config.pop(
        "dsv4_cpu_attention_backend", "auto"
    )
    dsv4_cpu_threads = extra_config.pop("dsv4_cpu_threads", 0)
    dsv4_profile = extra_config.pop("dsv4_profile", False)
    dsv4_profile_log_interval = extra_config.pop(
        "dsv4_profile_log_interval", 100
    )

    if not isinstance(dsv4_prefetch_correction, bool):
        raise ValueError("dsv4_prefetch_correction must be a boolean")
    if dsv4_prefetch_correction and not (
        dsv4_prefetch_mode_selection.explicit
        or bool(getattr(server_args, "enable_hisparse", False))
    ):
        raise ValueError(
            "dsv4_prefetch_correction requires ScoutAttention/InfiniGen prefetch; "
            "set dsv4_prefetch_mode or enable HiSparse"
        )
    if (
        not isinstance(dsv4_recall_interval, int)
        or isinstance(dsv4_recall_interval, bool)
    ):
        raise ValueError("dsv4_recall_interval must be an integer")
    if dsv4_cpu_attention_backend not in ("auto", "torch"):
        raise ValueError(
            "dsv4_cpu_attention_backend must be 'auto' or 'torch', got "
            f"{dsv4_cpu_attention_backend!r}"
        )
    if not isinstance(dsv4_cpu_threads, int) or isinstance(dsv4_cpu_threads, bool):
        raise ValueError("dsv4_cpu_threads must be an integer")
    if dsv4_cpu_threads < 0:
        raise ValueError("dsv4_cpu_threads must be non-negative")
    if not isinstance(dsv4_profile, bool):
        raise ValueError("dsv4_profile must be a boolean")
    if (
        not isinstance(dsv4_profile_log_interval, int)
        or isinstance(dsv4_profile_log_interval, bool)
        or dsv4_profile_log_interval <= 0
    ):
        raise ValueError("dsv4_profile_log_interval must be a positive integer")

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        swap_in_block_size=swap_in_block_size,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        dsv4_prefetch_mode=dsv4_prefetch_mode_selection.mode,
        dsv4_prefetch_mode_explicit=(
            dsv4_prefetch_mode_selection.explicit
        ),
        dsv4_prefetch_mode_deprecated_alias=(
            dsv4_prefetch_mode_selection.deprecated_alias
        ),
        dsv4_prefetch_correction=dsv4_prefetch_correction,
        dsv4_recall_interval=dsv4_recall_interval,
        dsv4_cpu_attention_backend=dsv4_cpu_attention_backend,
        dsv4_cpu_threads=dsv4_cpu_threads,
        dsv4_profile=dsv4_profile,
        dsv4_profile_log_interval=dsv4_profile_log_interval,
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
