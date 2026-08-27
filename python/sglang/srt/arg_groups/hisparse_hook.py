from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv", "flashinfer_sparse_mla"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")


def _is_hip() -> bool:
    from sglang.srt.server_args import is_hip

    return is_hip()


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if _is_hip():
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if _is_hip():
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv", "flashinfer_sparse_mla"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use one of {sorted(allowed_backends)}, or omit the option "
            "to let SGLang pick a backend for this platform."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate the legacy HiSparse flag and the unified DSV4 sparse runtime."""
    from sglang.srt.mem_cache.sparsity.runtime import (
        log_sparse_runtime_policy_warnings,
        resolve_sparse_runtime_policy,
    )

    policy = resolve_sparse_runtime_policy(server_args)
    log_sparse_runtime_policy_warnings(policy)
    if not policy.enabled and not policy.dsv4_prefetch_mode_explicit:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_sparse_runtime = is_deepseek_v4(hf_config)
    is_dsa_hisparse = is_deepseek_dsa(hf_config)
    if policy.dsv4_prefetch_mode_explicit and not is_v4_sparse_runtime:
        raise ValueError(
            "An explicit dsv4_prefetch_mode enables ScoutAttention/InfiniGen "
            "and is only supported for DeepSeek-V4 models."
        )
    if not policy.enabled:
        return

    is_hip = _is_hip()

    if policy.legacy_hisparse_enabled:
        assert is_dsa_hisparse or is_v4_sparse_runtime, (
            "--enable-hisparse is only supported for DSA (DeepSeek Sparse "
            "Attention) models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4."
        )

    assert server_args.disable_radix_cache, (
        "The HiSparse/ScoutAttention/InfiniGen sparse runtime currently "
        "requires --disable-radix-cache."
    )

    if getattr(server_args, "dcp_size", 1) > 1:
        raise NotImplementedError(
            "The HiSparse/ScoutAttention/InfiniGen sparse runtime with "
            "--dcp-size > 1 is not supported: the host pool has no DCP "
            "index translation."
        )

    # DeepSeek-V4 handles its own dtype/backend pairing. The checks below only
    # apply to the legacy DSA HiSparse path.
    if is_v4_sparse_runtime:
        if is_hip:
            # TEMPORARY GUARD: the DSV4 host-backed runtime is not supported on
            # the unified-KV path because it requires the separate packed C4 pool.
            from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
                is_unified_kv_triton,
            )

            if is_unified_kv_triton():
                raise ValueError(
                    "The DeepSeek-V4 HiSparse/ScoutAttention/InfiniGen runtime "
                    "is not supported with the unified-KV path on ROCm "
                    "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton). Set "
                    "SGLANG_HACK_FLASHMLA_BACKEND=triton or disable the sparse "
                    "runtime."
                )
        return

    from sglang.srt.arg_groups.overrides import resolved_view

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)
