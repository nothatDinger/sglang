from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

DSV4_PREFETCH_MODE_SCOUT = "scout"
DSV4_PREFETCH_MODE_INFINIGEN = "infinigen"

_DSV4_PREFETCH_MODE_ALIASES = {
    "cpu": DSV4_PREFETCH_MODE_SCOUT,
    "h2d": DSV4_PREFETCH_MODE_INFINIGEN,
}
_DSV4_PREFETCH_MODES = {
    DSV4_PREFETCH_MODE_SCOUT,
    DSV4_PREFETCH_MODE_INFINIGEN,
}


@dataclass(frozen=True)
class DSV4PrefetchModeSelection:
    mode: str
    explicit: bool
    deprecated_alias: Optional[str] = None


@dataclass(frozen=True)
class SparseRuntimePolicy:
    enabled: bool
    dsv4_prefetch_mode: str
    dsv4_prefetch_mode_explicit: bool
    legacy_hisparse_enabled: bool
    ignored_on_pd_prefill: bool = False
    deprecated_mode_alias: Optional[str] = None


def load_hisparse_extra_config(server_args: Any) -> dict[str, Any]:
    raw_config = getattr(server_args, "hisparse_config", None)
    if raw_config is None:
        return {}
    if isinstance(raw_config, dict):
        return dict(raw_config)
    try:
        extra_config = json.loads(raw_config)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse hisparse_config: {exc}") from exc
    if not isinstance(extra_config, dict):
        raise ValueError("hisparse_config must be a JSON object")
    return extra_config


def resolve_dsv4_prefetch_mode(
    extra_config: dict[str, Any],
) -> DSV4PrefetchModeSelection:
    explicit = "dsv4_prefetch_mode" in extra_config
    raw_mode = extra_config.get("dsv4_prefetch_mode", DSV4_PREFETCH_MODE_SCOUT)
    deprecated_alias = None
    if not isinstance(raw_mode, str):
        raise ValueError(
            "dsv4_prefetch_mode must be 'scout' or 'infinigen' "
            "(deprecated aliases: 'cpu' or 'h2d'), got "
            f"{raw_mode!r}"
        )
    if raw_mode in _DSV4_PREFETCH_MODE_ALIASES:
        deprecated_alias = raw_mode
        raw_mode = _DSV4_PREFETCH_MODE_ALIASES[raw_mode]
    if raw_mode not in _DSV4_PREFETCH_MODES:
        raise ValueError(
            "dsv4_prefetch_mode must be 'scout' or 'infinigen' "
            "(deprecated aliases: 'cpu' or 'h2d'), got "
            f"{raw_mode!r}"
        )
    return DSV4PrefetchModeSelection(
        mode=raw_mode,
        explicit=explicit,
        deprecated_alias=deprecated_alias,
    )


def resolve_sparse_runtime_policy(server_args: Any) -> SparseRuntimePolicy:
    selection = resolve_dsv4_prefetch_mode(load_hisparse_extra_config(server_args))
    legacy_enabled = bool(getattr(server_args, "enable_hisparse", False))
    disaggregation_mode = getattr(server_args, "disaggregation_mode", "null")
    disaggregation_mode = getattr(disaggregation_mode, "value", disaggregation_mode)

    ignored_on_pd_prefill = (
        selection.explicit
        and not legacy_enabled
        and disaggregation_mode == "prefill"
    )
    standalone_enabled = selection.explicit and not ignored_on_pd_prefill

    return SparseRuntimePolicy(
        enabled=legacy_enabled or standalone_enabled,
        dsv4_prefetch_mode=selection.mode,
        dsv4_prefetch_mode_explicit=selection.explicit,
        legacy_hisparse_enabled=legacy_enabled,
        ignored_on_pd_prefill=ignored_on_pd_prefill,
        deprecated_mode_alias=selection.deprecated_alias,
    )


def log_sparse_runtime_policy_warnings(policy: SparseRuntimePolicy) -> None:
    if policy.deprecated_mode_alias is not None:
        logger.warning(
            "dsv4_prefetch_mode=%r is deprecated; use %r instead.",
            policy.deprecated_mode_alias,
            policy.dsv4_prefetch_mode,
        )
    if policy.ignored_on_pd_prefill:
        logger.warning(
            "Ignoring explicitly configured dsv4_prefetch_mode=%r on the PD "
            "prefill server. ScoutAttention/InfiniGen runs on PD decode only; "
            "the prefill server keeps the regular KV cache pool.",
            policy.dsv4_prefetch_mode,
        )
