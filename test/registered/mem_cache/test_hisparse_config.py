import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups.hisparse_hook import validate_hisparse
from sglang.srt.mem_cache.sparsity.factory import parse_hisparse_config
from sglang.srt.mem_cache.sparsity.runtime import (
    log_sparse_runtime_policy_warnings,
    resolve_sparse_runtime_policy,
)


def _args(raw=None, *, enable_hisparse=False, disaggregation_mode="null"):
    return SimpleNamespace(
        hisparse_config=raw,
        enable_hisparse=enable_hisparse,
        disaggregation_mode=disaggregation_mode,
    )


def _parse(raw):
    return parse_hisparse_config(_args(raw))


def test_dsv4_prefetch_defaults_to_scout_without_explicit_activation():
    config = _parse(None)
    assert config.dsv4_prefetch_mode == "scout"
    assert config.dsv4_prefetch_mode_explicit is False
    assert config.dsv4_prefetch_mode_deprecated_alias is None
    assert config.dsv4_recall_interval == 8
    assert config.dsv4_cpu_attention_backend == "auto"
    assert config.dsv4_cpu_threads == 0
    assert config.dsv4_profile is False
    assert config.dsv4_profile_log_interval == 100

    policy = resolve_sparse_runtime_policy(_args())
    assert policy.enabled is False
    assert policy.dsv4_prefetch_mode == "scout"
    assert policy.dsv4_prefetch_mode_explicit is False


def test_non_mode_hisparse_config_does_not_activate_standalone_runtime():
    policy = resolve_sparse_runtime_policy(_args('{"top_k":512}'))
    assert policy.enabled is False
    assert policy.dsv4_prefetch_mode_explicit is False


def test_dsv4_prefetch_explicit_infinigen_and_profile():
    config = _parse(
        '{"dsv4_prefetch_mode":"infinigen","dsv4_recall_interval":0,'
        '"dsv4_cpu_attention_backend":"torch","dsv4_cpu_threads":16,'
        '"dsv4_profile":true,"dsv4_profile_log_interval":7}'
    )
    assert config.dsv4_prefetch_mode == "infinigen"
    assert config.dsv4_prefetch_mode_explicit is True
    assert config.dsv4_prefetch_mode_deprecated_alias is None
    assert config.dsv4_recall_interval == 0
    assert config.dsv4_cpu_attention_backend == "torch"
    assert config.dsv4_cpu_threads == 16
    assert config.dsv4_profile is True
    assert config.dsv4_profile_log_interval == 7


@pytest.mark.parametrize(
    ("legacy_mode", "canonical_mode"),
    [("cpu", "scout"), ("h2d", "infinigen")],
)
def test_dsv4_prefetch_legacy_aliases_are_normalized(
    legacy_mode, canonical_mode
):
    config = _parse(f'{{"dsv4_prefetch_mode":"{legacy_mode}"}}')
    assert config.dsv4_prefetch_mode == canonical_mode
    assert config.dsv4_prefetch_mode_explicit is True
    assert config.dsv4_prefetch_mode_deprecated_alias == legacy_mode


@pytest.mark.parametrize(
    (
        "raw",
        "enable_hisparse",
        "disaggregation_mode",
        "expected_enabled",
        "expected_ignored",
    ),
    [
        (None, False, "null", False, False),
        ('{"dsv4_prefetch_mode":"scout"}', False, "null", True, False),
        ('{"dsv4_prefetch_mode":"infinigen"}', False, "decode", True, False),
        ('{"dsv4_prefetch_mode":"scout"}', False, "prefill", False, True),
        (None, True, "null", True, False),
        (None, True, "decode", True, False),
        ('{"dsv4_prefetch_mode":"infinigen"}', True, "prefill", True, False),
    ],
)
def test_sparse_runtime_policy_matrix(
    raw,
    enable_hisparse,
    disaggregation_mode,
    expected_enabled,
    expected_ignored,
):
    policy = resolve_sparse_runtime_policy(
        _args(
            raw,
            enable_hisparse=enable_hisparse,
            disaggregation_mode=disaggregation_mode,
        )
    )
    assert policy.enabled is expected_enabled
    assert policy.ignored_on_pd_prefill is expected_ignored


def test_legacy_alias_and_pd_prefill_emit_warnings(caplog):
    policy = resolve_sparse_runtime_policy(
        _args('{"dsv4_prefetch_mode":"cpu"}', disaggregation_mode="prefill")
    )
    with caplog.at_level(
        logging.WARNING,
        logger="sglang.srt.mem_cache.sparsity.runtime",
    ):
        log_sparse_runtime_policy_warnings(policy)

    assert "deprecated" in caplog.text
    assert "Ignoring explicitly configured" in caplog.text


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"dsv4_prefetch_mode":"invalid"}', "dsv4_prefetch_mode"),
        ('{"dsv4_prefetch_mode":[]}', "dsv4_prefetch_mode"),
        ('{"dsv4_recall_interval":true}', "dsv4_recall_interval"),
        ('{"dsv4_cpu_attention_backend":"invalid"}', "dsv4_cpu_attention_backend"),
        ('{"dsv4_cpu_threads":-1}', "dsv4_cpu_threads"),
        ('{"dsv4_profile":1}', "dsv4_profile"),
        ('{"dsv4_profile_log_interval":0}', "dsv4_profile_log_interval"),
        ("[]", "JSON object"),
    ],
)
def test_dsv4_prefetch_config_validation(raw, message):
    with pytest.raises(ValueError, match=message):
        _parse(raw)


def _validation_args(
    raw, *, disaggregation_mode="null", disable_radix_cache=True
):
    args = _args(raw, disaggregation_mode=disaggregation_mode)
    args.disable_radix_cache = disable_radix_cache
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["TestArchitecture"])
    )
    return args


def test_pd_prefill_ignores_standalone_mode_after_dsv4_model_validation():
    args = _validation_args(
        '{"dsv4_prefetch_mode":"scout"}',
        disaggregation_mode="prefill",
        disable_radix_cache=False,
    )
    with (
        patch(
            "sglang.srt.configs.model_config.is_deepseek_v4",
            return_value=True,
        ),
        patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa",
            return_value=False,
        ),
    ):
        validate_hisparse(args)


def test_explicit_mode_on_non_dsv4_model_fails_even_on_pd_prefill():
    args = _validation_args(
        '{"dsv4_prefetch_mode":"infinigen"}',
        disaggregation_mode="prefill",
    )
    with (
        patch(
            "sglang.srt.configs.model_config.is_deepseek_v4",
            return_value=False,
        ),
        patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa",
            return_value=False,
        ),
        pytest.raises(ValueError, match="only supported for DeepSeek-V4"),
    ):
        validate_hisparse(args)


def test_standalone_mode_requires_radix_cache_to_be_disabled():
    args = _validation_args(
        '{"dsv4_prefetch_mode":"scout"}',
        disable_radix_cache=False,
    )
    with (
        patch(
            "sglang.srt.configs.model_config.is_deepseek_v4",
            return_value=True,
        ),
        patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa",
            return_value=False,
        ),
        pytest.raises(AssertionError, match="disable-radix-cache"),
    ):
        validate_hisparse(args)


def test_standalone_mode_rejects_dcp():
    args = _validation_args('{"dsv4_prefetch_mode":"infinigen"}')
    args.dcp_size = 2
    with (
        patch(
            "sglang.srt.configs.model_config.is_deepseek_v4",
            return_value=True,
        ),
        patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa",
            return_value=False,
        ),
        pytest.raises(NotImplementedError, match="dcp-size"),
    ):
        validate_hisparse(args)
