from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.sparsity.factory import parse_hisparse_config


def _parse(raw):
    return parse_hisparse_config(SimpleNamespace(hisparse_config=raw))


def test_dsv4_prefetch_defaults_to_cpu_with_eight_step_recall():
    config = _parse(None)
    assert config.dsv4_prefetch_mode == "cpu"
    assert config.dsv4_recall_interval == 8
    assert config.dsv4_cpu_attention_backend == "auto"
    assert config.dsv4_cpu_threads == 0
    assert config.dsv4_profile is False
    assert config.dsv4_profile_log_interval == 100


def test_dsv4_prefetch_explicit_h2d_and_profile():
    config = _parse(
        '{"dsv4_prefetch_mode":"h2d","dsv4_recall_interval":0,'
        '"dsv4_cpu_attention_backend":"torch","dsv4_cpu_threads":16,'
        '"dsv4_profile":true,"dsv4_profile_log_interval":7}'
    )
    assert config.dsv4_prefetch_mode == "h2d"
    assert config.dsv4_recall_interval == 0
    assert config.dsv4_cpu_attention_backend == "torch"
    assert config.dsv4_cpu_threads == 16
    assert config.dsv4_profile is True
    assert config.dsv4_profile_log_interval == 7


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"dsv4_prefetch_mode":"invalid"}', "dsv4_prefetch_mode"),
        ('{"dsv4_recall_interval":true}', "dsv4_recall_interval"),
        ('{"dsv4_cpu_attention_backend":"invalid"}', "dsv4_cpu_attention_backend"),
        ('{"dsv4_cpu_threads":-1}', "dsv4_cpu_threads"),
        ('{"dsv4_profile":1}', "dsv4_profile"),
        ('{"dsv4_profile_log_interval":0}', "dsv4_profile_log_interval"),
    ],
)
def test_dsv4_prefetch_config_validation(raw, message):
    with pytest.raises(ValueError, match=message):
        _parse(raw)

