"""``RPSConfig`` 与 ``validate_fault_config`` 的合法性校验测试。"""

import numpy as np
import pytest

from config import RPSConfig, validate_fault_config

# ---------------------------------------------------------------------------
# RPSConfig __post_init__
# ---------------------------------------------------------------------------

def test_default_config_is_valid():
    cfg = RPSConfig()
    assert cfg.h_hop == 2


@pytest.mark.parametrize("kwargs", [
    {"h_hop": 0},
    {"h_hop": -1},
    {"k_trunc": 0},
    {"window_len": 0},
    {"eta": 0.0},
    {"eta": -1.0},
    {"top_m": 0},
    {"top_agents_k": 0},
    {"diagnose_every": 0},
    {"gain": -0.1},
    {"proxy_std_weight": -0.1},
    {"proxy_global_weight": -0.5},
    {"uniform_factor": -0.1},
    {"uniform_factor": 1.5},
    {"chi2_confidence": 0.0},
    {"chi2_confidence": 1.0},
    {"tau_quantile": 0.0},
    {"tau_quantile": 1.0},
    {"record_agent_idx": -1},
])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError, match="Invalid RPSConfig"):
        RPSConfig(**kwargs)


def test_burn_in_must_be_at_least_window():
    """``burn_in < window_len`` 时滑窗在故障期前都没填满，配置无效。"""
    with pytest.raises(ValueError, match="burn_in"):
        RPSConfig(burn_in=10, window_len=20)


def test_replace_revalidates():
    """``replace`` 后无效字段也应被 __post_init__ 捕获。"""
    cfg = RPSConfig()
    with pytest.raises(ValueError):
        cfg.replace(h_hop=-3)


# ---------------------------------------------------------------------------
# validate_fault_config
# ---------------------------------------------------------------------------

def test_fault_config_minimal_valid():
    validate_fault_config({
        'onset': 100, 'agents': [3], 'type': 'constant',
        'delta': np.ones(5) * 0.01,
    })


def test_fault_config_empty_agents_no_delta_ok():
    """Ideal 用法：agents=[], delta=None 是合法的（无故障）。"""
    validate_fault_config({
        'onset': 9999, 'agents': [], 'type': 'constant', 'delta': None,
    })


def test_fault_config_missing_onset():
    with pytest.raises(ValueError, match="onset"):
        validate_fault_config({
            'agents': [1], 'type': 'constant', 'delta': np.ones(3),
        })


def test_fault_config_unknown_type():
    with pytest.raises(ValueError, match="type"):
        validate_fault_config({
            'onset': 100, 'agents': [1], 'type': 'unknown', 'delta': np.ones(3),
        })


def test_fault_config_intermittent_requires_prob():
    with pytest.raises(ValueError, match="prob"):
        validate_fault_config({
            'onset': 100, 'agents': [1], 'type': 'intermittent',
            'delta': np.ones(3),
        })


def test_fault_config_intermittent_prob_out_of_range():
    with pytest.raises(ValueError, match="prob"):
        validate_fault_config({
            'onset': 100, 'agents': [1], 'type': 'intermittent',
            'delta': np.ones(3), 'prob': 1.5,
        })


def test_fault_config_drift_cap_positive():
    with pytest.raises(ValueError, match="drift_cap"):
        validate_fault_config({
            'onset': 100, 'agents': [1], 'type': 'drift',
            'delta': np.ones(3), 'drift_cap': -1,
        })


def test_fault_config_constant_with_agents_needs_delta():
    with pytest.raises(ValueError, match="delta"):
        validate_fault_config({
            'onset': 100, 'agents': [1], 'type': 'constant',
        })


def test_fault_config_delta_dim_mismatch():
    """传入 d 时校验 delta.shape == (d,)。"""
    with pytest.raises(ValueError, match="delta"):
        validate_fault_config({
            'onset': 100, 'agents': [1], 'type': 'constant',
            'delta': np.ones(5),
        }, d=10)


def test_fault_config_delta_dim_match_passes():
    """正确维度通过。"""
    validate_fault_config({
        'onset': 100, 'agents': [1], 'type': 'constant',
        'delta': np.ones(10),
    }, d=10)
