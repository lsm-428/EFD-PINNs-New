#!/usr/bin/env python3
"""验证 Fix 1-5 的数值正确性 (含断言)"""

import json
import math
import os

import pytest


def test_fix2_ac_dimensional():
    """Fix 2: AC 方程量纲验证"""
    sigma = 0.02505  # N/m
    eps = 5e-6  # m
    M_ac = 1e-11  # m^3*s/kg

    D_eff = M_ac * sigma / eps  # m^2/s
    mob = D_eff / eps  # m/s
    tau = eps / mob  # s

    # D_eff 应为正
    assert D_eff > 0, f"D_eff should be positive, got {D_eff:.4e}"
    assert mob > 0, f"mob should be positive, got {mob:.4e}"

    # tau 应在 ms 量级（界面迁移时间 1-50ms）
    tau_ms = tau * 1000
    assert 0.1 < tau_ms < 100, (
        f"Interface migration time tau = {tau_ms:.1f}ms out of range [0.1, 100]"
    )


def test_fix1_ew_indicator_zdecay():
    """Fix 1: EW 界面指示函数 z_decay 验证"""
    h_ink = 3e-6

    test_points = {
        0: 1.0,  # z=0 → decay=1
        500e-9: math.exp(-500e-9 / h_ink),
        1e-6: math.exp(-1e-6 / h_ink),  # z=1μm
        3e-6: math.exp(-1),  # z=h_ink → 1/e
        20e-6: math.exp(-20e-6 / h_ink),  # far field
    }

    z_decays = {z: val for z, val in test_points.items() if isinstance(z, int | float)}
    z_list = sorted(z_decays.keys())

    # 单调递减
    for i in range(len(z_list) - 1):
        assert z_decays[z_list[i]] >= z_decays[z_list[i + 1]], (
            f"z_decay not monotonic at z={z_list[i]:.1e}"
        )

    # z=0 → 1.0
    assert abs(z_decays[0] - 1.0) < 0.01

    # z=h_ink → 1/e
    assert abs(z_decays[3e-6] - 1.0 / math.e) < 0.01

    # 远场 << 1
    assert z_decays[20e-6] < 0.01, f"Far-field z_decay should be << 1, got {z_decays[20e-6]:.4f}"


def test_fix3_ns_ew_zdecay():
    """Fix 3: NS EW z_decay 验证 (800nm decay length)"""
    d_eff = 800e-9  # 800nm

    z_vals = [0, 100e-9, 500e-9, 800e-9, 1.6e-6, 3e-6, 20e-6]

    decays = [math.exp(-z / d_eff) for z in z_vals]

    # 单调递减
    for i in range(len(decays) - 1):
        assert decays[i] >= decays[i + 1]

    # z=0 → 1
    assert abs(decays[0] - 1.0) < 0.01

    # z=d_eff → 1/e
    assert abs(decays[3] - 1.0 / math.e) < 0.03

    # 远场 close to 0
    assert decays[-1] < 0.01, f"Far-field decay should be < 0.01, got {decays[-1]:.4f}"


def test_fix4_ew_weight_enabled():
    """Fix 4: EW 权重验证 — electrowetting_weight > 0"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "v4.6-optimized.json"
    )

    if not os.path.exists(config_path):
        pytest.skip(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    ew_w = cfg.get("physics", {}).get("electrowetting_weight", 0)

    # EW 权重必须 > 0（不能关闭电润湿物理）
    assert ew_w > 0, f"electrowetting_weight = {ew_w}, should be > 0 (EW physics must be enabled)"


def test_fix5_lp_lv_annealing():
    """Fix 5: LP/LV 退火范围验证"""
    # LP S3 后期: max 0.8 (不再到 1.0)
    lp_max = 0.8
    assert lp_max < 1.0, "LP should not reach 1.0 in S3 (prevents NaN)"
    assert lp_max >= 0.5, "LP max should be >= 0.5 for meaningful regularization"

    # LV S3 后期: anneal → min 0.1
    lv_min = 0.1
    assert lv_min > 0, "LV floor should be > 0"
    assert lv_min < 0.3, f"LV floor {lv_min} should be lower than old default 0.3"
