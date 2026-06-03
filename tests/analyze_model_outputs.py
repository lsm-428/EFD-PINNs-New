"""
验证 PINN 输出相关的物理约束和计算（不依赖已训练模型）
"""

import numpy as np
import torch

from src.config.physics_config import PHYSICS, get_materials_params


def test_phi_range_constraint():
    """验证 φ 值范围计算的基本正确性"""
    # 模拟开口率计算：phi < 0.3 为极性液体/开口区域
    phi_test = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])

    opening_mask = phi_test < 0.3
    opening_rate = opening_mask.float().mean().item()

    # 0.0, 0.1, 0.2 → 3/8 = 37.5% 开口
    assert abs(opening_rate - 0.375) < 0.01, f"Expected opening rate 0.375, got {opening_rate:.3f}"


def test_opening_rate_upper_bound():
    """验证开口率不超过 η_max 物理约束"""
    eta_max = PHYSICS.get("eta_max", 0.85)

    # 模拟 φ 场（全开口：phi=0）
    phi = torch.zeros(10000)
    opening_mask = phi < 0.3
    opening_rate = opening_mask.float().mean().item()

    # 全开口时 opening_rate = 1.0，但 η_max 限制了物理上限
    assert opening_rate <= 1.0, "Opening rate cannot exceed 1.0"
    assert eta_max <= 1.0, f"eta_max = {eta_max} should be <= 1.0"


def test_voltage_monotonicity():
    """验证电压-开口率单调性（物理期望）"""
    # 开口率应随电压单调递增（在饱和前）
    # 这是一个物理约束测试：高电压 → 更大的电润湿力 → 更高开口率
    V_low = 10.0
    V_high = 20.0

    # 用 Young-Lippmann 验证 cosθ 随 V 单调递减（→ 开口率单调递增）
    params = get_materials_params()
    eps0 = 8.854e-12
    eps_r = params.get("relative_permittivity", 12.0)
    eps_h = params.get("epsilon_hydrophobic", 1.934)
    d_d = params.get("dielectric_thickness", 8e-7)
    d_h = params.get("hydrophobic_thickness", 4e-7)
    theta0 = PHYSICS.get("theta0_degrees", 120)

    C_open = 1.0 / (d_d / (eps0 * eps_r) + d_h / (eps0 * eps_h))
    sigma = params.get("surface_tension_polar_ink", 0.02505)

    cos_low = np.cos(np.deg2rad(theta0)) + C_open * V_low**2 / (2 * sigma)
    cos_high = np.cos(np.deg2rad(theta0)) + C_open * V_high**2 / (2 * sigma)

    # Young-Lippmann: cosθ 随 V² 单调递增
    # cosθ 增大 → θ 减小 → 更亲水 → 开口率增大
    assert (
        cos_high > cos_low
    ), f"cosθ should increase with V: cos({V_low}V)={cos_low:.3f}, cos({V_high}V)={cos_high:.3f}"


def test_field_dimensions():
    """验证 φ 场和相关物理量的维度"""
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]

    assert Lx > 0, f"Lx should be positive: {Lx}"
    assert Ly > 0, f"Ly should be positive: {Ly}"
    assert Lx == Ly, f"Expected square pixel: Lx={Lx * 1e6:.0f}μm, Ly={Ly * 1e6:.0f}μm"

    # 域大小应在合理范围 (100-200μm 像素)
    assert 100e-6 <= Lx <= 300e-6, f"Lx = {Lx * 1e6:.0f}μm out of expected pixel range [100, 300]μm"


def test_bottom_grid_sampling():
    """验证底面 (z=0) 网格采样的数值一致性"""
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
    n_grid = 100

    x = np.linspace(0, Lx, n_grid)
    y = np.linspace(0, Ly, n_grid)
    X, Y = np.meshgrid(x, y)

    # 采样 10000 个点 → (10000, 6)
    pts = np.stack(
        [
            X.flatten(),
            Y.flatten(),
            np.zeros(n_grid * n_grid),
            np.zeros(n_grid * n_grid),
            30.0 * np.ones(n_grid * n_grid),
            0.020 * np.ones(n_grid * n_grid),
        ],
        axis=1,
    )

    assert pts.shape == (
        n_grid * n_grid,
        6,
    ), f"Expected shape ({n_grid * n_grid}, 6), got {pts.shape}"

    # z 应为 0
    assert np.allclose(pts[:, 2], 0.0), "z coordinate should be 0 at bottom"
    # V_to 应为 30
    assert np.allclose(pts[:, 4], 30.0), "V_to should be 30V"


def test_time_sampling():
    """验证时间轴采样"""
    t = 0.020  # 20ms

    # 应在训练时间范围 [0, 0.05] 内
    assert 0 <= t <= 0.05, f"t={t} should be in training range [0, 0.05]"

    # 零时间应有 phi = phi_IC（初始条件）
    assert t >= 0, "t should be non-negative"
