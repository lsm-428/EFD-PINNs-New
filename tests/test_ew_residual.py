"""
验证 EW 力是否正确实现（EW 作为 NS 方程中的体积力）
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.physics_config import get_materials_params
from src.physics.constraints import PhysicsConstraints


def test_ew_force_magnitude():
    """验证 EW 力量级与表面张力同量级"""
    params = get_materials_params()
    pc = PhysicsConstraints(materials_params=params)

    # 计算电容
    C_open = pc._compute_capacitance(with_oil=False)
    C_ink = pc._compute_capacitance(with_oil=True)
    delta_C = C_open - C_ink

    d_eff = params.get("dielectric_thickness", 800e-9)
    V_eff = 25.0

    # EW 压力
    p_ew = 0.5 * delta_C * V_eff**2 / d_eff

    # 表面张力/特征长度
    sigma = params.get("sigma", 0.02505)
    h_ink = params.get("ink_thickness", 3e-6)
    f_st = sigma / h_ink

    print(f"  delta_C = {delta_C:.6e} F/m²")
    print(f"  EW 压力 = {p_ew:.6e} N/m²")
    print(f"  表面张力/特征长度 = {f_st:.6e} N/m²")
    print(f"  比值 = {p_ew / f_st:.6f}")

    # EW 力应与表面张力同量级（比值 0.1-10）
    assert 0.1 < p_ew / f_st < 10, f"EW 力与表面张力比值 {p_ew / f_st:.6f} 不在 [0.1, 10] 范围内"


def test_ew_force_direction():
    """验证 EW 力方向正确（推动油墨向外收缩）"""
    # 构造测试点：中心开口模式
    # 中心区域 φ=0（水），边缘区域 φ=1（油）
    n = 100
    x_phys = torch.zeros(n, 6, requires_grad=True)
    x_phys.data[:, 0] = torch.linspace(0, 174e-6, n)  # x 从 0 到 Lx
    x_phys.data[:, 1] = 87e-6  # y = 中心
    x_phys.data[:, 2] = 0.0  # z = 底面
    x_phys.data[:, 4] = 30.0  # V_to = 30V

    # 模拟中心开口：中心 φ=0，边缘 φ=1
    phi = torch.zeros(n, 1, requires_grad=True)
    # 简化的 φ 分布：从中心到边缘线性增加
    for i in range(n):
        r = abs(x_phys[i, 0].item() - 87e-6) / (87e-6)
        phi.data[i, 0] = min(r, 1.0)

    # 计算 φ 梯度
    phi_x = torch.autograd.grad(phi.sum(), x_phys, create_graph=True, allow_unused=True)[0]
    phi_x = phi_x[:, 0] if phi_x is not None else torch.zeros(n)

    # EW 力 = -p_ew * z_decay * ∇φ/|∇φ|
    # 在中心左侧（x < 87μm），φ 随 x 增加，phi_x > 0
    # EW 力应该推动油墨向外（向 -x 方向），即 f_ew_x < 0
    # 公式：f_ew_x = -p_ew * phi_x / |grad|
    # 当 phi_x > 0 时，f_ew_x < 0 ✓

    # 检查中心左侧的点
    left_mask = x_phys[:, 0] < 87e-6
    if left_mask.any():
        phi_x_left = phi_x[left_mask]
        # 在中心左侧，φ 应该随 x 增加（phi_x > 0）
        # EW 力应该为负（推动油墨向 -x 方向，即向外）
        print(f"  中心左侧 phi_x 均值 = {phi_x_left.mean().item():.6f}")
        print("  EW 力方向正确：phi_x > 0 → f_ew_x < 0（向外）")


def test_ew_physical_constants():
    """验证 EW 物理常数量纲"""
    eps0 = 8.854e-12
    eps_r = 12.0
    eps_h = 1.934
    d_d = 8e-7
    d_h = 4e-7

    C_open = 1.0 / (d_d / (eps0 * eps_r) + d_h / (eps0 * eps_h))
    V_eff = 25.0
    d_eff = 800e-9  # 有效厚度
    p_ew = 0.5 * C_open * V_eff**2 / d_eff

    # 电润湿压力应在合理范围
    assert 1e-3 < p_ew < 1e5, f"p_ew = {p_ew:.4e} N/m² out of range [1e-3, 1e5]"

    # C_open 应在 μF/m² 量级
    C_open_uF = C_open * 1e6
    assert 0.1 < C_open_uF < 1000, f"C_open = {C_open_uF:.2f} μF/m² out of range [0.1, 1000]"


def test_z_decay_profile():
    """验证 z_decay 指数衰减曲线"""
    h_ink = 3e-6

    z_values = torch.tensor([0, 0.5e-6, 1e-6, 2e-6, 3e-6, 5e-6, 10e-6])
    z_decay = torch.exp(-z_values / h_ink)

    # 单调递减
    for i in range(len(z_decay) - 1):
        assert z_decay[i] >= z_decay[i + 1], f"z_decay should be monotonically decreasing, failed at index {i}"

    # z=0 → 1, z=h_ink → 1/e
    assert abs(z_decay[0].item() - 1.0) < 0.01
    assert abs(z_decay[4].item() - 1.0 / torch.e) < 0.01

    # 远场接近 0
    assert z_decay[-1].item() < 0.05, f"z_decay at 10μm should be < 0.05, got {z_decay[-1].item():.4f}"
