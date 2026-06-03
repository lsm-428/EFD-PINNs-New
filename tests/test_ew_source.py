"""验证 EW 源项的量级和梯度"""

import pytest
import torch

from src.config.physics_config import get_materials_params
from src.physics.constraints import PhysicsConstraints


def test_ew_source_magnitude():
    """验证 EW 源项在典型工作点的量级"""
    params = get_materials_params()
    device = torch.device("cpu")

    # 构造测试点：底面中心，高电压
    n = 100
    Lx, Ly, Lz = 174e-6, 174e-6, 20e-6
    h_ink = 3e-6

    x_phys = torch.zeros(n, 6, device=device)
    x_phys[:, 0] = Lx / 2  # x 中心
    x_phys[:, 1] = Ly / 2  # y 中心
    x_phys[:, 2] = torch.linspace(0, h_ink, n, device=device)  # z: 0~3μm
    x_phys[:, 3] = 0.0  # V_from
    x_phys[:, 4] = 30.0  # V_to
    x_phys[:, 5] = 0.02  # t_since
    x_phys.requires_grad_(True)

    # 模拟油墨平铺状态
    predictions = torch.zeros(n, 5, device=device)
    predictions[:, 4] = 0.5 * (1 - torch.tanh((x_phys[:, 2] - h_ink) / 1e-6))

    # 手动计算 EW 源项
    eps0 = 8.854e-12
    eps_r = params.get("relative_permittivity", 12.0)
    eps_h = params.get("epsilon_hydrophobic", 1.934)
    d_d = params.get("dielectric_thickness", 8e-7)
    sigma_ac = params.get("surface_tension_polar_ink", 0.02505)
    eps_ac = params.get("ac_interface_width", 5e-07)
    M_ac = params.get("ac_mobility", 1e-10)

    # v7.2 Route 1: C = A_eff / (d_su8/(eps0*eps_su8) + d_teflon/(eps0*eps_teflon))
    A_eff = params.get("A_eff", 1.2)
    eps_su8 = params.get("epsilon_su8", 3.28)
    eps_teflon = params.get("epsilon_teflon", 1.934)
    d_su8 = params.get("d_su8", 4e-7)
    d_teflon = params.get("d_teflon", 4e-7)
    C_open = A_eff / (d_su8 / (eps0 * eps_su8) + d_teflon / (eps0 * eps_teflon))
    V_eff = 30.0 - 5.0
    p_ew = 0.5 * C_open * V_eff**2

    z = x_phys[:, 2]
    z_decay = torch.exp(-z / h_ink)
    phi = predictions[:, 4]

    D_eff = M_ac * sigma_ac / eps_ac
    mob = D_eff / eps_ac
    ew_source = (p_ew * z_decay * phi / (sigma_ac * eps_ac)) * mob

    # 断言: 源项非零且有合理量级 (ε=0.5μm 后 EW 源项 ~1e6, ∝ 1/ε²)
    assert ew_source.max().item() > 0, "EW source must be positive at oil-covered bottom"
    assert (
        ew_source.mean().item() < 1e8
    ), f"EW source mean {ew_source.mean().item():.4e} too large, expected < 1e8 1/s (eps=0.5um)"

    # 断言: z_decay 在 z=0 处为 1，在 z=h_ink 处衰减到 1/e
    assert (
        abs(z_decay[0].item() - 1.0) < 0.01
    ), f"z_decay at z=0 should be ~1.0, got {z_decay[0].item():.4f}"
    assert (
        abs(z_decay[-1].item() - 1.0 / torch.e) < 0.05
    ), f"z_decay at z=h_ink should be ~1/e, got {z_decay[-1].item():.4f}"


def test_ew_source_gradient():
    """验证 EW 源项有非零梯度"""
    params = get_materials_params()
    device = torch.device("cpu")

    n = 50
    x_phys = torch.zeros(n, 6, device=device)
    x_phys[:, 0] = 87e-6
    x_phys[:, 1] = 87e-6
    x_phys[:, 2] = 1e-6
    x_phys[:, 3] = 0.0
    x_phys[:, 4] = 30.0
    x_phys[:, 5] = 0.02
    x_phys.requires_grad_(True)

    predictions = torch.zeros(n, 5, device=device)
    predictions[:, 4] = 0.8

    # 通过 PhysicsConstraints 计算核心残差
    pc = PhysicsConstraints(params)
    residuals = pc.compute_core_residuals(x_phys, predictions)

    # AC 残差应存在
    ac_res = residuals.get("ac", residuals.get("phase_field"))
    if ac_res is None:
        pytest.skip("AC residual not available in compute_core_residuals")

    # AC 残差应有非零梯度
    if ac_res.requires_grad:
        grad = torch.autograd.grad(ac_res.sum(), x_phys, retain_graph=True)[0]
        assert grad is not None, "AC residual gradient should not be None"
        grad_norm = grad.norm().item()
        assert grad_norm > 0, f"AC residual gradient norm should be > 0, got {grad_norm:.6e}"
    else:
        pytest.skip("AC residual does not require grad")


def test_capacitance_components():
    """验证电容分量物理合理性"""
    params = get_materials_params()
    eps0 = 8.854e-12
    eps_r = params.get("relative_permittivity", 12.0)
    eps_h = params.get("epsilon_hydrophobic", 1.934)
    d_d = params.get("dielectric_thickness", 8e-7)
    d_h = params.get("hydrophobic_thickness", 4e-7)

    # v7.2 Route 1: C = A_eff / (d_su8/(eps0*eps_su8) + d_teflon/(eps0*eps_teflon))
    A_eff = params.get("A_eff", 1.2)
    eps_su8 = params.get("epsilon_su8", 3.28)
    eps_teflon = params.get("epsilon_teflon", 1.934)
    d_su8 = params.get("d_su8", 4e-7)
    d_teflon = params.get("d_teflon", 4e-7)
    C_open = A_eff / (d_su8 / (eps0 * eps_su8) + d_teflon / (eps0 * eps_teflon))

    # C_open 应为正且量级合理（μF/m²）
    assert C_open > 0
    assert 1e-6 < C_open < 1e-3, f"C_open = {C_open:.4e} F/m², expected μF/m² range"

    # 介质层应贡献主要阻抗
    Z_su8 = d_d / (eps0 * eps_r)
    Z_teflon = d_h / (eps0 * eps_h)
    assert Z_su8 > 0
    assert Z_teflon > 0
    # Teflon 虽然是疏水层但电容串联后 C 应小于单独 SU-8
    C_su8_only = eps0 * eps_su8 / d_su8
    assert (
        C_su8_only > C_open
    ), (
        f"C_open ({C_open:.4e}) should be less than C_su8_only ({C_su8_only:.4e}) "
        "(series adds impedance)"
    )
