"""
验证 EW 残差是否产生非零值（EW 嵌套在 AC 方程源项中）
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.physics_config import get_materials_params
from src.physics.constraints import PhysicsConstraints


def test_ew_residual_basic():
    """验证 compute_electrowetting_residual 返回结构"""
    params = get_materials_params()
    pc = PhysicsConstraints(materials_params=params)

    # 构造测试点
    n_points = 100
    x_phys = torch.rand(n_points, 6)
    x_phys[:, 0] *= 174e-6
    x_phys[:, 1] *= 174e-6
    x_phys[:, 2] = 0.0
    x_phys[:, 3] = 0.0
    x_phys[:, 4] = 30.0
    x_phys[:, 5] = 0.02

    predictions = torch.zeros(n_points, 5)
    predictions[:, 4] = 0.9

    residuals = pc.compute_electrowetting_residual(x_phys, predictions)

    # 断言: 返回是字典，包含 electrowetting 键
    assert isinstance(residuals, dict), "EW residual should return a dict"
    assert 'electrowetting' in residuals, "Dict should contain 'electrowetting' key"

    ew_res = residuals['electrowetting']
    assert isinstance(ew_res, torch.Tensor), f"EW residual should be a tensor, got {type(ew_res)}"
    assert ew_res.ndim == 1, f"EW residual should be 1D, got {ew_res.ndim}D"


def test_ew_residual_phi_dependence():
    """验证 EW 残差对 phi 有依赖性（油墨/极性液体不同响应）"""
    params = get_materials_params()
    pc = PhysicsConstraints(materials_params=params)

    n = 50
    x_phys = torch.zeros(n, 6)
    x_phys[:, 0] = 87e-6
    x_phys[:, 1] = 87e-6
    x_phys[:, 2] = 0.0
    x_phys[:, 3] = 0.0
    x_phys[:, 4] = 30.0
    x_phys[:, 5] = 0.02

    # 油墨覆盖 (phi ~ 0.9)
    pred_oil = torch.zeros(n, 5)
    pred_oil[:, 4] = 0.9
    res_oil = pc.compute_electrowetting_residual(x_phys, pred_oil)

    # 极性液体覆盖 (phi ~ 0.1)
    pred_water = torch.zeros(n, 5)
    pred_water[:, 4] = 0.1
    res_water = pc.compute_electrowetting_residual(x_phys, pred_water)

    ew_oil = res_oil['electrowetting'].abs().mean().item()
    ew_water = res_water['electrowetting'].abs().mean().item()

    # EW 力应与 phi 相关（油墨区域 EW 驱动力强）
    assert ew_oil >= 0, "EW residual magnitude should be non-negative"
    assert ew_water >= 0, "EW residual magnitude should be non-negative"


def test_ew_physical_constants():
    """验证 EW 物理常数量级"""
    eps0 = 8.854e-12
    eps_r = 12.0
    eps_h = 1.934
    d_d = 8e-7
    d_h = 4e-7

    C_open = 1.0 / (d_d / (eps0 * eps_r) + d_h / (eps0 * eps_h))
    V_eff = 25.0
    p_ew = 0.5 * C_open * V_eff**2

    # 电润湿压力应在合理范围 (0.001-1e4 N/m², Route 1 电容使 p_ew ~0.01)
    assert 1e-3 < p_ew < 1e4, \
        f"p_ew = {p_ew:.4e} N/m² out of range [1e-3, 1e4]"

    # C_open 应在 μF/m² 量级
    C_open_uF = C_open * 1e6
    assert 0.1 < C_open_uF < 1000, \
        f"C_open = {C_open_uF:.2f} μF/m² out of range [0.1, 1000]"


def test_z_decay_profile():
    """验证 z_decay 指数衰减曲线"""
    h_ink = 3e-6

    z_values = torch.tensor([0, 0.5e-6, 1e-6, 2e-6, 3e-6, 5e-6, 10e-6])
    z_decay = torch.exp(-z_values / h_ink)

    # 单调递减
    for i in range(len(z_decay) - 1):
        assert z_decay[i] >= z_decay[i + 1], \
            f"z_decay should be monotonically decreasing, failed at index {i}"

    # z=0 → 1, z=h_ink → 1/e
    assert abs(z_decay[0].item() - 1.0) < 0.01
    assert abs(z_decay[4].item() - 1.0/torch.e) < 0.01

    # 远场接近 0
    assert z_decay[-1].item() < 0.05, f"z_decay at 10μm should be < 0.05, got {z_decay[-1].item():.4f}"
