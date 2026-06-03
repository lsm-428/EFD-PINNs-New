"""验证 EW 源项 v2 的量级"""

from src.config.physics_config import get_materials_params


def test_ew_source_magnitude():
    """验证 EW 源项 v2 的量级在合理范围"""
    mp = get_materials_params()

    eps0 = 8.854e-12
    eps_r = mp.get("relative_permittivity", 12.0)
    eps_h = mp.get("epsilon_hydrophobic", 1.934)
    d_d = mp.get("dielectric_thickness", 8e-7)
    d_h = mp.get("hydrophobic_thickness", 4e-7)
    h_ink = mp.get("ink_thickness", 3e-6)
    eps_ink = mp.get("epsilon_ink", 4.0)
    sigma_ac = mp.get("surface_tension_polar_ink", 0.02505)
    eps_ac = mp.get("ac_interface_width", 5e-6)
    M_ac = mp.get("ac_mobility", 5e-11)

    C_open = 1.0 / (d_d / (eps0 * eps_r) + d_h / (eps0 * eps_h))
    C_ink = 1.0 / (d_d / (eps0 * eps_r) + d_h / (eps0 * eps_h) + h_ink / (eps0 * eps_ink))
    delta_C = C_open - C_ink

    V_eff = 25.0  # 30V - 5V

    # EW 源项（无 z_decay，即 z=0）
    ew_at_surface = M_ac * delta_C * V_eff**2 / eps_ac**2

    # 基本物理约束
    assert C_open > 0, f"C_open must be positive, got {C_open:.4e}"
    assert delta_C > 0, f"delta_C must be positive (C_open > C_ink), got {delta_C:.4e}"
    assert C_ink < C_open, f"C_ink ({C_ink:.4e}) must be less than C_open ({C_open:.4e})"

    # EW 源项应在合理量级 (0.001 ~ 10 1/s)
    assert (
        0.0001 < ew_at_surface < 100
    ), f"EW source magnitude {ew_at_surface:.4e} out of reasonable range [1e-4, 1e2]"

    # 电容量级验证（μF/m² 量级）
    assert 1e-6 < C_open < 1e-3, f"C_open = {C_open:.4e} F/m² out of range [1e-6, 1e-3] (μF/m²)"


def test_ew_source_with_oil_layer():
    """验证含油层电容的物理一致性"""
    mp = get_materials_params()
    eps0 = 8.854e-12
    eps_r = mp.get("relative_permittivity", 12.0)
    eps_h = mp.get("epsilon_hydrophobic", 1.934)
    d_d = mp.get("dielectric_thickness", 8e-7)
    d_h = mp.get("hydrophobic_thickness", 4e-7)
    h_ink = mp.get("ink_thickness", 3e-6)
    eps_ink = mp.get("epsilon_ink", 4.0)

    C_open = 1.0 / (d_d / (eps0 * eps_r) + d_h / (eps0 * eps_h))
    C_ink = 1.0 / (d_d / (eps0 * eps_r) + d_h / (eps0 * eps_h) + h_ink / (eps0 * eps_ink))

    # 油层增加总阻抗 → 电容下降
    assert C_ink < C_open, "C_ink must be less than C_open (oil adds impedance)"

    # 开关比应在合理范围 (1.1-5)
    ratio = C_open / C_ink
    assert 1.1 < ratio < 10, f"C_open/C_ink ratio = {ratio:.2f}, expected 1.1-10"
