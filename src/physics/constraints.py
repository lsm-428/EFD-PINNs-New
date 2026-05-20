"""
EWPINN 物理约束模块
包含物理方程计算、材料参数和边界条件处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger("EWPINN_Physics")

# 导入统一物理配置
try:
    from src.config import get_materials_params, PHYSICS

    _DEFAULT_MATERIALS = None  # 延迟加载
except ImportError:
    logger.warning("统一配置模块不可用，使用本地默认参数")
    _DEFAULT_MATERIALS = None
    PHYSICS = {}

# 导入动态权重调整模块
try:
    from src.training.scheduler import (
        DynamicPhysicsWeightScheduler,
        PhysicsWeightIntegration,
    )

    DYNAMIC_WEIGHT_AVAILABLE = True
except ImportError:
    logger.warning("动态权重调整模块不可用，将使用固定权重")
    DYNAMIC_WEIGHT_AVAILABLE = False


def _get_default_materials_params() -> Dict:
    """获取默认材料参数（优先从统一配置加载）"""
    global _DEFAULT_MATERIALS
    if _DEFAULT_MATERIALS is not None:
        return _DEFAULT_MATERIALS.copy()

    try:
        from src.config import get_materials_params

        _DEFAULT_MATERIALS = get_materials_params()
        return _DEFAULT_MATERIALS.copy()
    except Exception:
        pass

    # 回退到硬编码默认值
    return {
        "viscosity": 1.01e-3,
        "density": 998.0,
        "surface_tension": 0.048,
        "permittivity": 80.1,
        "conductivity": 5.5e7,
        "youngs_modulus": 210e9,
        "allen_cahn_gamma": 4.5e-7,
        "poisson_ratio": 0.3,
        "contact_angle_theta0": 120.0,
        "epsilon_0": 8.854e-12,
        "dielectric_thickness": 4e-7,
        "relative_permittivity": 3.28,
        "dynamic_contact_angle_advancing": 120.0,
        "dynamic_contact_angle_receding": 100.0,
        "contact_line_friction": 1e-3,
        "pinning_energy": 1e-5,
        "slip_length": 1e-6,
        "dielectric_conductivity": 1e-12,
        "charge_relaxation_time": 1e-3,
        "leakage_current_coefficient": 1e-6,
        "max_charge_density": 1e-4,
        "ambient_temperature": 293.15,
        "thermal_conductivity_water": 0.6,
        "thermal_conductivity_oil": 0.15,
        "thermal_conductivity_dielectric": 0.02,
        "specific_heat_water": 4186.0,
        "thermal_expansion_water": 2.1e-4,
        "temperature_coefficient_surface_tension": -1.5e-4,
        "temperature_coefficient_viscosity": -3.5e-3,
        "density_polar": 998.0,
        "density_ink": 763.0,
        "viscosity_polar": 1.01e-3,
        "viscosity_ink": 9.41e-4,
        "surface_tension_polar_ink": 0.02505,
        "contact_angle_ink": 120.0,
        "theta_wall": 71.0,
        "ink_potential_min": 0.0,
        "ink_initial_fraction": 0.15,
        "ink_thickness": 3e-6,
        "domain_height": 20e-6,
        "wall_height": 3.5e-6,
        # 物理模型配置开关
        "use_convection": False,          # Re<<1, 默认关闭对流项
        "use_legacy_ac": False,           # 标准化 Allen-Cahn (界面宽度可控)
        "ac_interface_width": 5e-6,       # 界面宽度 (m)
        "ac_mobility": 1e-7,              # 迁移率 (m³·s/kg), τ_reaction≈2000s
        "use_adaptive_loss_scale": False, # 自适应损失归一化 (EMA)
    }


class PhysicsConstraints:
    """物理约束类 - 处理Navier-Stokes方程和材料属性"""

    def __init__(self, materials_params=None):
        # 从统一配置获取默认参数，允许外部覆盖
        default_params = _get_default_materials_params()
        if materials_params is not None:
            default_params.update(materials_params)
        self.materials_params = default_params

        # 预定义的边界条件权重
        self.boundary_weights = {"dirichlet": 100.0, "neumann": 10.0, "interface": 50.0}

        # 全局步数计数器，用于控制日志输出频率
        self.global_step = 0

    def compute_navier_stokes_residual(self, x, predictions, model=None):
        """
        计算Navier-Stokes方程残差 (优化版，与 pinn_two_phase.py 保持一致)

        关键点:
        - 包含对流项 (Inertia)
        - 使用混合密度和粘度
        - 包含表面张力 CSF 模型
        - 包含电润湿驱动力 (Electrowetting force)
        """
        try:
            if x is None:
                return self._empty_residual(x, predictions)

            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

            if not x.requires_grad:
                x = x.clone().detach().requires_grad_(True)

            if model is not None:
                predictions = model(x)

            # 提取变量 (u, v, w, p, phi)
            u, v, w, p = (
                predictions[:, 0],
                predictions[:, 1],
                predictions[:, 2],
                predictions[:, 3],
            )
            phi = (
                predictions[:, 4] if predictions.shape[1] >= 5 else torch.zeros_like(u)
            )

            # 材料参数 (基于 PHYSICS 常量)
            rho_oil = self.materials_params["density_ink"]
            rho_polar = self.materials_params["density_polar"]
            mu_oil = self.materials_params["viscosity_ink"]
            mu_polar = self.materials_params["viscosity_polar"]
            sigma = self.materials_params["surface_tension_polar_ink"]

            # 电润湿参数
            epsilon_0 = self.materials_params["epsilon_0"]
            epsilon_r = self.materials_params.get("relative_permittivity", 3.28)
            d_dielectric = self.materials_params.get("dielectric_thickness", 4e-7)
            epsilon_h = self.materials_params.get("epsilon_hydrophobic", 1.934)
            d_hydrophobic = self.materials_params.get("hydrophobic_thickness", 4e-7)
            Lx = self.materials_params.get("Lx", 174e-6)
            Ly = self.materials_params.get("Ly", 174e-6)
            Lz = self.materials_params.get("domain_height", 20e-6)

            # SU-8 + Teflon 双层串联单位面积电容
            C_ew = 1.0 / (d_dielectric/(epsilon_0*epsilon_r) + d_hydrophobic/(epsilon_0*epsilon_h))

            # 混合属性
            rho = phi * rho_oil + (1 - phi) * rho_polar
            mu = phi * mu_oil + (1 - phi) * mu_polar

            # 计算梯度
            def get_grad(y, x_in):
                return torch.autograd.grad(
                    y,
                    x_in,
                    grad_outputs=torch.ones_like(y),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

            g_u = get_grad(u.sum(), x)
            g_v = get_grad(v.sum(), x)
            g_w = get_grad(w.sum(), x)
            g_p = get_grad(p.sum(), x)
            g_phi = get_grad(phi.sum(), x)

            u_x, u_y, u_z = g_u[:, 0], g_u[:, 1], g_u[:, 2]
            v_x, v_y, v_z = g_v[:, 0], g_v[:, 1], g_v[:, 2]
            w_x, w_y, w_z = g_w[:, 0], g_w[:, 1], g_w[:, 2]
            p_x, p_y, p_z = g_p[:, 0], g_p[:, 1], g_p[:, 2]
            phi_x, phi_y, phi_z = g_phi[:, 0], g_phi[:, 1], g_phi[:, 2]

            # 时间导数 (如果有)
            u_t = g_u[:, 5] if x.shape[1] >= 6 else torch.zeros_like(u)
            v_t = g_v[:, 5] if x.shape[1] >= 6 else torch.zeros_like(v)
            w_t = g_w[:, 5] if x.shape[1] >= 6 else torch.zeros_like(w)

            # 二阶导数 (Laplacian 和 混合导数)
            def get_second_derivs(first_grad, x_in, is_phi=False):
                lap = torch.zeros_like(first_grad[:, 0])
                second_grads = {}
                coords = ["x", "y", "z"]
                for i in range(3):
                    g2 = get_grad(first_grad[:, i].sum(), x_in)
                    lap += g2[:, i]
                    if is_phi:
                        second_grads[f"{coords[i]}{coords[i]}"] = g2[:, i]
                        for j in range(i + 1, 3):
                            second_grads[f"{coords[i]}{coords[j]}"] = g2[:, j]
                return lap, second_grads

            lap_u, _ = get_second_derivs(g_u, x)
            lap_v, _ = get_second_derivs(g_v, x)
            lap_w, _ = get_second_derivs(g_w, x)
            lap_phi, phi_2nd = get_second_derivs(g_phi, x, is_phi=True)

            # 表面张力 (CSF 模型 - 精确曲率)
            phi_xx, phi_yy, phi_zz = phi_2nd["xx"], phi_2nd["yy"], phi_2nd["zz"]
            phi_xy, phi_xz, phi_yz = phi_2nd["xy"], phi_2nd["xz"], phi_2nd["yz"]

            grad_phi_mag_sq = phi_x**2 + phi_y**2 + phi_z**2 + 1e-10
            grad_phi_mag = torch.sqrt(grad_phi_mag_sq)

            # 精确曲率公式: kappa = -div(grad(phi)/|grad(phi)|)
            numerator = (
                phi_xx * (phi_y**2 + phi_z**2)
                + phi_yy * (phi_x**2 + phi_z**2)
                + phi_zz * (phi_x**2 + phi_y**2)
                - 2
                * (
                    phi_x * phi_y * phi_xy
                    + phi_x * phi_z * phi_xz
                    + phi_y * phi_z * phi_yz
                )
            )
            kappa = -numerator / (grad_phi_mag_sq * grad_phi_mag + 1e-10)

            f_st_x = sigma * kappa * phi_x
            f_st_y = sigma * kappa * phi_y
            f_st_z = sigma * kappa * phi_z

            # ============================================================
            # 电润湿驱动力 (Electrowetting Force)
            # ============================================================
            # EWD 物理机制：
            # - 升压时，电润湿力降低接触角，油墨向边缘收缩
            # - 极性液体占据中心区域，形成透明开口
            # - 力作用在三相接触线，驱动油墨向外移动
            #
            # 力的大小: f_ew = ε₀εᵣV²/(2d) * δ(interface)
            # 力的方向: 沿界面法向，使油墨向外收缩
            #
            # 实现方式: 使用 CSF 类似的体积力形式
            # f_ew = f_ew_magnitude * interface_indicator * ∇phi / |∇phi|
            # ∇phi/|∇phi| 给出界面法向方向（从水指向油墨侧，向外）
            # 物理: 极性液体在电压下润湿基底，将油墨向外排挤

            # 提取电压信息 (V_from, V_to 在索引 3, 4)
            V_to = x[:, 4] if x.shape[1] >= 5 else torch.zeros_like(u)

            # 电润湿有效电压: 低于阈值无驱动力
            V_T = self.materials_params.get("V_T_base", 5.0)
            V_eff_ew = torch.clamp(V_to - V_T, min=0.0)

            # 电润湿力幅值: f_ew = ½·C_ew·V_eff² (双层串联电容)
            f_ew_magnitude = 0.5 * C_ew * V_eff_ew**2

            # 空间坐标
            x_coord = x[:, 0]
            y_coord = x[:, 1]
            z_coord = x[:, 2]

            # 电润湿力只在底面附近有效 (z < 2*h_ink)
            # 使用平滑的高度衰减函数
            h_ink = self.materials_params.get("ink_thickness", 3e-6)
            z_decay = torch.exp(-z_coord / (2 * h_ink))

            # 界面指示函数: 在 phi ≈ 0.5 处最大
            # 使用 4*phi*(1-phi) 作为界面 delta 函数的近似
            interface_indicator = 4 * phi * (1 - phi)

            # 电润湿力方向: 沿界面法向，从水(φ=0)指向油(φ=1)
            # 使用 +∇φ/|∇φ|: 正梯度方向指向 φ 增大方向（即从水到油，向外排油）
            grad_phi_xy_mag = torch.sqrt(phi_x**2 + phi_y**2 + 1e-12)
            dir_x = phi_x / grad_phi_xy_mag
            dir_y = phi_y / grad_phi_xy_mag

            # 电润湿体积力（作用在界面处，φ≈0.5）
            # interface_indicator 已确保力峰值在接触线，无需额外 ink_fraction
            f_ew_x = (
                f_ew_magnitude * interface_indicator * z_decay * dir_x
            )
            f_ew_y = (
                f_ew_magnitude * interface_indicator * z_decay * dir_y
            )
            f_ew_z = torch.zeros_like(f_ew_x)  # z 方向无电润湿力

            # 连续性方程
            continuity = u_x + v_y + w_z

            # 动量方程 (Re ≈ 1-5, Womersley ≈ 0.03 → 准稳态 Stokes)
            # 对流项默认关闭以减少训练噪声; 高电压快速响应时可开启
            use_convection = self.materials_params.get("use_convection", False)
            if use_convection:
                u_conv = u * u_x + v * u_y + w * u_z
                v_conv = u * v_x + v * v_y + w * v_z
                w_conv = u * w_x + v * w_y + w * w_z
            else:
                u_conv = torch.zeros_like(u)
                v_conv = torch.zeros_like(v)
                w_conv = torch.zeros_like(w)

            # 粘度梯度: μ = φ·μ_oil + (1-φ)·μ_polar
            # ∇μ = (μ_oil - μ_polar)·∇φ
            dmu_dphi = mu_oil - mu_polar  # 标量, 负值 (μ_oil < μ_polar)
            mu_x = dmu_dphi * phi_x
            mu_y = dmu_dphi * phi_y
            mu_z = dmu_dphi * phi_z

            # ∇·[μ(∇u + ∇uᵀ)] 的各分量
            viscous_u = (
                mu * lap_u
                + mu_x * (2 * u_x)
                + mu_y * (u_y + v_x)
                + mu_z * (u_z + w_x)
            )
            viscous_v = (
                mu * lap_v
                + mu_x * (v_x + u_y)
                + mu_y * (2 * v_y)
                + mu_z * (v_z + w_y)
            )
            viscous_w = (
                mu * lap_w
                + mu_x * (w_x + u_z)
                + mu_y * (w_y + v_z)
                + mu_z * (2 * w_z)
            )

            # N-S: ρ(∂u/∂t + u·∇u) = -∇p + ∇·[μ(∇u+∇uᵀ)] + f_st + f_ew
            momentum_u = rho * (u_t + u_conv) + p_x - viscous_u - f_st_x - f_ew_x
            momentum_v = rho * (v_t + v_conv) + p_y - viscous_v - f_st_y - f_ew_y
            momentum_w = rho * (w_t + w_conv) + p_z - viscous_w - f_st_z - f_ew_z

            return {
                "continuity": continuity,
                "momentum_u": momentum_u,
                "momentum_v": momentum_v,
                "momentum_w": momentum_w,
            }

        except Exception as e:
            logger.error(f"Navier-Stokes残差计算失败: {str(e)}")
            return self._empty_residual(x, predictions)

    def _compute_laplacian(self, scalar_field, coords, spatial_dims=3):
        """计算标量场的拉普拉斯算子"""
        try:
            laplacian = torch.zeros_like(scalar_field)

            # 计算一阶梯度
            grad_outputs = torch.ones_like(scalar_field)
            first_grad = torch.autograd.grad(
                outputs=scalar_field,
                inputs=coords,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if first_grad is None:
                return laplacian

            # 计算二阶梯度 (对角线元素之和)
            for i in range(min(spatial_dims, first_grad.shape[-1])):
                grad_i = first_grad[:, i]
                second_grad = torch.autograd.grad(
                    outputs=grad_i,
                    inputs=coords,
                    grad_outputs=torch.ones_like(grad_i),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

                if second_grad is not None and i < second_grad.shape[-1]:
                    laplacian = laplacian + second_grad[:, i]

            return laplacian

        except Exception:
            return torch.zeros_like(scalar_field)

    def compute_volume_conservation_residual(
        self, x_phys: torch.Tensor, predictions: torch.Tensor
    ):
        try:
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = (
                predictions.shape[0] if isinstance(predictions, torch.Tensor) else 1
            )

            residuals = {
                "volume_conservation": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "volume_consistency": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "ink_potential_min": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }

            if isinstance(predictions, torch.Tensor):
                if predictions.shape[1] >= 5:
                    alpha = predictions[:, 4]
                    alpha_clamped = torch.clamp(alpha, 0.0, 1.0)
                    base_consistency = alpha - alpha_clamped

                    ink_fraction_target = self.materials_params.get(
                        "ink_initial_fraction", 0.15
                    )
                    alpha_mean = torch.mean(alpha_clamped)
                    global_volume_residual = (alpha_mean - ink_fraction_target) / max(
                        ink_fraction_target, 1e-6
                    )
                    global_volume_residual_tensor = global_volume_residual.expand(
                        batch_size
                    )

                    overflow_penalty = torch.zeros(
                        batch_size, device=device, requires_grad=True
                    )
                    if (
                        isinstance(x_phys, torch.Tensor)
                        and x_phys.dim() == 2
                        and x_phys.size(1) >= 3
                    ):
                        coords = x_phys.detach()
                        x = coords[:, 0]
                        y = coords[:, 1]
                        z = coords[:, 2]

                        x_min, x_max = torch.min(x), torch.max(x)
                        y_min, y_max = torch.min(y), torch.max(y)

                        Lx = (x_max - x_min).clamp(min=1e-9)
                        Ly = (y_max - y_min).clamp(min=1e-9)

                        ink_thickness = self.materials_params.get("ink_thickness", 3e-6)
                        margin_x = 0.1 * Lx
                        margin_y = 0.1 * Ly

                        near_left = (x - x_min).abs() < margin_x
                        near_right = (x_max - x).abs() < margin_x
                        near_front = (y - y_min).abs() < margin_y
                        near_back = (y_max - y_min).abs() < margin_y

                        near_wall = near_left | near_right | near_front | near_back
                        wall_h = self.materials_params.get("wall_height", 3.5e-6)
                        above_wall = z > wall_h
                        overflow_mask = near_wall & above_wall

                        if overflow_mask.any():
                            overflow_alpha = alpha_clamped * overflow_mask.float()
                            overflow_penalty = overflow_alpha

                    residuals["volume_conservation"] = global_volume_residual_tensor
                    residuals["volume_consistency"] = (
                        base_consistency + overflow_penalty
                    )

            if isinstance(predictions, torch.Tensor) and predictions.shape[1] >= 6:
                ink_potential = predictions[:, 5]
                min_potential = self.materials_params.get("ink_potential_min", 0.0)
                residuals["ink_potential_min"] = torch.nn.functional.relu(
                    min_potential - ink_potential
                )

            return residuals

        except Exception as e:
            logger.error(f"计算体积守恒残差失败: {str(e)}")
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {
                "volume_conservation": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "volume_consistency": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "ink_potential_min": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }

    def compute_two_phase_flow_residual(
        self, x_phys: torch.Tensor, predictions: torch.Tensor
    ):
        """
        [DEPRECATED] 两相流 N-S 残差 — 未被训练管线调用。

        实际训练通过 compute_navier_stokes_residual() 处理所有 N-S 方程。
        此方法使用简化的粘性项和不同的连续性公式，与主线不一致。
        保留仅用于可能的对比研究，不要在生产训练中使用。
        """
        try:
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = (
                predictions.shape[0] if isinstance(predictions, torch.Tensor) else 1
            )

            # 初始化残差字典
            residuals = {
                "two_phase_continuity": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "two_phase_momentum_u": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "two_phase_momentum_v": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "two_phase_momentum_w": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }

            # 检查预测是否包含足够的分量
            if isinstance(predictions, torch.Tensor):
                # 假设体积分数是预测的第5个分量 (0=u, 1=v, 2=w, 3=p, 4=α)
                if predictions.shape[1] >= 5:
                    u = predictions[:, 0]
                    v = predictions[:, 1]
                    w = predictions[:, 2]
                    p = predictions[:, 3]
                    alpha = predictions[:, 4]

                    # 材料参数
                    rho_polar = self.materials_params["density_polar"]
                    rho_ink = self.materials_params["density_ink"]
                    mu_polar = self.materials_params["viscosity_polar"]
                    mu_ink = self.materials_params["viscosity_ink"]

                    # 混合密度和粘度（基于体积分数）
                    rho = alpha * rho_polar + (1 - alpha) * rho_ink
                    mu = alpha * mu_polar + (1 - alpha) * mu_ink

                    # 计算速度梯度
                    try:
                        grad_u = self.safe_compute_gradient(u, x_phys)
                        grad_v = self.safe_compute_gradient(v, x_phys)
                        grad_w = self.safe_compute_gradient(w, x_phys)
                        grad_p = self.safe_compute_gradient(p, x_phys)

                        # 连续性方程：∇·(ρu) = 0
                        continuity = rho * (grad_u[:, 0] + grad_v[:, 1] + grad_w[:, 2])
                        residuals["two_phase_continuity"] = continuity

                        # 动量方程：ρ(∂u/∂t + u·∇u) = -∇p + ∇·(μ∇u) + σκ∇α + g
                        # 简化计算，忽略时间导数项和重力项
                        momentum_u = (
                            rho
                            * (u * grad_u[:, 0] + v * grad_u[:, 1] + w * grad_u[:, 2])
                            - grad_p[:, 0]
                            + mu * self.safe_compute_laplacian(u, x_phys)
                        )
                        momentum_v = (
                            rho
                            * (u * grad_v[:, 0] + v * grad_v[:, 1] + w * grad_v[:, 2])
                            - grad_p[:, 1]
                            + mu * self.safe_compute_laplacian(v, x_phys)
                        )
                        momentum_w = (
                            rho
                            * (u * grad_w[:, 0] + v * grad_w[:, 1] + w * grad_w[:, 2])
                            - grad_p[:, 2]
                            + mu * self.safe_compute_laplacian(w, x_phys)
                        )

                        residuals["two_phase_momentum_u"] = momentum_u
                        residuals["two_phase_momentum_v"] = momentum_v
                        residuals["two_phase_momentum_w"] = momentum_w

                    except Exception as e:
                        logger.warning(f"双相流动量方程梯度计算失败: {str(e)}")

            return residuals

        except Exception as e:
            logger.error(f"计算双相流残差失败: {str(e)}")
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {
                "two_phase_continuity": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "two_phase_momentum_u": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "two_phase_momentum_v": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "two_phase_momentum_w": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }

    def compute_surface_tension_residual(
        self, x_phys: torch.Tensor, predictions: torch.Tensor
    ):
        """
        [DEPRECATED] 需要 predictions[:,5] (界面曲率) 但标准模型只输出5列。
        始终返回零张量。实际表面张力通过 N-S CSF 项和 compute_interface_energy_residual 处理。
        """
        try:
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = (
                predictions.shape[0] if isinstance(predictions, torch.Tensor) else 1
            )

            # 初始化残差字典
            residuals = {
                "surface_tension": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "contact_angle_constraint": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "interface_curvature": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }

            # 检查预测是否包含足够的分量
            if isinstance(predictions, torch.Tensor):
                # 假设体积分数是预测的第5个分量，界面曲率是第6个分量
                if predictions.shape[1] >= 6:
                    alpha = predictions[:, 4]  # 体积分数
                    interface_curvature = predictions[:, 5]  # 界面曲率

                    # 表面张力参数
                    sigma = self.materials_params["surface_tension_polar_ink"]
                    contact_angle_ink = self.materials_params.get(
                        "contact_angle_ink", 120.0
                    )

                    # 计算表面张力引起的力：σκ∇α
                    try:
                        grad_alpha = self.safe_compute_gradient(alpha, x_phys)
                        surface_tension_force = (
                            sigma * interface_curvature * torch.norm(grad_alpha, dim=1)
                        )
                        residuals["surface_tension"] = surface_tension_force
                    except Exception as e:
                        logger.warning(f"表面张力力计算失败: {str(e)}")

                    # 接触角约束：确保接触角在合理范围内
                    contact_angle_rad = torch.tensor(
                        np.radians(contact_angle_ink), device=device
                    )
                    # 简化的接触角约束：基于界面曲率
                    residuals["contact_angle_constraint"] = torch.nn.functional.relu(
                        torch.abs(interface_curvature)
                        - 1.0 / torch.tan(contact_angle_rad / 2.0)
                    )

                    # 界面曲率约束：确保界面平滑
                    residuals["interface_curvature"] = torch.nn.functional.relu(
                        torch.abs(interface_curvature) - 10.0  # 曲率上限
                    )

            return residuals

        except Exception as e:
            logger.error(f"计算表面张力残差失败: {str(e)}")
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {
                "surface_tension": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "contact_angle_constraint": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "interface_curvature": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }

    def compute_ink_potential_residual(
        self, x_phys: torch.Tensor, predictions: torch.Tensor
    ):
        """
        计算油墨势能最小化残差
        - 确保油墨始终以最小势能存在
        - 考虑表面张力和接触角变化
        """
        try:
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = (
                predictions.shape[0] if isinstance(predictions, torch.Tensor) else 1
            )

            # 初始化残差字典
            residuals = {
                "ink_potential_min": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "ink_energy_balance": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }

            # 检查预测是否包含足够的分量
            if isinstance(predictions, torch.Tensor):
                # 假设体积分数是第5个分量，油墨势能是第7个分量
                if predictions.shape[1] >= 7:
                    alpha = predictions[:, 4]  # 体积分数 (0=油墨, 1=极性液体)
                    ink_potential = predictions[:, 6]  # 油墨势能

                    # 材料参数
                    sigma = self.materials_params["surface_tension_polar_ink"]
                    min_potential = self.materials_params.get("ink_potential_min", 0.0)

                    # 油墨势能最小化约束：确保油墨势能不大于当前值
                    residuals["ink_potential_min"] = torch.nn.functional.relu(
                        ink_potential - min_potential
                    )

                    # 油墨能量平衡：考虑表面张力和接触角变化
                    # 简化计算：油墨能量与表面积和接触角相关
                    try:
                        grad_alpha = self.safe_compute_gradient(alpha, x_phys)
                        interface_area = torch.norm(grad_alpha, dim=1)
                        ink_energy = sigma * interface_area * (1 - alpha)
                        residuals["ink_energy_balance"] = ink_energy - ink_potential
                    except Exception as e:
                        logger.warning(f"油墨能量平衡计算失败: {str(e)}")

            return residuals

        except Exception as e:
            logger.error(f"计算油墨势能残差失败: {str(e)}")
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {
                "ink_potential_min": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "ink_energy_balance": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }

    def compute_sidewall_contact_angle_residual(
        self, x_phys: torch.Tensor, predictions: torch.Tensor
    ):
        """
        壁面接触角约束: n̂·n̂_wall = cos(θ_wall)

        物理: 极性液体在像素壁上的接触角 θ_wall=71°。
        在壁面处，油墨-极性液体界面的法向 n̂ = ∇φ/|∇φ| 与壁面法向 n̂_wall
        的夹角必须等于 θ_wall。此约束打破 PINN 的 1D 退化解，
        迫使界面在壁面处倾斜（油墨沿壁面爬升）。

        Args:
            x_phys: (batch, 6) 物理点坐标
            predictions: (batch, 5) 模型预测 (u,v,w,p,phi)
        """
        try:
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            zero = torch.zeros(batch_size, device=device, requires_grad=True)
            residuals = {"sidewall_contact_angle": zero}

            if not (
                isinstance(x_phys, torch.Tensor)
                and isinstance(predictions, torch.Tensor)
            ):
                return residuals
            if x_phys.dim() != 2 or x_phys.size(1) < 3:
                return residuals
            if predictions.dim() < 2 or predictions.size(1) < 5:
                return residuals

            phi = predictions[:, 4]
            coords = x_phys.detach()
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

            # 域边界
            Lx = self.materials_params.get("Lx", 174e-6)
            Ly = self.materials_params.get("Ly", 174e-6)
            margin = 0.08 * min(Lx, Ly)

            # 壁面检测
            near_left = x < margin
            near_right = x > Lx - margin
            near_front = y < margin
            near_back = y > Ly - margin
            near_wall = near_left | near_right | near_front | near_back

            # 仅界面附近 (φ≈0.5) 有效
            interface_weight = torch.exp(-100.0 * (phi - 0.5) ** 2)
            mask = near_wall & (interface_weight > 0.01)

            if mask.sum() < 3:
                return residuals

            # 对 x_phys 求 ∇φ（x_phys 在 compute_all_residuals 中已设 requires_grad=True）
            try:
                grad_all = torch.autograd.grad(
                    phi.sum(), x_phys, create_graph=True, retain_graph=True
                )[0]
            except Exception:
                return residuals

            if grad_all is None:
                return residuals

            grad_x = grad_all[:, 0]
            grad_y = grad_all[:, 1]
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_all[:, 2] ** 2) + 1e-10

            # 壁面法向分量
            n_x = grad_x / grad_mag
            n_y = grad_y / grad_mag

            # 壁面法向: x=0 → (1,0), x=Lx → (-1,0), y=0 → (0,1), y=Ly → (0,-1)
            wall_nx = torch.where(
                x < margin,
                torch.ones_like(n_x),
                torch.where(
                    x > Lx - margin, -torch.ones_like(n_x), torch.zeros_like(n_x)
                ),
            )
            wall_ny = torch.where(
                y < margin,
                torch.ones_like(n_y),
                torch.where(
                    y > Ly - margin, -torch.ones_like(n_y), torch.zeros_like(n_y)
                ),
            )

            dot = n_x * wall_nx + n_y * wall_ny

            # 接触角滞后 (CAH): 根据壁面法向速度选择前进/后退角
            # u·n̂_wall > 0 → 油墨被推开 → 前进角 θ_A
            # u·n̂_wall < 0 → 油墨拉回   → 后退角 θ_R
            u_vel = predictions[:, 0]
            v_vel = predictions[:, 1]
            u_dot_n = u_vel * wall_nx + v_vel * wall_ny

            theta_wall = self.materials_params.get("theta_wall", 71.0)
            delta_cah = self.materials_params.get("cah_hysteresis", 4.0)
            theta_A = theta_wall + delta_cah  # 75°
            theta_R = theta_wall - delta_cah  # 67°

            cos_A = np.cos(np.radians(theta_A))
            cos_R = np.cos(np.radians(theta_R))
            cos_eq = np.cos(np.radians(theta_wall))

            # 死区: |u·n̂| 很小时用平衡角，避免零速度时的错误分类
            deadband = 1e-4
            target_cos = torch.where(
                u_dot_n > deadband,
                torch.full_like(dot, cos_A),
                torch.where(
                    u_dot_n < -deadband,
                    torch.full_like(dot, cos_R),
                    torch.full_like(dot, cos_eq),
                ),
            )

            # 壁面底面边缘钉扎: z≈0 处的接触线被锐缘钉扎
            # 钉扎条件: |cos_θ_local - cos_θ_eq| < H_pin → 接触线不可移动
            is_bottom_edge = (z < 1e-6) & near_wall
            if is_bottom_edge.any():
                cos_diff = torch.abs(dot - target_cos)
                H_pin = 0.15  # 钉扎强度阈值
                pinning_factor = torch.where(
                    cos_diff < H_pin,
                    torch.ones_like(cos_diff) * 5.0,  # 钉扎: 加权惩罚
                    torch.ones_like(cos_diff),  # 突破: 正常权重
                )
                edge_weight = pinning_factor * interface_weight
                # 边缘点单独加权
                base_residual = (
                    ((dot - target_cos) ** 2) * mask.float() * interface_weight
                )
                edge_residual = (
                    ((dot - target_cos) ** 2) * is_bottom_edge.float() * edge_weight
                )
                residual = base_residual + edge_residual * 0.5  # 0.5 避免边角主导
            else:
                residual = ((dot - target_cos) ** 2) * mask.float() * interface_weight

            n_active = mask.float().sum().clamp(min=1)
            residuals["sidewall_contact_angle"] = residual.sum() / n_active

            return residuals
        except Exception as e:
            logger.warning(f"壁面接触角残差计算失败: {e}")
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {
                "sidewall_contact_angle": torch.zeros(
                    batch_size, device=device, requires_grad=True
                )
            }

    def compute_laplace_pressure_residual(
        self, x_phys: torch.Tensor, predictions: torch.Tensor
    ):
        """
        Laplace 压力一致约束: 沿界面 κ = 常数

        物理: ΔP = σ·κ 在界面各处相等。约束界面曲率的方差趋近于零，
        迫使 PINN 产生物理正确的 3D 界面形状（中心开口、壁面堆积）。
        此约束与壁面接触角约束联立，唯一确定 3D 界面形状。

        计算: κ_xy = -(φ_xx + φ_yy)/|∇φ|  (xy 平面曲率分量)
              沿界面采样点计算 κ_xy 的方差作为 loss

        Args:
            x_phys: (batch, 6) 物理点坐标
            predictions: (batch, 5) 模型预测
        """
        try:
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {"laplace_pressure": zero}

            if not (
                isinstance(x_phys, torch.Tensor)
                and isinstance(predictions, torch.Tensor)
            ):
                return res
            if x_phys.dim() != 2 or x_phys.size(1) < 3:
                return res
            if predictions.dim() < 2 or predictions.size(1) < 5:
                return res

            phi = predictions[:, 4]

            # 只在界面区域 (φ∈[0.3, 0.7]) 计算曲率
            interface_mask = (phi > 0.3) & (phi < 0.7)
            if interface_mask.sum() < 10:
                return res

            # 计算 ∇φ (一阶导)
            try:
                grad_phi = torch.autograd.grad(
                    phi.sum(), x_phys, create_graph=True, retain_graph=True
                )[0]
            except Exception:
                return res

            if grad_phi is None:
                return res

            phi_x = grad_phi[:, 0]
            phi_y = grad_phi[:, 1]
            grad_mag_sq = phi_x**2 + phi_y**2 + grad_phi[:, 2] ** 2 + 1e-10
            grad_mag = torch.sqrt(grad_mag_sq)

            # 计算 φ_xx + φ_yy (xy Laplacian，通过二阶 autograd)
            try:
                laplacian_xy = torch.zeros_like(phi_x)
                lap_x = torch.autograd.grad(
                    phi_x.sum(), x_phys, create_graph=True, retain_graph=True
                )[0][:, 0]
                lap_y = torch.autograd.grad(
                    phi_y.sum(), x_phys, create_graph=True, retain_graph=True
                )[0][:, 1]
                laplacian_xy = lap_x + lap_y
            except Exception:
                return res

            # κ_xy = -(φ_xx + φ_yy) / |∇φ|
            kappa_xy = -laplacian_xy / (grad_mag + 1e-10)

            # 沿界面 κ 的方差 → 驱动力使 κ 恒定
            k_if = kappa_xy[interface_mask]
            if k_if.numel() < 5:
                return res

            kappa_mean = k_if.mean()
            laplace_loss = ((k_if - kappa_mean) ** 2).mean()

            res["laplace_pressure"] = laplace_loss.unsqueeze(0)
            return res
        except Exception as e:
            logger.warning(f"Laplace 压力残差计算失败: {e}")
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            return {
                "laplace_pressure": torch.zeros(1, device=device, requires_grad=True)
            }

    def compute_electrowetting_residual(self, x_phys, predictions):
        """电润湿变分残差 — z=0底面能量梯度

        自由能: G_ew(φ) = -½·C(η)·V², η = 1-φ (开口率)
        变分导数: δG_ew/δφ = ½·V²·(C_open - C_ink_region) > 0
        (常数驱动力, 推动 φ→0 即油墨从底面清除)

        平衡条件 (Young-Lippmann):
          σ·κ_xy + ½·(C_open - C_ink_region)·V² = 0  @ 接触线
        即毛细压力 + 电润湿压力 = 0

        残差: |σ·κ_xy + p_ew| × δ(interface) 沿底面接触线
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {'electrowetting': zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            phi = predictions[:, 4]
            z = x_phys[:, 2]; V_to = x_phys[:, 4]
            is_bottom = z < 1e-6
            if not is_bottom.any():
                return res

            # 材料参数
            eps0 = 8.854e-12
            eps_r = self.materials_params.get('relative_permittivity', 3.28)
            eps_h = self.materials_params.get('epsilon_hydrophobic', 1.934)
            eps_ink = 4.0; h_ink = self.materials_params.get('ink_thickness', 3e-6)
            d_d = self.materials_params.get('dielectric_thickness', 4e-7)
            d_h = self.materials_params.get('hydrophobic_thickness', 4e-7)
            sigma = self.materials_params.get('surface_tension_polar_ink', 0.02505)

            C_open = 1.0/(d_d/(eps0*eps_r) + d_h/(eps0*eps_h))
            C_ink  = eps0*eps_ink/h_ink
            C_ink_region = 1.0/(1.0/C_open + 1.0/C_ink)

            V_T = self.materials_params.get('V_T_base', 5.0)
            V_eff = torch.clamp(V_to - V_T, min=0.0)

            # 电润湿压力 (常数, 不依赖φ): p_ew = ½·ΔC·V²
            delta_C = C_open - C_ink_region  # > 0
            p_ew = 0.5 * delta_C * V_eff**2  # [J/m² = N/m = Pa·m]

            # 计算底面接触线处的毛细压力: σ·κ_xy
            # 使用梯度计算
            try:
                grad_phi = torch.autograd.grad(
                    phi.sum(), x_phys, create_graph=True, retain_graph=True
                )[0]
            except Exception:
                return res
            if grad_phi is None:
                return res

            phi_x, phi_y = grad_phi[:, 0], grad_phi[:, 1]
            grad_mag = torch.sqrt(phi_x**2 + phi_y**2 + grad_phi[:, 2]**2) + 1e-10

            # 界面指示函数 (φ≈0.5)
            interface_weight = torch.exp(-100.0 * (phi - 0.5)**2)

            # 对于接触线, 能量平衡: σ·κ_xy + p_ew = 0
            # κ_xy ≈ -(φ_xx + φ_yy) / |∇φ|
            # 简化: 用 |∇φ| 加权, 只在界面处有效
            # residual = |∇φ| × (σ·κ_xy + p_ew) × interface_weight

            # 为降低计算成本, 用 φ 的梯度信息替代 κ_xy
            # 在平衡态: φ 的分布由 σ·κ_xy + p_ew = 0 决定
            # 非平衡态: 残差驱动 φ 调整
            phi_b = phi[is_bottom]; V_b = V_eff[is_bottom]
            interface_b = interface_weight[is_bottom]

            # 电润湿驱动: 惩罚底面存在油墨 (φ→1)
            # 变分一致形式: 能量梯度 × interface_indicator
            # δG_ew/δφ = p_ew (常数, 驱动 φ→0)
            # 实际驱动力需克服表面张力, 使用平滑形式
            driving = p_ew[is_bottom] * phi_b * interface_b

            n_bottom = is_bottom.float().sum().clamp(min=1)
            res['electrowetting'] = driving.sum() / n_bottom
            return res
        except Exception as e:
            logger.warning(f'电润湿残差失败: {e}')
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            return {'electrowetting': torch.zeros(1, device=device, requires_grad=True)}

    def compute_interface_energy_residual(self, x_phys, predictions):
        """界面能 — 纯 sigma*|grad(phi)|, 塑造圆润液滴

        电润湿项已拆分到 compute_electrowetting_residual。
        此项只做表面张力最小化(最小界面面积 = 圆润形状)。
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            zero = torch.zeros(x_phys.shape[0], device=device, requires_grad=True)
            res = {'interface_energy': zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            phi = predictions[:, 4]
            try:
                grad_phi = torch.autograd.grad(phi.sum(), x_phys, create_graph=True, retain_graph=True)[0]
            except Exception:
                return res
            if grad_phi is None:
                return res
            sigma = self.materials_params.get('surface_tension_polar_ink', 0.02505)
            grad_mag = torch.norm(grad_phi[:, :3], dim=1)
            res['interface_energy'] = sigma * grad_mag
            return res
        except Exception as e:
            logger.warning(f'界面能失败: {e}')
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            return {'interface_energy': torch.zeros(x_phys.shape[0], device=device, requires_grad=True)}

    def compute_wall_wetting_residual(
        self, x_phys: torch.Tensor, predictions: torch.Tensor
    ):
        try:
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = (
                predictions.shape[0] if isinstance(predictions, torch.Tensor) else 1
            )
            residuals = {
                "wall_wetting": torch.zeros(
                    batch_size, device=device, requires_grad=True
                )
            }
            if not (
                isinstance(x_phys, torch.Tensor)
                and isinstance(predictions, torch.Tensor)
            ):
                return residuals
            if x_phys.dim() != 2 or x_phys.size(1) < 3:
                return residuals
            if predictions.dim() < 2 or predictions.size(1) < 5:
                return residuals
            alpha = predictions[:, 4]
            alpha_clamped = torch.clamp(alpha, 0.0, 1.0)
            coords = x_phys.detach()
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            x_min, x_max = torch.min(x), torch.max(x)
            y_min, y_max = torch.min(y), torch.max(y)
            Lx = (x_max - x_min).clamp(min=1e-9)
            Ly = (y_max - y_min).clamp(min=1e-9)
            margin_x = 0.1 * Lx
            margin_y = 0.1 * Ly
            near_left = (x - x_min).abs() < margin_x
            near_right = (x_max - x).abs() < margin_x
            near_front = (y - y_min).abs() < margin_y
            near_back = (y_max - y).abs() < margin_y
            near_wall = near_left | near_right | near_front | near_back
            domain_height = self.materials_params.get("domain_height", 20e-6)
            dh = max(domain_height, 1e-9)
            z_norm = (z / dh).clamp(0.0, 1.0)
            penalty = alpha_clamped * z_norm * near_wall.float()
            residuals["wall_wetting"] = penalty
            return residuals
        except Exception as e:
            logger.error(f"计算墙润湿残差失败: {str(e)}")
            device = (
                x_phys.device
                if isinstance(x_phys, torch.Tensor)
                else torch.device("cpu")
            )
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {
                "wall_wetting": torch.zeros(
                    batch_size, device=device, requires_grad=True
                )
            }

    def _compute_dielectric_charge_residual(self, x_phys, predictions):
        """介电层RC充电约束 — z=0底面

        物理: 介电层电荷积累遵循RC电路模型
          σ(t) = C_dielectric × V × (1 - exp(-t/τ_RC))
          电润湿力在介电层充电完成后才完全建立
          τ_RC = ε₀ε_r / σ_conduct ≈ 25μs (远快于机械响应)

        约束: 在t < 3τ_RC的早期时刻，底面电润湿能应遵循充电曲线
          G_ew(t) = G_ew_steady × (1 - exp(-t/τ_RC))²
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {'dielectric_charge': zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            if x_phys.shape[1] < 6:
                return res
            phi = predictions[:, 4]
            z = x_phys[:, 2]; V_to = x_phys[:, 4]; t_since = x_phys[:, 5]
            is_bottom = z < 1e-6
            if not is_bottom.any():
                return res

            phi_b = phi[is_bottom]; V_b = V_to[is_bottom]; t_b = t_since[is_bottom]
            V_T = self.materials_params.get('V_T_base', 5.0)
            V_eff = torch.clamp(V_b - V_T, min=0.0)

            # τ_RC: 介电层RC时间常数 (SU-8电阻率≈10¹⁴Ω·cm → τ_RC≈25μs)
            tau_rc = self.materials_params.get('charge_relaxation_time', 2.5e-5)

            # RC充电曲线: (1-exp(-t/τ_RC))²
            charge_factor = (1.0 - torch.exp(-t_b / tau_rc)) ** 2
            # 稳态电润湿能: ½·C·V²
            eps0 = 8.854e-12
            eps_r = self.materials_params.get('relative_permittivity', 3.28)
            eps_h = self.materials_params.get('epsilon_hydrophobic', 1.934)
            d_d = self.materials_params.get('dielectric_thickness', 4e-7)
            d_h = self.materials_params.get('hydrophobic_thickness', 4e-7)
            C_ew = 1.0/(d_d/(eps0*eps_r) + d_h/(eps0*eps_h))
            G_steady = 0.5 * C_ew * V_eff**2
            G_expected = G_steady * charge_factor

            # 约束: φ_b × G_steady (电润湿推动力) 不应超过 RC 充电水平
            # 即: φ_b × G_steady ≤ (1 - φ_b) × G_expected
            # 残差: relu(φ_b × G_steady - (1 - φ_b) × G_expected)
            ew_driving = phi_b * G_steady
            ew_available = (1.0 - phi_b) * G_expected
            residual = torch.relu(ew_driving - ew_available)

            n_bottom = is_bottom.float().sum().clamp(min=1)
            res['dielectric_charge'] = residual.sum() / n_bottom
            return res
        except Exception as e:
            logger.warning(f'介电电荷残差失败: {e}')
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            return {'dielectric_charge': torch.zeros(1, device=device, requires_grad=True)}

    def _compute_contact_line_dynamics_residual(self, x_phys, predictions):
        """接触线动力学约束 — Hoffman-Voinov-Tanner 模型适配

        物理: 接触线滑移速度与接触角偏差的关系
          v_cl = k_cl × (cos(θ_eq) - cos(θ_local)) × sign(cos(θ_eq) - cos(θ_local))

        在相场中: v_cl ≈ (∂φ/∂t) / |∇φ|  (在 φ≈0.5, z=0 处)
        接触角信息: cos(θ_local) = (∂φ/∂z) / |∇φ|  (在底面)

        Young-Laplace 决定界面平衡形状:
          ΔP = σ·κ = const (由 Laplace 压力约束保证)
          接触线处的 θ_local 需满足 cos(θ_eq) = cos(θ₀) + C×V²/(2σ)
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {'contact_line_dynamics': zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            if x_phys.shape[1] < 6:
                return res
            phi = predictions[:, 4]
            z = x_phys[:, 2]; V_to = x_phys[:, 4]
            # 接触线区域: z=0 且 φ≈0.5
            is_bottom = z < 1e-6
            is_interface = (phi > 0.2) & (phi < 0.8)
            mask = is_bottom & is_interface
            if mask.sum() < 3:
                return res

            # 计算 ∇φ (需梯度)
            try:
                grad_phi = torch.autograd.grad(
                    phi.sum(), x_phys, create_graph=True, retain_graph=True
                )[0]
            except Exception:
                return res
            if grad_phi is None:
                return res

            phi_t = grad_phi[:, 5] if grad_phi.shape[1] >= 6 else torch.zeros_like(phi)
            phi_z = grad_phi[:, 2]
            grad_mag = torch.norm(grad_phi[:, :3], dim=1) + 1e-10

            v_cl = phi_t[mask] / grad_mag[mask]
            cos_local = phi_z[mask] / grad_mag[mask]

            # cos(θ_eq) from Young-Lippmann
            theta0_deg = self.materials_params.get('contact_angle_theta0', 120.0)
            eps0 = 8.854e-12
            eps_r = self.materials_params.get('relative_permittivity', 3.28)
            eps_h = self.materials_params.get('epsilon_hydrophobic', 1.934)
            sigma_po = self.materials_params.get('surface_tension_polar_ink',
                        self.materials_params.get('sigma', 0.02505))
            d_d = self.materials_params.get('dielectric_thickness', 4e-7)
            d_h = self.materials_params.get('hydrophobic_thickness', 4e-7)
            C_yl = 1.0/(d_d/(eps0*eps_r) + d_h/(eps0*eps_h))

            cos_theta0 = np.cos(np.radians(theta0_deg))
            V_m = V_to[mask]; V_T = self.materials_params.get('V_T_base', 5.0)
            V_eff = torch.clamp(V_m - V_T, min=0.0)
            ew_term = C_yl * V_eff**2 / (2 * sigma_po)
            cos_eq = torch.clamp(
                torch.tensor(cos_theta0, device=device) + ew_term, -1.0, 1.0
            )

            # HVT: v_cl ∝ cos_eq - cos_local
            k_cl = self.materials_params.get('contact_line_friction', 1e-3)
            residual = v_cl - k_cl * (cos_eq - cos_local)

            res['contact_line_dynamics'] = torch.mean(residual**2)
            return res
        except Exception as e:
            logger.warning(f'接触线动力学残差失败: {e}')
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            return {'contact_line_dynamics': torch.zeros(1, device=device, requires_grad=True)}

    def _compute_top_boundary_residual(self, x_phys, predictions):
        """顶面自由表面边界条件 (z≈Lz)

        物理: 极性液体-气界面
        - 零剪切: du/dz≈0, dv/dz≈0 (无表面应力)
        - 无穿透: w=0 (界面不移动)
        - φ=0 (纯极性液体在顶部, 油墨在下层)
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {'top_boundary': zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            if x_phys.shape[1] < 3:
                return res
            Lz = self.materials_params.get('domain_height', 20e-6)
            z = x_phys[:, 2]
            is_top = z > (Lz - 1e-6)
            if not is_top.any():
                return res

            u_t = predictions[is_top, 0]
            v_t = predictions[is_top, 1]
            w_t = predictions[is_top, 2]
            phi_t = predictions[is_top, 4]

            try:
                grad_u = torch.autograd.grad(
                    u_t.sum(), x_phys, create_graph=True, retain_graph=True
                )[0]
                grad_v = torch.autograd.grad(
                    v_t.sum(), x_phys, create_graph=True, retain_graph=True
                )[0]
            except Exception:
                return res
            if grad_u is None or grad_v is None:
                return res

            du_dz = grad_u[is_top, 2]
            dv_dz = grad_v[is_top, 2]

            shear = (du_dz**2 + dv_dz**2).mean()
            normal = (w_t**2).mean()
            phi_ok = (phi_t**2).mean()

            res['top_boundary'] = shear + normal + 0.5 * phi_ok
            return res
        except Exception as e:
            logger.warning(f'顶面BC失败: {e}')
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device('cpu')
            return {'top_boundary': torch.zeros(1, device=device, requires_grad=True)}

    def safe_compute_laplacian_spatial(
        self, scalar_field: torch.Tensor, coords: torch.Tensor, spatial_dims: int = 3
    ):
        try:
            grad = self.safe_compute_gradient(scalar_field, coords)
            lap = torch.zeros_like(scalar_field)
            dims = min(spatial_dims, coords.shape[-1])
            for i in range(dims):
                gi = grad[..., i]
                g2 = torch.autograd.grad(
                    outputs=gi,
                    inputs=coords,
                    grad_outputs=torch.ones_like(gi),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if g2 is not None:
                    lap = lap + g2[..., i]
            return lap
        except Exception:
            return torch.zeros_like(scalar_field)

    def safe_compute_gradient(self, output: torch.Tensor, input_tensor: torch.Tensor):
        try:
            if not isinstance(output, torch.Tensor) or not isinstance(
                input_tensor, torch.Tensor
            ):
                return torch.zeros_like(input_tensor)
            if not input_tensor.requires_grad:
                input_tensor = input_tensor.clone().requires_grad_(True)
            grad_outputs = torch.ones_like(output)
            grad = torch.autograd.grad(
                outputs=output,
                inputs=input_tensor,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros(
                    (*input_tensor.shape[:-1], input_tensor.shape[-1]),
                    device=input_tensor.device,
                )
            return grad
        except Exception:
            return torch.zeros(
                (*input_tensor.shape[:-1], input_tensor.shape[-1]),
                device=input_tensor.device,
            )

    def compute_core_residuals(
        self,
        x_phys: torch.Tensor,
        predictions: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        统一物理残差计算入口 - 作为"唯一物理真相"

        此方法整合所有核心物理方程残差计算，供 PhysicsLoss 等适配层调用。

        Args:
            x_phys: 物理点坐标 (batch, 6) - (x, y, z, V_from, V_to, t_since)
            predictions: 模型预测 (batch, 5+) - (u, v, w, p, phi, ...)
            model: 可选的模型引用（用于需要重新前向传播的情况）

        Returns:
            残差字典，包含：
            - continuity: 连续性方程残差 ∇·u = 0
            - momentum_u/v/w: N-S 动量方程残差
            - vof: VOF 方程残差 ∂φ/∂t + u·∇φ = 0
            - surface_tension: 表面张力相关残差
            - volume_conservation: 体积守恒残差
            - temporal_smoothness: 时间连续性残差（流场平滑变化）
        """
        device = (
            x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
        )
        batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1

        residuals = {}

        # 1. Navier-Stokes 残差（包含连续性和动量方程）
        try:
            ns_residuals = self.compute_navier_stokes_residual(
                x_phys, predictions, model=model
            )
            residuals.update(ns_residuals)
        except Exception as e:
            logger.warning(f"N-S 残差计算失败: {e}")
            residuals.update(self._empty_residual(x_phys, predictions))

        # 2. VOF 方程残差（体积分数输运）
        try:
            vof_residual = self._compute_vof_residual(x_phys, predictions, model=model)
            residuals["vof"] = vof_residual
        except Exception as e:
            logger.warning(f"VOF 残差计算失败: {e}")
            residuals["vof"] = torch.zeros(batch_size, device=device)

        # 3. 表面张力 — 已通过 N-S CSF 项和 interface_energy 处理，无需单独调用
        # compute_surface_tension_residual 需要 predictions[:,5] (曲率)，标准模型不提供此输出

        # 4. 体积守恒残差
        try:
            vc_residuals = self.compute_volume_conservation_residual(
                x_phys, predictions
            )
            residuals.update(vc_residuals)
        except Exception as e:
            logger.warning(f"体积守恒残差计算失败: {e}")

        # 5. 电润湿驱动力 (z=0逐点, 驱动开口)
        try:
            ew_residuals = self.compute_electrowetting_residual(x_phys, predictions)
            residuals.update(ew_residuals)
        except Exception as e:
            logger.warning(f"电润湿残差失败: {e}")

        # 6. 界面能 (纯 sigma*|grad(phi)|, 塑造液滴)
        try:
            ie_residuals = self.compute_interface_energy_residual(x_phys, predictions)
            residuals.update(ie_residuals)
        except Exception as e:
            logger.warning(f"界面能约束计算失败: {e}")

        # 7. Laplace 压力一致约束（界面曲率恒定）
        try:
            lp_residuals = self.compute_laplace_pressure_residual(x_phys, predictions)
            residuals.update(lp_residuals)
        except Exception as e:
            logger.warning(f"Laplace 压力残差计算失败: {e}")

        # 6. 壁面接触角约束（打破 1D 退化解）
        try:
            sw_residuals = self.compute_sidewall_contact_angle_residual(
                x_phys, predictions
            )
            residuals.update(sw_residuals)
        except Exception as e:
            logger.warning(f"壁面接触角残差计算失败: {e}")

        # 7. 时间连续性正则化（流场平滑变化）
        try:
            temporal_residual = self._compute_temporal_smoothness(
                x_phys, predictions, model=model
            )
            residuals["temporal_smoothness"] = temporal_residual
        except Exception as e:
            logger.warning(f"时间连续性残差计算失败: {e}")
            residuals["temporal_smoothness"] = torch.zeros(batch_size, device=device)

        # 8. 壁面润湿约束（油墨不应沿壁面爬升）
        try:
            ww_residuals = self.compute_wall_wetting_residual(x_phys, predictions)
            residuals.update(ww_residuals)
        except Exception as e:
            logger.warning(f"壁面润湿残差计算失败: {e}")

        # 9. 介电层RC充电约束 (z=0, 电荷积累动态)
        try:
            dc_residuals = self._compute_dielectric_charge_residual(x_phys, predictions)
            residuals.update(dc_residuals)
        except Exception as e:
            logger.warning(f"介电电荷残差计算失败: {e}")

        # 10. 接触线动力学约束 (Hoffman-Voinov-Tanner)
        try:
            cld_residuals = self._compute_contact_line_dynamics_residual(
                x_phys, predictions
            )
            residuals.update(cld_residuals)
        except Exception as e:
            logger.warning(f"接触线动力学残差计算失败: {e}")

        # 11. 顶面自由表面边界条件 (z=Lz, 零剪切 + 无穿透)
        try:
            tbc_residuals = self._compute_top_boundary_residual(x_phys, predictions)
            residuals.update(tbc_residuals)
        except Exception as e:
            logger.warning(f"顶面BC残差计算失败: {e}")

        # 12. 压力钉扎 (封闭域中压力仅确定到常数, 需参考点)
        try:
            p = predictions[:, 3] if predictions.shape[1] >= 4 else None
            if p is not None:
                residuals["pressure_pin"] = torch.mean(p).unsqueeze(0)
        except Exception:
            pass

        return residuals

    def _compute_temporal_smoothness(
        self,
        x_phys: torch.Tensor,
        predictions: torch.Tensor,
        model: Optional[nn.Module] = None,
        dt: float = 0.001,  # 1ms 时间步长
    ) -> torch.Tensor:
        """
        计算时间平滑性正则化残差 (基于二阶导数/加速度)

        改进：
        原版约束 ||f(t+dt) - f(t)||² 实际上是惩罚速度，导致模型倾向于静止。
        新版约束 ||f(t+dt) - 2f(t) + f(t-dt)||² 惩罚加速度（曲率），
        允许模型线性变化，但禁止突变，从而消除"PINN T"的剧烈波动。

        Args:
            x_phys: 物理点坐标 (batch, 6) - (x, y, z, V_from, V_to, t_since)
            predictions: 当前时刻的模型预测 (batch, 5)
            model: 模型引用
            dt: 时间步长 (s)

        Returns:
            时间平滑性残差 (batch,)
        """
        device = (
            x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
        )
        batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1

        # 如果没有模型，无法计算时间连续性
        if model is None:
            return torch.zeros(batch_size, device=device)

        try:
            if predictions is None:
                with torch.enable_grad():
                    predictions = model(x_phys)

            # 当前时刻 t 的预测
            u_t = predictions[:, 0]
            v_t = predictions[:, 1]
            w_t = predictions[:, 2]
            phi_t = (
                predictions[:, 4]
                if predictions.shape[1] >= 5
                else torch.zeros_like(u_t)
            )

            # 构造 t+dt 和 t-dt 时刻的输入
            # 注意：t_since 必须 >= 0。对于 t < dt 的点，我们需要特殊处理
            # 简单策略：如果 t < dt，则只计算向前的一阶差分（退化为速度惩罚，但仅限起始点）
            t_since = x_phys[:, 5]

            x_next = x_phys.clone()
            x_next[:, 5] = t_since + dt

            x_prev = x_phys.clone()
            x_prev[:, 5] = torch.clamp(t_since - dt, min=0.0)

            # 计算 t+dt 和 t-dt 时刻的预测
            with torch.enable_grad():
                pred_next = model(x_next)
                pred_prev = model(x_prev)

            # t+dt
            u_next, v_next, w_next = pred_next[:, 0], pred_next[:, 1], pred_next[:, 2]
            phi_next = (
                pred_next[:, 4] if pred_next.shape[1] >= 5 else torch.zeros_like(u_next)
            )

            # t-dt
            u_prev, v_prev, w_prev = pred_prev[:, 0], pred_prev[:, 1], pred_prev[:, 2]
            phi_prev = (
                pred_prev[:, 4] if pred_prev.shape[1] >= 5 else torch.zeros_like(u_prev)
            )

            # 二阶差分 (近似加速度/曲率): f(t+dt) - 2f(t) + f(t-dt)
            # 对于 t < dt 的点 (x_prev ≈ x_t)，公式退化为 f(t+dt) - f(t)，即一阶差分
            # 这是合理的，因为起始时刻确实不应该有剧烈速度

            # 速度场平滑性
            acc_u = (u_next - 2 * u_t + u_prev) ** 2
            acc_v = (v_next - 2 * v_t + v_prev) ** 2
            acc_w = (w_next - 2 * w_t + w_prev) ** 2
            vel_smoothness = acc_u + acc_v + acc_w

            # φ 场平滑性
            # φ 允许快速变化，但加速度不应过大
            phi_smoothness = (phi_next - 2 * phi_t + phi_prev) ** 2 * 50.0

            return vel_smoothness + phi_smoothness

        except Exception as e:
            logger.warning(f"时间连续性计算异常: {e}")
            return torch.zeros(batch_size, device=device)

    def _compute_vof_residual(
        self,
        x_phys: torch.Tensor,
        predictions: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        计算 Allen-Cahn 相场方程残差 (标准化相场模型)

        标准形式: ∂φ/∂t + u·∇φ = M [ σ·ε·∇²φ − (σ/ε)·W'(φ) ]

        - ε: 界面宽度 (m), 默认 5μm
        - σ: 界面张力 (N/m), 使用 surface_tension_polar_ink
        - M: 迁移率 (m³·s/kg), 控制界面移动速度
        - W'(φ) = 2φ(1-φ)(1-2φ): 双阱势导数

        当 use_legacy_ac=True 时退回原有公式:
          ∂φ/∂t + u·∇φ = γ [ ∇²φ − W'(φ) ]

        Args:
            x_phys: (batch, 6) — (x, y, z, V_from, V_to, t_since)
            predictions: (batch, 5+) — (u, v, w, p, phi, ...)
        """
        device = (
            x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
        )
        batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1

        try:
            if not isinstance(x_phys, torch.Tensor):
                return torch.zeros(batch_size, device=device)

            if not x_phys.requires_grad:
                x_phys = x_phys.clone().detach().requires_grad_(True)

            if model is not None:
                predictions = model(x_phys)

            u, v, w = predictions[:, 0], predictions[:, 1], predictions[:, 2]
            phi = (
                predictions[:, 4] if predictions.shape[1] >= 5 else torch.zeros_like(u)
            )

            def get_grad(y, x_in):
                return torch.autograd.grad(
                    y, x_in,
                    grad_outputs=torch.ones_like(y),
                    create_graph=True, retain_graph=True, allow_unused=True,
                )[0]

            g_phi = get_grad(phi.sum(), x_phys)
            if g_phi is None:
                return torch.zeros(batch_size, device=device)

            phi_x, phi_y, phi_z = g_phi[:, 0], g_phi[:, 1], g_phi[:, 2]
            phi_t = g_phi[:, 5] if x_phys.shape[1] >= 6 else torch.zeros_like(phi)

            # 对流项
            advection = phi_t + u * phi_x + v * phi_y + w * phi_z

            # 双阱势导数
            f_prime = 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

            use_legacy = self.materials_params.get("use_legacy_ac", False)
            if use_legacy:
                gamma = self.materials_params.get("allen_cahn_gamma", 4.5e-7)
                lap_phi = self.safe_compute_laplacian(phi, x_phys)
                ac_residual = advection - gamma * (lap_phi - f_prime)
            else:
                # 标准化相场模型: M [ σ·ε·∇²φ − (σ/ε)·W'(φ) ]
                sigma_ac = self.materials_params.get(
                    "surface_tension_polar_ink", 0.02505
                )
                eps_ac = self.materials_params.get("ac_interface_width", 5e-6)
                M_ac = self.materials_params.get("ac_mobility", 1e-8)

                lap_phi = self.safe_compute_laplacian(phi, x_phys)
                ac_residual = advection - M_ac * (
                    sigma_ac * eps_ac * lap_phi
                    - (sigma_ac / eps_ac) * f_prime
                )

            return ac_residual

        except Exception as e:
            logger.warning(f"Allen-Cahn 残差计算异常: {e}")
            return torch.zeros(batch_size, device=device)

    def safe_compute_laplacian(self, scalar_field: torch.Tensor, coords: torch.Tensor):
        try:
            grad = self.safe_compute_gradient(scalar_field, coords)
            lap = torch.zeros_like(scalar_field)
            for i in range(coords.shape[-1]):
                gi = grad[..., i]
                g2 = torch.autograd.grad(
                    outputs=gi,
                    inputs=coords,
                    grad_outputs=torch.ones_like(gi),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if g2 is not None:
                    lap = lap + g2[..., i]
            return lap
        except Exception:
            return torch.zeros_like(scalar_field)

    def safe_compute_hessian(self, scalar_field: torch.Tensor, coords: torch.Tensor):
        try:
            dim = coords.shape[-1]
            H = torch.zeros((*coords.shape[:-1], dim, dim), device=coords.device)
            grad = self.safe_compute_gradient(scalar_field, coords)
            for i in range(dim):
                gi = grad[..., i]
                g2 = torch.autograd.grad(
                    outputs=gi,
                    inputs=coords,
                    grad_outputs=torch.ones_like(gi),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if g2 is not None:
                    for j in range(dim):
                        H[..., i, j] = g2[..., j]
            return H
        except Exception:
            dim = coords.shape[-1]
            return torch.zeros((*coords.shape[:-1], dim, dim), device=coords.device)

    def compute_young_lippmann_residual(self, x_phys, predictions, applied_voltage):
        """
        计算Young-Lippmann方程残差

        参数:
            x_phys: 物理点输入
            predictions: 模型预测输出
            applied_voltage: 施加的电压

        返回:
            包含Young-Lippmann残差的字典
        """
        try:
            # 安全检查
            if x_phys is None or predictions is None or applied_voltage is None:
                logger.error("Young-Lippmann: 输入参数为None")
                return self._empty_young_lippmann_residual(x_phys)

            # 确保设备一致
            device = x_phys.device
            applied_voltage = applied_voltage.to(device)

            batch_size = x_phys.shape[0]

            # 提取材料参数
            theta0_deg = self.materials_params["contact_angle_theta0"]
            epsilon_0 = self.materials_params.get("epsilon_0", 8.854e-12)
            epsilon_r = self.materials_params.get("relative_permittivity", 3.28)
            epsilon_h = self.materials_params.get("epsilon_hydrophobic", 1.934)
            sigma_po = self.materials_params.get("surface_tension_polar_ink",
                        self.materials_params.get("sigma", 0.02505))
            d_d = self.materials_params.get("dielectric_thickness", 4e-7)
            d_h = self.materials_params.get("hydrophobic_thickness", 4e-7)

            # SU-8 + Teflon 双层串联单位面积电容
            C_yl = 1.0 / (d_d/(epsilon_0*epsilon_r) + d_h/(epsilon_0*epsilon_h))

            # 转换角度为弧度
            theta0_rad = torch.tensor(np.radians(theta0_deg), device=device)

            # 计算cos(theta0)
            cos_theta0 = torch.cos(theta0_rad)

            # 从预测中提取接触角信息
            # 阶段3输出24维：[p, u, v, w, phi, E_z, E_x, vorticity_z, h, kappa, theta, ...]
            # theta在索引10
            try:
                if predictions.shape[1] >= 11:
                    # 索引10是接触角theta（单位：弧度）
                    theta_pred_rad = predictions[:, 10]
                    cos_theta_pred = torch.cos(theta_pred_rad)
                else:
                    # 如果输出维度不足，使用零残差
                    cos_theta_pred = torch.zeros(batch_size, device=device)
                    logger.warning(
                        f"Young-Lippmann: 预测维度不足({predictions.shape[1]}<11)，无法提取接触角"
                    )
            except Exception as e:
                logger.error(f"Young-Lippmann: 提取接触角信息失败: {str(e)}")
                cos_theta_pred = torch.zeros(batch_size, device=device)

            # Young-Lippmann: cos(θ) = cos(θ₀) + C_yl·V²/(2σ_po)
            # σ_po = 极性液体-油界面张力 (液-液体系)
            V_squared = applied_voltage**2
            term = C_yl * V_squared / (2 * sigma_po)
            cos_theta_theory = cos_theta0 + term

            # 限制cos_theta_theory的范围在[-1, 1]内
            cos_theta_theory = torch.clamp(cos_theta_theory, -1.0, 1.0)

            # 计算残差：cosθ_pred - cosθ_theory
            residual = cos_theta_pred - cos_theta_theory

            logger.debug(f"Young-Lippmann残差计算完成，批次大小: {batch_size}")

            return {
                "young_lippmann": residual,
                "cos_theta_pred": cos_theta_pred,
                "cos_theta_theory": cos_theta_theory,
            }

        except Exception as e:
            logger.error(f"Young-Lippmann残差计算失败: {str(e)}")
            return self._empty_young_lippmann_residual(x_phys)

    def _empty_residual(self, x, predictions):
        """返回空的残差字典"""
        device = torch.device("cpu")
        if isinstance(x, torch.Tensor):
            device = x.device
        elif isinstance(predictions, torch.Tensor):
            device = predictions.device

        batch_size = 1
        if isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
        elif isinstance(predictions, torch.Tensor):
            batch_size = predictions.shape[0]

        return {
            "continuity": torch.zeros(batch_size, device=device),
            "momentum_u": torch.zeros(batch_size, device=device),
            "momentum_v": torch.zeros(batch_size, device=device),
            "momentum_w": torch.zeros(batch_size, device=device),
        }

    def _empty_young_lippmann_residual(self, x_phys):
        """返回空的Young-Lippmann残差字典"""
        device = torch.device("cpu")
        if isinstance(x_phys, torch.Tensor):
            device = x_phys.device

        batch_size = 1
        if isinstance(x_phys, torch.Tensor):
            batch_size = x_phys.shape[0]

        return {
            "young_lippmann": torch.zeros(batch_size, device=device),
            "cos_theta_pred": torch.zeros(batch_size, device=device),
            "cos_theta_theory": torch.zeros(batch_size, device=device),
        }

    def compute_contact_line_dynamics_residual(
        self, x_interface, predictions, velocity
    ):
        """
        [DEPRECATED] 接触线动力学 — 假设 11+ 列输出，标准模型不兼容。

        实际训练使用 _compute_contact_line_dynamics_residual() (私有版)，
        通过 autograd 从 φ 场计算所有量，不依赖扩展输出维度。
        此方法假设 predictions[:, 10] 包含接触角，与 TwoPhasePINN (5列输出) 不兼容。
        保留仅用于可能的旧版 24 列模型兼容。
        """
        try:
            # 安全检查
            if x_interface is None or predictions is None or velocity is None:
                logger.error("接触线动力学: 输入参数为None")
                return self._empty_contact_line_residual(x_interface)

            # 确保设备一致
            device = x_interface.device
            velocity = velocity.to(device)

            batch_size = x_interface.shape[0]

            # 提取材料参数
            theta_adv_deg = self.materials_params.get(
                "dynamic_contact_angle_advancing", 120.0
            )
            theta_rec_deg = self.materials_params.get(
                "dynamic_contact_angle_receding", 100.0
            )
            theta0_deg = self.materials_params.get("contact_angle_0", 120.0)
            mu = self.materials_params.get("viscosity_water", 0.001)
            gamma = self.materials_params.get("surface_tension_water", 0.048)
            pinning_energy = self.materials_params.get("pinning_energy", 1e-5)

            # 转换角度为弧度
            theta_adv_rad = torch.tensor(np.radians(theta_adv_deg), device=device)
            theta_rec_rad = torch.tensor(np.radians(theta_rec_deg), device=device)
            theta0_rad = torch.tensor(np.radians(theta0_deg), device=device)

            # 计算cos值
            cos_theta_adv = torch.cos(theta_adv_rad)
            cos_theta_rec = torch.cos(theta_rec_rad)
            cos_theta0 = torch.cos(theta0_rad)

            # 从预测中提取接触角信息
            # 输出索引定义 (与数据生成一致):
            # 0-2: u, v, w (速度)
            # 3: p (压力)
            # 4: alpha (体积分数)
            # 10: theta (接触角，弧度)
            try:
                if predictions.shape[1] >= 11:
                    # 索引10是接触角theta（单位：弧度）
                    theta_pred_rad = predictions[:, 10]
                    cos_theta_pred = torch.cos(theta_pred_rad)
                else:
                    # 如果没有直接的接触角预测，使用静态接触角
                    cos_theta_pred = cos_theta0 * torch.ones(batch_size, device=device)
                    logger.warning("接触线动力学: 预测中没有接触角信息，使用静态接触角")
            except Exception as e:
                logger.error(f"接触线动力学: 提取接触角信息失败: {str(e)}")
                cos_theta_pred = cos_theta0 * torch.ones(batch_size, device=device)

            # 计算接触线动力学模型 - 基于Hoffman-Voinov-Tanner方程
            # 限制cos(theta_pred)的范围在[cos(theta_rec), cos(theta_adv)]内
            cos_theta_pred_clamped = torch.clamp(
                cos_theta_pred, cos_theta_adv, cos_theta_rec
            )

            # 计算接触角滞后引起的力平衡方程
            # 对于动态接触线，考虑速度效应和钉扎效应
            velocity_abs = torch.abs(velocity)

            # 构建幂律关系的动态接触角模型
            # 使用修正版的Tanner定律: cos(theta) - cos(theta0) = k * |v|^n
            n = 0.3  # Tanner指数，通常在0.3-0.5之间
            k = 1e-2  # 比例常数，根据实验调整

            # 计算动态接触角理论值
            cos_theta_theory = cos_theta0 + torch.sign(velocity) * k * torch.pow(
                velocity_abs, n
            )

            # 限制cos_theta_theory的范围
            cos_theta_theory = torch.clamp(
                cos_theta_theory, cos_theta_adv, cos_theta_rec
            )

            # 计算残差：考虑接触角滞后和钉扎效应
            residual = cos_theta_pred - cos_theta_theory

            # 添加钉扎项（能量壁垒）
            pinning_term = pinning_energy * torch.sign(residual)
            residual += pinning_term

            logger.debug(f"接触线动力学残差计算完成，批次大小: {batch_size}")

            return {
                "contact_line_dynamics": residual,
                "cos_theta_pred": cos_theta_pred,
                "cos_theta_theory": cos_theta_theory,
                "velocity": velocity,
            }

        except Exception as e:
            logger.error(f"接触线动力学残差计算失败: {str(e)}")
            return self._empty_contact_line_residual(x_interface)

    def _empty_contact_line_residual(self, x_interface):
        """返回空的接触线动力学残差字典"""
        device = torch.device("cpu")
        if isinstance(x_interface, torch.Tensor):
            device = x_interface.device

        batch_size = 1
        if isinstance(x_interface, torch.Tensor):
            batch_size = x_interface.shape[0]

        return {
            "contact_line_dynamics": torch.zeros(batch_size, device=device),
            "cos_theta_pred": torch.zeros(batch_size, device=device),
            "cos_theta_theory": torch.zeros(batch_size, device=device),
            "velocity": torch.zeros(batch_size, device=device),
        }

    def compute_dielectric_charge_accumulation_residual(
        self, x_dielectric, predictions, voltage, time=None
    ):
        """
        计算介电层电荷积累约束残差

        参数:
            x_dielectric: 介电层点输入
            predictions: 模型预测输出
            voltage: 施加的电压
            time: 当前时间（用于动态电荷积累模型，可选）

        返回:
            包含介电层电荷积累残差的字典
        """
        try:
            # 安全检查
            if x_dielectric is None or predictions is None or voltage is None:
                logger.error("介电层电荷积累: 输入参数为None")
                return self._empty_dielectric_charge_residual(x_dielectric)

            # 确保设备一致
            device = x_dielectric.device
            voltage = voltage.to(device)
            if time is not None:
                time = time.to(device)

            batch_size = x_dielectric.shape[0]

            # 提取材料参数
            epsilon_0 = self.materials_params.get("epsilon_0", 8.854e-12)
            relative_permittivity = self.materials_params.get(
                "relative_permittivity", 3.28
            )
            dielectric_thickness = self.materials_params.get(
                "dielectric_thickness", 1e-6
            )
            dielectric_conductivity = self.materials_params.get(
                "dielectric_conductivity", 1e-12
            )
            charge_relaxation_time = self.materials_params.get(
                "charge_relaxation_time", 1e-3
            )
            leakage_current_coefficient = self.materials_params.get(
                "leakage_current_coefficient", 1e-6
            )
            max_charge_density = self.materials_params.get("max_charge_density", 1e-4)

            # 转换为tensor
            epsilon_0_tensor = torch.tensor(epsilon_0, device=device)
            relative_permittivity_tensor = torch.tensor(
                relative_permittivity, device=device
            )
            dielectric_thickness_tensor = torch.tensor(
                dielectric_thickness, device=device
            )
            dielectric_conductivity_tensor = torch.tensor(
                dielectric_conductivity, device=device
            )
            charge_relaxation_time_tensor = torch.tensor(
                charge_relaxation_time, device=device
            )
            leakage_current_coefficient_tensor = torch.tensor(
                leakage_current_coefficient, device=device
            )
            max_charge_density_tensor = torch.tensor(max_charge_density, device=device)

            # 计算理论电荷密度：σ = ε₀εᵣV/d
            theoretical_charge_density = (
                epsilon_0_tensor
                * relative_permittivity_tensor
                * voltage
                / dielectric_thickness_tensor
            )

            # 限制最大电荷密度
            theoretical_charge_density = torch.clamp(
                theoretical_charge_density,
                -max_charge_density_tensor,
                max_charge_density_tensor,
            )

            # 从预测中提取电荷密度信息（如果有）
            try:
                # 假设预测中包含电荷密度信息
                if hasattr(predictions, "get") and "charge_density" in predictions:
                    predicted_charge_density = predictions["charge_density"]
                elif (
                    isinstance(predictions, torch.Tensor) and predictions.shape[1] >= 6
                ):
                    # 假设第6个分量是电荷密度
                    predicted_charge_density = predictions[:, 5]
                else:
                    # 使用理论值作为默认值
                    predicted_charge_density = theoretical_charge_density
                    logger.warning("介电层电荷积累: 预测中没有电荷密度信息，使用理论值")
            except Exception as e:
                logger.error(f"介电层电荷积累: 提取电荷密度信息失败: {str(e)}")
                predicted_charge_density = theoretical_charge_density

            # 计算电荷积累动力学（RC电路模型）
            if time is not None:
                # 动态电荷积累模型：σ(t) = σ_max(1 - exp(-t/τ))
                charge_accumulation = theoretical_charge_density * (
                    1.0 - torch.exp(-time / charge_relaxation_time_tensor)
                )
            else:
                charge_accumulation = theoretical_charge_density

            # 计算泄漏电流效应
            leakage_term = (
                leakage_current_coefficient_tensor * voltage * voltage
            )  # 简化的泄漏电流模型
            charge_with_leakage = charge_accumulation - leakage_term

            # 计算残差
            residual = predicted_charge_density - charge_with_leakage

            logger.debug(f"介电层电荷积累残差计算完成，批次大小: {batch_size}")

            return {
                "dielectric_charge": residual,
                "predicted_charge_density": predicted_charge_density,
                "theoretical_charge_density": theoretical_charge_density,
                "charge_with_leakage": charge_with_leakage,
            }

        except Exception as e:
            logger.error(f"介电层电荷积累残差计算失败: {str(e)}")
            return self._empty_dielectric_charge_residual(x_dielectric)

    def _empty_dielectric_charge_residual(self, x_dielectric):
        """返回空的介电层电荷积累残差字典"""
        device = torch.device("cpu")
        if isinstance(x_dielectric, torch.Tensor):
            device = x_dielectric.device

        batch_size = 1
        if isinstance(x_dielectric, torch.Tensor):
            batch_size = x_dielectric.shape[0]

        return {
            "dielectric_charge": torch.zeros(batch_size, device=device),
            "predicted_charge_density": torch.zeros(batch_size, device=device),
            "theoretical_charge_density": torch.zeros(batch_size, device=device),
            "charge_with_leakage": torch.zeros(batch_size, device=device),
        }

    def compute_thermodynamic_residual(
        self, x, predictions, temperature, applied_voltage=None
    ):
        """
        计算热力学约束残差
        包括：温度对表面张力和粘度的影响，焦耳热效应

        参数：
        - x: 空间坐标输入
        - predictions: 模型预测输出（速度、压力等）
        - temperature: 温度场预测
        - applied_voltage: 施加的电压（用于焦耳热计算）

        返回：
        - 热力学约束残差字典
        """
        try:
            # 安全检查
            if x is None or predictions is None or temperature is None:
                logger.error("输入x、predictions或temperature为None")
                return self._empty_thermodynamic_residual(x, predictions)

            # 确保x是可微分的
            if not isinstance(x, torch.Tensor):
                logger.error(f"输入x类型错误，应为torch.Tensor，实际为{type(x)}")
                x = torch.tensor(x, dtype=torch.float32).requires_grad_(True)

            # 确保predictions和temperature是tensor并在正确设备上
            if not isinstance(predictions, torch.Tensor):
                logger.error(
                    f"predictions类型错误，应为torch.Tensor，实际为{type(predictions)}"
                )
                predictions = torch.tensor(predictions, dtype=torch.float32)

            if not isinstance(temperature, torch.Tensor):
                logger.error(
                    f"temperature类型错误，应为torch.Tensor，实际为{type(temperature)}"
                )
                temperature = torch.tensor(temperature, dtype=torch.float32)

            # 确保设备一致
            device = x.device
            predictions = predictions.to(device)
            temperature = temperature.to(device)

            # 验证x需要梯度
            if not x.requires_grad:
                logger.warning("物理点x不需要梯度，克隆并设置requires_grad=True")
                x = x.clone().requires_grad_(True)

            batch_size = x.shape[0]
            logger.info(f"计算热力学约束残差，批大小: {batch_size}")

            # 提取材料参数
            ambient_temp = self.materials_params.get("ambient_temperature", 293.15)
            temp_coef_surface_tension = self.materials_params.get(
                "temperature_coefficient_surface_tension", -1.5e-4
            )
            temp_coef_viscosity = self.materials_params.get(
                "temperature_coefficient_viscosity", -3.5e-3
            )
            thermal_conductivity_water = self.materials_params.get(
                "thermal_conductivity_water", 0.6
            )
            specific_heat_water = self.materials_params.get(
                "specific_heat_water", 4186.0
            )
            density_water = self.materials_params["density"]
            dielectric_conductivity = self.materials_params.get(
                "dielectric_conductivity", 1e-12
            )
            dielectric_thickness = self.materials_params.get(
                "dielectric_thickness", 1e-6
            )

            # 计算温度影响的参数
            # 1. 温度对表面张力的影响
            surface_tension_ambient = self.materials_params["surface_tension"]
            surface_tension_temp = surface_tension_ambient * (
                1.0 + temp_coef_surface_tension * (temperature - ambient_temp)
            )

            # 2. 温度对粘度的影响
            viscosity_ambient = self.materials_params.get("viscosity", 1.0)
            viscosity_temp = viscosity_ambient * torch.exp(
                temp_coef_viscosity * (temperature - ambient_temp)
            )

            # 3. 热传导方程残差（简化版傅里叶热传导）
            # 计算温度梯度
            try:
                dT_dx = torch.autograd.grad(
                    temperature,
                    x,
                    grad_outputs=torch.ones_like(temperature),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                # 确保梯度计算成功
                if dT_dx is None:
                    logger.error("温度梯度计算失败")
                    return self._empty_thermodynamic_residual(x, predictions)

                # 计算温度的拉普拉斯算子（简化为梯度的散度）
                laplacian_T = torch.zeros_like(temperature)
                for i in range(x.shape[1]):
                    d2T_dx2 = torch.autograd.grad(
                        dT_dx[:, i],
                        x,
                        grad_outputs=torch.ones_like(dT_dx[:, i]),
                        create_graph=True,
                        retain_graph=True,
                    )[0][:, i]
                    if d2T_dx2 is not None:
                        laplacian_T += d2T_dx2
            except Exception as e:
                logger.error(f"温度梯度计算错误: {str(e)}")
                return self._empty_thermodynamic_residual(x, predictions)

            # 4. 焦耳热效应（如果有电压）
            joule_heating = torch.zeros_like(temperature)
            if applied_voltage is not None:
                # 计算焦耳热：P = σ * E²，其中E是电场强度（V/d）
                electric_field = applied_voltage / dielectric_thickness
                joule_heating = (
                    dielectric_conductivity
                    * electric_field**2
                    * torch.ones_like(temperature)
                )

            # 5. 能量守恒方程残差
            # 简化的能量方程：ρ*Cp*∂T/∂t = k*∇²T + Q_joule
            # 这里我们只计算空间部分：k*∇²T + Q_joule
            heat_equation_residual = (
                thermal_conductivity_water * laplacian_T - joule_heating
            )

            # 6. 热膨胀引起的密度变化
            thermal_expansion = self.materials_params.get(
                "thermal_expansion_water", 2.1e-4
            )
            density_temp = density_water * (
                1.0 - thermal_expansion * (temperature - ambient_temp)
            )

            # 7. 确保温度在物理合理范围内
            temp_min = 273.15  # 水的冰点
            temp_max = 373.15  # 水的沸点
            temperature_constraint = torch.maximum(
                torch.zeros_like(temperature),
                torch.minimum(temperature - temp_min, temp_max - temperature),
            )

            # 收集残差项
            residuals = {
                "surface_tension_temp_effect": surface_tension_temp
                - surface_tension_ambient,
                "viscosity_temp_effect": viscosity_temp - viscosity_ambient,
                "heat_equation": heat_equation_residual,
                "temperature_limits": temperature_constraint,
                "thermal_expansion": density_temp - density_water,
            }

            # 返回标准化的残差字典
            return {
                key: (
                    residual / (torch.max(torch.abs(residual)) + 1e-12)
                    if torch.any(torch.abs(residual) > 0)
                    else residual
                )
                for key, residual in residuals.items()
            }

        except Exception as e:
            logger.error(f"计算热力学残差时出错: {str(e)}")
            return self._empty_thermodynamic_residual(x, predictions)

    def _empty_thermodynamic_residual(self, x, predictions):
        """返回空的热力学残差字典（错误情况处理）"""
        try:
            if x is None or predictions is None:
                return {
                    "surface_tension_temp_effect": torch.tensor(
                        [0.0], requires_grad=True
                    ),
                    "viscosity_temp_effect": torch.tensor([0.0], requires_grad=True),
                    "heat_equation": torch.tensor([0.0], requires_grad=True),
                    "temperature_limits": torch.tensor([0.0], requires_grad=True),
                    "thermal_expansion": torch.tensor([0.0], requires_grad=True),
                }

            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1

            return {
                "surface_tension_temp_effect": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "viscosity_temp_effect": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "heat_equation": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "temperature_limits": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "thermal_expansion": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }
        except Exception:
            return {
                "surface_tension_temp_effect": torch.tensor([0.0], requires_grad=True),
                "viscosity_temp_effect": torch.tensor([0.0], requires_grad=True),
                "heat_equation": torch.tensor([0.0], requires_grad=True),
                "temperature_limits": torch.tensor([0.0], requires_grad=True),
                "thermal_expansion": torch.tensor([0.0], requires_grad=True),
            }

    def compute_interface_stability_residual(self, x, predictions):
        """
        计算界面稳定性约束残差

        Args:
            x: 输入坐标 (t, x, y, z)
            predictions: 模型预测结果，包含速度场、压力等

        Returns:
            residual: 包含各种界面稳定性约束残差的字典
        """
        # 安全检查
        if predictions is None:
            logger.warning("compute_interface_stability_residual: predictions 为 None")
            return self._empty_interface_stability_residual(x, predictions)

        try:
            # 提取材料参数
            material_params = self.materials_params or {}

            # 1. 界面能量最小化约束
            # 获取界面位置（假设predictions中包含界面指示器或水平集函数）
            interface_indicator = None

            # 尝试从predictions中提取界面信息
            if isinstance(predictions, dict):
                # 检查各种可能的键名
                for key in ["interface", "level_set", "phase_indicator"]:
                    if key in predictions:
                        interface_indicator = predictions[key]
                        break

            # 如果没有明确的界面指示器，尝试从其他预测中推导
            if interface_indicator is None:
                # 假设使用水平集方法，将压力场或速度场的某些特性作为界面指示器
                # 这是一个简化处理，实际应用需要根据具体模型调整
                if isinstance(predictions, torch.Tensor) and predictions.size(-1) >= 3:
                    # 假设使用压力场作为近似
                    interface_indicator = predictions[..., 3:4]  # 假设第四维是压力
                else:
                    # 创建一个默认的界面指示器
                    interface_indicator = torch.zeros_like(x[..., :1], device=x.device)

            # 2. 计算界面梯度和曲率
            # 界面梯度用于衡量界面的尖锐程度
            try:
                coords3 = x[:, :3]
            except Exception:
                coords3 = x
            interface_grad = self.safe_compute_gradient(interface_indicator, coords3)
            interface_grad_norm = torch.norm(interface_grad, dim=-1, keepdim=True)

            # 界面曲率计算
            # 使用水平集方法计算曲率: κ = ∇·(∇φ/|∇φ|)
            # 这里使用简化的拉普拉斯算子作为近似
            interface_laplacian = self.safe_compute_laplacian(
                interface_indicator, coords3
            )

            # 避免除以零
            safe_grad_norm = torch.clamp(interface_grad_norm, min=1e-6)

            # 计算曲率（近似）
            curvature = interface_laplacian / safe_grad_norm

            # 3. 界面稳定性约束 - 防止Rayleigh-Taylor和Kelvin-Helmholtz不稳定性
            # Rayleigh-Taylor不稳定性约束 (重流体在轻流体上方时的稳定性)
            density_gradient_constraint = torch.zeros_like(
                interface_indicator, device=x.device
            )

            # Kelvin-Helmholtz不稳定性约束 (切向速度差引起的稳定性)
            kelvin_helmholtz_constraint = torch.zeros_like(
                interface_indicator, device=x.device
            )

            # 提取速度场计算切向速度差
            if isinstance(predictions, dict) and "velocity" in predictions:
                velocity = predictions["velocity"]
            elif isinstance(predictions, torch.Tensor) and predictions.size(-1) >= 3:
                velocity = predictions[..., :3]  # 假设前三维是速度
            else:
                velocity = torch.zeros((*x.shape[:-1], 3), device=x.device)

            # 计算速度的切向分量
            # 界面法向量
            interface_normal = interface_grad / safe_grad_norm

            # 速度的法向分量
            velocity_normal = torch.sum(
                velocity * interface_normal, dim=-1, keepdim=True
            )

            # 速度的切向分量
            velocity_tangential = velocity - velocity_normal * interface_normal

            # 计算切向速度梯度
            velocity_tangential_grad = self.safe_compute_gradient(
                velocity_tangential, coords3
            )

            # Kelvin-Helmholtz不稳定性的简化约束：切向速度梯度不应过大
            kelvin_helmholtz_constraint = torch.norm(
                velocity_tangential_grad, dim=-1
            ).mean(dim=-1, keepdim=True)

            # 4. 界面平滑性约束 - 防止指纹或其他高频缺陷
            # 使用高阶导数约束界面的平滑性
            interface_hessian = self.safe_compute_hessian(interface_indicator, x)
            hessian_norm = torch.norm(interface_hessian, dim=(-2, -1), keepdim=True)

            # 5. 界面能量最小化原理约束
            # 界面能量与界面面积和表面张力成正比
            # 这里使用梯度的散度作为界面面积变化的近似
            interface_area_change = self.safe_compute_laplacian(interface_grad_norm, x)

            # 合并所有残差
            residual = {
                "interface_curvature": curvature,
                "interface_gradient": interface_grad_norm,
                "density_gradient_constraint": density_gradient_constraint,
                "kelvin_helmholtz_constraint": kelvin_helmholtz_constraint,
                "interface_smoothness": hessian_norm,
                "interface_area_change": interface_area_change,
                "velocity_tangential": torch.norm(
                    velocity_tangential, dim=-1, keepdim=True
                ),
            }

            return residual

        except Exception as e:
            logger.error(f"compute_interface_stability_residual 计算异常: {str(e)}")
            return self._empty_interface_stability_residual(x, predictions)

    def _empty_interface_stability_residual(self, x, predictions):
        """
        返回空的界面稳定性约束残差

        Args:
            x: 输入坐标
            predictions: 模型预测结果

        Returns:
            空的界面稳定性约束残差字典
        """
        try:
            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1

            return {
                "interface_curvature": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "interface_gradient": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "density_gradient_constraint": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "kelvin_helmholtz_constraint": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "interface_smoothness": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "interface_area_change": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "velocity_tangential": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }
        except Exception:
            return {
                "interface_curvature": torch.tensor([0.0], requires_grad=True),
                "interface_gradient": torch.tensor([0.0], requires_grad=True),
                "density_gradient_constraint": torch.tensor([0.0], requires_grad=True),
                "kelvin_helmholtz_constraint": torch.tensor([0.0], requires_grad=True),
                "interface_smoothness": torch.tensor([0.0], requires_grad=True),
                "interface_area_change": torch.tensor([0.0], requires_grad=True),
                "velocity_tangential": torch.tensor([0.0], requires_grad=True),
            }

    def compute_frequency_response_residual(
        self, x, predictions, frequency=None, applied_voltage=None
    ):
        """
        计算频率响应约束残差，考虑交流电场下的特性和介电弛豫效应

        Args:
            x: 输入坐标 (t, x, y, z)
            predictions: 模型预测结果，包含速度场、压力等
            frequency: 交流电场频率 (Hz)
            applied_voltage: 施加的电压信号

        Returns:
            residual: 包含各种频率响应约束残差的字典
        """
        # 安全检查
        if predictions is None:
            logger.warning("compute_frequency_response_residual: predictions 为 None")
            return self._empty_frequency_response_residual(x, predictions)

        try:
            # 提取材料参数
            material_params = self.materials_params or {}

            # 1. 介电弛豫效应约束
            # 提取相对介电常数和电导率
            permittivity_params = material_params.get("permittivity", {})
            conductivity_params = material_params.get("conductivity", {})
            if isinstance(permittivity_params, (float, int)):
                permittivity_params = {
                    "epsilon_rel_1": float(permittivity_params),
                    "epsilon_rel_2": float(permittivity_params),
                    "relaxation_time": 1e-6,
                }
            if isinstance(conductivity_params, (float, int)):
                conductivity_params = {"sigma": float(conductivity_params)}

            # 默认值
            epsilon_rel_1 = permittivity_params.get(
                "epsilon_rel_1", 2.0
            )  # 低频相对介电常数
            epsilon_rel_2 = permittivity_params.get(
                "epsilon_rel_2", 1.0
            )  # 高频相对介电常数
            relaxation_time = permittivity_params.get(
                "relaxation_time", 1e-6
            )  # 弛豫时间常数
            sigma = conductivity_params.get("sigma", 1e-4)  # 电导率

            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")

            # 2. 计算介电常数的频率依赖性 (Debye模型)
            if frequency is not None:
                omega = 2 * torch.tensor(
                    np.pi * frequency, device=device, requires_grad=True
                )

                # Debye模型: ε(ω) = ε_∞ + (ε_s - ε_∞)/(1 + jωτ)
                epsilon_complex = epsilon_rel_2 + (epsilon_rel_1 - epsilon_rel_2) / (
                    1 + 1j * omega * relaxation_time
                )

                # 计算介电损耗角正切
                tan_delta = (
                    (epsilon_rel_1 - epsilon_rel_2)
                    * omega
                    * relaxation_time
                    / (epsilon_rel_1 * epsilon_rel_2 + (omega * relaxation_time) ** 2)
                )

                # 3. 位移电流与传导电流的比例约束
                # 在高频下，位移电流主导；在低频下，传导电流主导
                displacement_conduction_ratio = (
                    (omega * epsilon_rel_1) / sigma if sigma > 0 else 0
                )
            else:
                tan_delta = torch.tensor([0.0], device=device, requires_grad=True)
                displacement_conduction_ratio = torch.tensor(
                    [0.0], device=device, requires_grad=True
                )

            # 4. 交流电场下的界面响应约束
            # 获取界面位置指示器
            interface_indicator = None

            # 尝试从predictions中提取界面信息
            if isinstance(predictions, dict):
                # 检查各种可能的键名
                for key in ["interface", "level_set", "phase_indicator"]:
                    if key in predictions:
                        interface_indicator = predictions[key]
                        break

            # 如果没有明确的界面指示器，尝试从其他预测中推导
            if interface_indicator is None:
                if isinstance(predictions, torch.Tensor) and predictions.size(-1) >= 3:
                    # 假设使用压力场作为近似
                    interface_indicator = predictions[..., 3:4]  # 假设第四维是压力
                else:
                    # 创建一个默认的界面指示器
                    interface_indicator = torch.zeros_like(x[..., :1], device=device)

            # 5. 计算电场随时间的变化率（对于交流电场）
            dEdt = torch.zeros_like(
                interface_indicator, device=device, requires_grad=True
            )

            if applied_voltage is not None and frequency is not None:
                # 假设电压是时间的函数
                voltage_tensor = (
                    applied_voltage
                    if isinstance(applied_voltage, torch.Tensor)
                    else torch.tensor(
                        float(applied_voltage), device=device, requires_grad=True
                    )
                )

                # 确保voltage_tensor具有适当的形状
                if len(voltage_tensor.shape) == 0:
                    voltage_tensor = voltage_tensor.expand(x.shape[0], 1)

                # 计算电压的时间导数
                try:
                    dEdt = self.safe_compute_gradient(
                        voltage_tensor.squeeze(-1), x[..., 0:1]
                    )
                except Exception:
                    dEdt = torch.zeros_like(
                        interface_indicator, device=device, requires_grad=True
                    )

            # 6. 频率响应一致性约束
            # 确保高频和低频下的物理行为一致
            frequency_consistency = torch.zeros_like(
                interface_indicator, device=device, requires_grad=True
            )

            # 7. 介电损耗约束
            # 限制介电损耗在合理范围内
            dielectric_loss_constraint = torch.zeros_like(
                interface_indicator, device=device, requires_grad=True
            )
            if isinstance(tan_delta, torch.Tensor):
                dielectric_loss_constraint = torch.clamp(
                    tan_delta - 0.1, min=0
                )  # 限制损耗角正切不超过0.1

            # 8. 电容响应约束
            # 确保电容在不同频率下的响应符合物理规律
            capacitance_response = torch.zeros_like(
                interface_indicator, device=device, requires_grad=True
            )

            # 9. 泄漏电流约束
            leakage_current = torch.zeros_like(
                interface_indicator, device=device, requires_grad=True
            )

            # 合并所有残差
            residual = {
                "dielectric_relaxation": (
                    tan_delta
                    if isinstance(tan_delta, torch.Tensor)
                    else torch.tensor([0.0], device=device, requires_grad=True)
                ),
                "displacement_conduction_ratio": (
                    displacement_conduction_ratio
                    if isinstance(displacement_conduction_ratio, torch.Tensor)
                    else torch.tensor([0.0], device=device, requires_grad=True)
                ),
                "ac_field_response": dEdt,
                "frequency_consistency": frequency_consistency,
                "dielectric_loss_constraint": dielectric_loss_constraint,
                "capacitance_response": capacitance_response,
                "leakage_current": leakage_current,
            }

            return residual

        except Exception as e:
            logger.error(f"compute_frequency_response_residual 计算异常: {str(e)}")
            return self._empty_frequency_response_residual(x, predictions)

    def _empty_frequency_response_residual(self, x, predictions):
        """
        返回空的频率响应约束残差

        Args:
            x: 输入坐标
            predictions: 模型预测结果

        Returns:
            空的频率响应约束残差字典
        """
        try:
            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1

            return {
                "dielectric_relaxation": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "displacement_conduction_ratio": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "ac_field_response": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "frequency_consistency": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "dielectric_loss_constraint": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "capacitance_response": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "leakage_current": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }
        except Exception:
            return {
                "dielectric_relaxation": torch.tensor([0.0], requires_grad=True),
                "displacement_conduction_ratio": torch.tensor(
                    [0.0], requires_grad=True
                ),
                "ac_field_response": torch.tensor([0.0], requires_grad=True),
                "frequency_consistency": torch.tensor([0.0], requires_grad=True),
                "dielectric_loss_constraint": torch.tensor([0.0], requires_grad=True),
                "capacitance_response": torch.tensor([0.0], requires_grad=True),
                "leakage_current": torch.tensor([0.0], requires_grad=True),
            }

    def compute_optical_properties_residual(
        self, x, predictions, light_wavelength=None, angle_of_incidence=None
    ):
        """
        计算光学特性约束残差，包括反射/透射特性和对比度约束

        Args:
            x: 输入坐标 (t, x, y, z)
            predictions: 模型预测结果，包含速度场、压力等
            light_wavelength: 入射光波长 (m)
            angle_of_incidence: 入射角 (rad)

        Returns:
            residual: 包含各种光学特性约束残差的字典
        """
        # 安全检查
        if predictions is None:
            logger.warning("compute_optical_properties_residual: predictions 为 None")
            return self._empty_optical_properties_residual(x, predictions)

        try:
            # 提取材料参数
            material_params = self.materials_params or {}

            # 1. 光学参数获取
            # 提取折射率和消光系数
            optical_params = material_params.get("optical", {})

            # 默认值 - 水和油的典型光学参数
            refractive_index_water = optical_params.get("refractive_index_water", 1.33)
            refractive_index_oil = optical_params.get("refractive_index_oil", 1.47)
            extinction_coeff_water = optical_params.get("extinction_coeff_water", 1e-5)
            extinction_coeff_oil = optical_params.get("extinction_coeff_oil", 1e-4)

            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")

            # 2. 获取界面位置指示器
            interface_indicator = None

            # 尝试从predictions中提取界面信息
            if isinstance(predictions, dict):
                # 检查各种可能的键名
                for key in ["interface", "level_set", "phase_indicator"]:
                    if key in predictions:
                        interface_indicator = predictions[key]
                        break

            # 如果没有明确的界面指示器，尝试从其他预测中推导
            if interface_indicator is None:
                if isinstance(predictions, torch.Tensor) and predictions.size(-1) >= 3:
                    # 假设使用压力场作为近似
                    interface_indicator = predictions[..., 3:4]  # 假设第四维是压力
                else:
                    # 创建一个默认的界面指示器
                    interface_indicator = torch.zeros_like(x[..., :1], device=device)

            # 3. 计算反射率约束
            # 基于菲涅尔方程计算反射率
            # 假设垂直入射作为简化情况
            if angle_of_incidence is None:
                angle_of_incidence = torch.tensor(
                    0.0, device=device, requires_grad=True
                )

            # 4. 计算反射率
            # 简化的菲涅尔反射率计算 (垂直入射)
            n1 = refractive_index_water
            n2 = refractive_index_oil

            # 计算菲涅尔反射率
            fresnel_reflectance = ((n1 - n2) / (n1 + n2)) ** 2

            # 5. 界面对比度约束
            # 对比度 C = (I_max - I_min) / (I_max + I_min)
            # 我们希望在界面处有高对比度，便于成像和检测
            contrast_threshold = 0.3  # 期望的最小对比度

            # 6. 透射率约束
            # 计算透射率
            fresnel_transmittance = 1.0 - fresnel_reflectance

            # 7. 光吸收约束
            # 考虑消光系数的影响
            absorption_coefficient_water = 4 * np.pi * extinction_coeff_water
            absorption_coefficient_oil = 4 * np.pi * extinction_coeff_oil

            # 8. 界面锐利度约束
            # 确保界面处的折射率变化足够锐利，以获得清晰的光学边界
            interface_sharpness = torch.zeros_like(
                interface_indicator, device=device, requires_grad=True
            )

            # 如果有梯度计算能力，计算界面梯度
            if hasattr(self, "safe_compute_gradient"):
                try:
                    # 计算界面位置的梯度
                    grad_interface = self.safe_compute_gradient(
                        interface_indicator, x[..., 1:4]
                    )  # 空间梯度
                    interface_sharpness = torch.norm(
                        grad_interface, dim=-1, keepdim=True
                    )
                except Exception:
                    interface_sharpness = torch.zeros_like(
                        interface_indicator, device=device, requires_grad=True
                    )

            # 9. 光学一致性约束
            # 确保光学特性在整个域内保持物理一致性
            optical_consistency = torch.zeros_like(
                interface_indicator, device=device, requires_grad=True
            )

            # 10. 波长依赖性约束
            # 如果提供了波长，考虑不同波长的影响
            wavelength_dependency = torch.zeros_like(
                interface_indicator, device=device, requires_grad=True
            )

            if light_wavelength is not None:
                # 模拟不同波长下的折射率变化
                lambda_0 = 550e-9  # 参考波长 (绿色光)
                delta_lambda = (light_wavelength - lambda_0) / lambda_0

                # 色散效应 - 简化的柯西色散公式
                dispersion_factor = 1.0 + 0.01 * delta_lambda
                wavelength_dependency = delta_lambda * torch.ones_like(
                    interface_indicator, device=device, requires_grad=True
                )

            # 合并所有残差
            residual = {
                "reflectance_constraint": torch.tensor(
                    [fresnel_reflectance], device=device, requires_grad=True
                )
                * torch.ones_like(interface_indicator),
                "transmittance_constraint": torch.tensor(
                    [fresnel_transmittance], device=device, requires_grad=True
                )
                * torch.ones_like(interface_indicator),
                "contrast_constraint": torch.tensor(
                    [1.0 - contrast_threshold], device=device, requires_grad=True
                )
                * torch.ones_like(interface_indicator),
                "interface_sharpness": interface_sharpness,
                "optical_consistency": optical_consistency,
                "absorption_water": torch.tensor(
                    [absorption_coefficient_water], device=device, requires_grad=True
                )
                * torch.ones_like(interface_indicator),
                "absorption_oil": torch.tensor(
                    [absorption_coefficient_oil], device=device, requires_grad=True
                )
                * torch.ones_like(interface_indicator),
                "wavelength_dependency": wavelength_dependency,
            }

            return residual

        except Exception as e:
            logger.error(f"compute_optical_properties_residual 计算异常: {str(e)}")
            return self._empty_optical_properties_residual(x, predictions)

    def _empty_optical_properties_residual(self, x, predictions):
        """
        返回空的光学特性约束残差

        Args:
            x: 输入坐标
            predictions: 模型预测结果

        Returns:
            空的光学特性约束残差字典
        """
        try:
            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1

            return {
                "reflectance_constraint": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "transmittance_constraint": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "contrast_constraint": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "interface_sharpness": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "optical_consistency": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "absorption_water": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "absorption_oil": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "wavelength_dependency": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }
        except Exception:
            return {
                "reflectance_constraint": torch.tensor([0.0], requires_grad=True),
                "transmittance_constraint": torch.tensor([0.0], requires_grad=True),
                "contrast_constraint": torch.tensor([0.0], requires_grad=True),
                "interface_sharpness": torch.tensor([0.0], requires_grad=True),
                "optical_consistency": torch.tensor([0.0], requires_grad=True),
                "absorption_water": torch.tensor([0.0], requires_grad=True),
                "absorption_oil": torch.tensor([0.0], requires_grad=True),
                "wavelength_dependency": torch.tensor([0.0], requires_grad=True),
            }

    def compute_energy_efficiency_residual(self, x, predictions, applied_voltage=None):
        """
        计算能量效率约束残差，包括功耗限制和能量转换效率

        Args:
            x: 输入坐标 (t, x, y, z)
            predictions: 模型预测结果，包含速度场、压力等
            applied_voltage: 施加的电压

        Returns:
            residual: 包含各种能量效率约束残差的字典
        """
        # 安全检查
        if predictions is None:
            logger.warning("compute_energy_efficiency_residual: predictions 为 None")
            return self._empty_energy_efficiency_residual(x, predictions)

        try:
            # 提取材料参数
            material_params = self.materials_params or {}

            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1

            # 1. 计算电场强度和电流密度分布
            # 从模型预测中提取必要的物理量
            e_field = None
            charge_density = None
            velocity = None
            pressure = None

            if isinstance(predictions, dict):
                e_field = predictions.get("e_field", None)
                charge_density = predictions.get("charge_density", None)
                velocity = predictions.get("velocity", None)
                pressure = predictions.get("pressure", None)
            elif isinstance(predictions, torch.Tensor) and predictions.size(-1) >= 4:
                # 假设速度场是前三个分量，压力是第四个分量
                velocity = predictions[..., :3]
                pressure = predictions[..., 3:4]

            residual = {}

            # 2. 功耗限制约束
            if e_field is not None:
                # 计算电场能量密度
                e_energy_density = 0.5 * torch.sum(e_field**2, dim=-1)

                # 计算体积积分得到总电场能量
                total_electric_energy = torch.mean(e_energy_density)

                # 添加功耗约束残差
                # 基于目标功耗水平（这里使用一个参考值）
                target_power = torch.tensor(
                    1.0, dtype=torch.float32, device=device, requires_grad=True
                )
                residual["power_consumption"] = torch.abs(
                    total_electric_energy - target_power
                )

            # 3. 能量转换效率约束
            if (
                applied_voltage is not None
                and velocity is not None
                and pressure is not None
            ):
                # 计算机械能输出（简化为速度和压力的函数）
                velocity_magnitude = torch.sqrt(
                    torch.sum(velocity**2, dim=-1, keepdim=True)
                )
                mechanical_energy = torch.mean(velocity_magnitude * pressure)

                # 计算电能输入（基于施加的电压）
                voltage_magnitude = torch.abs(applied_voltage)

                # 简单模型：电流与电荷密度和速度相关
                if charge_density is not None:
                    current_estimate = torch.mean(
                        torch.abs(charge_density) * velocity_magnitude
                    )
                    electrical_power = voltage_magnitude * current_estimate

                    # 确保分母不为零
                    safe_electrical_power = torch.maximum(
                        electrical_power, torch.tensor(1e-10, device=device)
                    )

                    # 计算能量转换效率
                    energy_efficiency = mechanical_energy / safe_electrical_power

                    # 目标效率（可以根据应用需求调整）
                    target_efficiency = torch.tensor(
                        0.7, dtype=torch.float32, device=device, requires_grad=True
                    )

                    # 添加能量效率约束残差
                    # 使用tanh函数来降低过大值的影响
                    residual["energy_efficiency"] = torch.tanh(
                        torch.abs(energy_efficiency - target_efficiency)
                    )

            # 4. 能量耗散约束
            if velocity is not None and "viscosity" in material_params:
                try:
                    # 计算速度梯度
                    if hasattr(self, "safe_compute_gradient"):
                        velocity_grad = self.safe_compute_gradient(
                            velocity, x[..., 1:4]
                        )

                        # 计算粘性耗散率
                        viscosity = torch.tensor(
                            material_params["viscosity"],
                            dtype=torch.float32,
                            device=device,
                        )
                        viscous_dissipation = viscosity * torch.sum(
                            velocity_grad**2, dim=[-2, -1]
                        )

                        # 限制粘性耗散
                        max_dissipation = torch.tensor(
                            0.5, dtype=torch.float32, device=device, requires_grad=True
                        )
                        residual["viscous_dissipation"] = torch.nn.functional.relu(
                            viscous_dissipation - max_dissipation
                        )
                except Exception:
                    pass

            # 5. 电压利用效率约束
            if applied_voltage is not None and e_field is not None:
                # 计算电场利用率
                e_field_magnitude = torch.sqrt(torch.sum(e_field**2, dim=-1))
                mean_e_field = torch.mean(e_field_magnitude)

                # 理想情况下，电场应该有效地用于驱动流体，而不是耗散
                optimal_e_field = voltage_magnitude / torch.tensor(
                    1.0, dtype=torch.float32, device=device
                )

                # 添加电压利用效率约束
                residual["voltage_efficiency"] = torch.abs(
                    mean_e_field - optimal_e_field
                )

            # 6. 最低能量状态约束（可选）
            # 确保系统趋向于能量最小状态
            if velocity is not None:
                kinetic_energy = 0.5 * torch.sum(velocity**2, dim=-1)
                total_kinetic_energy = torch.mean(kinetic_energy)
                residual["kinetic_energy"] = total_kinetic_energy

            return residual

        except Exception as e:
            logger.error(f"compute_energy_efficiency_residual 计算异常: {str(e)}")
            return self._empty_energy_efficiency_residual(x, predictions)

    def _empty_energy_efficiency_residual(self, x, predictions):
        """
        返回空的能量效率约束残差

        Args:
            x: 输入坐标
            predictions: 模型预测结果

        Returns:
            空的能量效率约束残差字典
        """
        try:
            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1

            return {
                "power_consumption": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "energy_efficiency": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "viscous_dissipation": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "voltage_efficiency": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
                "kinetic_energy": torch.zeros(
                    batch_size, device=device, requires_grad=True
                ),
            }
        except Exception:
            return {
                "power_consumption": torch.tensor([0.0], requires_grad=True),
                "energy_efficiency": torch.tensor([0.0], requires_grad=True),
                "viscous_dissipation": torch.tensor([0.0], requires_grad=True),
                "voltage_efficiency": torch.tensor([0.0], requires_grad=True),
                "kinetic_energy": torch.tensor([0.0], requires_grad=True),
            }


