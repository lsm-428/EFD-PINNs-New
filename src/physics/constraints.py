"""
EWPINN 物理约束模块
包含物理方程计算、材料参数和边界条件处理
"""

import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger("EWPINN_Physics")

# 导入统一物理配置
try:
    from src.config import PHYSICS

    _DEFAULT_MATERIALS = None  # 延迟加载
except ImportError:
    logger.warning("统一配置模块不可用，使用本地默认参数")
    _DEFAULT_MATERIALS = None
    PHYSICS = {}

# 共享梯度工具 (消除跨文件公式重复)
from src.utils.gradients import compute_gradient, gradient_magnitude, mean_curvature_3d

# 导入动态权重调整模块
try:
    import src.training.scheduler  # noqa: F401

    DYNAMIC_WEIGHT_AVAILABLE = True
except ImportError:
    logger.warning("动态权重调整模块不可用，将使用固定权重")
    DYNAMIC_WEIGHT_AVAILABLE = False


def _get_default_materials_params() -> dict:
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

    # 回退到 PHYSICS 配置（单一来源）
    # 使用 PHYSICS.get() 确保向后兼容（PHYSICS 不可用时使用默认值）
    from src.config import PHYSICS

    return {
        # 基础流体属性
        "viscosity": PHYSICS.get("mu_polar", 1.01e-3),
        "density": PHYSICS.get("rho_polar", 998.0),
        "surface_tension": PHYSICS.get("gamma", 0.048),
        # 材料属性（PHYSICS 中无对应，保留硬编码）
        "permittivity": 80.1,
        "conductivity": 5.5e7,
        "youngs_modulus": 210e9,
        "poisson_ratio": 0.3,
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
        # 电学属性
        "epsilon_0": PHYSICS.get("epsilon_0", 8.854e-12),
        # 注意：epsilon_su8 使用 PHYSICS 中的 epsilon_r（SU-8 相对介电常数）
        # 但 PHYSICS 中 epsilon_r 已被更新为四层串联等效值（12.0）
        # 这里使用 PhysicsConfig 中的 epsilon_su8 字段（3.28）
        "epsilon_su8": PHYSICS.get("epsilon_su8", 3.28),  # SU-8 相对介电常数
        "epsilon_teflon": PHYSICS.get("epsilon_h", 1.934),
        # 注意：d_su8 使用 PHYSICS 中的 d_dielectric（介电层厚度）
        # 但 PHYSICS 中 d_dielectric 已被更新为有效值（8e-7）
        # 这里使用 PhysicsConfig 中的 d_su8 字段（400e-9）
        "d_su8": PHYSICS.get("d_su8", 400e-9),  # SU-8 厚度 (m) = 400nm
        "d_teflon": PHYSICS.get("d_hydrophobic", 400e-9),
        "A_eff": 1.20,  # PHYSICS 中无对应
        "dielectric_thickness": PHYSICS.get("d_dielectric", 8e-7),
        "relative_permittivity": PHYSICS.get("epsilon_r", 12.0),
        # 两相流属性
        "density_polar": PHYSICS.get("rho_polar", 998.0),
        "density_ink": PHYSICS.get("rho_oil", 763.0),
        "viscosity_polar": PHYSICS.get("mu_polar", 1.01e-3),
        "viscosity_ink": PHYSICS.get("mu_oil", 9.41e-4),
        "surface_tension_polar_ink": PHYSICS.get("sigma", 0.02505),
        # 接触角
        "contact_angle_theta0": PHYSICS.get("theta0", 120.0),
        "contact_angle_ink": PHYSICS.get("theta0", 120.0),
        "dynamic_contact_angle_advancing": PHYSICS.get("theta0", 120.0),
        "dynamic_contact_angle_receding": 100.0,
        "theta_wall": PHYSICS.get("theta_wall", 71.0),
        "contact_line_friction": 1e-3,
        "pinning_energy": 1e-5,
        "slip_length": 1e-6,
        # 几何参数
        "ink_thickness": PHYSICS.get("h_ink", 3e-6),
        "domain_height": PHYSICS.get("Lz", 20e-6),
        "wall_height": PHYSICS.get("wall_height", 3.5e-6),
        "ink_initial_fraction": PHYSICS.get("ink_initial_fraction", 0.15),
        "ink_potential_min": 0.0,
        # 电润湿 EW 力参数
        "lambda_debye": PHYSICS.get("lambda_debye", 50e-9),
        # 物理模型配置开关
        "use_convection": PHYSICS.get("use_convection", False),
        "use_unified_wetting": PHYSICS.get("use_unified_wetting", False),
        "use_legacy_ac": False,  # 保留（PHYSICS 中无对应）
        "use_adaptive_loss_scale": False,  # 保留（PHYSICS 中无对应）
        # Allen-Cahn 相场参数
        "ac_interface_width": PHYSICS.get("ac_interface_width", 5e-07),
        "ac_mobility": PHYSICS.get("ac_mobility", 1e-10),
        "electrowetting_weight": PHYSICS.get("electrowetting_weight", 1.0),
        # 侧壁 Teflon 污染接触角
        "theta_wall_teflon": PHYSICS.get("theta_wall_teflon", 110.0),
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

    def _compute_capacitance(self, with_oil=False):
        eps0 = self.materials_params.get("epsilon_0", 8.854e-12)
        eps_su8 = self.materials_params.get("epsilon_su8", 3.28)
        eps_teflon = self.materials_params.get("epsilon_teflon", 1.934)
        d_su8 = self.materials_params.get("d_su8", 400e-9)
        d_teflon = self.materials_params.get("d_teflon", 400e-9)
        A_eff = self.materials_params.get("A_eff", 1.20)

        Z = d_su8 / (eps0 * eps_su8) + d_teflon / (eps0 * eps_teflon)

        if with_oil:
            eps_oil = self.materials_params.get("epsilon_ink", 4.0)
            h_ink = self.materials_params.get("ink_thickness", 3e-6)
            Z = Z + h_ink / (eps0 * eps_oil)

        return A_eff / Z

    def compute_navier_stokes_residual(self, x, predictions, model=None, grads=None):
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
            phi = predictions[:, 4] if predictions.shape[1] >= 5 else torch.zeros_like(u)

            # 材料参数 (基于 PHYSICS 常量)
            rho_oil = self.materials_params["density_ink"]
            rho_polar = self.materials_params["density_polar"]
            mu_oil = self.materials_params["viscosity_ink"]
            mu_polar = self.materials_params["viscosity_polar"]
            sigma = self.materials_params["surface_tension_polar_ink"]

            # 电润湿参数
            self.materials_params.get("Lx", 174e-6)
            self.materials_params.get("Ly", 174e-6)
            self.materials_params.get("domain_height", 20e-6)

            # 双层串联电容 (SU-8 + Teflon)，含有效面积校正因子 A_eff
            C_ew = self._compute_capacitance()

            # 混合属性
            rho = phi * rho_oil + (1 - phi) * rho_polar
            mu = phi * mu_oil + (1 - phi) * mu_polar

            # 从统一梯度计算结果读取（由 compute_core_residuals 提供）
            if grads is not None:
                u_x, u_y, u_z = grads["u_x"], grads["u_y"], grads["u_z"]
                v_x, v_y, v_z = grads["v_x"], grads["v_y"], grads["v_z"]
                w_x, w_y, w_z = grads["w_x"], grads["w_y"], grads["w_z"]
                p_x, p_y, p_z = grads["p_x"], grads["p_y"], grads["p_z"]
                phi_x, phi_y, phi_z = grads["phi_x"], grads["phi_y"], grads["phi_z"]
                u_t, v_t, w_t = grads["u_t"], grads["v_t"], grads["w_t"]
                lap_u, lap_v, lap_w = grads["lap_u"], grads["lap_v"], grads["lap_w"]
                lap_phi = grads["lap_phi"]
                phi_xx, phi_yy, phi_zz = grads["phi_xx"], grads["phi_yy"], grads["phi_zz"]
                phi_xy, phi_xz, phi_yz = grads["phi_xy"], grads["phi_xz"], grads["phi_yz"]
            else:
                # 回退：独立计算（兼容直接调用场景）
                g_u = compute_gradient(u.sum(), x)
                g_v = compute_gradient(v.sum(), x)
                g_w = compute_gradient(w.sum(), x)
                g_p = compute_gradient(p.sum(), x)
                g_phi = compute_gradient(phi.sum(), x)
                u_x, u_y, u_z = g_u[:, 0], g_u[:, 1], g_u[:, 2]
                v_x, v_y, v_z = g_v[:, 0], g_v[:, 1], g_v[:, 2]
                w_x, w_y, w_z = g_w[:, 0], g_w[:, 1], g_w[:, 2]
                p_x, p_y, p_z = g_p[:, 0], g_p[:, 1], g_p[:, 2]
                phi_x, phi_y, phi_z = g_phi[:, 0], g_phi[:, 1], g_phi[:, 2]
                u_t = g_u[:, 5] if x.shape[1] >= 6 else torch.zeros_like(u)
                v_t = g_v[:, 5] if x.shape[1] >= 6 else torch.zeros_like(v)
                w_t = g_w[:, 5] if x.shape[1] >= 6 else torch.zeros_like(w)
                lap_u = self._compute_laplacian(u, x)
                lap_v = self._compute_laplacian(v, x)
                lap_w = self._compute_laplacian(w, x)
                lap_phi = self._compute_laplacian(phi, x)
                phi_xx = lap_phi
                phi_yy = lap_phi
                phi_zz = lap_phi
                phi_xy = lap_phi
                phi_xz = lap_phi
                phi_yz = lap_phi

            # 表面张力 (CSF 模型 - 精确曲率)

            _grad_phi_mag_sq, _grad_phi_mag = gradient_magnitude(phi_x, phi_y, phi_z)

            # 精确曲率公式: kappa = -div(grad(phi)/|grad(phi)|)
            kappa = mean_curvature_3d(phi_x, phi_y, phi_z, phi_xx, phi_yy, phi_zz, phi_xy, phi_xz, phi_yz)

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
            x[:, 0]
            x[:, 1]
            z_coord = x[:, 2]

            # 电润湿力作用在底面 (z=0)
            # z 方向衰减尺度: 用油墨厚度 h_ink 作为特征长度
            # 电润湿力是界面力，作用在油墨-极性液体界面附近，尺度 ~h_ink (3μm)
            # 与 EW 残差 (compute_electrowetting_residual) 保持一致
            h_ink_ns = self.materials_params.get("ink_thickness", 3e-6)
            z_decay = torch.exp(-z_coord / h_ink_ns)

            # 电润湿体积力: f_ew = ½ C_ew V_eff² * z_decay * ∇φ
            # 量纲: [N/m] * [1/m] = [N/m³] ✅
            # 方向: 沿 ∇φ（从水指向油），极性液体推动油墨向外
            # z_decay: 在底面 ~100nm 内急剧衰减（德拜屏蔽）
            f_ew_x = f_ew_magnitude * z_decay * phi_x
            f_ew_y = f_ew_magnitude * z_decay * phi_y
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
            viscous_u = mu * lap_u + mu_x * (2 * u_x) + mu_y * (u_y + v_x) + mu_z * (u_z + w_x)
            viscous_v = mu * lap_v + mu_x * (v_x + u_y) + mu_y * (2 * v_y) + mu_z * (v_z + w_y)
            viscous_w = mu * lap_w + mu_x * (w_x + u_z) + mu_y * (w_y + v_z) + mu_z * (2 * w_z)

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
            logger.error(f"Navier-Stokes残差计算失败: {e!s}")
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

    def compute_volume_conservation_residual(self, x_phys: torch.Tensor, predictions: torch.Tensor):
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            batch_size = predictions.shape[0] if isinstance(predictions, torch.Tensor) else 1

            residuals = {
                "volume_conservation": torch.zeros(batch_size, device=device, requires_grad=True),
                "volume_consistency": torch.zeros(batch_size, device=device, requires_grad=True),
                "ink_potential_min": torch.zeros(batch_size, device=device, requires_grad=True),
            }

            if isinstance(predictions, torch.Tensor) and predictions.shape[1] >= 5:
                alpha = predictions[:, 4]
                alpha_clamped = torch.clamp(alpha, 0.0, 1.0)
                base_consistency = alpha - alpha_clamped

                ink_fraction_target = self.materials_params.get("ink_initial_fraction", 0.15)
                alpha_mean = torch.mean(alpha_clamped)
                global_volume_residual = (alpha_mean - ink_fraction_target) / max(ink_fraction_target, 1e-6)
                global_volume_residual_tensor = global_volume_residual.expand(batch_size)

                overflow_penalty = torch.zeros(batch_size, device=device, requires_grad=True)
                if isinstance(x_phys, torch.Tensor) and x_phys.dim() == 2 and x_phys.size(1) >= 3:
                    coords = x_phys.detach()
                    x = coords[:, 0]
                    y = coords[:, 1]
                    z = coords[:, 2]

                    x_min, x_max = torch.min(x), torch.max(x)
                    y_min, y_max = torch.min(y), torch.max(y)

                    Lx = (x_max - x_min).clamp(min=1e-9)
                    Ly = (y_max - y_min).clamp(min=1e-9)

                    self.materials_params.get("ink_thickness", 3e-6)
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
                residuals["volume_consistency"] = base_consistency + overflow_penalty

            if isinstance(predictions, torch.Tensor) and predictions.shape[1] >= 6:
                ink_potential = predictions[:, 5]
                min_potential = self.materials_params.get("ink_potential_min", 0.0)
                residuals["ink_potential_min"] = torch.nn.functional.relu(min_potential - ink_potential)

            return residuals

        except Exception as e:
            logger.error(f"计算体积守恒残差失败: {e!s}")
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {
                "volume_conservation": torch.zeros(batch_size, device=device, requires_grad=True),
                "volume_consistency": torch.zeros(batch_size, device=device, requires_grad=True),
                "ink_potential_min": torch.zeros(batch_size, device=device, requires_grad=True),
            }

    def compute_sidewall_contact_angle_residual(self, x_phys: torch.Tensor, predictions: torch.Tensor, grads=None):
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
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            zero = torch.zeros(batch_size, device=device, requires_grad=True)
            residuals = {"sidewall_contact_angle": zero}

            if not (isinstance(x_phys, torch.Tensor) and isinstance(predictions, torch.Tensor)):
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

            # 从统一梯度计算结果读取 phi 梯度
            if grads is not None:
                grad_x = grads["phi_x"]
                grad_y = grads["phi_y"]
                grad_z = grads["phi_z"]
            else:
                try:
                    grad_all = torch.autograd.grad(phi.sum(), x_phys, create_graph=True, retain_graph=True)[0]
                except Exception:
                    return residuals
                if grad_all is None:
                    return residuals
                grad_x = grad_all[:, 0]
                grad_y = grad_all[:, 1]
                grad_z = grad_all[:, 2]

            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2) + 1e-10

            # 壁面法向分量
            n_x = grad_x / grad_mag
            n_y = grad_y / grad_mag

            # 壁面法向: x=0 → (1,0), x=Lx → (-1,0), y=0 → (0,1), y=Ly → (0,-1)
            wall_nx = torch.where(
                x < margin,
                torch.ones_like(n_x),
                torch.where(x > Lx - margin, -torch.ones_like(n_x), torch.zeros_like(n_x)),
            )
            wall_ny = torch.where(
                y < margin,
                torch.ones_like(n_y),
                torch.where(y > Ly - margin, -torch.ones_like(n_y), torch.zeros_like(n_y)),
            )

            dot = n_x * wall_nx + n_y * wall_ny

            # 接触角滞后 (CAH): 接触线沿围堰壁顶移动时的动态接触角
            # 物理位置: 围堰壁顶部(z≈3.5μm)与顶面交界处
            # u·n̂_wall > 0 → 油墨被推向壁外 → 前进角 θ_A (更大)
            # u·n̂_wall < 0 → 油墨被拉回壁内 → 后退角 θ_R (更小)
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

            # 接触线位置: 围堰壁顶部与顶面交界处 (z ≈ wall_height)
            wall_height = self.materials_params.get("wall_height", 3.5e-6)
            is_wall_top = (z > wall_height - 1e-6) & (z < wall_height + 1e-6)
            contact_line_mask = is_wall_top & near_wall & (interface_weight > 0.01)

            # 接触线约束: 在壁顶接触线区域施加约束
            if contact_line_mask.any():
                residual = ((dot - target_cos) ** 2) * contact_line_mask.float() * interface_weight
                n_active = contact_line_mask.float().sum().clamp(min=1)
            else:
                residual = ((dot - target_cos) ** 2) * mask.float() * interface_weight
                n_active = mask.float().sum().clamp(min=1)

            residuals["sidewall_contact_angle"] = residual.sum() / n_active

            return residuals
        except Exception as e:
            logger.warning(f"壁面接触角残差计算失败: {e}")
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {"sidewall_contact_angle": torch.zeros(batch_size, device=device, requires_grad=True)}

    def compute_laplace_pressure_residual(self, x_phys: torch.Tensor, predictions: torch.Tensor, grads=None):
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
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {"laplace_pressure": zero}

            if not (isinstance(x_phys, torch.Tensor) and isinstance(predictions, torch.Tensor)):
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

            # 从统一梯度计算结果读取
            if grads is not None:
                phi_x = grads["phi_x"]
                phi_y = grads["phi_y"]
                phi_z = grads["phi_z"]
                _grad_mag_sq, grad_mag = gradient_magnitude(phi_x, phi_y, phi_z)
                laplacian_xy = grads["phi_xx"] + grads["phi_yy"]
            else:
                # 回退：独立计算
                try:
                    grad_phi = torch.autograd.grad(phi.sum(), x_phys, create_graph=True, retain_graph=True)[0]
                except Exception:
                    return res
                if grad_phi is None:
                    return res
                phi_x = grad_phi[:, 0]
                phi_y = grad_phi[:, 1]
                phi_z = grad_phi[:, 2]
                _grad_mag_sq, grad_mag = gradient_magnitude(phi_x, phi_y, phi_z)
                try:
                    lap_x = compute_gradient(phi_x.sum(), x_phys)[:, 0]
                    lap_y = compute_gradient(phi_y.sum(), x_phys)[:, 1]
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
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            return {"laplace_pressure": torch.zeros(1, device=device, requires_grad=True)}

    def compute_electrowetting_residual(self, x_phys, predictions, grads=None):
        """电润湿驱动力残差 — 已移除

        EW 驱动力已整合到 AC 方程（_compute_vof_residual）的 ew_source 项中，
        直接驱动相场演化，不再需要独立残差。

        保留此方法以保持 compute_core_residuals 调用兼容，但始终返回零。
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            return {"electrowetting": torch.zeros(1, device=device, requires_grad=True)}
        except Exception:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            return {"electrowetting": torch.zeros(1, device=device, requires_grad=True)}

    def compute_interface_energy_residual(self, x_phys, predictions, grads=None):
        """界面能 — 纯 sigma*|grad(phi)|, 塑造圆润液滴

        电润湿项已拆分到 compute_electrowetting_residual。
        此项只做表面张力最小化(最小界面面积 = 圆润形状)。
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            zero = torch.zeros(x_phys.shape[0], device=device, requires_grad=True)
            res = {"interface_energy": zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            phi = predictions[:, 4]
            # 从统一梯度计算结果读取 phi 梯度
            if grads is not None:
                grad_mag = torch.sqrt(grads["phi_x"] ** 2 + grads["phi_y"] ** 2 + grads["phi_z"] ** 2)
            else:
                try:
                    g = torch.autograd.grad(phi.sum(), x_phys, create_graph=True, retain_graph=True)[0]
                except Exception:
                    return res
                if g is None:
                    return res
                grad_mag = torch.norm(g[:, :3], dim=1)

            sigma = self.materials_params.get("surface_tension_polar_ink", 0.02505)
            res["interface_energy"] = sigma * grad_mag
            return res
        except Exception as e:
            logger.warning(f"界面能失败: {e}")
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            return {"interface_energy": torch.zeros(x_phys.shape[0], device=device, requires_grad=True)}

    def compute_wall_wetting_residual(self, x_phys: torch.Tensor, predictions: torch.Tensor, grads=None):
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            batch_size = predictions.shape[0] if isinstance(predictions, torch.Tensor) else 1
            residuals = {"wall_wetting": torch.zeros(batch_size, device=device, requires_grad=True)}
            if not (isinstance(x_phys, torch.Tensor) and isinstance(predictions, torch.Tensor)):
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
            logger.error(f"计算墙润湿残差失败: {e!s}")
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1
            return {"wall_wetting": torch.zeros(batch_size, device=device, requires_grad=True)}

    def _compute_unified_wetting_bc(self, x_phys, predictions):
        """
        统一相场润湿边界条件 — 基于能量泛函 F[φ] 的自然 BC

        Bulk: F[φ] = ∫[σ/ε·f(φ) + σ·ε/2·|∇φ|²]dV + ∫f_w(φ)dS
        Wall: f_w(φ) = +σ·cos(θ_eq)·φ²(3-2φ)
           θ_eq > 90° (亲油): cos < 0 → φ=1 能量低
           θ_eq < 90° (亲水): cos > 0 → φ=0 能量低

        变分得到自然 BC: ε·n·∇φ + cos(θ_eq)·6φ(1-φ) = 0
          底面 (z=0, n=(0,0,-1)): ε·φ_z - cos(θ_eq)·6φ(1-φ) = 0
          侧壁: ε·n_wall·∇φ - cos(θ_wall)·6φ(1-φ) = 0

        cos(θ_eq) 通过 Young-Lippmann: cos(θ_eq) = cos(θ₀) + C_yl·V_eff²/(2σ)

        与 bulk AC 方程共享 σ, ε 系数，确保量级一致。
        界面加权: exp(-100*(φ-0.5)²) — 体相连续消失。
        归一化: sum(w·R²)/sum(w)，不对全 batch 平均。
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1

            if not isinstance(x_phys, torch.Tensor) or not isinstance(predictions, torch.Tensor):
                return {"phase_field_wetting": torch.tensor(0.0, device=device)}
            if x_phys.dim() != 2 or x_phys.size(1) < 3:
                return {"phase_field_wetting": torch.tensor(0.0, device=device)}
            if predictions.dim() < 2 or predictions.size(1) < 5:
                return {"phase_field_wetting": torch.tensor(0.0, device=device)}

            # 物理参数
            sigma = self.materials_params.get("surface_tension_polar_ink", 0.02505)
            eps = self.materials_params.get("ac_interface_width", 5e-07)
            theta0 = self.materials_params.get("contact_angle_theta0", 120.0)
            V_T_base = self.materials_params.get("V_T_base", 5.0)
            theta_wall_teflon = self.materials_params.get("theta_wall_teflon", 110.0)

            # 双层串联电容 (SU-8 + Teflon)，含有效面积校正因子 A_eff
            C_yl = self._compute_capacitance()

            # 提取坐标和 φ
            x_coord = x_phys[:, 0]
            y_coord = x_phys[:, 1]
            z_coord = x_phys[:, 2]
            V_to = x_phys[:, 4] if x_phys.shape[1] >= 5 else torch.zeros_like(x_coord)
            phi = predictions[:, 4]

            # 界面加权: w = exp(-100*(φ-0.5)²)
            interface_w = torch.exp(-100.0 * (phi - 0.5) ** 2)

            X = self.materials_params.get("Lx", 174e-6)
            Ly = self.materials_params.get("Ly", 174e-6)

            # 安全梯度计算
            x_grad = x_phys.clone().detach().requires_grad_(True) if not x_phys.requires_grad else x_phys

            g_phi = torch.autograd.grad(
                phi.sum(),
                x_grad,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g_phi is None:
                return {"phase_field_wetting": torch.tensor(0.0, device=device)}

            phi_x = g_phi[:, 0]
            phi_y = g_phi[:, 1]
            phi_z = g_phi[:, 2]

            # ---- 底面 BC (z=0): ε·φ_z - cos(θ_eq)·6φ(1-φ) = 0 ----
            # 注意减号: n=(0,0,-1) → n·∇φ = -φ_z
            # Young-Lippmann: cos(θ_eq) = cos(θ₀) + C_yl·V_eff²/(2σ)
            V_eff = torch.clamp(V_to - V_T_base, min=0.0)
            cos_theta0 = np.cos(np.radians(theta0))
            cos_theta_eq = cos_theta0 + C_yl * V_eff**2 / (2.0 * sigma)
            cos_theta_eq = torch.clamp(cos_theta_eq, -1.0, 1.0)

            f_wall = 6.0 * phi * (1.0 - phi)
            bc_bottom = eps * phi_z - cos_theta_eq * f_wall

            # 底面 mask: z < 0.5 * h_ink
            h_ink = self.materials_params.get("ink_thickness", 3e-6)
            near_bottom = (z_coord < 0.5 * h_ink).float()
            w_bottom = interface_w * near_bottom

            # ---- 侧壁 BC: ε·n·∇φ - cos(θ_wall_teflon)·6φ(1-φ) = 0 ----
            cos_theta_wall = np.cos(np.radians(theta_wall_teflon))

            near_x0 = (x_coord < 0.1 * X).float()
            near_xX = (x_coord > 0.9 * X).float()
            near_y0 = (y_coord < 0.1 * Ly).float()
            near_yY = (y_coord > 0.9 * Ly).float()

            bc_x0 = -eps * phi_x - cos_theta_wall * f_wall
            bc_xX = eps * phi_x - cos_theta_wall * f_wall
            bc_y0 = -eps * phi_y - cos_theta_wall * f_wall
            bc_yY = eps * phi_y - cos_theta_wall * f_wall

            w_x0 = interface_w * near_x0
            w_xX = interface_w * near_xX
            w_y0 = interface_w * near_y0
            w_yY = interface_w * near_yY

            # ---- 加权损失 ----
            total_w = w_bottom + w_x0 + w_xX + w_y0 + w_yY
            total_w_sum = total_w.sum().clamp(min=1e-12)

            loss = w_bottom * bc_bottom**2 + w_x0 * bc_x0**2 + w_xX * bc_xX**2 + w_y0 * bc_y0**2 + w_yY * bc_yY**2
            scalar_loss = loss.sum() / total_w_sum

            return {"phase_field_wetting": scalar_loss}

        except Exception as e:
            logger.warning(f"统一润湿 BC 计算异常: {e}")
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            return {"phase_field_wetting": torch.tensor(0.0, device=device)}

    def _compute_dielectric_charge_residual(self, x_phys, predictions, grads=None):
        """介电层RC充电约束 — z=0底面

        物理: 介电层电荷积累遵循RC电路模型
          σ(t) = C_dielectric × V × (1 - exp(-t/τ_RC))
          电润湿力在介电层充电完成后才完全建立
          τ_RC = ε₀ε_r / σ_conduct ≈ 25μs (远快于机械响应)

        约束: 在t < 3τ_RC的早期时刻，底面电润湿能应遵循充电曲线
          G_ew(t) = G_ew_steady × (1 - exp(-t/τ_RC))²
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {"dielectric_charge": zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            if x_phys.shape[1] < 6:
                return res
            phi = predictions[:, 4]
            z = x_phys[:, 2]
            V_to = x_phys[:, 4]
            t_since = x_phys[:, 5]
            is_bottom = z < 1e-6
            if not is_bottom.any():
                return res

            phi_b = phi[is_bottom]
            V_b = V_to[is_bottom]
            t_b = t_since[is_bottom]
            V_T = self.materials_params.get("V_T_base", 5.0)
            V_eff = torch.clamp(V_b - V_T, min=0.0)

            # τ_RC: 介电层RC时间常数 (SU-8电阻率≈10¹⁴Ω·cm → τ_RC≈25μs)
            tau_rc = self.materials_params.get("charge_relaxation_time", 2.5e-5)

            # RC充电曲线: (1-exp(-t/τ_RC))²
            charge_factor = (1.0 - torch.exp(-t_b / tau_rc)) ** 2
            # 稳态电润湿能: ½·C·V²
            # 双层串联电容 (SU-8 + Teflon)，含有效面积校正因子 A_eff
            C_ew = self._compute_capacitance()
            G_steady = 0.5 * C_ew * V_eff**2
            G_expected = G_steady * charge_factor

            # 约束: φ_b × G_steady (电润湿推动力) 不应超过 RC 充电水平
            # 即: φ_b × G_steady ≤ (1 - φ_b) × G_expected
            # 残差: relu(φ_b × G_steady - (1 - φ_b) × G_expected)
            ew_driving = phi_b * G_steady
            ew_available = (1.0 - phi_b) * G_expected
            residual = torch.relu(ew_driving - ew_available)

            n_bottom = is_bottom.float().sum().clamp(min=1)
            res["dielectric_charge"] = residual.sum() / n_bottom
            return res
        except Exception as e:
            logger.warning(f"介电电荷残差失败: {e}")
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            return {"dielectric_charge": torch.zeros(1, device=device, requires_grad=True)}

    def _compute_contact_line_dynamics_residual(self, x_phys, predictions, grads=None):
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
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {"contact_line_dynamics": zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            if x_phys.shape[1] < 6:
                return res
            phi = predictions[:, 4]
            z = x_phys[:, 2]
            V_to = x_phys[:, 4]
            # 接触线区域: z=0 且 φ≈0.5
            is_bottom = z < 1e-6
            is_interface = (phi > 0.2) & (phi < 0.8)
            mask = is_bottom & is_interface
            if mask.sum() < 3:
                return res

            # 从统一梯度计算结果读取 phi 梯度
            if grads is not None:
                g_x = grads["phi_x"]
                g_y = grads["phi_y"]
                g_z = grads["phi_z"]
                phi_t = grads["phi_t"]
            else:
                try:
                    g = torch.autograd.grad(phi.sum(), x_phys, create_graph=True, retain_graph=True)[0]
                except Exception:
                    return res
                if g is None:
                    return res
                g_x, g_y, g_z = g[:, 0], g[:, 1], g[:, 2]
                phi_t = g[:, 5] if g.shape[1] >= 6 else torch.zeros_like(phi)

            phi_z = g_z
            grad_mag = torch.sqrt(g_x**2 + g_y**2 + g_z**2) + 1e-10

            v_cl = phi_t[mask] / grad_mag[mask]
            cos_local = phi_z[mask] / grad_mag[mask]

            # cos(θ_eq) from Young-Lippmann
            theta0_deg = self.materials_params.get("contact_angle_theta0", 120.0)
            sigma_po = self.materials_params.get(
                "surface_tension_polar_ink", self.materials_params.get("sigma", 0.02505)
            )
            # 双层串联电容 (SU-8 + Teflon)，含有效面积校正因子 A_eff
            C_yl = self._compute_capacitance()

            cos_theta0 = np.cos(np.radians(theta0_deg))
            V_m = V_to[mask]
            V_T = self.materials_params.get("V_T_base", 5.0)
            V_eff = torch.clamp(V_m - V_T, min=0.0)
            ew_term = C_yl * V_eff**2 / (2 * sigma_po)
            cos_eq = torch.clamp(torch.tensor(cos_theta0, device=device) + ew_term, -1.0, 1.0)

            # HVT: v_cl ∝ cos_eq - cos_local
            k_cl = self.materials_params.get("contact_line_friction", 1e-3)
            residual = v_cl - k_cl * (cos_eq - cos_local)

            res["contact_line_dynamics"] = torch.mean(residual**2)
            return res
        except Exception as e:
            logger.warning(f"接触线动力学残差失败: {e}")
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            return {"contact_line_dynamics": torch.zeros(1, device=device, requires_grad=True)}

    def _compute_top_boundary_residual(self, x_phys, predictions, grads=None):
        """顶面自由表面边界条件 (z≈Lz)

        物理: 极性液体-气界面
        - 零剪切: du/dz≈0, dv/dz≈0 (无表面应力)
        - 无穿透: w=0 (界面不移动)
        - φ=0 (纯极性液体在顶部, 油墨在下层)
        """
        try:
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            zero = torch.zeros(1, device=device, requires_grad=True)
            res = {"top_boundary": zero}
            if not (isinstance(x_phys, torch.Tensor) and predictions.dim() >= 2):
                return res
            if x_phys.shape[1] < 3:
                return res
            Lz = self.materials_params.get("domain_height", 20e-6)
            z = x_phys[:, 2]
            is_top = z > (Lz - 1e-6)
            if not is_top.any():
                return res

            predictions[is_top, 0]
            predictions[is_top, 1]
            w_t = predictions[is_top, 2]
            phi_t = predictions[is_top, 4]

            # 从统一梯度计算结果读取 u_z, v_z
            if grads is not None:
                du_dz = grads["u_z"][is_top]
                dv_dz = grads["v_z"][is_top]
            else:
                try:
                    grad_u = torch.autograd.grad(predictions[:, 0].sum(), x_phys, create_graph=True, retain_graph=True)[
                        0
                    ]
                    grad_v = torch.autograd.grad(predictions[:, 1].sum(), x_phys, create_graph=True, retain_graph=True)[
                        0
                    ]
                except Exception:
                    return res
                if grad_u is None or grad_v is None:
                    return res
                du_dz = grad_u[is_top, 2]
                dv_dz = grad_v[is_top, 2]

            shear = (du_dz**2 + dv_dz**2).mean()
            normal = (w_t**2).mean()
            phi_ok = (phi_t**2).mean()

            res["top_boundary"] = shear + normal + 0.5 * phi_ok
            return res
        except Exception as e:
            logger.warning(f"顶面BC失败: {e}")
            device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
            return {"top_boundary": torch.zeros(1, device=device, requires_grad=True)}

    def safe_compute_laplacian_spatial(self, scalar_field: torch.Tensor, coords: torch.Tensor, spatial_dims: int = 3):
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
            if not isinstance(output, torch.Tensor) or not isinstance(input_tensor, torch.Tensor):
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

    def _compute_all_gradients(self, x_phys: torch.Tensor, predictions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        统一计算所有一阶和二阶梯度。

        一次性计算 u,v,w,p,phi 的所有空间梯度和时间导数，
        以及 Laplacian 和二阶混合导数（用于曲率）。

        Returns:
            grads dict，键名格式：
            一阶: u_x, u_y, u_z, u_t, v_x, ..., phi_x, phi_y, phi_z, phi_t
            二阶: lap_u, lap_v, lap_w, lap_phi
                   phi_xx, phi_yy, phi_zz, phi_xy, phi_xz, phi_yz
        """
        device = x_phys.device
        batch_size = x_phys.shape[0]
        grads = {}

        def get_grad(y, x_in):
            return compute_gradient(y, x_in)

        u = predictions[:, 0]
        v = predictions[:, 1]
        w = predictions[:, 2]
        p = predictions[:, 3]
        phi = predictions[:, 4] if predictions.shape[1] >= 5 else torch.zeros_like(u)

        # 一阶梯度
        g_u = get_grad(u.sum(), x_phys)
        g_v = get_grad(v.sum(), x_phys)
        g_w = get_grad(w.sum(), x_phys)
        g_p = get_grad(p.sum(), x_phys)
        g_phi = get_grad(phi.sum(), x_phys)

        # 空间梯度
        for prefix, g in [("u", g_u), ("v", g_v), ("w", g_w), ("p", g_p), ("phi", g_phi)]:
            grads[f"{prefix}_x"] = g[:, 0]
            grads[f"{prefix}_y"] = g[:, 1]
            grads[f"{prefix}_z"] = g[:, 2]

        # 时间导数 (t_since 在索引 5)
        n_coord = x_phys.shape[1]
        for prefix, g in [("u", g_u), ("v", g_v), ("w", g_w), ("p", g_p), ("phi", g_phi)]:
            grads[f"{prefix}_t"] = g[:, 5] if n_coord >= 6 else torch.zeros_like(u)

        # Laplacian 和二阶导数
        def compute_laplacian_and_hessian(first_grad, compute_hessian=False):
            lap = torch.zeros(batch_size, device=device)
            hess = {}
            for i in range(3):
                g2 = get_grad(first_grad[:, i].sum(), x_phys)
                lap += g2[:, i]
                if compute_hessian:
                    for j in range(3):
                        key = "xyz"[i] + "xyz"[j]
                        hess[key] = g2[:, j]
            return lap, hess

        lap_u, _ = compute_laplacian_and_hessian(g_u)
        lap_v, _ = compute_laplacian_and_hessian(g_v)
        lap_w, _ = compute_laplacian_and_hessian(g_w)
        grads["lap_u"] = lap_u
        grads["lap_v"] = lap_v
        grads["lap_w"] = lap_w

        # phi 需要完整 Hessian（用于曲率计算）
        lap_phi, phi_hess = compute_laplacian_and_hessian(g_phi, compute_hessian=True)
        grads["lap_phi"] = lap_phi
        grads["phi_xx"] = phi_hess["xx"]
        grads["phi_yy"] = phi_hess["yy"]
        grads["phi_zz"] = phi_hess["zz"]
        grads["phi_xy"] = phi_hess["xy"]
        grads["phi_xz"] = phi_hess["xz"]
        grads["phi_yz"] = phi_hess["yz"]

        return grads

    def compute_core_residuals(
        self,
        x_phys: torch.Tensor,
        predictions: torch.Tensor,
        model: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        统一物理残差计算入口 - 作为"唯一物理真相"

        先统一计算所有梯度，再分发给各约束子方法。
        避免重复 autograd 调用，训练速度提升 3-4 倍。
        """
        device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
        batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1

        # === 统一梯度计算（一次性完成所有 autograd） ===
        try:
            grads = self._compute_all_gradients(x_phys, predictions)
        except Exception as e:
            logger.warning(f"梯度计算失败: {e}")
            return self._empty_residual(x_phys, predictions)

        residuals = {}

        # 1. Navier-Stokes 残差（包含连续性和动量方程）
        try:
            ns_residuals = self.compute_navier_stokes_residual(x_phys, predictions, model=model, grads=grads)
            residuals.update(ns_residuals)
        except Exception as e:
            logger.warning(f"N-S 残差计算失败: {e}")
            residuals.update(self._empty_residual(x_phys, predictions))

        # 2. VOF 方程残差（体积分数输运）
        try:
            vof_residual = self._compute_vof_residual(x_phys, predictions, model=model, grads=grads)
            residuals["vof"] = vof_residual
        except Exception as e:
            logger.warning(f"VOF 残差计算失败: {e}")
            residuals["vof"] = torch.zeros(batch_size, device=device)

        # 3. 体积守恒残差
        try:
            vc_residuals = self.compute_volume_conservation_residual(x_phys, predictions)
            residuals.update(vc_residuals)
        except Exception as e:
            logger.warning(f"体积守恒残差计算失败: {e}")

        # 4. 电润湿驱动力
        try:
            ew_residuals = self.compute_electrowetting_residual(x_phys, predictions, grads=grads)
            residuals.update(ew_residuals)
        except Exception as e:
            logger.warning(f"电润湿残差失败: {e}")

        # 5. 界面能
        try:
            ie_residuals = self.compute_interface_energy_residual(x_phys, predictions, grads=grads)
            residuals.update(ie_residuals)
        except Exception as e:
            logger.warning(f"界面能约束计算失败: {e}")

        # 6. Laplace 压力一致约束
        try:
            lp_residuals = self.compute_laplace_pressure_residual(x_phys, predictions, grads=grads)
            residuals.update(lp_residuals)
        except Exception as e:
            logger.warning(f"Laplace 压力残差计算失败: {e}")

        # 7. 壁面接触角约束
        if not self.materials_params.get("use_unified_wetting", False):
            try:
                sw_residuals = self.compute_sidewall_contact_angle_residual(x_phys, predictions, grads=grads)
                residuals.update(sw_residuals)
            except Exception as e:
                logger.warning(f"壁面接触角残差计算失败: {e}")

        # 8. 时间连续性正则化（需要 model 做三时间点前向）
        try:
            temporal_residual = self._compute_temporal_smoothness(x_phys, predictions, model=model)
            residuals["temporal_smoothness"] = temporal_residual
        except Exception as e:
            logger.warning(f"时间连续性残差计算失败: {e}")
            residuals["temporal_smoothness"] = torch.zeros(batch_size, device=device)

        # 9. 壁面润湿约束
        if not self.materials_params.get("use_unified_wetting", False):
            try:
                ww_residuals = self.compute_wall_wetting_residual(x_phys, predictions, grads=grads)
                residuals.update(ww_residuals)
            except Exception as e:
                logger.warning(f"壁面润湿残差计算失败: {e}")

        # 10. 统一相场润湿 BC
        if self.materials_params.get("use_unified_wetting", False):
            try:
                pfw_residuals = self._compute_unified_wetting_bc(x_phys, predictions, grads=grads)
                residuals.update(pfw_residuals)
            except Exception as e:
                logger.warning(f"统一润湿 BC 计算失败: {e}")

        # 11. 介电层RC充电约束
        try:
            dc_residuals = self._compute_dielectric_charge_residual(x_phys, predictions, grads=grads)
            residuals.update(dc_residuals)
        except Exception as e:
            logger.warning(f"介电电荷残差计算失败: {e}")

        # 12. 接触线动力学约束
        try:
            cld_residuals = self._compute_contact_line_dynamics_residual(x_phys, predictions, grads=grads)
            residuals.update(cld_residuals)
        except Exception as e:
            logger.warning(f"接触线动力学残差计算失败: {e}")

        # 13. 顶面自由表面边界条件
        try:
            tbc_residuals = self._compute_top_boundary_residual(x_phys, predictions, grads=grads)
            residuals.update(tbc_residuals)
        except Exception as e:
            logger.warning(f"顶面BC残差计算失败: {e}")

        # 14. 压力钉扎
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
        model: nn.Module | None = None,
        dt: float = 0.001,  # 1ms 时间步长
    ) -> torch.Tensor:
        """
        计算时间平滑性正则化残差 (基于二阶导数/加速度) — 批量前向版本。

        优化：将 t-dt, t, t+dt 三个时间点的输入拼接为一次大前向，
        替代原有的两次独立前向（t+dt 和 t-dt）。
        t 时刻的预测已在外层传入，无需重复计算。
        """
        device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
        batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1

        if model is None:
            return torch.zeros(batch_size, device=device)

        try:
            t_since = x_phys[:, 5]

            # 构造 t+dt 和 t-dt 的输入
            x_next = x_phys.clone()
            x_next[:, 5] = t_since + dt
            x_prev = x_phys.clone()
            x_prev[:, 5] = torch.clamp(t_since - dt, min=0.0)

            # 拼接为 (2*batch, 6)，一次前向
            pts_temporal = torch.cat([x_prev, x_next], dim=0)
            pred_temporal = model(pts_temporal)  # (2*batch, 5)

            pred_prev = pred_temporal[:batch_size]
            pred_next = pred_temporal[batch_size:]

            # t 时刻的预测
            u_t = predictions[:, 0]
            v_t = predictions[:, 1]
            w_t = predictions[:, 2]
            phi_t = predictions[:, 4] if predictions.shape[1] >= 5 else torch.zeros_like(u_t)

            # t+dt
            u_next = pred_next[:, 0]
            v_next = pred_next[:, 1]
            w_next = pred_next[:, 2]
            phi_next = pred_next[:, 4] if pred_next.shape[1] >= 5 else torch.zeros_like(u_next)

            # t-dt
            u_prev = pred_prev[:, 0]
            v_prev = pred_prev[:, 1]
            w_prev = pred_prev[:, 2]
            phi_prev = pred_prev[:, 4] if pred_prev.shape[1] >= 5 else torch.zeros_like(u_prev)

            # 二阶差分 (加速度/曲率)
            acc_u = (u_next - 2 * u_t + u_prev) ** 2
            acc_v = (v_next - 2 * v_t + v_prev) ** 2
            acc_w = (w_next - 2 * w_t + w_prev) ** 2
            vel_smoothness = acc_u + acc_v + acc_w
            phi_smoothness = (phi_next - 2 * phi_t + phi_prev) ** 2 * 50.0

            return vel_smoothness + phi_smoothness

        except Exception as e:
            logger.warning(f"时间连续性计算异常: {e}")
            return torch.zeros(batch_size, device=device)

    def _compute_vof_residual(
        self,
        x_phys: torch.Tensor,
        predictions: torch.Tensor,
        model: nn.Module | None = None,
        grads: dict[str, torch.Tensor] | None = None,
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
        device = x_phys.device if isinstance(x_phys, torch.Tensor) else torch.device("cpu")
        batch_size = x_phys.shape[0] if isinstance(x_phys, torch.Tensor) else 1

        try:
            if not isinstance(x_phys, torch.Tensor):
                return torch.zeros(batch_size, device=device)

            if not x_phys.requires_grad:
                x_phys = x_phys.clone().detach().requires_grad_(True)

            if model is not None:
                predictions = model(x_phys)

            u, v, w = predictions[:, 0], predictions[:, 1], predictions[:, 2]
            phi = predictions[:, 4] if predictions.shape[1] >= 5 else torch.zeros_like(u)

            # 从统一梯度计算结果读取（由 compute_core_residuals 提供）
            if grads is not None:
                phi_x = grads["phi_x"]
                phi_y = grads["phi_y"]
                phi_z = grads["phi_z"]
                phi_t = grads["phi_t"]
                lap_phi = grads["lap_phi"]
            else:
                # 回退：独立计算
                g_phi = torch.autograd.grad(
                    phi.sum(),
                    x_phys,
                    grad_outputs=torch.ones_like(phi),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if g_phi is None:
                    return torch.zeros(batch_size, device=device)
                phi_x, phi_y, phi_z = g_phi[:, 0], g_phi[:, 1], g_phi[:, 2]
                phi_t = g_phi[:, 5] if x_phys.shape[1] >= 6 else torch.zeros_like(phi)
                lap_phi = self.safe_compute_laplacian(phi, x_phys)

            # 对流项
            advection = phi_t + u * phi_x + v * phi_y + w * phi_z

            # 双阱势导数
            f_prime = 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

            # lap_phi 已从 grads 读取（或回退计算），不重复计算
            use_legacy = self.materials_params.get("use_legacy_ac", False)
            if use_legacy:
                gamma = self.materials_params.get("allen_cahn_gamma", 4.5e-7)
                ac_residual = advection - gamma * (lap_phi - f_prime)
            else:
                # 标准化相场模型: D_eff [ ε·∇²φ − (1/ε)·W'(φ) ]
                # 其中 D_eff = M_ac·σ·ε / L_ref² 是有效扩散系数 [1/s]
                # L_ref = ε (界面宽度) 作为特征长度
                # 量纲: advection [1/s] = D_eff [1/s] * (ε·∇²φ [1/m²] * L_ref² [m²]) ✓
                sigma_ac = self.materials_params.get("surface_tension_polar_ink", 0.02505)
                eps_ac = self.materials_params.get("ac_interface_width", 5e-07)
                M_ac = self.materials_params.get("ac_mobility", 1e-10)

                # 量纲修正:
                # M_ac [m³·s/kg] * sigma [kg/s²] / eps [m] = [m²/s] (扩散系数 D_eff)
                # D_eff / eps [m²/s / m] = [m/s] (迁移速度尺度 mob)
                # ac_residual = advection [1/s] - mob [m/s] * (eps*∇²φ [1/m] - f'/eps [1/m])
                # = [1/s] - [m/s * 1/m] = [1/s] - [1/s] ✓ 量纲匹配
                #
                # M_ac = 1e-11 → mob = 1e-11 * 0.025 / (5e-6)² = 0.01 m/s
                # 对应界面迁移时间 τ ~ eps/v = 5μm/0.01ms⁻¹ = 0.5ms (与电润湿 5ms 匹配)
                D_eff = M_ac * sigma_ac / eps_ac  # [m²/s] 扩散系数
                mob = D_eff / eps_ac  # [m/s] 迁移速度尺度

                # 标准 AC 方程残差: ∂φ/∂t + u·∇φ = mob·[ε·∇²φ − σ·W'(φ)/ε]
                ac_residual = advection - mob * (eps_ac * lap_phi - sigma_ac * f_prime / eps_ac)

            # === 电润湿驱动力源项 (直接加入 AC 方程) ===
            # 物理: 电润湿自由能 G_ew = -½·C(φ)·V²
            #   C(φ) = φ·C_ink + (1-φ)·C_open  (线性插值)
            #   δG_ew/δφ = ½·(C_open - C_ink)·V² = ½·delta_C·V²
            # AC 方程: ∂φ/∂t = -Γ·δF/δφ = ... - Γ·δG_ew/δφ
            # EW 源项: S_ew = -M_ac · ½ · delta_C · V_eff² · z_decay / eps_ac
            # 量纲: [m³·s/kg]·[F/m²]·[V²]/[m] = [m³·s/kg]·[J/m²]/[m] = [m³·s/kg]·[N/m²] = [m²/s]
            # 需要再除以 eps_ac → [m²/s]/[m] = [m/s]
            # 不对，让我重新算：
            # M_ac [m³·s/kg] * delta_C [F/m²] * V² [V²] / eps_ac [m]
            # = [m³·s/kg] * [C/V·m²] * [V²] / [m]
            # = [m³·s/kg] * [A·s/V·m²] * [V²] / [m]
            # = [m³·s/kg] * [A·s·V / m³]
            # = [m³·s/kg] * [J / m³]  (因为 J = A·s·V)
            # = [m³·s/kg] · [kg·m/s² · m / m³]
            # = [m²/s]
            # 所以 S_ew 量纲是 [m²/s]，需要再除以 eps_ac [m] 得到 [m/s]
            # 再除以 eps_ac [m] 得到 [1/s] ← 与 advection 匹配
            #
            # 归一化因子：S_ew = M_ac * delta_C * V² * z_decay / eps_ac²
            try:
                # 开口区域电容（SU-8 + Teflon），含有效面积校正因子 A_eff
                C_open = self._compute_capacitance(with_oil=False)
                # 油墨区域电容（SU-8 + Teflon + 油墨层串联），含 A_eff
                C_ink = self._compute_capacitance(with_oil=True)
                # 电容差
                delta_C = C_open - C_ink  # > 0

                # V_to 在 x_phys 索引 4
                V_to = x_phys[:, 4] if x_phys.shape[1] >= 5 else torch.zeros(batch_size, device=device)

                V_T = self.materials_params.get("V_T_base", 5.0)
                V_eff = torch.clamp(V_to - V_T, min=0.0)

                # z 方向衰减
                z_coord = x_phys[:, 2]
                h_ink_v = self.materials_params.get("ink_thickness", 3e-6)
                z_decay = torch.exp(-z_coord / h_ink_v)

                # EW 源项: S_ew = M_ac * delta_C * V_eff² * z_decay / eps_ac²
                # 量纲: [m³·s/kg] * [F/m²] * [V²] / [m²] = [m²/s] / [m] = [1/s] ✓
                ew_source = M_ac * delta_C * V_eff**2 * z_decay / (eps_ac**2 + 1e-20)

                # 负号：驱动力使 phi 减小（油墨被极性液体替代）
                ac_residual = ac_residual - ew_source
            except Exception:
                pass  # EW 源项失败不影响主残差

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
            "vof": torch.zeros(batch_size, device=device),
            "volume_conservation": torch.zeros(batch_size, device=device),
        }
