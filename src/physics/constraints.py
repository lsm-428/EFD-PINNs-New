"""
EWPINN 物理约束模块
包含物理方程计算、材料参数和边界条件处理
"""

import logging

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
    # 所有物理参数必须从 PHYSICS 字典导入，禁止本地硬编码（R1 规则）
    # 如果 PHYSICS 不可用，抛出异常而不是使用错误的默认值
    from src.config import PHYSICS

    def _get(key, required=True):
        """从 PHYSICS 获取参数，缺失时抛出异常"""
        if key in PHYSICS:
            return PHYSICS[key]
        if required:
            msg = f"PHYSICS['{key}'] missing, cannot compute physics constraints"
            raise KeyError(msg)
        return None

    return {
        # 基础流体属性
        "viscosity": _get("mu_polar"),
        "density": _get("rho_polar"),
        "surface_tension": _get("gamma"),
        # 电学属性
        "epsilon_0": _get("epsilon_0"),
        "epsilon_su8": _get("epsilon_r"),
        "epsilon_teflon": _get("epsilon_h"),
        "d_su8": _get("d_dielectric"),
        "d_teflon": _get("d_hydrophobic"),
        "A_eff": _get("A_eff"),
        "dielectric_thickness": _get("d_dielectric"),
        "relative_permittivity": _get("epsilon_r"),
        # 两相流属性
        "density_polar": _get("rho_polar"),
        "density_ink": _get("rho_oil"),
        "viscosity_polar": _get("mu_polar"),
        "viscosity_ink": _get("mu_oil"),
        "surface_tension_polar_ink": _get("sigma"),
        # 接触角
        "contact_angle_theta0": _get("theta0"),
        "contact_angle_ink": _get("theta0"),
        "dynamic_contact_angle_advancing": _get("theta0"),
        "dynamic_contact_angle_receding": 100.0,  # PHYSICS 中无对应
        "theta_wall": _get("theta_wall"),
        "contact_line_friction": 1e-3,  # 经验值
        "pinning_energy": 1e-5,  # 经验值
        "slip_length": 1e-6,  # 经验值
        # 几何参数
        "ink_thickness": _get("h_ink"),
        "domain_height": _get("Lz"),
        "wall_height": _get("wall_height"),
        "wall_top_half_width": _get("wall_top_half_width"),
        "ink_initial_fraction": _get("ink_initial_fraction"),
        # 电润湿 EW 力参数
        "lambda_debye": _get("lambda_debye"),
        # 物理模型配置开关
        "use_convection": _get("use_convection"),
        "use_legacy_ac": False,
        "use_adaptive_loss_scale": False,
        # Allen-Cahn 相场参数
        "ac_interface_width": _get("ac_interface_width"),
        "ac_mobility": _get("ac_mobility"),
        "electrowetting_weight": _get("electrowetting_weight"),
        # 阈值电压
        "V_T_base": _get("V_T_base"),
        # 侧壁 Teflon 污染接触角
        "theta_wall_teflon": _get("theta_wall_teflon"),
    }


class PhysicsConstraints:
    """物理约束类 - 处理Navier-Stokes方程和材料属性"""

    def __init__(self, materials_params=None):
        # 从统一配置获取默认参数，允许外部覆盖
        default_params = _get_default_materials_params()
        if materials_params is not None:
            default_params.update(materials_params)
        self.materials_params = default_params

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

                # 回退路径：独立计算 phi 的 Hessian 各分量（用于曲率）
                # 注意：不能用 Laplacian 代替各二阶导数，曲率公式需要独立分量
                def _get_hessian_diag(first_grad):
                    """从一阶梯度计算 Hessian 对角线和混合分量"""
                    hess = {}
                    for i, key_i in enumerate("xyz"):
                        g2 = torch.autograd.grad(
                            outputs=first_grad[:, i].sum(),
                            inputs=x,
                            create_graph=True,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if g2 is not None:
                            for j, key_j in enumerate("xyz"):
                                hess[key_i + key_j] = g2[:, j]
                        else:
                            for _j, key_j in enumerate("xyz"):
                                hess[key_i + key_j] = torch.zeros_like(phi)
                    return hess

                g_phi = torch.autograd.grad(
                    outputs=phi.sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if g_phi is not None:
                    phi_hess = _get_hessian_diag(g_phi)
                    phi_xx = phi_hess["xx"]
                    phi_yy = phi_hess["yy"]
                    phi_zz = phi_hess["zz"]
                    phi_xy = phi_hess["xy"]
                    phi_xz = phi_hess["xz"]
                    phi_yz = phi_hess["yz"]
                else:
                    phi_xx = phi_yy = phi_zz = phi_xy = phi_xz = phi_yz = torch.zeros_like(phi)

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
            V_T = self.materials_params.get("V_T_base", 3.0)
            V_eff_ew = torch.clamp(V_to - V_T, min=0.0)

            # 电润湿压力幅值: p_ew = ½·C_ew·V_eff² / d_eff
            # C_ew [F/m²] * V² [V²] / d_eff [m] = [J/m²] / [m] = [N/m²] ✅
            d_eff = self.materials_params.get("dielectric_thickness", 800e-9)
            f_ew_magnitude = 0.5 * C_ew * V_eff_ew**2 / d_eff

            # 空间坐标
            x[:, 0]
            x[:, 1]
            z_coord = x[:, 2]

            # 电润湿力作用在底面 (z=0)
            # z 方向衰减尺度: 德拜屏蔽长度 ~50nm（不是油墨厚度 3μm）
            # 电润湿力是表面力，作用在 Z=0 疏水层表面纳米尺度
            lambda_d = self.materials_params.get("lambda_debye", 50e-9)
            z_decay = torch.exp(-z_coord / lambda_d)

            # 电润湿体积力: f_ew = -p_ew * z_decay * ∇φ / |∇φ|
            # 量纲: [N/m²] * [1/m] = [N/m³] ✅
            # 方向: 沿 -∇φ（从油指向水/中心），极性液体推动油墨向外收缩
            # z_decay: 在底面 ~50nm 内急剧衰减
            # 数值稳定性：|∇φ| 过小时关闭 EW 力（避免除零）
            grad_mag = torch.sqrt(phi_x**2 + phi_y**2 + 1e-10)
            # 当界面过度扩散时（|∇φ| < 1e-3），EW 力趋近于零
            ew_active = (grad_mag > 1e-3).float()
            f_ew_x = -f_ew_magnitude * z_decay * phi_x / grad_mag * ew_active
            f_ew_y = -f_ew_magnitude * z_decay * phi_y / grad_mag * ew_active
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

    def compute_interface_energy_residual(self, x_phys, predictions, grads=None):
        """界面能 — 纯 sigma*|grad(phi)|, 塑造圆润液滴

        注意：电润湿力已作为体积力加入 NS 方程（f_ew_x, f_ew_y）。
        AC 方程中不再有独立的 EW 源项。
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
            # [2026-06-11] NaN 修复：加 eps 保护，防止 sqrt(0) 在反向传播时产生 NaN 梯度
            # 根因：当 φ 在均匀区域 (φ=0 或 φ=1) 时，phi_x=phi_y=phi_z=0，
            # sqrt(0) 的梯度 = 0/0 = NaN，通过计算图传播到所有参数。
            if grads is not None:
                grad_mag = torch.sqrt(grads["phi_x"] ** 2 + grads["phi_y"] ** 2 + grads["phi_z"] ** 2 + 1e-10)
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

        # 3. 界面能残差
        try:
            ie_residuals = self.compute_interface_energy_residual(x_phys, predictions, grads=grads)
            residuals.update(ie_residuals)
        except Exception as e:
            logger.warning(f"界面能约束计算失败: {e}")

        # 4. Laplace 压力一致约束
        try:
            lp_residuals = self.compute_laplace_pressure_residual(x_phys, predictions, grads=grads)
            residuals.update(lp_residuals)
        except Exception as e:
            logger.warning(f"Laplace 压力残差计算失败: {e}")

        # 5. 时间连续性正则化（需要 model 做三时间点前向）
        try:
            temporal_residual = self._compute_temporal_smoothness(x_phys, predictions, model=model)
            residuals["temporal_smoothness"] = temporal_residual
        except Exception as e:
            logger.warning(f"时间连续性残差计算失败: {e}")
            residuals["temporal_smoothness"] = torch.zeros(batch_size, device=device)

        # 7. 压力钉扎
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

                # 标准 AC 方程残差: ∂φ/∂t + u·∇φ = mob·[ε·∇²φ − W'(φ)/ε]
                # 注意: W'(φ) = f_prime = 2φ(1-φ)(1-2φ)，量纲 [1]
                # eps_ac * lap_phi [1/m] - f_prime / eps_ac [1/m] → 量纲匹配
                ac_residual = advection - mob * (eps_ac * lap_phi - f_prime / eps_ac)

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
            # 注意: 电润湿力已作为体积力加入 NS 方程（f_ew_x, f_ew_y）
            # AC 方程中不再需要独立的 EW 源项
            # 相场演化由 NS 方程中的 EW 力自然驱动

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
        }
