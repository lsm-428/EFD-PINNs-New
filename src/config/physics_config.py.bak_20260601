#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EFD3D 统一物理参数配置模块
==========================

本模块是项目中所有物理参数的唯一来源（Single Source of Truth）。

所有需要物理参数的模块都应该从这里导入，而不是自己定义：
    from src.config import PHYSICS, get_physics_config

参数分类：
    1. 几何参数：像素尺寸、油墨厚度、围堰高度等
    2. 流体属性：密度、粘度、界面张力等
    3. 电学参数：介电常数、介电层厚度等
    4. 动力学参数：响应时间常数、阻尼比等
    5. 边界条件：接触角、阈值电压等

使用方法：
    # 方式1：直接使用全局 PHYSICS 字典（简单场景）
    from src.config import PHYSICS
    theta0 = PHYSICS["theta0"]

    # 方式2：使用 PhysicsConfig 类（需要从 JSON 加载或自定义）
    from src.config import get_physics_config
    config = get_physics_config("config/device_calibrated_physics.json")
    theta0 = config.theta0

    # 方式3：获取 materials_params 格式（兼容 PhysicsConstraints）
    materials = config.to_materials_params()

作者: EFD-PINNs Team
日期: 2025-12-31
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# ============================================================================
# 默认配置路径 - 从 paths 模块导入
# ============================================================================
from .paths import DEFAULT_CONFIG_PATH

# ============================================================================
# 统一物理常量字典（全局单例）
# ============================================================================
# 这是项目中所有物理参数的权威来源
# 值来自 config/device_calibrated_physics.json 和实验校准

PHYSICS: Dict[str, Any] = {
    # ========== 几何参数 ==========
    "Lx": 174e-6,  # 像素宽度 (m)
    "Ly": 174e-6,  # 像素高度 (m)
    "Lz": 20e-6,  # 围堰/流体层高度 (m)
    "h_ink": 3e-6,  # 油墨层厚度 (m)
    "h_polar": 17e-6,  # 极性液体层厚度 (m)
    "wall_height": 3.5e-6,  # 围堰高度 (m)，实际器件
    # ========== 流体属性 ==========
    # 油墨（非极性，深色）
    "rho_oil": 763.0,  # 油墨密度 (kg/m³)，实测
    "mu_oil": 9.41e-4,  # 油墨动力粘度 (Pa·s)，实测
    "density_ink": 763.0,  # 别名，兼容 constraints.py
    "viscosity_ink": 9.41e-4,  # 别名，兼容 constraints.py
    # 极性液体（水+乙二醇 38:62）
    "rho_polar": 998.0,  # 极性液体密度 (kg/m³)
    "mu_polar": 1.01e-3,  # 极性液体动力粘度 (Pa·s)
    "density_polar": 998.0,  # 别名
    "viscosity_polar": 1.01e-3,  # 别名
    # 界面张力
    "sigma": 0.02505,  # 油墨-极性液体界面张力 (N/m)，实测
    "gamma": 0.048,  # 极性液体表面张力 (N/m)，水+EG 38:62实测
    "surface_tension_polar_ink": 0.02505,  # 别名
    # ========== 电学参数 ==========
    "epsilon_0": 8.854e-12,  # 真空介电常数 (F/m)
    "epsilon_r": 3.28,  # SU-8 相对介电常数（实测值）
    "epsilon_h": 1.934,  # Teflon AF 相对介电常数（实测值）
    "d_dielectric": 4e-7,  # 介电层厚度 (m) = 400nm (SU-8)
    "d_hydrophobic": 4e-7,  # 疏水层厚度 (m) = 400nm (Teflon)
    # ========== 接触角参数 ==========
    "theta0": 120.0,  # 本征接触角 (度)，无电压时
    "theta_wall": 71.0,  # 围堰壁接触角 (度), 原生SU-8, 极性液体润湿
    "theta_wall_teflon": 110.0,  # Teflon污染侧壁接触角 (度), 油墨润湿
    "wall_top_contact_angle": 71.0,  # 围堰顶面接触角 (度), 原生SU-8未污染
    "theta_min": 60.0,  # 最小接触角 (度)，物理下限
    "contact_angle_theta0": 120.0,  # 别名，兼容 constraints.py
    "contact_angle_ink": 120.0,  # 别名
    # ========== 动力学参数 ==========
    "tau": 0.005,  # 电润湿响应时间常数 (s) = 5ms
    "tau_onset": 0.0075,  # 低电压区 τ (s)
    "tau_saturation": 0.003,  # 高电压区 τ (s)
    "tau_recovery_factor": 0.4,  # 恢复因子
    "tau_recovery": 0.002,  # 恢复时间常数 = tau × factor = 5ms × 0.4 = 2ms
    "zeta": 0.8,  # 阻尼比（欠阻尼）
    "dynamic_order": 2,  # 动态阶数：2=二阶欠阻尼, 1=一阶指数
    "t_max": 0.05,  # 最大仿真时间 (s) = 50ms
    # ========== 电压参数 ==========
    "V_T_base": 5.0,  # 3μm油膜对应的阈值电压 (V)
    "V_T_sensitivity": 2e6,  # 阈值电压灵敏度 (V/m) = 2V/μm
    "V_max": 30.0,  # 最大工作电压 (V)
    "V_threshold": 5.0,  # 阈值电压 (V) = V_T_base + (h_ink-3μm)×sensitivity，计算值
    # ========== Allen-Cahn 相场参数 ==========
    "ac_interface_width": 5e-6,  # 界面宽度 (m)
    "ac_mobility": 1e-10,        # 迁移率 (m³·s/kg), mob≈0.1m/s, τ~0.05ms
    # ========== 开口率参数 ==========
    "eta_max": 0.85,  # 最大开口率
    "ink_initial_fraction": 0.15,  # 初始油墨体积分数
    # ========== 电润湿 EW 力参数 ==========
    "lambda_debye": 50e-9,  # 德拜屏蔽长度 [m] ~50nm，EW 力 z 方向衰减尺度
    # ========== 物理模型开关 ==========
    "use_convection": False,        # Re≈1-5, 默认关闭对流项
    "use_unified_wetting": False,   # 统一相场润湿 BC (替代旧版 BW/WW/SW)
    # ========== 开口率映射参数（用真实材料参数后需重新标定）==========
    "aperture_k": 3.0,  # 映射陡度（提高以补偿 Δθ 缩小）
    "aperture_theta_scale": 19.0,  # 角度缩放因子（降低使 tanh 更早饱和）
    "aperture_alpha": 0.03,  # 电容反馈强度（稍增）
}


@dataclass
class PhysicsConfig:
    """
    物理参数配置类

    提供类型安全的参数访问和 JSON 序列化支持。

    Attributes:
        所有物理参数作为类属性，带类型注解
    """

    # 几何参数
    Lx: float = 174e-6
    Ly: float = 174e-6
    Lz: float = 20e-6
    h_ink: float = 3e-6
    h_polar: float = 17e-6
    wall_height: float = 3.5e-6

    # 流体属性 - 油墨
    rho_oil: float = 763.0
    mu_oil: float = 9.41e-4

    # 流体属性 - 极性液体
    rho_polar: float = 998.0
    mu_polar: float = 1.01e-3

    # 界面张力
    sigma: float = 0.02505
    gamma: float = 0.048

    # 电学参数
    epsilon_0: float = 8.854e-12
    epsilon_r: float = 3.28
    epsilon_h: float = 1.934
    d_dielectric: float = 4e-7
    d_hydrophobic: float = 4e-7

    # 接触角
    theta0: float = 120.0
    theta_wall: float = 71.0
    theta_min: float = 60.0

    # 动力学
    tau: float = 0.005
    tau_onset: float = 0.0075
    tau_saturation: float = 0.003
    tau_recovery_factor: float = 0.4
    zeta: float = 0.8
    dynamic_order: int = 2
    t_max: float = 0.05

    # 电压 — V_threshold 是 property，从 V_T_base + 油膜厚度计算
    V_T_base: float = 5.0
    V_T_sensitivity: float = 2e6
    V_max: float = 30.0

    # 开口率
    eta_max: float = 0.85
    ink_initial_fraction: float = 0.15
    aperture_k: float = 3.0
    aperture_theta_scale: float = 19.0
    aperture_alpha: float = 0.03

    # 电润湿 EW 力参数
    lambda_debye: float = 50e-9  # 德拜屏蔽长度 [m] ~50nm

    # 物理模型开关
    use_convection: bool = False        # Re≈1-5, 默认关闭对流项
    use_unified_wetting: bool = False   # 统一相场润湿 BC (替代旧版 BW/WW/SW)

    # 侧壁 Teflon 污染接触角 (°)
    theta_wall_teflon: float = 110.0

    # 配置来源（用于追踪）
    _source: str = field(default="default", repr=False)

    @property
    def V_threshold(self) -> float:
        """阈值电压，基于油膜厚度动态计算"""
        return self.V_T_base + (self.h_ink - 3.0e-6) * self.V_T_sensitivity

    @property
    def tau_recovery(self) -> float:
        """恢复时间常数 = 驱动τ × 恢复因子（恢复快于驱动）"""
        return self.tau * self.tau_recovery_factor

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PhysicsConfig":
        """
        从 JSON 配置文件加载物理参数

        Args:
            path: JSON 文件路径

        Returns:
            PhysicsConfig 实例
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"配置文件不存在: {path}，使用默认参数")
            return cls(_source="default")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 从各个配置节提取参数
        materials = data.get("materials", {})
        geometry = data.get("geometry", {})
        data_cfg = data.get("data", {})
        dynamics = data_cfg.get("dynamics_params", {})
        aperture_mapping = data.get("aperture_mapping", {})

        return cls(
            # 几何参数
            Lx=geometry.get("Lx", cls.Lx),
            Ly=geometry.get("Ly", cls.Ly),
            Lz=geometry.get("Lz", cls.Lz),
            h_ink=geometry.get("ink_thickness", cls.h_ink),
            wall_height=geometry.get("wall_height", cls.wall_height),
            # 流体属性
            rho_oil=materials.get("rho_oil", cls.rho_oil),
            mu_oil=materials.get("mu_oil", cls.mu_oil),
            rho_polar=materials.get("rho_polar", cls.rho_polar),
            mu_polar=materials.get("mu_polar", cls.mu_polar),
            sigma=materials.get("sigma", cls.sigma),
            # 材料参数
            theta0=materials.get("theta0", cls.theta0),
            theta_wall=materials.get("theta_wall", cls.theta_wall),
            epsilon_r=materials.get("epsilon_r", cls.epsilon_r),
            epsilon_h=materials.get("epsilon_hydrophobic", cls.epsilon_h),
            gamma=materials.get("gamma", cls.gamma),
            d_dielectric=materials.get("dielectric_thickness", cls.d_dielectric),
            d_hydrophobic=materials.get("hydrophobic_thickness", cls.d_hydrophobic),
            # V_T 参数（V_threshold 是 property，从 V_T_base + h_ink 计算）
            V_T_base=materials.get("V_T_base", dynamics.get("V_T_base", cls.V_T_base)),
            V_T_sensitivity=materials.get(
                "V_T_sensitivity", dynamics.get("V_T_sensitivity", cls.V_T_sensitivity)
            ),
            # 动力学参数
            tau=dynamics.get("tau", cls.tau),
            tau_onset=dynamics.get("tau_onset", cls.tau_onset),
            tau_saturation=dynamics.get("tau_saturation", cls.tau_saturation),
            tau_recovery_factor=dynamics.get(
                "tau_recovery_factor", cls.tau_recovery_factor
            ),
            zeta=dynamics.get("zeta", cls.zeta),
            dynamic_order=dynamics.get("dynamic_order", cls.dynamic_order),
            # 开口率映射
            eta_max=aperture_mapping.get("aperture_max", cls.eta_max),
            aperture_k=aperture_mapping.get("k", cls.aperture_k),
            aperture_theta_scale=aperture_mapping.get(
                "theta_scale", cls.aperture_theta_scale
            ),
            aperture_alpha=aperture_mapping.get("alpha", cls.aperture_alpha),
            use_convection=materials.get("use_convection", cls.use_convection),
            _source=str(path),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容 PHYSICS 格式）"""
        d = asdict(self)
        d.pop("_source", None)

        # 添加别名以兼容旧代码
        d["density_ink"] = d["rho_oil"]
        d["density_polar"] = d["rho_polar"]
        d["viscosity_ink"] = d["mu_oil"]
        d["viscosity_polar"] = d["mu_polar"]
        d["surface_tension_polar_ink"] = d["sigma"]
        d["contact_angle_theta0"] = d["theta0"]
        d["contact_angle_ink"] = d["theta0"]

        return d

    def to_materials_params(self) -> Dict[str, Any]:
        """
        转换为 PhysicsConstraints 兼容的 materials_params 格式

        Returns:
            materials_params 字典
        """
        return {
            # 基础流体属性
            "viscosity": self.mu_polar,
            "density": self.rho_polar,
            "surface_tension": self.gamma,
            # 电学属性
            "epsilon_0": self.epsilon_0,
            "relative_permittivity": self.epsilon_r,
            "dielectric_thickness": self.d_dielectric,
            # 两相流属性
            "density_polar": self.rho_polar,
            "density_ink": self.rho_oil,
            "viscosity_polar": self.mu_polar,
            "viscosity_ink": self.mu_oil,
            "surface_tension_polar_ink": self.sigma,
            # 接触角
            "contact_angle_theta0": self.theta0,
            "contact_angle_ink": self.theta0,
            "theta_wall": self.theta_wall,
            # 几何参数
            "Lx": self.Lx,
            "Ly": self.Ly,
            "Lz": self.Lz,
            "ink_thickness": self.h_ink,
            "domain_height": self.Lz,
            "wall_height": self.wall_height,
            "ink_initial_fraction": self.ink_initial_fraction,
            # 电润湿 EW 力参数
            "lambda_debye": self.lambda_debye,
            # 物理模型开关
            "use_convection": self.use_convection,
            "use_unified_wetting": getattr(self, "use_unified_wetting", False),
            "theta_wall_teflon": getattr(self, "theta_wall_teflon", 110.0),
        }

    def to_predictor_params(self) -> Dict[str, Any]:
        """
        转换为 HybridPredictor 兼容的 params 格式

        Returns:
            params 字典
        """
        return {
            "theta0": self.theta0,
            "epsilon_0": self.epsilon_0,
            "gamma": self.gamma,
            "epsilon_r": self.epsilon_r,
            "d": self.d_dielectric,
            "epsilon_h": self.epsilon_h,
            "d_h": self.d_hydrophobic,
            "tau": self.tau,
            "tau_onset": self.tau_onset,
            "tau_saturation": self.tau_saturation,
            "tau_recovery_factor": self.tau_recovery_factor,
            "zeta": self.zeta,
            "dynamic_order": self.dynamic_order,
            "sigma": self.sigma,
            "V_max": self.V_max,
            "V_threshold": self.V_threshold,
            "V_T_base": self.V_T_base,
            "V_T_sensitivity": self.V_T_sensitivity,
        }

    def update_global_physics(self) -> None:
        """
        将当前配置更新到全局 PHYSICS 字典

        警告：这会修改全局状态，请谨慎使用
        """
        global PHYSICS
        PHYSICS.update(self.to_dict())
        logger.info(f"全局 PHYSICS 已更新，来源: {self._source}")


# ============================================================================
# 便捷函数
# ============================================================================

_config_cache: Dict[str, PhysicsConfig] = {}


def get_physics_config(
    path: Optional[Union[str, Path]] = None, use_cache: bool = True
) -> PhysicsConfig:
    """
    获取物理配置实例

    Args:
        path: 配置文件路径，None 则使用默认路径
        use_cache: 是否使用缓存（同一路径只加载一次）

    Returns:
        PhysicsConfig 实例
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH

    path_str = str(path)

    if use_cache and path_str in _config_cache:
        return _config_cache[path_str]

    config = PhysicsConfig.from_json(path)

    if use_cache:
        _config_cache[path_str] = config

    return config


def get_materials_params(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    获取 PhysicsConstraints 兼容的 materials_params

    这是一个便捷函数，用于替换 PhysicsConstraints 中的硬编码默认值。

    Args:
        path: 配置文件路径

    Returns:
        materials_params 字典
    """
    config = get_physics_config(path)
    return config.to_materials_params()


# ============================================================================
# 模块初始化：从默认配置更新 PHYSICS
# ============================================================================


def _init_physics_from_config():
    """尝试从默认配置文件初始化 PHYSICS"""
    try:
        if Path(DEFAULT_CONFIG_PATH).exists():
            config = get_physics_config(DEFAULT_CONFIG_PATH)
            # 只更新存在的键，保留 PHYSICS 中的额外键
            for key, value in config.to_dict().items():
                if (
                    key in PHYSICS
                    or key.startswith("density_")
                    or key.startswith("viscosity_")
                ):
                    PHYSICS[key] = value
            logger.debug(f"PHYSICS 已从 {DEFAULT_CONFIG_PATH} 初始化")
    except Exception as e:
        logger.warning(f"从配置文件初始化 PHYSICS 失败: {e}")


# 模块加载时自动初始化
_init_physics_from_config()
