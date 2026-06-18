#!/usr/bin/env python3
"""
EWP 两相流 PINN — 数据生成器模块
====================================

从 pinn_two_phase.py 中抽取的数据生成相关类，包含：
1. PhysicsBasedSampler — 基于物理机制的采样器（从 src/data/physics_sampling.py 迁入）
2. _sample_bc_z — 壁面 BC 的 z 坐标分层采样函数
3. DataGenerator — 训练数据生成器

本文件仅包含数据生成逻辑，不包含网络架构和训练逻辑。

作者: EFD-PINNs Team
日期: 2024-12
重构: 2026-06-18
"""

import logging
from typing import Any

import numpy as np
import torch

from src.config import PHYSICS, get_default_training_config

logger = logging.getLogger("EWP-PINN")

# 可选导入
try:
    from src.models.aperture_model import EnhancedApertureModel

    HAS_APERTURE = True
except ImportError:
    HAS_APERTURE = False

try:
    from src.predictors.hybrid_predictor import HybridPredictor

    HAS_HYBRID_PREDICTOR = True
except ImportError:
    HAS_HYBRID_PREDICTOR = False


# ============================================================================
# 默认配置（单一来源：config/device_calibrated_physics.json）
# ============================================================================

DEFAULT_CONFIG = get_default_training_config()


# ============================================================================
# 基于物理机制的采样器（从 src/data/physics_sampling.py 迁入）
# ============================================================================


class PhysicsBasedSampler:
    """
    基于物理机制的采样器

    实现四种电压分区采样策略：
    - 无响应区 (0-V_T): 采样权重可配置
    - 起始响应区 (V_T-2V_T): 采样权重可配置
    - 线性响应区 (2V_T-4V_T): 采样权重可配置
    - 饱和区 (>4V_T): 采样权重可配置
    """

    def __init__(self, config: dict, stage1_predictor=None, stage1_aperture_model=None):
        """
        初始化物理采样器

        Args:
            config: 配置字典，包含采样参数
            stage1_predictor: HybridPredictor 实例（用于精确开口率估算）
            stage1_aperture_model: EnhancedApertureModel 实例
        """
        self.config = config

        # Stage1 模型引用（用于精确物理计算）
        self._stage1_predictor = stage1_predictor
        self._stage1_aperture = stage1_aperture_model

        # 电压分区配置
        self.voltage_weights = config.get(
            "voltage_weights",
            {
                "no_response": 0.10,
                "onset": 0.50,
                "linear": 0.30,
                "saturation": 0.10,
            },
        )

        # 阈值电压参数 — 优先从 Stage1 读取，回退到 PHYSICS（单一来源）
        if stage1_predictor is not None:
            V_T_fallback = PHYSICS["V_T_base"]
            self.V_T_base = stage1_predictor.params.get(
                "V_T_base", stage1_predictor.params.get("V_threshold", V_T_fallback)
            )
        else:
            self.V_T_base = PHYSICS["V_T_base"]
        self.V_T_sensitivity = PHYSICS["V_T_sensitivity"]

        # 时间采样配置 — 从 PHYSICS 读取校准后的 tau 值
        if stage1_predictor is not None:
            default_tau_onset = stage1_predictor.params.get("tau_onset", PHYSICS["tau_onset"])
            default_tau_linear = stage1_predictor.params.get("tau", PHYSICS["tau"])
            default_tau_sat = stage1_predictor.params.get("tau_saturation", PHYSICS["tau_saturation"])
        else:
            default_tau_onset = PHYSICS["tau_onset"]
            default_tau_linear = PHYSICS["tau"]
            default_tau_sat = PHYSICS["tau_saturation"]

        self.time_config = config.get(
            "time_sampling",
            {
                "critical_points_density": 0.6,
                "adaptive_tau": True,
                "tau_onset": default_tau_onset,
                "tau_linear": default_tau_linear,
                "tau_saturation": default_tau_sat,
            },
        )

        # 物理阶段时间点
        self.physical_stages = {
            "electric_field": (0.0, 0.001),
            "marangoni": (0.001, 0.003),
            "film_instability": (0.003, 0.010),
            "local_rupture": (0.010, 0.020),
        }

        logger.info("物理采样器初始化完成")
        logger.info(f"电压分区权重: {self.voltage_weights}")
        logger.info(f"阈值电压: {self.V_T_base}V (基础值, 3μm油膜)")
        if self._stage1_predictor is not None:
            logger.info("使用 Stage1 模型进行精确开口率估算")

    def _calculate_threshold_voltage(self, oil_thickness: float) -> float:
        """基于油膜厚度计算阈值电压"""
        V_T = self.V_T_base + (oil_thickness - 3.0e-6) * self.V_T_sensitivity
        return max(0.1, V_T)

    def sample_voltage_physics_based(self, n_samples: int, oil_thickness: float = 3e-6) -> np.ndarray:
        """基于物理机制的电压采样"""
        V_T = self._calculate_threshold_voltage(oil_thickness)
        boundaries = {
            "no_response": (0.0, V_T),
            "onset": (V_T, 2 * V_T),
            "linear": (2 * V_T, 4 * V_T),
            "saturation": (4 * V_T, 30.0),
        }
        voltages = []
        for region, weight in self.voltage_weights.items():
            n_region = int(n_samples * weight)
            if n_region == 0:
                continue
            v_min, v_max = boundaries[region]
            if region == "no_response":
                v_samples = np.random.uniform(v_min, v_max, n_region)
            elif region == "onset":
                v_samples = self._sample_onset_region(v_min, v_max, n_region)
            elif region == "linear":
                v_samples = np.random.uniform(v_min, v_max, n_region)
            else:
                v_samples = self._sample_saturation_region(v_min, v_max, n_region)
            voltages.extend(v_samples)

        V_max = PHYSICS["V_max"]
        if len(voltages) < n_samples:
            remaining = n_samples - len(voltages)
            v_extra = np.random.uniform(0, V_max, remaining)
            voltages.extend(v_extra)

        voltages = np.clip(voltages, 0, V_max)
        return np.array(voltages[:n_samples])

    def _sample_onset_region(self, v_min: float, v_max: float, n_samples: int) -> np.ndarray:
        """在起始响应区采样（阈值附近加密）"""
        beta_samples = np.random.beta(0.3, 2.0, n_samples)
        return v_min + beta_samples * (v_max - v_min)

    def _sample_saturation_region(self, v_min: float, v_max: float, n_samples: int) -> np.ndarray:
        """在饱和区采样（边界附近加密）"""
        beta_samples = np.random.beta(2.0, 0.3, n_samples)
        return v_min + beta_samples * (v_max - v_min)

    def sample_time_adaptive(self, n_samples: int, voltage: float, voltage_prev: float = 0.0) -> np.ndarray:
        """自适应时间采样"""
        is_ramp_up = voltage > voltage_prev
        if voltage < self.V_T_base:
            tau = self.time_config["tau_onset"]
        elif voltage < 2 * self.V_T_base:
            tau = self.time_config["tau_linear"]
        else:
            tau = self.time_config["tau_saturation"]

        if not is_ramp_up:
            tau_recovery_factor = PHYSICS.get("tau_recovery_factor", 0.85)
            tau *= tau_recovery_factor

        critical_density = self.time_config["critical_points_density"]
        n_critical = int(n_samples * critical_density)
        n_continuous = n_samples - n_critical

        times = []
        if n_critical > 0:
            critical_times = self._sample_critical_physics_times(n_critical, voltage, voltage_prev)
            times.extend(critical_times)
        if n_continuous > 0:
            continuous_times = self._sample_continuous_times(n_continuous, tau)
            times.extend(continuous_times)

        return np.array(times[:n_samples])

    def _sample_critical_physics_times(self, n_samples: int, voltage: float, voltage_prev: float) -> np.ndarray:
        """在关键物理阶段时间点采样"""
        critical_times = []
        if voltage > voltage_prev:
            focus_stages = ["marangoni", "film_instability", "local_rupture"]
        else:
            focus_stages = ["local_rupture", "film_instability"]

        samples_per_stage = n_samples // len(focus_stages)
        for stage in focus_stages:
            t_min, t_max = self.physical_stages[stage]
            t_center = (t_min + t_max) / 2
            t_std = (t_max - t_min) / 6
            stage_times = np.random.normal(t_center, t_std, samples_per_stage)
            stage_times = np.clip(stage_times, t_min, t_max)
            critical_times.extend(stage_times)

        return np.array(critical_times)

    def _sample_continuous_times(self, n_samples: int, tau: float) -> np.ndarray:
        """连续时间采样（指数衰减分布）"""
        times = np.random.exponential(scale=tau, size=n_samples)
        t_max = 0.1  # 100ms 上限
        return np.clip(times, 0, t_max)

    def sample_spatial_physics_based(
        self, n_samples: int, voltage: float, time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """基于物理机制的空间采样"""
        eta = self._estimate_opening_rate(voltage, time)
        r_open = np.sqrt(eta * 174e-6 * 174e-6 / np.pi) if eta > 0.01 else 0.0
        theta = self._estimate_contact_angle(voltage, time)

        x_samples = np.random.uniform(0, 174e-6, n_samples)
        y_samples = np.random.uniform(0, 174e-6, n_samples)
        z_samples = self._sample_z_interface(n_samples, eta, voltage, time, r_open, theta)

        return x_samples, y_samples, z_samples

    def _estimate_opening_rate(self, voltage: float, time: float) -> float:
        """估算开口率"""
        if self._stage1_predictor is not None and self._stage1_aperture is not None:
            theta_ss = self._stage1_predictor.young_lippmann(voltage)
            if time > 1e-6:
                theta_start = self._stage1_predictor.young_lippmann(0.0)
                theta_t = self._stage1_predictor.dynamic_response(time, theta_start, theta_ss, V_to=voltage)
            else:
                theta_t = theta_ss
            return float(self._stage1_aperture.contact_angle_to_aperture_ratio(theta_t))

        V_T = self.V_T_base
        if voltage <= V_T:
            return 0.0

        theta = self._estimate_contact_angle(voltage, time)
        theta0 = PHYSICS["theta0"]
        eta_max_aperture = PHYSICS.get("eta_max_aperture", 0.85)
        k = PHYSICS.get("aperture_k", 2.02)
        theta_scale = PHYSICS.get("aperture_theta_scale", 27.46)

        dtheta = theta0 - theta
        if dtheta <= 0:
            return 0.0
        return float(np.clip(eta_max_aperture * np.tanh(k * dtheta / theta_scale), 0.0, eta_max_aperture))

    def _estimate_contact_angle(self, voltage: float, time: float) -> float:
        """估算接触角（度）"""
        if self._stage1_predictor is not None:
            theta_ss = self._stage1_predictor.young_lippmann(voltage)
            if time > 1e-6:
                theta_start = self._stage1_predictor.young_lippmann(0.0)
                return self._stage1_predictor.dynamic_response(time, theta_start, theta_ss, V_to=voltage)
            return theta_ss

        theta0 = PHYSICS["theta0"]
        V_threshold = PHYSICS["V_threshold"]
        V_eff = max(0.0, voltage - V_threshold)

        if V_eff <= 0.0:
            return theta0

        epsilon_0 = PHYSICS["epsilon_0"]
        epsilon_r = PHYSICS["epsilon_r"]
        d_dielectric = PHYSICS["d_dielectric"]
        sigma = PHYSICS["sigma"]
        C_total = epsilon_0 * epsilon_r / d_dielectric

        cos_theta0 = np.cos(np.radians(theta0))
        ew_term = C_total * V_eff**2 / (2.0 * sigma)
        cos_theta_eq = np.clip(cos_theta0 + ew_term, -1.0, 1.0)
        theta_eq = np.degrees(np.arccos(cos_theta_eq))

        tau = PHYSICS["tau"]
        zeta = PHYSICS["zeta"]
        omega_0 = 1.0 / tau
        if zeta >= 1.0:
            theta_t = theta_eq + (theta0 - theta_eq) * np.exp(-omega_0 * time)
        else:
            omega_d = omega_0 * np.sqrt(1.0 - zeta**2)
            exp_term = np.exp(-zeta * omega_0 * time)
            damping = zeta / np.sqrt(1.0 - zeta**2)
            theta_t = theta_eq + (theta0 - theta_eq) * exp_term * (
                np.cos(omega_d * time) + damping * np.sin(omega_d * time)
            )
        return theta_t

    def _sample_z_interface(
        self, n_samples: int, eta: float, voltage: float, time: float, r_open: float = 0.0, theta: float = 120.0
    ) -> np.ndarray:
        """在界面附近采样Z坐标（考虑倾斜界面）"""
        h_edge = 3e-6 / max(1.0 - eta, 0.15)
        interface_width = self._calculate_interface_width(voltage, time)

        n_interface = int(n_samples * 0.7)
        n_domain = n_samples - n_interface

        z_samples = []
        if n_interface > 0:
            z_interface = np.random.normal(h_edge, interface_width, n_interface)
            z_interface = np.clip(z_interface, 0, 20e-6)
            z_samples.extend(z_interface)
        if n_domain > 0:
            z_domain = np.random.uniform(0, 20e-6, n_domain)
            z_samples.extend(z_domain)

        return np.array(z_samples)

    def _calculate_interface_width(self, voltage: float, time: float) -> float:
        """计算界面宽度"""
        V_factor = min(1.0, voltage / 30.0)
        t_factor = min(1.0, time / 0.02)
        base_width = 0.5e-6
        max_width = 2e-6
        return base_width + (max_width - base_width) * V_factor * t_factor


# ============================================================================
# 壁面 BC z 采样辅助函数
# ============================================================================


def _sample_bc_z(rand: float, h_ink: float, Lz: float) -> float:
    """壁面 BC 的 z 坐标分层采样。

    物理：侧壁必须覆盖全壁高 [0, Lz]，不能只采油墨层附近。
    但油墨层附近需要更高采样密度。

    分层策略（rand ∈ [0,1) 均匀随机数）：
      - rand < 0.50: 油墨层及界面区 [0, h_ink*3]（最密）
      - rand < 0.80: 壁面上部 [h_ink*3, Lz]（稀疏但覆盖）
      - else:        过渡区 [h_ink*2, h_ink*4]（界面过渡加密）
    """
    if rand < 0.50:
        # 油墨层及界面区：占 50% 采样
        return rand / 0.50 * h_ink * 3
    if rand < 0.80:
        # 壁面上部：占 30% 采样
        z_lo = h_ink * 3
        return z_lo + (rand - 0.50) / 0.30 * (Lz - z_lo)
    # 过渡区：占 20% 采样
    z_lo = h_ink * 2
    return z_lo + (rand - 0.80) / 0.20 * h_ink * 2


# ============================================================================
# 训练数据生成器
# ============================================================================


class DataGenerator:
    """
    训练数据生成器 - 物理正确的边界条件方式

    核心思想：
    - 接触角 θ(t) 是边界条件，决定了油墨在基底上的润湿行为
    - PINN 自己学习 φ 场的演化，开口率是求解后的结果
    - 不预设"开口半径"，让物理方程自己决定界面位置

    边界条件：
    - 底面 (z=0): 接触角边界条件，∇φ·n = |∇φ|cos(θ)
    - 侧壁: 无滑移 + 固定接触角
    - 初始条件: 油墨均匀铺在底部

    generate_all_data() 重构说明（2026-06-18）：
    - _build_scenarios(): 消除稳态/升压/降压/跳变场景构建的重复代码
    - _sample_times(): 替代嵌套函数 sample_continuous_times()
    - _make_6d(): 消除约 20 处 6D 元组构造重复
    """

    def __init__(self, config: dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
        self.Lz = PHYSICS["Lz"]
        self.h_ink = PHYSICS["h_ink"]
        self.t_max = PHYSICS["t_max"]
        self.cx, self.cy = self.Lx / 2, self.Ly / 2
        self.r_max = np.sqrt(self.cx**2 + self.cy**2)

        # 初始化 Stage 1 接触角预测器（使用校准后的配置）
        self.contact_angle_predictor = None
        if HAS_HYBRID_PREDICTOR:
            try:
                from src.config import CONFIG_PATH

                self.contact_angle_predictor = HybridPredictor(
                    config_path=str(CONFIG_PATH), use_model_for_steady_state=False
                )
                logger.info("✅ 已集成 Stage 1 HybridPredictor 作为接触角边界条件")
            except Exception as e:
                logger.warning(f"HybridPredictor 初始化失败: {e}")

        self.theta0 = PHYSICS["theta0"]
        self.wall_height = PHYSICS["wall_height"]
        self.wall_top_z_tol = 1e-8  # z 方向浮点容差，与 TwoPhasePINN.forward 中的 _z_eps 一致
        self.wall_top_half_width = PHYSICS["wall_top_half_width"]
        self.use_stage1_eta = self.config.get("stage1_eta_from_model", False)

        # 采样策略：uniform（默认）或 physics_based
        sampling_cfg = self.config.get("sampling", {})
        self.sampling_strategy = sampling_cfg.get("strategy", "uniform")
        self.physics_sampler = None
        if self.sampling_strategy == "physics_based":
            try:
                # PhysicsBasedSampler 定义在本文件顶部（从 src.data.physics_sampling 迁入）
                self.physics_sampler = PhysicsBasedSampler(
                    sampling_cfg,
                    stage1_predictor=self.contact_angle_predictor,
                    stage1_aperture_model=None,
                )
                logger.info("✅ 使用物理采样策略 (PhysicsBasedSampler)")
            except ImportError as e:
                logger.warning(f"PhysicsBasedSampler 不可用，回退到均匀采样: {e}")
                self.sampling_strategy = "uniform"

    def _get_eta_from_stage1(self, V_prev: float | None, V: float, t: float) -> float | None:
        """从 Stage 1 模型获取开口率 eta(V_prev, V, t_since)。

        优先使用 EnhancedApertureModel.theta_eta_from_triad，
        回退到 get_opening_rate（解析公式）。
        """
        if V_prev is None:
            V_prev = V
        t_since = max(0.0, t)
        if HAS_APERTURE:
            try:
                if not hasattr(self, "_aperture_model"):
                    from src.config import CONFIG_PATH

                    self._aperture_model = EnhancedApertureModel(config_path=str(CONFIG_PATH))
                _, eta = self._aperture_model.theta_eta_from_triad(V_prev, V, t_since)
                return float(np.clip(eta, 0.0, 1.0))
            except Exception:
                pass
        return self.get_opening_rate(V, t)

    def compute_interface_phi(
        self,
        x: float,
        y: float,
        z: float,
        V_from: float,
        V_to: float,
        t_since: float,
    ) -> float:
        """定义 interface 数据点的 phi 硬约束值（6元组签名）。

        phi 是油水体积分数：0(纯水), 1(纯油), 0.5(油水界面零宽度)
        油水分离，没有混合态。

        硬约束区域：
        1. 壁顶面或以上（z >= wall_height）：水 → phi=0
        2. 接触线（d_wall ≈ 0, z = wall_height）：三相线 → phi=0.5
        3. 壁顶面以上（d_wall ≈ 0, z > wall_height）：水 → phi=0
        4. 夹角区（d_wall < wall_height, z < wall_height）：毛细钉扎 → phi=1
        5. z=0 非夹角区：由 Stage 1 eta 指导
           - 开口区内（r < r_open）：phi=0（水）
           - 油墨区（r >= r_open）：phi=1（油墨）
           - 界面（r ≈ r_open）：phi=0.5（接触线位置由 PINN 自己学习，这里不采样）

        Returns:
            phi ∈ {0, 0.5, 1}，NaN 表示该点不需要 interface 约束
        """
        wall_h = self.wall_height

        # 到最近围堰壁面的距离
        d_wall = min(x, self.Lx - x, y, self.Ly - y)

        # ===== 接触线：围堰立面与壁顶面顶角附近，phi=0.5 =====
        if d_wall < 1e-6 and abs(z - wall_h) < 1e-6:
            return 0.5

        # ===== z >= wall_height：壁顶面或以上，全是水 =====
        if z >= wall_h:
            return 0.0

        # ===== d_wall < wall_height 且 z < wall_height：夹角区 =====
        # Teflon 亲油，毛细钉扎，油墨被固定
        if d_wall < wall_h:
            return 1.0

        # ===== z=0 非夹角区（d_wall >= wall_height）：Stage 1 eta 指导 =====
        if z < 1e-8:
            eta = self._get_eta_from_stage1(V_from, V_to, t_since)
            if eta is None or eta < 0.01:
                return float("nan")

            # 开口半径：eta = pi * r_open^2 / (Lx * Ly)
            r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
            r_open = min(r_open, self.Lx / 2 * 0.98)

            cx, cy = self.Lx / 2, self.Ly / 2
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # 界面附近（r ≈ r_open）由 PINN 自己学习，跳过
            if r < r_open * 0.85:
                return 0.0  # 开口区：水
            if r > r_open * 1.15:
                return 1.0  # 油墨区：油墨
            return float("nan")  # 界面过渡区，PINN 自己学

        # ===== z ∈ (0, wall_height) 非夹角区：垂直分布由网络学习 =====
        return float("nan")

    def get_contact_angle(self, V: float, t: float) -> float:
        """
        获取动态接触角 - Stage 1 输出作为边界条件

        Args:
            V: 电压 (V)
            t: 时间 (s)

        Returns:
            接触角 θ(t) (度)
        """
        if self.contact_angle_predictor is not None:
            return self.contact_angle_predictor.predict(voltage=V, time=t, V_initial=0.0, t_step=0.0)
        return self._analytical_contact_angle(V, t)

    def _analytical_contact_angle(self, V: float, t: float) -> float:
        """
        内置解析公式（备用）— SU-8 + Teflon 双层串联电容

        Young-Lippmann：cos(θ) = cos(θ₀) + C_total × V_eff² / (2σ)
        1/C_total = 1/C_SU8 + 1/C_Teflon
        σ = 极性液体-油界面张力 (液-液体系)

        动态:
        - V > V_T: 二阶欠阻尼电润湿驱动
        - V ≤ V_T: 一阶指数表面张力恢复 (τ_recovery = τ × 0.4)
        """
        theta0 = PHYSICS["theta0"]
        epsilon_0 = PHYSICS["epsilon_0"]
        epsilon_r = PHYSICS["epsilon_r"]
        epsilon_h = PHYSICS["epsilon_h"]
        d_dielectric = PHYSICS["d_dielectric"]
        d_hydrophobic = PHYSICS["d_hydrophobic"]
        sigma_po = PHYSICS["sigma"]
        tau = PHYSICS["tau"]
        zeta = PHYSICS["zeta"]
        V_threshold = PHYSICS["V_threshold"]
        tau_recovery_factor = PHYSICS.get("tau_recovery_factor", 0.4)

        V_eff = max(0, V - V_threshold)

        # SU-8 + Teflon 双层串联
        C_su8 = epsilon_0 * epsilon_r / d_dielectric
        C_teflon = epsilon_0 * epsilon_h / d_hydrophobic
        C_total = 1.0 / (1.0 / C_su8 + 1.0 / C_teflon)

        cos_theta0 = np.cos(np.radians(theta0))
        ew_term = C_total * V_eff**2 / (2 * sigma_po)
        cos_theta_eq = np.clip(cos_theta0 + ew_term, -1, 1)
        theta_eq = np.degrees(np.arccos(cos_theta_eq))

        if V_eff > 0:
            omega_0 = 1.0 / tau
            if zeta >= 1.0:
                theta_t = theta_eq + (theta0 - theta_eq) * np.exp(-omega_0 * t)
            else:
                omega_d = omega_0 * np.sqrt(1 - zeta**2)
                exp_term = np.exp(-zeta * omega_0 * t)
                damping = zeta / np.sqrt(1 - zeta**2)
                theta_t = theta_eq + (theta0 - theta_eq) * exp_term * (
                    np.cos(omega_d * t) + damping * np.sin(omega_d * t)
                )
        else:
            tau_rec = tau * tau_recovery_factor
            theta_t = theta0 + (theta_eq - theta0) * np.exp(-t / tau_rec)

        return theta_t

    def compute_contact_angle_gradient(self, theta_deg: float) -> tuple[float, float]:
        """
        计算接触角对应的 φ 梯度方向

        接触角定义：液体内部与固体表面的夹角
        θ < 90°: 亲水（润湿）
        θ > 90°: 疏水（不润湿）

        在底面 z=0，法向量 n = (0, 0, 1)
        接触角边界条件：∂φ/∂z = |∇φ| * cos(θ)

        Returns:
            (cos_theta, sin_theta) 用于边界条件
        """
        theta_rad = np.radians(theta_deg)
        return np.cos(theta_rad), np.sin(theta_rad)

    def get_opening_rate(self, V: float, t: float) -> float:
        theta = self.get_contact_angle(V, t)
        # tanh 饱和映射: eta = eta_max * tanh(k * Δθ / theta_scale)
        theta0 = PHYSICS["theta0"]
        eta_max_aperture = PHYSICS.get("eta_max_aperture", 0.85)
        k = PHYSICS.get("aperture_k", 2.02)
        theta_scale = PHYSICS.get("aperture_theta_scale", 27.46)

        dtheta = theta0 - theta
        eta = 0.0 if dtheta <= 0 else eta_max_aperture * np.tanh(k * dtheta / theta_scale)

        return np.clip(eta, 0, eta_max_aperture)

    def target_phi_3d(
        self,
        x: float,
        y: float,
        z: float,
        V_from: float,
        V_to: float,
        t_since: float,
    ) -> float:
        """计算目标 φ 值（支持升压和降压）。

        物理模型：
        - 初始：底部 3μm 油墨 (φ=1)，上部 17μm 极性液体 (φ=0)
        - 升压（V_to > V_from）：电润湿驱动，油墨被推到边缘/角落
        - 降压（V_to < V_from）：表面张力恢复，油墨从边缘/角落铺展回中心

        升压模式：
        - 开口率 < 50%：中心开口模式（油墨环形分布）
        - 开口率 > 50%：四角液滴模式（油墨在角落）

        降压模式：
        - 油墨从当前位置向中心铺展
        - 开口率指数衰减：η(t) = η_at_step × exp(-t_since / τ_recovery)

        Args:
            x, y, z: 空间坐标 (m)
            V_from: 跳变前电压 (V)
            V_to: 跳变后电压 (V)
            t_since: 跳变后经过的时间 (s)

        Returns:
            φ ∈ [0, 1]
        """
        h_ink = self.h_ink

        # 判断是否降压
        is_voltage_down = V_to < V_from

        if is_voltage_down:
            # 降压模式：开口率指数衰减
            tau_recovery = PHYSICS.get("tau_recovery", 0.0075)
            eta_max = self.get_opening_rate(V_from, 0.020)
            eta = eta_max * np.exp(-t_since / tau_recovery)
        else:
            # 升压或稳态模式
            eta = self.get_opening_rate(V_to, t_since)

        # 使用统一的中心开口模式计算
        return self._phi_center_opening_mode(x, y, z, eta, h_ink)

    def _phi_center_opening_mode(self, x: float, y: float, z: float, eta: float, h_ink: float) -> float:
        """中心开口模式的 φ 分布（用于升压和降压）。

        物理模型：
        - 中心区域透明（φ=0），油墨在边缘形成环形分布
        - 开口半径 r_open = sqrt(η × Lx × Ly / π)
        - 油墨堆高满足体积守恒

        Args:
            x, y, z: 空间坐标 (m)
            eta: 开口率 ∈ [0, 1]
            h_ink: 初始油墨厚度 (m)

        Returns:
            φ ∈ [0, 1]
        """
        interface_width = PHYSICS["ac_interface_width"]

        if eta < 0.01:
            # 无开口：初始状态，油墨均匀铺在底部
            return 0.5 * (1 - np.tanh((z - h_ink) / (interface_width / 3)))

        if eta > 0.99:
            # 完全开口：几乎没有油墨
            return 0.0

        r = np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2)

        # 开口半径
        r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)

        # 油墨堆高（体积守恒）
        max_eta = 0.85
        eta = min(eta, max_eta)
        r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
        ink_area = self.Lx * self.Ly - np.pi * r_open**2
        h_ink_edge = self.Lx * self.Ly * h_ink / max(ink_area, 1e-12)

        # 径向分布
        radial_factor = 0.5 * (1 + np.tanh((r - r_open) / interface_width))

        if r < r_open - interface_width:
            phi_z = 0.0  # 中心透明
        elif r > r_open + interface_width:
            phi_z = 0.5 * (1 - np.tanh((z - h_ink_edge) / (interface_width / 2)))
        else:
            phi_center = 0.0
            phi_edge = 0.5 * (1 - np.tanh((z - h_ink_edge) / (interface_width / 2)))
            phi_z = phi_center * (1 - radial_factor) + phi_edge * radial_factor

        return np.clip(phi_z, 0, 1)

    def _sample_point_by_eta(self, eta: float) -> tuple:
        """根据开口率采样空间点，XY 方向界面加密"""
        eta_lo, eta_hi = 0.40, 0.50

        if eta > 0.01 and np.random.rand() < 0.4:
            if eta <= eta_lo:
                use_center = True
            elif eta >= eta_hi:
                use_center = False
            else:
                use_center = np.random.rand() < 0.5

            if use_center:
                r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
                n_angles = 8
                angle_idx = np.random.randint(n_angles)
                theta_angle = angle_idx * 2 * np.pi / n_angles + np.random.uniform(-0.2, 0.2)
                r = r_open + np.random.randn() * PHYSICS.get("sample_spread_small", 2e-6)
                r = max(0, min(r, self.r_max))
                x = self.cx + r * np.cos(theta_angle)
                y = self.cy + r * np.sin(theta_angle)
                x = np.clip(x, 0, self.Lx)
                y = np.clip(y, 0, self.Ly)
            else:
                corners = [(0, 0), (self.Lx, 0), (0, self.Ly), (self.Lx, self.Ly)]
                corner_idx = np.random.randint(4)
                cx, cy = corners[corner_idx]
                r_grid = np.linspace(0, PHYSICS.get("sample_spread_large", 15e-6), 5)
                theta_grid = np.linspace(0, np.pi / 2, 5)
                r = r_grid[np.random.randint(5)]
                theta_angle = theta_grid[np.random.randint(5)]
                sign_x = 1 if cx == 0 else -1
                sign_y = 1 if cy == 0 else -1
                x = cx + r * np.cos(theta_angle) * sign_x + np.random.randn() * 1e-6
                y = cy + r * np.sin(theta_angle) * sign_y + np.random.randn() * 1e-6
                x = np.clip(x, 0, self.Lx)
                y = np.clip(y, 0, self.Ly)
        else:
            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly

        h_edge = self.h_ink / max(1.0 - eta, PHYSICS["ink_initial_fraction"])

        if np.random.rand() < 0.7:
            interface_width = PHYSICS["ac_interface_width"]
            z = np.random.normal(h_edge, interface_width / 4)
            z = np.clip(z, 0, self.Lz)
        else:
            z = np.random.uniform(0, self.Lz)

        return x, y, z

    def _sample_times(self, n_samples: int) -> np.ndarray:
        """
        连续时间采样（偏重早期动态）

        使用 Beta(0.5, 1.0) 分布，自然地在早期 (t=0) 附近采样更多点，
        同时平滑覆盖整个时间域，避免人为的分段点造成的密度突变。

        Args:
            n_samples: 采样数量

        Returns:
            时间数组 (s)
        """
        return np.random.beta(0.5, 1.0, n_samples) * self.t_max

    def _make_6d(self, x, y, z, V_from, V_to, t) -> list:
        """
        构造 6D Triad 输入元组

        Args:
            x, y, z: 空间坐标 (m)
            V_from: 跳变前电压 (V)
            V_to: 当前电压 (V)
            t: 跳变后经过时间 (s)

        Returns:
            [x, y, z, V_from, V_to, t] 列表
        """
        return [x, y, z, V_from, V_to, t]

    def _build_scenarios(
        self, n_steady: int, n_up: int, n_down: int, n_jump: int = 0, jump_pairs: list | None = None
    ) -> tuple[list, list, list, list]:
        """
        构建稳态/升压/降压/跳变四类场景，消除重复代码。

        返回四类场景列表，每个元素为 (V_from, V_to, t) 元组。

        Args:
            n_steady: 稳态场景数 (V_from = V_to)
            n_up: 升压场景数 (0 -> V)
            n_down: 降压场景数 (V -> 0)
            n_jump: 跳变场景数 (V_a -> V_b)
            jump_pairs: 跳变电压对列表，默认 [(10,25),(25,10),(10,20),(20,10)]

        Returns:
            (steady_scenarios, up_scenarios, down_scenarios, jump_scenarios)
            每个为 (V_from, V_to, t) 元组列表
        """
        if jump_pairs is None:
            jump_pairs = [(10, 25), (25, 10), (10, 20), (20, 10)]

        # 稳态: V_from = V_to
        V_s = np.random.uniform(0, 30.0, n_steady)
        t_s = self._sample_times(n_steady)
        steady_scenarios = [(V_s[i], V_s[i], t_s[i]) for i in range(n_steady)]

        # 升压: 0 -> V
        V_u = np.random.uniform(1.0, 30.0, n_up)
        t_u = self._sample_times(n_up)
        up_scenarios = [(0.0, V_u[i], t_u[i]) for i in range(n_up)]

        # 降压: V -> 0
        V_d = np.random.uniform(1.0, 30.0, n_down)
        t_d = self._sample_times(n_down)
        down_scenarios = [(V_d[i], 0.0, t_d[i]) for i in range(n_down)]

        # 跳变: V_a -> V_b
        t_j = self._sample_times(n_jump)
        jump_scenarios = []
        for i in range(n_jump):
            Vf, Vt = jump_pairs[i % len(jump_pairs)]
            jump_scenarios.append((float(Vf), float(Vt), t_j[i]))

        return steady_scenarios, up_scenarios, down_scenarios, jump_scenarios

    def generate_all_data(self) -> dict[str, torch.Tensor]:
        """
        生成训练数据 - 支持升压和降压

        数据格式: (x, y, z, V_from, V_to, t_since) - 6D Triad
        φ = 1: 油墨
        φ = 0: 极性液体（透明）

        重构说明（2026-06-18）：
        - 使用 _build_scenarios() 消除 4 处场景构建重复代码
        - 使用 _sample_times() 替代嵌套函数 sample_continuous_times()
        - 使用 _make_6d() 消除约 20 处 6D 元组构造重复
        - 所有物理计算完全不变
        """
        data_cfg = self.config.get("data", DEFAULT_CONFIG["data"])

        logger.info("生成训练数据（连续采样模式）...")
        logger.info("  数据格式: (x, y, z, V_from, V_to, t_since)")

        # ============================================================
        # 1. 界面数据（核心训练数据）
        # ============================================================
        n_interface = data_cfg.get("n_interface", 100000)
        interface_points = []
        interface_targets = []

        use_physics_sampler = self.sampling_strategy == "physics_based" and self.physics_sampler is not None

        if use_physics_sampler:
            sampler = self.physics_sampler
            oil_thickness = self.h_ink
            if sampler._stage1_aperture is None and hasattr(self, "_aperture_model"):
                sampler._stage1_aperture = self._aperture_model
            if sampler._stage1_aperture is None:
                try:
                    from src.config import CONFIG_PATH as _CONFIG_PATH
                    from src.models.aperture_model import EnhancedApertureModel as _EAM

                    sampler._stage1_aperture = _EAM(config_path=str(_CONFIG_PATH))
                except Exception:
                    pass

            # 使用物理采样器的场景构建
            n_steady = int(n_interface * 0.4)
            n_up = int(n_interface * 0.3)
            n_down = n_interface - n_steady - n_up  # 剩余给降压
            n_jump = int(n_interface * 0.1)

            # 1.1 稳态 (40%)
            V_steady_arr = sampler.sample_voltage_physics_based(n_steady, oil_thickness=oil_thickness)
            for V in V_steady_arr:
                t = sampler.sample_time_adaptive(1, V, V)[0]
                x, y, z = sampler.sample_spatial_physics_based(1, V, t)
                phi = self.target_phi_3d(x[0], y[0], z[0], float(V), float(V), t)
                if not np.isnan(phi):
                    interface_points.append(self._make_6d(x[0], y[0], z[0], float(V), float(V), t))
                    interface_targets.append(phi)

            # 1.2 升压 (30%)
            V_up_arr = sampler.sample_voltage_physics_based(n_up, oil_thickness=oil_thickness)
            for V in V_up_arr:
                if float(V) < 1.0:
                    continue
                t = sampler.sample_time_adaptive(1, float(V), 0.0)[0]
                x, y, z = sampler.sample_spatial_physics_based(1, float(V), t)
                phi = self.target_phi_3d(x[0], y[0], z[0], 0.0, float(V), t)
                if not np.isnan(phi):
                    interface_points.append(self._make_6d(x[0], y[0], z[0], 0.0, float(V), t))
                    interface_targets.append(phi)

            # 1.3 降压 (剩余)
            n_down_actual = n_interface - len(interface_points)
            if n_down_actual > 0:
                V_from_arr = sampler.sample_voltage_physics_based(n_down_actual, oil_thickness=oil_thickness)
                for Vf in V_from_arr:
                    if float(Vf) < 1.0:
                        continue
                    t = sampler.sample_time_adaptive(1, 0.0, float(Vf))[0]
                    x, y, z = sampler.sample_spatial_physics_based(1, 0.0, t)
                    phi = self.target_phi_3d(x[0], y[0], z[0], float(Vf), 0.0, t)
                    if not np.isnan(phi):
                        interface_points.append(self._make_6d(x[0], y[0], z[0], float(Vf), 0.0, t))
                        interface_targets.append(phi)

            # 1.4 中间电压跳变 (10%)
            jump_pairs = [(10, 25), (25, 10), (10, 20), (20, 10)]
            for _ in range(n_jump):
                V_from, V_to = jump_pairs[np.random.randint(len(jump_pairs))]
                t = sampler.sample_time_adaptive(1, float(V_to), float(V_from))[0]
                x, y, z = sampler.sample_spatial_physics_based(1, float(V_to), t)
                phi = self.target_phi_3d(x[0], y[0], z[0], float(V_from), float(V_to), t)
                if not np.isnan(phi):
                    interface_points.append(self._make_6d(x[0], y[0], z[0], float(V_from), float(V_to), t))
                    interface_targets.append(phi)
        else:
            # 原有均匀采样逻辑（使用 _build_scenarios 和 _make_6d 重构）
            n_steady = int(n_interface * 0.4)
            n_up = int(n_interface * 0.3)
            n_down = n_interface - n_steady - n_up
            n_jump = int(n_interface * 0.1)

            steady_sc, up_sc, down_sc, jump_sc = self._build_scenarios(n_steady, n_up, n_down, n_jump)

            # 1.1 稳态数据 (40%) - V_from = V_to
            for V_from, V_to, t in steady_sc:
                eta = self.get_opening_rate(V_to, t)
                x, y, z = self._sample_point_by_eta(eta)
                phi = self.target_phi_3d(x, y, z, V_from, V_to, t)
                if not np.isnan(phi):
                    interface_points.append(self._make_6d(x, y, z, V_from, V_to, t))
                    interface_targets.append(phi)

            # 1.2 升压响应 (30%) - 0 -> V
            for V_from, V_to, t in up_sc:
                eta = self.get_opening_rate(V_to, t)
                x, y, z = self._sample_point_by_eta(eta)
                phi = self.target_phi_3d(x, y, z, V_from, V_to, t)
                if not np.isnan(phi):
                    interface_points.append(self._make_6d(x, y, z, V_from, V_to, t))
                    interface_targets.append(phi)

            # 1.3 降压响应 (30%) - V -> 0
            tau_recovery = PHYSICS["tau_recovery"]
            for V_from, V_to, t in down_sc:
                eta_at_fall = self.get_opening_rate(V_from, 0.020)
                eta = eta_at_fall * np.exp(-t / tau_recovery)
                x, y, z = self._sample_point_by_eta(eta)
                phi = self.target_phi_3d(x, y, z, V_from, V_to, t)
                if not np.isnan(phi):
                    interface_points.append(self._make_6d(x, y, z, V_from, V_to, t))
                    interface_targets.append(phi)

            # 1.4 中间电压跳变 (10%)
            for V_from, V_to, t in jump_sc:
                eta = self.get_opening_rate(V_to, t)
                x, y, z = self._sample_point_by_eta(eta)
                phi = self.target_phi_3d(x, y, z, V_from, V_to, t)
                if not np.isnan(phi):
                    interface_points.append(self._make_6d(x, y, z, V_from, V_to, t))
                    interface_targets.append(phi)

        logger.info(f"  界面数据点: {len(interface_points)}")

        # ============================================================
        # 1b. 底面数据点 (z=0)
        # ============================================================
        n_bottom = data_cfg.get("n_bottom", 10000)
        bottom_added = 0
        for _ in range(n_bottom):
            x = np.random.uniform(0, self.Lx)
            y = np.random.uniform(0, self.Ly)
            z = 0.0
            V_from = np.random.uniform(0, 30.0)
            V_to = np.random.uniform(0, 30.0)
            if np.random.random() < 0.8:
                V_to = V_from
                t = self._sample_times(1)[0]
            else:
                t = self._sample_times(1)[0]
            phi = self.target_phi_3d(x, y, z, V_from, V_to, t)
            if not np.isnan(phi):
                interface_points.append(self._make_6d(x, y, z, V_from, V_to, t))
                interface_targets.append(phi)
                bottom_added += 1

        logger.info(f"  底面数据点 (z=0): +{bottom_added}")

        # ============================================================
        # 2. 初始条件：t=0 时油墨均匀铺在底部 3μm
        # ============================================================
        n_ic = data_cfg.get("n_initial", 10000)
        ic_points, ic_values = [], []

        V_ic = np.random.uniform(0, 30.0, n_ic)

        for V in V_ic:
            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly
            z = np.random.rand() * self.wall_height

            interface_width = PHYSICS["ic_width"]
            d_wall = min(x, self.Lx - x, y, self.Ly - y)

            if abs(z - self.wall_height) < self.wall_top_z_tol and d_wall < self.wall_top_z_tol:
                phi = 0.5
            elif z < self.h_ink:
                phi = 1.0
            else:
                phi = 0.5 * (1 + np.tanh((self.h_ink - z) / interface_width))

            phi = np.clip(phi, 0, 1)
            ic_points.append(self._make_6d(x, y, z, V, V, 0.0))
            ic_values.append([0.0, 0.0, 0.0, 0.0, phi])

        logger.info(f"  初始条件点: {len(ic_points)}")

        # ============================================================
        # 3. 壁面边界条件 — 使用 _build_scenarios 构建混合场景
        # ============================================================
        n_bc = data_cfg.get("n_boundary", 10000)
        bc_points, bc_values = [], []

        n_bc_steady = int(n_bc * 0.4)
        n_bc_up = int(n_bc * 0.3)
        n_bc_down = n_bc - n_bc_steady - n_bc_up

        bc_steady_sc, bc_up_sc, bc_down_sc, _ = self._build_scenarios(n_bc_steady, n_bc_up, n_bc_down)
        scenarios = bc_steady_sc + bc_up_sc + bc_down_sc

        for V_from, V_to, t in scenarios:
            boundary_type = np.random.randint(0, 4)

            if boundary_type == 0:
                x, y = 0, np.random.rand() * self.Ly
            elif boundary_type == 1:
                x, y = self.Lx, np.random.rand() * self.Ly
            elif boundary_type == 2:
                x, y = np.random.rand() * self.Lx, 0
            else:
                x, y = np.random.rand() * self.Lx, self.Ly

            z = _sample_bc_z(np.random.rand(), self.h_ink, self.Lz)

            d_wall = min(x, self.Lx - x, y, self.Ly - y)
            on_wall_top_z = abs(z - self.wall_height) < self.wall_top_z_tol
            in_corner_x = (x < self.wall_height) or (x > self.Lx - self.wall_height)
            in_corner_y = (y < self.wall_height) or (y > self.Ly - self.wall_height)
            in_corner = (in_corner_x or in_corner_y) and z < self.wall_height

            if on_wall_top_z and d_wall < self.wall_top_z_tol:
                phi = 0.5
            elif z >= self.wall_height - self.wall_top_z_tol:
                phi = 0.0
            elif in_corner:
                phi = 1.0
            else:
                phi = self.target_phi_3d(x, y, z, V_from, V_to, t)
                if np.isnan(phi):
                    phi = 1.0 if z < self.h_ink else 0.0

            bc_points.append(self._make_6d(x, y, z, V_from, V_to, t))
            bc_values.append([0.0, 0.0, 0.0, 0.0, phi])

        logger.info(f"  壁面边界条件点: {len(bc_points)}")

        # ============================================================
        # 3b. 壁顶接触线过采样
        # ============================================================
        n_wall_top = n_bc // 4
        wt_added = 0
        for _ in range(n_wall_top):
            V_from = np.random.uniform(0, 30.0)
            V_to = np.random.uniform(0, 30.0)
            t = self._sample_times(1)[0]

            boundary_type = np.random.randint(0, 4)
            if boundary_type == 0:
                x = 0
                y = np.random.uniform(self.wall_height, self.Ly - self.wall_height)
            elif boundary_type == 1:
                x = self.Lx
                y = np.random.uniform(self.wall_height, self.Ly - self.wall_height)
            elif boundary_type == 2:
                x = np.random.uniform(self.wall_height, self.Lx - self.wall_height)
                y = 0
            else:
                x = np.random.uniform(self.wall_height, self.Lx - self.wall_height)
                y = self.Ly

            z = self.wall_height + np.random.uniform(-self.wall_top_z_tol, self.wall_top_z_tol)

            bc_points.append(self._make_6d(x, y, z, V_from, V_to, t))
            bc_values.append([0.0, 0.0, 0.0, 0.0, 0.5])
            wt_added += 1

        logger.info(f"  壁顶接触线过采样: +{wt_added} 点 (z~{self.wall_height*1e6:.1f}μm, φ=0.5)")

        # ============================================================
        # 3c. 侧壁夹角区域过采样
        # ============================================================
        n_corner = n_bc // 4
        corner_added = 0
        for _ in range(n_corner):
            V_from = np.random.uniform(0, 30.0)
            V_to = np.random.uniform(0, 30.0)
            t = self._sample_times(1)[0]

            corner = np.random.randint(0, 4)
            if corner == 0:
                x = np.random.uniform(0, self.wall_height)
                y = np.random.uniform(0, self.wall_height)
            elif corner == 1:
                x = np.random.uniform(self.Lx - self.wall_height, self.Lx)
                y = np.random.uniform(0, self.wall_height)
            elif corner == 2:
                x = np.random.uniform(0, self.wall_height)
                y = np.random.uniform(self.Ly - self.wall_height, self.Ly)
            else:
                x = np.random.uniform(self.Lx - self.wall_height, self.Lx)
                y = np.random.uniform(self.Ly - self.wall_height, self.Ly)

            z = 0.0

            bc_points.append(self._make_6d(x, y, z, V_from, V_to, t))
            bc_values.append([0.0, 0.0, 0.0, 0.0, 1.0])
            corner_added += 1

        logger.info(f"  侧壁夹角过采样: +{corner_added} 点 (z=0, 角落, φ=1)")

        # ============================================================
        # 4. 域内配点 — 使用 _build_scenarios 构建混合场景
        # ============================================================
        n_domain = data_cfg.get("n_domain", 20000)
        domain_points = []

        n_dom_steady = int(n_domain * 0.35)
        n_dom_up = int(n_domain * 0.25)
        n_dom_down = int(n_domain * 0.25)
        n_dom_jump = n_domain - n_dom_steady - n_dom_up - n_dom_down

        dom_steady_sc, dom_up_sc, dom_down_sc, dom_jump_sc = self._build_scenarios(
            n_dom_steady,
            n_dom_up,
            n_dom_down,
            n_dom_jump,
            jump_pairs=[(10, 25), (25, 10), (10, 20), (20, 10), (5, 15), (15, 25)],
        )
        dom_scenarios = dom_steady_sc + dom_up_sc + dom_down_sc + dom_jump_sc

        use_vertical_sampling = data_cfg.get("use_vertical_sampling", True)
        n_vertical_samples = data_cfg.get("n_vertical_samples", 50)

        for V_from, V_to, t in dom_scenarios:
            x = np.random.uniform(0, self.Lx)
            y = np.random.uniform(0, self.Ly)

            if use_vertical_sampling and n_vertical_samples > 1:
                eta = self.get_opening_rate(V_to, t)
                h_ink_edge = self.h_ink / max(1.0 - eta, PHYSICS["ink_initial_fraction"])
                interface_width = PHYSICS["ac_interface_width"]

                n_interface = int(n_vertical_samples * 0.8)
                n_bottom_v = int(n_vertical_samples * 0.1)
                n_top_v = n_vertical_samples - n_interface - n_bottom_v

                z_points = []

                z_bottom_max = max(0, h_ink_edge - 3 * interface_width)
                if n_bottom_v > 0 and z_bottom_max > 0:
                    z_bottom = np.linspace(0, z_bottom_max, n_bottom_v)
                    z_points.extend(z_bottom)

                z_interface_min = max(0, h_ink_edge - 3 * interface_width)
                z_interface_max = min(self.Lz, h_ink_edge + 3 * interface_width)
                if n_interface > 0:
                    z_interface = np.linspace(z_interface_min, z_interface_max, n_interface)
                    z_points.extend(z_interface)

                z_top_min = min(self.Lz, h_ink_edge + 3 * interface_width)
                if n_top_v > 0 and z_top_min < self.Lz:
                    z_top = np.linspace(z_top_min, self.Lz, n_top_v)
                    z_points.extend(z_top)

                z_all = np.unique(z_points)

                for z in z_all:
                    domain_points.append(self._make_6d(x, y, z, V_from, V_to, t))
            else:
                z = np.random.uniform(0, self.Lz)
                domain_points.append(self._make_6d(x, y, z, V_from, V_to, t))

        logger.info(f"  域内配点: {len(domain_points)}")

        # ============================================================
        # 5. 接触角边界条件 — 使用 _build_scenarios
        # ============================================================
        n_contact = data_cfg.get("n_interface", 100000) // 2
        contact_points = []
        contact_theta = []

        n_con_steady = int(n_contact * 0.4)
        n_con_up = int(n_contact * 0.3)
        n_con_down = n_contact - n_con_steady - n_con_up

        con_steady_sc, con_up_sc, con_down_sc, _ = self._build_scenarios(n_con_steady, n_con_up, n_con_down)
        con_scenarios = con_steady_sc + con_up_sc + con_down_sc

        for V_from, V_to, t in con_scenarios:
            # 接触角计算逻辑（对齐 EFD-PINNs 参考实现）
            if V_from > V_to:  # 降压过程 (Step Down)
                # 模拟接触角弛豫：从 theta(V_from) 恢复到 theta0
                theta_high = self.get_contact_angle(V_from, 1.0)
                theta_low = self.theta0
                tau_recovery = PHYSICS.get("tau_recovery", 0.0075)
                decay = np.exp(-t / tau_recovery)
                theta = theta_low + (theta_high - theta_low) * decay
            else:  # 升压或稳态
                theta = self.get_contact_angle(V_to, t)
            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly
            contact_points.append(self._make_6d(x, y, 0.0, V_from, V_to, t))
            contact_theta.append(theta)

        logger.info(f"  接触角采样点: {len(contact_points)} (仅用于采样位置引导，不接入损失)")

        # ============================================================
        # 5b. 底面接触线精细采样 (z=0, r~r_open 高斯扩展)
        # ============================================================
        n_cl = data_cfg.get("n_interface", 100000) // 4
        cl_added = 0
        for _ in range(n_cl):
            V = np.random.uniform(0, 30.0)
            V_prev = 0.0 if np.random.rand() > 0.5 else V
            t = self._sample_times(1)[0]

            eta = self.get_opening_rate(V, t)
            r_open = np.sqrt(max(0, eta) * self.Lx * self.Ly / np.pi)
            r_open = min(r_open, self.Lx / 2 * 0.95)

            angle = np.random.uniform(0, 2 * np.pi)
            r = r_open + np.random.normal(0, PHYSICS["contact_line_sigma"])
            r = np.clip(r, 1e-6, self.Lx / 2 * 0.98)
            x_cl = self.cx + r * np.cos(angle)
            y_cl = self.cy + r * np.sin(angle)
            x_cl = np.clip(x_cl, 0, self.Lx)
            y_cl = np.clip(y_cl, 0, self.Ly)
            z_cl = 0.0

            phi_cl = self.target_phi_3d(x_cl, y_cl, z_cl, V_prev, V, t)
            if not np.isnan(phi_cl):
                interface_points.append(self._make_6d(x_cl, y_cl, z_cl, V_prev, V, t))
                interface_targets.append(phi_cl)
                cl_added += 1

        logger.info(f"  底面接触线精细采样: +{cl_added} 点 (z=0, r~r_open)")

        # ============================================================
        # 5c. 突破时刻底面过采样 (t∈[0,5ms], 升压场景)
        # ============================================================
        n_breakthrough = data_cfg.get("n_interface", 100000) // 8
        bt_added = 0
        for _ in range(n_breakthrough):
            V = np.random.uniform(5.0, 30.0)
            t = np.random.exponential(scale=PHYSICS["breakthrough_tau"])
            t = np.clip(t, 0, PHYSICS["breakthrough_t_max"])
            eta = self.get_opening_rate(V, 0.02)
            r_open = np.sqrt(max(0.01, eta) * self.Lx * self.Ly / np.pi)
            r_open = min(r_open, self.Lx / 2 * 0.95)
            angle = np.random.uniform(0, 2 * np.pi)
            r = r_open * (0.5 + 0.5 * np.random.random())
            x_bt = np.clip(self.cx + r * np.cos(angle), 0, self.Lx)
            y_bt = np.clip(self.cy + r * np.sin(angle), 0, self.Ly)
            z_bt = 0.0

            phi_bt = self.target_phi_3d(x_bt, y_bt, z_bt, 0.0, V, t)
            if not np.isnan(phi_bt):
                interface_points.append(self._make_6d(x_bt, y_bt, z_bt, 0.0, V, t))
                interface_targets.append(phi_bt)
                bt_added += 1

        logger.info(f"  突破时刻底面过采样: +{bt_added} 点 (t~[0,5ms], 升压)")

        return {
            "interface_points": torch.tensor(np.array(interface_points), dtype=torch.float32, device=self.device),
            "interface_targets": torch.tensor(np.array(interface_targets), dtype=torch.float32, device=self.device),
            "contact_points": torch.tensor(np.array(contact_points), dtype=torch.float32, device=self.device),
            "contact_theta": torch.tensor(np.array(contact_theta), dtype=torch.float32, device=self.device),
            "ic_points": torch.tensor(np.array(ic_points), dtype=torch.float32, device=self.device),
            "ic_values": torch.tensor(np.array(ic_values), dtype=torch.float32, device=self.device),
            "bc_points": torch.tensor(np.array(bc_points), dtype=torch.float32, device=self.device),
            "bc_values": torch.tensor(np.array(bc_values), dtype=torch.float32, device=self.device),
            "domain_points": torch.tensor(np.array(domain_points), dtype=torch.float32, device=self.device),
        }


def create_physics_sampling_dataset(config: dict, n_total: int = 100000) -> dict:
    """
    创建完整的物理优化数据集

    Args:
        config: 配置字典
        n_total: 总样本数

    Returns:
        包含所有数据的字典
    """
    sampler = PhysicsBasedSampler(config)

    # 数据分配
    n_interface = int(n_total * 0.6)  # 60% 界面数据
    n_initial = int(n_total * 0.2)  # 20% 初始条件
    n_boundary = n_total - n_interface - n_initial  # 剩余为边界条件

    logger.info(f"数据集分配: 界面={n_interface}, 初始={n_initial}, 边界={n_boundary}")

    # 生成数据
    dataset = {
        "interface_points": [],
        "interface_targets": [],
        "initial_points": [],
        "initial_targets": [],
        "boundary_points": [],
        "boundary_targets": [],
    }

    # 1. 界面数据
    logger.info("生成界面数据...")
    for _ in range(n_interface):
        voltage = sampler.sample_voltage_physics_based(1)[0]
        voltage_prev = 0.0 if np.random.rand() > 0.5 else voltage
        time = sampler.sample_time_adaptive(1, voltage, voltage_prev)[0]
        x, y, z = sampler.sample_spatial_physics_based(1, voltage, time)
        # 物理参数从 PHYSICS 读取（单一来源）
        from src.config import PHYSICS

        h_ink = PHYSICS["h_ink"]
        interface_width = PHYSICS.get("interface_width", 1e-6)
        phi = 0.5 * (1 - np.tanh((z[0] - h_ink) / interface_width))
        dataset["interface_points"].append([x[0], y[0], z[0], voltage_prev, voltage, time])
        dataset["interface_targets"].append(phi)

    # 转换为numpy数组
    for key, values in dataset.items():
        dataset[key] = np.array(values)

    total_samples = len(dataset["interface_points"]) + len(dataset["initial_points"]) + len(dataset["boundary_points"])
    logger.info(f"数据集生成完成，总样本数: {total_samples}")

    return dataset


if __name__ == "__main__":
    """独立运行数据生成。

    使用方法:
        python -m src.models.pinn_data_generator [--config CONFIG_PATH] [--output OUTPUT_DIR] [--n-samples N]
    """
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="PINN 两相流数据生成器")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--output", type=str, default="outputs/data", help="输出目录")
    parser.add_argument("--n-samples", type=int, default=100000, help="样本数量")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    # 创建设备
    device = torch.device(args.device)

    # 创建数据生成器
    logger.info("初始化 DataGenerator...")
    data_gen = DataGenerator(config, device)

    # 生成数据
    logger.info(f"生成数据 (n_samples={args.n_samples})...")
    data = data_gen.generate_all_data()

    # 保存数据
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "training_data.pt")

    torch.save(data, output_file)
    logger.info(f"数据已保存到: {output_file}")

    # 打印统计信息
    logger.info("=" * 60)
    logger.info("数据统计:")
    for key, value in data.items():
        if hasattr(value, "shape"):
            logger.info(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            logger.info(f"  {key}: {len(value)} items")
    logger.info("=" * 60)
