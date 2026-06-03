#!/usr/bin/env python3
"""
物理优化采样器
=================

基于EFD3D物理采样优化分析报告实现的采样策略：
1. 电压分区采样（无响应区、起始响应区、线性响应区、饱和区）
2. 动态时间采样（基于电压大小调整时间常数）
3. 油膜破裂物理阶段采样
4. 空间-时间联合采样

作者: EFD-PINNs Team
日期: 2026-04-26
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
                "no_response": 0.10,  # 无响应区
                "onset": 0.50,  # 起始响应区
                "linear": 0.30,  # 线性响应区
                "saturation": 0.10,  # 饱和区
            },
        )

        # 阈值电压参数 — 优先从 Stage1 读取
        if stage1_predictor is not None:
            self.V_T_base = stage1_predictor.params.get(
                "V_T_base", stage1_predictor.params.get("V_threshold", 5.0)
            )
        else:
            self.V_T_base = config.get("threshold_voltage_base", 5.0)
        self.V_T_sensitivity = config.get("threshold_voltage_sensitivity", 2e6)

        # 时间采样配置 — 对齐 Stage1 参数
        if stage1_predictor is not None:
            default_tau_onset = stage1_predictor.params.get("tau_onset", 0.0075)
            default_tau_linear = stage1_predictor.params.get("tau", 0.005)
            default_tau_sat = stage1_predictor.params.get("tau_saturation", 0.003)
        else:
            default_tau_onset, default_tau_linear, default_tau_sat = 0.0075, 0.005, 0.003

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
            "electric_field": (0.0, 0.001),  # 电场建立: 0-1ms
            "marangoni": (0.001, 0.003),  # Marangoni效应: 1-3ms
            "film_instability": (0.003, 0.010),  # 薄膜失稳: 3-10ms
            "local_rupture": (0.010, 0.020),  # 局部破裂: 10-20ms
        }

        logger.info("物理采样器初始化完成")
        logger.info(f"电压分区权重: {self.voltage_weights}")
        logger.info(f"阈值电压: {self.V_T_base}V (基础值, 3μm油膜)")
        if self._stage1_predictor is not None:
            logger.info("使用 Stage1 模型进行精确开口率估算")

    def _calculate_threshold_voltage(self, oil_thickness: float) -> float:
        """
        基于油膜厚度计算阈值电压

        Args:
            oil_thickness: 油膜厚度 (m)

        Returns:
            阈值电压 (V)
        """
        # 基于实际器件观测的线性关系，参考点 3.0μm
        V_T = self.V_T_base + (oil_thickness - 3.0e-6) * self.V_T_sensitivity
        return max(0.1, V_T)

    def sample_voltage_physics_based(
        self, n_samples: int, oil_thickness: float = 3e-6
    ) -> np.ndarray:
        """
        基于物理机制的电压采样

        Args:
            n_samples: 采样数量
            oil_thickness: 油膜厚度，用于计算阈值电压

        Returns:
            采样电压数组
        """
        # 计算当前几何下的阈值电压
        V_T = self._calculate_threshold_voltage(oil_thickness)

        # 定义电压分区边界
        boundaries = {
            "no_response": (0.0, V_T),
            "onset": (V_T, 2 * V_T),
            "linear": (2 * V_T, 4 * V_T),
            "saturation": (4 * V_T, 30.0),
        }

        # 根据权重分配采样数量
        voltages = []

        for region, weight in self.voltage_weights.items():
            n_region = int(n_samples * weight)
            if n_region == 0:
                continue

            v_min, v_max = boundaries[region]

            if region == "no_response":
                # 无响应区：均匀采样
                v_samples = np.random.uniform(v_min, v_max, n_region)
            elif region == "onset":
                # 起始响应区：在阈值附近加密
                v_samples = self._sample_onset_region(v_min, v_max, n_region)
            elif region == "linear":
                # 线性响应区：均匀采样
                v_samples = np.random.uniform(v_min, v_max, n_region)
            else:  # saturation
                # 饱和区：在边界附近加密
                v_samples = self._sample_saturation_region(v_min, v_max, n_region)

            voltages.extend(v_samples)

        # 补充剩余样本
        if len(voltages) < n_samples:
            remaining = n_samples - len(voltages)
            v_extra = np.random.uniform(0, 30.0, remaining)
            voltages.extend(v_extra)

        return np.array(voltages[:n_samples])

    def _sample_onset_region(self, v_min: float, v_max: float, n_samples: int) -> np.ndarray:
        """
        在起始响应区采样（阈值附近加密）
        """
        # 使用Beta分布，在阈值附近采样更多点
        # Beta(0.3, 2.0) 在左边界附近有更高的概率密度
        beta_samples = np.random.beta(0.3, 2.0, n_samples)
        return v_min + beta_samples * (v_max - v_min)

    def _sample_saturation_region(self, v_min: float, v_max: float, n_samples: int) -> np.ndarray:
        """
        在饱和区采样（边界附近加密）
        """
        # 使用Beta分布，在右边界附近有更高的概率密度
        beta_samples = np.random.beta(2.0, 0.3, n_samples)
        return v_min + beta_samples * (v_max - v_min)

    def sample_time_adaptive(
        self, n_samples: int, voltage: float, voltage_prev: float = 0.0
    ) -> np.ndarray:
        """
        自适应时间采样

        Args:
            n_samples: 采样数量
            voltage: 当前电压
            voltage_prev: 前一个电压（用于判断升压/降压）

        Returns:
            采样时间数组
        """
        # 判断电压变化方向
        is_ramp_up = voltage > voltage_prev

        # 基于电压大小选择时间常数
        if voltage < self.V_T_base:
            tau = self.time_config["tau_onset"]
        elif voltage < 2 * self.V_T_base:
            tau = self.time_config["tau_linear"]
        else:
            tau = self.time_config["tau_saturation"]

        # 调整时间常数（降压响应更快）
        if not is_ramp_up:
            tau *= 0.4  # 降压响应速度约为升压的2.5倍

        # 关键时间点采样
        critical_density = self.time_config["critical_points_density"]
        n_critical = int(n_samples * critical_density)
        n_continuous = n_samples - n_critical

        times = []

        # 1. 关键物理阶段时间点采样
        if n_critical > 0:
            critical_times = self._sample_critical_physics_times(n_critical, voltage, voltage_prev)
            times.extend(critical_times)

        # 2. 连续时间采样
        if n_continuous > 0:
            continuous_times = self._sample_continuous_times(n_continuous, tau)
            times.extend(continuous_times)

        return np.array(times[:n_samples])

    def _sample_critical_physics_times(
        self, n_samples: int, voltage: float, voltage_prev: float
    ) -> np.ndarray:
        """
        在关键物理阶段时间点采样
        """
        critical_times = []

        # 根据电压变化选择重点阶段
        if voltage > voltage_prev:  # 升压
            focus_stages = ["marangoni", "film_instability", "local_rupture"]
        else:  # 降压
            focus_stages = ["local_rupture", "film_instability"]

        # 在每个关注阶段采样
        samples_per_stage = n_samples // len(focus_stages)

        for stage in focus_stages:
            t_min, t_max = self.physical_stages[stage]

            # 在该阶段内使用高斯分布采样（集中在阶段中期）
            t_center = (t_min + t_max) / 2
            t_std = (t_max - t_min) / 6  # 覆盖99.7%的范围

            stage_times = np.random.normal(t_center, t_std, samples_per_stage)
            stage_times = np.clip(stage_times, t_min, t_max)

            critical_times.extend(stage_times)

        return np.array(critical_times)

    def _sample_continuous_times(self, n_samples: int, tau: float) -> np.ndarray:
        """
        连续时间采样（指数衰减分布）
        """
        # 使用指数分布模拟暂态过程
        # 在早期时间点采样更密集
        1.0 / tau
        times = np.random.exponential(scale=tau, size=n_samples)

        # 限制最大时间
        t_max = 0.1  # 100ms
        return np.clip(times, 0, t_max)

    def sample_spatial_physics_based(
        self, n_samples: int, voltage: float, time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        基于物理机制的空间采样

        Args:
            n_samples: 采样数量
            voltage: 电压
            time: 时间

        Returns:
            (x, y, z) 坐标数组
        """
        # 基于电压和时间确定界面位置
        eta = self._estimate_opening_rate(voltage, time)

        # 空间采样策略
        x_samples = np.random.uniform(0, 174e-6, n_samples)
        y_samples = np.random.uniform(0, 174e-6, n_samples)

        # Z方向：在界面附近加密采样
        z_samples = self._sample_z_interface(n_samples, eta, voltage, time)

        return x_samples, y_samples, z_samples

    def _estimate_opening_rate(self, voltage: float, time: float) -> float:
        """
        估算开口率

        优先使用 Stage1 精确模型（Young-Lippmann + tanh + 电容反馈）；
        若不可用则回退到修正后的简化模型。
        """
        if self._stage1_predictor is not None and self._stage1_aperture is not None:
            theta_ss = self._stage1_predictor.young_lippmann(voltage)
            if time > 1e-6:
                theta_start = self._stage1_predictor.young_lippmann(0.0)
                theta_t = self._stage1_predictor.dynamic_response(
                    time, theta_start, theta_ss, V_to=voltage
                )
            else:
                theta_t = theta_ss
            return float(self._stage1_aperture.contact_angle_to_aperture_ratio(theta_t))

        # 回退：修正后的简化模型（仅 Stage1 不可用时使用）
        V_T = self.V_T_base
        if voltage <= V_T:
            return 0.0

        V_eff = voltage - V_T
        tau = self.time_config.get("tau_linear", 0.005)
        try:
            from src.config import PHYSICS

            eta_max_ref = PHYSICS["eta_max"]
        except ImportError:
            eta_max_ref = 0.85
        eta_max = min(eta_max_ref, V_eff / 18.0)
        return eta_max * (1.0 - np.exp(-time / tau))

    def _sample_z_interface(
        self, n_samples: int, eta: float, voltage: float, time: float
    ) -> np.ndarray:
        """
        在界面附近采样Z坐标
        """
        # 界面位置（基于开口率）
        h_interface = 3e-6 * (1 - eta)  # 界面高度

        # 界面宽度（动态调整）
        interface_width = self._calculate_interface_width(voltage, time)

        # 采样策略：70%在界面附近，30%在整个域内
        n_interface = int(n_samples * 0.7)
        n_domain = n_samples - n_interface

        z_samples = []

        # 1. 界面附近采样（高斯分布）
        if n_interface > 0:
            z_interface = np.random.normal(h_interface, interface_width, n_interface)
            z_interface = np.clip(z_interface, 0, 20e-6)
            z_samples.extend(z_interface)

        # 2. 全域采样
        if n_domain > 0:
            z_domain = np.random.uniform(0, 20e-6, n_domain)
            z_samples.extend(z_domain)

        return np.array(z_samples)

    def _calculate_interface_width(self, voltage: float, time: float) -> float:
        """
        计算界面宽度
        """
        # 基于电压和时间的界面宽度
        # 高电压和长时间导致更宽的界面
        V_factor = min(1.0, voltage / 30.0)
        t_factor = min(1.0, time / 0.02)

        base_width = 0.5e-6  # 基础界面宽度
        max_width = 2e-6  # 最大界面宽度

        return base_width + (max_width - base_width) * V_factor * t_factor


def create_physics_sampling_dataset(config: dict, n_total: int = 100000) -> dict[str, np.ndarray]:
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
        # 电压采样
        voltage = sampler.sample_voltage_physics_based(1)[0]
        voltage_prev = 0.0 if np.random.rand() > 0.5 else voltage

        # 时间采样
        time = sampler.sample_time_adaptive(1, voltage, voltage_prev)[0]

        # 空间采样
        x, y, z = sampler.sample_spatial_physics_based(1, voltage, time)

        # 目标值（简化）
        phi = 0.5 * (1 - np.tanh((z[0] - 3e-6) / 1e-6))

        dataset["interface_points"].append([x[0], y[0], z[0], voltage_prev, voltage, time])
        dataset["interface_targets"].append(phi)

    # 转换为numpy数组
    for key in dataset:
        dataset[key] = np.array(dataset[key])

    logger.info(
        f"数据集生成完成，总样本数: {len(dataset['interface_points']) + len(dataset['initial_points']) + len(dataset['boundary_points'])}"
    )

    return dataset


if __name__ == "__main__":
    # 测试采样器
    test_config = {
        "voltage_weights": {"no_response": 0.15, "onset": 0.35, "linear": 0.35, "saturation": 0.15},
        "threshold_voltage_base": 5.0,
        "time_sampling": {"critical_points_density": 0.6, "adaptive_tau": True},
    }

    # 创建采样器
    sampler = PhysicsBasedSampler(test_config)

    # 测试电压采样
    voltages = sampler.sample_voltage_physics_based(1000)
    logger.info("电压采样统计:")
    logger.info(f"  最小值: {voltages.min():.2f}V")
    logger.info(f"  最大值: {voltages.max():.2f}V")
    logger.info(f"  平均值: {voltages.mean():.2f}V")
    logger.info(
        f"  阈值附近比例: {np.sum((voltages >= 4.5) & (voltages <= 6.5)) / len(voltages) * 100:.1f}%"
    )

    # 测试时间采样
    times_up = sampler.sample_time_adaptive(100, 10.0, 0.0)  # 升压
    times_down = sampler.sample_time_adaptive(100, 0.0, 10.0)  # 降压

    logger.info("\n时间采样统计:")
    logger.info(f"  升压 - 平均时间: {times_up.mean() * 1000:.2f}ms")
    logger.info(f"  降压 - 平均时间: {times_down.mean() * 1000:.2f}ms")

    # 生成完整数据集
    logger.info("\n生成完整数据集...")
    dataset = create_physics_sampling_dataset(test_config, 10000)
    logger.info(f"数据集大小: {len(dataset['interface_points'])} 界面点")
