# physics_sampling.py — 物理优化采样器

> 基于EFD3D物理采样优化分析报告实现的采样策略。包含4种电压分区采样、动态时间采样、油膜破裂物理阶段采样、空间-时间联合采样。

> 文件路径: `/home/scnu/Gitee/EFD3D/src/data/physics_sampling.py`
> 总行数: 556

---

## 目录

### 类 (class)

- **L23** `PhysicsBasedSampler`

### 公开函数

- **L121** `sample_voltage_physics_based`
- **L199** `sample_time_adaptive`
- **L290** `sample_spatial_physics_based`
- **L467** `create_physics_sampling_dataset`

### 私有函数

- **L34** `__init__`
- **L107** `_calculate_threshold_voltage`
- **L182** `_sample_onset_region`
- **L191** `_sample_saturation_region`
- **L248** `_sample_critical_physics_times`
- **L277** `_sample_continuous_times`
- **L320** `_estimate_opening_rate`
- **L354** `_estimate_contact_angle`
- **L405** `_sample_z_interface`
- **L452** `_calculate_interface_width`

---

## 源码

```python
   1 | #!/usr/bin/env python3
     | """
     | 物理优化采样器
     | =================
     |
   6 | 基于EFD3D物理采样优化分析报告实现的采样策略：
     | 1. 电压分区采样（无响应区、起始响应区、线性响应区、饱和区）
     | 2. 动态时间采样（基于电压大小调整时间常数）
     | 3. 油膜破裂物理阶段采样
     | 4. 空间-时间联合采样
  11 |
     | 作者: EFD-PINNs Team
     | 日期: 2026-04-26
     | """
     |
  16 | import logging
     |
     | import numpy as np
     |
     | logger = logging.getLogger(__name__)
  21 |
     |
  23 | class PhysicsBasedSampler:
     |     """
     |     基于物理机制的采样器
  26 |
     |     实现四种电压分区采样策略：
     |     - 无响应区 (0-V_T): 采样权重可配置
     |     - 起始响应区 (V_T-2V_T): 采样权重可配置
     |     - 线性响应区 (2V_T-4V_T): 采样权重可配置
  31 |     - 饱和区 (>4V_T): 采样权重可配置
     |     """
     |
  34 |     def __init__(self, config: dict, stage1_predictor=None, stage1_aperture_model=None):
     |         """
  36 |         初始化物理采样器
     |
     |         Args:
     |             config: 配置字典，包含采样参数
     |             stage1_predictor: HybridPredictor 实例（用于精确开口率估算）
  41 |             stage1_aperture_model: EnhancedApertureModel 实例
     |         """
     |         self.config = config
     |
     |         # Stage1 模型引用（用于精确物理计算）
  46 |         self._stage1_predictor = stage1_predictor
     |         self._stage1_aperture = stage1_aperture_model
     |
     |         # 电压分区配置
     |         self.voltage_weights = config.get(
  51 |             "voltage_weights",
     |             {
     |                 "no_response": 0.10,  # 无响应区
     |                 "onset": 0.50,  # 起始响应区
     |                 "linear": 0.30,  # 线性响应区
  56 |                 "saturation": 0.10,  # 饱和区
     |             },
     |         )
     |
     |         # 阈值电压参数 — 优先从 Stage1 读取，回退到 PHYSICS（单一来源）
  61 |         from src.config import PHYSICS
     |
     |         if stage1_predictor is not None:
     |             V_T_fallback = PHYSICS["V_T_base"]
     |             self.V_T_base = stage1_predictor.params.get(
  66 |                 "V_T_base", stage1_predictor.params.get("V_threshold", V_T_fallback)
     |             )
     |         else:
     |             self.V_T_base = PHYSICS["V_T_base"]
     |         self.V_T_sensitivity = PHYSICS["V_T_sensitivity"]
  71 |
     |         # 时间采样配置 — 从 PHYSICS 读取校准后的 τ 值
     |         if stage1_predictor is not None:
     |             default_tau_onset = stage1_predictor.params.get("tau_onset", PHYSICS["tau_onset"])
     |             default_tau_linear = stage1_predictor.params.get("tau", PHYSICS["tau"])
  76 |             default_tau_sat = stage1_predictor.params.get("tau_saturation", PHYSICS["tau_saturation"])
     |         else:
     |             default_tau_onset = PHYSICS["tau_onset"]
     |             default_tau_linear = PHYSICS["tau"]
     |             default_tau_sat = PHYSICS["tau_saturation"]
  81 |
     |         self.time_config = config.get(
     |             "time_sampling",
     |             {
     |                 "critical_points_density": 0.6,
  86 |                 "adaptive_tau": True,
     |                 "tau_onset": default_tau_onset,
     |                 "tau_linear": default_tau_linear,
     |                 "tau_saturation": default_tau_sat,
     |             },
  91 |         )
     |
     |         # 物理阶段时间点
     |         self.physical_stages = {
     |             "electric_field": (0.0, 0.001),  # 电场建立: 0-1ms
  96 |             "marangoni": (0.001, 0.003),  # Marangoni效应: 1-3ms
     |             "film_instability": (0.003, 0.010),  # 薄膜失稳: 3-10ms
     |             "local_rupture": (0.010, 0.020),  # 局部破裂: 10-20ms
     |         }
     |
 101 |         logger.info("物理采样器初始化完成")
     |         logger.info(f"电压分区权重: {self.voltage_weights}")
     |         logger.info(f"阈值电压: {self.V_T_base}V (基础值, 3μm油膜)")
     |         if self._stage1_predictor is not None:
     |             logger.info("使用 Stage1 模型进行精确开口率估算")
 106 |
 107 |     def _calculate_threshold_voltage(self, oil_thickness: float) -> float:
     |         """
     |         基于油膜厚度计算阈值电压
     |
 111 |         Args:
     |             oil_thickness: 油膜厚度 (m)
     |
     |         Returns:
     |             阈值电压 (V)
 116 |         """
     |         # 基于实际器件观测的线性关系，参考点 3.0μm
     |         V_T = self.V_T_base + (oil_thickness - 3.0e-6) * self.V_T_sensitivity
     |         return max(0.1, V_T)
     |
 121 |     def sample_voltage_physics_based(self, n_samples: int, oil_thickness: float = 3e-6) -> np.ndarray:
     |         """
     |         基于物理机制的电压采样
     |
     |         Args:
 126 |             n_samples: 采样数量
     |             oil_thickness: 油膜厚度，用于计算阈值电压
     |
     |         Returns:
     |             采样电压数组
 131 |         """
     |         # 计算当前几何下的阈值电压
     |         V_T = self._calculate_threshold_voltage(oil_thickness)
     |
     |         # 定义电压分区边界
 136 |         boundaries = {
     |             "no_response": (0.0, V_T),
     |             "onset": (V_T, 2 * V_T),
     |             "linear": (2 * V_T, 4 * V_T),
     |             "saturation": (4 * V_T, 30.0),
 141 |         }
     |
     |         # 根据权重分配采样数量
     |         voltages = []
     |
 146 |         for region, weight in self.voltage_weights.items():
     |             n_region = int(n_samples * weight)
     |             if n_region == 0:
     |                 continue
     |
 151 |             v_min, v_max = boundaries[region]
     |
     |             if region == "no_response":
     |                 # 无响应区：均匀采样
     |                 v_samples = np.random.uniform(v_min, v_max, n_region)
 156 |             elif region == "onset":
     |                 # 起始响应区：在阈值附近加密
     |                 v_samples = self._sample_onset_region(v_min, v_max, n_region)
     |             elif region == "linear":
     |                 # 线性响应区：均匀采样
 161 |                 v_samples = np.random.uniform(v_min, v_max, n_region)
     |             else:  # saturation
     |                 # 饱和区：在边界附近加密
     |                 v_samples = self._sample_saturation_region(v_min, v_max, n_region)
     |
 166 |             voltages.extend(v_samples)
     |
     |         # 补充剩余样本（上限从 PHYSICS 读取，与 DataGenerator 一致）
     |         from src.config import PHYSICS
     |
 171 |         V_max = PHYSICS["V_max"]  # 30.0
     |         if len(voltages) < n_samples:
     |             remaining = n_samples - len(voltages)
     |             v_extra = np.random.uniform(0, V_max, remaining)
     |             voltages.extend(v_extra)
 176 |
     |         # 限制最大电压为 V_max
     |         voltages = np.clip(voltages, 0, V_max)
     |
     |         return np.array(voltages[:n_samples])
 181 |
 182 |     def _sample_onset_region(self, v_min: float, v_max: float, n_samples: int) -> np.ndarray:
     |         """
     |         在起始响应区采样（阈值附近加密）
     |         """
 186 |         # 使用Beta分布，在阈值附近采样更多点
     |         # Beta(0.3, 2.0) 在左边界附近有更高的概率密度
     |         beta_samples = np.random.beta(0.3, 2.0, n_samples)
     |         return v_min + beta_samples * (v_max - v_min)
     |
 191 |     def _sample_saturation_region(self, v_min: float, v_max: float, n_samples: int) -> np.ndarray:
     |         """
     |         在饱和区采样（边界附近加密）
     |         """
     |         # 使用Beta分布，在右边界附近有更高的概率密度
 196 |         beta_samples = np.random.beta(2.0, 0.3, n_samples)
     |         return v_min + beta_samples * (v_max - v_min)
     |
 199 |     def sample_time_adaptive(self, n_samples: int, voltage: float, voltage_prev: float = 0.0) -> np.ndarray:
     |         """
 201 |         自适应时间采样
     |
     |         Args:
     |             n_samples: 采样数量
     |             voltage: 当前电压
 206 |             voltage_prev: 前一个电压（用于判断升压/降压）
     |
     |         Returns:
     |             采样时间数组
     |         """
 211 |         # 判断电压变化方向
     |         is_ramp_up = voltage > voltage_prev
     |
     |         # 基于电压大小选择时间常数
     |         if voltage < self.V_T_base:
 216 |             tau = self.time_config["tau_onset"]
     |         elif voltage < 2 * self.V_T_base:
     |             tau = self.time_config["tau_linear"]
     |         else:
     |             tau = self.time_config["tau_saturation"]
 221 |
     |         # 调整时间常数（降压响应更快，因子从 PHYSICS 读取）
     |         if not is_ramp_up:
     |             from src.config import PHYSICS
     |
 226 |             tau_recovery_factor = PHYSICS.get("tau_recovery_factor", 0.85)
     |             tau *= tau_recovery_factor
     |
     |         # 关键时间点采样
     |         critical_density = self.time_config["critical_points_density"]
 231 |         n_critical = int(n_samples * critical_density)
     |         n_continuous = n_samples - n_critical
     |
     |         times = []
     |
 236 |         # 1. 关键物理阶段时间点采样
     |         if n_critical > 0:
     |             critical_times = self._sample_critical_physics_times(n_critical, voltage, voltage_prev)
     |             times.extend(critical_times)
     |
 241 |         # 2. 连续时间采样
     |         if n_continuous > 0:
     |             continuous_times = self._sample_continuous_times(n_continuous, tau)
     |             times.extend(continuous_times)
     |
 246 |         return np.array(times[:n_samples])
     |
 248 |     def _sample_critical_physics_times(self, n_samples: int, voltage: float, voltage_prev: float) -> np.ndarray:
     |         """
     |         在关键物理阶段时间点采样
 251 |         """
     |         critical_times = []
     |
     |         # 根据电压变化选择重点阶段
     |         if voltage > voltage_prev:  # 升压
 256 |             focus_stages = ["marangoni", "film_instability", "local_rupture"]
     |         else:  # 降压
     |             focus_stages = ["local_rupture", "film_instability"]
     |
     |         # 在每个关注阶段采样
 261 |         samples_per_stage = n_samples // len(focus_stages)
     |
     |         for stage in focus_stages:
     |             t_min, t_max = self.physical_stages[stage]
     |
 266 |             # 在该阶段内使用高斯分布采样（集中在阶段中期）
     |             t_center = (t_min + t_max) / 2
     |             t_std = (t_max - t_min) / 6  # 覆盖99.7%的范围
     |
     |             stage_times = np.random.normal(t_center, t_std, samples_per_stage)
 271 |             stage_times = np.clip(stage_times, t_min, t_max)
     |
     |             critical_times.extend(stage_times)
     |
     |         return np.array(critical_times)
 276 |
 277 |     def _sample_continuous_times(self, n_samples: int, tau: float) -> np.ndarray:
     |         """
     |         连续时间采样（指数衰减分布）
     |         """
 281 |         # 使用指数分布模拟暂态过程
     |         # 在早期时间点采样更密集
     |         1.0 / tau
     |         times = np.random.exponential(scale=tau, size=n_samples)
     |
 286 |         # 限制最大时间
     |         t_max = 0.1  # 100ms
     |         return np.clip(times, 0, t_max)
     |
 290 |     def sample_spatial_physics_based(
 291 |         self, n_samples: int, voltage: float, time: float
     |     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
     |         """
     |         基于物理机制的空间采样
     |
 296 |         Args:
     |             n_samples: 采样数量
     |             voltage: 电压
     |             time: 时间
     |
 301 |         Returns:
     |             (x, y, z) 坐标数组
     |         """
     |         # 基于电压和时间确定界面位置
     |         eta = self._estimate_opening_rate(voltage, time)
 306 |
     |         # 计算开口半径和接触角
     |         r_open = np.sqrt(eta * 174e-6 * 174e-6 / np.pi) if eta > 0.01 else 0.0
     |         theta = self._estimate_contact_angle(voltage, time)
     |
 311 |         # 空间采样策略
     |         x_samples = np.random.uniform(0, 174e-6, n_samples)
     |         y_samples = np.random.uniform(0, 174e-6, n_samples)
     |
     |         # Z方向：在界面附近加密采样（考虑倾斜界面）
 316 |         z_samples = self._sample_z_interface(n_samples, eta, voltage, time, r_open, theta)
     |
     |         return x_samples, y_samples, z_samples
     |
 320 |     def _estimate_opening_rate(self, voltage: float, time: float) -> float:
 321 |         """
     |         估算开口率
     |
     |         优先使用 Stage1 精确模型（Young-Lippmann + tanh + 电容反馈）；
     |         若不可用则回退到余弦映射模型（与 DataGenerator.get_opening_rate 一致）。
 326 |         """
     |         if self._stage1_predictor is not None and self._stage1_aperture is not None:
     |             theta_ss = self._stage1_predictor.young_lippmann(voltage)
     |             if time > 1e-6:
     |                 theta_start = self._stage1_predictor.young_lippmann(0.0)
 331 |                 theta_t = self._stage1_predictor.dynamic_response(time, theta_start, theta_ss, V_to=voltage)
     |             else:
     |                 theta_t = theta_ss
     |             return float(self._stage1_aperture.contact_angle_to_aperture_ratio(theta_t))
     |
 336 |         # 回退：tanh 饱和映射（与 DataGenerator.get_opening_rate 一致，参数从 PHYSICS 读取）
     |         from src.config import PHYSICS
     |
     |         V_T = self.V_T_base
     |         if voltage <= V_T:
 341 |             return 0.0
     |
     |         theta = self._estimate_contact_angle(voltage, time)
     |         theta0 = PHYSICS["theta0"]
     |         eta_max_aperture = PHYSICS.get("eta_max_aperture", 0.85)
 346 |         k = PHYSICS.get("aperture_k", 2.02)
     |         theta_scale = PHYSICS.get("aperture_theta_scale", 27.46)
     |
     |         dtheta = theta0 - theta
     |         if dtheta <= 0:
 351 |             return 0.0
     |         return float(np.clip(eta_max_aperture * np.tanh(k * dtheta / theta_scale), 0.0, eta_max_aperture))
     |
 354 |     def _estimate_contact_angle(self, voltage: float, time: float) -> float:
     |         """
 356 |         估算接触角（度）
     |
     |         优先使用 Stage1 精确模型；若不可用则回退到 Young-Lippmann + 动态响应。
     |         回退模型所有参数从 PHYSICS 读取，与 DataGenerator._analytical_contact_angle 一致。
     |         """
 361 |         if self._stage1_predictor is not None:
     |             theta_ss = self._stage1_predictor.young_lippmann(voltage)
     |             if time > 1e-6:
     |                 theta_start = self._stage1_predictor.young_lippmann(0.0)
     |                 return self._stage1_predictor.dynamic_response(time, theta_start, theta_ss, V_to=voltage)
 366 |             return theta_ss
     |
     |         # 回退：Young-Lippmann + 一阶动态响应（参数全部从 PHYSICS 读取）
     |         from src.config import PHYSICS
     |
 371 |         theta0 = PHYSICS["theta0"]
     |         V_threshold = PHYSICS["V_threshold"]
     |         V_eff = max(0.0, voltage - V_threshold)
     |
     |         if V_eff <= 0.0:
 376 |             return theta0
     |
     |         # 介电层串联电容（简化单层模型，与 _analytical_contact_angle 一致）
     |         epsilon_0 = PHYSICS["epsilon_0"]
     |         epsilon_r = PHYSICS["epsilon_r"]
 381 |         d_dielectric = PHYSICS["d_dielectric"]
     |         sigma = PHYSICS["sigma"]
     |         C_total = epsilon_0 * epsilon_r / d_dielectric
     |
     |         cos_theta0 = np.cos(np.radians(theta0))
 386 |         ew_term = C_total * V_eff**2 / (2.0 * sigma)
     |         cos_theta_eq = np.clip(cos_theta0 + ew_term, -1.0, 1.0)
     |         theta_eq = np.degrees(np.arccos(cos_theta_eq))
     |
     |         # 一阶动态响应（zeta=1.0 → 临界阻尼，无振荡）
 391 |         tau = PHYSICS["tau"]
     |         zeta = PHYSICS["zeta"]
     |         omega_0 = 1.0 / tau
     |         if zeta >= 1.0:
     |             theta_t = theta_eq + (theta0 - theta_eq) * np.exp(-omega_0 * time)
 396 |         else:
     |             omega_d = omega_0 * np.sqrt(1.0 - zeta**2)
     |             exp_term = np.exp(-zeta * omega_0 * time)
     |             damping = zeta / np.sqrt(1.0 - zeta**2)
     |             theta_t = theta_eq + (theta0 - theta_eq) * exp_term * (
 401 |                 np.cos(omega_d * time) + damping * np.sin(omega_d * time)
     |             )
     |         return theta_t
     |
 405 |     def _sample_z_interface(
 406 |         self, n_samples: int, eta: float, voltage: float, time: float, r_open: float = 0.0, theta: float = 120.0
     |     ) -> np.ndarray:
     |         """
     |         在界面附近采样Z坐标（考虑倾斜界面）
     |
 411 |         target3D 使用倾斜界面：z_tilt = z + (r - r_open) * tan(theta)
     |         界面位置：z = h_edge - (r - r_open) * tan(theta)
     |
     |         Args:
     |             n_samples: 采样数量
 416 |             eta: 开口率
     |             voltage: 电压
     |             time: 时间
     |             r_open: 开口半径
     |             theta: 接触角（度）
 421 |         """
     |         # 界面高度（基于开口率）
     |         h_edge = 3e-6 / max(1.0 - eta, 0.15)
     |
     |         # 界面宽度（动态调整）
 426 |         interface_width = self._calculate_interface_width(voltage, time)
     |
     |         # 采样策略：70%在界面附近，30%在整个域内
     |         n_interface = int(n_samples * 0.7)
     |         n_domain = n_samples - n_interface
 431 |
     |         z_samples = []
     |
     |         # 1. 界面附近采样（考虑倾斜界面）
     |         if n_interface > 0:
 436 |             # 在 r_open 附近采样（界面最清晰的位置）
     |             # 界面位置：z = h_edge - (r - r_open) * tan(theta)
     |             # 在 r = r_open 处，z = h_edge
     |
     |             # 在界面附近采样 z
 441 |             z_interface = np.random.normal(h_edge, interface_width, n_interface)
     |             z_interface = np.clip(z_interface, 0, 20e-6)
     |             z_samples.extend(z_interface)
     |
     |         # 2. 全域采样
 446 |         if n_domain > 0:
     |             z_domain = np.random.uniform(0, 20e-6, n_domain)
     |             z_samples.extend(z_domain)
     |
     |         return np.array(z_samples)
 451 |
 452 |     def _calculate_interface_width(self, voltage: float, time: float) -> float:
     |         """
     |         计算界面宽度
     |         """
 456 |         # 基于电压和时间的界面宽度
     |         # 高电压和长时间导致更宽的界面
     |         V_factor = min(1.0, voltage / 30.0)
     |         t_factor = min(1.0, time / 0.02)
     |
 461 |         base_width = 0.5e-6  # 基础界面宽度
     |         max_width = 2e-6  # 最大界面宽度
     |
     |         return base_width + (max_width - base_width) * V_factor * t_factor
     |
 466 |
 467 | def create_physics_sampling_dataset(config: dict, n_total: int = 100000) -> dict[str, np.ndarray]:
     |     """
     |     创建完整的物理优化数据集
     |
 471 |     Args:
     |         config: 配置字典
     |         n_total: 总样本数
     |
     |     Returns:
 476 |         包含所有数据的字典
     |     """
     |     sampler = PhysicsBasedSampler(config)
     |
     |     # 数据分配
 481 |     n_interface = int(n_total * 0.6)  # 60% 界面数据
     |     n_initial = int(n_total * 0.2)  # 20% 初始条件
     |     n_boundary = n_total - n_interface - n_initial  # 剩余为边界条件
     |
     |     logger.info(f"数据集分配: 界面={n_interface}, 初始={n_initial}, 边界={n_boundary}")
 486 |
     |     # 生成数据
     |     dataset = {
     |         "interface_points": [],
     |         "interface_targets": [],
 491 |         "initial_points": [],
     |         "initial_targets": [],
     |         "boundary_points": [],
     |         "boundary_targets": [],
     |     }
 496 |
     |     # 1. 界面数据
     |     logger.info("生成界面数据...")
     |     for _ in range(n_interface):
     |         # 电压采样
 501 |         voltage = sampler.sample_voltage_physics_based(1)[0]
     |         voltage_prev = 0.0 if np.random.rand() > 0.5 else voltage
     |
     |         # 时间采样
     |         time = sampler.sample_time_adaptive(1, voltage, voltage_prev)[0]
 506 |
     |         # 空间采样
     |         x, y, z = sampler.sample_spatial_physics_based(1, voltage, time)
     |
     |         # 目标值（简化）
 511 |         phi = 0.5 * (1 - np.tanh((z[0] - 3e-6) / 1e-6))
     |
     |         dataset["interface_points"].append([x[0], y[0], z[0], voltage_prev, voltage, time])
     |         dataset["interface_targets"].append(phi)
     |
 516 |     # 转换为numpy数组
     |     for key, values in dataset.items():
     |         dataset[key] = np.array(values)
     |
     |     total_samples = len(dataset["interface_points"]) + len(dataset["initial_points"]) + len(dataset["boundary_points"])
 521 |     logger.info(f"数据集生成完成，总样本数: {total_samples}")
     |
     |     return dataset
     |
     |
 526 | if __name__ == "__main__":
     |     # 测试采样器
     |     test_config = {
     |         "voltage_weights": {"no_response": 0.15, "onset": 0.35, "linear": 0.35, "saturation": 0.15},
     |         "threshold_voltage_base": 5.0,
 531 |         "time_sampling": {"critical_points_density": 0.6, "adaptive_tau": True},
     |     }
     |
     |     # 创建采样器
     |     sampler = PhysicsBasedSampler(test_config)
 536 |
     |     # 测试电压采样
     |     voltages = sampler.sample_voltage_physics_based(1000)
     |     logger.info("电压采样统计:")
     |     logger.info(f"  最小值: {voltages.min():.2f}V")
 541 |     logger.info(f"  最大值: {voltages.max():.2f}V")
     |     logger.info(f"  平均值: {voltages.mean():.2f}V")
     |     logger.info(f"  阈值附近比例: {np.sum((voltages >= 4.5) & (voltages <= 6.5)) / len(voltages) * 100:.1f}%")
     |
     |     # 测试时间采样
 546 |     times_up = sampler.sample_time_adaptive(100, 10.0, 0.0)  # 升压
     |     times_down = sampler.sample_time_adaptive(100, 0.0, 10.0)  # 降压
     |
     |     logger.info("\n时间采样统计:")
     |     logger.info(f"  升压 - 平均时间: {times_up.mean() * 1000:.2f}ms")
 551 |     logger.info(f"  降压 - 平均时间: {times_down.mean() * 1000:.2f}ms")
     |
     |     # 生成完整数据集
     |     logger.info("\n生成完整数据集...")
     |     dataset = create_physics_sampling_dataset(test_config, 10000)
 556 |     logger.info(f"数据集大小: {len(dataset['interface_points'])} 界面点")
```
