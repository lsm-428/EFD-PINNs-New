#!/usr/bin/env python3
"""
EWP 两相流 PINN - 整合优化版
============================

整合所有最佳实践的物理信息神经网络：
1. 径向坐标转换（利用轴对称性）
2. 分离网络架构（phi 和速度场分开）
3. 完整物理损失（连续性 + VOF + Navier-Stokes）
4. 渐进式训练策略
5. 界面加密采样

物理方程：
- 连续性：∇·u = 0
- VOF：∂φ/∂t + u·∇φ = 0
- N-S：ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u

作者: EFD-PINNs Team
日期: 2024-12
"""

import argparse
import concurrent.futures
import datetime
import json
import logging
import os
import random
import threading
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.physics.constraints import PhysicsConstraints


def set_seed(seed: int = 42, deterministic: bool = False):
    """设置全局随机种子，确保训练可复现。

    Args:
        seed: 随机种子
        deterministic: 是否强制确定性计算（降低性能但可复现）。
                       默认 False，启用 cudnn.benchmark 加速 GPU 运算。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# 配置日志
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("EWP-PINN")

# 导入项目模块
try:
    from src.models.aperture_model import EnhancedApertureModel

    HAS_APERTURE = True
except ImportError:
    HAS_APERTURE = False
    logger.warning("EnhancedApertureModel 不可用，使用解析公式")

try:
    from src.predictors.hybrid_predictor import HybridPredictor

    HAS_HYBRID_PREDICTOR = True
except ImportError:
    HAS_HYBRID_PREDICTOR = False
    logger.warning("HybridPredictor 不可用，使用解析公式计算接触角")


# ============================================================================
# 物理常量 - 从统一配置模块导入
# ============================================================================
from src.config import PHYSICS

# ============================================================================
# φ 场定义（标准 VOF）
# ============================================================================
# φ = 1: 纯油墨
# φ = 0: 纯极性液体（透明）
# 0 < φ < 1: 界面过渡区
#
# 初始状态 (t=0, V=0):
#   z < h_ink (3μm): φ = 1 (油墨层)
#   z > h_ink: φ = 0 (极性液体层)
#
# 电压响应后:
#   中心区域: φ → 0 (透明，极性液体下沉到基底)
#   边缘区域: φ = 1 且油墨堆高 (体积守恒)
# ============================================================================


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_CONFIG = {
    "model": {
        "hidden_phi": [64, 64, 64, 32],  # phi 网络
        "hidden_vel": [64, 64, 32],  # 速度网络
    },
    "training": {
        "epochs": 30000,
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "min_lr": 1e-6,
        "gradient_clip": 1.0,
        # 渐进式训练阶段
        "stage1_epochs": 5000,  # 纯数据学习
        "stage2_epochs": 15000,  # 引入连续性+VOF
        "stage3_epochs": 30000,  # 完整物理约束
    },
    "physics": {
        # 数据损失权重
        "interface_weight": 500.0,
        "ic_weight": 100.0,
        "bc_weight": 50.0,
        # 物理损失权重（使用 log1p 缩放后的权重）
        "continuity_weight": 0.5,  # 连续性方程
        "vof_weight": 0.5,  # VOF 方程
        "ns_weight": 0.1,  # Navier-Stokes
        "surface_tension_weight": 0.01,  # 表面张力 CSF
        "sharpening": 0.1,  # [Added] VOF Sharpening Loss
    },
    "data": {
        "n_interface": 100000,
        "n_initial": 10000,
        "n_boundary": 10000,
        "n_domain": 20000,
        "voltages": [0.5 * i for i in range(0, 61)],
        "times": 50,
    },
}


# ============================================================================
# 两相流 PINN 模型
# ============================================================================


class FourierFeature(nn.Module):
    """Fourier 特征编码 — 缓解 PINN 谱偏置，改善 XY 高频特征学习"""

    def __init__(self, in_dim: int, mapping_size: int = 32, sigma: float = 3.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(mapping_size, in_dim) * sigma, requires_grad=False)
        self.out_dim = 2 * mapping_size

    def forward(self, x):
        proj = 2 * torch.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TwoPhasePINN(nn.Module):
    """
    两相流物理信息神经网络

    特点：
    - 支持升压和降压过程
    - 分离 phi 网络和速度网络
    - 输入: (x, y, z, V_from, V_to, t_since) - 6维（三元组格式）
      * V_from: 跳变前电压
      * V_to: 跳变后电压（当前电压）
      * t_since: 跳变后经过的时间
    - 输出: (u, v, w, p, phi)

    电压历史的物理意义：
    - V_from = V_to: 恒定电压状态
    - V_from < V_to: 升压过程（电润湿驱动）
    - V_from > V_to: 降压过程（表面张力恢复）
    """

    def __init__(self, config: dict[str, Any] = None):
        super().__init__()
        config = config or DEFAULT_CONFIG

        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
        self.Lz = PHYSICS["Lz"]
        self.t_max = PHYSICS["t_max"]
        self.cx = self.Lx / 2
        self.cy = self.Ly / 2
        self.r_max = np.sqrt(self.cx**2 + self.cy**2)

        # 网络配置
        model_cfg = config.get("model", {})
        hidden_phi = model_cfg.get("hidden_phi", [128, 128, 64, 32])
        hidden_vel = model_cfg.get("hidden_vel", [64, 64, 32])

        # Fourier 特征编码（空间坐标 → 高频特征，缓解谱偏置）
        fourier_size = model_cfg.get("fourier_mapping_size", 16)
        fourier_sigma = model_cfg.get("fourier_sigma", 3.0)
        self.use_fourier = model_cfg.get("use_fourier", True)
        if self.use_fourier:
            self.fourier = FourierFeature(3, mapping_size=fourier_size, sigma=fourier_sigma)
            spatial_features = self.fourier.out_dim  # 2*16*3 = 96
        else:
            spatial_features = 3

        # Phi 网络: Fourier(spatial) + V_from + V_to + t → phi
        phi_input = spatial_features + 3
        self.phi_net = self._build_network(phi_input, 1, hidden_phi)

        # 速度网络: Fourier(spatial) + V_from + V_to + t + phi → u,v,w,p
        vel_input = spatial_features + 4
        self.vel_net = self._build_network(vel_input, 4, hidden_vel)

        # 硬约束配置
        hard_cfg = config.get("hard_constraints", {})
        self.use_hard_constraints = hard_cfg.get("enable", False)
        self.h_ink = hard_cfg.get("h_ink", 3e-6)
        self.hard_ic_width = hard_cfg.get("ic_width", 1e-6)
        self.sigmoid_temperature = hard_cfg.get("sigmoid_temperature", 1.0)

        self.apply(self._init_weights)

    def _build_network(self, input_dim: int, output_dim: int, hidden_layers: list) -> nn.Sequential:
        """构建全连接网络"""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        """Xavier 初始化"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (支持硬约束编码)

        Args:
            x: (batch, 6) - (x, y, z, V_from, V_to, t_since)

        Returns:
            (batch, 5) - (u, v, w, p, phi)

        硬约束 (通过构造保证, 不需loss):
          - 顶面 BC: φ(z=Lz) = 0  →  φ *= (1 - z/Lz)
          - 初始条件: φ(t=0) = φ_IC(z)  →  φ = φ_IC + t/t_max * (φ_learned - φ_IC)
        """
        if not torch.jit.is_tracing() and x.shape[1] != 6:
            raise ValueError("TwoPhasePINN expects input of shape (batch, 6).")
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        V_from = x[:, 3:4]
        V_to = x[:, 4:5]
        t_since = x[:, 5:6]

        # 归一化到 [0, 1]
        x_norm = x_coord / self.Lx
        y_norm = y_coord / self.Ly
        z_norm = z_coord / self.Lz
        t_norm = t_since / self.t_max
        V_from_norm = V_from / 30.0
        V_to_norm = V_to / 30.0

        # 空间 Fourier 特征编码
        if self.use_fourier:
            spatial_ff = self.fourier(torch.cat([x_norm, y_norm, z_norm], dim=-1))
            phi_features = torch.cat([spatial_ff, V_from_norm, V_to_norm, t_norm], dim=-1)
        else:
            phi_features = torch.cat(
                [x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm], dim=-1
            )

        # Phi 预测
        phi_input = phi_features
        phi_raw = self.phi_net(phi_input)
        sigmoid_T = getattr(self, "sigmoid_temperature", 1.0)
        phi_learned = torch.sigmoid(sigmoid_T * phi_raw)  # 限制在 [0, 1]

        # --- 硬约束编码 ---
        if getattr(self, "use_hard_constraints", False):
            # 1. 顶面 BC: φ(z=Lz) = 0 (纯极性液体)
            phi = phi_learned * (1.0 - z_norm)

            # 2. 初始条件: φ(t=0) = φ_IC(z)
            #    φ_IC: tanh 平滑台阶, 油墨(z<h_ink)→1, 极性液体(z>h_ink)→0
            z_phys = z_coord  # 物理单位 (m)
            h_ink = getattr(self, "h_ink", 3e-6)
            delta_ic = getattr(self, "hard_ic_width", 1e-6)
            phi_ic = 0.5 * (1.0 + torch.tanh((h_ink - z_phys) / delta_ic))

            #    Blend: 混合 phi_ic（初始条件）和 phi（模型预测+顶面BC）
            #      公式：phi = (1 - blend) * phi_ic + blend * phi
            #      blend = 0 → phi = phi_ic（完全约束到初始条件）
            #      blend = 1 → phi = phi（完全自由）
            #      设计：
            #        z=0 底面：blend ≡ 1（永远自由，不约束 xy 径向分布）
            #        z>0：blend = t_norm（t=0→0 完全IC，t>0→逐渐自由）
            z_mask = (z_norm > 1e-6).float()  # z=0→0, z>0→1
            blend = 1.0 - z_mask * (1.0 - t_norm)  # z=0→1, z>0→t_norm
            phi = (1.0 - blend) * phi_ic + blend * phi
        else:
            phi = phi_learned

        # 速度预测 - 使用 Fourier 特征 + phi
        vel_input = torch.cat([phi_features, phi], dim=-1)
        vel_out = self.vel_net(vel_input)
        u, v, w, p = vel_out[:, 0:1], vel_out[:, 1:2], vel_out[:, 2:3], vel_out[:, 3:4]

        return torch.cat([u, v, w, p, phi], dim=-1)

    def forward_triplet(
        self, spatial_coords: torch.Tensor, voltage_triplet: torch.Tensor
    ) -> torch.Tensor:
        """
        三元组格式的前向传播（与 LSTM-Hybrid-PINN 接口一致）

        Args:
            spatial_coords: (batch, 3) - (x, y, z)
            voltage_triplet: (batch, 3) - (V_from, V_to, t_since)，已归一化

        Returns:
            (batch, 5) - (u, v, w, p, phi)
        """
        # 归一化空间坐标
        x_norm = spatial_coords[:, 0:1] / self.Lx
        y_norm = spatial_coords[:, 1:2] / self.Ly
        z_norm = spatial_coords[:, 2:3] / self.Lz

        # 电压三元组已经归一化
        V_from_norm = voltage_triplet[:, 0:1]
        V_to_norm = voltage_triplet[:, 1:2]
        t_norm = voltage_triplet[:, 2:3]

        # Phi 网络输入
        phi_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm], dim=-1)
        phi_raw = self.phi_net(phi_input)
        phi = torch.sigmoid(phi_raw)

        # 速度网络输入
        vel_input = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm, phi], dim=-1)
        vel_out = self.vel_net(vel_input)
        u, v, w, p = vel_out[:, 0:1], vel_out[:, 1:2], vel_out[:, 2:3], vel_out[:, 3:4]

        return torch.cat([u, v, w, p, phi], dim=-1)


# ============================================================================
# 物理损失计算
# ============================================================================


class PhysicsLoss:
    """
    两相流物理损失（重构版）

    设计原则：
    - PhysicsConstraints 作为"唯一物理真相"，负责物理方程计算
    - PhysicsLoss 作为"适配层 + 数值保护壳"，负责：
      1. 调用 PhysicsConstraints.compute_core_residuals() 获取原始残差
      2. 对残差进行 NaN/Inf 清理、裁剪
      3. 应用 log1p 缩放和归一化
      4. 界面加权和损失聚合

    包含：
    - 连续性方程：∇·u = 0
    - VOF 方程：∂φ/∂t + u·∇φ = 0
    - Navier-Stokes 方程
    - 表面张力 CSF 模型：F_st = σκδ_s n
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.rho_oil = PHYSICS["rho_oil"]
        self.rho_polar = PHYSICS["rho_polar"]
        self.mu_oil = PHYSICS["mu_oil"]
        self.mu_polar = PHYSICS["mu_polar"]
        self.sigma = PHYSICS["sigma"]

        # 特征尺度（用于归一化）
        self.L_char = PHYSICS["Lx"]  # 特征长度
        self.U_char = 1e-3  # 特征速度 (m/s)
        self.T_char = self.L_char / self.U_char  # 特征时间

        # 自适应损失归一化: 指数移动平均
        self._loss_ema = {}  # key → EMA value
        self._ema_decay = 0.99  # EMA 衰减因子
        self._ema_warmup_steps = 200  # 预热步数
        self._ema_step = 0

        # 材料参数引用 (用于自适应归一化等)
        try:
            from src.config import get_materials_params

            self.materials_params = get_materials_params()
        except ImportError:
            self.materials_params = {}

        # 实例化 PhysicsConstraints 作为物理方程计算核心
        try:
            from src.config import get_materials_params

            self.physics_constraints = PhysicsConstraints(materials_params=get_materials_params())
        except ImportError:
            self.physics_constraints = PhysicsConstraints()

    def _sanitize_tensor(self, tensor: torch.Tensor, name: str = "") -> torch.Tensor:
        """
        清理张量中的 NaN/Inf 值

        Args:
            tensor: 输入张量
            name: 张量名称（用于日志）

        Returns:
            清理后的张量
        """
        if not torch.isfinite(tensor).all():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"张量 {name} 包含 {nan_count} 个 NaN, {inf_count} 个 Inf，已清零")
            tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
        return tensor

    def compute_all_residuals(
        self, model: nn.Module, points: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        计算所有物理残差（通过 PhysicsConstraints）

        Args:
            model: PINN 模型
            points: 物理点坐标 (batch, 6)

        Returns:
            残差字典
        """
        points = points.clone().requires_grad_(True)

        predictions = model(points)
        residuals = self.physics_constraints.compute_core_residuals(
            x_phys=points, predictions=predictions, model=model
        )

        # 清理所有残差中的 NaN/Inf
        for key in residuals:
            residuals[key] = self._sanitize_tensor(residuals[key], key)

        return residuals

    def explicit_volume_conservation_loss(self, phi_pred: torch.Tensor) -> torch.Tensor:
        """
        显式体积守恒损失

        直接计算体积积分误差: |∫φ dV - V₀| / V₀
        这是对VOF方程的补充，提供直接的体积守恒约束

        Args:
            phi_pred: 预测的φ场 (batch,)

        Returns:
            体积守恒误差（相对误差）
        """
        # 数值积分（使用黎曼和）
        dV = (PHYSICS["Lx"] * PHYSICS["Ly"] * PHYSICS["Lz"]) / phi_pred.numel()
        volume_integral = torch.sum(phi_pred) * dV

        # 初始油墨体积
        V_initial = PHYSICS["Lx"] * PHYSICS["Ly"] * PHYSICS["h_ink"]

        # 相对误差
        volume_error = torch.abs(volume_integral - V_initial) / (V_initial + 1e-10)

        return volume_error

    def compute_total_loss(
        self, model: nn.Module, points: torch.Tensor, weights: dict[str, float] = None
    ) -> dict[str, torch.Tensor]:
        """
        计算总物理损失（推荐入口）

        通过 PhysicsConstraints.compute_core_residuals() 获取残差，
        然后应用归一化、log1p 缩放和加权求和。

        Args:
            model: PINN 模型
            points: 物理点坐标 (batch, 6)
            weights: 各残差项的权重，默认使用均匀权重

        Returns:
            包含各项损失和总损失的字典
        """
        default_weights = {
            "continuity": 2.0,
            "momentum_u": 0.5,
            "momentum_v": 0.5,
            "momentum_w": 0.5,
            "vof": 5.0,
            "electrowetting": 2.0,
            "laplace_pressure": 0.2,
            "sidewall_contact_angle": 5.0,
            "interface_energy": 0.05,
            "volume_conservation": 0.3,
            "explicit_volume": 1.0,
            "sharpening": 0.0,
            "bottom_wetting": 0.5,
            "wall_wetting": 0.1,
            "phase_field_wetting": 10.0,
            "temporal_smoothness": 0.1,
            "dielectric_charge": 0.05,
            "contact_line_dynamics": 0.1,
            "top_boundary": 0.05,
            "pressure_pin": 0.01,
        }
        weights = weights or default_weights

        # 获取所有残差
        residuals = self.compute_all_residuals(model, points)

        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # 预先计算模型输出，供后续特殊Loss使用
        outputs = model(points)
        phi_pred = outputs[:, 4]  # 注意：这里可能未经过 sigmoid，具体取决于模型输出
        # 根据 TwoPhasePINN.forward，输出已经是 sigmoid 后的 [0,1] 值

        # 是否使用自适应归一化 (基于物理量级自动平衡)
        use_adaptive = self.materials_params.get("use_adaptive_loss_scale", False)
        self._ema_step += 1

        # 处理标准物理残差
        for key, residual in residuals.items():
            if residual is None:
                continue

            # 计算 MSE
            mse = torch.mean(residual**2)

            # 应用 log1p 缩放稳定大损失
            scaled_loss = torch.log1p(mse)

            # 自适应归一化: 除以 EMA 使各损失量级统一
            if use_adaptive and torch.isfinite(scaled_loss):
                val = scaled_loss.detach().item()
                if key not in self._loss_ema:
                    self._loss_ema[key] = val
                else:
                    self._loss_ema[key] = (
                        self._ema_decay * self._loss_ema[key] + (1 - self._ema_decay) * val
                    )
                # 预热后使用归一化
                if self._ema_step > self._ema_warmup_steps:
                    ref = self._loss_ema.get(key, 1.0)
                    scaled_loss = scaled_loss / max(ref, 1e-8)

            # 获取权重
            w = weights.get(key, 0.1)

            # 加权损失
            weighted_loss = w * scaled_loss
            losses[key] = weighted_loss
            total_loss = total_loss + weighted_loss

        # 方案B: 添加显式体积守恒损失
        if "explicit_volume" in weights and weights["explicit_volume"] > 0:
            phi_clamped = torch.clamp(phi_pred, 0.0, 1.0)
            vol_loss = self.explicit_volume_conservation_loss(phi_clamped)
            losses["explicit_volume"] = weights["explicit_volume"] * vol_loss
            total_loss = total_loss + losses["explicit_volume"]

        # Allen-Cahn 双阱势正则化: φ²(1-φ)²
        # 与 AC 方程的 f'(φ)=2φ(1-φ)(1-2φ) 一致（同一泛函的导数）
        # 极小值在 φ=0 和 φ=1，不惩罚 φ≈0.5 的扩散界面
        if "sharpening" in weights and weights["sharpening"] > 0:
            sharpening_val = torch.mean(phi_pred**2 * (1.0 - phi_pred) ** 2)
            losses["sharpening"] = weights["sharpening"] * sharpening_val
            total_loss = total_loss + losses["sharpening"]

        losses["total"] = total_loss
        return losses


# ============================================================================
# 数据生成器
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
        self.use_stage1_eta = self.config.get("stage1_eta_from_model", False)

        # 采样策略：uniform（默认）或 physics_based
        sampling_cfg = self.config.get("sampling", {})
        self.sampling_strategy = sampling_cfg.get("strategy", "uniform")
        self.physics_sampler = None
        if self.sampling_strategy == "physics_based":
            try:
                from src.data.physics_sampling import PhysicsBasedSampler

                self.physics_sampler = PhysicsBasedSampler(
                    sampling_cfg,
                    stage1_predictor=self.contact_angle_predictor,
                    stage1_aperture_model=None,  # 延迟初始化
                )
                logger.info("✅ 使用物理采样策略 (PhysicsBasedSampler)")
            except ImportError as e:
                logger.warning(f"PhysicsBasedSampler 不可用，回退到均匀采样: {e}")
                self.sampling_strategy = "uniform"

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
            return self.contact_angle_predictor.predict(
                voltage=V, time=t, V_initial=0.0, t_step=0.0
            )
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
            # 电润湿驱动: 二阶欠阻尼
            omega_0 = 1.0 / tau
            omega_d = omega_0 * np.sqrt(max(0, 1 - zeta**2))
            exp_term = np.exp(-zeta * omega_0 * t)
            damping = zeta / np.sqrt(1 - zeta**2) if zeta < 1 else 1.0
            theta_t = theta_eq + (theta0 - theta_eq) * exp_term * (
                np.cos(omega_d * t) + damping * np.sin(omega_d * t)
            )
        else:
            # 表面张力恢复: 一阶指数 (过阻尼, 无振荡)
            # τ_recovery = τ × 0.4 (恢复快于驱动)
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
        if HAS_APERTURE and self.use_stage1_eta:
            try:
                if not hasattr(self, "_aperture_model"):
                    from src.config import CONFIG_PATH

                    self._aperture_model = EnhancedApertureModel(config_path=str(CONFIG_PATH))
                return self._aperture_model.contact_angle_to_aperture_ratio(theta)
            except Exception as e:
                logger.warning(f"EnhancedApertureModel 调用失败: {e}")

        theta0 = PHYSICS["theta0"]
        theta_min = PHYSICS["theta_min"]
        eta_max = PHYSICS["eta_max"]

        if theta >= theta0:
            eta = 0.0
        elif theta <= theta_min:
            eta = eta_max
        else:
            cos_change = np.cos(np.radians(theta)) - np.cos(np.radians(theta0))
            cos_max_change = np.cos(np.radians(theta_min)) - np.cos(np.radians(theta0))
            eta = eta_max * cos_change / cos_max_change

        return np.clip(eta, 0, eta_max)

    def target_phi_3d(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        V: float,
        V_prev: float = None,
        t_step: float = 0.0,
    ) -> float:
        """
        计算目标 φ 值（支持升压和降压）

        物理模型：
        - 初始：底部 3μm 油墨 (φ=1)，上部 17μm 极性液体 (φ=0)
        - 升压（V > V_prev）：电润湿驱动，油墨被推到边缘/角落
        - 降压（V < V_prev）：表面张力恢复，油墨从边缘/角落铺展回中心

        升压模式：
        - 开口率 < 50%：中心开口模式（油墨环形分布）
        - 开口率 > 50%：四角液滴模式（油墨在角落）

        降压模式：
        - 油墨从当前位置向中心铺展
        - 开口率指数衰减：η(t) = η_at_step × exp(-(t - t_step) / τ_recovery)

        Args:
            x, y, z: 空间坐标 (m)
            t: 时间 (s)
            V: 当前电压 (V)
            V_prev: 之前的电压 (V)，默认等于 V（恒定电压）
            t_step: 电压变化时刻 (s)

        Returns:
            φ ∈ [0, 1]
        """
        h_ink = self.h_ink
        if V_prev is None:
            V_prev = V
        if t_step is None:
            t_step = 0.0
        t_since = max(0.0, t - t_step)
        eta = None
        if HAS_APERTURE:
            try:
                if not hasattr(self, "_aperture_model"):
                    from src.config import CONFIG_PATH

                    self._aperture_model = EnhancedApertureModel(config_path=str(CONFIG_PATH))
                _, eta_stage1 = self._aperture_model.theta_eta_from_triad(V_prev, V, t_since)
                eta = float(eta_stage1)
            except Exception as e:
                logger.warning(f"EnhancedApertureModel 在 target_phi_3d 中失败: {e}")
        if eta is None:
            is_voltage_down = V_prev > V
            if is_voltage_down:
                tau_recovery = PHYSICS["tau_recovery"]
                eta_max = self.get_opening_rate(V_prev, 0.020)
                if t_step is not None and t_step >= 0:
                    t_since_local = max(0, t - t_step)
                else:
                    t_since_local = max(0, t - 0.015)
                eta = eta_max * np.exp(-t_since_local / tau_recovery)
                return self._phi_center_opening_mode(x, y, z, eta, h_ink)
            eta = self.get_opening_rate(V, t)

        # 界面宽度
        interface_width = 1.5e-6  # 1.5 μm

        # 开口率阈值：超过此值从中心开口切换到单角墨滴
        # 物理上 η≈50% 是环形分布→角落汇聚的质变点
        # 取 0.45 让过渡更平缓，减少 PINN 训练难度
        eta_threshold = 0.45

        if eta < 0.01:
            # 无开口：初始状态，油墨均匀铺在底部
            phi_z = 0.5 * (1 - np.tanh((z - h_ink) / (interface_width / 3)))

        elif eta < eta_threshold:
            # ============================================================
            # 中心开口模式 (η < 50%)：油墨环形分布
            # ============================================================
            r = np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2)
            r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
            radial_factor = 0.5 * (1 + np.tanh((r - r_open) / interface_width))
            h_edge = h_ink / max(1.0 - eta, 0.15)

            if r < r_open - interface_width:
                phi_z = 0.0  # 开口区: 极性液体柱
            elif r > r_open + interface_width:
                phi_z = 0.5 * (1 - np.tanh((z - h_edge) / (interface_width / 2)))
            else:
                phi_center = 0.0
                phi_edge = 0.5 * (1 - np.tanh((z - h_edge) / (interface_width / 2)))
                phi_z = phi_center * (1 - radial_factor) + phi_edge * radial_factor

        else:
            # ============================================================
            # 单角墨滴模式 (η ≥ 50%)：油墨汇聚到 (0,0) 角落
            #
            # 物理：底面疏油 + 侧壁亲油 → 油墨沿壁面流动
            #       单 blob 表面能 < 多 blob → 汇聚到单角落
            #       角落三面夹持（底面+两面壁）→ 最深毛细势阱
            #
            # 模型：以角落为中心的 1/4 超椭球 blob
            #       ink_volume = Lx * Ly * h_ink（守恒）
            #       等效半径 r_blob = sqrt(4 * V_ink / (π * h_edge))
            # ============================================================
            corner_x, corner_y = 0.0, 0.0  # 固定 (0,0) 角落

            # 油墨堆高（体积守恒，可超过围堰形成凸面）
            h_edge = h_ink / max(1.0 - eta, 0.15)

            # blob 等效半径：1/4 圆柱近似
            ink_volume = self.Lx * self.Ly * h_ink
            r_blob = np.sqrt(4.0 * ink_volume / (np.pi * h_edge))

            # 到角落的距离
            r_c = np.sqrt((x - corner_x) ** 2 + (y - corner_y) ** 2)

            # 径向分布：角落处 φ=1，远离角落 φ=0
            radial = 0.5 * (1 - np.tanh((r_c - r_blob) / interface_width))

            # z 分布：堆高 h_edge 以下 φ=1，以上 φ=0
            phi_z = radial * 0.5 * (1 - np.tanh((z - h_edge) / (interface_width / 2)))

            # 不截断 z：油墨可堆高到围堰以上（凸面），由 PINN 物理约束决定最终形貌

        return np.clip(phi_z, 0, 1)

    def _phi_center_opening_mode(
        self, x: float, y: float, z: float, eta: float, h_ink: float
    ) -> float:
        """
        中心开口模式的 φ 分布（用于升压和降压）

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
        interface_width = 1.5e-6  # 1.5 μm (方案B: 减小界面宽度提高精度)

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
        # 修复: 限制最大开口率以保持体积守恒
        max_eta = PHYSICS["eta_max"]
        eta = min(eta, max_eta)
        r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
        ink_area = self.Lx * self.Ly - np.pi * r_open**2
        h_ink_edge = self.Lx * self.Ly * h_ink / max(ink_area, 1e-12)
        # 移除了: min(h_ink_edge, self.Lz * 0.8) - 破坏体积守恒

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
        """根据开口率采样空间点"""
        eta_threshold = 0.50

        if eta < eta_threshold and np.random.rand() < 0.4 and eta > 0.01:
            # 中心开口模式：在界面附近采样
            r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
            r = r_open + np.random.randn() * 10e-6
            r = max(0, min(r, self.r_max))
            theta_angle = np.random.rand() * 2 * np.pi
            x = self.cx + r * np.cos(theta_angle)
            y = self.cy + r * np.sin(theta_angle)
            x = np.clip(x, 0, self.Lx)
            y = np.clip(y, 0, self.Ly)
        elif eta >= eta_threshold and np.random.rand() < 0.4:
            # 四角/单角模式：在角落附近采样
            corners = [(0, 0), (self.Lx, 0), (0, self.Ly), (self.Lx, self.Ly)]
            cx, cy = corners[np.random.randint(4)]
            r = np.abs(np.random.randn()) * 30e-6
            theta_angle = np.random.rand() * np.pi / 2
            x = cx + r * np.cos(theta_angle) * (1 if cx == 0 else -1)
            y = cy + r * np.sin(theta_angle) * (1 if cy == 0 else -1)
            x = np.clip(x, 0, self.Lx)
            y = np.clip(y, 0, self.Ly)
        else:
            # 均匀采样
            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly

        # z 方向：在油墨层附近加密
        if np.random.rand() < 0.5:
            z = np.random.rand() * self.h_ink * 3
        else:
            z = np.random.rand() * self.h_ink * 2

        return x, y, z

    def generate_all_data(self) -> dict[str, torch.Tensor]:
        """
        生成训练数据 - 支持升压和降压

        数据格式: (x, y, z, V_from, V_to, t_since) - 三元组格式
        φ = 1: 油墨
        φ = 0: 极性液体（透明）

        三元组语义：
        - V_from: 跳变前电压
        - V_to: 跳变后电压（当前电压）
        - t_since: 跳变后经过的时间
        """
        data_cfg = self.config.get("data", DEFAULT_CONFIG["data"])

        logger.info("生成训练数据（连续采样模式）...")
        logger.info("  数据格式: (x, y, z, V_from, V_to, t_since)")

        # 辅助函数：连续时间采样（偏重早期动态）
        def sample_continuous_times(n_samples):
            """
            连续时间采样 (Beta 分布)

            使用 Beta(0.5, 1.0) 分布，自然地在早期 (t=0) 附近采样更多点，
            同时平滑覆盖整个时间域，避免人为的分段点 (如 15ms) 造成的密度突变。
            """
            # Beta(0.5, 1.0) 类似于 1/sqrt(x)，在 0 处概率密度高，向右递减
            # 这非常适合模拟暂态过程（变化集中在早期）
            t_samples = np.random.beta(0.5, 1.0, n_samples) * self.t_max
            return t_samples

        # ============================================================
        # 1. 界面数据（核心训练数据）
        # ============================================================
        n_interface = data_cfg.get("n_interface", 100000)
        interface_points = []
        interface_targets = []

        use_physics_sampler = (
            self.sampling_strategy == "physics_based" and self.physics_sampler is not None
        )

        if use_physics_sampler:
            sampler = self.physics_sampler
            oil_thickness = self.h_ink
            # 更新 aperture_model 引用（延迟初始化）
            if sampler._stage1_aperture is None and hasattr(self, "_aperture_model"):
                sampler._stage1_aperture = self._aperture_model

            # 确保 stage1_aperture 可用
            if sampler._stage1_aperture is None:
                try:
                    from src.config import CONFIG_PATH
                    from src.models.aperture_model import EnhancedApertureModel

                    sampler._stage1_aperture = EnhancedApertureModel(config_path=str(CONFIG_PATH))
                except Exception:
                    pass

            # 1.1 稳态 (40%)
            n_steady = int(n_interface * 0.4)
            V_steady = sampler.sample_voltage_physics_based(n_steady, oil_thickness=oil_thickness)
            for V in V_steady:
                t = sampler.sample_time_adaptive(1, V, V)[0]
                x, y, z = sampler.sample_spatial_physics_based(1, V, t)
                phi = self.target_phi_3d(x[0], y[0], z[0], t, float(V), V_prev=float(V))
                interface_points.append([x[0], y[0], z[0], float(V), float(V), t])
                interface_targets.append(phi)

            # 1.2 升压 (30%)
            n_up = int(n_interface * 0.3)
            V_up = sampler.sample_voltage_physics_based(n_up, oil_thickness=oil_thickness)
            for V in V_up:
                if float(V) < 1.0:
                    continue
                t = sampler.sample_time_adaptive(1, float(V), 0.0)[0]
                x, y, z = sampler.sample_spatial_physics_based(1, float(V), t)
                phi = self.target_phi_3d(x[0], y[0], z[0], t, float(V), V_prev=0.0)
                interface_points.append([x[0], y[0], z[0], 0.0, float(V), t])
                interface_targets.append(phi)

            # 1.3 降压 (30%)
            n_down = n_interface - len(interface_points)
            if n_down > 0:
                V_from = sampler.sample_voltage_physics_based(n_down, oil_thickness=oil_thickness)
                for Vf in V_from:
                    if float(Vf) < 1.0:
                        continue
                    t = sampler.sample_time_adaptive(1, 0.0, float(Vf))[0]
                    x, y, z = sampler.sample_spatial_physics_based(1, 0.0, t)
                    phi = self.target_phi_3d(x[0], y[0], z[0], t, 0.0, V_prev=float(Vf))
                    interface_points.append([x[0], y[0], z[0], float(Vf), 0.0, t])
                    interface_targets.append(phi)
        else:
            # 原有均匀采样逻辑
            # 1.1 稳态数据 (40%) - V_from = V_to
            # 连续电压采样 [0, 30]
            n_steady = int(n_interface * 0.4)
            V_steady = np.random.uniform(0, 30.0, n_steady)
            t_steady = sample_continuous_times(n_steady)

            for V, t in zip(V_steady, t_steady, strict=False):
                eta = self.get_opening_rate(V, t)
                x, y, z = self._sample_point_by_eta(eta)
                phi = self.target_phi_3d(x, y, z, t, V, V_prev=V)
                interface_points.append([x, y, z, V, V, t])
                interface_targets.append(phi)

            # 1.2 升压响应 (30%) - 0 -> V
            n_up = int(n_interface * 0.3)
            V_up = np.random.uniform(1.0, 30.0, n_up)
            t_up = sample_continuous_times(n_up)

            for V, t in zip(V_up, t_up, strict=False):
                eta = self.get_opening_rate(V, t)
                x, y, z = self._sample_point_by_eta(eta)
                phi = self.target_phi_3d(x, y, z, t, V, V_prev=0.0)
                interface_points.append([x, y, z, 0.0, V, t])
                interface_targets.append(phi)

            # 1.3 降压响应 (30%) - V -> 0
            n_down = n_interface - n_steady - n_up
            V_down = np.random.uniform(1.0, 30.0, n_down)
            t_down = sample_continuous_times(n_down)
            tau_recovery = PHYSICS["tau_recovery"]

            for V, t in zip(V_down, t_down, strict=False):
                # 降压逻辑：从 V 的稳态开始指数衰减
                eta_at_fall = self.get_opening_rate(V, 0.020)
                eta = eta_at_fall * np.exp(-t / tau_recovery)
                x, y, z = self._sample_point_by_eta(eta)
                phi = self.target_phi_3d(x, y, z, t, 0.0, V_prev=V)
                interface_points.append([x, y, z, V, 0.0, t])
                interface_targets.append(phi)

        logger.info(f"  界面数据点: {len(interface_points)}")

        # ============================================================
        # 1b. 底面数据点 (z=0) — 关键: 直接告诉模型底面开口率
        # ============================================================
        n_bottom = data_cfg.get("n_bottom", 10000)
        bottom_added = 0
        for _ in range(n_bottom):
            x = np.random.uniform(0, self.Lx)
            y = np.random.uniform(0, self.Ly)
            z = 0.0
            V_from = np.random.uniform(0, 30.0)
            V_to = np.random.uniform(0, 30.0)
            # 偏向稳态 (80%稳态, 20%瞬态)
            if np.random.random() < 0.8:
                V_to = V_from  # 稳态
            t = sample_continuous_times(1)[0]
            phi = self.target_phi_3d(x, y, z, t, V_to, V_prev=V_from)
            interface_points.append([x, y, z, V_from, V_to, t])
            interface_targets.append(phi)
            bottom_added += 1

        logger.info(f"  底面数据点 (z=0): +{bottom_added}")

        # ============================================================
        # 2. 初始条件：t=0 时油墨均匀铺在底部 3μm
        # ============================================================
        n_ic = data_cfg.get("n_initial", 10000)
        ic_points, ic_values = [], []

        # 连续电压采样
        V_ic = np.random.uniform(0, 30.0, n_ic)

        for V in V_ic:
            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly
            z = np.random.rand() * self.Lz  # 全域采样 [0, Lz]

            interface_width = 1e-6
            phi = 0.5 * (1 - np.tanh((z - self.h_ink) / interface_width))
            phi = np.clip(phi, 0, 1)

            # 三元组格式: (x, y, z, V_from, V_to, t_since=0)
            ic_points.append([x, y, z, V, V, 0.0])
            ic_values.append([0.0, 0.0, 0.0, 0.0, phi])

        logger.info(f"  初始条件点: {len(ic_points)}")

        # ============================================================
        # 3. 壁面边界条件
        # ============================================================
        n_bc = data_cfg.get("n_boundary", 10000)
        bc_points, bc_values = [], []

        # 混合场景采样：40% 稳态, 30% 升压, 30% 降压
        n_bc_steady = int(n_bc * 0.4)
        n_bc_up = int(n_bc * 0.3)
        n_bc_down = n_bc - n_bc_steady - n_bc_up

        # 3.1 稳态 BC
        V_steady = np.random.uniform(0, 30.0, n_bc_steady)
        t_steady = sample_continuous_times(n_bc_steady)

        # 3.2 升压 BC
        V_up = np.random.uniform(1.0, 30.0, n_bc_up)
        t_up = sample_continuous_times(n_bc_up)

        # 3.3 降压 BC
        V_down = np.random.uniform(1.0, 30.0, n_bc_down)
        t_down = sample_continuous_times(n_bc_down)

        # 合并所有场景
        scenarios = []
        for i in range(n_bc_steady):
            scenarios.append((V_steady[i], V_steady[i], t_steady[i]))
        for i in range(n_bc_up):
            scenarios.append((0.0, V_up[i], t_up[i]))
        for i in range(n_bc_down):
            scenarios.append((V_down[i], 0.0, t_down[i]))

        for V_from, V_to, t in scenarios:
            # 随机选择一个壁面
            boundary_type = np.random.randint(0, 4)  # 0:x0, 1:xL, 2:y0, 3:yL

            if boundary_type == 0:  # x=0
                x, y = 0, np.random.rand() * self.Ly
            elif boundary_type == 1:  # x=Lx
                x, y = self.Lx, np.random.rand() * self.Ly
            elif boundary_type == 2:  # y=0
                x, y = np.random.rand() * self.Lx, 0
            else:  # y=Ly
                x, y = np.random.rand() * self.Lx, self.Ly

            z = np.random.rand() * self.h_ink * 2

            # 计算目标 phi
            phi = self.target_phi_3d(x, y, z, t, V_to, V_prev=V_from)

            # 三元组格式: (x, y, z, V_from, V_to, t_since)
            bc_points.append([x, y, z, V_from, V_to, t])
            bc_values.append([0.0, 0.0, 0.0, 0.0, phi])

        logger.info(f"  壁面边界条件点: {len(bc_points)}")

        # ============================================================
        # 4. 域内配点
        # ============================================================
        n_domain = data_cfg.get("n_domain", 20000)
        domain_points = []

        # 混合场景采样
        n_dom_steady = int(n_domain * 0.4)
        n_dom_up = int(n_domain * 0.3)
        n_dom_down = n_domain - n_dom_steady - n_dom_up

        # 生成域内点场景
        dom_scenarios = []

        # 稳态
        V_s = np.random.uniform(0, 30.0, n_dom_steady)
        t_s = sample_continuous_times(n_dom_steady)
        for i in range(n_dom_steady):
            dom_scenarios.append((V_s[i], V_s[i], t_s[i]))

        # 升压
        V_u = np.random.uniform(1.0, 30.0, n_dom_up)
        t_u = sample_continuous_times(n_dom_up)
        for i in range(n_dom_up):
            dom_scenarios.append((0.0, V_u[i], t_u[i]))

        # 降压
        V_d = np.random.uniform(1.0, 30.0, n_dom_down)
        t_d = sample_continuous_times(n_dom_down)
        for i in range(n_dom_down):
            dom_scenarios.append((V_d[i], 0.0, t_d[i]))

        for V_from, V_to, t in dom_scenarios:
            x = np.random.uniform(0, self.Lx)
            y = np.random.uniform(0, self.Ly)
            z = np.random.uniform(0, self.Lz)
            domain_points.append([x, y, z, V_from, V_to, t])

        logger.info(f"  域内配点: {len(domain_points)}")

        # ============================================================
        # 5. 接触角边界条件
        # ============================================================
        n_contact = data_cfg.get("n_interface", 100000) // 2
        contact_points = []
        contact_theta = []

        # 混合场景采样
        n_con_steady = int(n_contact * 0.4)
        n_con_up = int(n_contact * 0.3)
        n_con_down = n_contact - n_con_steady - n_con_up

        con_scenarios = []

        # 稳态
        V_cs = np.random.uniform(0, 30.0, n_con_steady)
        t_cs = sample_continuous_times(n_con_steady)
        for i in range(n_con_steady):
            con_scenarios.append((V_cs[i], V_cs[i], t_cs[i]))

        # 升压
        V_cu = np.random.uniform(1.0, 30.0, n_con_up)
        t_cu = sample_continuous_times(n_con_up)
        for i in range(n_con_up):
            con_scenarios.append((0.0, V_cu[i], t_cu[i]))

        # 降压
        V_cd = np.random.uniform(1.0, 30.0, n_con_down)
        t_cd = sample_continuous_times(n_con_down)
        for i in range(n_con_down):
            con_scenarios.append((V_cd[i], 0.0, t_cd[i]))

        for V_from, V_to, t in con_scenarios:
            # 接触角计算逻辑
            if V_from > V_to:  # 降压过程 (Step Down)
                # 模拟接触角弛豫：从 theta(V_from) 恢复到 theta0
                # 1. 计算初始高压下的稳态接触角
                # 使用 t=1.0 (足够长的时间) 获取稳态值
                theta_high = self.get_contact_angle(V_from, 1.0)
                theta_low = self.theta0

                # 2. 一阶指数弛豫, τ_recovery = τ_drive(V_from) × 0.4
                tau_recovery_factor = PHYSICS.get("tau_recovery_factor", 0.4)
                if V_from <= PHYSICS.get("V_threshold", 5.0):
                    tau_drive = PHYSICS.get("tau_onset", 0.0075)
                elif V_from <= 2 * PHYSICS.get("V_threshold", 5.0):
                    tau_drive = PHYSICS["tau"]
                else:
                    tau_drive = PHYSICS.get("tau_saturation", 0.003)
                tau_rec = tau_drive * tau_recovery_factor
                decay = np.exp(-t / tau_rec)
                theta = theta_low + (theta_high - theta_low) * decay

            else:  # 升压或稳态
                theta = self.get_contact_angle(V_to, t)

            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly
            contact_points.append([x, y, 0.0, V_from, V_to, t])
            contact_theta.append(theta)

        logger.info(f"  接触角边界条件点: {len(contact_points)}")

        # ============================================================
        # 5b. 三部接触线过采样 (z=0, φ≈0.5)
        # ============================================================
        # 接触线是 z=0 与 φ=0.5 等值面的交线 (1D 曲线在 3D 空间中)
        # 均匀采样命中概率 ≈ 0 → 需显式过采样
        n_cl = data_cfg.get("n_interface", 100000) // 4  # 额外 25% 接触线点
        cl_added = 0
        for _ in range(n_cl):
            V = np.random.uniform(0, 30.0)
            V_prev = 0.0 if np.random.rand() > 0.5 else V
            t = sample_continuous_times(1)[0]

            # 从 Stage1 估算开口半径 → 接触线位置
            eta = self.get_opening_rate(V, t)
            r_open = np.sqrt(max(0, eta) * self.Lx * self.Ly / np.pi)
            r_open = min(r_open, self.Lx / 2 * 0.95)

            # 接触线附近采样 (r ≈ r_open, z=0)
            angle = np.random.uniform(0, 2 * np.pi)
            r = r_open + np.random.normal(0, 5e-6)  # 高斯扩展 (±5μm)
            r = np.clip(r, 1e-6, self.Lx / 2 * 0.98)
            x_cl = self.cx + r * np.cos(angle)
            y_cl = self.cy + r * np.sin(angle)
            x_cl = np.clip(x_cl, 0, self.Lx)
            y_cl = np.clip(y_cl, 0, self.Ly)

            theta_cl = self.get_contact_angle(V, t)
            contact_points.append([x_cl, y_cl, 0.0, V_prev, V, t])
            contact_theta.append(theta_cl)
            cl_added += 1

        logger.info(f"  三部接触线过采样: +{cl_added} 点 (z=0, r≈r_open)")

        # ============================================================
        # 5c. 突破时刻过采样 (t≈0, 升压场景)
        # ============================================================
        n_breakthrough = data_cfg.get("n_interface", 100000) // 8
        bt_added = 0
        for _ in range(n_breakthrough):
            V = np.random.uniform(5.0, 30.0)  # 突破只发生在 V>V_T
            # 时间集中在 0-2ms (突破窗口)
            t = np.random.exponential(scale=0.0005)  # τ=0.5ms 指数分布
            t = np.clip(t, 0, 0.005)
            # z=0 底面 + r 靠近开口半径
            eta = self.get_opening_rate(V, 0.02)  # 用稳态 η 估计初始 r
            r_open = np.sqrt(max(0.01, eta) * self.Lx * self.Ly / np.pi)
            r_open = min(r_open, self.Lx / 2 * 0.95)
            angle = np.random.uniform(0, 2 * np.pi)
            r = r_open * (0.5 + 0.5 * np.random.random())  # r ∈ [0.5r_open, r_open]
            x_bt = np.clip(self.cx + r * np.cos(angle), 0, self.Lx)
            y_bt = np.clip(self.cy + r * np.sin(angle), 0, self.Ly)

            theta_bt = self.get_contact_angle(V, t)
            contact_points.append([x_bt, y_bt, 0.0, 0.0, V, t])
            contact_theta.append(theta_bt)
            bt_added += 1

        logger.info(f"  突破时刻过采样: +{bt_added} 点 (t∈[0,5ms], 升压)")

        return {
            # 界面数据（核心训练数据）
            "interface_points": torch.tensor(
                np.array(interface_points), dtype=torch.float32, device=self.device
            ),
            "interface_targets": torch.tensor(
                np.array(interface_targets), dtype=torch.float32, device=self.device
            ),
            # 接触角边界条件
            "contact_points": torch.tensor(
                np.array(contact_points), dtype=torch.float32, device=self.device
            ),
            "contact_theta": torch.tensor(
                np.array(contact_theta), dtype=torch.float32, device=self.device
            ),
            # 初始条件
            "ic_points": torch.tensor(np.array(ic_points), dtype=torch.float32, device=self.device),
            "ic_values": torch.tensor(np.array(ic_values), dtype=torch.float32, device=self.device),
            # 壁面边界条件
            "bc_points": torch.tensor(np.array(bc_points), dtype=torch.float32, device=self.device),
            "bc_values": torch.tensor(np.array(bc_values), dtype=torch.float32, device=self.device),
            # 域内配点
            "domain_points": torch.tensor(
                np.array(domain_points), dtype=torch.float32, device=self.device
            ),
        }


# ============================================================================
# 训练器
# ============================================================================


class Trainer:
    """两相流 PINN 训练器"""

    def __init__(self, config: dict[str, Any] = None, resume_path: str | None = None):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 模型
        self.model = TwoPhasePINN(self.config).to(self.device)
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        self.data_generator = DataGenerator(self.config, self.device)
        self.physics_loss = PhysicsLoss(self.device)
        self.physics_constraints = PhysicsConstraints()

        # 统一相场润湿模式: 用自然BC替代碎片化壁面约束
        if self.config.get("physics", {}).get("use_unified_wetting", False):
            self.physics_loss.materials_params["use_unified_wetting"] = True
            self.physics_loss.physics_constraints.materials_params["use_unified_wetting"] = True
            logger.info("✅ 启用统一相场润湿BC (use_unified_wetting=True)")

        # Stage 1 模型 - 使用统一配置路径
        if HAS_APERTURE:
            from src.config import CONFIG_PATH

            self.stage1_model = EnhancedApertureModel(config_path=str(CONFIG_PATH))
        else:
            self.stage1_model = None
        self.enable_stage1_tutor = self.config.get(
            "stage1_eta_tutor", True
        )  # 默认启用 Stage1 tutor
        # 训练配置
        training_cfg = self.config.get("training", {})
        self.epochs = training_cfg.get("epochs", 30000)
        self.batch_size = training_cfg.get("batch_size", 4096)
        self.lr = training_cfg.get("learning_rate", 5e-4)

        # 渐进式训练阶段
        self.stage1_epochs = training_cfg.get("stage1_epochs", 5000)
        self.stage2_epochs = training_cfg.get("stage2_epochs", 15000)
        # S3 物理约束平滑过渡跨度 (epoch), 默认5000适配60000epoch训练
        self.s3_smooth_span = float(training_cfg.get("s3_smooth_span", 5000))

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1500, min_lr=1e-6
        )

        # 学习率预热
        self.warmup_epochs = training_cfg.get("warmup_epochs", 500)
        self.warmup_start_lr = self.lr * 0.01  # 从 1% 开始预热

        # 训练历史
        self.history = {
            "epoch": [],
            "loss": [],
            "interface": [],
            "physics": [],
            "lr": [],
            "pinn_continuity": [],
            "pinn_vof": [],
            "pinn_momentum_u": [],
            "pinn_momentum_v": [],
            "pinn_momentum_w": [],
            "pinn_electrowetting": [],
            "pinn_temporal_smoothness": [],
            "pinn_laplace_pressure": [],
            "pinn_sidewall_contact_angle": [],
            "pinn_interface_energy": [],
            "pinn_wall_wetting": [],
            "pinn_bottom_wetting": [],
            "pinn_phase_field_wetting": [],
            "pinn_dielectric_charge": [],
            "pinn_contact_line_dynamics": [],
            "pinn_top_boundary": [],
            "contact_angle": [],
            "volume": [],
        }
        self.best_loss = float("inf")
        self.best_physics_loss = float("inf")  # 新增：物理损失最佳值
        self.patience_counter = 0
        self.early_stop_patience = training_cfg.get("early_stop_patience", 5000)

        # 阶段3才开始记录 best_loss（物理约束完整后）
        self.best_loss_recording_start = self.stage2_epochs  # 从阶段3开始记录
        self.start_epoch = 0

        # 输出目录与续训
        if resume_path is not None and os.path.exists(resume_path):
            logger.info(f"正在从 {resume_path} 恢复训练...")
            ckpt = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.best_loss = ckpt.get("best_loss", self.best_loss)
            self.best_physics_loss = ckpt.get("best_physics_loss", self.best_physics_loss)
            self.history = ckpt.get("history", self.history)
            self.start_epoch = ckpt.get("epoch", -1) + 1
            # 续训时，总 epochs = 起始 epoch + 新增 epochs
            self.epochs = self.start_epoch + self.epochs
            self.output_dir = os.path.dirname(resume_path)
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"恢复成功，从 Epoch {self.start_epoch} 继续，将训练到 Epoch {self.epochs}")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # [2026-01-16] 使用统一路径管理，输出到 outputs/train/
            from src.config.paths import get_output_dir

            self.output_dir = str(get_output_dir(f"train/pinn_{timestamp}"))

        # 添加文件日志处理器到 root logger, 捕获所有模块日志
        log_file = os.path.join(self.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(logging.INFO)
        logger.info(f"日志文件: {log_file}")

        # 初始化 TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "runs"))

        # 预先保存一份配置
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """
        保存训练检查点
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "best_physics_loss": self.best_physics_loss,
            "history": self.history,
            "config": self.config,
        }

        # 保存最新检查点
        latest_path = os.path.join(self.output_dir, "latest_model.pth")
        torch.save(checkpoint, latest_path)

        # 如果是最佳模型，则额外保存
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            # 同时也保留一个带 epoch 的副本，防止覆盖
            if epoch % 1000 == 0:
                torch.save(
                    checkpoint,
                    os.path.join(self.output_dir, f"best_model_epoch_{epoch}.pth"),
                )

        # 如果是最终模型
        if is_final:
            final_path = os.path.join(self.output_dir, "final_model.pth")
            torch.save(checkpoint, final_path)

    def _plot_curves(self, epoch: int = None):
        """
        绘制并保存训练曲线 — 所有 loss 在一张图上
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            ep = self.history["epoch"]
            h = self.history

            # (history_key, label, alpha, lw, color, linestyle)
            curve_specs = [
                ("loss", "Total", 1.0, 2.0, "black", "-"),
                ("interface", "IF(data)", 0.7, 0.8, "C0", "-"),
                ("contact_angle", "θ(contact)", 0.5, 0.6, "C1", "--"),
                ("volume", "Vol", 0.5, 0.6, "C2", "--"),
                ("pinn_continuity", "C(∇·u=0)", 0.6, 0.7, "C3", "-"),
                ("pinn_vof", "AC(vof)", 0.5, 0.6, "C4", ":"),
                ("pinn_momentum_u", "NS_u", 0.3, 0.4, "C5", ":"),
                ("pinn_momentum_w", "NS_w", 0.3, 0.4, "C6", ":"),
                ("pinn_electrowetting", "EW", 0.5, 0.6, "C7", "-."),
                ("pinn_laplace_pressure", "LP", 0.4, 0.5, "C8", ":"),
                ("pinn_sidewall_contact_angle", "SW", 0.4, 0.5, "C9", "-."),
                ("pinn_interface_energy", "IE", 0.3, 0.5, "C10", ":"),
                ("pinn_wall_wetting", "WW", 0.3, 0.4, "C11", "-."),
                ("pinn_bottom_wetting", "BW", 0.3, 0.4, "C12", "--"),
                ("pinn_phase_field_wetting", "PFW", 0.3, 0.5, "C13", "-"),
                ("pinn_contact_line_dynamics", "CLD", 0.3, 0.4, "C14", "-."),
                ("pinn_dielectric_charge", "DC", 0.2, 0.4, "C15", ":"),
                ("pinn_top_boundary", "TB", 0.2, 0.4, "C16", "-."),
                ("pinn_temporal_smoothness", "TS", 0.1, 0.3, "C17", ":"),
            ]
            for key, label, alpha, lw, color, ls in curve_specs:
                ax.semilogy(ep, h[key], label=label, alpha=alpha, lw=lw, color=color, linestyle=ls)

            # 阶段分界线
            if len(self.history["epoch"]) > 0:
                last_ep = self.history["epoch"][-1]
                if self.stage1_epochs < last_ep:
                    ax.axvline(
                        x=self.stage1_epochs,
                        color="r",
                        linestyle="--",
                        alpha=0.5,
                        lw=1.0,
                        label="S1→S2",
                    )
                if self.stage2_epochs < last_ep:
                    ax.axvline(
                        x=self.stage2_epochs,
                        color="g",
                        linestyle="--",
                        alpha=0.5,
                        lw=1.0,
                        label="S2→S3",
                    )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log scale)")
            ax.legend(loc="upper right", ncol=3, fontsize=6.5, framealpha=0.8)
            ax.grid(True, alpha=0.15)

            title = "Training Loss (all constraints)"
            if epoch is not None:
                title += f"  (Epoch {epoch})"
            ax.set_title(title, fontsize=12)

            filename = (
                "training_curve.png" if epoch is None else f"training_curve_epoch_{epoch}.png"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "training_curve.png"), dpi=150)
            milestones = {0, 5000, 10000, 20000, 30000, 40000, 50000, 60000}
            if epoch is not None and epoch in milestones:
                plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"保存训练曲线失败: {e}")

    def get_physics_weights(self, epoch: int) -> dict[str, float]:
        """
        根据训练阶段返回物理损失权重（平滑渐进）

        改进：使用平滑过渡而不是阶跃变化
        """
        physics_cfg = self.config.get("physics", {})

        if epoch < self.stage1_epochs:
            # 阶段1：纯数据学习，所有物理约束为零
            return {
                "continuity": 0.0,
                "vof": 0.0,
                "ns": 0.0,
                "electrowetting": 0.0,
                "laplace_pressure": 0.0,
                "sidewall_contact_angle": 0.0,
                "interface_energy": 0.0,
                "wall_wetting": 0.0,
                "bottom_wetting": 0.0,
                "phase_field_wetting": 0.0,
                "temporal_smoothness": 0.0,
                "dielectric_charge": 0.0,
                "contact_line_dynamics": 0.0,
                "top_boundary": 0.0,
                "explicit_volume": 0.0,
                "sharpening": 0.0,
            }

        if epoch < self.stage2_epochs:
            # 阶段2：平滑引入连续性和VOF，并以很小权重预热 NS
            progress = (epoch - self.stage1_epochs) / (self.stage2_epochs - self.stage1_epochs)
            smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))  # S形曲线

            ns_base = physics_cfg.get("ns_weight", 0.01)
            ns_scale = 0.05  # 阶段2仅使用 5% 的 NS 权重

            return {
                "continuity": physics_cfg.get("continuity_weight", 0.1) * smooth_factor * 0.1,
                "vof": physics_cfg.get("vof_weight", 0.1) * smooth_factor * 0.1,
                "ns": ns_base * smooth_factor * ns_scale,
                "electrowetting": 0.0,
                "laplace_pressure": 0.0,
                "sidewall_contact_angle": 0.0,
                "interface_energy": 0.0,
                "wall_wetting": 0.0,
                "bottom_wetting": 0.0,
                "phase_field_wetting": 0.0,
                "temporal_smoothness": 0.0,
                "dielectric_charge": 0.0,
                "contact_line_dynamics": 0.0,
                "top_boundary": 0.0,
                "explicit_volume": physics_cfg.get("explicit_volume_weight", 0.0)
                * smooth_factor
                * 0.1,
                "sharpening": physics_cfg.get("sharpening_weight", 0.0) * smooth_factor * 0.1,
            }
        # 阶段3：完整物理约束（平滑增加，跨度可配置）
        progress = min(1.0, (epoch - self.stage2_epochs) / max(1.0, self.s3_smooth_span))
        smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))

        return {
            "continuity": physics_cfg.get("continuity_weight", 0.1) * (0.1 + 0.9 * smooth_factor),
            "vof": physics_cfg.get("vof_weight", 0.1) * (0.1 + 0.9 * smooth_factor),
            "ns": physics_cfg.get("ns_weight", 0.01) * smooth_factor,
            "electrowetting": physics_cfg.get("electrowetting_weight", 2.0) * smooth_factor,
            "laplace_pressure": physics_cfg.get("laplace_pressure_weight", 0.2) * smooth_factor,
            "sidewall_contact_angle": physics_cfg.get("sidewall_contact_angle", 5.0)
            * smooth_factor,
            "interface_energy": physics_cfg.get("interface_energy_weight", 0.05) * smooth_factor,
            "wall_wetting": physics_cfg.get("wall_wetting_weight", 0.1) * smooth_factor,
            "bottom_wetting": physics_cfg.get("bottom_wetting_weight", 0.5) * smooth_factor,
            "phase_field_wetting": physics_cfg.get("phase_field_wetting_weight", 10.0)
            * smooth_factor,
            "temporal_smoothness": physics_cfg.get("temporal_smoothness_weight", 0.1)
            * smooth_factor,
            "dielectric_charge": physics_cfg.get("dielectric_charge_weight", 0.05) * smooth_factor,
            "contact_line_dynamics": physics_cfg.get("contact_line_dynamics_weight", 0.1)
            * smooth_factor,
            "top_boundary": physics_cfg.get("top_boundary_weight", 0.05) * smooth_factor,
            "explicit_volume": physics_cfg.get("explicit_volume_weight", 0.0) * smooth_factor,
            "sharpening": physics_cfg.get("sharpening_weight", 0.0) * smooth_factor,
        }

    def get_stage1_weight_factor(self, epoch: int) -> float:
        """
        获取 Stage1 约束的退火权重（控制 Interface loss）。

        S1 (0 ~ stage1_epochs):              factor = 1.0    (interface_weight = 500)
        S2 (stage1_epochs ~ stage2_epochs):   factor 退火 1.0 → 0.1  (500 → 50)
        S3 (stage2_epochs ~ end):             factor 退火 0.1 → 0.05 (50 → 25)

        S3 后期保留适度数据锚定（min_factor=0.05），防止物理约束权重过大时
        梯度在接触线高阶项处爆炸（NaN 崩溃，见 pinn_20260528_192413 复盘）。
        """
        training_cfg = (
            self.config.get("training", {})
            if isinstance(self.config.get("training", {}), dict)
            else {}
        )

        # S2 退火: stage1_epochs → stage2_epochs, factor 1.0 → 0.1
        s2_start = self.stage1_epochs
        s2_end = self.stage2_epochs
        s2_span = max(1, s2_end - s2_start)

        # S3 退火: stage2_epochs → stage2_epochs + anneal_span, factor 0.1 → 0.05
        s3_start = self.stage2_epochs
        s3_anneal_epochs = int(training_cfg.get("s3_anneal_span", 15000))
        s3_end = s3_start + s3_anneal_epochs
        s3_span = max(1, s3_end - s3_start)

        # 可配置最低因子，防止数据锚定完全消失
        min_factor = float(training_cfg.get("stage1_tutor_min_factor", 0.05))
        min_factor = float(np.clip(min_factor, 0.01, 0.5))

        if epoch < s2_start:
            return 1.0
        if epoch < s2_end:
            # S2: 余弦退火 1.0 → 0.1
            progress = (epoch - s2_start) / s2_span
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
        if epoch < s3_end:
            # S3: 余弦退火 0.1 → min_factor
            progress = (epoch - s3_start) / s3_span
            return min_factor + (0.1 - min_factor) * 0.5 * (1 + np.cos(np.pi * progress))
        return min_factor

    def compute_losses(self, data: dict[str, torch.Tensor], epoch: int) -> dict[str, torch.Tensor]:
        """
        计算所有损失 - 模块化重构

        φ = 1: 油墨, φ = 0: 极性液体
        """
        losses = {}
        physics_cfg = self.config.get("physics", {})
        physics_weights = self.get_physics_weights(epoch)
        stage1_factor = self.get_stage1_weight_factor(epoch)  # Stage1 约束退火

        # 记录当前 epoch，供各子损失方法读取（用于 S3 权重退火）
        self._current_epoch = epoch

        current_stage = (
            1 if epoch < self.stage1_epochs else (2 if epoch < self.stage2_epochs else 3)
        )

        # 1. 界面数据拟合损失 (核心)
        losses["interface"], data_fit_loss = self._compute_data_loss(
            data, physics_cfg, stage1_factor
        )

        # 2. 接触角边界条件损失 + Phase调度
        losses["contact_angle"] = self._compute_contact_angle_loss(data)
        s3_start = self.stage2_epochs
        s3_prog = max(0.0, (epoch - s3_start) / max(self.epochs - s3_start, 1))
        if s3_prog < 0.2:
            contact_mult = 0.5
        elif s3_prog < 0.6:
            contact_mult = 1.0
        else:
            contact_mult = 1.0 + 1.5 * min(1.0, (s3_prog - 0.6) / 0.1)  # max 2.5
        losses["contact_angle"] *= contact_mult

        # 3. 初始条件 & 壁面边界条件损失
        ib_losses = self._compute_initial_boundary_loss(data, physics_cfg)
        losses.update(ib_losses)

        # 4. 早期时间 & 零电压约束
        ez_losses = self._compute_early_zero_voltage_loss()
        losses.update(ez_losses)

        # 5. 单调性 & 电压响应约束
        mr_losses = self._compute_monotonicity_response_loss()
        losses.update(mr_losses)

        # 6. 开口率相关约束 (eta_ceiling, eta_stage1_early, eta_stage1_late, eta_monotonic)
        eta_losses = self._compute_eta_constraints_loss(stage1_factor)
        losses.update(eta_losses)

        # 6.5+6.6 φ 场空间分布 + 几何一致性约束（已合并为批量前向）
        spatial_loss, geom_loss = self._compute_phi_spatial_loss(stage1_factor)
        losses["phi_spatial"] = spatial_loss
        losses["phi_geometry"] = geom_loss

        # 7. 体积守恒约束
        losses["volume_conservation"] = self._compute_volume_conservation_loss(stage1_factor)

        # 7.5 η 匹配 loss（Two-Stage 核心锚定，不退火）
        losses["eta_matching"] = self.compute_eta_matching_loss(epoch) * 500.0

        # 7.6 φ target3D loss（从 η 构造 φ target，遵守 target3D）
        # S1/S2 高权重，S3 渐降但不归零
        if epoch < self.stage1_epochs:
            phi_target3d_weight = 300.0
        elif epoch < self.stage2_epochs:
            phi_target3d_weight = 200.0
        else:
            s3_progress = (epoch - self.stage2_epochs) / max(self.epochs - self.stage2_epochs, 1)
            phi_target3d_weight = max(50.0, 200.0 * (1.0 - s3_progress * 0.75))
        losses["phi_target3d"] = self.compute_phi_target3d_loss(epoch) * phi_target3d_weight

        # 8. 物理方程损失
        if any(w > 0 for w in physics_weights.values()):
            phys_losses = self._compute_physics_equation_loss(
                physics_weights, data_fit_loss, epoch, current_stage, data=data
            )
            losses.update(phys_losses)

        # 9. 降压过程：开口率恢复约束（阶段2及以后）
        if epoch >= self.stage1_epochs:
            losses["eta_recovery"] = self.eta_recovery_constraint_loss()

        # 10. 连续性约束：升压→降压转换时刻必须连续（方案B新增）
        if epoch >= self.stage1_epochs:
            losses["continuity_transition"] = self._compute_continuity_transition_loss()

        # 计算总损失
        total_loss = torch.tensor(0.0, device=self.device)
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                # 检查损失是否有效
                if torch.isfinite(val):
                    total_loss = total_loss + val
                else:
                    logger.warning(f"检测到无效损失 {name}: {val.item()}, 将其忽略")

        losses["total"] = total_loss
        return losses

    def _compute_data_loss(
        self, data: dict[str, torch.Tensor], physics_cfg: dict, stage1_factor: float
    ):
        """1. 界面数据拟合损失"""
        idx = torch.randperm(len(data["interface_points"]))[: self.batch_size]
        interface_pts = data["interface_points"][idx]
        interface_tgt = data["interface_targets"][idx]

        phi_pred = self.model(interface_pts)[:, 4]
        interface_loss = F.mse_loss(phi_pred, interface_tgt)

        base_weight = physics_cfg.get("interface_weight", 500.0)
        return interface_loss * base_weight * stage1_factor, interface_loss

    def _compute_contact_angle_loss(self, data: dict[str, torch.Tensor]):
        """2. 接触角边界条件损失 + 接触线滑移动态

        静态约束: ∂φ/∂z = |∇φ|cos(θ_eq)  @ z=0
        滑移约束: v_slip = α × (cos θ_eq - cos θ_local)  @ 接触线
        """
        idx = torch.randperm(len(data["contact_points"]))[: self.batch_size // 2]
        contact_pts = data["contact_points"][idx].clone().requires_grad_(True)
        phi = self.model(contact_pts)[:, 4]

        grad_phi = torch.autograd.grad(
            phi.sum(), contact_pts, create_graph=True, retain_graph=True
        )[0]
        dphi_dz = grad_phi[:, 2]
        grad_mag = torch.sqrt(torch.sum(grad_phi[:, :3] ** 2, dim=1) + 1e-10)

        theta_rad = data["contact_theta"][idx] * np.pi / 180.0
        target_cos = torch.cos(theta_rad)
        actual_cos = dphi_dz / grad_mag

        # 界面加权 (只作用于 φ≈0.5)
        interface_weight = torch.exp(-100 * (phi - 0.5) ** 2)

        # 静态接触角损失
        loss_static = torch.mean((actual_cos - target_cos) ** 2 * interface_weight) * 100.0

        # 接触线滑移损失 (CAH 动态)
        # v_slip = φ_t / (|∇φ| + ε), Δcos = cos_θ_eq - cos_θ_local
        phi_t = grad_phi[:, 5] if grad_phi.shape[1] >= 6 else torch.zeros_like(phi)
        v_slip = phi_t / (grad_mag + 1e-10)
        delta_cos = target_cos - actual_cos
        alpha_slip = 0.1  # 滑移系数

        loss_slip = torch.mean((v_slip - alpha_slip * delta_cos) ** 2 * interface_weight) * 10.0

        return loss_static + loss_slip

    def _compute_initial_boundary_loss(self, data: dict[str, torch.Tensor], physics_cfg: dict):
        """3. 初始条件 & 壁面边界条件损失"""
        res = {}
        # IC
        idx_ic = torch.randperm(len(data["ic_points"]))[: self.batch_size // 4]
        pred_ic = self.model(data["ic_points"][idx_ic])
        ic_phi_loss = F.mse_loss(pred_ic[:, 4:5], data["ic_values"][idx_ic][:, 4:5])
        ic_vel_loss = F.mse_loss(pred_ic[:, :4], data["ic_values"][idx_ic][:, :4])
        res["ic"] = ic_phi_loss * physics_cfg.get("ic_weight", 100.0) + ic_vel_loss * 50.0

        # BC
        idx_bc = torch.randperm(len(data["bc_points"]))[: self.batch_size // 4]
        pred_bc = self.model(data["bc_points"][idx_bc])
        res["bc"] = F.mse_loss(pred_bc[:, :3], data["bc_values"][idx_bc][:, :3]) * physics_cfg.get(
            "bc_weight", 50.0
        )
        res["bc"] += F.mse_loss(pred_bc[:, 4:5], data["bc_values"][idx_bc][:, 4:5]) * 80.0
        return res

    def _compute_early_zero_voltage_loss(self):
        """4. 早期时间 & 零电压约束 — 批量前向版本（3次→1次）。"""
        res = {}
        n = self.batch_size // 4
        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
        h_ink = PHYSICS["h_ink"]
        delta_ic = getattr(self, "hard_ic_width", 1e-6)

        x = torch.rand(n, device=self.device) * Lx
        y = torch.rand(n, device=self.device) * Ly
        z = torch.rand(n, device=self.device) * Lz
        xyz = torch.stack([x, y, z], dim=1)  # (n, 3)

        # 3 组场景参数
        t_early = torch.rand(n, device=self.device) * 0.002
        v_early = torch.rand(n, device=self.device) * 30.0
        t_0v = torch.rand(n, device=self.device) * PHYSICS["t_max"]
        v_low = torch.rand(n, device=self.device) * 5.0
        t_low = 0.005 + torch.rand(n, device=self.device) * 0.015

        # 拼接为 (3n, 6)，一次前向
        pts_all = torch.cat(
            [
                torch.cat(
                    [
                        xyz,
                        torch.zeros(n, 1, device=self.device),
                        v_early.unsqueeze(1),
                        t_early.unsqueeze(1),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        xyz,
                        torch.zeros(n, 1, device=self.device),
                        torch.zeros(n, 1, device=self.device),
                        t_0v.unsqueeze(1),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        xyz,
                        torch.zeros(n, 1, device=self.device),
                        v_low.unsqueeze(1),
                        t_low.unsqueeze(1),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )  # (3n, 6)

        phi_all = self.model(pts_all)[:, 4]  # (3n,)
        phi_early = phi_all[:n]
        phi_0v = phi_all[n : 2 * n]
        phi_low = phi_all[2 * n :]

        # early_time target: phi_IC(z)
        phi_ic_target = 0.5 * (1.0 + torch.tanh((h_ink - z) / delta_ic))

        # S3 退火：zero_voltage 和 low_voltage 在 S3 后半段退火到 0.3（不完全归零）
        # 保留弱锚定防止梯度在接触线高阶项处爆炸（NaN 崩溃复盘，20260529）
        _epoch = getattr(self, "_current_epoch", 0)
        if _epoch > self.stage2_epochs:
            _s3_total = max(1, self.epochs - self.stage2_epochs)
            _s3_prog = (_epoch - self.stage2_epochs) / _s3_total
            # S3 前半段权重=1.0，后半段线性退火到 0.3
            if _s3_prog < 0.5:
                _volt_anneal = 1.0
            else:
                _volt_anneal = max(0.1, 1.0 - 0.9 * ((_s3_prog - 0.5) / 0.5))
        else:
            _volt_anneal = 1.0

        res["early_time"] = F.mse_loss(phi_early, phi_ic_target) * 100.0
        res["zero_voltage"] = F.mse_loss(phi_0v, phi_ic_target) * 100.0 * _volt_anneal
        res["low_voltage"] = F.mse_loss(phi_low, phi_ic_target) * 100.0 * _volt_anneal
        return res

    def _compute_monotonicity_response_loss(self):
        """5. 单调性 & 电压响应约束 — 批量前向版本（4次→1次）。"""
        res = {}
        n = self.batch_size // 4
        cx, cy = PHYSICS["Lx"] / 2, PHYSICS["Ly"] / 2
        x = cx + (torch.rand(n, device=self.device) - 0.5) * PHYSICS["Lx"] * 0.4
        y = cy + (torch.rand(n, device=self.device) - 0.5) * PHYSICS["Ly"] * 0.4
        z = torch.rand(n, device=self.device) * PHYSICS["h_ink"]
        xyz = torch.stack([x, y, z], dim=1)  # (n, 3)

        # Monotonicity
        v_mono = 10.0 + torch.rand(n, device=self.device) * 20.0
        t1 = torch.rand(n, device=self.device) * 0.01
        t2 = torch.rand(n, device=self.device) * 0.01 + 0.002

        # Voltage Response
        v1 = 5.0 + torch.rand(n, device=self.device) * 10.0
        v2 = torch.clamp(v1 + 5.0 + torch.rand(n, device=self.device) * 10.0, max=30.0)
        t_v = torch.full((n,), 0.015, device=self.device)

        # 拼接 4 组 (n, 6) → (4n, 6)，一次前向
        pts_all = torch.cat(
            [
                torch.cat(
                    [
                        xyz,
                        torch.zeros(n, 1, device=self.device),
                        v_mono.unsqueeze(1),
                        t1.unsqueeze(1),
                    ],
                    dim=1,
                ),  # p1
                torch.cat(
                    [
                        xyz,
                        torch.zeros(n, 1, device=self.device),
                        v_mono.unsqueeze(1),
                        t2.unsqueeze(1),
                    ],
                    dim=1,
                ),  # p2
                torch.cat(
                    [xyz, torch.zeros(n, 1, device=self.device), v1.unsqueeze(1), t_v.unsqueeze(1)],
                    dim=1,
                ),  # pv1
                torch.cat(
                    [xyz, torch.zeros(n, 1, device=self.device), v2.unsqueeze(1), t_v.unsqueeze(1)],
                    dim=1,
                ),  # pv2
            ],
            dim=0,
        )  # (4n, 6)

        phi_all = self.model(pts_all)[:, 4]  # (4n,)
        phi1, phi2 = phi_all[:n], phi_all[n : 2 * n]
        phiv1, phiv2 = phi_all[2 * n : 3 * n], phi_all[3 * n :]

        res["monotonicity"] = torch.mean(F.relu(phi2 - phi1 + 0.02) ** 2) * 50.0
        res["voltage_response"] = torch.mean(F.relu(phiv2 - phiv1 + 0.02) ** 2) * 100.0
        return res

    def _compute_eta_constraints_loss(self, stage1_factor: float):
        """6. 开口率相关约束 - 批量 aperture 前向版本。

        eta_ceiling + eta_monotonic 收集所有 triplet 后一次性批量计算，
        将 ~90 次独立前向降为常数次（2次：ceiling 一次 + monotonic 一次）。
        """
        res = {}
        eta_max = PHYSICS["eta_max"]

        # ===== Ceiling: 收集所有 triplet → 一次批量前向 =====
        v_cap = [10.0, 15.0, 20.0, 25.0, 30.0]
        t_cap = [0.010, 0.012, 0.014, 0.016, 0.018, 0.020]
        ceil_triplets = [(0.0, float(v), float(t)) for v in v_cap for t in t_cap]
        if ceil_triplets:
            eta_all = self.compute_aperture_ratio_batch(ceil_triplets, n_grid=25)
            loss_cap = torch.mean(F.relu(eta_all - eta_max) ** 2)
            res["eta_ceiling"] = loss_cap * 200.0
        else:
            res["eta_ceiling"] = torch.tensor(0.0, device=self.device)

        # ===== Monotonic: 收集所有 triplet → 一次批量前向 =====
        v_smooth = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        t_on = sorted(np.random.uniform(0.001, 0.020, 5).tolist())
        t_off = sorted(np.random.uniform(0.001, 0.040, 5).tolist())

        # 升压 triplet: (0.0, v, t)
        on_triplets = [(0.0, float(v), float(t)) for v in v_smooth for t in t_on]
        # 降压 triplet: (v, 0.0, t)
        off_triplets = [(float(v), 0.0, float(t)) for v in v_smooth for t in t_off]
        all_mono_triplets = on_triplets + off_triplets

        if all_mono_triplets:
            eta_mono = self.compute_aperture_ratio_batch(all_mono_triplets, n_grid=20)
            n_on = len(v_smooth) * len(t_on)
            eta_on = eta_mono[:n_on].view(len(v_smooth), len(t_on))
            eta_off = eta_mono[n_on:].view(len(v_smooth), len(t_off))

            loss_m = torch.tensor(0.0, device=self.device)
            count_m = 0
            # 升压单调递增: eta(t_i) <= eta(t_{i+1})
            for i in range(len(t_on) - 1):
                loss_m = loss_m + torch.mean(F.relu(eta_on[:, i] - eta_on[:, i + 1] + 0.005) ** 2)
                count_m += 1
            # 降压单调递减: eta(t_i) >= eta(t_{i+1})
            for i in range(len(t_off) - 1):
                loss_m = loss_m + torch.mean(F.relu(eta_off[:, i + 1] - eta_off[:, i] + 0.005) ** 2)
                count_m += 1
            res["eta_monotonic"] = loss_m * 50.0 / count_m if count_m > 0 else loss_m
        else:
            res["eta_monotonic"] = torch.tensor(0.0, device=self.device)

        return res

    def _compute_phi_spatial_loss(self, stage1_factor: float):
        """6.5+6.6 φ 场空间分布 + 几何一致性约束（批量前向版本）。

        合并原 _compute_phi_spatial_loss + _compute_phi_geometry_loss：
        - 收集所有 test_cases 的查询点，按 case 拼接为一次大前向
        - 从 20 次独立前向降为 2 次（spatial 一次 + geometry 一次）
        - geometry 部分的空间采样也合并为一次前向

        Returns:
            (spatial_loss, geometry_loss) 两个独立标量
        """
        if self.stage1_model is None:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        Lx, Ly, Lz, h_ink = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"], PHYSICS["h_ink"]
        cx, cy = Lx / 2, Ly / 2

        # ===== spatial: 中心+边缘两点约束（批量前向） =====
        n_spatial_pts = 20
        spatial_test_cases = [
            (0.0, 10.0, float(np.random.uniform(0.010, 0.020))),
            (0.0, 20.0, float(np.random.uniform(0.010, 0.020))),
            (0.0, 30.0, float(np.random.uniform(0.005, 0.012))),
            (0.0, 30.0, float(np.random.uniform(0.018, 0.030))),
        ]

        # 收集所有 spatial 查询点
        spatial_pts_center = []  # (N_total, 6) for center
        spatial_pts_edge = []

        spatial_loss = torch.tensor(0.0, device=self.device)
        spatial_count = 0
        spatial_offset_center = 0
        spatial_offset_edge = 0
        center_valid_ranges = []  # (start, n_pts) for valid cases
        edge_valid_ranges = []

        for V_from, V_to, t_since in spatial_test_cases:
            _, eta_s1 = self.stage1_model.theta_eta_from_triad(V_from, V_to, t_since)
            if eta_s1 < 0.05:
                center_valid_ranges.append(None)
                edge_valid_ranges.append(None)
                continue

            r_open = np.sqrt(eta_s1 * Lx * Ly / np.pi)
            Vf, Vt, ts = float(V_from), float(V_to), float(t_since)

            # 中心
            r_center = r_open * 0.3
            if r_center > 5e-6:
                theta_a = torch.rand(n_spatial_pts, device=self.device) * 2 * np.pi
                rr = torch.rand(n_spatial_pts, device=self.device) * r_center
                xc = cx + rr * torch.cos(theta_a)
                yc = cy + rr * torch.sin(theta_a)
                zc = torch.zeros(n_spatial_pts, device=self.device)
                pts = torch.stack(
                    [
                        xc,
                        yc,
                        zc,
                        torch.full((n_spatial_pts,), Vf, device=self.device),
                        torch.full((n_spatial_pts,), Vt, device=self.device),
                        torch.full((n_spatial_pts,), ts, device=self.device),
                    ],
                    dim=1,
                )
                spatial_pts_center.append(pts)
                center_valid_ranges.append((spatial_offset_center, n_spatial_pts))
                spatial_offset_center += n_spatial_pts
            else:
                center_valid_ranges.append(None)

            # 边缘
            r_edge_min = min(r_open * 1.5, min(Lx, Ly) / 2 * 0.8)
            r_edge_max = min(Lx, Ly) / 2 * 0.95
            if r_edge_max > r_edge_min:
                theta_a = torch.rand(n_spatial_pts, device=self.device) * 2 * np.pi
                rr = r_edge_min + torch.rand(n_spatial_pts, device=self.device) * (
                    r_edge_max - r_edge_min
                )
                xe = cx + rr * torch.cos(theta_a)
                ye = cy + rr * torch.sin(theta_a)
                xe = torch.clamp(xe, 1e-6, Lx - 1e-6)
                ye = torch.clamp(ye, 1e-6, Ly - 1e-6)
                ze = torch.zeros(n_spatial_pts, device=self.device)
                pts = torch.stack(
                    [
                        xe,
                        ye,
                        ze,
                        torch.full((n_spatial_pts,), Vf, device=self.device),
                        torch.full((n_spatial_pts,), Vt, device=self.device),
                        torch.full((n_spatial_pts,), ts, device=self.device),
                    ],
                    dim=1,
                )
                spatial_pts_edge.append(pts)
                edge_valid_ranges.append((spatial_offset_edge, n_spatial_pts))
                spatial_offset_edge += n_spatial_pts
            else:
                edge_valid_ranges.append(None)

        # 一次前向计算所有 spatial 中心点
        if spatial_pts_center:
            all_center_pts = torch.cat(spatial_pts_center, dim=0)
            all_phi_center = self.model(all_center_pts)[:, 4]
            for i, rng in enumerate(center_valid_ranges):
                if rng is not None:
                    s, n = rng
                    phi_c = all_phi_center[s : s + n]
                    spatial_loss = spatial_loss + torch.mean(torch.relu(phi_c - 0.3) ** 2)
                    spatial_count += 1

        # 一次前向计算所有 spatial 边缘点
        if spatial_pts_edge:
            all_edge_pts = torch.cat(spatial_pts_edge, dim=0)
            all_phi_edge = self.model(all_edge_pts)[:, 4]
            for i, rng in enumerate(edge_valid_ranges):
                if rng is not None:
                    s, n = rng
                    phi_e = all_phi_edge[s : s + n]
                    spatial_loss = spatial_loss + torch.mean(torch.relu(0.7 - phi_e) ** 2)
                    spatial_count += 1

        if spatial_count == 0:
            spatial_loss = torch.tensor(0.0, device=self.device)
        else:
            spatial_loss = spatial_loss * 500.0 * stage1_factor / spatial_count

        # geom_loss 已在下方计算，此处不返回

        # ===== geometry: 三点约束（中心+边缘油墨+边缘极性，批量前向） =====
        v0 = Lx * Ly * h_ink
        n_geom_pts = 256
        geom_test_cases = [
            (0.0, 20.0, 0.015),
            (0.0, 20.0, 0.020),
            (0.0, 30.0, 0.010),
            (0.0, 30.0, 0.020),
        ]

        geom_pts_all = []
        geom_valid_ranges = []  # list of (start, h_edge, r_open_val, margin) or None

        geom_offset = 0
        for V_from, V_to, t_since in geom_test_cases:
            _, eta_s1 = self.stage1_model.theta_eta_from_triad(V_from, V_to, t_since)
            eta_s1 = float(np.clip(eta_s1, 0.0, 0.95))
            if eta_s1 < 0.02:
                geom_valid_ranges.append(None)
                continue

            r_open = float(np.sqrt(eta_s1 * Lx * Ly / np.pi))
            r_open = min(r_open, 0.95 * min(cx, cy))
            ink_area = max(Lx * Ly - np.pi * r_open**2, 1e-12)
            h_edge = float(min(v0 / ink_area, Lz))
            margin = 3e-6
            r_center = max(0.0, r_open - margin)
            r_edge_min = min(r_open + margin, 0.9 * min(cx, cy))
            r_edge_max_g = 0.95 * min(cx, cy)
            Vf2, Vt2, ts2 = float(V_from), float(V_to), float(t_since)

            start = geom_offset
            # z 范围: center 用 [0, min(h_ink, Lz)], ink 用 [0, min(h_edge*0.8, Lz)], polar 用 [h_edge*1.1, Lz]

            pts_list = []
            if r_center > 5e-6:
                theta = torch.rand(n_geom_pts, device=self.device) * 2 * np.pi
                rr = torch.sqrt(torch.rand(n_geom_pts, device=self.device)) * r_center
                pts_list.append(
                    torch.stack(
                        [
                            cx + rr * torch.cos(theta),
                            cy + rr * torch.sin(theta),
                            torch.rand(n_geom_pts, device=self.device) * min(h_ink, Lz),
                            torch.full((n_geom_pts,), Vf2, device=self.device),
                            torch.full((n_geom_pts,), Vt2, device=self.device),
                            torch.full((n_geom_pts,), ts2, device=self.device),
                        ],
                        dim=1,
                    )
                )

            if r_edge_max_g > r_edge_min and h_edge > 1e-7:
                theta = torch.rand(n_geom_pts, device=self.device) * 2 * np.pi
                rr = r_edge_min + torch.rand(n_geom_pts, device=self.device) * (
                    r_edge_max_g - r_edge_min
                )
                xe = torch.clamp(cx + rr * torch.cos(theta), 1e-9, Lx - 1e-9)
                ye = torch.clamp(cy + rr * torch.sin(theta), 1e-9, Ly - 1e-9)
                # 油墨层内
                pts_list.append(
                    torch.stack(
                        [
                            xe,
                            ye,
                            torch.rand(n_geom_pts, device=self.device) * min(h_edge * 0.8, Lz),
                            torch.full((n_geom_pts,), Vf2, device=self.device),
                            torch.full((n_geom_pts,), Vt2, device=self.device),
                            torch.full((n_geom_pts,), ts2, device=self.device),
                        ],
                        dim=1,
                    )
                )
                z_top_min = min(h_edge * 1.1, Lz * 0.99)
                if z_top_min < Lz:
                    pts_list.append(
                        torch.stack(
                            [
                                xe,
                                ye,
                                z_top_min
                                + torch.rand(n_geom_pts, device=self.device) * (Lz - z_top_min),
                                torch.full((n_geom_pts,), Vf2, device=self.device),
                                torch.full((n_geom_pts,), Vt2, device=self.device),
                                torch.full((n_geom_pts,), ts2, device=self.device),
                            ],
                            dim=1,
                        )
                    )

            n_pts_case = len(pts_list) * n_geom_pts
            if pts_list:
                geom_pts_all.append(torch.cat(pts_list, dim=0))
                geom_valid_ranges.append((start, h_edge, r_open, margin, len(pts_list)))
                geom_offset += n_pts_case
            else:
                geom_valid_ranges.append(None)

        if geom_pts_all:
            all_geom_pts = torch.cat(geom_pts_all, dim=0)
            all_phi_geom = self.model(all_geom_pts)[:, 4]
            geom_loss = torch.tensor(0.0, device=self.device)
            geom_count = 0
            for rng in geom_valid_ranges:
                if rng is None:
                    continue
                start, h_edge, r_open, margin, n_regions = rng
                s = start
                # 中心区域: φ < 0.2
                if n_regions >= 1:
                    phi_c = all_phi_geom[s : s + n_geom_pts]
                    geom_loss = geom_loss + torch.mean(torch.relu(phi_c - 0.2) ** 2)
                    s += n_geom_pts
                # 边缘油墨区: φ > 0.8
                if n_regions >= 2:
                    phi_ink = all_phi_geom[s : s + n_geom_pts]
                    geom_loss = geom_loss + torch.mean(torch.relu(0.8 - phi_ink) ** 2)
                    s += n_geom_pts
                # 边缘极性区: φ < 0.2
                if n_regions >= 3:
                    phi_polar = all_phi_geom[s : s + n_geom_pts]
                    geom_loss = geom_loss + torch.mean(torch.relu(phi_polar - 0.2) ** 2)
                geom_count += n_regions
            if geom_count > 0:
                keep_factor = 0.2 + 0.8 * stage1_factor
                geom_loss = geom_loss * 200.0 * keep_factor / geom_count
            else:
                geom_loss = torch.tensor(0.0, device=self.device)
        else:
            geom_loss = torch.tensor(0.0, device=self.device)

        return spatial_loss, geom_loss

    def _compute_volume_conservation_loss(self, stage1_factor: float):
        """体积守恒约束 — 批量前向版本。

        将 N 个 (V_from, V_to, t_since) 场景的空间点与坐标拼接为一次大前向，
        替代 N 次独立前向，并用 torch.no_grad() 省去梯度构建开销。
        """
        training_cfg = (
            self.config.get("training", {})
            if isinstance(self.config.get("training", {}), dict)
            else {}
        )
        n_vol = int(training_cfg.get("volume_n_vol", 10000))  # 30000→10000
        n_vol = max(2000, n_vol)
        Lx, Ly, Lz, h_ink = (
            PHYSICS["Lx"],
            PHYSICS["Ly"],
            PHYSICS["Lz"],
            PHYSICS["h_ink"],
        )
        v0 = Lx * Ly * h_ink
        v_domain = Lx * Ly * Lz

        # 收集所有测试场景的 triplets
        t_steady = np.random.uniform(0.020, 0.050, 3).tolist()
        t_on = np.random.uniform(0.002, 0.025, 5).tolist()
        t_off = np.random.uniform(0.002, 0.025, 4).tolist()

        tests = []
        for v in [0.0, 10.0, 20.0, 25.0, 30.0]:
            for t in t_steady:
                tests.append((float(v), float(v), float(t)))
        for t in t_on:
            tests.append((0.0, 30.0, float(t)))
            tests.append((0.0, 25.0, float(t)))
        for t in t_off:
            tests.append((30.0, 0.0, float(t)))
            tests.append((25.0, 0.0, float(t)))

        N = len(tests)
        if N == 0:
            return torch.tensor(0.0, device=self.device)

        # 空间坐标只生成一次，每个场景复用
        x = torch.rand(n_vol, device=self.device) * Lx
        y = torch.rand(n_vol, device=self.device) * Ly
        z = torch.rand(n_vol, device=self.device) * Lz

        # 构造 (N * n_vol, 6) 的大 batch，通过 repeat_interleave 复用空间坐标
        xyz = torch.stack([x, y, z], dim=1)  # (n_vol, 3)
        xyz_all = xyz.repeat_interleave(N, dim=0)  # (N*n_vol, 3)

        v_from_all = torch.tensor([t[0] for t in tests], device=self.device).repeat(
            n_vol
        )  # (N*n_vol,)
        v_to_all = torch.tensor([t[1] for t in tests], device=self.device).repeat(n_vol)
        t_since_all = torch.tensor([t[2] for t in tests], device=self.device).repeat(n_vol)

        pts = torch.cat(
            [
                xyz_all,
                v_from_all.unsqueeze(1),
                v_to_all.unsqueeze(1),
                t_since_all.unsqueeze(1),
            ],
            dim=1,
        )  # (N*n_vol, 6)

        with torch.no_grad():
            phi = torch.clamp(self.model(pts)[:, 4], 0.0, 1.0)  # (N*n_vol,)

        # reshape → (n_vol, N) → 按列（每个场景）计算均值
        phi_by_scene = phi.view(N, n_vol).mean(dim=1)  # (N,)
        v_curr = v_domain * phi_by_scene
        rel_errors = (v_curr - v0) / (v0 + 1e-12)
        loss_vol = torch.mean(rel_errors**2)

        base_weight = float(training_cfg.get("volume_base_weight", 1000.0))  # 2000→1000
        stage_weight = 0.2 + (1.0 - float(stage1_factor))

        # S3 退火：S3 前期（油墨运动阶段）弱化体积约束，后期恢复
        # 避免油墨动态变形中体积守恒与电润湿驱动冲突
        _epoch = getattr(self, "_current_epoch", 0)
        if _epoch > self.stage2_epochs:
            _s3_total = max(1, self.epochs - self.stage2_epochs)
            _s3_prog = (_epoch - self.stage2_epochs) / _s3_total
            # epoch=15000→ramp=0, epoch=37500→ramp=0.5, epoch=60000→ramp=1.0
            _vol_ramp = min(1.0, _s3_prog * 2.0)
        else:
            _vol_ramp = 1.0

        return loss_vol * base_weight * stage_weight * _vol_ramp

    def _compute_continuity_transition_loss(self):
        """10. 连续性约束 — 批量前向版本。

        将 N 个 (V, t) 的 rise/fall 空间坐标拼接为两次大前向（rise 一次 + fall 一次），
        替代 N×2 次独立前向，并用 torch.no_grad() 省去梯度构建。
        """
        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]

        n = min(4000, self.batch_size)
        x = torch.rand(n, device=self.device) * Lx
        y = torch.rand(n, device=self.device) * Ly
        n_ink = int(n * 0.6)
        z_ink = torch.rand(n_ink, device=self.device) * PHYSICS["h_ink"] * 2
        z_uniform = torch.rand(n - n_ink, device=self.device) * Lz
        z = torch.cat([z_ink, z_uniform])

        t_rise_samples = np.linspace(0.005, 0.030, 6).tolist()
        V_samples = [10.0, 20.0, 30.0]
        test_cases = [(float(V), float(t)) for V in V_samples for t in t_rise_samples]
        N = len(test_cases)

        if N == 0:
            return torch.tensor(0.0, device=self.device)

        xyz = torch.stack([x, y, z], dim=1)  # (n, 3)

        # 构造所有 rise 场景的点: (R, n, 6) → (R*n, 6)
        # rise: V_from=0, V_to=V_high, t=t_rise_end
        V_high_all = torch.tensor([tc[0] for tc in test_cases], device=self.device)
        t_rise_all = torch.tensor([tc[1] for tc in test_cases], device=self.device)

        xyz_r = xyz.unsqueeze(0).expand(N, -1, -1).reshape(N * n, 3)
        pts_rise = torch.cat(
            [
                xyz_r,
                torch.zeros(N * n, 1, device=self.device),
                V_high_all.unsqueeze(1).expand(-1, n).reshape(-1, 1),
                t_rise_all.unsqueeze(1).expand(-1, n).reshape(-1, 1),
            ],
            dim=1,
        )

        # fall: V_from=V_high, V_to=0, t=0
        xyz_f = xyz.unsqueeze(0).expand(N, -1, -1).reshape(N * n, 3)
        pts_fall = torch.cat(
            [
                xyz_f,
                V_high_all.unsqueeze(1).expand(-1, n).reshape(-1, 1),
                torch.zeros(N * n, 1, device=self.device),
                torch.zeros(N * n, 1, device=self.device),
            ],
            dim=1,
        )

        with torch.no_grad():
            pred_rise_all = self.model(pts_rise)  # (N*n, 5)
            pred_fall_all = self.model(pts_fall)

        # reshape → (N, n, 5)
        pred_r = pred_rise_all.view(N, n, 5)
        pred_f = pred_fall_all.view(N, n, 5)

        loss_phi = torch.mean((pred_r[:, :, 4] - pred_f[:, :, 4]) ** 2, dim=1)  # (N,)
        loss_vel = torch.mean((pred_r[:, :, :3] - pred_f[:, :, :3]) ** 2, dim=(1, 2))  # (N,)
        loss_continuity = torch.mean(loss_phi + 0.1 * loss_vel)

        training_cfg = (
            self.config.get("training", {})
            if isinstance(self.config.get("training", {}), dict)
            else {}
        )
        base_weight = float(training_cfg.get("continuity_transition_weight", 2000.0))
        return loss_continuity * base_weight

    def _compute_physics_equation_loss(
        self,
        weights: dict,
        data_fit_loss: torch.Tensor,
        epoch: int,
        stage: int,
        data: dict[str, torch.Tensor] = None,
    ):
        """
        8. 物理方程损失 - 通过 PhysicsLoss.compute_total_loss 计算

        [2026-05-19] 统一通过 PhysicsLoss.compute_total_loss (→ compute_core_residuals)
        计算所有物理约束，包括 electrowetting, VOF, NS, Laplace, 接触角等。
        """
        res = {}

        # 1. 准备物理点
        # 策略：混合 50% 域内点 (包含界面附近) + 50% 随机点
        n_batch = min(2000, self.batch_size)  # 增加物理点采样数
        pts_list = []

        # (A) 从 DataGenerator 获取的点 (高质量，包含界面加密)
        if data is not None and "domain_points" in data:
            n_domain = len(data["domain_points"])
            idx_dom = torch.randperm(n_domain)[: n_batch // 2]
            pts_list.append(data["domain_points"][idx_dom])

        # (B) 从界面点获取 (VOF 关键区域)
        if data is not None and "interface_points" in data:
            n_interface = len(data["interface_points"])
            idx_int = torch.randperm(n_interface)[: n_batch // 4]
            pts_list.append(data["interface_points"][idx_int])

        # (C) 补充随机点 (确保全域覆盖)
        n_needed = n_batch - sum(len(p) for p in pts_list)
        if n_needed > 0:
            pts_random = torch.stack(
                [
                    torch.rand(n_needed, device=self.device) * PHYSICS["Lx"],
                    torch.rand(n_needed, device=self.device) * PHYSICS["Ly"],
                    torch.rand(n_needed, device=self.device) * PHYSICS["Lz"],
                    torch.rand(n_needed, device=self.device) * 30.0,  # V_from (随机)
                    torch.rand(n_needed, device=self.device) * 30.0,  # V_to
                    torch.rand(n_needed, device=self.device) * PHYSICS["t_max"],  # t_since
                ],
                dim=1,
            )
            pts_list.append(pts_random)

        pts = torch.cat(pts_list, dim=0)

        try:
            # 分阶段权重调度: 相对于S3物理训练窗口
            s3_start = self.stage2_epochs
            s3_end = self.epochs
            s3_progress = max(0.0, (epoch - s3_start) / max(s3_end - s3_start, 1))
            if s3_progress < 0.2:  # S3早期: 拓扑成型
                ac_mult, ns_mult, contact_mult = 8.0, 0.2, 0.5
                ew_penalty_mult = 1.0
                lp_mult = 0.1  # Laplace 压力弱化，避免曲率震荡
            elif s3_progress < 0.6:  # S3中期: NS驱动
                ac_mult, ns_mult, contact_mult = 1.0, 1.0, 1.0
                ew_penalty_mult = 0.5
                lp_mult = 0.5
            else:  # S3后期: 接触线精修
                ramp = min(1.0, (s3_progress - 0.6) / 0.1)
                ac_mult = 1.0 - 0.5 * ramp
                ns_mult = 1.0 + 0.3 * ramp
                contact_mult = 1.0 + 1.5 * ramp
                ew_penalty_mult = max(0.3, 1.0 - ramp)  # 不完全归零，保留弱EW驱动
                lp_mult = 0.3 + 0.2 * ramp  # 0.3→0.5，降低LP权重防止NaN

            phys_weights = {
                "continuity": float(weights.get("continuity", 0.0)) * ns_mult,
                "vof": float(weights.get("vof", 0.0)) * ac_mult,
                "momentum_u": float(weights.get("ns", 0.0)) * ns_mult,
                "momentum_v": float(weights.get("ns", 0.0)) * ns_mult,
                "momentum_w": float(weights.get("ns", 0.0)) * ns_mult,
                "electrowetting": float(weights.get("electrowetting", 2.0)) * ew_penalty_mult,
                "laplace_pressure": float(weights.get("laplace_pressure", 0.2)) * lp_mult,
                "sidewall_contact_angle": float(weights.get("sidewall_contact_angle", 5.0))
                * contact_mult,
                "interface_energy": float(weights.get("interface_energy", 0.05)) * ac_mult,
                "wall_wetting": float(weights.get("wall_wetting", 0.1)) * contact_mult,
                "bottom_wetting": float(weights.get("bottom_wetting", 0.5)) * contact_mult,
                "phase_field_wetting": float(weights.get("phase_field_wetting", 10.0))
                * contact_mult,
                "temporal_smoothness": float(weights.get("temporal_smoothness", 0.1)) * ns_mult,
                "dielectric_charge": float(weights.get("dielectric_charge", 0.05)) * ns_mult,
                "contact_line_dynamics": float(weights.get("contact_line_dynamics", 0.1))
                * contact_mult,
                "top_boundary": float(weights.get("top_boundary", 0.05)) * ns_mult,
                "volume_conservation": float(
                    self.config.get("physics", {}).get("volume_conservation_weight", 0.0)
                ),
                "explicit_volume": float(weights.get("explicit_volume", 0.0)),
                "sharpening": float(weights.get("sharpening", 0.0)),
            }

            # 使用 PhysicsLoss 计算损失
            losses = self.physics_loss.compute_total_loss(self.model, pts, weights=phys_weights)

            pinn_loss = losses.get("total", None)
            if isinstance(pinn_loss, torch.Tensor) and torch.isfinite(pinn_loss) and pinn_loss > 0:
                res["pinn_physics"] = pinn_loss
                for k in [
                    "continuity",
                    "vof",
                    "momentum_u",
                    "momentum_v",
                    "momentum_w",
                    "electrowetting",
                    "temporal_smoothness",
                    "sidewall_contact_angle",
                    "laplace_pressure",
                    "interface_energy",
                    "wall_wetting",
                    "bottom_wetting",
                    "phase_field_wetting",
                    "dielectric_charge",
                    "contact_line_dynamics",
                    "top_boundary",
                ]:
                    if (
                        k in losses
                        and isinstance(losses[k], torch.Tensor)
                        and torch.isfinite(losses[k])
                    ):
                        res[f"pinn_{k}"] = losses[k]

        except Exception as e:
            logger.warning(f"Physics loss failed: {e}")
        return res

    def compute_aperture_ratio(
        self, V_from: float, V_to: float, t_since: float, n_grid: int = 30
    ) -> torch.Tensor:
        """
        计算开口率：z=0 底面上“极性液体覆盖面积”的比例（软统计）

        Args:
            V_from: 跳变前电压 (V)
            V_to: 跳变后电压 (V)
            t_since: 跳变后经过的时间 (s)
            n_grid: 网格分辨率
        """
        if V_from is None:
            V_from = V_to

        return self.compute_aperture_ratio_batch(
            [(float(V_from), float(V_to), float(t_since))], n_grid=n_grid
        )[0]

    def compute_aperture_ratio_batch(self, triplets: list, n_grid: int = 30) -> torch.Tensor:
        """
        批量计算开口率：一次性前向传播，返回每个 triplet 的 eta。

        将 N 个 triplet 的网格点拼接为一次大前向，替代 N 次独立前向，
        消除 eta/spatial/geometry/volume 约束中的重复 model 调用。

        Args:
            triplets: list of (V_from, V_to, t_since)
            n_grid: 网格分辨率（每个维度）

        Returns:
            (N,) tensor，每个元素是对应 triplet 的开口率
        """
        if not triplets:
            return torch.zeros(0, device=self.device)

        N = len(triplets)
        Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]

        x = torch.linspace(0, Lx, n_grid, device=self.device)
        y = torch.linspace(0, Ly, n_grid, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_x = X.flatten()  # (G,)
        grid_y = Y.flatten()
        G = grid_x.shape[0]

        # 构造 (N*G, 6) 的大 batch
        all_points = torch.zeros(N * G, 6, device=self.device)
        for i, (Vf, Vt, ts) in enumerate(triplets):
            s = slice(i * G, (i + 1) * G)
            all_points[s, 0] = grid_x
            all_points[s, 1] = grid_y
            # z=0 已初始化为 0
            all_points[s, 3] = float(Vf)
            all_points[s, 4] = float(Vt)
            all_points[s, 5] = float(ts)

        eval_cfg = (
            self.config.get("eval", {}) if isinstance(self.config.get("eval", {}), dict) else {}
        )
        phi0 = float(eval_cfg.get("aperture_phi0", 0.3))
        eval_eps = max(1e-6, float(eval_cfg.get("aperture_eps", 0.05)))

        with torch.no_grad():
            phi = torch.clamp(self.model(all_points)[:, 4], 0.0, 1.0)  # (N*G,)

        # reshape → (N, G)，对每个 triplet 在 G 个网格点上统计
        phi_grid = phi.view(N, G)
        masks = torch.sigmoid((phi0 - phi_grid) / eval_eps)  # (N, G)
        return masks.mean(dim=1)  # (N,)

    def compute_aperture_ratio_differentiable(
        self, V_from: float, V_to: float, t_since: float, n_grid: int = 20
    ) -> torch.Tensor:
        """
        可微的开口率计算：去掉 torch.no_grad()，让 η 的梯度能反传。

        用于训练 loss（η 匹配），区别于 evaluate 用的 compute_aperture_ratio。

        Args:
            V_from: 跳变前电压 (V)
            V_to: 跳变后电压 (V)
            t_since: 跳变后经过的时间 (s)
            n_grid: 网格分辨率

        Returns:
            标量 tensor，η ∈ [0, 1]，带梯度
        """
        Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]

        x = torch.linspace(0, Lx, n_grid, device=self.device)
        y = torch.linspace(0, Ly, n_grid, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_x = X.flatten()
        grid_y = Y.flatten()
        G = grid_x.shape[0]

        # 构造 (G, 6) 输入：z=0, V_from, V_to, t_since
        points = torch.zeros(G, 6, device=self.device)
        points[:, 0] = grid_x
        points[:, 1] = grid_y
        # z=0 已默认
        points[:, 3] = float(V_from)
        points[:, 4] = float(V_to)
        points[:, 5] = float(t_since)

        eval_cfg = (
            self.config.get("eval", {}) if isinstance(self.config.get("eval", {}), dict) else {}
        )
        phi0 = float(eval_cfg.get("aperture_phi0", 0.3))
        eval_eps = max(1e-6, float(eval_cfg.get("aperture_eps", 0.05)))

        # 关键：不用 torch.no_grad()！
        phi = torch.clamp(self.model(points)[:, 4], 0.0, 1.0)
        masks = torch.sigmoid((phi0 - phi) / eval_eps)  # 软二值化
        return masks.mean()

    def compute_eta_matching_loss(self, epoch: int) -> torch.Tensor:
        """
        η 匹配 loss：让 PINN 的 η(V,t) 追踪 Teacher 的 η(V,t)。

        这是 Two-Stage Design 的核心——Stage 1 解析模型先学开口率，
        Stage 2 PINN 从开口率出发学习油墨形态。

        特点：
        - 使用可微的 compute_aperture_ratio_differentiable
        - **不退火**：η 匹配始终保留，是锚不是辅助项
        - 覆盖多电压多时刻，确保 η(V,t) 全局匹配
        """
        if not HAS_APERTURE:
            return torch.tensor(0.0, device=self.device)

        # 懒初始化 Teacher 模型
        if not hasattr(self, "_aperture_model") or self._aperture_model is None:
            try:
                from src.config import CONFIG_PATH as _CP

                self._aperture_model = EnhancedApertureModel(config_path=str(_CP))
            except Exception as e:
                logger.warning(f"Teacher 模型初始化失败: {e}")
                return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        count = 0

        # 稀疏采样：5 电压 × 3 时间 = 15 次前向（原 54+12=66 次）
        # 每 epoch 随机采一部分，多 epoch 统计覆盖全空间
        import random

        all_triplets = []
        # 升压
        for V in [0.0, 8.0, 15.0, 25.0, 30.0]:
            for t in [0.005, 0.015, 0.035]:
                all_triplets.append((0.0, V, t))
        # 降压
        for V in [15.0, 30.0]:
            for t in [0.010, 0.030]:
                all_triplets.append((V, 0.0, t))

        # 每 epoch 随机选 8 个 triplet（梯度噪声=正则化）
        selected = random.sample(all_triplets, min(8, len(all_triplets)))

        for V_from, V_to, t in selected:
            _, eta_teacher = self._aperture_model.theta_eta_from_triad(V_from, V_to, t)
            eta_teacher = float(eta_teacher)
            eta_pinn = self.compute_aperture_ratio_differentiable(V_from, V_to, t, n_grid=12)
            loss = loss + (eta_pinn - eta_teacher) ** 2
            count += 1

        return loss / max(count, 1)

    def compute_phi_target3d_loss(self, epoch: int) -> torch.Tensor:
        """
        φ target3D loss：从 η 构造 φ target，让 PINN 的 φ 场匹配。

        物理逻辑（用户指定：油膜需要遵守 target3D）：
          1. Teacher 给出 η(V,t)
          2. η → h_oil = h_ink / (1-η)（体积守恒）
          3. 用 target_phi_3d 逻辑构造 φ(x,y,z) target
          4. MSE(φ_PINN, φ_target)

        三个约束唯一确定 φ：
          ① 开口区无油 → η 决定"哪一圈没有油"
          ② 墙壁束缚   → 油不能越过边界
          ③ 体积守恒   → V_oil = h_ink × Lx × Ly = const
        """
        if not HAS_APERTURE:
            return torch.tensor(0.0, device=self.device)

        # 懒初始化 Teacher 模型
        if not hasattr(self, "_aperture_model") or self._aperture_model is None:
            try:
                from src.config import CONFIG_PATH as _CP

                self._aperture_model = EnhancedApertureModel(config_path=str(_CP))
            except Exception as e:
                logger.warning(f"Teacher 模型初始化失败: {e}")
                return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        count = 0

        # 选取典型工况（减少到 4 个，每 epoch 随机选 2-3 个）
        test_cases = [
            (0.0, 0.0, 0.040),  # V=0V 稳态：油膜平铺底部
            (0.0, 15.0, 0.020),  # V=15V：中等开口
            (0.0, 30.0, 0.020),  # V=30V：大开口
            (20.0, 0.0, 0.020),  # 降压
        ]

        # 每 epoch 随机选 2 个工况
        import random

        selected_cases = random.sample(test_cases, min(2, len(test_cases)))

        n_pts_per_case = 25  # 每个工况采样 25 个空间点

        all_points = []
        all_phi_targets = []

        for V_from, V_to, t_since in selected_cases:
            # Teacher η
            _, eta = self._aperture_model.theta_eta_from_triad(V_from, V_to, t_since)
            eta = float(eta)

            # 采样空间点（在油膜区域加密）
            for _ in range(n_pts_per_case):
                x = np.random.uniform(0, self.model.Lx)
                y = np.random.uniform(0, self.model.Ly)
                # z 在油膜层加密
                if np.random.rand() < 0.6:
                    z = np.random.uniform(
                        0, min(self.model.h_ink * 3 / max(1 - eta, 0.15), self.model.Lz)
                    )
                else:
                    z = np.random.uniform(0, self.model.Lz)

                # 用 target_phi_3d 构造 φ target
                phi_target = self.data_generator.target_phi_3d(
                    x, y, z, t_since, V_to, V_prev=V_from, t_step=0.0
                )

                all_points.append([x, y, z, V_from, V_to, t_since])
                all_phi_targets.append(phi_target)

        if not all_points:
            return torch.tensor(0.0, device=self.device)

        points_tensor = torch.tensor(all_points, dtype=torch.float32, device=self.device)
        phi_targets = torch.tensor(all_phi_targets, dtype=torch.float32, device=self.device)

        # PINN 前向（带梯度）
        phi_pinn = self.model(points_tensor)[:, 4]

        loss = F.mse_loss(phi_pinn, phi_targets)
        return loss

    def eta_recovery_constraint_loss(
        self, t_fall: float = 0.015, tau_recovery: float = None, weight: float = 100.0
    ) -> torch.Tensor:
        """
        开口率恢复的软约束（降压后）

        物理：η(t) = η_at_fall × exp(-(t - t_fall) / τ_recovery)

        Args:
            t_fall: 降压时刻 (s)，默认 12ms
            tau_recovery: 恢复时间常数 (s)，默认从 PHYSICS 读取
            weight: 损失权重
        """
        if tau_recovery is None:
            tau_recovery = PHYSICS["tau_recovery"]
        if self.stage1_model is not None and self.enable_stage1_tutor:
            t_max = PHYSICS.get("t_max", 0.10)
            max_t_since = max(0.0, min(t_max - t_fall, 0.05))
            t_since_samples = np.linspace(0.0, max_t_since, 8)
            loss = torch.tensor(0.0, device=self.device)
            for t_since in t_since_samples:
                t_since_val = float(t_since)
                _, eta_stage1 = self.stage1_model.theta_eta_from_triad(30.0, 0.0, t_since_val)
                eta_target = torch.tensor(float(eta_stage1), device=self.device)
                eta_pred = self.compute_aperture_ratio(V_from=30.0, V_to=0.0, t_since=t_since_val)
                loss = loss + (eta_pred - eta_target) ** 2
            return loss * weight / len(t_since_samples)

        eta_at_fall = self.compute_aperture_ratio(V_from=0.0, V_to=30.0, t_since=t_fall)
        t_samples = [t_fall + dt for dt in [0.002, 0.004, 0.006, 0.008]]
        loss = torch.tensor(0.0, device=self.device)
        for t in t_samples:
            t_since = t - t_fall
            eta_target = eta_at_fall * torch.exp(
                torch.tensor(-t_since / tau_recovery, device=self.device)
            )
            eta_pred = self.compute_aperture_ratio(V_from=30.0, V_to=0.0, t_since=t_since)
            loss = loss + (eta_pred - eta_target) ** 2
        return loss * weight / len(t_samples)

    def fine_tune_lbfgs(self, max_iter=5000):
        """L-BFGS 二阶优化微调"""
        if not self.config.get("training", {}).get("use_lbfgs", False):
            return

        logger.info("=" * 60)
        logger.info(f"开始 L-BFGS 微调 (max_iter={max_iter})...")
        logger.info("=" * 60)

        # 重新生成一批高质量数据
        data = self.data_generator.generate_all_data()

        # L-BFGS 优化器
        optimizer_lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=max_iter,
            max_eval=int(max_iter * 1.25),
            history_size=100,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        pbar = tqdm(total=max_iter, desc="L-BFGS Tuning")

        def closure():
            optimizer_lbfgs.zero_grad()
            # 使用当前 epoch (训练结束后的状态)
            losses = self.compute_losses(data, self.epochs)
            loss = losses["total"]

            if torch.isfinite(loss) and loss.requires_grad:
                loss.backward()

                # 记录
                current_loss = loss.item()
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    # 实时保存最佳模型
                    torch.save(
                        {
                            "epoch": self.epochs,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer_lbfgs.state_dict(),
                            "loss": self.best_loss,
                            "config": self.config,
                        },
                        os.path.join(self.output_dir, "best_model_lbfgs.pth"),
                    )

                pbar.set_postfix({"Loss": f"{current_loss:.4e}"})
                pbar.update(1)
            else:
                # loss 非 finite 或无梯度：返回带梯度的零值，避免 L-BFGS crash
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                pbar.update(1)

            return loss

        try:
            optimizer_lbfgs.step(closure)
        except Exception as e:
            logger.error(f"L-BFGS 优化过程出错: {e}")

        pbar.close()
        logger.info(f"L-BFGS 微调完成! 最终最佳 Loss: {self.best_loss:.6e}")

    def train(self):
        """训练主循环"""
        logger.info("=" * 60)
        logger.info("开始两相流 PINN 训练")
        logger.info("=" * 60)

        data = self.data_generator.generate_all_data()

        # 数据重采样间隔 (0=禁用)
        _train_cfg = self.config.get("training", {})
        resample_interval = _train_cfg.get("resample_interval", 5000)

        # 后台重采样线程（避免阻塞训练循环）
        _resample_future = None
        _resample_lock = threading.Lock()
        _resample_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1) if resample_interval > 0 else None
        )

        # 记录网络结构图到 TensorBoard
        try:
            for key, value in data.items():
                if isinstance(value, torch.Tensor) and value.dim() > 1:
                    self.tb_writer.add_graph(self.model, value[:1])
                    break
        except Exception as e:
            logger.warning(f"无法添加网络图到 TensorBoard: {e}")
        start_time = time.time()
        _consecutive_nan = 0  # 连续 NaN 计数器

        for epoch in range(self.start_epoch, self.epochs):
            # 学习率预热
            if epoch < self.warmup_epochs:
                warmup_lr = self.warmup_start_lr + (self.lr - self.warmup_start_lr) * (
                    epoch / self.warmup_epochs
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            # 检查训练阶段
            if epoch == self.stage1_epochs:
                logger.info("\n进入阶段 2：引入连续性和VOF约束")
            elif epoch == self.stage2_epochs:
                logger.info("\n进入阶段 3：完整物理约束")

            # 定期重采样数据（后台线程），防止对固定采样点过拟合
            if resample_interval > 0 and epoch > 0 and epoch % resample_interval == 0:
                if _resample_future is None:
                    # 提交后台重采样任务
                    _resample_future = _resample_executor.submit(
                        self.data_generator.generate_all_data
                    )
                    logger.info(f"Epoch {epoch}: 后台重采样已启动")
                elif _resample_future.done():
                    # 后台任务完成，原子替换数据
                    try:
                        data = _resample_future.result()
                        _resample_future = None
                        logger.info(f"Epoch {epoch}: 后台重采样完成，已更新训练数据")
                    except Exception as e:
                        logger.warning(f"后台重采样失败: {e}")
                        _resample_future = None

            self.model.train()
            self.optimizer.zero_grad()

            losses = self.compute_losses(data, epoch)
            total_loss = losses["total"]

            # 增加稳定性检查：只有在损失有效且大于 0 时才进行反向传播
            if torch.isfinite(total_loss) and total_loss > 0:
                # 保存当前有效状态，用于 NaN 恢复
                self._last_valid_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                total_loss.backward()
                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # 检查 step 后是否产生 NaN 权重
                has_nan = any(torch.isnan(p).any() for p in self.model.parameters())
                if not has_nan:
                    _consecutive_nan = 0  # 正常 step，重置 NaN 计数
                if has_nan:
                    if hasattr(self, "_last_valid_state"):
                        self.model.load_state_dict(self._last_valid_state)
                        logger.warning(
                            f"Epoch {epoch}: step 后检测到 NaN 权重，已回滚到上一个有效状态"
                        )
                    self.optimizer.zero_grad()
            else:
                logger.warning(
                    f"Epoch {epoch}: 检测到无效总损失 {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss}, 跳过本次优化步"
                )
                if hasattr(self, "_last_valid_state"):
                    self.model.load_state_dict(self._last_valid_state)
                    logger.warning(f"Epoch {epoch}: 已回滚模型权重到上一个有效状态")

            # 记录历史 (使用更健壮的获取方式)
            def get_val(d, k):
                return d.get(k, torch.tensor(0.0)).item()

            # TensorBoard 日志记录 (每 100 epoch)
            if epoch % 100 == 0:
                e = epoch
                w = self.tb_writer
                w.add_scalar("Loss/total", total_loss.item(), e)
                for tb_name, loss_key in [
                    ("Loss/interface", "interface"),
                    ("Loss/contact_angle", "contact_angle"),
                    ("Loss/volume_conservation", "volume_conservation"),
                    ("Loss/phi_spatial", "phi_spatial"),
                    ("Loss/phi_geometry", "phi_geometry"),
                    ("Loss/eta_ceiling", "eta_ceiling"),
                    ("Loss/eta_monotonic", "eta_monotonic"),
                    ("Loss/eta_recovery", "eta_recovery"),
                    ("Loss/pinn_physics", "pinn_physics"),
                    ("Loss/continuity", "pinn_continuity"),
                    ("Loss/vof", "pinn_vof"),
                    ("Loss/momentum_u", "pinn_momentum_u"),
                    ("Loss/momentum_v", "pinn_momentum_v"),
                    ("Loss/momentum_w", "pinn_momentum_w"),
                    ("Loss/electrowetting", "pinn_electrowetting"),
                    ("Loss/laplace_pressure", "pinn_laplace_pressure"),
                    ("Loss/sidewall_contact_angle", "pinn_sidewall_contact_angle"),
                    ("Loss/interface_energy", "pinn_interface_energy"),
                    ("Loss/wall_wetting", "pinn_wall_wetting"),
                    ("Loss/bottom_wetting", "pinn_bottom_wetting"),
                    ("Loss/phase_field_wetting", "pinn_phase_field_wetting"),
                    ("Loss/contact_line_dynamics", "pinn_contact_line_dynamics"),
                    ("Loss/dielectric_charge", "pinn_dielectric_charge"),
                    ("Loss/top_boundary", "pinn_top_boundary"),
                    ("Loss/temporal_smoothness", "pinn_temporal_smoothness"),
                ]:
                    w.add_scalar(tb_name, get_val(losses, loss_key), e)
                w.add_scalar("Learning_Rate", self.optimizer.param_groups[0]["lr"], e)

            # 梯度/权重直方图: 仅记录首层和末层, 每 5000 epoch
            if epoch % 5000 == 0 and epoch > 0:
                for name, param in [
                    ("phi_net.0.weight", self.model.phi_net[0].weight),
                    ("phi_net.last.weight", list(self.model.phi_net.modules())[-1].weight),
                    ("vel_net.0.weight", self.model.vel_net[0].weight),
                ]:
                    try:
                        if param.grad is not None:
                            self.tb_writer.add_histogram(f"grad/{name}", param.grad, epoch)
                            self.tb_writer.add_histogram(f"weight/{name}", param.data, epoch)
                    except Exception:
                        pass

            # [2025-12-31] 物理损失统一从 pinn_physics 获取
            # 旧的 continuity, vof, ns, surface_tension 已不再单独计算
            physics_loss = get_val(losses, "pinn_physics")
            interface_loss = get_val(losses, "interface")

            self.history["epoch"].append(epoch)
            self.history["loss"].append(total_loss.item())
            self.history["interface"].append(interface_loss)
            self.history["physics"].append(physics_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.history["pinn_continuity"].append(get_val(losses, "pinn_continuity"))
            self.history["pinn_vof"].append(get_val(losses, "pinn_vof"))
            self.history["pinn_momentum_u"].append(get_val(losses, "pinn_momentum_u"))
            self.history["pinn_momentum_v"].append(get_val(losses, "pinn_momentum_v"))
            self.history["pinn_momentum_w"].append(get_val(losses, "pinn_momentum_w"))
            self.history["pinn_electrowetting"].append(get_val(losses, "pinn_electrowetting"))
            self.history["pinn_temporal_smoothness"].append(
                get_val(losses, "pinn_temporal_smoothness")
            )
            self.history["pinn_laplace_pressure"].append(get_val(losses, "pinn_laplace_pressure"))
            self.history["pinn_sidewall_contact_angle"].append(
                get_val(losses, "pinn_sidewall_contact_angle")
            )
            self.history["pinn_interface_energy"].append(get_val(losses, "pinn_interface_energy"))
            self.history["pinn_wall_wetting"].append(get_val(losses, "pinn_wall_wetting"))
            self.history["pinn_bottom_wetting"].append(get_val(losses, "pinn_bottom_wetting"))
            self.history["pinn_phase_field_wetting"].append(
                get_val(losses, "pinn_phase_field_wetting")
            )
            self.history["pinn_dielectric_charge"].append(get_val(losses, "pinn_dielectric_charge"))
            self.history["pinn_contact_line_dynamics"].append(
                get_val(losses, "pinn_contact_line_dynamics")
            )
            self.history["pinn_top_boundary"].append(get_val(losses, "pinn_top_boundary"))
            self.history["contact_angle"].append(get_val(losses, "contact_angle"))
            self.history["volume"].append(get_val(losses, "volume_conservation"))

            # 定期评估与保存
            if epoch % 100 == 0:
                # 预热期后才使用 scheduler
                if epoch >= self.warmup_epochs and torch.isfinite(total_loss):
                    self.scheduler.step(total_loss)

                current_loss = total_loss.item()

                # 计算物理损失（用于评估物理合理性）
                # 优先使用统一后的 pinn_physics
                physics_loss_val = losses.get("pinn_physics", torch.tensor(0.0)).item()
                if physics_loss_val == 0.0:
                    # 兼容旧逻辑（以防万一）
                    physics_loss_val = sum(
                        losses.get(k, torch.tensor(0.0)).item()
                        for k in ["volume_conservation", "continuity", "vof", "ns"]
                    )

                # 只有在阶段3（完整物理约束）之后才更新 best_loss
                # 这样确保保存的模型是物理上合理的
                if epoch >= self.best_loss_recording_start:
                    is_best = current_loss < self.best_loss
                    if is_best:
                        self.best_loss = current_loss
                        self.patience_counter = 0
                        logger.info(f"  ✓ 新最佳损失: {current_loss:.4e} (epoch {epoch})")
                    else:
                        self.patience_counter += 100

                    # 同时跟踪物理损失最佳值
                    self.best_physics_loss = min(self.best_physics_loss, physics_loss_val)
                else:
                    # 阶段1/2：不更新 best_loss，但仍保存检查点
                    is_best = False
                    self.patience_counter = 0  # 阶段1/2不计入早停

                # 保存检查点 (包括最新和可能的最佳)
                self._save_checkpoint(epoch, is_best=is_best)

                elapsed = time.time() - start_time

                # 构建详细的损失字符串
                physics_str = ""
                if "low_voltage" in losses:
                    physics_str += f" | LV: {losses['low_voltage'].item():.2e}"
                if "volume_conservation" in losses:
                    vol_val = losses["volume_conservation"].item()
                    # 体积守恒：显示趋势箭头
                    vol_trend = "↑" if vol_val > 2000 else ("→" if vol_val > 1000 else "↓")
                    physics_str += f" | Vol: {vol_val:.2e}{vol_trend}"
                if "contact_angle" in losses:
                    physics_str += f" | θ: {losses['contact_angle'].item():.2e}"
                # eta_stage1 tutor 已移除 — Stage1 仅提供 θ(V) 和参考 η(V)
                if "phi_spatial" in losses:
                    physics_str += f" | φS: {losses['phi_spatial'].item():.2e}"

                # 物理场损失 (优先检查 pinn_ 前缀)
                continuity_val = losses.get("pinn_continuity", losses.get("continuity"))
                if continuity_val is not None:
                    physics_str += f" | C: {continuity_val.item():.2e}"

                interface_val = losses.get("interface")
                if interface_val is not None:
                    physics_str += f" | IF: {interface_val.item():.2e}"

                # 新增: Allen-Cahn / 表面张力 / Laplace / 壁面接触角 / 界面能
                ac_val = losses.get("pinn_vof")
                if ac_val is not None:
                    physics_str += f" | AC: {ac_val.item():.2e}"
                lp_val = losses.get("pinn_laplace_pressure")
                if lp_val is not None:
                    physics_str += f" | LP: {lp_val.item():.2e}"
                sw_val = losses.get("pinn_sidewall_contact_angle")
                if sw_val is not None:
                    physics_str += f" | SW: {sw_val.item():.2e}"
                ie_val = losses.get("pinn_interface_energy")
                if ie_val is not None:
                    physics_str += f" | IE: {ie_val.item():.2e}"
                ew_val = losses.get("pinn_electrowetting")
                if ew_val is not None:
                    physics_str += f" | EW: {ew_val.item():.2e}"
                ww_val = losses.get("pinn_wall_wetting")
                if ww_val is not None:
                    physics_str += f" | WW: {ww_val.item():.2e}"
                bw_val = losses.get("pinn_bottom_wetting")
                if bw_val is not None:
                    physics_str += f" | BW: {bw_val.item():.2e}"
                pfw_val = losses.get("pinn_phase_field_wetting")
                if pfw_val is not None:
                    physics_str += f" | PFW: {pfw_val.item():.2e}"
                dc_val = losses.get("pinn_dielectric_charge")
                if dc_val is not None:
                    physics_str += f" | DC: {dc_val.item():.2e}"
                cld_val = losses.get("pinn_contact_line_dynamics")
                if cld_val is not None:
                    physics_str += f" | CLD: {cld_val.item():.2e}"
                tb_val = losses.get("pinn_top_boundary")
                if tb_val is not None:
                    physics_str += f" | TB: {tb_val.item():.2e}"

                # 显示当前阶段
                stage_str = f"S{1 if epoch < self.stage1_epochs else (2 if epoch < self.stage2_epochs else 3)}"

                logger.info(
                    f"Epoch {epoch:5d} [{stage_str}] | Loss: {current_loss:.4e}{physics_str} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s"
                )

                # 每 1000 轮保存一次中间训练曲线，防止丢失
                if epoch % 1000 == 0 and epoch > 0:
                    self._plot_curves(epoch=epoch)

            # 早停检查
            if self.patience_counter >= self.early_stop_patience:
                logger.info(f"\n早停触发：连续 {self.early_stop_patience} 轮损失未下降")
                break

        # 训练结束，保存最终结果
        self._save_checkpoint(epoch, is_final=True)
        self._plot_curves()

        # [2026-01-30] L-BFGS 微调
        if self.config.get("training", {}).get("use_lbfgs", False):
            lbfgs_iter = self.config.get("training", {}).get("lbfgs_iter", 5000)
            self.fine_tune_lbfgs(max_iter=lbfgs_iter)

        # 调用专业评估套件生成全套图表
        try:
            from evaluate import PINNEvaluator

            evaluator = PINNEvaluator()
            # 优先使用 final_model (含完整物理约束), 其次 best_model
            eval_ckpt = os.path.join(self.output_dir, "final_model.pth")
            if not os.path.exists(eval_ckpt):
                eval_ckpt = os.path.join(self.output_dir, "best_model.pth")
            if os.path.exists(eval_ckpt):
                model, _ = evaluator.load_model(eval_ckpt)
                if model:
                    evaluator.plot_dashboard(
                        model,
                        os.path.join(self.output_dir, "pro_dashboard.png"),
                        model_name=self.output_dir,
                    )
                    evaluator.plot_phi_grid(
                        model, os.path.join(self.output_dir, "phi_grid_evolution.png")
                    )
                    evaluator.plot_interface_3d(
                        model,
                        os.path.join(self.output_dir, "interface_3d_steady.png"),
                        30.0,
                        0.02,
                        30.0,
                    )
                    logger.info(
                        f"✅ 专业评估套件已生成 (model={os.path.basename(eval_ckpt)}): {self.output_dir}/"
                    )
        except Exception as e:
            logger.error(f"生成专业评估仪表盘失败: {e}")

        # 关闭 TensorBoard writer
        self.tb_writer.close()
        logger.info(f"TensorBoard 日志已保存到: {os.path.join(self.output_dir, 'runs')}")

        logger.info("=" * 60)
        logger.info(f"训练完成! 最佳损失: {self.best_loss:.6e}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

    def visualize(self):
        """
        可视化结果 - 已弃用，由 evaluate.py 替代
        """
        pass


# ============================================================================
# 主函数
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="EWP 两相流 PINN 训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--resume_from", type=str, default=None, help="checkpoint 路径，用于续训")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="强制确定性计算（降低性能但可复现）。默认关闭，启用 cudnn.benchmark 加速。",
    )

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed, deterministic=args.deterministic)
    logger.info(f"🌱 随机种子: {args.seed} | deterministic: {args.deterministic}")

    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()

    # 命令行参数覆盖
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr

    # 训练
    trainer = Trainer(config, resume_path=args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
