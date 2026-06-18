#!/usr/bin/env python3
"""
PhysicsLoss — 两相流物理损失适配层
====================================

从 pinn_two_phase.py 中抽取的 PhysicsLoss 类，独立为单独模块。

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

import logging

import torch
from torch import nn

from src.config import PHYSICS
from src.physics.constraints import PhysicsConstraints

logger = logging.getLogger("EWP-PINN")


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

    def compute_all_residuals(self, model: nn.Module, points: torch.Tensor) -> dict[str, torch.Tensor]:
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
        residuals = self.physics_constraints.compute_core_residuals(x_phys=points, predictions=predictions, model=model)

        # 清理所有残差中的 NaN/Inf
        for key in residuals:
            residuals[key] = self._sanitize_tensor(residuals[key], key)

        return residuals

    def compute_total_loss(
        self, model: nn.Module, points: torch.Tensor, weights: dict[str, float] | None = None
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
            "continuity": 0.5,
            "momentum_u": 0.1,
            "momentum_v": 0.1,
            "momentum_w": 0.1,
            "vof": 0.5,
            "interface_energy": 2.0,
            "laplace_pressure": 0.05,
            "sharpening": 1.0,
            "temporal_smoothness": 0.1,
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

            # 计算 MSE，添加数值保护防止溢出
            mse = torch.mean(torch.clamp(residual**2, min=0.0, max=1e36))

            # 应用 log1p 缩放稳定大损失
            scaled_loss = torch.log1p(mse)
            if not torch.isfinite(scaled_loss):
                scaled_loss = torch.tensor(100.0, device=self.device)

            # 自适应归一化: 除以 EMA 使各损失量级统一
            if use_adaptive and torch.isfinite(scaled_loss):
                val = scaled_loss.detach().item()
                if key not in self._loss_ema:
                    self._loss_ema[key] = val
                else:
                    self._loss_ema[key] = self._ema_decay * self._loss_ema[key] + (1 - self._ema_decay) * val
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

        # Allen-Cahn 双阱势正则化: φ²(1-φ)²
        # 与 AC 方程的 f'(φ)=2φ(1-φ)(1-2φ) 一致（同一泛函的导数）
        # 极小值在 φ=0 和 φ=1，不惩罚 φ≈0.5 的扩散界面
        if "sharpening" in weights and weights["sharpening"] > 0:
            sharpening_val = torch.mean(phi_pred**2 * (1.0 - phi_pred) ** 2)
            losses["sharpening"] = weights["sharpening"] * sharpening_val
            total_loss = total_loss + losses["sharpening"]

        # 显式体积约束
        if "explicit_volume" in weights and weights["explicit_volume"] > 0:
            ink_fraction_target = self.materials_params.get("ink_initial_fraction", 0.15)
            explicit_volume_val = torch.abs(torch.mean(phi_pred) - ink_fraction_target)
            losses["explicit_volume"] = weights["explicit_volume"] * explicit_volume_val
            total_loss = total_loss + losses["explicit_volume"]

        losses["total"] = total_loss
        return losses
