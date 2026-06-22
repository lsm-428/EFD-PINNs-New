#!/usr/bin/env python3
"""
EWP 两相流 PINN 网络架构
==========================

从 pinn_two_phase.py 中抽取的网络定义模块，包含：
1. FourierFeature — Fourier 特征编码，缓解 PINN 谱偏置
2. TwoPhasePINN — 两相流物理信息神经网络（6D Triad 输入）

本文件仅包含网络架构定义，不包含训练逻辑（PhysicsLoss/DataGenerator/Trainer）。

输入: (x, y, z, V_from, V_to, t_since) - 6维 Triad
输出: (u, v, w, p, phi) - 5维，phi ∈ [0,1]（1=油墨，0=极性液体）

作者: EFD-PINNs Team
日期: 2024-12
"""

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from src.config import PHYSICS, get_default_training_config

logger = logging.getLogger("EWP-PINN")


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
    两相流物理信息神经网络（6D Triad 输入）

    输入: (x, y, z, V_from, V_to, t_since) - 6维
      * V_from: 跳变前电压，V_to: 当前电压，t_since: 跳变后经过时间
      * V_from=V_to: 恒压状态
      * V_from<V_to: 升压过程（电润湿驱动，油墨向外收缩形成开口）
      * V_from>V_to: 降压过程（表面张力恢复，油墨向中心铺展）
      * 单个模型支持任意电压序列的时间连续仿真

    输出: (u, v, w, p, phi)，phi ∈ [0,1]（1=油墨，0=极性液体）

    硬约束（use_hard_constraints=True 时启用）：
      - 顶面 BC: z=Lz → φ=0（ITO 玻璃）
      - 壁顶面 BC: z≈wall_height, 非接触线区 → φ=0（极性液体）
      - 壁顶接触线 BC: z≈wall_height, 壁面边缘 → φ=0.5（三相线）
      - 夹角区 BC: z=0, x/y∈[0,wall_height]∪[L-wall_height,L] → φ=1（油墨堆积）
      - IC blend:
        * z=0 非夹角区: 始终自由
        * t<2ms 或 V<V_T: 强制 IC（φ=phi_ic）
        * t>2ms, V>V_T: 逐渐自由（blend=1-t_norm/t_early）
      - IC 目标 phi_ic:
        * z < h_ink=3μm: φ=1（油墨）
        * z ≈ wall_height=3.5μm: φ=0.5（接触线）
        * z > wall_height: φ=0（水）
        * h_ink < z < wall_height: tanh 过渡
    """

    def __init__(self, config: dict[str, Any] | None = None):
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
        self.use_hard_constraints = hard_cfg.get("enable", True)  # 默认启用
        self.h_ink = hard_cfg.get("h_ink", 3e-6)
        self.hard_ic_width = hard_cfg.get("ic_width", PHYSICS["ic_width"])
        self.sigmoid_temperature = hard_cfg.get("sigmoid_temperature", 1.0)
        # 围堰几何（硬约束用）
        self.wall_height = PHYSICS["wall_height"]
        self.wall_top_half_width = PHYSICS["wall_top_half_width"]

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
              V_from: 跳变前电压; V_to: 当前电压; t_since: 跳变后经过时间
              V_from=V_to: 恒压状态
              V_from<V_to: 升压过程（电润湿驱动，油墨向外收缩）
              V_from>V_to: 降压过程（表面张力恢复，油墨向中心铺展）

        Returns:
            (batch, 5) - (u, v, w, p, phi), phi ∈ [0,1] (1=油墨, 0=极性液体)

        硬约束 (use_hard_constraints=True 时通过构造保证):
          - 顶面 BC: z=Lz → φ=0 (ITO 玻璃)
          - 壁顶面 BC: z≈wall_height, 非接触线区 → φ=0 (极性液体)
          - 壁顶接触线 BC: z≈wall_height, 壁面边缘 → φ=0.5 (三相线)
          - 夹角区 BC: z=0, x/y∈[0,wall_height]∪[L-wall_height,L] → φ=1 (油墨堆积)
          - IC blend:
            * z=0 非夹角区: 始终自由
            * t<2ms 或 V<V_T: 强制 IC (blend=0, φ=phi_ic)
            * t>2ms, V>V_T, z>0: 渐变自由 (blend=1-t_norm/t_early)
          - IC 目标 phi_ic:
            * z < h_ink=3μm: φ=1 (油墨)
            * z ≈ wall_height=3.5μm: φ=0.5 (接触线)
            * z > wall_height: φ=0 (水)
            * h_ink < z < wall_height: tanh 过渡
        """
        if not torch.jit.is_tracing() and x.shape[1] != 6:
            msg = "TwoPhasePINN expects input of shape (batch, 6)."
            raise ValueError(msg)
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
            phi_features = torch.cat([x_norm, y_norm, z_norm, V_from_norm, V_to_norm, t_norm], dim=-1)

        # Phi 预测
        phi_input = phi_features
        phi_raw = self.phi_net(phi_input)
        sigmoid_T = getattr(self, "sigmoid_temperature", 1.0)
        phi_learned = torch.sigmoid(sigmoid_T * phi_raw)  # 限制在 [0, 1]

        # --- 硬约束编码 ---
        if getattr(self, "use_hard_constraints", False):
            # 统一升维到 (batch, 1)，避免 torch.where 广播 (batch,1) 和 (batch,) → (batch,batch)
            z_phys = z_coord  # (batch, 1)
            h_ink = getattr(self, "h_ink", PHYSICS["h_ink"])
            wall_height = getattr(self, "wall_height", PHYSICS["wall_height"])
            _z_eps = 1e-8  # z 方向浮点容差，非物理参数
            delta_ic = getattr(self, "hard_ic_width", PHYSICS["ic_width"])
            V_T = PHYSICS["V_T_base"]
            t_early = 0.002  # 2ms

            # 输入坐标（用于判断位置）— 保持 (batch, 1) 以匹配 phi 维度
            x_phys = x[:, 0:1]  # x 物理坐标
            y_phys = x[:, 1:2]  # y 物理坐标
            V_eff = x[:, 4:5]  # V_to, (batch, 1)
            t_since = x[:, 5:6]  # t_since, (batch, 1)

            # ============================================================
            # 1. 顶面 BC: φ(z=Lz) = 0 (ITO 玻璃) [软约束]
            # ============================================================
            phi = phi_learned * (1.0 - z_norm)

            # ============================================================
            # 2. 初始条件 IC(z) — z ≤ wall_height 区域
            #    z < h_ink: φ=1（油墨）
            #    h_ink ≤ z ≤ wall_height: tanh 过渡
            #    z > wall_height: 中性值 0.5（不产生梯度）
            # ============================================================
            in_ink = z_phys < h_ink
            in_ic_zone = z_phys <= wall_height
            phi_ic = torch.where(
                in_ic_zone,
                torch.where(
                    in_ink,
                    torch.ones_like(phi),
                    0.5 * (1.0 + torch.tanh((h_ink - z_phys) / delta_ic)),
                ),
                torch.full_like(phi, 0.5),
            )

            # ============================================================
            # 3. Blend 因子
            #    t<2ms 或 V<V_T：blend=0（强制 IC）
            #    其他：blend=1（自由）
            #    z > wall_height：始终 blend=1
            # ============================================================
            above_wall_mask = (z_phys > wall_height).float()
            force_ic = ((t_since < t_early) | (V_eff < V_T)).float()
            blend = 1.0 - (1.0 - above_wall_mask) * force_ic

            phi = (1.0 - blend) * phi_ic + blend * phi

            # ============================================================
            # 4. 壁顶面 BC（SU-8 亲水面）[硬约束，在 IC blend 之后]
            #    z = wall_height, d_wall ≥ wall_height: φ=0
            #    z = wall_height, d_wall < wall_height: φ=0.5（接触线）
            # ============================================================
            on_wall_top_z = torch.abs(z_phys - wall_height) < _z_eps
            d_wall = torch.min(torch.min(x_phys, self.Lx - x_phys), torch.min(y_phys, self.Ly - y_phys))
            on_contact_line = on_wall_top_z & (d_wall < _z_eps)
            on_wall_top_face = on_wall_top_z & ~on_contact_line

            phi = torch.where(on_wall_top_face, torch.zeros_like(phi), phi)
            phi = torch.where(on_contact_line, torch.full_like(phi, 0.5), phi)

            # ============================================================
            # 5. 夹角区域（围堰立面与底板夹角，Teflon 亲油）[最高优先级]
            #    d_wall < wall_height, z < wall_height: φ=1
            # ============================================================
            on_corner_x = (x_phys < wall_height) | (x_phys > self.Lx - wall_height)
            on_corner_y = (y_phys < wall_height) | (y_phys > self.Ly - wall_height)
            on_corner_z0 = (z_phys < wall_height) & (on_corner_x | on_corner_y)
            phi = torch.where(on_corner_z0, torch.ones_like(phi), phi)
        else:
            phi = phi_learned

        # 速度预测 - 使用 Fourier 特征 + phi
        vel_input = torch.cat([phi_features, phi], dim=-1)
        vel_out = self.vel_net(vel_input)
        u, v, w, p = vel_out[:, 0:1], vel_out[:, 1:2], vel_out[:, 2:3], vel_out[:, 3:4]

        output = torch.cat([u, v, w, p, phi], dim=-1)

        # [2026-06-11] NaN 诊断：前向传播输出检查（仅首次 NaN 时记录，避免刷屏）
        if not getattr(self, "_nan_forward_reported", False) and torch.isnan(output).any():
            nan_fields = []
            field_names = ["u", "v", "w", "p", "phi"]
            for i, name in enumerate(field_names):
                col = output[:, i]
                if torch.isnan(col).any():
                    nan_count = torch.isnan(col).sum().item()
                    nan_fields.append(f"{name}({nan_count}/{col.shape[0]})")
            logger.warning(
                f"TwoPhasePINN.forward 输出 NaN: {', '.join(nan_fields)}\n"
                f"  Input stats: x∈[{x_coord.min().item():.2f}, {x_coord.max().item():.2f}], "
                f"y∈[{y_coord.min().item():.2f}, {y_coord.max().item():.2f}], "
                f"z∈[{z_coord.min().item():.2f}, {z_coord.max().item():.2f}], "
                f"V_from∈[{V_from.min().item():.2f}, {V_from.max().item():.2f}], "
                f"V_to∈[{V_to.min().item():.2f}, {V_to.max().item():.2f}], "
                f"t∈[{t_since.min().item():.4f}, {t_since.max().item():.4f}]"
            )
            self._nan_forward_reported = True  # 只报一次，避免刷屏

        return output


# ============================================================================
# 默认配置（单一来源：config/device_calibrated_physics.json）
# ============================================================================

DEFAULT_CONFIG = get_default_training_config()
