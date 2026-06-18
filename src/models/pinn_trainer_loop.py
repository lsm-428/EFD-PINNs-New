#!/usr/bin/env python3
"""
Trainer(TrainerBase) — 损失计算 + 训练循环
============================================

从 pinn_two_phase.py 中抽取的训练逻辑模块，包含：
1. compute_losses() — 总损失调度
2. _compute_data_loss() — 界面数据拟合损失
3. _get_ic_annealed_weight() — IC 退火权重
4. _compute_initial_boundary_loss() — IC + BC 损失
5. _compute_phi_spatial_loss() — phi 空间分布 + 几何一致性约束
6. _compute_volume_conservation_loss() — 体积守恒约束
7. _compute_physics_equation_loss() — PDE 物理损失（通过 PhysicsLoss）
8. compute_aperture_ratio / compute_aperture_ratio_batch / compute_aperture_ratio_differentiable — 开口率计算
9. fine_tune_lbfgs() — L-BFGS 二阶优化微调
10. train() — 主训练循环

本文件继承 TrainerBase（pinn_trainer_base.py），不重复实现 __init__、调度器、checkpoint 等。

作者: EFD-PINNs Team
日期: 2024-12
"""

import concurrent.futures
import logging
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import PHYSICS
from src.models.pinn_trainer_base import TrainerBase

logger = logging.getLogger("EWD-PINN")


class Trainer(TrainerBase):
    """两相流 PINN 训练器（损失计算 + 训练循环）"""

    # ========================================================================
    # 损失计算方法
    # ========================================================================

    def _compute_data_loss(self, data: dict[str, torch.Tensor], physics_cfg: dict, interface_mult: float):
        """1. 界面数据拟合损失"""
        idx = torch.randperm(len(data["interface_points"]))[: self.batch_size]
        interface_pts = data["interface_points"][idx]
        interface_tgt = data["interface_targets"][idx]

        phi_pred = self.model(interface_pts)[:, 4]
        interface_loss = F.mse_loss(phi_pred, interface_tgt)

        base_weight = physics_cfg.get("interface_weight", 500.0)
        return interface_loss * base_weight * interface_mult, interface_loss

    def _compute_initial_boundary_loss(self, data: dict[str, torch.Tensor], physics_cfg: dict, ic_phi_mult: float):
        """3. 初始条件 & 壁面边界条件损失"""
        res = {}
        # IC
        idx_ic = torch.randperm(len(data["ic_points"]))[: self.batch_size // 4]
        pred_ic = self.model(data["ic_points"][idx_ic])
        ic_phi_loss = F.mse_loss(pred_ic[:, 4:5], data["ic_values"][idx_ic][:, 4:5])
        ic_vel_loss = F.mse_loss(pred_ic[:, :4], data["ic_values"][idx_ic][:, :4])
        # IC phi 部分退火：基础权重 * 统一调度乘子
        ic_phi_weight = physics_cfg.get("ic_weight", 300.0) * ic_phi_mult
        res["ic"] = ic_phi_loss * ic_phi_weight + ic_vel_loss * 50.0

        # BC
        idx_bc = torch.randperm(len(data["bc_points"]))[: self.batch_size // 4]
        pred_bc = self.model(data["bc_points"][idx_bc])
        res["bc"] = F.mse_loss(pred_bc[:, :3], data["bc_values"][idx_bc][:, :3]) * physics_cfg.get("bc_weight", 50.0)
        res["bc"] += F.mse_loss(pred_bc[:, 4:5], data["bc_values"][idx_bc][:, 4:5]) * 80.0
        return res

    def _compute_phi_spatial_loss(self, interface_mult: float):
        """phi 场空间分布 + 几何一致性约束（批量前向版本）。

        替代原 eta_matching + phi_target3D 方案：
        - spatial: 中心+边缘两点约束（中心 phi<0.3, 边缘 phi>0.7）
        - geometry: 三点约束（中心 phi<0.2, 边缘油墨 phi>0.8, 边缘极性 phi<0.2）

        从 eta 直接计算 r_open 和 h_edge，不依赖复杂的 target_phi_3d 函数。
        批量前向：所有工况点拼接为一次大前向，效率高。

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

        spatial_pts_center = []
        spatial_pts_edge = []

        spatial_loss = torch.tensor(0.0, device=self.device)
        spatial_count = 0
        spatial_offset_center = 0
        spatial_offset_edge = 0
        center_valid_ranges = []
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
                rr = r_edge_min + torch.rand(n_spatial_pts, device=self.device) * (r_edge_max - r_edge_min)
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
            for _i, rng in enumerate(center_valid_ranges):
                if rng is not None:
                    s, n = rng
                    phi_c = all_phi_center[s : s + n]
                    spatial_loss = spatial_loss + torch.mean(torch.relu(phi_c - 0.3) ** 2)
                    spatial_count += 1

        # 一次前向计算所有 spatial 边缘点
        if spatial_pts_edge:
            all_edge_pts = torch.cat(spatial_pts_edge, dim=0)
            all_phi_edge = self.model(all_edge_pts)[:, 4]
            for _i, rng in enumerate(edge_valid_ranges):
                if rng is not None:
                    s, n = rng
                    phi_e = all_phi_edge[s : s + n]
                    spatial_loss = spatial_loss + torch.mean(torch.relu(0.7 - phi_e) ** 2)
                    spatial_count += 1

        if spatial_count == 0:
            spatial_loss = torch.tensor(0.0, device=self.device)
        else:
            spatial_loss = spatial_loss * 500.0 * interface_mult / spatial_count

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
        geom_valid_ranges = []

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
                rr = r_edge_min + torch.rand(n_geom_pts, device=self.device) * (r_edge_max_g - r_edge_min)
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
                                z_top_min + torch.rand(n_geom_pts, device=self.device) * (Lz - z_top_min),
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
                # 中心区域: phi < 0.2
                if n_regions >= 1:
                    phi_c = all_phi_geom[s : s + n_geom_pts]
                    geom_loss = geom_loss + torch.mean(torch.relu(phi_c - 0.2) ** 2)
                    s += n_geom_pts
                # 边缘油墨区: phi > 0.8
                if n_regions >= 2:
                    phi_ink = all_phi_geom[s : s + n_geom_pts]
                    geom_loss = geom_loss + torch.mean(torch.relu(0.8 - phi_ink) ** 2)
                    s += n_geom_pts
                # 边缘极性区: phi < 0.2
                if n_regions >= 3:
                    phi_polar = all_phi_geom[s : s + n_geom_pts]
                    geom_loss = geom_loss + torch.mean(torch.relu(phi_polar - 0.2) ** 2)
                geom_count += n_regions
            if geom_count > 0:
                keep_factor = 0.2 + 0.8 * interface_mult
                geom_loss = geom_loss * 200.0 * keep_factor / geom_count
            else:
                geom_loss = torch.tensor(0.0, device=self.device)
        else:
            geom_loss = torch.tensor(0.0, device=self.device)

        return spatial_loss, geom_loss

    def _compute_volume_conservation_loss(self, interface_mult: float):
        """体积守恒约束 — 批量前向版本。

        将 N 个 (V_from, V_to, t_since) 场景分 batch 前向，避免显存峰值过高。
        场景列表和采样参数可通过 training config 配置。
        """
        training_cfg = self.config.get("training", {}) if isinstance(self.config.get("training", {}), dict) else {}
        n_vol = int(training_cfg.get("volume_n_vol", 8000))
        n_vol = max(2000, n_vol)
        vol_batch = int(training_cfg.get("volume_batch", 4096))  # 每批最大点数，控制显存
        vol_batch = max(1024, vol_batch)
        Lx, Ly, Lz, h_ink = (
            PHYSICS["Lx"],
            PHYSICS["Ly"],
            PHYSICS["Lz"],
            PHYSICS["h_ink"],
        )
        v0 = Lx * Ly * h_ink
        v_domain = Lx * Ly * Lz

        # --- 场景列表（可通过 config 覆盖）---
        vol_sc_cfg = training_cfg.get("volume_scenarios", {})
        # 稳态电压列表
        v_steady = vol_sc_cfg.get("v_steady", [0.0, 10.0, 20.0, 25.0, 30.0])
        n_t_steady = vol_sc_cfg.get("n_t_steady", 3)
        t_steady = np.random.uniform(0.020, 0.050, n_t_steady).tolist()
        # 升压：(v_from, v_to) — 5V 一档从 0→30V
        v_pairs_on = vol_sc_cfg.get(
            "v_pairs_on",
            [[0, 5], [0, 10], [0, 15], [0, 20], [0, 25], [0, 30]],
        )
        n_t_on = vol_sc_cfg.get("n_t_on", 5)
        t_on = np.random.uniform(0.002, 0.025, n_t_on).tolist()
        # 降压 — 5V 一档从 30V→0
        v_pairs_off = vol_sc_cfg.get(
            "v_pairs_off",
            [[5, 0], [10, 0], [15, 0], [20, 0], [25, 0], [30, 0]],
        )
        n_t_off = vol_sc_cfg.get("n_t_off", 5)
        t_off = np.random.uniform(0.002, 0.025, n_t_off).tolist()

        tests = []
        for v in v_steady:
            for t in t_steady:
                tests.append((float(v), float(v), float(t)))
        for vf, vt in v_pairs_on:
            for t in t_on:
                tests.append((float(vf), float(vt), float(t)))
        for vf, vt in v_pairs_off:
            for t in t_off:
                tests.append((float(vf), float(vt), float(t)))

        N = len(tests)
        if N == 0:
            return torch.tensor(0.0, device=self.device)

        # --- 分 batch 前向，控制显存峰值 ---
        # 空间坐标只生成一次，每个场景复用
        xyz = torch.stack(
            [
                torch.rand(n_vol, device=self.device) * Lx,
                torch.rand(n_vol, device=self.device) * Ly,
                torch.rand(n_vol, device=self.device) * Lz,
            ],
            dim=1,
        )  # (n_vol, 3)

        phi_by_scene = torch.zeros(N, device=self.device)
        # 将 N 个场景分成若干组，每组最多 vol_batch//n_vol 个场景
        scenes_per_batch = max(1, vol_batch // n_vol)
        for i in range(0, N, scenes_per_batch):
            j = min(i + scenes_per_batch, N)
            n_sc = j - i
            # 当前批次的场景
            batch_tests = tests[i:j]

            xyz_b = xyz.repeat(n_sc, 1)  # (n_sc*n_vol, 3)
            v_from_b = torch.tensor([t[0] for t in batch_tests], device=self.device).repeat_interleave(n_vol)
            v_to_b = torch.tensor([t[1] for t in batch_tests], device=self.device).repeat_interleave(n_vol)
            t_since_b = torch.tensor([t[2] for t in batch_tests], device=self.device).repeat_interleave(n_vol)

            pts = torch.cat([xyz_b, v_from_b.unsqueeze(1), v_to_b.unsqueeze(1), t_since_b.unsqueeze(1)], dim=1)
            # 模型输出已经过 sigmoid ∈ (0,1)，无需 clamp
            phi_b = self.model(pts)[:, 4]  # (n_sc*n_vol,)
            phi_by_scene[i:j] = phi_b.view(n_sc, n_vol).mean(dim=1)  # (n_sc,)

        v_curr = v_domain * phi_by_scene
        rel_errors = (v_curr - v0) / (v0 + 1e-12)
        loss_vol = torch.mean(rel_errors**2)

        base_weight = float(training_cfg.get("volume_base_weight", 2000.0))
        stage_weight = 0.2 + (1.0 - float(interface_mult))

        # S3 退火：S3 前期（油墨运动阶段）弱化体积约束，后期恢复
        # 避免油墨动态变形中体积守恒与电润湿驱动冲突
        # epoch=4000→ramp=0.1, epoch≈37500→ramp=1.0, epoch=60000→ramp=1.0
        # 下限 0.1 确保 S3 前期仍有弱体积约束，防止油墨体积漂移
        _epoch = getattr(self, "_current_epoch", 0)
        if _epoch > self.stage2_epochs:
            _s3_total = max(1, self.epochs - self.stage2_epochs)
            _s3_prog = (_epoch - self.stage2_epochs) / _s3_total
            _vol_ramp = max(0.1, min(1.0, _s3_prog * 2.0))
        else:
            _vol_ramp = 1.0

        return loss_vol * base_weight * stage_weight * _vol_ramp

    def _compute_physics_equation_loss(
        self,
        weights: dict,
        epoch: int,
        data: dict[str, torch.Tensor] | None = None,
    ):
        """
        8. 物理方程损失 - 通过 PhysicsLoss.compute_total_loss 计算

        [2026-05-19] 统一通过 PhysicsLoss.compute_total_loss (→ compute_core_residuals)
        计算所有物理约束，包括 VOF, NS, Laplace, 润湿BC, 接触线动力学等。
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
            phys_weights = {
                "continuity": float(weights.get("continuity", 0.0)),
                "vof": float(weights.get("vof", 0.0)),
                "momentum_u": float(weights.get("ns", 0.0)),
                "momentum_v": float(weights.get("ns", 0.0)),
                "momentum_w": float(weights.get("ns", 0.0)),
                "laplace_pressure": float(weights.get("laplace_pressure", 0.05)),
                "interface_energy": float(weights.get("interface_energy", 2.0)),
                "temporal_smoothness": float(weights.get("temporal_smoothness", 0.1)),
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
                    "temporal_smoothness",
                    "laplace_pressure",
                    "interface_energy",
                ]:
                    if k in losses and isinstance(losses[k], torch.Tensor) and torch.isfinite(losses[k]):
                        res[f"pinn_{k}"] = losses[k]

        except Exception as e:
            logger.warning(f"Physics loss failed: {e}")
        return res

    # ========================================================================
    # 开口率计算方法
    # ========================================================================

    def compute_aperture_ratio(self, V_from: float, V_to: float, t_since: float, n_grid: int = 30) -> torch.Tensor:
        """
        计算开口率：z=0 底面上"极性液体覆盖面积"的比例（软统计）

        Args:
            V_from: 跳变前电压 (V)
            V_to: 跳变后电压 (V)
            t_since: 跳变后经过的时间 (s)
            n_grid: 网格分辨率
        """
        if V_from is None:
            V_from = V_to

        return self.compute_aperture_ratio_batch([(float(V_from), float(V_to), float(t_since))], n_grid=n_grid)[0]

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

        eval_cfg = self.config.get("eval", {}) if isinstance(self.config.get("eval", {}), dict) else {}
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
        可微的开口率计算：去掉 torch.no_grad()，让 eta 的梯度能反传。

        用于训练 loss（eta 匹配），区别于 evaluate 用的 compute_aperture_ratio。

        Args:
            V_from: 跳变前电压 (V)
            V_to: 跳变后电压 (V)
            t_since: 跳变后经过的时间 (s)
            n_grid: 网格分辨率

        Returns:
            标量 tensor，eta ∈ [0, 1]，带梯度
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

        eval_cfg = self.config.get("eval", {}) if isinstance(self.config.get("eval", {}), dict) else {}
        phi0 = float(eval_cfg.get("aperture_phi0", 0.3))
        eval_eps = max(1e-6, float(eval_cfg.get("aperture_eps", 0.05)))

        # 关键：不用 torch.no_grad()！
        phi = torch.clamp(self.model(points)[:, 4], 0.0, 1.0)
        masks = torch.sigmoid((phi0 - phi) / eval_eps)  # 软二值化
        return masks.mean()

    # ========================================================================
    # 总损失调度
    # ========================================================================

    def compute_losses(self, data: dict[str, torch.Tensor], epoch: int) -> dict[str, torch.Tensor]:
        """
        计算所有损失 - 模块化重构

        phi = 1: 油墨, phi = 0: 极性液体
        """
        losses = {}
        physics_cfg = self.config.get("physics", {})
        physics_weights = self.get_physics_weights(epoch)

        # 统一退火调度器 — 所有乘子来自同一来源，避免此消彼长
        mult = self._get_phase_mult(epoch)
        stage1_factor = mult["interface"]  # 兼容旧变量名，减少 diff

        # 记录当前 epoch，供各子损失方法读取（用于 S3 权重退火）
        self._current_epoch = epoch

        # 1. 界面数据拟合损失 (核心)
        losses["interface"], data_fit_loss = self._compute_data_loss(data, physics_cfg, stage1_factor)

        # 2. 初始条件 & 壁面边界条件损失
        ib_losses = self._compute_initial_boundary_loss(data, physics_cfg, mult["ic_phi"])
        losses.update(ib_losses)

        # 3. 体积守恒约束
        losses["volume_conservation"] = self._compute_volume_conservation_loss(stage1_factor)

        # 4. phi 空间分布 + 几何一致性约束
        spatial_loss, geom_loss = self._compute_phi_spatial_loss(stage1_factor)
        losses["phi_spatial"] = spatial_loss
        losses["phi_geometry"] = geom_loss

        # 6. 物理方程损失（整体乘子来自统一调度器）
        if any(w > 0 for w in physics_weights.values()):
            # 将 physics 乘子注入 weights，使 _compute_physics_equation_loss 无需改动
            scaled_weights = {k: v * mult["physics"] for k, v in physics_weights.items()}
            phys_losses = self._compute_physics_equation_loss(scaled_weights, epoch, data=data)
            losses.update(phys_losses)

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

    # ========================================================================
    # L-BFGS 微调
    # ========================================================================

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

    # ========================================================================
    # 主训练循环
    # ========================================================================

    def train(self):
        """训练主循环"""
        logger.info("=" * 60)
        logger.info("开始两相流 PINN 训练")
        logger.info("=" * 60)

        data = self.data_generator.generate_all_data()

        # 数据重采样间隔 (0=禁用)
        _train_cfg = self.config.get("training", {})
        resample_interval = _train_cfg.get("resample_interval", 0)  # 默认禁止重采样（训练铁律 R3）

        # 后台重采样线程（避免阻塞训练循环）
        _resample_future = None
        _resample_lock = threading.Lock()
        _resample_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if resample_interval > 0 else None

        # 记录网络结构图到 TensorBoard（用 dummy input 避免依赖 data tensor 维度）
        try:
            dummy_input = torch.zeros(1, 6, device=self.device)
            self.tb_writer.add_graph(self.model, dummy_input)
        except Exception as e:
            logger.warning(f"无法添加网络图到 TensorBoard: {e}")
        start_time = time.time()
        self._consecutive_nan = 0  # 连续 NaN 计数器

        for epoch in range(self.start_epoch, self.epochs):
            # 学习率预热
            if epoch < self.warmup_epochs:
                warmup_lr = self.warmup_start_lr + (self.lr - self.warmup_start_lr) * (epoch / self.warmup_epochs)
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
                    _resample_future = _resample_executor.submit(self.data_generator.generate_all_data)
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

            # [2026-06-11] NaN 诊断：前向传播阶段检查各 loss 项
            if not torch.isfinite(total_loss):
                loss_report = []
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        is_nan = torch.isnan(v).any().item()
                        is_inf = torch.isinf(v).any().item()
                        val = v.item() if not (is_nan or is_inf) else ("NaN" if is_nan else "Inf")
                        loss_report.append(f"  {k}: {val}")
                logger.warning(
                    f"Epoch {epoch}: 前向传播检测到无效总损失 {total_loss.item()}\n" + "\n".join(loss_report)
                )

            # 增加稳定性检查：只有在损失有效且大于 0 时才进行反向传播
            if torch.isfinite(total_loss) and total_loss > 0:
                # 保存当前有效状态，用于 NaN 恢复
                self._last_valid_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                total_loss.backward()

                # [2026-06-11] NaN 诊断：检查反向传播后梯度状态
                grad_nan_params = []
                grad_inf_params = []
                max_grad_norm = 0.0
                max_grad_name = ""
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            grad_nan_params.append(name)
                        if torch.isinf(param.grad).any():
                            grad_inf_params.append(name)
                        gnorm = param.grad.norm().item()
                        if gnorm > max_grad_norm:
                            max_grad_norm = gnorm
                            max_grad_name = name
                if grad_nan_params or grad_inf_params:
                    logger.warning(
                        f"Epoch {epoch}: 反向传播后梯度异常\n"
                        f"  NaN grad in: {grad_nan_params}\n"
                        f"  Inf grad in: {grad_inf_params}\n"
                        f"  Max grad norm: {max_grad_norm:.2e} ({max_grad_name})"
                    )

                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # [2026-06-11] NaN 诊断：裁剪后再次检查
                clip_nan_params = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        clip_nan_params.append(name)
                if clip_nan_params:
                    logger.warning(
                        f"Epoch {epoch}: clip_grad_norm 后梯度仍 NaN: {clip_nan_params}\n"
                        f"  grad_clip={self.grad_clip}, total_norm before clip was {max_grad_norm:.2e}"
                    )

                self.optimizer.step()

                # 检查 step 后是否产生 NaN 权重
                has_nan = any(torch.isnan(p).any() for p in self.model.parameters())
                if not has_nan:
                    self._consecutive_nan = 0  # 正常 step，重置 NaN 计数
                else:
                    # [2026-06-11] NaN 诊断：找出第一个 NaN 参数及其梯度/优化器状态
                    nan_report = []
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param).any():
                            grad_norm = param.grad.norm().item() if param.grad is not None else float("nan")
                            param_max = (
                                param[~torch.isnan(param)].abs().max().item() if (~torch.isnan(param)).any() else 0.0
                            )
                            # Adam 状态
                            opt_state = self.optimizer.state.get(param, {})
                            exp_avg_sq_max = (
                                opt_state.get("exp_avg_sq", torch.tensor([])).abs().max().item()
                                if "exp_avg_sq" in opt_state
                                else 0.0
                            )
                            step_count = opt_state.get("step", 0)
                            nan_report.append(
                                f"  NaN in {name}: shape={list(param.shape)}, "
                                f"grad_norm={grad_norm:.2e}, param_max={param_max:.2e}, "
                                f"exp_avg_sq_max={exp_avg_sq_max:.2e}, step={step_count}"
                            )
                    logger.warning(
                        f"Epoch {epoch}: step 后检测到 NaN 权重，已回滚到上一个有效状态\n" + "\n".join(nan_report)
                    )
                    if hasattr(self, "_last_valid_state"):
                        self.model.load_state_dict(self._last_valid_state)
                    # 清除 Adam 动量状态，防止被污染的动量继续导致 NaN
                    self.optimizer.state.clear()
                    self.optimizer.zero_grad()
                    # 连续 NaN 时自动降学习率，避免反复触发
                    self._consecutive_nan = getattr(self, "_consecutive_nan", 0) + 1
                    if self._consecutive_nan >= 3:
                        min_lr = self.config.get("training", {}).get("min_lr", 1e-6)
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = max(pg["lr"] * 0.5, min_lr)
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        logger.warning(
                            f"Epoch {epoch}: 连续 {self._consecutive_nan} 次 NaN，" f"学习率降至 {current_lr:.2e}"
                        )
                        self._consecutive_nan = 0
            else:
                loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                logger.warning(f"Epoch {epoch}: 检测到无效总损失 {loss_val}, 跳过本次优化步")
                if hasattr(self, "_last_valid_state"):
                    self.model.load_state_dict(self._last_valid_state)
                    self.optimizer.state.clear()
                    logger.warning(f"Epoch {epoch}: 已回滚模型权重并清除优化器状态")

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
                    ("Loss/volume_conservation", "volume_conservation"),
                    ("Loss/pinn_physics", "pinn_physics"),
                    ("Loss/continuity", "pinn_continuity"),
                    ("Loss/vof", "pinn_vof"),
                    ("Loss/momentum_u", "pinn_momentum_u"),
                    ("Loss/momentum_v", "pinn_momentum_v"),
                    ("Loss/momentum_w", "pinn_momentum_w"),
                    ("Loss/laplace_pressure", "pinn_laplace_pressure"),
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
            self.history["pinn_temporal_smoothness"].append(get_val(losses, "pinn_temporal_smoothness"))
            self.history["pinn_laplace_pressure"].append(get_val(losses, "pinn_laplace_pressure"))
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
                        for k in [
                            "pinn_continuity",
                            "pinn_vof",
                            "pinn_momentum_u",
                            "pinn_momentum_v",
                            "pinn_momentum_w",
                        ]
                    )

                # 只有在阶段3（完整物理约束）之后才更新 best_loss
                # 这样确保保存的模型是物理上合理的
                if epoch >= self.best_loss_recording_start:
                    is_best = current_loss < self.best_loss
                    if is_best:
                        self.best_loss = current_loss
                        self.patience_counter = 0
                        logger.info(f"  新最佳损失: {current_loss:.4e} (epoch {epoch})")
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
                if "volume_conservation" in losses:
                    vol_val = losses["volume_conservation"].item()
                    # 体积守恒：显示趋势箭头
                    vol_trend = "↑" if vol_val > 2000 else ("→" if vol_val > 1000 else "↓")
                    physics_str += f" | Vol: {vol_val:.2e}{vol_trend}"
                # 物理场损失 (优先检查 pinn_ 前缀)
                continuity_val = losses.get("pinn_continuity", losses.get("continuity"))
                if continuity_val is not None:
                    physics_str += f" | C: {continuity_val.item():.2e}"

                interface_val = losses.get("interface")
                if interface_val is not None:
                    physics_str += f" | IF: {interface_val.item():.2e}"

                # 新增: Allen-Cahn / Laplace / 界面能 / EW / 润湿BC
                ac_val = losses.get("pinn_vof")
                if ac_val is not None:
                    physics_str += f" | AC: {ac_val.item():.2e}"
                lp_val = losses.get("pinn_laplace_pressure")
                if lp_val is not None:
                    physics_str += f" | LP: {lp_val.item():.2e}"
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
                    evaluator.plot_phi_grid(model, os.path.join(self.output_dir, "phi_grid_evolution.png"))
                    evaluator.plot_interface_3d(
                        model,
                        os.path.join(self.output_dir, "interface_3d_steady.png"),
                        30.0,
                        0.02,
                        30.0,
                    )
                    logger.info(f"✅ 专业评估套件已生成 (model={os.path.basename(eval_ckpt)}): {self.output_dir}/")
        except Exception as e:
            logger.error(f"生成专业评估仪表盘失败: {e}")

        # 关闭 TensorBoard writer
        self.tb_writer.close()
        logger.info(f"TensorBoard 日志已保存到: {os.path.join(self.output_dir, 'runs')}")

        logger.info("=" * 60)
        logger.info(f"训练完成! 最佳损失: {self.best_loss:.6e}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)
