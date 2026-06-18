#!/usr/bin/env python3
"""
PINN 训练器基类
================

从原始 Trainer 类中提取初始化和工具方法，形成 TrainerBase 基类。

包含：
- TrainerBase.__init__(): 模型初始化、优化器、调度器、输出目录、续训逻辑
- _validate_config(): 配置参数范围验证
- _save_checkpoint(): 保存训练检查点
- _plot_curves(): 绘制并保存训练曲线
- get_physics_weights(): 分阶段物理损失权重（平滑渐进）
- _get_phase_mult(): 统一退火调度器
- get_stage1_weight_factor(): 兼容旧接口，返回 interface 乘子

子类（pinn_trainer_loop.py 中的 Trainer）需实现所有 _compute_* 损失方法。

作者: EFD-PINNs Team
日期: 2024-12
重构: 2026-06-18
"""

import datetime
import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import PHYSICS
from src.models.pinn_data_generator import DataGenerator
from src.models.pinn_network import TwoPhasePINN
from src.models.pinn_physics_loss import PhysicsLoss

logger = logging.getLogger("EWP-PINN")

# aperture_model 可选导入
try:
    from src.models.aperture_model import EnhancedApertureModel

    HAS_APERTURE = True
except ImportError:
    HAS_APERTURE = False


class TrainerBase:
    """两相流 PINN 训练器基类 — 初始化 + 调度 + checkpoint"""

    def __init__(self, config: dict[str, Any] | None = None, resume_path: str | None = None):
        # 延迟导入避免循环依赖（pinn_two_phase 从本模块导入 Trainer）
        if config is None:
            from src.models.pinn_two_phase import DEFAULT_CONFIG

            config = DEFAULT_CONFIG
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 配置参数范围验证
        self._validate_config()

        # 模型
        self.model = TwoPhasePINN(self.config).to(self.device)
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        self.data_generator = DataGenerator(self.config, self.device)
        self.physics_loss = PhysicsLoss(self.device)

        # Stage 1 模型 - 使用统一配置路径
        if HAS_APERTURE:
            from src.config import CONFIG_PATH

            self.stage1_model = EnhancedApertureModel(config_path=str(CONFIG_PATH))
        else:
            self.stage1_model = None
        # 训练配置
        training_cfg = self.config.get("training", {})
        self.epochs = training_cfg.get("epochs", 30000)
        self.batch_size = training_cfg.get("batch_size", 4096)
        self.lr = training_cfg.get("learning_rate", 5e-4)
        # NaN 修复：从默认 1.0 改为 0.1，防止 Adam per-parameter 自适应溢出
        # 根因：clip_grad_norm 只裁剪总范数，Adam 对每个参数独立缩放——
        # 某些参数 exp_avg_sq 很小，裁剪后的小梯度除以很小的 exp_avg_sq 仍可溢出
        self.grad_clip = training_cfg.get("gradient_clip", 0.1)

        # 渐进式训练阶段
        self.stage1_epochs = training_cfg.get("stage1_epochs", 5000)
        self.stage2_epochs = training_cfg.get("stage2_epochs", 15000)
        # S3 物理约束平滑过渡跨度 (epoch), 默认5000适配60000epoch训练
        self.s3_smooth_span = float(training_cfg.get("s3_smooth_span", 5000))

        # 优化器
        # [2026-06-11] NaN 修复：Adam eps 从默认 1e-8 增大到 1e-4
        # 根因：clip_grad_norm 只裁剪总范数到 0.1，但 Adam per-parameter 更新步长
        # = lr * m_i / (√v_i + eps)。当某个参数在 S2 新 loss 上梯度很小时 (v_i ≈ 0)，
        # 更新步长 ≈ 0.0003 × 0.1 / 1e-8 = 3000，一次 step 就能让参数溢出。
        # eps=1e-4 把最大更新步长限制到 0.0003 × 0.1 / 1e-4 = 0.3，安全得多。
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-4)
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
            "pinn_temporal_smoothness": [],
            "pinn_laplace_pressure": [],
            "volume": [],
        }
        self.best_loss = float("inf")
        self.best_physics_loss = float("inf")  # 新增：物理损失最佳值
        self.patience_counter = 0
        self._consecutive_nan = 0  # 连续 NaN 计数器
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
            # 续训时保持目标总轮数不变（epochs 是总数，不是新增轮数）
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
            logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(logging.INFO)
        logger.info(f"日志文件: {log_file}")

        # 初始化 TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "runs"))

        # 预先保存一份配置
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def _validate_config(self):
        """验证配置参数范围，提前发现错误"""
        errors = []

        # 训练参数
        training = self.config.get("training", {})
        if training.get("epochs", 0) <= 0:
            errors.append(f"epochs 应为正数，当前 {training.get('epochs')}")
        if training.get("batch_size", 0) <= 0:
            errors.append(f"batch_size 应为正数，当前 {training.get('batch_size')}")
        if training.get("learning_rate", 0) <= 0:
            errors.append(f"learning_rate 应为正数，当前 {training.get('learning_rate')}")

        # 阶段顺序
        s1 = training.get("stage1_epochs", 0)
        s2 = training.get("stage2_epochs", 0)
        s3 = training.get("epochs", 0)
        if not (s1 < s2 < s3):
            errors.append(f"阶段 epoch 应递增: S1={s1}, S2={s2}, S3={s3}")

        # 物理权重非负
        physics = self.config.get("physics", {})
        for key in ["interface_weight", "continuity_weight", "vof_weight", "ns_weight"]:
            if physics.get(key, 0) < 0:
                errors.append(f"{key} 应非负，当前 {physics.get(key)}")

        # 采样参数
        data = self.config.get("data", {})
        n_vert = data.get("n_vertical_samples", 50)
        if n_vert < 2:
            errors.append(f"n_vertical_samples 应 >= 2，当前 {n_vert}")

        # 物理参数
        if PHYSICS.get("V_T_base", 0) < 0:
            errors.append(f"V_T_base 应非负，当前 {PHYSICS['V_T_base']}")
        if PHYSICS.get("tau", 0) <= 0:
            errors.append(f"tau 应为正数，当前 {PHYSICS['tau']}")

        if errors:
            for e in errors:
                logger.warning(f"配置验证: {e}")
            raise ValueError("配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))
        logger.info("✅ 配置验证通过")

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

    def _plot_curves(self, epoch: int | None = None):
        """
        绘制并保存训练曲线 — 所有 loss 在一张图上
        """
        try:
            _fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            ep = self.history["epoch"]
            h = self.history

            # (history_key, label, alpha, lw, color, linestyle)
            curve_specs = [
                ("loss", "Total", 1.0, 2.0, "black", "-"),
                ("interface", "IF(data)", 0.7, 0.8, "C0", "-"),
                ("volume", "Vol", 0.5, 0.6, "C2", "--"),
                ("pinn_continuity", "C(∇·u=0)", 0.6, 0.7, "C3", "-"),
                ("pinn_vof", "AC(vof)", 0.5, 0.6, "C4", ":"),
                ("pinn_momentum_u", "NS_u", 0.3, 0.4, "C5", ":"),
                ("pinn_momentum_w", "NS_w", 0.3, 0.4, "C6", ":"),
                ("pinn_laplace_pressure", "LP", 0.4, 0.5, "C8", ":"),
                ("pinn_temporal_smoothness", "TS", 0.1, 0.3, "C15", ":"),
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

            filename = "training_curve.png" if epoch is None else f"training_curve_epoch_{epoch}.png"
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
                "interface_energy": 0.0,
                "laplace_pressure": 0.0,
                "temporal_smoothness": 0.0,
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
                "interface_energy": physics_cfg.get("interface_energy_weight", 2.0) * smooth_factor * 0.1,
                "laplace_pressure": 0.0,
                "temporal_smoothness": 0.0,
                "sharpening": physics_cfg.get("sharpening_weight", 0.0) * smooth_factor * 0.1,
            }
        # 阶段3：完整物理约束（平滑增加，跨度可配置）
        progress = min(1.0, (epoch - self.stage2_epochs) / max(1.0, self.s3_smooth_span))
        smooth_factor = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))

        return {
            "continuity": physics_cfg.get("continuity_weight", 0.1) * (0.1 + 0.9 * smooth_factor),
            "vof": physics_cfg.get("vof_weight", 0.1) * (0.1 + 0.9 * smooth_factor),
            "ns": physics_cfg.get("ns_weight", 0.01) * smooth_factor,
            "interface_energy": physics_cfg.get("interface_energy_weight", 2.0) * smooth_factor,
            "laplace_pressure": physics_cfg.get("laplace_pressure_weight", 0.2) * smooth_factor,
            "temporal_smoothness": physics_cfg.get("temporal_smoothness_weight", 0.1) * smooth_factor,
            "sharpening": physics_cfg.get("sharpening_weight", 0.0) * smooth_factor,
        }

    def _get_phase_mult(self, epoch: int) -> dict[str, float]:
        """
        统一退火调度器 — 所有损失乘子同向变化，避免此消彼长。

        三个阶段：
          early (S1): 数据拟合主导，物理权重=0
          mid   (S2): 逐渐引入物理约束，数据权重衰减
          late  (S3): 物理约束主导，数据权重保持底线

        返回 dict，包含所有乘子：
          interface, ic_phi, eta_match(已禁用), phi_target3d(已禁用), physics,
          phi_spatial, phi_geometry(新增，替代 target3D)
        """
        s1_end = self.stage1_epochs
        s2_end = self.stage2_epochs

        if epoch < s1_end:
            # === early: 纯数据拟合 ===
            return {
                "interface": 1.0,
                "ic_phi": 1.0,
                "phi_spatial": 1.0,  # spatial 乘子 = interface_mult
                "phi_geometry": 1.0,  # 旧 geo 方案：geometry 乘子 = interface_mult
                "physics": 0.0,
            }

        if epoch < s2_end:
            # === mid: 数据→物理过渡 ===
            p = (epoch - s1_end) / max(s2_end - s1_end, 1)  # 0→1
            # 数据权重：1→0.3（保持底线）
            data_w = 1.0 - 0.7 * p
            # 物理权重：0→0.5
            phys_w = 0.5 * p
            return {
                "interface": data_w,
                "ic_phi": 1.0 - 0.5 * p,  # 1→0.5
                "phi_spatial": data_w,  # 跟随 interface 退火
                "phi_geometry": data_w,  # 跟随 interface 退火
                "physics": phys_w,
            }

        # === late: 物理主导 ===
        s3_anneal_epochs = int(self.config.get("training", {}).get("s3_anneal_span", 15000))
        p = min(1.0, (epoch - s2_end) / max(s3_anneal_epochs, 1))  # 0→1
        # 数据权重：0.3→0.1（保持底线，防止遗忘）
        data_w = 0.3 - 0.2 * p
        # 物理权重：0.5→1.0
        phys_w = 0.5 + 0.5 * p
        return {
            "interface": max(0.05, data_w),  # 最低 0.05
            "ic_phi": 0.5 - 0.2 * p,  # 0.5→0.3
            "phi_spatial": max(0.05, data_w),  # 跟随 interface 退火，最低 0.05
            "phi_geometry": max(0.05, data_w),  # 跟随 interface 退火
            "physics": phys_w,
        }

    def get_stage1_weight_factor(self, epoch: int) -> float:
        """兼容旧接口，返回 interface 乘子"""
        return self._get_phase_mult(epoch)["interface"]
