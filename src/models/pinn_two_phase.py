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

模块结构（重构后）：
- pinn_network.py: FourierFeature + TwoPhasePINN
- pinn_physics_loss.py: PhysicsLoss
- pinn_data_generator.py: PhysicsBasedSampler + DataGenerator + _sample_bc_z
- pinn_trainer_loop.py: Trainer（损失计算 + train）

本文件为向后兼容的 re-export 入口 + main() 函数。

作者: EFD-PINNs Team
日期: 2024-12
"""

import argparse
import json
import logging
import os
import random

import torch

from src.config import PHYSICS, get_default_training_config


def set_seed(seed: int = 42, deterministic: bool = False):
    """设置全局随机种子，确保训练可复现。

    Args:
        seed: 随机种子
        deterministic: 是否强制确定性计算（降低性能但可复现）。
                       默认 False，启用 cudnn.benchmark 加速 GPU 运算。
    """
    random.seed(seed)
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

# ============================================================================
# 默认配置（单一来源：config/device_calibrated_physics.json）
# ============================================================================

DEFAULT_CONFIG = get_default_training_config()

# ============================================================================
# Re-export（向后兼容）
# ============================================================================

from .pinn_data_generator import DataGenerator, PhysicsBasedSampler
from .pinn_network import FourierFeature, TwoPhasePINN
from .pinn_physics_loss import PhysicsLoss
from .pinn_trainer_loop import Trainer

__all__ = [
    "DEFAULT_CONFIG",
    "PHYSICS",
    "DataGenerator",
    "FourierFeature",
    "PhysicsBasedSampler",
    "PhysicsLoss",
    "Trainer",
    "TwoPhasePINN",
    "main",
    "set_seed",
]

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
