#!/usr/bin/env python3
"""
物理优化采样器（薄包装）
=======================

本模块已重构：PhysicsBasedSampler 和 create_physics_sampling_dataset
已迁移到 src.models.pinn_data_generator 中。

本文件保留为向后兼容的 re-export 薄包装。
新代码请直接从 src.models.pinn_data_generator 导入。

作者: EFD-PINNs Team
日期: 2024-12
重构: 2026-06-18
"""

# 向后兼容：从 pinn_data_generator 重新导出
from src.models.pinn_data_generator import (
    PhysicsBasedSampler,
    create_physics_sampling_dataset,
)

__all__ = ["PhysicsBasedSampler", "create_physics_sampling_dataset"]
