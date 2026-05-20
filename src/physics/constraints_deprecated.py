"""
归档的物理约束层代码 — 2026-05-19

此文件包含以下从未在训练主循环中被调用的类：
- Swish: 自定义 Swish 激活函数
- ResidualBlock: 残差块
- PINNConstraintLayer: 物理约束层（~675 行，被 PhysicsLoss.compute_total_loss 替代）
- PhysicsEnhancedLoss: 物理增强损失包装器

这些类由 pPhysicConstraintLayer 实例化（pinn_two_phase.py:1575），
但 self.constraint_layer 从未在 compute_losses() 或 train() 中被调用。
实际物理损失通过 PhysicsLoss.compute_total_loss()（调用 compute_core_residuals）计算。

保留此文件仅供参考。如需恢复这些类，请从 git 历史中恢复。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# 注意: 此文件依赖 src.physics.constraints.PhysicsConstraints 和
# src.training.scheduler 中的 DynamicPhysicsWeightScheduler 等类。
# 仅供代码参考，不保证可直接导入运行。

class Swish(nn.Module):
    """Swish激活函数"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return out + residual


class PINNConstraintLayer(nn.Module):
    """
    PINN物理约束层 - 将物理方程集成到神经网络中
    实现方式：
    1. 接收模型预测输出
    2. 计算物理方程残差
    3. 使用残差作为额外约束，优化模型
    4. 支持多种物理方程和权重调整
    5. 集成动态权重调整机制
    """

    def __init__(
        self,
        physics_constraints=None,
        residual_weights=None,
        enable_dynamic_weight=True,
        dynamic_weight_config=None,
        config=None,
    ):
        super(PINNConstraintLayer, self).__init__()
        # 物理约束对象
        self.physics_constraints = physics_constraints or PhysicsConstraints()

        # 优化后的残差权重 - 基于真实器件物理（2024-11-27修正）
        # 参考：generate_pyvista_3d.py 中的器件结构
        self.residual_weights = residual_weights or {
            # === 核心物理约束（最重要）===
            "young_lippmann": 2.0,  # ⬆️ Young-Lippmann方程 - 电润湿核心物理
            "contact_angle_constraint": 2.0,  # ⬆️ 接触角约束 - 边界条件
            "sidewall_contact_angle": 2.0,
            "wall_wetting": 1.0,
            "interface_stability": 0.5,  # 界面稳定性 - 保持平滑
            "volume_conservation": 0.3,  # ⬆️ 体积守恒 - 油墨+极性液体=常数
            # === 界面相关约束 ===
            "interface_curvature": 0.3,  # ⬆️ 界面曲率 - Laplace压力
            "surface_tension": 0.3,  # ⬆️ 表面张力 - 界面能量
            "contact_line_dynamics": 0.2,  # 接触线动力学
            "vof": 0.5,  # [2025-12-31] VOF 方程残差 - 体积分数输运
            # === 时间连续性约束 ===
            "temporal_smoothness": 0.1,  # [2026-01-07] 时间连续性正则化 - 防止流场突变
            # === 简化的流动约束（低Reynolds数）===
            "continuity": 0.1,  # ⬇️ 连续性方程（低Re数不重要）
            "momentum_u": 0.02,  # ⬇️ 动量方程（惯性可忽略）
            "momentum_v": 0.02,  # ⬇️
            "momentum_w": 0.02,  # ⬇️
            "two_phase_continuity": 0.05,  # ⬇️ 双相流连续性（简化）
            "two_phase_momentum_u": 0.01,  # ⬇️ 双相流动量（简化）
            "two_phase_momentum_v": 0.01,  # ⬇️
            "two_phase_momentum_w": 0.01,  # ⬇️
            # === 电学约束 ===
            "dielectric_charge": 0.2,  # ⬇️ 介电层电荷积累
            "frequency_response": 0.15,  # ⬇️ 频率响应
            # === 热学约束 ===
            "thermodynamic": 0.1,  # ⬇️ 热力学（次要）
            # === 光学和能效 ===
            "optical_properties": 0.15,  # ⬇️ 光学特性
            "energy_efficiency": 0.15,  # ⬇️ 能量效率
            # === 需要重新定义的约束 ===
            "volume_consistency": 0.05,  # 体积分数一致性
            "ink_potential_min": 0.0,  # ⬇️ 暂时关闭（定义不清）
            "ink_energy_balance": 0.05,  # ⬇️ 降低（需要重新定义）
            # === 数据拟合 ===
            "data_fit": 1.0,  # 保持
        }

        # 从配置中加载残差权重（如果提供）
        if (
            config is not None
            and "physics" in config
            and "residual_weights" in config["physics"]
        ):
            config_residual_weights = config["physics"]["residual_weights"]
            for key, value in config_residual_weights.items():
                if key in self.residual_weights:
                    logger.info(f"从配置加载残差权重: {key} = {value}")
                    self.residual_weights[key] = value

        # 用于自适应权重调整的历史残差
        self.residual_history = {}
        self.history_length = 10

        # 是否启用自适应权重 - 默认启用
        self.adaptive_weights = True

        # 权重平滑参数 - 增加平滑度，使调整更稳定
        self.weight_smoothing = 0.1  # 增加平滑参数，更稳定的权重调整

        # 权重调整范围限制
        self.min_weight_factor = 0.5  # 最小权重因子
        self.max_weight_factor = 2.0  # 最大权重因子

        # 历史记录长度调整
        self.history_length = 15  # 增加历史记录长度，提高调整精度

        # 全局步数计数器，用于日志记录
        self.global_step = 0

        # 动态权重调整相关
        self.enable_dynamic_weight = enable_dynamic_weight and DYNAMIC_WEIGHT_AVAILABLE
        self.dynamic_weight_scheduler = None
        self.physics_weight_integration = None

        if self.enable_dynamic_weight:
            # 配置动态权重调度器
            dynamic_config = dynamic_weight_config or {}
            self.dynamic_weight_scheduler = DynamicPhysicsWeightScheduler(
                initial_weight=dynamic_config.get("initial_weight", 0.1),
                min_weight=dynamic_config.get("min_weight", 0.01),
                max_weight=dynamic_config.get("max_weight", 5.0),
                adjustment_strategy=dynamic_config.get(
                    "adjustment_strategy", "combined"
                ),
                smoothing_factor=dynamic_config.get("smoothing_factor", 0.9),
                adjustment_interval=dynamic_config.get("adjustment_interval", 100),
                target_loss_ratio=dynamic_config.get("target_loss_ratio", 1.0),
                patience=dynamic_config.get("patience", 500),
                verbose=dynamic_config.get("verbose", True),
            )

            self.physics_weight_integration = PhysicsWeightIntegration(
                weight_scheduler=self.dynamic_weight_scheduler,
                integration_method=dynamic_config.get(
                    "integration_method", "multiplicative"
                ),
            )

            logger.info(
                f"动态物理权重调整已启用，策略: {dynamic_config.get('adjustment_strategy', 'combined')}"
            )
        else:
            logger.info("使用固定物理权重")

    def compute_physics_loss(
        self,
        x_phys,
        model_predictions,
        data_loss=None,
        val_loss=None,
        epoch=None,
        stage=None,
        applied_voltage=None,
        contact_line_velocity=None,
        time=None,
        temperature=None,
        model=None,
    ):
        """
        计算物理约束损失

        参数:
            x_phys: 物理点输入
            model_predictions: 模型预测输出
            data_loss: 数据损失 (可选，用于动态权重调整)
            val_loss: 验证损失 (可选，用于动态权重调整)
            epoch: 当前训练轮次 (可选，用于动态权重调整)
            stage: 当前训练阶段 (可选，用于动态权重调整)
            applied_voltage: 施加的电压 (用于Young-Lippmann方程)
            contact_line_velocity: 接触线速度 (用于接触线动力学约束)
            time: 当前时间 (用于介电层电荷积累约束)
            temperature: 温度场预测 (用于热力学约束)

        返回:
            物理损失和加权残差详情
        """
        # 更新全局步数
        self.global_step += 1

        # [2025-12-31] 使用统一入口 compute_core_residuals 获取核心物理残差
        # 包含: continuity, momentum_u/v/w, vof, surface_tension, volume_conservation 等
        residuals = self.physics_constraints.compute_core_residuals(
            x_phys, model_predictions, model=model
        )

        # 计算Young-Lippmann方程残差（如果提供了电压）
        if applied_voltage is not None:
            yl_residuals = self.physics_constraints.compute_young_lippmann_residual(
                x_phys, model_predictions, applied_voltage
            )
            # 将Young-Lippmann残差合并到总残差中
            residuals.update(yl_residuals)

        # 计算接触线动力学残差（如果提供了速度）
        if contact_line_velocity is not None and hasattr(
            self.physics_constraints, "compute_contact_line_dynamics_residual"
        ):
            cl_residuals = (
                self.physics_constraints.compute_contact_line_dynamics_residual(
                    x_phys, model_predictions, contact_line_velocity
                )
            )
            # 将接触线动力学残差合并到总残差中
            residuals.update(cl_residuals)

        # 计算介电层电荷积累残差（如果提供了电压）
        if applied_voltage is not None and hasattr(
            self.physics_constraints, "compute_dielectric_charge_accumulation_residual"
        ):
            dc_residuals = self.physics_constraints.compute_dielectric_charge_accumulation_residual(
                x_phys, model_predictions, applied_voltage, time
            )
            # 将介电层电荷积累残差合并到总残差中
            residuals.update(dc_residuals)

        # 计算热力学约束残差（如果提供了温度）
        if temperature is not None and hasattr(
            self.physics_constraints, "compute_thermodynamic_residual"
        ):
            try:
                td_residuals = self.physics_constraints.compute_thermodynamic_residual(
                    x_phys, model_predictions, temperature, applied_voltage
                )
                # 将热力学残差合并到总残差中
                residuals.update(td_residuals)
            except Exception as e:
                logger.error(f"计算热力学约束残差失败: {str(e)}")

        # 计算界面稳定性约束残差
        if hasattr(self.physics_constraints, "compute_interface_stability_residual"):
            try:
                is_residuals = (
                    self.physics_constraints.compute_interface_stability_residual(
                        x_phys, model_predictions
                    )
                )
                # 将界面稳定性残差合并到总残差中
                residuals.update(is_residuals)
            except Exception as e:
                logger.error(f"计算界面稳定性约束残差失败: {str(e)}")

        # 计算频率响应约束残差
        if hasattr(self.physics_constraints, "compute_frequency_response_residual"):
            try:
                fr_residuals = (
                    self.physics_constraints.compute_frequency_response_residual(
                        x_phys, model_predictions, applied_voltage=applied_voltage
                    )
                )
                # 将频率响应残差合并到总残差中
                residuals.update(fr_residuals)
            except Exception as e:
                logger.error(f"计算频率响应约束残差失败: {str(e)}")

        # 计算光学特性约束残差
        if hasattr(self.physics_constraints, "compute_optical_properties_residual"):
            try:
                op_residuals = (
                    self.physics_constraints.compute_optical_properties_residual(
                        x_phys, model_predictions
                    )
                )
                # 将光学特性残差合并到总残差中
                residuals.update(op_residuals)
            except Exception as e:
                logger.error(f"计算光学特性约束残差失败: {str(e)}")

        # 计算能量效率约束残差
        if hasattr(self.physics_constraints, "compute_energy_efficiency_residual"):
            try:
                ee_residuals = (
                    self.physics_constraints.compute_energy_efficiency_residual(
                        x_phys, model_predictions, applied_voltage=applied_voltage
                    )
                )
                # 将能量效率残差合并到总残差中
                residuals.update(ee_residuals)
            except Exception as e:
                logger.error(f"计算能量效率约束残差失败: {str(e)}")

        if hasattr(self.physics_constraints, "compute_wall_wetting_residual"):
            try:
                ww_residuals = self.physics_constraints.compute_wall_wetting_residual(
                    x_phys, model_predictions
                )
                residuals.update(ww_residuals)
            except Exception as e:
                logger.error(f"计算墙润湿残差失败: {str(e)}")

        # 计算体积守恒与体积分数一致性（若可用）
        if hasattr(self.physics_constraints, "compute_volume_conservation_residual"):
            try:
                vc_residuals = (
                    self.physics_constraints.compute_volume_conservation_residual(
                        x_phys, model_predictions
                    )
                )
                residuals.update(vc_residuals)
            except Exception as e:
                logger.error(f"计算体积守恒残差失败: {str(e)}")

        # 计算 Laplace 压力一致约束（若可用）
        if hasattr(self.physics_constraints, "compute_laplace_pressure_residual"):
            try:
                lp_residuals = (
                    self.physics_constraints.compute_laplace_pressure_residual(
                        x_phys, model_predictions
                    )
                )
                residuals.update(lp_residuals)
            except Exception as e:
                logger.error(f"计算 Laplace 压力残差失败: {str(e)}")

        # 计算双相流Navier-Stokes方程残差（若可用）
        if hasattr(self.physics_constraints, "compute_two_phase_flow_residual"):
            try:
                tp_residuals = self.physics_constraints.compute_two_phase_flow_residual(
                    x_phys, model_predictions
                )
                residuals.update(tp_residuals)
            except Exception as e:
                logger.error(f"计算双相流残差失败: {str(e)}")

        # 计算表面张力和接触角动态残差（若可用）
        if hasattr(self.physics_constraints, "compute_surface_tension_residual"):
            try:
                st_residuals = (
                    self.physics_constraints.compute_surface_tension_residual(
                        x_phys, model_predictions
                    )
                )
                residuals.update(st_residuals)
            except Exception as e:
                logger.error(f"计算表面张力残差失败: {str(e)}")

        if hasattr(self.physics_constraints, "compute_sidewall_contact_angle_residual"):
            try:
                sw_residuals = (
                    self.physics_constraints.compute_sidewall_contact_angle_residual(
                        x_phys, model_predictions
                    )
                )
                residuals.update(sw_residuals)
            except Exception as e:
                logger.error(f"计算侧墙接触角残差失败: {str(e)}")

        # 计算油墨势能最小化残差（若可用）
        if hasattr(self.physics_constraints, "compute_ink_potential_residual"):
            try:
                ip_residuals = self.physics_constraints.compute_ink_potential_residual(
                    x_phys, model_predictions
                )
                residuals.update(ip_residuals)
            except Exception as e:
                logger.error(f"计算油墨势能残差失败: {str(e)}")

        # 计算加权残差损失
        physics_loss = 0.0
        weighted_residuals = {}

        # 记录残差统计信息
        residual_stats = {}

        for key, residual in residuals.items():
            if key in self.residual_weights:
                # 计算残差统计信息
                residual_mean = torch.mean(residual).item()
                residual_std = torch.std(residual).item()
                residual_min = torch.min(residual).item()
                residual_max = torch.max(residual).item()
                residual_abs_mean = torch.mean(torch.abs(residual)).item()

                residual_stats[key] = {
                    "mean": residual_mean,
                    "std": residual_std,
                    "min": residual_min,
                    "max": residual_max,
                    "abs_mean": residual_abs_mean,
                }

                # 获取自适应权重
                weight = self._get_current_weight(key, residual)

                # 计算平方残差均值
                raw_residual_squared = torch.mean(residual**2)
                weighted_loss = weight * raw_residual_squared

                # 累加总损失
                physics_loss += weighted_loss
                if self.global_step <= 3:
                    print(
                        f"[DEBUG PINN] key={key} | residual_mean={residual_mean:.6f} | residual_std={residual.std().item():.6f} | residual_max={residual.abs().max().item():.6f} | weight={weight:.6f} | weighted_loss={weighted_loss.item():.6f}",
                        flush=True,
                    )

                # 记录加权残差信息
                weighted_residuals[key] = {
                    "loss": weighted_loss.item(),
                    "weight": weight,
                    "raw_value": raw_residual_squared.item(),
                }

        # 应用动态权重调整（如果启用）
        if self.enable_dynamic_weight and data_loss is not None:
            physics_loss = self.physics_weight_integration.apply_dynamic_weight(
                physics_loss, data_loss, val_loss, epoch, stage
            )

            # 更新加权残差信息以反映动态权重
            current_dynamic_weight = (
                self.physics_weight_integration.get_current_weight()
            )
            for key in weighted_residuals:
                weighted_residuals[key]["dynamic_weight"] = current_dynamic_weight
                weighted_residuals[key]["dynamic_loss"] = (
                    weighted_residuals[key]["loss"] * current_dynamic_weight
                )

        # 添加详细的调试日志
        if self.global_step % 50 == 0:
            logger.info(f"📊 物理损失计算详情 (步骤 {self.global_step}):")
            logger.info(f"  物理点数量: {x_phys.shape[0]}")
            if self.enable_dynamic_weight:
                logger.info(
                    f"  动态权重: {self.physics_weight_integration.get_current_weight():.6f}"
                )
            logger.info("  残差项统计:")
            for key, stats in residual_stats.items():
                logger.info(f"    {key}:")
                logger.info(
                    f"      均值: {stats['mean']:.6f}, 标准差: {stats['std']:.6f}"
                )
                logger.info(
                    f"      最小值: {stats['min']:.6f}, 最大值: {stats['max']:.6f}"
                )
                logger.info(f"      绝对均值: {stats['abs_mean']:.6f}")
            logger.info("  加权损失详情:")
            for key, info in weighted_residuals.items():
                if "dynamic_loss" in info:
                    logger.info(
                        f"    {key}: 权重={info['weight']:.4f}, 动态权重={info['dynamic_weight']:.4f}, "
                        f"原始值={info['raw_value']:.6f}, 加权损失={info['loss']:.6f}, "
                        f"动态损失={info['dynamic_loss']:.6f}"
                    )
                else:
                    logger.info(
                        f"    {key}: 权重={info['weight']:.4f}, 原始值={info['raw_value']:.6f}, "
                        f"加权损失={info['loss']:.6f}"
                    )
            # 确保类型安全，只有当physics_loss是tensor时才调用.item()
            physics_loss_value = (
                physics_loss.item()
                if isinstance(physics_loss, torch.Tensor)
                else physics_loss
            )
            logger.info(f"  总物理损失: {physics_loss_value:.6f}")
            if self.global_step <= 3:
                print(
                    f"[DEBUG PINN] global_step={self.global_step} | total_physics_loss={physics_loss_value:.6f}",
                    flush=True,
                )

        return physics_loss, weighted_residuals

    def _get_current_weight(self, residual_key, current_residual):
        """获取当前残差项的权重（支持自适应调整）"""
        base_weight = self.residual_weights.get(residual_key, 1.0)

        if not self.adaptive_weights:
            return base_weight

        # 计算当前残差值的均值
        current_value = torch.mean(current_residual**2).item()

        # 更新历史记录
        if residual_key not in self.residual_history:
            self.residual_history[residual_key] = []

        self.residual_history[residual_key].append(current_value)

        # 保持历史长度
        if len(self.residual_history[residual_key]) > self.history_length:
            self.residual_history[residual_key] = self.residual_history[residual_key][
                -self.history_length :
            ]

        # 如果历史记录不足，返回基础权重
        if (
            len(self.residual_history[residual_key]) < 8
        ):  # 增加历史记录要求，使调整更稳定
            return base_weight

        # 计算历史均值和当前值的比值
        history_mean = np.mean(self.residual_history[residual_key][:-1])

        if history_mean > 0:
            ratio = current_value / history_mean

            # 动态调整权重 - 残差增大时增加权重，残差减小时减小权重
            # 使用更平滑的调整方式，增加指数衰减
            adaptive_factor = 1.0 + self.weight_smoothing * np.sign(
                ratio - 1.0
            ) * np.sqrt(abs(ratio - 1.0))

            # 使用新增的权重范围限制参数
            adaptive_factor = max(
                self.min_weight_factor, min(self.max_weight_factor, adaptive_factor)
            )

            # 增加权重变化的平滑性 - 指数平滑
            smoothed_factor = adaptive_factor
            if len(self.residual_history[residual_key]) > 10:
                # 最近几次的权重因子进行平滑
                recent_ratios = []
                for i in range(1, min(6, len(self.residual_history[residual_key]))):
                    prev_val = self.residual_history[residual_key][-i - 1]
                    if prev_val > 0:
                        recent_ratio = (
                            self.residual_history[residual_key][-i] / prev_val
                        )
                        recent_ratios.append(
                            1.0
                            + self.weight_smoothing
                            * np.sign(recent_ratio - 1.0)
                            * np.sqrt(abs(recent_ratio - 1.0))
                        )

                if recent_ratios:
                    # 使用指数加权平均平滑
                    weights = np.exp(
                        -np.arange(len(recent_ratios)) * 0.3
                    )  # 指数衰减权重
                    weights /= weights.sum()
                    smoothed_factor = np.sum(np.array(recent_ratios) * weights)
                    smoothed_factor = max(
                        self.min_weight_factor,
                        min(self.max_weight_factor, smoothed_factor),
                    )

            adjusted_weight = base_weight * smoothed_factor

            # 记录权重调整信息（便于调试）
            if hasattr(self, "weight_adjustment_history"):
                if residual_key not in self.weight_adjustment_history:
                    self.weight_adjustment_history[residual_key] = []
                self.weight_adjustment_history[residual_key].append(
                    {
                        "current_value": current_value,
                        "history_mean": history_mean,
                        "ratio": ratio,
                        "adaptive_factor": adaptive_factor,
                        "smoothed_factor": smoothed_factor,
                        "adjusted_weight": adjusted_weight,
                    }
                )
                # 限制历史记录长度
                if len(self.weight_adjustment_history[residual_key]) > 50:
                    self.weight_adjustment_history[residual_key] = (
                        self.weight_adjustment_history[residual_key][-50:]
                    )

            return adjusted_weight

        return base_weight

    def __call__(self, *args, **kwargs):
        """
        兼容多种调用方式的统一接口
        支持：
        1. 2参数调用: constraint_layer(physics_points, model_predictions)
        2. 4参数调用: constraint_layer(x_data, x_phys, model_predictions, true_labels)
        """
        # 2参数调用（训练时使用）
        if len(args) == 2 and not kwargs:
            physics_points, model_predictions = args
            # 计算简化的物理约束
            return self._compute_simple_constraint(physics_points, model_predictions)

        # 4+参数调用（完整功能）
        else:
            return self.forward(*args, **kwargs)

    def _compute_simple_constraint(self, physics_points, model_predictions):
        """
        简化的物理约束计算（用于训练循环）
        只计算梯度平滑性约束，确保快速且稳定
        """
        device = physics_points.device

        # 确保输入需要梯度
        if not physics_points.requires_grad:
            physics_points = physics_points.clone().requires_grad_(True)

        try:
            # 计算梯度
            grad_outputs = torch.ones_like(model_predictions)
            gradients = torch.autograd.grad(
                outputs=model_predictions,
                inputs=physics_points,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if gradients is None:
                # 如果梯度计算失败，使用预测值的方差
                return torch.var(model_predictions) * 0.1

            # 梯度平滑性约束
            grad_smoothness = torch.mean(gradients**2)

            # 输出合理性约束
            output_penalty = torch.mean(torch.abs(model_predictions)) * 0.01

            # 梯度一致性约束
            if gradients.shape[1] >= 3:
                grad_diff = torch.std(gradients[:, :3], dim=1)
                grad_consistency = torch.mean(grad_diff)
            else:
                grad_consistency = torch.tensor(0.0, device=device)

            # 组合约束
            total_constraint = grad_smoothness + output_penalty + grad_consistency * 0.1

            return total_constraint

        except Exception as e:
            logger.error(f"简化约束计算失败: {e}")
            return torch.tensor(0.1, device=device, requires_grad=True)

    def forward(
        self,
        x_data,
        x_phys,
        model_predictions,
        true_labels=None,
        applied_voltage=None,
        contact_line_velocity=None,
        time=None,
        temperature=None,
    ):
        """
        前向传播 - 计算总损失（完整功能）
        输入:
        - x_data: 数据点输入
        - x_phys: 物理点输入
        - model_predictions: 模型预测输出
        - true_labels: 真实标签（可选）
        - applied_voltage: 施加的电压（用于Young-Lippmann方程）
        - contact_line_velocity: 接触线速度（用于接触线动力学约束）
        - time: 当前时间（用于介电层电荷积累约束）
        - temperature: 温度分布（用于热力学约束）

        返回:
        - total_loss: 总损失
        - loss_components: 各部分损失的详细信息
        """
        loss_components = {}
        total_loss = 0.0

        # 计算物理约束损失
        physics_loss, weighted_residuals = self.compute_physics_loss(
            x_phys,
            model_predictions,
            applied_voltage=applied_voltage,
            contact_line_velocity=contact_line_velocity,
            time=time,
            temperature=temperature,
        )
        loss_components["physics"] = weighted_residuals
        total_loss += physics_loss

        # 计算数据拟合损失（如果提供了真实标签）
        if true_labels is not None:
            data_loss = self.residual_weights.get("data_fit", 1.0) * torch.mean(
                (model_predictions - true_labels) ** 2
            )
            loss_components["data_fit"] = {
                "loss": data_loss.item(),
                "weight": self.residual_weights.get("data_fit", 1.0),
            }
            total_loss += data_loss

        return total_loss, loss_components


class PhysicsEnhancedLoss(nn.Module):
    """物理增强损失函数 - 用于EWPINN模型"""

    def __init__(self, pinn_layer=None, alpha=0.001, model_parameters=None):
        super(PhysicsEnhancedLoss, self).__init__()
        self.pinn_layer = pinn_layer or PINNConstraintLayer()
        self.alpha = alpha
        self.alpha_decay = 0.999
        self.global_step = 0
        self.loss_clipping = 1e5
        self.use_log_scaling = False
        self.model_parameters = model_parameters  # 添加模型参数用于正则化

    def safe_loss_computation(self, loss_tensor, name=""):
        """安全的损失计算，防止数值不稳定"""
        try:
            # 不再为物理损失设置固定值，而是使用实际计算的物理损失
            if name == "物理":
                logger.debug("处理物理损失 - 使用实际计算值")
                # 继续正常处理流程，确保物理约束的正确性

            # 检查输入是否有效
            if loss_tensor is None:
                logger.warning(f"{name}损失张量为None")
                return torch.tensor(1e-6, requires_grad=True)

            # 确保输入是Tensor类型
            if isinstance(loss_tensor, (int, float)):
                # 对于标量值，直接检查并返回安全值
                if np.isnan(loss_tensor) or np.isinf(loss_tensor):
                    logger.warning(f"{name}损失包含NaN或无穷大值，替换为安全值")
                    return torch.tensor(1e-6, requires_grad=True)
                return torch.tensor(float(loss_tensor), requires_grad=True)

            # 对于Tensor类型的处理，确保保留梯度
            if not isinstance(loss_tensor, torch.Tensor):
                logger.error(f"{name}损失类型错误: {type(loss_tensor)}")
                return torch.tensor(1e-6, requires_grad=True)

            # 检查是否为无穷大或NaN
            if torch.any(torch.isnan(loss_tensor)) or torch.any(
                torch.isinf(loss_tensor)
            ):
                logger.warning(f"{name}损失包含NaN或无穷大值，替换为安全值")
                # 替换NaN和无穷大值
                replacement_value = torch.tensor(
                    1e-6, device=loss_tensor.device, requires_grad=True
                )
                loss_tensor = torch.where(
                    torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
                    replacement_value,
                    loss_tensor,
                )

            # 确保损失值不会太小
            loss_tensor = torch.clamp(loss_tensor, min=1e-8, max=1e6)

            return loss_tensor

        except Exception as e:
            logger.error(f"{name}损失计算异常: {str(e)}")
            # 异常情况下返回一个更大的值，确保训练过程不会被卡住
            return torch.tensor(1.0, requires_grad=True)

    def forward(
        self,
        x_data,
        x_phys,
        predictions,
        targets=None,
        applied_voltage=None,
        contact_line_velocity=None,
        time=None,
        temperature=None,
    ):
        """计算物理增强的损失"""
        # 更新全局步数
        self.global_step += 1
        # 更新物理约束层的全局步数，确保日志记录正确触发
        self.pinn_layer.global_step = self.global_step

        # 改进的物理权重策略 - 更快地达到目标值
        target_alpha = self.alpha
        # 使用更快的增长策略，100步后达到目标值的50%，500步后完全达到目标值
        if self.global_step < 100:
            current_alpha = target_alpha * (self.global_step / 100)
        elif self.global_step < 500:
            current_alpha = target_alpha * (0.5 + 0.5 * (self.global_step - 100) / 400)
        else:
            current_alpha = target_alpha

        # 计算物理约束损失
        # 使用统一的 helper 提取预测张量以避免多个提取点导致的不一致
        try:
            from src.utils.model_utils import extract_predictions

            main_predictions = extract_predictions(predictions)
        except Exception:
            # 回退到兼容逻辑
            if isinstance(predictions, dict):
                main_predictions = predictions.get("main_predictions", predictions)
            elif isinstance(predictions, torch.Tensor):
                main_predictions = predictions
            else:
                main_predictions = predictions

        physics_loss, physics_components = self.pinn_layer.compute_physics_loss(
            x_phys,
            main_predictions,
            applied_voltage=applied_voltage,
            contact_line_velocity=contact_line_velocity,
            time=time,
            temperature=temperature,
        )

        # 安全处理物理损失 - 添加额外的裁剪
        physics_loss = self.safe_loss_computation(physics_loss, "物理")

        # 计算数据损失（使用上面提取到的 main_predictions）
        data_loss = 0.0
        if targets is not None:
            try:
                # 确保main_predictions是张量类型
                if isinstance(main_predictions, torch.Tensor):
                    # 计算MSE，但先使用对异常值更鲁棒的 Huber/smooth_l1
                    if main_predictions.numel() > 0 and targets.numel() > 0:
                        data_loss = F.smooth_l1_loss(
                            main_predictions, targets, reduction="mean"
                        )
                    else:
                        data_loss = torch.tensor(0.0, device=physics_loss.device)
                else:
                    logger.warning(
                        "提取到的 main_predictions 不是张量类型，无法计算数据损失"
                    )
                    data_loss = torch.tensor(0.0, device=physics_loss.device)
            except Exception as e:
                logger.error(f"数据损失计算失败: {str(e)}")
                data_loss = torch.tensor(0.0, device=physics_loss.device)

        # 安全处理数据损失
        data_loss = self.safe_loss_computation(data_loss, "数据")

        # 添加梯度正则化，防止参数过大
        reg_loss = torch.tensor(0.0, device=data_loss.device)
        if hasattr(self, "model_parameters") and self.model_parameters is not None:
            for param in self.model_parameters:
                reg_loss += 0.0001 * torch.norm(param, p=2)

        physics_contribution = current_alpha * physics_loss
        total_loss = data_loss + physics_contribution + reg_loss

        # 记录损失统计信息 - 增加详细的物理损失日志
        if self.global_step % 50 == 0:
            logger.info(f"📊 损失统计 - 步骤 {self.global_step}:")
            logger.info(f"  数据损失: {data_loss.item():.6f}")
            logger.info(f"  物理损失: {physics_loss.item():.6f}")
            logger.info(f"  物理贡献: {physics_contribution.item():.6f}")
            logger.info(f"  正则化损失: {reg_loss.item():.6f}")
            logger.info(f"  物理权重: {current_alpha:.8f}")

            # 添加物理组件的详细日志
            if physics_components is not None and isinstance(physics_components, dict):
                logger.info("  物理约束组件详情:")
                for comp_name, comp_value in physics_components.items():
                    if isinstance(comp_value, dict):
                        loss_val = comp_value.get("loss", 0.0)
                        weight_val = comp_value.get("weight", 1.0)
                        raw_val = comp_value.get("raw_value", 0.0)
                        logger.info(
                            f"    {comp_name}: 权重={weight_val:.4f}, 原始值={raw_val:.6f}, 加权损失={loss_val:.6f}"
                        )
                    else:
                        logger.info(f"    {comp_name}: {comp_value:.6f}")

            # 计算损失分布比例
            total_loss_value = total_loss.item()
            if total_loss_value > 1e-10:  # 避免除零
                data_ratio = (data_loss.item() / total_loss_value) * 100
                physics_ratio = (physics_contribution.item() / total_loss_value) * 100
                reg_ratio = (reg_loss.item() / total_loss_value) * 100
                logger.info("  损失分布比例:")
                logger.info(f"    数据损失: {data_ratio:.2f}%")
                logger.info(f"    物理贡献: {physics_ratio:.2f}%")
                logger.info(f"    正则化损失: {reg_ratio:.2f}%")

        return {
            "total": total_loss,
            "data": data_loss,
            "physics": physics_loss,
            "physics_contribution": physics_contribution,
            "regularization": reg_loss,
            "physics_components": physics_components,
            "alpha": current_alpha,
        }
