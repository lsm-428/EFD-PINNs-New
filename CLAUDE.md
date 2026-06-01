# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EFD3D 是一个基于物理信息神经网络 (PINN) 的 3D 电润湿两相流仿真框架。采用 VOF 方法追踪油墨-极性液体界面，结合 Navier-Stokes 方程和电润湿力场建模。核心创新是 **6D Triad 输入** `(x, y, z, V_from, V_to, t_since)`，支持单一模型模拟任意电压序列。

## Commands

```bash
# 训练（推荐配置）
uv run train_two_phase.py --config config/v4.5-standard.json

# 从检查点恢复
uv run train_two_phase.py --config config/v4.5-standard.json --resume_from outputs/train/pinn_YYYYMMDD_HHMMSS/best_model.pth

# 评估与可视化
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/

# 启动 Streamlit 仪表板
uv run scripts/dashboard.py

# 运行所有测试
uv run pytest tests/ -v

# 运行单个测试
uv run pytest tests/test_pinn_complete.py -v

# 覆盖率报告
uv run pytest tests/ --cov=src --cov-report=html

# 消融研究
uv run scripts/run_ablation.sh

# Lint
uv run ruff check src/ tests/
```

## Architecture

### 两阶段设计

- **Stage 1 (解析模型)**: `HybridPredictor` (电压→接触角, Young-Lippmann方程) + `EnhancedApertureModel` (接触角→开口率). 已校准，为 Stage 2 提供监督信号 (eta tutor).
- **Stage 2 (PINN)**: `TwoPhasePINN` 求解完整流场 — Navier-Stokes + VOF 界面追踪. 输出 `(u, v, w, p, phi)`.

### 核心模块 API 速查

> **⚠️ 修改代码时，必须检查所有相关模块，不只是 `pinn_two_phase.py`。**
> `src/` 下共 42 个 Python 文件，分属 7 个子包。

#### 📁 完整模块清单

```
src/
├── config/              # 物理参数 + 路径管理
│   ├── __init__.py      # 导出 PHYSICS, PhysicsConfig, get_physics_config, get_materials_params
│   ├── physics_config.py    # ⭐ 物理参数唯一来源（三层体系）
│   └── paths.py         # 路径常量 + get_config_path(), get_output_dir()
├── models/              # 神经网络模型
│   ├── pinn_two_phase.py    # ⭐ TwoPhasePINN + PhysicsLoss + DataGenerator + Trainer
│   └── aperture_model.py    # Stage 1: ApertureModel + EnhancedApertureModel
├── physics/             # 物理约束引擎
│   ├── constraints.py       # ⭐ PhysicsConstraints（~3441行，所有PDE残差）
│   └── constraints_deprecated.py  # 旧版（不活跃）
├── predictors/          # Stage 1 预测器
│   ├── hybrid_predictor.py  # HybridPredictor: 电压→接触角 (Young-Lippmann)
│   └── pinn_aperture.py     # PINNAperturePredictor
├── training/            # 训练基础设施
│   ├── components.py    # DataNormalizer, LossStabilizer, EnhancedDataAugmenter
│   ├── scheduler.py     # DynamicPhysicsWeightScheduler, PhysicsWeightIntegration
│   └── stabilizer.py    # TrainingStabilizer (NaN恢复/预热/梯度裁剪)
├── data/                # 数据采样
│   └── physics_sampling.py  # PhysicsBasedSampler
├── utils/               # 工具函数
│   ├── logging_config.py    # setup_logging(), get_logger()
│   └── model_utils.py       # extract_predictions(), load_model_with_mismatch_handling()
├── solvers/             # CFD求解器
│   └── flow_solver.py       # Mesh, FlowSolver, PINNSolver, FlowFieldSimulator
└── dashboard/           # Streamlit交互式面板（10个文件）
```

---

#### `src/config/physics_config.py` — ⭐ 物理参数唯一来源
| 行号 | 类/函数 | 用途 |
|------|---------|------|
| 47 | `DEFAULT_CONFIG_PATH` | 指向 `config/device_calibrated_physics.json` |
| 55 | `PHYSICS: Dict` | 全局物理参数字典（代码内嵌默认值，模块加载时从JSON覆盖） |
| 122 | `PhysicsConfig` dataclass | 类型安全的物理参数类 |
| 196 | `PhysicsConfig.V_threshold` | property，从 V_T_base + h_ink 动态计算 |
| 201 | `PhysicsConfig.tau_recovery` | property，tau × tau_recovery_factor |
| 206 | `PhysicsConfig.from_json(path)` | 从 JSON 文件加载配置 |
| 278 | `PhysicsConfig.to_dict()` | 转 PHYSICS 格式（含别名） |
| 294 | `PhysicsConfig.to_materials_params()` | 转 PhysicsConstraints 兼容格式 |
| 334 | `PhysicsConfig.to_predictor_params()` | 转 HybridPredictor 兼容格式 |
| 362 | `PhysicsConfig.update_global_physics()` | 更新全局 PHYSICS |
| 380 | `get_physics_config(path)` | 获取 PhysicsConfig 实例（带缓存） |
| 409 | `get_materials_params(path)` | 获取 materials_params 字典 |

**三层参数体系**（修改参数时必须三层同步）：
1. `PHYSICS` dict (L55) — 代码内嵌默认值
2. `config/device_calibrated_physics.json` — 实验校准值
3. `constraints.py` → `_get_default_materials_params()` — 回退硬编码

---

#### `src/physics/constraints.py` — 物理约束引擎（~3441行）
| 行号 | 函数/方法 | 用途 | 活跃？ |
|------|-----------|------|--------|
| 38 | `_get_default_materials_params()` | 回退硬编码材料参数 | ✅ |
| 122 | `compute_navier_stokes_residual()` | NS方程（连续性+动量+EWF+CSF）| ✅ |
| 366 | `_compute_laplacian()` | 标量场 Laplacian | ✅ |
| 405 | `compute_volume_conservation_residual()` | 体积守恒 | ✅ |
| 453 | `_compute_laplacian_spatial()` | 空间 Laplacian | ✅ |
| 498 | `compute_surface_tension_residual()` | CSF表面张力+曲率 | 🔴 DEPRECATED |
| 577 | `compute_volume_conservation_full()` | 完整体积守恒 | 🔴 DEPRECATED |
| 638 | `compute_two_phase_flow_residual()` | 简化版NS两相流 | 🔴 DEPRECATED |
| 728 | `compute_ink_potential_residual()` | 油墨电势 | 🔴 DEPRECATED |
| 801 | `compute_sidewall_contact_angle_residual()` | 壁面接触角 | ✅ |
| 965 | `compute_laplace_pressure_residual() ` | Laplace压力一致 | ✅ |
| 1062 | `compute_electrowetting_residual()` | 电润湿变分残差 | ✅ |
| 1149 | `compute_interface_energy_residual()` | 界面能 σ\|∇φ\| | ✅ |
| 1177 | `compute_wall_wetting_residual()` | 壁面润湿 | ✅ |
| 1240 | `_compute_unified_wetting_bc()` | 统一相场润湿BC | 开关 |
| 1369 | `_compute_dielectric_charge_residual()` | 介电层RC充电 | ⚠️ 零 |
| 1428 | `_compute_contact_line_dynamics_residual()` | 接触线动力学 | ⚠️ 零 |
| 1505 | `_compute_top_boundary_residual()` | 顶面自由表面BC | ⚠️ 零 |
| 1558 | `safe_compute_laplacian_spatial()` | 安全空间Laplacian | ✅ |
| 1581 | `safe_compute_gradient()` | 安全梯度计算 | ✅ |
| 1610 | **`compute_core_residuals()`** | **统一入口，调用所有活跃约束** | ✅ |
| 1764 | `_compute_temporal_smoothness()` | 时间连续性正则化 | ✅ |
| 1860 | `_compute_vof_residual()` | Allen-Cahn相场方程 | ✅ |
| 1948 | `safe_compute_laplacian()` | 安全Laplacian | ✅ |
| 1968 | `safe_compute_hessian()` | 安全Hessian | ✅ |
| 1991 | `compute_young_lippmann_residual()` | Young-Lippmann接触角 | ✅ |
| 2113 | `compute_contact_line_dynamics_residual()` | 接触线动力学(Hoffman) | ✅ |
| 2239 | `compute_dielectric_charge_accumulation_residual()` | 介电电荷积累 | ✅ |
| 2387 | `compute_thermodynamic_residual()` | 热力学一致性 | ✅ |
| 2607 | `compute_interface_stability_residual()` | 界面稳定性 | ✅ |
| 2791 | `compute_frequency_response_residual()` | 频响残差 | ✅ |

---

#### `src/models/pinn_two_phase.py` — 主PINN模型（~3474行）
| 行号 | 类/方法 | 用途 |
|------|---------|------|
| 43 | `set_seed(seed)` | 全局随机种子 |
| 150 | `FourierFeature` | Fourier特征映射 (x,y,z→96) |
| 184 | `TwoPhasePINN.__init__()` | phi_net[128,128,64,32] + vel_net[64,64,32] |
| 250 | `TwoPhasePINN.forward()` | (B,6)→(B,5): u,v,w,p,phi |
| 330 | `TwoPhasePINN.forward_triplet()` | 三时间点前向，用于时间导数 |
| 375 | `PhysicsLoss.__init__()` | 加载参数，注册权重 |
| 453 | `PhysicsLoss.compute_all_residuals()` | 调用 compute_core_residuals |
| 479 | `PhysicsLoss.explicit_volume_conservation_loss()` | 显式体积守恒 |
| 504 | `PhysicsLoss.compute_total_loss()` | 加权总loss（log1p缩放+自适应归一化） |
| 612 | `PhysicsLoss.compute_gradients()` | 计算所有梯度（含二阶） |
| 678 | `PhysicsLoss.continuity_residual()` | ∇·u = 0（旧版，已整合到compute_total_loss） |
| 691 | `PhysicsLoss.vof_residual()` | ∂φ/∂t + u·∇φ = 0（旧版） |
| 710 | `PhysicsLoss.navier_stokes_residual()` | NS方程（旧版） |
| 811 | `PhysicsLoss.surface_tension_residual()` | CSF（旧版） |
| 835 | `DataGenerator` | 训练数据生成器 |
| 1013 | `DataGenerator.target_phi_3d()` | 目标φ场构造 |
| 1242 | `DataGenerator.generate_all_data()` | 生成完整数据集 |
| 1693 | `Trainer.__init__()` | 训练器初始化（三阶段+权重调度） |
| 1930 | `Trainer.get_physics_weights(epoch)` | 分阶段物理损失权重 |
| 2022 | `Trainer.get_stage1_weight_factor(epoch)` | Stage1退火因子 |
| 2050 | `Trainer.compute_losses()` | 总损失调度（10+个损失项） |
| 2141 | `_compute_data_loss()` | 界面数据拟合损失（权重500×） |
| 2155 | `_compute_contact_angle_loss()` | 底面接触角BC |
| 2196 | `_compute_initial_boundary_loss()` | IC+BC损失 |
| 2221 | `_compute_early_zero_voltage_loss()` | 早期时间+零电压 |
| 2275 | `_compute_monotonicity_response_loss()` | 单调性响应 |
| 2309 | `_compute_eta_constraints_loss()` | 开口率约束 |
| 2369 | `_compute_phi_spatial_loss()` | φ场空间分布 |
| 2485 | `_compute_phi_geometry_loss()` | φ场几何一致性 |
| 2604 | `_compute_volume_conservation_loss()` | 体积守恒 |
| 2757 | `_compute_physics_equation_loss()` | PDE物理损失（三阶段调度） |
| 2912 | `compute_aperture_ratio()` | φ→开口率 |
| 2959 | `eta_recovery_constraint_loss()` | 降压恢复约束 |
| 3069 | `Trainer.train()` | 主训练循环 |

---

#### `src/models/aperture_model.py` — Stage 1 开口率模型（~1713行）
| 行号 | 类 | 用途 |
|------|-----|------|
| 49 | `ApertureModel` | 基类：接触角→开口率（能量最小化） |
| 392 | `EnhancedApertureModel` | 增强版：含围堰几何+接触角滞后 |

#### `src/predictors/hybrid_predictor.py` — Stage 1 预测器
| 行号 | 类 | 用途 |
|------|-----|------|
| 43 | `HybridPredictor` | 电压→接触角+开口率（Young-Lippmann+动态模型） |

#### `src/training/scheduler.py` — 动态权重调度
| 行号 | 类 | 用途 |
|------|-----|------|
| 21 | `DynamicPhysicsWeightScheduler` | combined/linear/step 三种权重策略 |
| 290 | `PhysicsWeightIntegration` | 权重与训练循环集成 |

#### `src/training/stabilizer.py` — 训练稳定性
| 行号 | 类 | 用途 |
|------|-----|------|
| 15 | `TrainingStabilizer` | NaN检测+恢复、预热、梯度裁剪 |

#### `src/training/components.py` — 训练组件
| 行号 | 类 | 用途 |
|------|-----|------|
| 20 | `DataNormalizer` | 输入/输出归一化 |
| 79 | `LossStabilizer` | Loss稳定性控制 |
| 137 | `EnhancedDataAugmenter` | 数据增强（噪声/扰动） |

#### `src/solvers/flow_solver.py` — CFD求解器（~2648行）
| 行号 | 类 | 用途 |
|------|-----|------|
| 74 | `Mesh` | 计算网格 |
| 164 | `SolverConfig` | 求解器配置 |
| 260 | `MeshGenerator` | 网格生成 |
| 450 | `InterfaceTracker` | VOF界面追踪 |
| 763 | `ContactLineHandler` | 接触线动力学 |
| 1013 | `FlowSolver` | NS+VOF耦合求解 |
| 1612 | `PINNSolver` | PINN推断封装 |
| 2086 | `FlowFieldSimulator` | 流场可视化后处理 |
| 2522 | `compute_aperture_ratio_from_phi()` | φ→开口率后处理 |

### 6D Triad 输入格式

```
(x, y, z, V_from, V_to, t_since)
```

- `V_from`: 跳变前电压. `V_to`: 当前电压. `t_since`: 跳变后经过时间
- `V_from == V_to` → 恒压状态; `V_from < V_to` → 升压(电润湿驱动); `V_from > V_to` → 降压(表面张力恢复)
- 这个设计让单个模型能连续模拟任意电压序列，避免为每个电压组合单独训练

### TwoPhasePINN 数据流

```
输入 (batch, 6)
  → 空间坐标归一化 [0,1] + 电压/时间归一化
  → FourierFeature(3 → 96) 缓解谱偏置
  → phi_net: [128,128,64,32] → sigmoid → phi ∈ [0,1]
  → vel_net: phi_features + phi → [64,64,32] → (u,v,w,p)
  → 输出 (batch, 5): (u, v, w, p, phi)
```

### PhysicsLoss 设计

PhysicsLoss 是适配层，委托 `PhysicsConstraints.compute_core_residuals()` 获取原始残差，然后做 NaN/Inf 清理、log1p 缩放、界面加权。各残差函数签名见上方 API 速查表。

### φ 场语义 (VOF)

- φ = 1: 纯油墨
- φ = 0: 纯极性液体(透明)
- 0 < φ < 1: 界面过渡区
- 初始状态: z < h_ink (3μm) → φ=1, z > h_ink → φ=0

### 配置系统

- `config/v4.5-standard.json` — 推荐训练配置, 已收敛验证 (30V 开口率 83.4%, 体积误差 <1%)
- `config/v4.6-standard.json` — v4.6 实验配置 (新增)
- `config/v4.5-physics-sampling.json` — 物理采样配置
- `config/v4.5-smoke.json` — 快速冒烟测试配置
- `config/device_calibrated_physics.json` — 器件物理参数
- `config/ablation/` — 消融实验配置 (no_continuity, no_vof, no_interface 等)
- 物理参数单一来源: `src.config.physics_config` 模块。三层体系：① `PHYSICS` 全局 dict（代码内嵌默认值）② `config/device_calibrated_physics.json`（实验校准值）③ `constraints.py` 中的回退硬编码（必须与前两层一致）。修改参数时三层同步更新
- 环境变量覆盖: `EFD_CONFIG_PATH`, `EFD_OUTPUT_DIR`

### 器件物理参数 (PHYSICS 关键常量)

```
几何:     Lx=Ly=174μm | Lz=20μm | h_ink=3μm | wall_height=3.5μm
流体:     ρ_oil=763 kg/m³ | μ_oil=9.41e-4 Pa·s | ρ_polar=998 | μ_polar=1.01e-3
界面:     σ=0.02505 N/m (油墨-极性) | γ=0.048 N/m (表面张力)
电学:     ε_SU8=3.28 | ε_Teflon=1.934 | d_dielectric=400nm | d_hydrophobic=400nm
接触角:   θ₀=120° (本征) | θ_wall=71° (围堰) | θ_min=60° | V_T=5V (阈值) | V_max=30V
动力学:   τ=5ms | ζ=0.8 (欠阻尼) | t_max=50ms
开口率:   η_max=0.85 | k=3.0 | θ_scale=19.0 | α=0.03
```
器件层叠: ITO → SU-8(400nm) → Teflon(400nm) → [油墨3μm + 极性液体17μm] → 围堰SU-8 → 顶层ITO

### 输出目录结构

```
outputs/train/pinn_YYYYMMDD_HHMMSS/
├── best_model.pth              # 最佳模型权重
├── best_model_epoch_XXXXX.pth  # 特定 epoch 检查点
├── training.log                # 训练日志
├── training_curve.png          # loss 曲线图
├── interface_3d_steady.png     # 3D 界面可视化
└── config.json                 # 训练配置快照
```

### 渐进式训练三阶段

1. **Stage 1 (前 1500 epochs)**: 纯数据拟合 (interface/IC/BC losses), 可选 eta tutor 约束
2. **Stage 2 (1500-4000 epochs)**: 引入连续性 + VOF 物理损失
3. **Stage 3 (4000+ epochs)**: 完整物理约束 (Navier-Stokes + 表面张力)

训练默认 60000 epochs, 支持 Adam→LBFGS 切换. 动态损失权重 (`combined` 策略) 自适应平衡物理约束和数据拟合.

### 工具链

- 包管理器: `uv` (pip 源: tuna.tsinghua, PyTorch: pytorch.org/whl/cu118)
- Python 3.12+, PyTorch 2.7.1 (CUDA 11.8)
- Lint: `ruff` (line-length=88). Format: `black` (line-length=88)
- 测试: `pytest`, 属性测试: `hypothesis`
- 可视化: `matplotlib` (IEEE 出版标准, Times New Roman, 300 DPI), `pyvista`, `streamlit`

### 已知问题

- **稳态正确但瞬态失败**: `TwoPhasePINN.forward()` 缺少时间门控, 导致 ∂φ/∂t 在电压跳变附近学习不好
- **physics_sampling.py 与 Stage 1 模型有 3 处不一致**: 待论文完成后修复

### 废弃文件

- `src/physics/constraints_deprecated.py` — 旧版物理约束, 已被 `constraints.py` 取代, 保留仅供历史参考

## 项目 Memory

每次会话开始时，请先读取 `.claude/memory/` 目录下的 memory 文件，了解用户偏好和项目当前状态。关键要点:
- 用户是 PINN/电润湿研究者，AI 物理推理需第一性原理验证，不确定时明确提出而非编造
- PHYSICS dict 是物理参数唯一来源，禁止本地定义
- 当前论文写作中，代码修改以实验支撑为优先
