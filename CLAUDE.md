# CLAUDE.md — EFD3D 项目快速参考

## 项目概述

EFD3D (Electrowetting Fluid Dynamics 3D) 是一个基于物理信息神经网络 (PINN) 的 3D 电润湿两相流仿真框架。项目位于 `/home/scnu/Gitee/EFD3D`。

**核心创新**: 6D Triad 输入 `(x, y, z, V_from, V_to, t_since)`，支持单一模型模拟任意电压序列。

## 首次会话启动流程

每次会话开始时，按以下顺序执行：

1. 读取本文件（CLAUDE.md）— 架构、API 速查、物理参数、规则
2. 读取 `.claude/memory/MEMORY.md` — 索引所有 memory 文件
3. 读取 `.claude/memory/` 下的所有 `.md` 文件 — 了解用户偏好、项目状态、已知问题
4. 检查 `git status` — 了解当前分支和未提交修改

## 关键规则

### 物理规则
- **物理参数单一来源**: 所有物理参数必须从 `src.config.PHYSICS` 导入，禁止在任何模块中本地定义物理常量
- **AI 物理推理不可信**: 涉及 PDE 修改、物理参数调整、边界条件变更时，必须从第一性原理推导。不确定时明确提出，不编造
- **优先相信代码和实验数据**，而非 AI 的物理直觉

### ⚠️ BC 物理事实（必须遵守，修改代码前确认）

**顶面 (z = Lz = 20μm)**：ITO 玻璃极板，始终接触极性液体。
- **φ = 0 永远成立**。任何位置出现 φ → 1 意味着油墨接触顶面 → **器件永久失效**
- 硬约束：`φ *= (1 - z_norm)` 保证 z=Lz 时 φ=0

**壁顶接触线（四周壁顶，z ≈ wall_height = 3.5μm）**：SU-8 暴露面（亲水，θ=125°）
- **φ = 0 始终成立**（极性液体接触）
- 油墨如果突破壁顶接触线 → 翻过围堰 → **器件永久失效**
- 接触线位置限定在矩形环内：x ∈ [wall_height, Lx-wall_height], y ∈ [wall_height, Ly-wall_height]

**侧壁夹角区域（z=0 四周等墙高夹角）**：Teflon 涂层（亲油，θ=110°）
- **φ = 1 始终成立**（油墨堆积区）
- 夹角区域：x ∈ [0, wall_top_half_width] ∪ [Lx-wall_top_half_width, Lx]，y 同理

**侧壁（x=0/Lx, y=0/Ly）**：Teflon 涂层
- z 方向采样必须到 Lz（全壁高），不能只采到 2*h_ink
- 壁顶边沿（z ≈ wall_height）油墨始终 φ=0，突破=失效

**接触线**：限定在底面 z=0 上的矩形环内
- 内半径 r_open（由 η 决定），外半径受壁顶限制
- 接触线不会越过壁顶接触线

### 训练铁律
1. **禁止重采样**: `resample_interval` 必须永远设为 0
2. **物理配点必须覆盖全空间域**: z 范围必须是 `[0, PHYSICS["Lz"]]`
3. **早期开口率诊断**: 每 2000-5000 epochs 检查开口率，若 η_max < 20% 且不增长，停止并诊断

### 代码规范
- 沟通语言: 中文
- 代码注释和文档: 中文
- 代码标识符: 英文
- 提交信息: 中文
- Lint: `ruff check` + `black`（已通过 hooks 自动执行）

### ⚠️ 修改前必须备份
- **修改任何 `.py` 文件前，必须先创建备份**：`cp file.py file.py.bak.YYYYMMDD`
- 备份应保留到修改被验证通过（测试通过 + 训练收敛）后再删除
- 对于核心文件（`pinn_two_phase.py`, `constraints.py`, `physics_config.py`），备份应提交到 git
- 目的：防止修改导致代码无法回退，特别是物理约束和训练逻辑的改动

### 禁止操作
- 不允许直接编辑 `.pth` 或 `.pt` 模型检查点文件（已通过 hook 拦截）
- 禁止在非 `outputs/` 目录中生成临时文件

---

## 架构

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

#### `src/physics/constraints.py` — 物理约束引擎（~1277行）

> **重构说明**：旧版 ~3441 行，经清理后缩减到 ~1277 行。删除了所有 DEPRECATED 函数、
> `compute_sidewall_contact_angle_residual`（与统一润湿BC冲突）、死权重和死代码。
> 统一梯度计算消除重复 autograd 调用，训练速度提升 3-4 倍。
>
> **BC 体系**（2026-06-11 重构后）：
> - 顶面（z=Lz，ITO 玻璃）：φ=0 硬约束，φ=1=器件失效
> - 壁顶（z≈wall_height，SU-8）：φ=0 硬约束 + bc 过采样，突破=失效
> - 侧壁（x=0/Lx, y=0/Ly，Teflon）：无滑移 + φ 匹配，z 采样到 [0, Lz]
> - 夹角区域（z=0 角落）：φ=1 bc 过采样（油墨堆积）
> - 底面（z=0，Teflon）：`_compute_contact_angle_loss`，Young-Lippmann 调制
> - ~~`_compute_unified_wetting_bc`~~ 已删除（与 bc 重叠）
> - ~~`_compute_contact_line_dynamics_residual`~~ 已删除（与 contact_angle 重叠）

| 行号 | 函数/方法 | 用途 | 活跃？ |
|------|-----------|------|--------|
| 37 | `_get_default_materials_params()` | 回退硬编码材料参数 | ✅ |
| 118 | `class PhysicsConstraints` | 物理约束类 | ✅ |
| 134 | `_compute_capacitance()` | 介电层串联电容计算 | ✅ |
| 151 | `compute_navier_stokes_residual()` | NS方程（连续性+动量+EWF+CSF）| ✅ |
| 345 | `_compute_laplacian()` | 标量场 Laplacian | ✅ |
| 384 | `compute_volume_conservation_residual()` | 体积守恒 | ✅ |
| 528 | `compute_interface_energy_residual()` | 界面能 σ\|∇φ\| | ✅ |
| 546 | `safe_compute_laplacian_spatial()` | 安全空间Laplacian | ✅ |
| 567 | `safe_compute_gradient()` | 安全梯度计算 | ✅ |
| 592 | `compute_laplace_pressure_residual()` | Laplace压力一致 | ✅ |
| 648 | `_compute_all_gradients()` | **统一梯度计算**（一次性 autograd，分发给所有约束）| ✅ |
| 689 | **`compute_core_residuals()`** | **统一入口，调用所有活跃约束** | ✅ |
| 748 | `_compute_temporal_smoothness()` | 时间连续性正则化 | ✅ |
| 789 | `_compute_vof_residual()` | Allen-Cahn相场方程 | ✅ |
| 855 | `safe_compute_laplacian()` | 安全Laplacian | ✅ |
| 875 | `safe_compute_hessian()` | 安全Hessian | ✅ |
| 898 | `_empty_residual()` | 空残差（降级用） | ✅ |

**`compute_core_residuals()` 调用链**（L689）：
1. `_compute_all_gradients()` — 统一计算所有梯度（一次性 autograd）
2. `compute_navier_stokes_residual()` — NS 方程残差
3. `_compute_vof_residual()` — VOF 相场方程
4. `compute_volume_conservation_residual()` — 体积守恒
5. `compute_interface_energy_residual()` — 界面能
6. `compute_laplace_pressure_residual()` — Laplace 压力
7. `_compute_temporal_smoothness()` — 时间连续性
8. `_compute_unified_wetting_bc()` — 统一相场润湿 BC（底面+侧壁，含 Young-Lippmann 电压调制）
9. `_compute_contact_line_dynamics_residual()` — 接触线动力学
10. 压力钉扎（pressure_pin）

---

#### `src/models/pinn_two_phase.py` — 主PINN模型（~3512行）

| 行号 | 类/方法 | 用途 |
|------|---------|------|
| 44 | `set_seed(seed)` | 全局随机种子 |
| 159 | `FourierFeature` | Fourier特征映射 (x,y,z→96) |
| 172 | `class TwoPhasePINN` | 主PINN网络 | ✅ |
| 191 | `TwoPhasePINN.__init__()` | phi_net[128,128,64,32] + vel_net[64,64,32] |
| 253 | `TwoPhasePINN.forward()` | (B,6)→(B,5): u,v,w,p,phi |
| 330 | `TwoPhasePINN.forward_triplet()` | 三时间点前向，用于时间导数 |
| 369 | `class PhysicsLoss` | 物理损失适配层 | ✅ |
| 388 | `PhysicsLoss.__init__()` | 加载参数，实例化 PhysicsConstraints |
| 423 | `PhysicsLoss._sanitize_tensor()` | NaN/Inf 清理 |
| 442 | `PhysicsLoss.compute_all_residuals()` | 调用 compute_core_residuals + 清理 |
| 464 | `PhysicsLoss.explicit_volume_conservation_loss()` | 显式体积守恒 |
| 487 | `PhysicsLoss.compute_total_loss()` | 加权总loss（log1p缩放+自适应EMA归一化） |
| 595 | `class DataGenerator` | 训练数据生成器 | ✅ |
| 610 | `DataGenerator.__init__()` | 初始化 |
| 668 | `DataGenerator.get_contact_angle()` | 电压+时间→接触角 |
| 683 | `DataGenerator._analytical_contact_angle()` | 解析接触角模型 |
| 742 | `DataGenerator.compute_contact_angle_gradient()` | 接触角梯度 |
| 759 | `DataGenerator.get_opening_rate()` | 电压+时间→开口率 |
| 786 | `DataGenerator.target_phi_3d()` | 目标φ场构造（从η构造） |
| 936 | `DataGenerator._phi_center_opening_mode()` | 中心开口模式φ场 |
| 1002 | `DataGenerator._sample_point_by_eta()` | 按η采样空间点 |
| 1036 | `DataGenerator.generate_all_data()` | 生成完整数据集 |
| 1505 | `class Trainer` | 训练器 | ✅ |
| 1508 | `Trainer.__init__()` | 训练器初始化（三阶段+权重调度） |
| 1633 | `Trainer._validate_config()` | 配置验证 |
| 1677 | `Trainer._save_checkpoint()` | 保存检查点 |
| 1712 | `Trainer._plot_curves()` | 绘制训练曲线 |
| 1785 | `Trainer.get_physics_weights(epoch)` | 分阶段物理损失权重 |
| 1854 | `Trainer.get_stage1_weight_factor(epoch)` | Stage1退火因子 |
| 1894 | `Trainer.compute_losses()` | 总损失调度（10+个损失项） |
| 1991 | `_compute_data_loss()` | 界面数据拟合损失 |
| 2003 | `_compute_contact_angle_loss()` | 底面接触角BC |
| 2044 | `_compute_initial_boundary_loss()` | IC+BC损失 |
| 2061 | `_compute_early_zero_voltage_loss()` | 早期时间+零电压 |
| 2139 | `_compute_monotonicity_response_loss()` | 单调性响应 |
| 2200 | `_compute_eta_constraints_loss()` | 开口率约束 |
| 2253 | `_compute_phi_spatial_loss()` | φ场空间分布 |
| 2512 | `_compute_volume_conservation_loss()` | 体积守恒 |
| 2598 | `_compute_continuity_transition_loss()` | 连续性约束（批量前向优化） |
| 2668 | `_compute_physics_equation_loss()` | PDE物理损失（三阶段调度） |
| 2787 | `compute_aperture_ratio()` | φ→开口率（网格法） |
| 2802 | `compute_aperture_ratio_batch()` | 批量φ→开口率 |
| 2852 | `compute_aperture_ratio_differentiable()` | 可微φ→开口率 |
| 2896 | `compute_eta_matching_loss()` | **η匹配loss**：PINN的η(V,t)追踪Teacher |
| 2950 | `compute_phi_target3d_loss()` | **φ target3D loss**：PINN的φ场匹配target |
| 3035 | `eta_recovery_constraint_loss()` | 降压恢复约束 |
| 3073 | `fine_tune_lbfgs()` | L-BFGS 二阶优化微调 |
| 3141 | `Trainer.train()` | 主训练循环 |
| 3461 | `Trainer.visualize()` | 可视化 |
| 3473 | `main()` | 入口函数 |

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

---

## 配置系统

- `config/v4.5-standard.json` — 推荐训练配置, 已收敛验证 (30V 开口率 83.4%, 体积误差 <1%)
- `config/v4.6-optimized.json` — v4.6 优化配置 (PFW:10, IE:2, LP:0.05, CLD:0.2, TS:0.1)
- `config/device_calibrated_physics.json` — 器件物理参数
- `config/ablation/` — 消融实验配置 (no_continuity, no_vof, no_interface 等)
- 物理参数单一来源: `src.config.physics_config` 模块。三层体系：① `PHYSICS` 全局 dict（代码内嵌默认值）② `config/device_calibrated_physics.json`（实验校准值）③ `constraints.py` 中的回退硬编码（必须与前两层一致）。修改参数时三层同步更新
- 环境变量覆盖: `EFD_CONFIG_PATH`, `EFD_OUTPUT_DIR`

### 器件物理参数（校正后，以 config/device_calibrated_physics.json 为准）

```
几何:       Lx=Ly=174μm | Lz=20μm | h_ink=3μm | wall_height=3.5μm
流体:       ρ_oil=763 kg/m³ | μ_oil=9.41e-4 Pa·s | ρ_polar=998 | μ_polar=1.01e-3
界面:       σ=0.02505 N/m (水油界面, CSF用) | γ=0.015 N/m (有效宏观, EW力用)
电学:       ε_r=12.0 (四层串联等效) | ε_SU8=3.28 | ε_Teflon=1.934 | d_dielectric=800nm (有效)
接触角:     θ₀=120° (本征) | θ_wall=71° (围堰) | θ_min=60° | V_T=3.0V (阈值, 校正后) | V_max=30V
动力学:     τ=11.9ms (校正后, 一阶) | ζ=1.0 (一阶系统) | t_max=80ms
开口率:     η_max=0.85 | k=3.0 | θ_scale=19.0 | α=0.03
```

**关键说明**:
- **ε_r = 12.0**: 四层串联等效（SU-8+Teflon+Oil+Polar），用于电容计算
- **σ = 0.02505 N/m**: 水油界面表面张力，用于 CSF 表面张力计算
- **γ = 0.015 N/m**: 有效宏观表面张力，从 CV 拟合，用于 EW 力公式
- **V_T = 3.0V**: 阈值电压（校正后，从 CV 数据）
- **τ = 11.9ms**: 时间常数（校正后，从 RT 数据一阶拟合）
- **一阶响应**: 1st-order RC charging/discharging（校正后确认，所有 R²>0.93）

器件层叠: ITO → SU-8(400nm) → Teflon(400nm) → [油墨3μm + 极性液体17μm] → 围堰SU-8 → 顶层ITO

**顶面物理** (z=Lz=20μm):
- 极性液体-固体界面（ITO 顶板），**不是自由表面**
- 边界条件：**无滑移** (u=v=0) + **无穿透** (w=0)
- ⚠️ 油膜一旦接触顶面 → **器件永久失效**（封闭像素无气体，油膜无法恢复）
- 顶面 φ=0 自然满足（油墨密度大，沉在底部），无需额外约束

---

## 命令速查

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

---

## 输出目录结构

```
outputs/train/pinn_YYYYMMDD_HHMMSS/
├── best_model.pth              # 最佳模型权重
├── best_model_epoch_XXXXX.pth  # 特定 epoch 检查点
├── training.log                # 训练日志
├── training_curve.png          # loss 曲线图
├── interface_3d_steady.png     # 3D 界面可视化
└── config.json                 # 训练配置快照
```

---

## 渐进式训练三阶段

1. **Stage 1 (前 1500 epochs)**: 纯数据拟合 (interface/IC/BC losses), 可选 eta tutor 约束
2. **Stage 2 (1500-4000 epochs)**: 引入连续性 + VOF 物理损失
3. **Stage 3 (4000+ epochs)**: 完整物理约束 (Navier-Stokes + 表面张力)

训练默认 60000 epochs, 支持 Adam→LBFGS 切换. 动态损失权重 (`combined` 策略) 自适应平衡物理约束和数据拟合.

---

## 工具链

- 包管理器: `uv` (pip 源: tuna.tsinghua, PyTorch: pytorch.org/whl/cu118)
- Python 3.12+, PyTorch 2.7.1 (CUDA 11.8)
- Lint: `ruff` (line-length=88). Format: `black` (line-length=88)
- 测试: `pytest`, 属性测试: `hypothesis`
- 可视化: `matplotlib` (IEEE 出版标准, Times New Roman, 300 DPI), `pyvista`, `streamlit`

---

## 当前项目状态

- **阶段**: 论文写作中
- **稳定基线**: v4.5-standard（30V 开口率 83.4%，体积误差 <1%）
- **实验中**: v4.6-optimized 配置
- **代码清理**: constraints.py 从 ~3441 行精简到 ~1277 行，删除了所有 DEPRECATED 函数、冲突的 `compute_sidewall_contact_angle_residual`、死权重（`electrowetting`/`bottom_wetting`/`wall_wetting`/`top_boundary`/`dielectric_charge`）和死代码
- **架构优化**: 统一梯度计算（`_compute_all_gradients`），训练速度提升 3-4 倍
- **BC 体系已统一**: 壁侧面 110°（Teflon）+ 壁顶面 71°（SU-8）+ 底面 Young-Lippmann 调制，无冲突
- **新增功能**: η匹配loss、φ target3D loss、L-BFGS微调、批量前向连续性约束
- **已知问题**: 瞬态精度不足（forward 缺少时间门控）、physics_sampling.py 与 Stage 1 有 3 处不一致（待论文完成后修复）

---

## 🎓 教学理念（Stage 1 → PINN 师生关系）

**核心原则**: Stage 1 是 PINN 的老师，不是 PINN 的拐杖。

| 角色 | 职责 | 目标 |
|------|------|------|
| **Stage 1（老师）** | 早期提供电压-开口率的映射关系 | 教会 PINN "电压如何影响油膜形状" |
| **PINN（学生）** | 先跟老师学，后来自主泛化 | 最终不需要老师也能正确预测 |

### 教学阶段

| 阶段 | Stage 1 的作用 | PINN 的状态 |
|------|---------------|-------------|
| **早期（S1）** | **强指导**: target3D 提供油膜形状，Interface loss 权重大 | 学生跟着老师学电压-开口率关系 |
| **中期（S2）** | **逐渐放手**: Interface loss 退火，物理约束权重上升 | 学生开始用物理方程验证 |
| **后期（S3）** | **独立学习**: Interface loss 权重很小甚至为零，PINN 靠物理约束自主学习 | 学生已经学会电压-开口率映射，不再需要老师 |

### target3D 的作用

- **不是最终答案**，而是**油膜形状的指导**
- 告诉 PINN："在这个电压下，油膜大概应该长这样"
- PINN 通过物理约束（NS/VOF/EW）来**细化**这个形状，而不是死记硬背

### 成功的标志

> **如果后期不需要 Stage 1 老师，PINN 学生也能学好，那就说明教学成功了——学生已经会走了。**

---

## 🔑 核心物理图像（必须记住，修改代码前仍需验证）

### 接触线系统（双接触线结构）

| 接触线 | 位置 | 特性 | 运动方式 |
|--------|------|------|----------|
| **第一接触线** | 围堰壁顶部与侧面交界处 (z≈3.5μm) | **基本固定**，除非油膜被推倒翻墙 | 沿壁面移动（壁顶+壁侧交界线） |
| **第二接触线** | 凹坑内疏水层表面 Z=0 界面 | **动态变化**，油膜最薄处随电压增加形成开口，向外扩散至像素墙角毛细半径宽度 | 在 Z=0 平面上扩散/收缩 |

### 油膜演化过程（0V → 工作状态）

1. **0V 初始状态**: 油膜在围堰内**几乎平铺**（无开口）
2. **施加电压**: 电润湿力驱动油膜变形
3. **围堰凹坑**: 油膜的主要战场——**油墨体积不变，形状发生改变**
4. **第二接触线形成与扩展**: 电压增加 → 油膜最薄处（Z=0 界面）形成开口 → 接触线向外扩散
5. **第一接触线**: 保持固定（除非油膜被推倒翻墙）

### 关键物理定义

| 物理量 | 定义 |
|--------|------|
| **围堰底面积** | 174μm × 174μm（开口率的分母） |
| **开口率** | 开口面积 / 围堰内底面积 |
| **体积守恒** | 油墨总体积不变，底面积 = (1 - 开口率) / 底部总面积 |
| **target3D** | Stage 1 预测的 φ 场，包含 Z=0 界面开口状态和预计的油膜形状 |
| **Interface loss** | 在整个 3D 域上计算 PINN 预测与 target3D 的差异 |
| **开关特性** | 线性，与用户提供的校正 CV 曲线相似 |
| **翻墙条件** | 电压过高，油膜在其中一个角落收缩太严重，表面张力绷不住 → 第一接触线松动 → 器件失效 |

### 翻墙失效机制

- **条件**: 电压过高 → 油膜在某个角落收缩太严重 → 表面张力绷不住
- **后果**: 第一接触线松动 → 油膜翻过这条线 → 器件永久失效

### ⚠️ 原则

修改涉及油膜/接触线的代码时，先从第一性原理验证物理正确性，不确定就问，问清楚再行动。

### 🔬 电润湿物理图像（2026-06-04 确认）

**物理链条**: 电润湿 → 接触角变化 → 边缘靠毛细力/中间表面张力 → 油膜变形

**关键理解**:
- **EW 不是体积力**：电润湿力是保守力（势能梯度），不直接推动流体，而是改变界面能
- **接触角是直接驱动力**：电润湿通过 Young-Lippmann 方程改变接触角，接触角变化驱动油膜变形
- **EW 保留在 AC 方程中**：作为相场演化的驱动力（源项），但量级需与 AC 残差匹配
- **训练权重**: 接触角 BC 应主导（权重~20），IF 不应主导（权重~50）

**当前配置（v4.6-optimized.json）**:
- IF: 50（原 500）
- 接触角 BC: 20（原 2）
- 表面张力: 2（原 0.01）
- 界面能: 2（原 0.05）

---

## 废弃文件

- `src/physics/constraints_deprecated.py` — 旧版物理约束, 已被 `constraints.py` 取代, 保留仅供历史参考

## 项目 Memory

每次会话开始时，请先读取 `.claude/memory/` 目录下的 memory 文件，了解用户偏好和项目当前状态。关键要点:
- 用户是 PINN/电润湿研究者，AI 物理推理需第一性原理验证，不确定时明确提出而非编造
- PHYSICS dict 是物理参数唯一来源，禁止本地定义
- 当前论文写作中，代码修改以实验支撑为优先
