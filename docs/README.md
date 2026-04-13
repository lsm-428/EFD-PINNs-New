# 📚 EFD-PINNs 文档中心

**最后更新**: 2026-04-13

---

## 🎯 快速导航

### 入门
- **[../README.md](../README.md)** - 项目入口
- **[../CLAUDE.md](../CLAUDE.md)** - 项目核心文档（包含知识库系统）⭐
- **[guides/installation.md](guides/installation.md)** - 安装与环境配置
- **[guides/usage.md](guides/usage.md)** - 使用指南
- **[guides/visualization_guide.md](guides/visualization_guide.md)** - 可视化指南

### 技术文档
- **[architecture/system_design.md](architecture/system_design.md)** - 系统架构设计
- **[guides/physics_and_device_guide.md](guides/physics_and_device_guide.md)** - 物理理论与器件规格

### 训练与验证
- **[research/TRAINING_REPORTS.md](research/TRAINING_REPORTS.md)** - 训练报告与结果
- **[guides/training_guide.md](guides/training_guide.md)** - 训练策略
- **[guides/troubleshooting.md](guides/troubleshooting.md)** - 故障排除
- **[architecture/problem_analysis/volume_conservation_complete_analysis.md](architecture/problem_analysis/volume_conservation_complete_analysis.md)** - 体积守恒问题深度分析

### 配置与贡献
- **[guides/configuration_guide.md](guides/configuration_guide.md)** - 配置系统详解
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - 贡献指南
- **[CHANGELOG.md](CHANGELOG.md)** - 更新日志

---

## 📁 文档结构

src/
├── models/                    # 神经网络模型 (3 modules)
│   ├── aperture_model.py      # Stage 1 开口率模型（已校准）
│   └── pinn_two_phase.py      # Stage 2 两相流 PINN（6D Triad 输入）
├── predictors/                # 预测器 (3 modules)
│   ├── pinn_aperture.py       # Stage 2 PINN 预测器
│   └── hybrid_predictor.py    # Stage 1 + Stage 2 集成预测器
├── physics/                   # 物理约束 (2 modules)
│   └── constraints.py         # Navier-Stokes, VOF, 连续性方程
├── training/                  # 训练基础设施 (4 modules)
│   ├── scheduler.py           # 动态损失权重调度
│   ├── stabilizer.py          # NaN恢复, 梯度裁剪
│   └── components.py          # 训练工具
├── config/                    # 配置管理 (3 modules)
│   ├── __init__.py            # PHYSICS参数导出
│   └── physics_config.py      # 类型安全配置
├── dashboard/                 # Streamlit仪表板 (21 modules)
│   ├── monitor/               # 监控和日志分析 (6 modules)
│   └── reports/               # 报告生成 (3 modules)
├── solvers/                   # CFD求解器 (2 modules)
└── utils/                     # 工具函数 (3 modules)
```

---

## 🔍 按需求查找

| 需求 | 文档 |
|------|------|
| 项目总览 | [../README.md](../README.md) |
| 核心文档 | [../CLAUDE.md](../CLAUDE.md) ⭐ |
| 安装配置 | [guides/installation.md](guides/installation.md) |
| 快速使用 | [guides/usage.md](guides/usage.md) |
| 系统架构 | [architecture/system_design.md](architecture/system_design.md) |
| 物理理论 | [guides/physics_and_device_guide.md](guides/physics_and_device_guide.md) |
| 训练策略 | [guides/training_guide.md](guides/training_guide.md) |
| 故障排除 | [guides/troubleshooting.md](guides/troubleshooting.md) |
| 可视化 | [guides/visualization_guide.md](guides/visualization_guide.md) |
| **新用户** | **[guides/quickstart.md](guides/quickstart.md)** |
| **开发者** | **[../CLAUDE.md](../CLAUDE.md)** |
| **研究人员** | **[research/DEEP_UNDERSTANDING_GUIDE.md](research/DEEP_UNDERSTANDING_GUIDE.md)** |

## 🆕 2026-02-04 重要更新

### 项目里程碑 (v4.5 / Stage 2)

| 指标 | 数值 | 说明 |
|------|------|------|
| 30V 开口率 | **83.4%** | 像素开口率优化至工业级水平 |
| 体积误差 | **<1%** | VOF 输运方程体积守恒修正成效 |
| 代码规模 | **57 文件** | 41 src + 1 scripts + 15 tests |
| 输入表示 | **6D Triad** | (x, y, z, V_from, V_to, t_since) |

### 技术参数详情 {#physics-parameters}

> 📖 **完整技术参数**: 材料参数、动态参数、训练配置等详见 **[物理理论与器件规格指南](guides/physics_and_device_guide.md#physics-parameters)**

#### 材料参数（校准值）

| 参数 | 符号 | 值 | 单位 | 描述 |
|-----------|--------|-------|------|-------------|
| 初始接触角 | `theta0` | 120.0 | 度 | Teflon表面的初始接触角 |
| 像素壁接触角 | `theta_wall` | 71.0 | 度 | 像素壁处的接触角 |
| 有效介电常数 | `epsilon_r` | 12.0 | - | 有效介电常数（SU-8 + 渗透效应） |
| 表面张力 | `gamma` | 0.015 | N/m | 有效界面张力 |
| 介电层厚度 | `d_dielectric` | 400e-9 | m | SU-8层厚度 |
| 疏水层厚度 | `d_hydrophobic` | 400e-9 | m | Teflon层厚度 |
| 阈值电压 | `V_threshold` | 3.0 | V | 电润湿阈值电压 |
| 真空介电常数 | `epsilon_0` | 8.854e-12 | F/m | 真空介电常数 |

#### 动态参数

| 参数 | 符号 | 值 | 单位 | 描述 |
|-----------|--------|-------|------|-------------|
| 时间常数 | `tau` | 5e-3 | s | 电润湿响应时间常数 |
| 恢复时间常数 | `tau_recovery` | 7.5e-3 | s | 电压下降期间的恢复时间常数 |
| 阻尼比 | `zeta` | 0.8 | - | 欠阻尼系统阻尼比 |

#### 几何参数

| 参数 | 符号 | 值 | 单位 | 描述 |
|-----------|--------|-------|------|-------------|
| 像素宽度 | `Lx` | 174e-6 | m | 像素宽度（内沿尺寸） |
| 像素长度 | `Ly` | 174e-6 | m | 像素长度 |
| 围堰高度 | `Lz` | 20e-6 | m | 围堰高度（模型值） |
| 油墨层厚度 | `h_ink` | 3e-6 | m | 油墨层厚度 |
| 实际围堰高度 | `h_wall_real` | 3.5e-6 | m | 实际围堰高度 |

### 训练配置 {#training-configuration}

#### 训练超参数

| 参数 | 默认值 | 描述 |
|-----------|---------------|-------------|
| 总训练轮次 | 60000 | 总训练轮次 |
| 批次大小 | 4096 | 训练批次大小 |
| 学习率 | 0.0003 | 基础学习率 |
| 第一阶段轮次 | 1500 | 几何阶段训练轮次 (0-1500) |
| 第二阶段轮次 | 4000 | 运动学阶段训练轮次 (1500-5500) |
| 第三阶段轮次 | 50000 | 完整物理阶段训练轮次 (5500-60000) |

#### 损失权重

| 损失项 | 默认权重 | 描述 |
|----------------|----------------|-------------|
| 界面 | 500.0 | 界面数据拟合 |
| 初始条件 | 300.0 | 初始条件约束 |
| 边界条件 | 80.0 | 边界条件约束 |
| 连续性 | 0.5 | 质量守恒（∇·u = 0） |
| VOF | 0.5 | 流体体积法输运方程 |
| Navier-Stokes | 0.1 | 动量守恒 |
| 表面张力 | 0.01 | 表面张力力 |
| 锐化 | 1.0 | 界面锐化损失 |
| 显式体积守恒 | 100.0 | 显式体积守恒约束 |

### 器件规格 {#device-specifications}

#### 像素尺寸

- **像素尺寸**: 174μm × 174μm
- **围堰高度**: 20μm（模型）/ 3.5μm（实际）
- **油墨层**: 3μm
- **工作电压范围**: 0-30V
- **典型工作电压**: 20V

#### 层结构

src/
├── models/                    # 神经网络模型 (3 modules)
│   ├── aperture_model.py      # Stage 1 开口率模型（已校准）
│   └── pinn_two_phase.py      # Stage 2 两相流 PINN（6D Triad 输入）
├── predictors/                # 预测器 (3 modules)
│   ├── pinn_aperture.py       # Stage 2 PINN 预测器
│   └── hybrid_predictor.py    # Stage 1 + Stage 2 集成预测器
├── physics/                   # 物理约束 (2 modules)
│   └── constraints.py         # Navier-Stokes, VOF, 连续性方程
├── training/                  # 训练基础设施 (4 modules)
│   ├── scheduler.py           # 动态损失权重调度
│   ├── stabilizer.py          # NaN恢复, 梯度裁剪
│   └── components.py          # 训练工具
├── config/                    # 配置管理 (3 modules)
│   ├── __init__.py            # PHYSICS参数导出
│   └── physics_config.py      # 类型安全配置
├── dashboard/                 # Streamlit仪表板 (21 modules)
│   ├── monitor/               # 监控和日志分析 (6 modules)
│   └── reports/               # 报告生成 (3 modules)
├── solvers/                   # CFD求解器 (2 modules)
└── utils/                     # 工具函数 (3 modules)
```

### API 参考

#### 关键类和函数

##### `TwoPhasePINN` (`src/models/pinn_two_phase.py`)
- **输入**: 6D Triad `[x,y,z,V_from,V_to,t_since]`
- **输出**: 5D物理场 `[u,v,w,p,φ]`
- **网络**: 双分支MLP（速度网络 + φ网络）
- **特性**: 支持三阶段渐进式训练，包含电润湿驱动力

##### `Trainer` (`src/models/pinn_two_phase.py`)
- **功能**: 管理三阶段训练流程
- **特性**: 动态权重调度，训练稳定性管理，NaN恢复机制

##### `PhysicsConstraints` (`src/physics/constraints.py`)
- **实现**: 物理方程约束集
  - 连续性: `∇·u = 0`
  - Navier-Stokes: `ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st + F_ew`
  - VOF: `∂φ/∂t + u·∇φ = 0`
  - 界面锐化: `L_sharp = λ·φ(1-φ)`

##### `EnhancedApertureModel` (`src/models/aperture_model.py`)
- **功能**: 基于Young-Lippmann方程的解析模型
- **输入**: 电压值
- **输出**: `[θ, η]`（接触角，开口率）
- **状态**: 已校准，30V开口率83.4%

### 训练进展

- **当前版本**: v4.5
- **核心改进**: 界面锐化 + 体积守恒强化
- **详细信息**: [research/TRAINING_REPORTS.md](research/TRAINING_REPORTS.md)

---

## 📊 脚本工具

项目包含以下脚本：

### 主要脚本

| 脚本 | 说明 |
|------|------|
| `scripts/dashboard.py` | Streamlit 交互式仪表板，集成了所有分析和可视化功能 ⭐ |
| `scripts/run_ablation.sh` | 消融研究脚本 |

### 仪表板功能模块

`scripts/dashboard.py` 包含8个功能模块：
1. **📊 2D场分析** - 截面场分析，支持任意电压序列模拟
2. **🧊 3D体渲染视图** - 三维体积渲染和界面重建
3. **📈 瞬态响应** - 时域响应分析和动态可视化
4. **🩺 物理诊断** - 物理约束验证和残差分析
5. **📊 训练输出分析** - 训练曲线、模型性能和收敛分析
6. **⏱️ 基准测试** - 性能指标和性能对比
7. **🔄 对比分析** - 多模型/多配置对比
8. **📐 Stage 1** - 解析模型验证和接触角分析

### 使用示例

```bash
# 启动交互式仪表板（推荐）
uv run scripts/dashboard.py

# 运行消融研究
uv run scripts/run_ablation.sh
src/
├── models/                    # 神经网络模型 (3 modules)
│   ├── aperture_model.py      # Stage 1 开口率模型（已校准）
│   └── pinn_two_phase.py      # Stage 2 两相流 PINN（6D Triad 输入）
├── predictors/                # 预测器 (3 modules)
│   ├── pinn_aperture.py       # Stage 2 PINN 预测器
│   └── hybrid_predictor.py    # Stage 1 + Stage 2 集成预测器
├── physics/                   # 物理约束 (2 modules)
│   └── constraints.py         # Navier-Stokes, VOF, 连续性方程
├── training/                  # 训练基础设施 (4 modules)
│   ├── scheduler.py           # 动态损失权重调度
│   ├── stabilizer.py          # NaN恢复, 梯度裁剪
│   └── components.py          # 训练工具
├── config/                    # 配置管理 (3 modules)
│   ├── __init__.py            # PHYSICS参数导出
│   └── physics_config.py      # 类型安全配置
├── dashboard/                 # Streamlit仪表板 (21 modules)
│   ├── monitor/               # 监控和日志分析 (6 modules)
│   └── reports/               # 报告生成 (3 modules)
├── solvers/                   # CFD求解器 (2 modules)
└── utils/                     # 工具函数 (3 modules)
```
