# EFD3D 项目 Code Wiki

## 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [核心模块详解](#核心模块详解)
4. [物理模型](#物理模型)
5. [使用指南](#使用指南)
6. [API 参考](#api-参考)

---

## 项目概述

### 项目简介

**EFD3D (Electrowetting Fluid Dynamics 3D)** 是一个工业级的物理信息神经网络 (Physics-Informed Neural Network, PINN) 仿真框架，专为解决微流控和电子纸显示技术 (Electronic Paper Display, EPD) 中极端复杂的三维两相流问题而设计。

### 核心特性

- **6D Triad 输入**: `(x, y, z, V_from, V_to, t_since)` - 支持任意电压跳变序列
- **两阶段架构**:
  - Stage 1: 解析模型预测接触角/开口率（已校准）
  - Stage 2: PINN求解完整流场（Navier-Stokes + VOF）
- **无网格仿真**: 连续空间采样，避免网格生成
- **工业级精度**: 体积守恒误差 < 1%
- **混合CFD-PINN求解器**: 传统CFD求解器用于验证，PINN用于加速
- **端到端训练**: 无需Stage 1依赖的替代训练方法
- **动态损失权重调度**: 自适应平衡各项物理约束
- **训练稳定性增强**: NaN恢复机制和梯度裁剪

### 版本历史

| 版本 | 发布日期 | 核心改进 |
|------|---------|---------|
| **v4.5** | 2026-01-29 | 界面锐化损失，30V开口率83.4%，体积误差<1% |
| v4.4 | 2026-01-13 | Stage 1 Tutor约束 |
| v4.3 | 2026-01-08 | 基础PINN架构 |

---

## 系统架构

### 目录结构

```
/workspace/
├── src/                           # 核心源代码
│   ├── models/                   # 神经网络模型
│   │   ├── pinn_two_phase.py     # 主PINN模型 (TwoPhasePINN + Trainer)
│   │   └── aperture_model.py     # Stage 1 解析模型
│   ├── physics/                  # 物理约束引擎
│   │   └── constraints.py        # Navier-Stokes, VOF, 连续性方程
│   ├── training/                 # 训练基础设施
│   │   ├── scheduler.py          # 动态损失权重调度
│   │   ├── stabilizer.py         # NaN恢复, 梯度裁剪
│   │   └── components.py         # 训练工具
│   ├── config/                   # 配置管理
│   │   ├── __init__.py           # PHYSICS参数导出
│   │   └── physics_config.py     # 类型安全配置
│   ├── predictors/               # 预测器
│   │   ├── hybrid_predictor.py   # 混合预测器
│   │   └── pinn_aperture.py      # PINN开口率预测
│   ├── solvers/                  # 求解器
│   │   └── flow_solver.py        # 流场求解器
│   ├── data/                     # 数据生成
│   │   └── physics_sampling.py   # 物理采样
│   ├── dashboard/                # 交互式仪表板
│   │   ├── app.py                # 主应用
│   │   ├── plotter.py            # 绘图工具
│   │   └── monitor/              # 监控模块
│   └── utils/                    # 工具函数
│       ├── model_utils.py        # 模型加载和预测提取
│       └── logging_config.py     # 统一日志配置
├── config/                       # 训练配置
│   ├── v4.5-standard.json        # 推荐配置 (已验证收敛)
│   └── device_calibrated_physics.json  # 物理校准
├── scripts/                      # 用户工具
│   ├── dashboard.py              # Streamlit交互式面板
│   └── run_ablation.sh           # 消融研究脚本
├── tests/                        # 测试套件
├── outputs/                      # 训练输出
├── train_two_phase.py            # 训练入口
├── evaluate.py                   # 评估入口
└── pyproject.toml                # 项目配置
```

### 依赖关系图

```
train_two_phase.py
    └── src/models/pinn_two_phase.py
            ├── src/physics/constraints.py
            ├── src/training/scheduler.py
            ├── src/training/stabilizer.py
            └── src/config/physics_config.py

evaluate.py
    ├── src/models/pinn_two_phase.py
    ├── src/models/aperture_model.py
    └── src/config/physics_config.py

scripts/dashboard.py
    └── src/dashboard/app.py
```

---

## 核心模块详解

### 1. 模型模块 (`src/models/`)

#### 1.1 TwoPhasePINN 类 ([`pinn_two_phase.py`](file:///workspace/src/models/pinn_two_phase.py))

**主要职责**：
- 实现两相流物理信息神经网络
- 处理 6D 输入: `(x, y, z, V_from, V_to, t_since)`
- 输出 5D 流场: `(u, v, w, p, phi)`
- 包含 Fourier 特征编码以缓解谱偏置

**关键代码**：

```python
class TwoPhasePINN(nn.Module):
    """
    两相流物理信息神经网络
    
    输入: (x, y, z, V_from, V_to, t_since) - 6维（三元组格式）
    输出: (u, v, w, p, phi)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # 初始化网络架构
        # 分离 phi 网络和速度网络
        # 使用 Fourier 特征编码
```

#### 1.2 EnhancedApertureModel 类 ([`aperture_model.py`](file:///workspace/src/models/aperture_model.py))

**主要职责**：
- Stage 1 解析模型
- 接触角 → 油墨分布 → 开口率
- 提供可视化功能

**关键代码**：

```python
class EnhancedApertureModel:
    """
    增强版开口率模型
    接触角 → 油墨分布 → 开口率 → 可视化
    """
    
    def predict_enhanced(self, voltage: float) -> Dict:
        """预测给定电压下的开口率和油墨分布"""
```

### 2. 物理约束模块 ([`constraints.py`](file:///workspace/src/physics/constraints.py))

#### PhysicsConstraints 类

**主要职责**：
- 计算 Navier-Stokes 方程残差
- 计算 VOF 输运方程残差
- 计算连续性方程残差
- 处理边界条件和初始条件
- 计算表面张力和电润湿力

**关键方法**：

```python
class PhysicsConstraints:
    """物理约束类 - 处理Navier-Stokes方程和材料属性"""
    
    def compute_navier_stokes_residual(self, x, predictions, model=None):
        """计算Navier-Stokes方程残差"""
    
    def compute_vof_residual(self, x, predictions):
        """计算VOF输运方程残差"""
    
    def compute_continuity_residual(self, x, predictions):
        """计算连续性方程残差"""
```

### 3. 训练模块 ([`src/training/`](file:///workspace/src/training/))

#### 3.1 DynamicPhysicsWeightScheduler 类 ([`scheduler.py`](file:///workspace/src/training/scheduler.py))

**主要职责**：
- 动态调整物理损失权重
- 支持多种调整策略: "stage", "adaptive", "performance", "combined"
- 平滑权重变化，避免剧烈波动

#### 3.2 TrainingStabilizer 类 ([`stabilizer.py`](file:///workspace/src/training/stabilizer.py))

**主要职责**：
- 处理训练过程中的 NaN
- 梯度裁剪
- 学习率调度

### 4. 配置模块 ([`src/config/`](file:///workspace/src/config/))

#### 4.1 PhysicsConfig 类 ([`physics_config.py`](file:///workspace/src/config/physics_config.py))

**主要职责**：
- 统一管理所有物理参数
- 提供类型安全的参数访问
- 支持从 JSON 配置文件加载
- 提供多种导出格式 (to_dict, to_materials_params, to_predictor_params)

**关键属性**：

- **几何参数**: Lx, Ly, Lz, h_ink, wall_height
- **流体属性**: rho_oil, mu_oil, rho_polar, mu_polar, sigma, gamma
- **电学参数**: epsilon_0, epsilon_r, epsilon_h, d_dielectric, d_hydrophobic
- **接触角**: theta0, theta_wall, theta_min
- **动力学参数**: tau, zeta, t_max

**关键方法**：

```python
@dataclass
class PhysicsConfig:
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PhysicsConfig":
        """从 JSON 配置文件加载物理参数"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容 PHYSICS 格式）"""
    
    def to_materials_params(self) -> Dict[str, Any]:
        """转换为 PhysicsConstraints 兼容的 materials_params 格式"""
```

#### 4.2 全局 PHYSICS 字典

```python
PHYSICS: Dict[str, Any] = {
    "Lx": 174e-6,           # 像素宽度 (m)
    "Ly": 174e-6,           # 像素高度 (m)
    "Lz": 20e-6,            # 围堰/流体层高度 (m)
    "h_ink": 3e-6,          # 油墨层厚度 (m)
    "rho_oil": 763.0,       # 油墨密度 (kg/m³)
    "mu_oil": 9.41e-4,      # 油墨粘度 (Pa·s)
    "rho_polar": 998.0,     # 极性液体密度 (kg/m³)
    "mu_polar": 1.01e-3,    # 极性液体粘度 (Pa·s)
    "sigma": 0.02505,       # 界面张力 (N/m)
    # ... 更多参数
}
```

### 5. 仪表板模块 ([`src/dashboard/`](file:///workspace/src/dashboard/))

**主要功能**：
- 2D 场分析
- 3D 体积渲染
- 瞬态响应分析
- 物理诊断
- 训练输出分析
- 基准测试
- 对比分析

**入口文件**: [`scripts/dashboard.py`](file:///workspace/scripts/dashboard.py)

---

## 物理模型

### 控制方程

#### 1. 连续性方程

```
∇ · u = 0
```

#### 2. VOF 输运方程

```
∂φ/∂t + u · ∇φ = 0
```

其中，φ 为体积分数：
- φ = 1: 纯油墨
- φ = 0: 纯极性液体
- 0 < φ < 1: 界面过渡区

#### 3. Navier-Stokes 方程

```
ρ(∂u/∂t + u · ∇u) = -∇p + μ∇²u + F_st
```

其中:
- ρ 为混合密度: ρ = φρ_oil + (1-φ)ρ_polar
- μ 为混合粘度: μ = φμ_oil + (1-φ)μ_polar
- F_st 为表面张力和电润湿力

### 物理参数

详见 [`src/config/physics_config.py`](file:///workspace/src/config/physics_config.py) 中的 PHYSICS 字典。

---

## 使用指南

### 1. 快速开始

#### 安装依赖

```bash
# 使用 uv 管理依赖
uv pip install -e .
```

#### 训练模型

```bash
# 使用推荐配置训练
uv run train_two_phase.py --config config/v4.5-standard.json

# 快速测试 (1,000轮)
uv run train_two_phase.py --epochs 1000

# 从检查点恢复训练
uv run train_two_phase.py --config config/v4.5-standard.json \
    --resume_from outputs/train/pinn_YYYYMMDD_HHMMSS/best_model.pth
```

#### 评估模型

```bash
# 运行评估并生成可视化
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/

# 启动交互式面板
uv run scripts/dashboard.py

# 运行消融研究
uv run scripts/run_ablation.sh
```

#### 运行测试

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行特定测试模块
uv run pytest tests/test_pinn_complete.py -v
```

### 2. 配置指南

配置文件格式为 JSON，包含以下主要部分：

```json
{
    "model": {
        "hidden_phi": [64, 64, 64, 32],
        "hidden_vel": [64, 64, 32]
    },
    "training": {
        "epochs": 60000,
        "batch_size": 4096,
        "learning_rate": 0.0003,
        "gradient_clip": 1.0
    },
    "physics": {
        "interface_weight": 500.0,
        "ic_weight": 300.0,
        "bc_weight": 80.0,
        "continuity_weight": 0.5,
        "vof_weight": 0.5,
        "ns_weight": 0.1
    }
}
```

推荐配置: [`config/v4.5-standard.json`](file:///workspace/config/v4.5-standard.json)

---

## API 参考

### 核心模型 API

#### TwoPhasePINN

```python
from src.models.pinn_two_phase import TwoPhasePINN

model = TwoPhasePINN(config)
predictions = model(x)  # x: (batch, 6) -> (batch, 5)
```

#### PhysicsConstraints

```python
from src.physics.constraints import PhysicsConstraints

physics = PhysicsConstraints()
residuals = physics.compute_navier_stokes_residual(x, predictions)
```

#### PhysicsConfig

```python
from src.config import get_physics_config, PHYSICS

# 方式1: 使用全局字典
theta0 = PHYSICS["theta0"]

# 方式2: 使用类型安全的配置类
config = get_physics_config("config/device_calibrated_physics.json")
theta0 = config.theta0
```

### 预测器 API

```python
from src.predictors.hybrid_predictor import HybridPredictor

predictor = HybridPredictor()
result = predictor.predict(voltage=30, t=0.05)
```

---

## 参考资料

### 相关文档

- [项目 README](file:///workspace/README.md)
- [配置指南](file:///workspace/docs/guides/configuration_guide.md)
- [物理理论](file:///workspace/docs/guides/physics_and_device_guide.md)
- [系统设计](file:///workspace/docs/architecture/system_design.md)

### 引用文献

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.jcp.2018.10.045}
}
```

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！详见 [`docs/CONTRIBUTING.md`](file:///workspace/docs/CONTRIBUTING.md)。

---

## 许可证

MIT License - 详见 [`LICENSE`](file:///workspace/LICENSE) 文件。
