# AGENTS.md — EFD3D 项目代理指南

## 项目概述

EFD3D (Electrowetting Fluid Dynamics 3D) 是一个基于物理信息神经网络 (PINN) 的 3D 电润湿两相流仿真框架。项目位于 `/home/scnu/Gitee/EFD3D`。

**核心创新**: 6D Triad 输入 `(x, y, z, V_from, V_to, t_since)`，支持单一模型模拟任意电压序列。

## 首次会话启动流程

每次会话开始时，按以下顺序执行：

1. 读取 `CLAUDE.md` — 项目架构、API 速查、物理参数
2. 读取 `.claude/memory/MEMORY.md` — 索引所有 memory 文件
3. 读取 `.claude/memory/` 下的所有 `.md` 文件 — 了解用户偏好、项目状态、已知问题
4. 检查 `git status` — 了解当前分支和未提交修改

## 关键规则

### 物理规则
- **物理参数单一来源**: 所有物理参数必须从 `src.config.PHYSICS` 导入，禁止在任何模块中本地定义物理常量
- **AI 物理推理不可信**: 涉及 PDE 修改、物理参数调整、边界条件变更时，必须从第一性原理推导。不确定时明确提出，不编造
- **优先相信代码和实验数据**，而非 AI 的物理直觉

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

### 禁止操作
- 不允许直接编辑 `.pth` 或 `.pt` 模型检查点文件（已通过 hook 拦截）
- 禁止在非 `outputs/` 目录中生成临时文件

## 架构速查

### 两阶段设计
- **Stage 1 (解析模型)**: `HybridPredictor` + `EnhancedApertureModel`，电压→接触角→开口率
- **Stage 2 (PINN)**: `TwoPhasePINN`，求解 Navier-Stokes + VOF，输出 `(u, v, w, p, phi)`

### 核心文件
| 文件 | 用途 |
|------|------|
| `src/models/pinn_two_phase.py` (~3325行) | 主 PINN 模型 + Trainer |
| `src/physics/constraints.py` (~3238行) | 物理约束引擎（唯一物理真相）|
| `src/config/physics_config.py` | PHYSICS 参数配置 |
| `src/training/scheduler.py` | 动态损失权重调度 |
| `src/training/stabilizer.py` | NaN 恢复、梯度裁剪 |
| `src/predictors/hybrid_predictor.py` | Stage 1 电压→接触角 |
| `src/models/aperture_model.py` | Stage 1 接触角→开口率 |

### 常用命令
```bash
uv run train_two_phase.py --config config/v4.5-standard.json
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/
uv run scripts/dashboard.py
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

## 当前项目状态

- **阶段**: 论文写作中
- **稳定基线**: v4.5-standard（30V 开口率 83.4%，体积误差 <1%）
- **实验中**: v4.6 系列配置
- **已知问题**: 瞬态精度不足（forward 缺少时间门控）、physics_sampling.py 与 Stage 1 有 3 处不一致（待论文完成后修复）
