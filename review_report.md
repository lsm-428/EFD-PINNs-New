# EFD3D 全面代码审核报告（第 4 轮 — 深度审核）

**审核日期**: 2026-06-08
**审核范围**: 全部 44 个 Python 源文件 + 配置文件 + 测试文件
**代码规模**: ~13,900+ 行 Python 代码，8 个子包
**最新 commit**: b65e1ba

---

## 架构总览

EFD3D — 基于 PINN 的 3D 电润湿两相流仿真框架

- **src/config/** (661 行): 物理参数单一来源 + 路径管理
- **src/models/** (5221 行): TwoPhasePINN + Stage1 开口率模型
- **src/physics/** (1571 行): 物理约束引擎 (N-S/VOF/EW/AC)
- **src/predictors/** (1393 行): HybridPredictor + PINNAperturePredictor
- **src/training/** (635 行): 训练组件/调度器/稳定器
- **src/data/** (509 行): 物理优化采样器
- **src/solvers/** (2439 行): CFD 求解器 (传统验证用)
- **src/utils/** (443 行): 梯度工具/日志/模型工具
- **src/dashboard/** (~2000 行): Streamlit/Panel 交互式面板
- **evaluate.py** (1649 行): 专业评估套件
- **train_two_phase.py** (966 行): 训练入口
- **tests/** (18 个文件): 单元测试 + 集成测试

**核心数据流**: PhysicsConfig → PHYSICS → PhysicsConstraints → Trainer → TwoPhasePINN

---

## I. 严重问题（CRITICAL）

### C-1: test_pinn_complete.py 安全风险 (weights_only=False)

**文件**: tests/test_pinn_complete.py L58

    torch.load(checkpoint_path, map_location=device, weights_only=False)

evaluate.py 已改为 weights_only=True（安全修复），但测试文件仍用 weights_only=False。
存在反序列化任意代码执行风险（CVE-2024-3519）。

**修复**: 改为 weights_only=True。

---

### C-2: evaluate.py 修改全局 PHYSICS 字典

**文件**: evaluate.py L86-91

    if "physics" in config:
        for k, v in config["physics"].items():
            PHYSICS[k] = v

直接修改全局 PHYSICS 字典。虽然注释说 evaluate.py 是独立运行的脚本，
但在以下场景会出错：
- 同一进程中评估多个不同配置的模型（模型 A 的参数污染模型 B）
- 评估后继续训练，被污染的 PHYSICS 影响训练
- 多线程/多进程评估

**修复**: 使用 copy.deepcopy(PHYSICS) 创建评估时的临时副本。

---

### C-3: HybridPredictor._load_config() 未读取 V_T_base

**文件**: src/predictors/hybrid_predictor.py L157-182

_load_config() 从 materials 读取了 theta0/epsilon_r/gamma/sigma/d/epsilon_h/d_h，
但没有读取 V_T_base 和 V_threshold。JSON 中 materials.V_T_base=3.0，
但 HybridPredictor 的 young_lippmann() 使用:

    V_threshold = self.params.get("V_threshold", self.params.get("V_T_base", 5.0))

由于 _load_config 没有更新 V_threshold 和 V_T_base，它们始终使用回退默认值 5.0。
而 PhysicsConfig 从 JSON 读到的 V_T_base=3.0。

**影响**: HybridPredictor（Stage 1 教师）使用 V_T=5.0，而 PINN（Stage 2 学生）使用 V_T=3.0。
30V 时 V_eff 差 2V，EW 力差约 17%。

**修复**: 在 _load_config 中添加 V_T_base 的读取。

---

### C-4: constraints.py 中 V_T_base 回退默认值仍为 5.0

**文件**: src/physics/constraints.py

虽然 JSON 中 V_T_base=3.0 且 PhysicsConfig 正确读取，
但 constraints.py 中多个位置的回退默认值仍为 5.0:

    V_T = self.materials_params.get("V_T_base", 5.0)

当 materials_params 正确传入时（通过 get_materials_params()），
V_T_base=3.0 是正确的。但如果 materials_params 不完整或被覆盖，
回退到 5.0 就会产生与 PhysicsConfig 不一致的结果。

**修复**: 将回退默认值从 5.0 改为 3.0，与 PHYSICS 默认值一致。

---

### C-5: HybridPredictor 回退默认值与校准值不一致

**文件**: src/predictors/hybrid_predictor.py L37-52

| 参数 | 回退默认值 | JSON 校准值 |
|------|-----------|------------|
| gamma | 0.048 | 0.015 |
| tau | 0.015 | 0.0119 |
| epsilon_r | 3.28 | 12.0 |
| d | 4e-7 | 8e-7 |
| tau_recovery_factor | 0.4 | 0.85 |

当 PHYSICS 模块 import 失败时（如路径问题），HybridPredictor 使用
这些回退值，与校准后的 JSON 参数严重不一致。

**修复**: 将回退默认值更新为与 JSON 校准值一致。

---

## II. 重要问题（IMPORTANT）

### I-1: constraints.py 中重复的梯度计算方法

**文件**: src/physics/constraints.py

constraints.py 中有多个功能相似的梯度计算方法:
- _compute_laplacian (L340): 使用 autograd 循环
- safe_compute_gradient (L1074): 带异常保护的梯度计算
- safe_compute_laplacian (L1493): 调用 safe_compute_gradient
- safe_compute_laplacian_spatial (L1053): 空间拉普拉斯
- safe_compute_hessian (L1513): 海森矩阵

虽然 _compute_all_gradients (L1155) 已统一了核心梯度计算，
但旧方法仍在多处被直接调用（如 compute_navier_stokes_residual 的回退路径）。
建议统一为 src/utils/gradients.py 中的共享实现。

---

### I-2: pinn_two_phase.py 中 _analytical_contact_angle 公式可读性差

**文件**: src/models/pinn_two_phase.py L724-738

zeta=1.0 时，二阶欠阻尼公式退化为:
    theta_t = theta_eq + (theta0 - theta_eq) * exp(-t/tau) * (cos(0) + 1.0*sin(0))
     = theta_eq + (theta0 - theta_eq) * exp(-t/tau)

这是一阶指数，但代码形式上是二阶欠阻尼的退化情况。
HybridPredictor.dynamic_response() 使用纯一阶指数。
两者在 zeta=1.0 时数学上等价，但代码形式不一致。

**修复**: 在 _analytical_contact_angle 中添加 zeta=1.0 的特殊情况处理。

---

### I-3: flow_solver.py 处于闲置状态

**文件**: src/solvers/flow_solver.py (2439 行)

根据 CLAUDE.md R10 规则，闲置模块暂时保留。
但 2439 行代码占据大量维护成本。

**修复**: 添加明显的 DEPRECATED 标记和废弃时间表。

---

## III. 改进建议（SUGGESTIONS）

### S1: 补充测试用例

当前改动涉及核心采样策略和物理方程，但缺少测试覆盖。建议:
- 测试垂直采样分布是否符合预期
- 测试阶段转换行为（Stage 1->2->3 权重切换）
- 测试 V_T_base 在 PhysicsConfig/constraints/HybridPredictor 中的一致性
- 测试 EW 力 z_decay 衰减尺度（lambda_debye=50nm vs h_ink=3um）

### S2: 添加配置参数范围验证

n_vertical_samples 过大时可能导致 z_points 为空。
建议在 PhysicsConfig 中添加参数验证。

### S3: 统一 random.seed 的使用

pinn_two_phase.py 中既用 random.seed() 又用 np.random.seed()，
还手动 import random 在函数内部。建议统一使用 np.random.seed()。

### S4: 考虑添加 __repr__ 到 PhysicsConfig

方便调试时查看当前配置状态。

---

## IV. 代码质量亮点

1. **物理参数单一来源**: physics_config.py 的三层体系设计良好
2. **模块职责清晰**: 7 个子包分工明确
3. **文档完善**: 每个模块都有详细的 docstring 和中文注释
4. **配置版本管理**: v4.5 -> v4.6 配置演进清晰
5. **安全修复及时**: evaluate.py 已改为 weights_only=True
6. **训练稳定性**: TrainingStabilizer 提供 NaN 恢复、预热、梯度裁剪
7. **物理约束完整**: constraints.py 涵盖 N-S/VOF/EW/AC 全套物理方程
8. **评估套件专业**: evaluate.py 提供 10+ 种可视化分析
9. **测试覆盖**: 18 个测试文件覆盖物理一致性、梯度传播、模型鲁棒性
10. **代码规范**: ruff + black + pre-commit hooks 保证代码质量

---

## V. 总结

| 类别 | 数量 | 优先级 |
|------|------|--------|
| CRITICAL | 5 | 立即修复 |
| IMPORTANT | 3 | 下一版本修复 |
| SUGGESTION | 4 | 后续迭代 |

**最关键的 3 个问题**:

1. **C-1 (weights_only=False)**: 安全风险，可能被恶意 checkpoint 利用
2. **C-3 (HybridPredictor 未读 V_T_base)**: Stage 1 教师使用错误的阈值电压
3. **C-5 (HybridPredictor 回退默认值不一致)**: 当 PHYSICS 不可用时使用错误参数

**建议修复顺序**: C-1 -> C-3 -> C-5 -> C-2 -> C-4
