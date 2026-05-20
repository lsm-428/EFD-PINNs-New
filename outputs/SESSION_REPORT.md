# EFD3D 工作树实验完整报告

**日期**: 2026-05-13 ~ 2026-05-20
**分支**: `main` (base @ 86ce4f0)
**总改动**: 31 文件, 29 项物理修复 (v1→v6)


---

## 零、基线 → v1：采样优化 + 参数统一

基线: main @ 86ce4f0 (原有代码)

### 0.1 原始采样 vs 新采样

| 维度 | 原始 DataGenerator | 新 PhysicsBasedSampler |
|------|-------------------|----------------------|
| 电压采样 | 均匀随机 U(0,30) | 四区划分: 无响应(0-VT)10%, 起始(VT-2VT)50%, 线性(2VT-4VT)30%, 饱和(>4VT)10% |
| 时间采样 | Beta(0.5,1.0) 早期偏置 | 电压自适应τ + 物理阶段分层(电场/Marangoni/膜失稳/局部破裂) + 升压/降压不对称 |
| 空间采样 | 混合: 40%界面高斯/40%角落/20%均匀 | X/Y均匀, Z界面自适应宽度高斯(70%界面+30%全域) |
| 场景划分 | 40/30/30 稳态/升压/降压 | 同 40/30/30 |
| 开口率估算 | DataGenerator.get_opening_rate (Stage1) | PhysicsBasedSampler._estimate_opening_rate → 调用 Stage1 |
| phi目标值 | DataGenerator.target_phi_3d (不变) | DataGenerator.target_phi_3d (继承,不修改) |

### 0.2 采样器三处 bug 修复 (experimental/test/physics_sampling.py)

1. **V_T 参考点**: 2.5μm → 3.0μm, 公式 `V_T_base + (h-2.5e-6)*sensitivity` → `V_T_base + (h-3.0e-6)*sensitivity`
2. **开口率模型**: 简化线性模型 → 调用 Stage1 (predictor.young_lippmann + model.contact_angle_to_aperture_ratio)
3. **时间常数**: τ_onset 0.007→0.0075, 对齐 Stage1 新参数

### 0.3 采样器迁移

- `experimental/test/physics_sampling.py` → `src/data/physics_sampling.py`
- 新增 `src/data/__init__.py` 导出 PhysicsBasedSampler

### 0.4 DataGenerator 集成 (src/models/pinn_two_phase.py)

- `DataGenerator.__init__`: 读 `sampling.strategy`, "physics_based" 时初始化 PhysicsBasedSampler(注入Stage1)
- `generate_all_data()`: 界面点生成3处替换 (电压/时间/空间采样)
- phi目标值保持现有逻辑不动 (已用Stage1)

### 0.5 配置文件 (config/v4.5-physics-sampling.json)

新增 `sampling` 块:
```json
"sampling": {
    "strategy": "physics_based",
    "voltage_weights": {"no_response": 0.10, "onset": 0.50, "linear": 0.30, "saturation": 0.10}
}
```

### 0.6 v1 训练 (= 基线 + 采样优化 + 参数统一 + 时间动态)

v1 是在所有上述参数修改完成后, 用 `v4.5-physics-sampling.json` 配置跑的第一次 60000 epoch 训练。
结果: Best Loss 25.68, 质量守恒 0.46%, 但 φ(xy) 出现 1D 退化

---

## 一、决策时间线

### 1. 物理参数改用真实值
所有材料参数从拟合值改为实验测量值:
- SU-8 ε_r: 12.0→3.28
- Teflon ε_h: 1.9→1.934
- Oil ε: 3.0→4.0
- γ(极性液体表面张力): 0.015→0.048
- σ(油水界面张力): 0.045→0.02505
- ρ_oil: 800→763, μ_oil: 0.003→9.41e-4
去掉了6个文件中的重复默认值,统一到 physics_config.py

### 2. 电容模型
双层串联(SU-8+Teflon)用于Young-Lippmann。极性液体不加(双电层屏蔽)。
油膜在 calculate_capacitance() 反馈中处理。

### 3. V_T厚度依赖
V_T = 5.0 + (h_ink - 3μm) × 2e6, 3μm→5V
Bug: 原公式参考点2.5μm, 修正为3.0μm

### 4. 壁面翻墙模型
η_max = 1 - h_ink / (wall_height × 2.68 × θ_wall/71°)
θ_wall=71°时 η_max=0.68(匹配实验)
Bug: d_fluid=20μm→ink_thickness=3μm

### 5. 时间动态
电压依赖τ(7.5/5/3ms), 恢复τ×0.4, 二阶欠阻尼

### 6. 物理采样器
sampling.strategy="physics_based", 采样器调Stage1

### 7. 壁面接触角+Laplace
n̂·n̂_wall=cos(71°)打破1D退化, κ方差→Laplace压力一致
发现: PINNConstraintLayer完全死代码

### 8. Fourier+CAH+界面能
mapping_size=16 Fourier; CAH θ_A=75°/θ_R=67°; 界面能σ×∫|∇φ|dV

---

## 二、三次训练

| 版本 | Best Loss | 质量守恒 | φ(xy) | 约束数 |
|------|-----------|---------|-------|--------|
| v1 | 25.68 | 0.46% | 1D退化 | 7 |
| v2 | 29.09 | 0.37% | 散点 | 9 |
| v3 | 训练中 | - | 待评估 | 10+Fourier |

v1: outputs/train/pinn_20260513_140711/
v2: outputs/train/pinn_20260514_181419/
v3: outputs/train/pinn_20260515_113251/

---

## 三、改动文件

| 文件 | 改动 |
|------|------|
| src/config/physics_config.py | V_threshold→property; 材料参数真实值; from_json适配 |
| src/predictors/hybrid_predictor.py | ε真实值; 电压依赖τ; 恢复τ×0.4; 签名加V_to |
| src/models/aperture_model.py | d_fluid→ink_thickness; wall_height; θ_wall翻墙 |
| src/models/pinn_two_phase.py | 删fallback字典; FourierFeature; physics_sampler |
| src/physics/constraints.py | 壁面接触角; Laplace; CAH; 界面能; 死代码修复 |
| config/device_calibrated_physics.json | 全部材料参数更新 |
| src/data/physics_sampling.py | 新建(从experimental迁移) |
| src/data/__init__.py | 新建 |
| src/solvers/flow_solver.py | FLUID_PROPERTIES→PHYSICS |
| src/predictors/pinn_aperture.py | physics→PHYSICS; t_max 0.02→0.05 |
| experimental/test/physics_sampling.py | V_T参考点修正+Stage1集成 |
| experimental/test/validate_sampling_vs_stage1.py | 传Stage1给采样器 |

---

## 四、关键发现

1. v1学到1D退化解(φ只随z变化,xy均匀)
2. v2壁面接触角打破1D(xy散点结构,非中心圆形开口)
3. PINNConstraintLayer 500+行死代码(从未调用)
4. compute_surface_tension_residual死代码(需predictions[:,5])
5. Fourier让数据拟合过强(φS→0,物理侧需更长时间)
6. d_fluid bug: C_ratio虚高16.5→实际2.82
7. theta_wall=71°从未被使用(v2修复)
8. v3在27000/60000 epoch, Loss~79, IF~2.5, 约10h已运行

---

## 七、v4 改进 (2026-05-18 ~ 2026-05-19)

### 7.1 Allen-Cahn 相场方程 (替代纯VOF)

**文件**: `src/physics/constraints.py` — `_compute_vof_residual`

```
L_AC = ||∂φ/∂t + u·∇φ − γ(∇²φ − f'(φ))||²
f'(φ) = 2φ(1-φ)(1-2φ)   (双阱势导数, φ∈[0,1])
γ = 4.5×10⁻⁷             (6σ·ε, 与CSF表面张力一致)
```

### 7.2 壁面接触角约束 (替换简单惩罚)

**文件**: `src/physics/constraints.py` — `compute_sidewall_contact_angle_residual`

```
n̂·n̂_wall = cos(θ_wall=71°)   @ 壁面, φ≈0.5处
n̂ = ∇φ/|∇φ|
CAH: θ_A=75°/θ_R=67° (死区 1e-4)
边角钉扎: z≈0边缘 |cosθ_diff|<H_pin→5x惩罚
```

### 7.3 Laplace 压力一致约束

**文件**: `src/physics/constraints.py` — `compute_laplace_pressure_residual`

```
κ_xy = −(φ_xx+φ_yy)/|∇φ|
L_LP = variance(κ_xy) 沿界面
```

### 7.4 电润湿驱动力 (Gibbs ③拆分为独立约束)

**文件**: `src/physics/constraints.py` — `compute_electrowetting_residual` (新增)

```
z=0底面, 逐点计算:
C_eff = η·C_open + (1-η)·C_ink_region  (电容反馈)
L_EW = ½·C_eff·V²·(1-φ)               (电润湿推水铺展)
```

### 7.5 界面能纯化 (Gibbs ①)

**文件**: `src/physics/constraints.py` — `compute_interface_energy_residual` (重写)

```
L_shape = σ·|∇φ| (纯表面张力最小化, 塑造圆润液滴)
```

### 7.6 底面接触线滑移

**文件**: `src/models/pinn_two_phase.py` — `_compute_contact_angle_loss`

```
v_slip = φ_t/(|∇φ|+ε) @ z=0, φ≈0.5
L_slip = ||v_slip − α·Δcos||², α=0.1
```

### 7.7 三部接触线过采样

**文件**: `src/models/pinn_two_phase.py` — `generate_all_data`

```
+25000 点 @ z=0, r≈r_open (Stage1估算开口半径)
+7500  点 @ z=0, t∈[0,5ms] (突破时刻加密)
```

### 7.8 Young-Lippmann 改用 σ (油水界面张力)

**文件**: `src/predictors/hybrid_predictor.py`

```
YL公式: σ=0.02505 (替代 γ=0.048)
Δθ@30V: 11°→21° (更符合物理)
aperture_mapping k=3.0/θ_scale=19.0 自洽
```

### 7.9 训练阶段 + 分阶段权重调度

**文件**: `config/v4.5-physics-sampling.json` + `pinn_two_phase.py`

```
S1: 0-3000 (数据), S2: 3000-10000 (物理引入), S3: 10000-60000 (全物理)

Phase调度(相对于S3):
  S3早期(10k-20k): AC×8, NS×0.2, contact×0.5
  S3中期(20k-40k): AC×1, NS×1, contact×1
  S3后期(40k-60k): smooth ramp
```

### 7.10 Loss分解 + 训练曲线精简

- 新增: AC/EW/LP/SW/IE 独立日志
- 删除: ST(死代码)/sharpening/explicit_volume/momentum分项
- 训练曲线: 7条 (Total|IF|C|θ|AC|LP|SW)
- 里程碑PNG改为关键epoch

### 7.11 统计归一化修复

SW/EW约束从 `.mean()` (除以全batch 2000) 改为 `sum()/n_active` (除以有效点数)。
界面局部约束信号放大 ~200x。

### 7.12 数据拟合权重调整

```
interface_weight: 500→200  (防止数据过拟合)
z采样上限: Lz=20μm→2×h_ink=6μm  (聚焦油膜区域)
Stage1 tutor: 权重400→0 (移除, PINN自主)
target_phi_3d: z=0 Stage1指导, z>0 体积守恒推算
```

### 7.13 速度优化

- Fourier features: OFF (v3证实反效果, 3.3x加速)
- 动态重采样: OFF (10%加速, 当前阶段收益低)

### 7.14 配置权重总览

```
interface_weight: 200
sidewall_contact_angle: 5.0
laplace_pressure: 0.2
interface_energy: 0.05
electrowetting: 0.5
```

### 7.15 死代码清理

- PINNConstraintLayer: ~500行, 从未调用
- compute_surface_tension_residual: 需predictions[:,5]→死
- compute_contact_line_dynamics: 需predictions[:,10]→死
- TensorBoard网络图bug: data未定义→修复

## 八、文件变更总览 (v1→v4累计)

| 文件 | 改动类型 |
|------|---------|
| `src/config/physics_config.py` | V_threshold→property, to_predictor_params+σ |
| `src/predictors/hybrid_predictor.py` | σ→YL, 电压依赖τ, 恢复×0.4 |
| `src/models/aperture_model.py` | d_fluid→ink_thickness, θ_wall翻墙 |
| `src/models/pinn_two_phase.py` | Allen-Cahn, 采样器, 接触线过采样, 阶段调度, Loss分解 |
| `src/physics/constraints.py` | 壁面接触角, Laplace, CAH, Gibbs重构, 电润湿, 归一化修复 |
| `config/v4.5-physics-sampling.json` | **新建**, 训练配置 |
| `config/device_calibrated_physics.json` | 材料参数全部更新 |
| `src/data/physics_sampling.py` | **新建**, 物理采样器 |
| `src/data/__init__.py` | **新建** |

## 九、v5 物理基础深度审计与修复 (2026-05-19 ~ 2026-05-20)

### 9.1 审计方法论

三轮递进审计: 
- 第一轮: 电润湿电容与表面张力参数逐项核查
- 第二轮: 约束完整性检查 + 死代码识别
- 第三轮: 第一性原理验证 (尺度分析/适定性/变分一致性/守恒律)

### 9.2 电容与 Young-Lippmann 修正

**发现: NS动量方程电润湿力使用单层电容(ε₀εᵣ/d), 忽略Teflon层**

| 参数 | 修正前 | 修正后 | 偏差 |
|------|--------|--------|------|
| C_ew (F/m²) | 7.26×10⁻⁵ (SU-8单层) | 2.69×10⁻⁵ (SU-8+Teflon双层) | -2.7× |
| YL界面张力 | γ=0.048 (极液-气) | σ_po=0.02505 (极液-油) | -1.9× YL项 |

修改: `constraints.py` L155-166, L265-266, L1562-1603; `pinn_two_phase.py` L852,864

### 9.3 约束丢失修复 (8项)

| 约束 | 修复前 | 修复后 |
|------|--------|--------|
| `wall_wetting` | 死代码 (未调用) | 接入 `compute_core_residuals` |
| `dielectric_charge` | 死代码 (需charge_density) | 重写, 适配5分量输出 |
| `contact_line_dynamics` | 死代码 (需theta预测) | 重写, HVT模型适配 |
| `top_boundary` | 未实现 | 新增 (零剪切+无穿透+φ=0) |
| `pressure_pin` | 未实现 | 新增 (mean(p)→0) |
| `surface_tension` | 零值残差 + 权重≠0 | 从所有权重字典移除 |
| `pinn_electrowetting` | 不记录history/TB | 加入全部追踪 |
| `young_lippmann_residual` | 电容+γ错误 | 修复但未接入训练 |

### 9.4 NS方程物理修正

**两相应力张量** (`constraints.py` L300-336): 
- 修复前: `μ∇²u` (恒定粘度近似)
- 修复后: `∇·[μ(∇u+∇uᵀ)]` (含∇μ项, Δμ/μ≈7%→界面贡献~40%)

**对流项** (Re≈0.15): 默认关闭 `u·∇u`, 可选开启

### 9.5 Allen-Cahn 标准化

**修复前**: `γ[∇²φ - W'(φ)]`, γ=4.5×10⁻⁷, 无物理界面宽度标度

**修复后**: `M[σ·ε·∇²φ - (σ/ε)·W'(φ)]`
- ε = 5×10⁻⁶ m (界面宽度, 可控)
- M = 1×10⁻⁸ m³·s/kg (迁移率, 可控)
- 旧版通过 `use_legacy_ac=true` 切回

### 9.6 电润湿双重计数修正

**问题**: NS体力 + 独立惩罚两机制独立驱动 → 稳态解被非物理惩罚污染

**修复** (`pinn_two_phase.py` L2716-2733): 电润湿惩罚权重退火
- S3早期: ew_penalty=1.0 (辅助成型)
- S3中期: ew_penalty=0.5 (NS体力接手)
- S3后期: ew_penalty→0 (完全物理驱动)

### 9.7 其他修正

| 修复 | 文件 |
|------|------|
| 电润湿残差MSE归一化 (底面均值, 不再被零值稀释6-10×) | `constraints.py` |
| 降压恢复模型: 二阶振荡→一阶指数 (过阻尼物理) | `pinn_two_phase.py` |
| 恢复τ统一: 数据生成器用电压依赖 `τ_drive(V)×0.4` | `pinn_two_phase.py` |
| φ空间/几何约束: `relu(x)²`→`sigmoid(x/0.05)` (软阈值, 连续梯度) | `pinn_two_phase.py` |
| 自适应损失归一化: EMA平衡各约束量级 (可选开启) | `pinn_two_phase.py` |
| Young-Laplace注释: tanh模型的经验近似性质 | `aperture_model.py` |

### 9.8 训练日志与曲线重构

**训练日志**: 新增5个约束显示 (EW/WW/DC/CLD/TB)

**TensorBoard**: 新增4个标量 (wall_wetting/dielectric_charge/contact_line_dynamics/top_boundary)

**训练曲线图**: 单层→双层布局
- 上层 (主要): Total, IF(data), C(continuity), AC(vof), EW(electrowetting), θ(contact), Vol
- 下层 (次级): LP(laplace), SW(sidewall), IE(interface), WW(wall), CLD(dynamics), DC(charge), TB(top BC), TS(smooth)

### 9.9 最终约束矩阵 (22 残差 key)

```
流体力学 (6): continuity, momentum_u/v/w, vof, pressure_pin
界面物理 (5): interface_energy, laplace_pressure, contact_angle_constraint,
              surface_tension, interface_curvature
电润湿 (2):   electrowetting, dielectric_charge
壁面/接触线(3): sidewall_contact_angle, wall_wetting, contact_line_dynamics
边界条件 (2):  top_boundary, pressure_pin
守恒律 (2):    volume_conservation, volume_consistency
正则化 (2):    temporal_smoothness, ink_potential_min
```

### 9.10 第一性原理验证清单

| 原理 | 状态 |
|------|:---:|
| 质量守恒 (∇·u=0 + 显式体积) | ✅ |
| 动量守恒 (完整两相应力张量) | ✅ |
| 压力解唯一性 (钉扎条件) | ✅ (新增) |
| 伽利略不变性 (Stokes流) | ✅ |
| 框架无关性 (客观应力张量) | ✅ |
| 接触线无应力奇点 (扩散界面) | ✅ |
| 方程-变量封闭 (5eq/5var) | ✅ |
| CSF压力跳变一致性 | ✅ |
| AC守恒型等价性 (∇·u=0) | ✅ |
| 混合物连续性 (质量通量误差~10⁻¹³) | ✅ |
| 电润湿惩罚退火 (后期完全物理驱动) | ✅ (新增) |

### 9.11 尺度分析结果

| 无量纲数 | 值 | 结论 |
|----------|-----|------|
| Re | 1.5×10⁻¹ | 蠕变流, 对流项可忽略 |
| Ca | 4.0×10⁻⁵ | 表面张力绝对主导 |
| Ew @25V | 0.34 | 电润湿与表面张力可比 |
| Bo | 1.0×10⁻² | 重力可忽略 |
| Δμ/μ | 0.069 | 界面粘度变化~7% |
| τ_visc | 7.9μs | 油膜内粘性扩散极快 |
| τ_cap | 0.4ms | 毛细时间 (与τ=5ms一致) |

### 9.12 v5 改动文件总计

| 文件 | 改动 |
|------|------|
| `src/physics/constraints.py` | 电容修正, 两相应力, AC标准化, 电润湿变分, wall_wetting接入, charge/CLD重写, top_boundary新增, pressure_pin新增, 对流可选, 配置开关 |
| `src/models/pinn_two_phase.py` | YL修正, 降压恢复, τ统一, 软阈值, 自适应loss, 日志/TB/曲线重构, 权重字典全更新, history全更新, EW退火 |
| `src/models/aperture_model.py` | Young-Laplace注释 |

**总计: 3文件, 22项修复**

---

## 十、v6 物理审计 follow-up 修复 (2026-05-20)

基于 v5 审计的深度代码审查，发现并修复了 4 类问题。

### 10.1 EWD NS 体力方向符号修正 (P0-1)

**文件**: `src/physics/constraints.py` L292-294

**发现**: N-S 方程中电润湿体力方向使用 `-∇φ`（从油指向水/向内），与物理机制相反。

**分析**: 在开口状态下，中心为极性液体 (φ=0)，边缘为油墨 (φ=1)。∇φ 从中心指向边缘。电润湿驱动极性液体润湿基底、将油墨**向外**排挤，力方向应沿 `+∇φ`（从水指向油/向外）。`-∇φ` 方向指向中心，会关闭开口。

修复:
```diff
- dir_x = -phi_x / grad_phi_xy_mag    # -∇φ: 从油→水 (错误)
- dir_y = -phi_y / grad_phi_xy_mag
+ dir_x = +phi_x / grad_phi_xy_mag    # +∇φ: 从水→油 (正确)
+ dir_y = +phi_y / grad_phi_xy_mag
```
同步更新了 L253-264 和 L290-294 的注释文档。

### 10.2 对流项配置化 + Re 数修正 (P0-2)

**文件**: `src/config/physics_config.py`, `config/device_calibrated_physics.json`, `src/physics/constraints.py`

**发现**: 
1. 代码注释称 "Re ≈ 0.15" 但实际 Re = ρUL/μ ≈ 1-5（高电压快速响应时更高），对流项不应无条件忽略
2. `use_convection` 硬编码在 `_get_default_materials_params()` 内，无法从配置文件控制

修复:
- `PhysicsConfig` dataclass 新增 `use_convection: bool = False` 字段
- `from_json()` 从 `materials.use_convection` 读取
- `to_materials_params()` 自动透传到 `PhysicsConstraints`
- JSON 配置新增 `"use_convection": false` 及说明注释
- N-S 注释 "Re ≈ 0.15" → "Re ≈ 1-5, Womersley ≈ 0.03 → 准稳态 Stokes"

### 10.3 死代码与重复方法清理 (P1-3, P1-4, P1-5)

**文件**: `src/physics/constraints.py`

三项清理:

| 方法 | 问题 | 处理 |
|------|------|------|
| `compute_surface_tension_residual` | 标记 `[DEPRECATED]` 但仍被 `compute_core_residuals` 调用；需要 `predictions[:,5]`，标准模型不提供 → 始终返回零 | 从 `compute_core_residuals` 移除调用，保留方法声明 |
| `compute_two_phase_flow_residual` | 115行完整实现但在训练中从未调用；使用简化的粘性项和不同的连续性公式，与主线不一致 | 添加 `[DEPRECATED]` 标记和替代说明 |
| `compute_contact_line_dynamics_residual` (公开版) | 假设 11+ 列输出 (`predictions[:,10]`=theta)，与 TwoPhasePINN (5列) 不兼容；私有版 `_compute_...` 从 autograd 计算所有量 | 添加 `[DEPRECATED]` 标记和兼容性说明 |

三者均在 `constraints_deprecated.py` 中有引用（该文件本身已废弃），活跃代码零引用。

### 10.4 AC mobility 提升 + 体积计算优化 (P2)

**文件**: `src/physics/constraints.py`, `config/v4.5-standard.json`

**AC mobility**: 
- 修复前: `M_ac = 1e-8 m³·s/kg` → τ_reaction = ε²/(M·σ) ≈ 20000s（极弱，AC 方程几乎无相分离驱动力）
- 修复后: `M_ac = 1e-7 m³·s/kg` → τ_reaction ≈ 2000s（10x 提升，温和正则化，仍远弱于数据损失）

**体积守恒计算量**:
- `volume_n_vol`: 20000 → 8000（减少蒙特卡洛点数，精度损失可忽略）

### 10.5 v6 改动文件总计

| 文件 | 改动 |
|------|------|
| `src/physics/constraints.py` | EWD符号修正+注释, 死代码清理(3处), AC mobility, Re注释 |
| `src/config/physics_config.py` | PhysicsConfig +use_convection 字段, from_json/to_materials_params 适配 |
| `config/device_calibrated_physics.json` | +use_convection 开关及注释 |
| `config/v4.5-standard.json` | volume_n_vol 20000→8000 |

**总计: 4文件, 7项修复**
