# EFD3D 实验版本改进摘要 (v1 ~ v16)

**日期**: 2026-05-13 ~ 2026-05-26
**代码基线**: main @ 86ce4f0

---

## v1 — 物理采样器 + 参数统一

**核心改动**:
- 新建 `PhysicsBasedSampler`（四区电压采样 + 自适应τ时间采样 + Z界面高斯采样）
- 所有材料参数从拟合值改为实验测量值（ε_r, ε_h, ρ, μ, σ, γ）
- V_T 公式参考点 2.5μm → 3.0μm
- `target_phi_3d` 调用 Stage1 而非线性模型

**结果**: Loss 25.68, 体积守恒 0.46%, 但 φ(xy) 1D 退化

**关键发现**: φ 只随 z 变化，xy 方向均匀 → 缺少 xy 方向的结构化约束

---

## v2 — 壁面接触角 + Laplace 压力

**核心改动**:
- 新增 `compute_sidewall_contact_angle_residual`（n̂·n̂_wall = cos(71°)）
- 新增 `compute_laplace_pressure_residual`（κ_xy 方差 → Laplace 压力一致）
- 发现 PINNConstraintLayer ~500 行死代码

**结果**: Loss 29.09, 体积守恒 0.37%, xy 散点结构出现

**关键发现**: 壁面接触角打破 1D 退化；Fourier 让数据拟合过强

---

## v3 — Fourier + CAH + 界面能

**核心改动**:
- 新增 Fourier Feature 映射（mapping_size=16）
- CAH 接触角滞后（θ_A=75°/θ_R=67°）
- 界面能 σ×∫|∇φ|dV

**结果**: 训练中（27000/60000 epoch, Loss~79）

---

## v4 — Allen-Cahn 相场 + 分阶段调度

**核心改动**:
- VOF → Allen-Cahn 相场方程: `∂φ/∂t + u·∇φ = γ(∇²φ - f'(φ))`
- 三阶段权重调度: S1(0-3000) 数据, S2(3000-10000) 物理引入, S3(10000-60000) 全物理
- 统计归一化修复: `.mean()` → `sum()/n_active`（界面约束信号放大 ~200x）
- interface_weight: 500→200, Fourier: OFF, 重采样: OFF

**关键发现**: AC 方程替代纯 VOF 是正确方向；分阶段调度必要

---

## v5 — 物理深度审计（22 项修复）

**核心改动**:
- **电容修正**: NS 电润湿力从单层电容（SU-8）→ 双层串联（SU-8 + Teflon），C_ew ↓2.7×
- **YL 界面张力**: γ=0.048（极液-气）→ σ=0.02505（极液-油），Δθ@30V: 11°→21°
- **两相应力张量**: `μ∇²u` → `∇·[μ(∇u+∇uᵀ)]`（含 ∇μ 项）
- **AC 标准化**: `γ[∇²φ - W'(φ)]` → `M[σ·ε·∇²φ - (σ/ε)·W'(φ)]`（ε=5μm, M=1e-8）
- **电润湿双重计数修正**: EW 惩罚权重退火（S3 早期 1.0 → 后期 0）
- 新增 `wall_wetting`, `dielectric_charge`, `contact_line_dynamics`, `top_boundary`, `pressure_pin` 约束
- 死代码清理: `compute_surface_tension_residual` 从 core 移除

**约束矩阵**: 22 残差 key（流体 6 + 界面 5 + 电润湿 2 + 壁面 3 + BC 2 + 守恒 2 + 正则 2）

**尺度分析**: Re≈0.15（蠕变流）, Ca≈4e-5（表面张力主导）, Ew@25V=0.34（电润湿可比）

---

## v6 — 审计 follow-up（7 项修复）

**核心改动**:
- **EWD 力方向确认**: `-∇φ`（对抗 CSF 力）正确，`+∇φ` 会导致开口率丢失
- **对流项配置化**: `use_convection` 字段加入 `PhysicsConfig`，Re 注释修正为 1-5
- **AC mobility**: 1e-8 → 1e-7（τ_reaction 20000s → 2000s）
- **体积计算优化**: volume_n_vol 20000 → 8000
- 死代码标记: `compute_two_phase_flow_residual`, `compute_contact_line_dynamics_residual`（公开版）

---

## v7 — 开口率丢失修复（9 项修复）

**核心改动**:
- **sigmoid → relu²**: 4 处软阈值全部替换（梯度消失根因，φS 472→24）
- **Fourier ON**: 标准 FC 网络无法表达径向过渡，Fourier 是必要条件
- **s3_smooth_span 可配置化**: 硬编码 5000 → 配置项
- **v4.6-standard.json**: 合并 v4.5-standard + v4.5-physics-sampling 最佳实践

**冒烟验证** (2000 epoch): V=0→30V, η: 7%→28%, 径向开口形成

**训练铁律 #1**: Fourier + relu² 是两个必要条件，缺任一都无法学到径向开口

---

## v8 — 训练收敛诊断（3 条根本原因）

**核心改动**:
- **resample_interval: 5000 → 0**（禁用重采样 — Stage 1 近似目标与 PDE 精确解不可调和）
- **z 采样扩展**: [0, 2h_ink=6μm] → [0, Lz=20μm]（物理配点必须覆盖全空间域）
- **EW 力单位修复**: `4φ(1-φ)`（无量纲）→ `|∇φ|`（[1/m]），线力 → 体力，量级修正 ~10¹¹
- **Stage 1 退火归零**: tutor_min_factor 0.2→0, anneal_span 5000→30000

**训练铁律**:
1. 禁止重采样
2. 物理配点必须覆盖全域
3. 早期开口率是领先失效指标

**验证**: EW 从 ~1e-13 → 2.7e-08（+10⁵），Loss -75%, C -85%, AC -86%

---

## v9 — 底面润湿约束探索（4 轮冒烟）

**核心问题**: 油墨相分离丢失 + z 向扩散

| 轮次 | 配置 | 关键发现 |
|:----:|------|---------|
| noew | EW 独立项=0 | interface loss 太强压制相分离 |
| pure-pde | interface=0 | PDE 约束不足，模型振荡 |
| +BW | 新增 bottom_wetting | BW 与 AC 梯度对抗（φ_z 冲突） |
| balanced | 弱监督+弱BW+PDE | 折中方案 |

**核心认知**: Stage 1 target_phi_3d 既是"脚手架"也是"毒药"。最佳策略：弱监督引导 z=0 油墨分布，PDE 控制 z>0 界面。

**新增代码**: `compute_bottom_wetting_residual`（~100 行）

---

## v10 — 统一相场润湿 BC（设计）

**核心思想**: 用单一能量泛函 F[φ] 统一所有界面物理

```
F[φ] = ∫[σ/ε·f(φ) + σ·ε/2·|∇φ|²]dV + ∫f_w(φ)dS
f_w(φ) = -σ·cos(θ_eq)·φ²(3-2φ)
自然 BC: ε·n·∇φ - cos(θ_eq)·6φ(1-φ) = 0
```

**关键优势**: f_w'(φ) ∝ 6φ(1-φ) 在体相自动消失，8 个碎片化项 → 2 个统一项

**⚠️ 问题**: 设计报告写了，但代码从未实现。PFW 权重一直空转。

---

## v11 — 润湿参数物理修正

**核心改动**:
- 侧壁接触角: θ_polar=71° → **θ_wall_teflon=110°**（Teflon 污染侧壁，亲油）
- 墙顶接触角: 55° → 71°（原生 SU-8）
- 统一 BC 修复: 界面连续加权 `exp(-100*(φ-0.5)²)` + BC 量级与 bulk AC 对齐

**物理分析**: V=30V 时底面疏油 + 侧壁亲油 → 油墨沿壁流动 → 单角落汇聚

**结果**: PFW 稳定在 ~2e-7（不爆炸），但 φ≈0.74 常数（PINN 无法从零学 6D 映射）

---

## v12 — Stage 1 老师回归

**策略**: interface=200 + AC=10 + PFW=5 + NS=0

**结果**: φ 坍缩到 ≈0.5 常数，无相分离

**结论**: 6 次冒烟测试证明 soft penalty PDE 太弱，无法从平凡解推出

---

## v13 — 硬约束策略（关键突破）

**核心思想**: IC + top BC 编码进网络输出构造（hard constraint），而非 soft penalty

```python
phi = phi_learned * (1.0 - z_norm)          # top BC: φ(z=Lz)=0
phi = phi_ic + t_norm * (phi - phi_ic)       # IC: φ(t=0)=φ_IC(z)
```

**实验 1** (smoke-hard, 2000 epoch): AC 从 63.7→8.95（-86%），φ 达到 0.001/0.998

**实验 2** (hard-long, 5000 epoch): AC 突破 -87%，相分离成功，体积 1.82x

**关键发现**:
- 硬约束是解决平凡解的关键
- AC "突破"现象：关键 epoch 突然暴跌 86-87%
- 电压响应几乎为零（模型忽略 V_from/V_to）

---

## v14 — 底面 target 修正 + φ 三值物理

**核心改动**:
- **target_phi_3d 修正**: z=0 开口区 φ=0.5 → **0.0**（固体表面只有 0/1）
- **sigmoid 温度**: T=5 → **T=20**（消灭无意义灰区，φ 三值化）

**电压梯度诊断**: dφ/dV 非零（-2.94e-3），方向正确，不是架构问题

**物理**: z=0 固体表面 → r<r_open: φ=0, r=r_open: φ=0.5, r>r_open: φ=1

---

## v15 — 统一相场 BC 真正实现

**根因**: v10 设计从未编码，PFW 一直空转

**实现**: `_compute_unified_wetting_bc()`（~50 行）
- 底面 BC: ε·φ_z - cos(θ_eq)·6φ(1-φ) = 0
- 侧壁 BC: ε·n_wall·∇φ - cos(θ_wall_teflon)·6φ(1-φ) = 0
- 符号修正: 报告写 `+cos`，正确为 `-cos`（外法向 n=(0,0,-1)）

**训练** (pfw-z0driven, 5000 epoch): PFW 非空转 ✓（4.03→0.3），Vol 1.93→0.72

**⚠️ 问题**: epochs=5000=stage2_epochs，Stage 3 零帧执行

---

## v16 — 单角墨滴 target

**核心改动**: η≥0.45 时 target_phi_3d 从中心开口 → 单角落墨滴模型

```
r_blob = sqrt(4·V_ink / (π·h_edge)), h_edge = h_ink/(1-η)
φ = 0.5·(1 - tanh((r_c - r_blob)/w))
```

**物理验证**: η=0.30→中心开口, η=0.50→角落汇聚（相变）

---

## 训练铁律汇总

| 铁律 | 来源 | 后果 |
|------|------|------|
| 禁止重采样 | v8 | Loss 振荡，η_max≈5% |
| 物理配点覆盖全域 [0, Lz] | v8 | 油墨入侵极性液体区 |
| Fourier + relu² 必须同时开启 | v7 | 无法学到径向开口 |
| 硬约束解决平凡解 | v13 | φ 不坍缩到常数 |
| sigmoid 温度 T≥20 | v14 | φ 三值化 |
| Stage 3 必须执行 | v15 bug | 全物理约束从未生效 |
| 早期开口率是领先指标 | v8 | η_max<20% = EW 力未打开界面 |
| blend 公式必须释放 z=0 底面 | v18 | z=0 被压制→均匀解 φ≈0.8 |
| IC 采样必须覆盖全域 [0, Lz] | v18 | 上层极性液体区域缺失 IC 约束 |
| early_time target 必须用 phi_IC(z) | v18 | 全 1 target 扭曲初始条件 |

---

## v17 — 代码 bug 修复 + 完整 S3 训练

**日期**: 2026-05-26 ~ 2026-05-27

**代码审查结论**：计划中的 Bug 1-6 已全部在之前的 session 中修复。

**配置修复**（v4.6-pfw-z0driven.json）：
- `epochs`: 5000 → 60000（Stage 3 完整执行）
- Stage 3 物理权重设为非零

**训练 pinn_20260527_002933**（已停止，epoch 44200/60000）：
- S3 后期进入平台期，Loss ~420-480，φS→0.00，EW→0.00
- **诊断结论**：模型不可逆地退化到均匀解，旧 blend 公式是根因

---

## v18 — 硬约束 blend 修复 + IC/early_time 全域采样（关键突破）

**日期**: 2026-05-27

### 根因分析

**旧 blend 公式**（`forward()` L311）：
```python
phi = phi_ic + t_norm * (phi - phi_ic)
```
在 z=0 处 `phi_ic≈0.998`，t_norm<1 时把输出拉向 1.0。
V=30V 远处需要 φ≈0，但旧 blend 最小只能到 ~0.8（t_norm=0.2 时）。
模型被夹在 target=0.0 和 hard≈0.8 之间，妥协到 φ≈0.798。

**数据分布佐证**：
- z=0 底面 75% 数据要求 φ≈0，25% 角落数据要求 φ≈1
- 旧 blend 下模型无法达到 φ<0.8，导致均匀解

### 修复方案

#### 修复 1：新 blend 公式

```python
z_mask = (z_norm > 1e-6).float()  # z=0→0, z>0→1
blend = 1.0 - z_mask * (1.0 - t_norm)  # z=0→1, z>0→t_norm
phi = (1.0 - blend) * phi_ic + blend * phi
```

| 位置 | 旧 blend | 新 blend |
|------|----------|----------|
| z=0, 任意t | φ→1.0（被压制） | φ=φ_model（完全自由） |
| z>0, t=0 | φ=φ_ic | φ=φ_ic（不变） |
| z>0, t>0 | φ 混合 | φ 混合（不变） |
| z=Lz | φ=0 | φ=0（不变） |

#### 修复 2：IC 采样扩展到全域

```python
# OLD: z = np.random.rand() * self.h_ink * 2  # [0, 6μm]
# NEW: z = np.random.rand() * self.Lz            # [0, 20μm]
```
物理配点必须覆盖全空间域，否则上层极性液体区域没有 IC 约束。

#### 修复 3：early_time target 修正

```python
# OLD: target=1.0（全部油墨，错误）
# NEW: target=phi_IC(z)（正确分层）
z = torch.rand(n, device=self.device) * PHYSICS["Lz"]
phi_ic_target = 0.5 * (1.0 + torch.tanh((h_ink - z) / delta_ic))
```
early_time 约束应编码正确的初始分层：z<h_ink 时 φ≈1（油墨），z>h_ink 时 φ≈0（极性液体）。

### 提交记录

| 提交 | 内容 |
|------|------|
| `f506f2e` | 修复硬约束 blend：z=0 底面完全自由 |
| `e85a7fb` | IC 和 early_time 采样改为全域 [0, Lz]，target 用 phi_IC(z) |

### 训练进展

**pinn_20260527_105409**（已停止，S1 epoch ~1700 → 被 kill 以修复 IC 采样）

**pinn_20260527_1108xx** → 实际目录 `pinn_20260527_111127`（运行中）
- 配置: `config/v4.6-pfw-z0driven.json`
- 当前进度: epoch ~11800/60000 (S3, ~20%)
- 监控记录:
  - Epoch 3400 [S2]: Loss 637, Vol 1.20, AC 36.0
  - Epoch 5500 [S3]: Loss 714, Vol 1.13, AC 74.3
  - Epoch 7800 [S3]: Loss 1081, Vol 1.15, AC 242
  - Epoch 9700 [S3]: Loss 1141, Vol 2.75, AC 268
  - Epoch 11500 [S3]: Loss 1133, Vol 3.30, AC 263
  - Epoch 11800 [S3]: Loss 1405, Vol 7.32, AC 320
- **观察**: S3 物理约束接管期 Loss 震荡上升，AC/IE/LP 权重生效，PFW 非零
- **待验证**: z=0 底面 φ 径向分布是否真正多元化

## 当前状态 (2026-05-27 13:53)

**最新训练**: pinn_20260527_111127, epoch ~11800/60000 (S3 20%)
- 训练正常运行，PID 303800
- S3 物理约束激活中（LP/IE/EW/PFW/DC/TB 均非零）
- Loss 在 1000-1500 区间震荡，待观察是否收敛
- 监控日志: `/tmp/v18_monitor.log`（每 30 分钟采样一次）

**代码文件变更**: `pinn_two_phase.py`（blend 公式 + IC 采样 + early_time target）
**配置文件**: v4.6-pfw-z0driven.json（完整 S3 训练）
**Git 提交**: `f506f2e`, `e85a7fb`

**风险评估**:
- S3 初期震荡属正常（物理权重从零到有）
- 关注指标: AC 是否出现"突破式下降"（v13 关键特征）
- 关注指标: Vol 是否稳定在 ~1.0（体积守恒）
- 如 epoch 20000 时 Loss 仍未下降 → 需检查物理权重过大问题
