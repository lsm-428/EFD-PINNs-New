# EFD3D 知识库补充文档

> 本文档补充 NotebookLM "EFD3D PINN 研究"知识库中缺失的技术细节，基于代码实现 (v4.5) 整理。

## 📋 目录
1. [模型架构补充](#模型架构补充)
2. [物理约束实现细节](#物理约束实现细节)
3. [训练策略补充](#训练策略补充)
4. [代码特有功能](#代码特有功能)
5. [建议纳入知识库的要点](#建议纳入知识库的要点)

---

## 模型架构补充

### 1. TwoPhasePINN 网络结构（完整版）

**知识库应补充的详细信息：**

```python
# 来源：src/models/pinn_two_phase.py 第174-311行

class TwoPhasePINN(nn.Module):
    """
    输入: (batch, 6) - [x, y, z, V_from, V_to, t_since]
    输出: (batch, 5) - [u, v, w, p, phi]
    
    网络架构:
    - phi_net: 输入6维 → 隐藏层[128,128,64,32] → 输出1维 (phi)
    - vel_net: 输入7维 (6+phi) → 隐藏层[64,64,32] → 输出4维 (u,v,w,p)
    
    激活函数: Tanh (所有隐藏层)
    初始化: Xavier normal
    """
```

**关键设计决策：**
- **分离网络**：phi和速度场分离，避免梯度耦合
- **7维速度输入**：速度网络接收phi作为额外输入，实现物理耦合
- **归一化**：所有输入归一化到[0,1]范围（x/Lx, y/Ly, z/Lz, t/t_max, V/30）

### 2. 6D Triad 输入详解

**三元组 (Triplet) 格式语义：**

| 分量 | 符号 | 物理意义 | 归一化 |
|------|------|---------|---------|
| x, y, z | 空间坐标 | 像素内位置 (0~174μm) | ÷ Lx, Ly, Lz |
| V_from | 跳变前电压 | 前一稳态电压值 | ÷ 30V |
| V_to | 跳变后电压 | 当前施加电压 | ÷ 30V |
| t_since | 时间 | 电压跳变后经过时间 | ÷ t_max (0.05s) |

**电压历史编码原理：**
- `V_from = V_to`：稳态（无跳变）
- `V_from < V_to`：升压（电润湿驱动油墨收缩）
- `V_from > V_to`：降压（表面张力恢复，油墨回弹）

**代码实现（第256-262行）：**
```python
x_norm = x_coord / self.Lx
y_norm = y_coord / self.Ly
z_norm = z_coord / self.Lz
t_norm = t_since / self.t_max
V_from_norm = V_from / 30.0
V_to_norm = V_to / 30.0
```

---

## 物理约束实现细节

### 1. Navier-Stokes 方程完整形式

**知识库应补充的数学公式：**

```
ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st + F_ew

其中：
- ρ = φ·ρ_oil + (1-φ)·ρ_polar  (混合密度)
- μ = φ·μ_oil + (1-φ)·μ_polar  (混合粘度)
- F_st = σ·κ·∇φ  (表面张力，CSF模型)
- F_ew = (ε₀εᵣV²/(2d)) · φ · 4φ(1-φ) · dir  (电润湿力)
```

**代码实现位置：** `src/physics/constraints.py` 第111-293行

### 2. 电润湿力 (Electrowetting Force) 详细实现

**物理量计算：**

```python
# 1. 电润湿压强幅值 (第241行)
f_ew_magnitude = epsilon_0 * epsilon_r * V_to**2 / (2 * d_dielectric)
# 单位: N/m² (Pascal)

# 2. 界面指示函数 (第255行)
# 4*phi*(1-phi) 在 phi=0.5 处最大，近似 delta 函数
interface_indicator = 4 * phi * (1 - phi)

# 3. 空间衰减 (第251行)
# 电润湿力只在底面附近有效 (z < 2*h_ink)
h_ink = 3e-6  # 油墨厚度
z_decay = torch.exp(-z_coord / (2 * h_ink))

# 4. 方向向量 (第257-266行)
# 力驱动油墨从中心向外收缩（径向向外）
dir_x = (x_coord - center_x) / dist_from_center
dir_y = (y_coord - center_y) / dist_from_center

# 5. 体积力合成 (第271-273行)
f_ew_x = f_ew_magnitude * phi * interface_indicator * z_decay * dir_x
f_ew_y = f_ew_magnitude * phi * interface_indicator * z_decay * dir_y
f_ew_z = 0  # z方向无力
```

**物理意义：**
- 升压时 (V_to > V_from)：油墨向边缘收缩 → 中心透明 (开口率↑)
- 降压时 (V_to < V_from)：表面张力恢复 → 油墨回流 (开口率↓)

### 3. VOF (Volume of Fluid) 传输方程

**方程形式：**
```
∂φ/∂t + u·∇φ = 0

φ = 1: 纯油墨
φ = 0: 纯极性液体
0<φ<1: 界面过渡区
```

**界面锐化 (Sharpening)：**
配置中启用 `sharpening_weight: 1.0`，通过人工压缩项防止界面扩散。

### 4. 表面张力与曲率计算

**CSF (Continuum Surface Force) 模型：**

```python
# 曲率计算 (第207-214行)
# κ = -div(∇φ/|∇φ|) 精确公式
numerator = (phi_xx*(phi_y**2+phi_z**2) + 
             phi_yy*(phi_x**2+phi_z**2) + 
             phi_zz*(phi_x**2+phi_y**2) - 
             2*(phi_x*phi_y*phi_xy + phi_x*phi_z*phi_xz + phi_y*phi_z*phi_yz))
kappa = -numerator / (grad_phi_mag_sq * grad_phi_mag + 1e-10)

# 表面张力体积力
f_st_x = sigma * kappa * phi_x
f_st_y = sigma * kappa * phi_y
f_st_z = sigma * kappa * phi_z
```

---

## 训练策略补充

### 1. 三阶段渐进训练 (v4.5配置)

**配置来源：** `config/v4.5-standard.json` 第19-37行

| 阶段 | Epochs | 物理约束 | 数据权重 | 目标 |
|------|---------|---------|---------|------|
| **Stage1** | 1500 | 无 | interface(500) + ic(300) + bc(80) | 学习稳态接触角 |
| **Stage2** | 4000 | +continuity(0.5) + vof(0.5) | 降低数据权重 | 引入流体连续性 |
| **Stage3** | 60000 | +ns(0.1) + surface_tension(0.01) | 最小 | 完整物理约束 |

**动态权重调整：**
```json
"dynamic_weight": {
  "enable": true,
  "adjustment_strategy": "combined",
  "target_loss_ratio": 1.0,
  "adjustment_interval": 100
}
```

### 2. 损失函数权重配置

**完整残差权重列表（第52-81行）：**

| 权重项 | 值 | 物理意义 |
|--------|-----|---------|
| young_lippmann | 2.0 | Young-Lippmann电润湿方程 |
| contact_angle_constraint | 2.0 | 接触角边界条件 |
| sidewall_contact_angle | 2.0 | 侧壁接触角 |
| continuity | 0.1 | 质量守恒 ∇·u=0 |
| momentum_u,v,w | 0.02 each | N-S动量方程各分量 |
| vof | 0.5 | VOF传输方程 |
| surface_tension | 0.3 | 表面张力 |
| volume_conservation | 0.3 | 体积守恒 |
| explicit_volume | 100.0 | 显式体积约束 |

### 3. 数据采样策略

**采样点配置（第83-93行）：**
```json
"data": {
  "n_interface": 60000,  // 界面点（高密度）
  "n_initial": 10000,    // 初始条件点
  "n_boundary": 8000,    // 边界条件点
  "n_domain": 50000,     // 域内部点
  "voltages": [0-30V, 步长1V],
  "times": 200           // 时间采样点
}
```

**电压等级：** 0-30V，共31个电压值，覆盖完整工作范围。

---

## 代码特有功能

### 1. 训练稳定性增强

**TrainingStabilizer（知识库未提及）：**
- **NaN恢复机制**：检测到NaN时自动回退到上一个checkpoint
- **梯度裁剪**：`gradient_clip: 1.0` 防止梯度爆炸
- **L-BFGS支持**：`use_lbfgs: true` 可选二阶优化

### 2. 界面锐化损失 (Sharpening Loss)

**新增功能（v4.5）：**
```python
"sharpening_weight": 1.0  # VOF界面锐化
```
防止数值扩散，保持界面陡峭。

### 3. 体积守恒显式约束

**Explicit Volume Constraint：**
```python
"explicit_volume_weight": 100.0
```
直接约束总体积变化，确保物理一致性。

### 4. 混合密度/粘度插值

**两相材料属性（第154-155行）：**
```python
rho = phi * rho_oil + (1 - phi) * rho_polar
mu = phi * mu_oil + (1 - phi) * mu_polar
```
基于VOF分数线性插值，实现光滑过渡。

---

## 建议纳入知识库的要点

### 🔴 高优先级（核心理论）

1. **6D Triad输入的设计原理**
   - 为什么需要V_from？捕获电压历史依赖
   - t_since的作用：时序动力学建模
   - 单次训练支持任意电压序列推理的机制

2. **电润湿力的体积力形式**
   - f_ew = ε₀εᵣV²/(2d) · φ · 4φ(1-φ) · dir
   - 为什么用4φ(1-φ)近似delta函数？
   - 空间衰减函数 exp(-z/(2h_ink)) 的物理依据

3. **分离网络架构的动机**
   - phi和速度场解耦的原因（避免梯度冲突）
   - 7维速度输入的意义（物理耦合）

### 🟡 中优先级（实现细节）

4. **三阶段训练的退化策略**
   - Stage1：纯数据驱动学习稳态
   - Stage2：引入连续性+VOF，建立流体场
   - Stage3：完整NS方程，平衡各物理约束

5. **动态损失权重调整**
   - combined策略：基于loss ratio自动调整
   - target_loss_ratio=1.0：各损失项平衡

6. **界面锐化技术**
   - Sharpening loss的数学形式
   - 与标准VOF的差异

### 🟢 低优先级（工程实践）

7. **训练稳定性技巧**
   - NaN恢复机制实现
   - 梯度裁剪阈值选择
   - L-BFGS与Adam的混合使用

8. **数据采样策略**
   - 界面加密采样的必要性
   - 电压离散化密度选择

---

## 📚 参考文献补充

**建议知识库添加以下参考文献：**

1. **PINN基础理论**：
   - Raissi et al., "Physics-informed neural networks", JCP 2019
   
2. **VOF方法**：
   - Hirt & Nichols, "Volume of Fluid (VOF) method", JCP 1981
   
3. **电润湿理论**：
   - Lippmann, "Relations entre les phénomènes électriques et capillaires", 1904
   - Berge, "Electrowetting of water", C.R. Acad. Sci. 1993
   
4. **CSF模型**：
   - Brackbill et al., "Continuum surface force model", JCP 1992
   
5. **两相流PINN应用**：
   - EFD3D相关论文（待发表）

---

## 📝 更新记录

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-05-08 | v1.0 | 初始版本，基于v4.5代码实现整理 |

---

*本文档旨在补充NotebookLM知识库，建议在下次更新时纳入上述内容。*
