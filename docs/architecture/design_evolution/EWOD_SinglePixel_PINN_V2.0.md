# 单像素电润湿 4D-XYZt PINN 精简模型

## 文档版本：V2.0
## 适用场景：硕士/博士论文核心章节、快速原型建模、单像素机理研究
## 核心覆盖：Navier-Stokes + VOF + 电场 + 接触角滞后 + 吉布斯自由能 + PINN

---

## 目录
1. [模型概述与基本假设](#1-模型概述与基本假设)
2. [控制方程](#2-控制方程)
   - 2.1 特征量与无量纲化
   - 2.2 连续性方程
   - 2.3 动量方程
   - 2.4 VOF 相场输运
   - 2.5 界面力学
   - 2.6 电场方程
3. [边界条件](#3-边界条件)
4. [非理想物理效应](#4-非理想物理效应)
   - 4.1 接触角滞后 (CAH)
   - 4.2 壁面滑移
   - 4.3 CAH 与滑移的协同机制
5. [热力学约束](#5-热力学约束)
   - 5.1 吉布斯自由能（可测量形式）
   - 5.2 油相体积守恒
6. [PINN 损失函数与网络架构](#6-pinn-损失函数与网络架构)
   - 6.1 总损失函数（含显式残差）
   - 6.2 初始条件损失
   - 6.3 自适应权重
   - 6.4 网络架构
   - 6.5 训练策略
7. [验证策略](#7-验证策略)
   - 7.1 分层验证
   - 7.2 单像素特有基准
   - 7.3 异常指纹诊断
8. [模型局限与适用条件](#8-模型局限与适用条件)

---

## 1 模型概述与基本假设

以单个电润湿像素为对象，构建四维时空 $(x,y,z,t)$ 物理信息神经网络（PINN），耦合弱可压缩 Navier-Stokes 方程、VOF 相场、界面应力连续及电场驱动，实现无网格高精度预测像素内油膜三维形貌与动态开关响应。

**单像素简化假设**：
- **几何**：像素腔体为长方体 $\Omega = [0,L]^2 \times [0,H]$，底部依次为均匀厚度的介电层 $d_d$ 与疏水层 $d_f$，上方为油水两相流体域。
- **电压均匀**：像素尺寸足够小（$L \le 200\ \mu\text{m}$），ITO 方阻忽略，底部电极电势 $\varphi = V_0$ 处处相等，无面内衰减。
- **工艺均匀**：$d_d$、$d_f$ 在像素内为常数，无空间起伏。
- **等温、无重力**：忽略温度变化与浮力效应（$Bo \ll 1$），物性为常数。
- **无界面活性剂、无气相**：仅考虑纯油‑水两相，无表面活性剂输运，无气泡产生。

---

## 2 控制方程

### 2.1 特征量与无量纲化

选取特征量如下：

| 特征量 | 符号 | 定义 | 典型值 |
|:---|:---:|:---|:---:|
| 特征长度 | $L_c$ | 像素半宽 $L/2$ | $75\ \mu\text{m}$ |
| 特征速度 | $U_c$ | $\varepsilon_0 \varepsilon_r V_0^2/(\mu_o d_d)$ | $10^{-3}\sim 10^{-2}\ \text{m/s}$ |
| 特征压力 | $p_c$ | 毛细压强 $\gamma_{ow}/L_c$ | $\sim 200\ \text{Pa}$ |
| 特征时间 | $t_c$ | $L_c/U_c$ | $10^{-3}\sim 10^{-1}\ \text{s}$ |
| 特征电势 | $\varphi_c$ | $V_0$ | $5\sim 30\ \text{V}$ |

定义无量纲变量（以上标 $*$ 表示，为简洁起见略去 $*$）：

$$x^* = \frac{x}{L_c},\ y^* = \frac{y}{L_c},\ z^* = \frac{z}{L_c},\ t^* = \frac{t}{t_c},\ \mathbf{u}^* = \frac{\mathbf{u}}{U_c},\ p^* = \frac{p}{p_c},\ \varphi^* = \frac{\varphi}{\varphi_c}$$

无量纲参数：

- Reynolds 数 $Re = \frac{\rho_o U_c L_c}{\mu_o}$
- Capillary 数 $Ca = \frac{\mu_o U_c}{\gamma_{ow}}$
- 电润湿数 $\eta = \frac{\Delta p_{\text{elec}}}{p_c} = \frac{C_{\text{eq}} V_0^2}{2p_c}$，其中等效电容 $C_{\text{eq}} = \frac{\varepsilon_0}{d_d/\varepsilon_d + d_f/\varepsilon_f}$
- Mach 数 $Ma = U_c/c$（弱可压缩声速 $c$ 取 $Ma<10^{-3}$）

以下所有方程均为无量纲形式。

### 2.2 连续性方程（弱可压缩）

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = -\frac{1}{Ma^2}\frac{Dp}{Dt} \tag{1}$$

物性插值：$\rho(\phi) = \phi\rho_o + (1-\phi)\rho_w$，$\mu(\phi) = \phi\mu_o + (1-\phi)\mu_w$，介电常数 $\varepsilon(\phi) = \phi\varepsilon_o + (1-\phi)\varepsilon_w$（均为无量纲比值）。

### 2.3 动量方程

$$\rho \frac{D\mathbf{u}}{Dt} = -\frac{1}{Ca}\nabla p + \nabla \cdot (\mu \nabla \mathbf{u}) + \frac{1}{Ca}\mathbf{F}_{\text{csf}} + \frac{1}{2}\mathbf{F}_{\text{elec}} \tag{2}$$

其中：
- $\mathbf{F}_{\text{csf}} = \kappa \mathbf{n}\,\delta_\Gamma$，$\delta_\Gamma = |\nabla\phi|$ 为界面 Dirac 函数（CSF 模型）。
- $\mathbf{F}_{\text{elec}} = -\frac{1}{2}\varepsilon_r |\nabla\varphi|^2 \nabla\varepsilon_r$（Korteweg‑Helmholtz 力，电致伸缩项已并入修正压力；系数 $1/2$ 源于特征量选取，推导见附注）。

> **附注**：若将电场效应完全归入界面压强（即忽略体积极化力），可令 $\mathbf{F}_{\text{elec}} \equiv 0$，对结果影响微小且进一步简化模型。本模型保留该项以维持多物理场耦合的完整性。

### 2.4 VOF 相场输运

$$\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = 0 \tag{3}$$

$\phi=1$ 为油相，$\phi=0$ 为水相，界面位于 $\phi=0.5$。

### 2.5 界面力学

- 界面法向量与曲率：$\mathbf{n} = \frac{\nabla\phi}{|\nabla\phi|+\delta}$，$\kappa = \nabla\cdot\mathbf{n}$，$\delta = 10^{-8}$。
- 法向应力连续（广义 Young‑Laplace）：

$$p_o - p_w = \kappa + \eta \tag{4}$$

其中 $\eta$ 为无量纲电润湿数，代表电场附加压强（常数，因电压与介质层均匀）。

### 2.6 电场方程

忽略介质层漏电流（$\sigma \to 0$）且为准静态电场（$\partial_t \nabla\varphi \to 0$），电势满足 Laplace 方程：

$$\nabla \cdot (\varepsilon \nabla \varphi) = 0 \tag{5}$$

（注：在单像素均匀介质层、均匀电压下，该方程的解为线性分布，但保留该方程以验证 PINN 的多物理场学习能力，并为未来扩展提供接口。）

---

## 3 边界条件

- **流动**
  底部疏水表面 $(z = d_d+d_f)$：Navier 滑移 $u_\tau = l_s \partial_n u_\tau$，壁面无穿透 $\mathbf{u}\cdot\mathbf{n}_{\text{wall}}=0$。
  侧壁和顶盖 $(z=H)$：无滑移 $\mathbf{u}=0$。

- **电场**
  底部电极 $(z=0)$：$\varphi = 1$（无量纲 $V_0$ 已被 $\varphi_c$ 归一化）。
  其他壁面：$\partial_n \varphi = 0$（绝缘）。
  油‑水界面：电势及法向电位移连续。

- **相场**
  底部疏水表面：接触角约束 $\mathbf{n}\cdot\mathbf{n}_{\text{wall}} = \cos\theta$（详见 §4）。
  其余固壁：中性润湿条件（$\partial_n \phi = 0$）。

---

## 4 非理想物理效应

### 4.1 接触角滞后 (CAH)

三相接触线运动时，微观接触角在前进角 $\theta_A$ 和后退角 $\theta_R$ 之间切换：

$$\mathbf{n}\cdot\mathbf{n}_{\text{wall}} =
\begin{cases}
\cos\theta_A, & \text{铺展} \;(\mathbf{u}\cdot\mathbf{n}_{\text{wall}} > 0) \\
\cos\theta_R, & \text{收缩} \;(\mathbf{u}\cdot\mathbf{n}_{\text{wall}} < 0)
\end{cases} \tag{6}$$

数值平滑（可选）：使用 $\tanh$ 过渡 $ \cos\theta = \cos\theta_0 + \frac{\Delta\theta}{2}\tanh(\beta\,\mathbf{u}\cdot\mathbf{n}_{\text{wall}})$，$\theta_0=(\theta_A+\theta_R)/2$，$\Delta\theta=\theta_A-\theta_R$，$\beta\sim 10^2\text{--}10^4\,\text{s/m}$，避免梯度间断。

### 4.2 壁面滑移

疏水表面采用 Navier 滑移边界条件：

$$u_\tau = l_s \frac{\partial u_\tau}{\partial n} \tag{7}$$

$l_s$ 为滑移长度，典型值 $10\text{--}100\,\text{nm}$。

### 4.3 CAH 与滑移的协同机制

接触角滞后决定了三相接触线运动的**力学阈值**（必须克服滞后角才能启动），而壁面滑移**消除了无滑移假设下的应力奇异性**，允许接触线以有限速度移动。两者共同作用才能真实再现电润湿的动态响应：CAH 提供运动阻力，滑移提供运动学可行性。在 PINN 损失函数中，接触角损失 $\mathcal{L}_{\text{Contact}}$ 与滑移边界硬编码（或软约束）需同时施加。

---

## 5 热力学约束

### 5.1 吉布斯自由能（可测量形式）

等温、无外加热源时，系统自由能简化为界面能与电场能之和：

$$G = \gamma_{ow} A_{ow} + \gamma_{so} A_{so} + \gamma_{sw} A_{sw} - \frac{1}{2}C_{\text{eq}} V_0^2 A_{sw} \tag{8}$$

利用 Young 方程 $\gamma_{so} - \gamma_{sw} = \gamma_{ow}\cos\theta_0$ 及 $A_{\text{total}} = A_{so}+A_{sw}$ 为常数，消去不可直接测量的固体界面张力，得到仅依赖 $\gamma_{ow}$ 和本征接触角 $\theta_0$ 的有效自由能：

$$G_{\text{eff}} = \gamma_{ow} A_{ow} - \gamma_{ow}\cos\theta_0\, A_{sw} - \frac{1}{2}C_{\text{eq}} V_0^2 A_{sw} \tag{8a}$$

稳态对应 $\delta G_{\text{eff}} = 0$，等价于 Young‑Laplace 方程与接触角条件。

### 5.2 油相体积守恒

$$\iiint_\Omega \phi \, dV = V_{\text{oil}} = \text{const} \tag{9}$$

---

## 6 PINN 损失函数与网络架构

### 6.1 总损失函数（显式残差）

总损失函数为各物理约束残差的加权均方误差和：

$$\mathcal{L} = \lambda_1\mathcal{L}_{\text{NS}} + \lambda_2\mathcal{L}_{\text{Cont}} + \lambda_3\mathcal{L}_{\text{VOF}} + \lambda_4\mathcal{L}_{\text{Intf}} + \lambda_5\mathcal{L}_{\text{Elec}} + \lambda_6\mathcal{L}_{\text{Contact}} + \lambda_7\mathcal{L}_{\text{Vol}} + \lambda_8\mathcal{L}_{\text{IC}} \tag{10}$$

各损失项具体残差形式如下表：

| 损失项 | 物理约束 | 残差表达式（均方误差） |
|:---|:---|:---|
| $\mathcal{L}_{\text{NS}}$ | 动量方程 (2) | $\|\rho \frac{D\mathbf{u}}{Dt} + \frac{1}{Ca}\nabla p - \nabla\cdot(\mu\nabla\mathbf{u}) - \frac{1}{Ca}\mathbf{F}_{\text{csf}} - \frac{1}{2}\mathbf{F}_{\text{elec}}\|_2^2$ |
| $\mathcal{L}_{\text{Cont}}$ | 连续性方程 (1) | $\|\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\mathbf{u}) + \frac{1}{Ma^2}\frac{Dp}{Dt}\|_2^2$ |
| $\mathcal{L}_{\text{VOF}}$ | 相场输运 (3) | $\|\frac{\partial\phi}{\partial t} + \mathbf{u}\cdot\nabla\phi\|_2^2$ |
| $\mathcal{L}_{\text{Intf}}$ | 界面应力 (4) + 法向量定义 | $\|p_o - p_w - \kappa - \eta\|_2^2 + \|\mathbf{n} - \frac{\nabla\phi}{|\nabla\phi|+\delta}\|_2^2$ |
| $\mathcal{L}_{\text{Elec}}$ | 电场方程 (5) | $\|\nabla\cdot(\varepsilon\nabla\varphi)\|_2^2$ |
| $\mathcal{L}_{\text{Contact}}$ | 接触角约束 (6) | $\|\mathbf{n}\cdot\mathbf{n}_{\text{wall}} - \cos\theta_{A/R}\|_2^2$ （壁面采样点） |
| $\mathcal{L}_{\text{Vol}}$ | 体积守恒 (9) | $\|\frac{1}{N}\sum_i \phi_i - \frac{V_{\text{oil}}}{L^2 H}\|_2^2$ |
| $\mathcal{L}_{\text{IC}}$ | 初始条件 (§6.2) | $\|\phi(x,y,z,0) - \phi_{\text{IC}}(x,y,z)\|_2^2$ |

可选正则项：界面曲率平滑 $\mathcal{L}_{\kappa\text{-reg}} = \lambda_\kappa \|\nabla\kappa\|_2^2$。

### 6.2 初始条件损失

初始时刻 $t=0$ 的油膜形貌由无外电场时的静态平衡决定。对于小 Bond 数（$Bo\ll1$），形貌为球冠状：

$$\phi_{\text{IC}}(x,y,z) = \frac{1}{2}\left[1 + \tanh\left(\frac{R - r}{2\varepsilon}\right)\right]$$

其中 $r = \sqrt{x^2+y^2+(z-(d_d+d_f))^2}$，$R = \left[\frac{3V_{\text{oil}}}{\pi(2-3\cos\theta_0+\cos^3\theta_0)}\right]^{1/3}$，$\varepsilon$ 为界面宽度参数。该初始场作为软约束加入 $\mathcal{L}_{\text{IC}}$。

### 6.3 自适应权重

采用 GradNorm 或 NeSII 算法动态调整各 $\lambda_i$，平衡梯度范数，防止某一损失项主导训练。典型初始权重可设为 $(1, 1, 1, 10, 1, 10, 10, 10)$，随后自动调节。

### 6.4 网络架构

- **输入**：$(x,y,z,t)$（4 维）
- **输出**：$(u,v,w,p,\phi,\varphi)$（6 维）
- **结构**：6–8 层全连接，每层 128–256 神经元，激活函数 GELU（保证高阶导数连续性）
- **傅里叶特征编码**：对空间坐标 $(x,y,z)$ 施加 $\gamma(\mathbf{x}) = [\sin(2\pi B\mathbf{x}), \cos(2\pi B\mathbf{x})]$，$B\sim\mathcal{N}(0,\sigma^2)$，$\sigma\sim 1\text{--}10$，缓解谱偏置
- **多任务头**：流场 $(u,v,w,p)$、相场 $\phi$、电势 $\varphi$ 共享前 4–5 层，后接独立 2–3 层
- **硬约束编码**：壁面无穿透条件通过 $\mathbf{u} = \hat{\mathbf{u}} \cdot d_{\text{wall}}(z)$ 实现，$d_{\text{wall}}(z)=\min(z-(d_d+d_f), H-z)$

### 6.5 训练策略

- **空间采样**：Latin Hypercube 全局采样 (10k–20k 点) + 界面附近 ($0.2<\phi<0.8$) 局部加密 (5k–10k 点)
- **时间采样**：电压切换时刻 $t=0, t_{\text{switch}}$ 加密（时间分辨率 $\sim 0.1t_c$），稳态段稀疏
- **优化器**：Adam + L-BFGS 两阶段训练
- **收敛判据**：总损失相对下降 $<10^{-4}$，各项物理残差 $<10^{-3}$，体积守恒误差 $<1\%$

---

## 7 验证策略

### 7.1 分层验证
- **L1 单元验证**：静态液滴解析解（$V_0=0$），MMS 收敛率测试。
- **L2 耦合验证**：与 COMSOL/OpenFOAM 的 VOF+电场基准对比（接触角、铺展半径、响应时间）。
- **L3 实验验证**：共聚焦/接触角仪测量稳态油膜形貌与动态开关曲线。

### 7.2 单像素特有基准
**电润湿铺展基准**：初始无电场油滴呈球冠（接触角 $\theta_0$），突加电压 $V_0$ 后，接触角向 Lippmann 角 $\theta_L = \arccos(\cos\theta_0 + \eta)$ 演化，铺展半径 $R$ 按 Tanner 律 $R \propto t^{1/10}$（或电润湿修正标度）增长。PINN 预测的瞬态 $R(t)$ 和稳态 $\theta$ 应与理论/CFD 一致。

### 7.3 异常指纹诊断
训练及验证过程中监控以下物理可行性指标：
- 接触角 $\theta \in [0^\circ, 180^\circ]$
- 相场无孤立碎片（体相域内无 $\phi=0.5$ 异常曲面）
- 速度场幅值 $|\mathbf{u}|_{\max}$ 在 $t > 2t_c$ 后不超过 $5U_c$
- 体积守恒漂移 $< 5\%$
- 压力场不出现负压（$p > -p_c$）或异常高压（$p < 10p_c$）

任一异常出现，优先调整权重或采样再继续训练。

---

## 8 模型局限与适用条件

- 仅适用于单个像素，无法模拟像素间流体串扰。
- 假设电极电压均匀，适用于 $L < 200\ \mu\text{m}$ 且方阻可忽略的 ITO 电极。
- 忽略温度变化、浮力、表面活性剂及双电层效应。
- 未考虑介质层电荷注入、油膜蒸发、柔性变形等长期服役效应。

**适用参数范围**：$50\ \mu\text{m} \le L \le 200\ \mu\text{m}$，$V_0 \le 30\,\text{V}$，水平放置，纯净油‑水体系。

---

*文档版本 V2.0 | 最后更新 2026-05-14*
