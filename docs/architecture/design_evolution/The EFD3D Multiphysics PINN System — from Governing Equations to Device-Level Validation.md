# 电润湿像素 4D-XYZt PINN 多物理场完备建模文档

## 文档版本：V2.5.1 (微调版)
## 适用场景：硕士/博士论文理论章节、SCI期刊机理模型、项目技术白皮书
## 核心覆盖：Navier-Stokes + VOF两相界面 + 四维时空PINN(XYZt) + 吉布斯自由能 + 器件各层厚度/导电方阻 + 所有非理想物理效应全耦合 + 模型验证 + 局限性

---

## 目录
1. [模型整体框架](#1-模型整体框架)
   - 1.1 模型逻辑流程图
   - 1.2 核心建模思路
   - 1.3 为什么选择 PINN 而非传统 CFD
   - 1.4 模型的跨尺度层次结构
2. [计算域与PINN输出变量定义](#2-计算域与pinn输出变量定义)
   - 2.1 四维时空计算域
   - 2.2 PINN 网络输入输出
   - 2.3 边界条件体系
   - 2.4 时空采样策略
   - 2.5 初始条件设定
3. [器件Z向工艺分层结构](#3-器件z向工艺分层结构)
   - 工艺厚度空间起伏的统计模型、器件各层材料物理性质表
4. [流体动力学控制方程](#4-流体动力学控制方程)
   - 4.1 弱可压缩连续性方程
   - 4.2 三维 Navier-Stokes 动量方程（含流动状态与无量纲分析、重力/浮力效应、各力项物理详述）
   - 4.3 弱可压缩近似的理论与数值依据 / VOF 温变物性加权
   - 4.4 壁面 Navier 滑移边界条件
   - 4.5 控制方程无量纲化（特征量选取 + 无量纲参数体系）
5. [VOF相场与界面力学连续模型](#5-vof相场与界面力学连续模型)
   - 5.1 VOF 相场输运方程（含物理本质与 Diffuse Interface 对比）
   - 5.2–5.4（CSF 数值实现、Young-Laplace 广义推导）
6. [电场-多层介质-导电方阻耦合模型](#6-电场-多层介质-导电方阻耦合模型)
   - 6.1–6.5（含 RC 传输线推导、频率依赖性、边缘电场畸变）
   - 6.6 电化学反应与水解阈值、介电层击穿场强与安全裕度
7. [固壁润湿边界与三相接触线模型](#7-固壁润湿边界与三相接触线模型)
   - 7.1–7.5（CAH、Wenzel、MKT、钉扎、Cassie-Baxter）
8. [双电层/表面活性剂/热场耦合模型](#8-双电层表面活性剂热场耦合模型)
   - 8.1–8.4（EHD 耦合路径、EDL-表面活性剂竞争效应、非等温量级估计）
9. [吉布斯自由能泛函与守恒约束](#9-吉布斯自由能泛函与守恒约束)
   - 9.1–9.3（变分推导、Onsager 变分原理）
10. [四维PINN总损失函数构造](#10-四维pinn总损失函数构造)
    - 10.1 多物理场联合损失
    - 10.2 数值稳定附加约束（NeSII 算法详解、推荐网络架构、收敛判据、硬约束编码）
11. [模型验证策略与基准测试](#11-模型验证策略与基准测试)
    - 11.1 四层验证体系 (L1–L4)
    - 11.2 L1: 解析解与制造解 (MMS)
    - 11.3 L2: CFD 基准对比
    - 11.4 L3: 实验数据对比（共聚焦/接触角仪/高速相机）
    - 11.5 验证指标体系
12. [不确定性量化 (UQ) 框架](#12-不确定性量化-uq-框架)
    - 12.1 前向 UQ — Monte Carlo 传播
    - 12.2 反向 UQ — Bayesian 参数标定
    - 12.3 Sobol 全局灵敏度分析
    - 12.4 验证失败判定：异常指纹清单
13. [模型局限性与适用条件](#13-模型局限性与适用条件)
    - 13.1 物理假设与适用范围边界（9项假设表）
    - 13.2 数值方法的已知局限（高阶导数退化、采样效率、权重脆弱性）
    - 13.3 未包含的物理效应（6项待扩展效应）
    - 13.4 适用场景声明
14. [模型完备性覆盖清单](#14-模型完备性覆盖清单)
15. [模型核心创新点总结](#15-模型核心创新点总结)

---

# 1 模型整体框架

## 1.1 模型逻辑流程图

```mermaid
flowchart TD
    A[输入参数<br/>像素几何/各层厚度/导电方阻/驱动电压/固水气接触角]
    B[四维时空PINN<br/>PINN(x,y,z,t)]
    C[多物理场并行求解<br/>流体动力学 | 相场界面 | 电场介电]
    D[热力学约束<br/>吉布斯自由能+动态弛豫+体积守恒]
    E[数值正则约束<br/>自适应权重+曲率正则+接触线奇点正则]
    F[输出结果<br/>3D油膜形貌/流场/电势/三相线/动态残影演化]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

## 1.2 核心建模思路

以电润湿微像素为研究对象，摒弃传统 CFD 网格求解，构建四维时空 $(x,y,z,t)$ 物理信息神经网络（PINN）；耦合瞬态弱可压缩 Navier-Stokes 方程、VOF 两相界面捕捉、界面应力连续、多层介质电场、润湿滞后、双电层、Marangoni 流、工艺厚度起伏、导电方阻非均匀等全物理效应；以吉布斯自由能最小化为稳态判据、动态弛豫为时域演化准则，通过多物理场约束构造 PINN 损失函数，实现无网格高精度预测像素内油膜三维形貌与动态开关响应。

## 1.3 为什么选择 PINN 而非传统 CFD

电润湿像素的仿真面临三个传统 CFD 难以逾越的挑战：

**移动界面追踪困难**：油-水两相界面在驱动电压作用下发生大变形铺展/收缩，传统 VOF/Level-Set 方法需要动态网格重构或界面重构，在三维情况下计算量极大且易出现质量守恒误差。PINN 以连续可微神经网络隐式表示相场 $\phi(x,y,z,t)$，无需显式界面追踪即可获得任意精度的界面位置。

**多物理场跨尺度耦合**：像素仿真涉及从纳米级双电层（$\sim 1\text{--}100\,\text{nm}$）到毫米级像素腔（$\sim 100\,\mu\text{m}$）的跨三个数量级尺度耦合。传统 CFD 对此类多尺度问题需要子网格模型或分区耦合策略，引入额外的模型误差。PINN 在统一网络框架内以损失函数项的形式并行施加所有物理约束，天然适合多物理场耦合。

**参数化设计与反问题**：传统 CFD 每次几何或物理参数变化都需要重新划分网格并完整求解，计算成本高。PINN 可将工艺参数（各层厚度、方阻、驱动电压）编码为网络输入，单次训练即可覆盖参数空间，实现"一次训练、实时推理"的正向预测，并便于后续开展反问题（如根据目标油膜形貌反推最优驱动电压波形）。

## 1.4 模型的跨尺度层次结构

本模型的核心物理效应分布在四个空间尺度上，构成层次化耦合体系：

| 尺度层级 | 空间范围 | 核心物理 | 耦合路径 |
|:---|:---:|:---|:---|
| 分子/纳米尺度 | $\sim 1\text{--}100\,\text{nm}$ | 双电层极化、表面活性剂吸附、分子滑移 | → 壁面滑移长度 $l_s$、EDL 体力 $\mathbf{F}_{\text{EDL}}$、界面张力调制 $\gamma(C)$ |
| 介观/界面尺度 | $\sim 0.1\text{--}10\,\mu\text{m}$ | 三相接触线动力学、界面曲率、Marangoni 流 | → 接触角边界条件、CSF 表面张力 $\mathbf{F}_{\text{csf}}$、切向应力 $\tau_M$ |
| 连续介质尺度 | $\sim 10\text{--}100\,\mu\text{m}$ | Navier-Stokes 流动、VOF 相场输运、热传导 | → 速度/压力/相场/温度场的主控方程 |
| 器件/工艺尺度 | $\sim 100\,\mu\text{m}\text{--}1\,\text{mm}$ | 介电层厚度起伏、ITO 方阻梯度、边缘电场畸变 | → 非均匀电容 $C_{\text{eq}}(x,y)$、面内电压衰减 $V(x,y)$ |

各尺度之间通过 PINN 的输出变量实现双向自洽耦合：纳米尺度的 EDL 电势 $\psi$ 提供壁面体力源项给连续介质尺度的动量方程；器件尺度的非均匀电容决定界面上的电场附加压强 $\Delta p_{\text{elec}}$，进而影响介观尺度的接触线运动。

---

# 2 计算域与 PINN 输出变量定义

## 2.1 四维时空计算域

$$\Omega: (x,y,z,t), \quad
\begin{cases}
x \in [0, L] \\
y \in [0, L] \\
z \in [0, H] \\
t \in [0, T_{\text{end}}]
\end{cases}$$

其中：
- $L$：像素平面边长
- $H$：像素腔体总高度
- $T_{\text{end}}$：动态演化时间区间

## 2.2 PINN 网络输入输出

**输入**：时空坐标 $(x, y, z, t)$

**输出**：多物理场耦合变量

$$\text{PINN}(x,y,z,t) = (u, v, w, p, \phi, \varphi, T, \psi, C)$$

| 符号 | 物理含义 |
|:---:|:---|
| $u, v, w$ | 三维流体速度分量 |
| $p$ | 油水两相流体压强场 |
| $\phi$ | VOF 相场变量（$\phi=1$ 油相，$\phi=0$ 水相） |
| $\varphi$ | 外驱动电场电势 |
| $T$ | 全域温度场 |
| $\psi$ | 固液界面双电层 EDL 电势 |
| $C$ | 油-水界面表面活性剂浓度场 |

## 2.3 边界条件体系

计算域的边界条件按物理变量和壁面类型分为三类：

**流动边界条件**：像素侧壁 $(x=0,L$ 或 $y=0,L)$ 和顶部盖板 $(z=H)$ 施加无滑移壁面条件 $\mathbf{u}=0$；底部疏水表面 $(z = d_d + d_f)$ 施加 Navier 滑移条件 $u_\tau = l_s \partial_n u_\tau$（式 4），以消除三相接触线的应力奇点。在疏水层上同时施加壁面无穿透约束 $\mathbf{u} \cdot \mathbf{n}_{\text{wall}} = 0$。

**电场边界条件**：底部 ITO 电极 $(z=0)$ 施加驱动电压 Dirichlet 条件 $\varphi = V(x,y)$，其中 $V(x,y)$ 由式 (11) 的 RC 衰减模型确定；顶部盖板和侧壁施加零通量 Neumann 条件 $\partial_n \varphi = 0$（理想绝缘假设）。在油-水界面上，电势及其法向导数满足介电位移连续性条件 $\varepsilon_o E_o^n - \varepsilon_w E_w^n = 0$。

**相场与浓度边界条件**：VOF 相场 $\phi$ 在疏水壁面上的法向梯度由接触角约束 $\mathbf{n} \cdot \mathbf{n}_{\text{wall}} = \cos\theta$ 隐式确定（式 14）；表面活性剂浓度 $C$ 在固壁上满足零通量条件 $\partial_n C = 0$。对于温度场 $T$，底部恒温 $T = T_0$（室温），侧壁和顶壁绝热 $\partial_n T = 0$。

无穿透条件通过硬约束编码实现（参见 §10.2），而 Navier 滑移条件以软约束形式加入损失函数。

## 2.4 时空采样策略

PINN 训练需要在高维时空域内高效分布采样点（collocation points），本模型采用分层自适应采样策略：

**空间采样**：采用 Latin Hypercube Sampling (LHS) 确保采样点在 $[0,L]^2 \times [0,H]$ 内均匀覆盖空间；在界面区域（$0.2 < \phi < 0.8$）和壁面附近（$z$ 在 $d_d+d_f$ 附近 $\pm 2\lambda_D$ 范围）进行局部加密。典型的采样点数配置为：体相域 10,000–20,000 点，界面域 5,000–10,000 点，壁面域 2,000–5,000 点。

**时间采样**：在时间轴 $[0, T_{\text{end}}]$ 上采用分段策略——驱动电压切换时刻（$t = 0^+, t_{\text{switch}}$）附近加密采样（时间分辨率 $\sim 0.1\,\text{ms}$），稳态区间稀疏采样（$\sim 1\text{--}5\,\text{ms}$），以捕捉动态开关行为。

**自适应重采样**：训练过程中每 $N_{\text{resample}}$ 个 epoch 根据当前 PINN 预测残差的幅值分布进行残差驱动自适应重采样——在高残差区域增加采样点，低残差区域减少采样点，使采样密度与物理约束的难满足程度匹配。这显著提高了高梯度区域（如界面和接触线附近）的求解精度。

## 2.5 初始条件设定

对于动态仿真，初始时刻 $(t=0)$ 的物理场分布通过两种方式设定：

- **稳态初始化**：在 $t=0$ 时刻，先求解无外电场 ($V=0$) 的静态平衡方程获得初始油膜平衡形貌（由 Young-Laplace 方程和接触角边界条件确定），作为 PINN 的初始条件软约束损失项 $\mathcal{L}_{\text{IC}} = \|\text{PINN}(x,y,z,0) - \text{IC}(x,y,z)\|_2^2$。
- **数据驱动初始化**：若具备实验初始形貌数据（如共聚焦显微镜扫描的油膜三维形貌），可直接将实验数据映射为 $\phi(x,y,z,0)$ 的初始条件约束。

---

# 3 器件 Z 向工艺分层结构

像素沿 $z$ 方向的多层结构（从底部 ITO 电极到顶部盖板）定义为：

$$
\begin{cases}
0 < z < d_d(x,y) & \text{介电层（如 SiN}_x\text{）} \\[4pt]
d_d(x,y) < z < d_d(x,y) + d_f(x,y) & \text{疏水绝缘层（如 AF 涂层）} \\[4pt]
d_d(x,y) + d_f(x,y) < z < H & \text{油/水两相流体域}
\end{cases}
$$

其中：
- $d_d(x,y)$：介电层厚度（可空间起伏，例如工艺不均匀导致的 $\pm 5\%$ 波动）
- $d_f(x,y)$：疏水 AF 层厚度（可空间起伏）
- $d_o$：油膜厚度，$d_w$：水层厚度，满足 $H = d_d + d_f + d_o + d_w$
- $R_s(x,y)$：透明导电层（ITO）方阻，支持非均匀分布（$\Omega/\square$）

**补充说明**：实际器件中，$d_d \sim 200\text{--}400\,\text{nm}$，$d_f \sim 10\text{--}50\,\text{nm}$，$d_o \sim 10\text{--}30\,\mu\text{m}$，$H \sim 50\text{--}100\,\mu\text{m}$。方阻 $R_s$ 通常在 $10\text{--}100\,\Omega/\square$ 范围内，其非均匀性来源于 ITO 溅射工艺的膜厚梯度。

### 工艺厚度空间起伏的统计模型

介电层和疏水层的厚度不均匀性来源于 CVD/PVD 薄膜沉积工艺的固有波动，可用空间相关随机场建模：

$$d(x,y) = d_0 \big[1 + \varepsilon(x,y)\big], \qquad \varepsilon(x,y) \sim \mathcal{N}(0, \sigma_d^2)$$

其中 $\sigma_d$ 为厚度相对标准偏差（典型值 $2\%\text{--}5\%$，取决于工艺成熟度）。$\varepsilon(x,y)$ 的空间相关长度 $\lambda_c$ 表示厚度起伏的特征尺度（典型值 $5\text{--}20\,\mu\text{m}$），可由指数型协方差函数 $\text{Cov}[\varepsilon(\mathbf{r}_1), \varepsilon(\mathbf{r}_2)] = \sigma_d^2 \exp(-|\mathbf{r}_1 - \mathbf{r}_2|/\lambda_c)$ 建模。

对于多层结构，各层厚度起伏相互独立（不同沉积步骤），因此等效电容 $C_{\text{eq}}$ 的总体非均匀性为各层贡献的 RSS（Root Sum Square）合成。

### Karhunen-Loève (KL) 展开与 PINN 集成

空间相关随机场 $\varepsilon(x,y)$ 可通过 Karhunen-Loève 展开实现有限维参数化，以便在 PINN 训练和推理中高效表示：

$$\varepsilon(x,y) \approx \sum_{k=1}^{N_{\text{KL}}} \sqrt{\lambda_k}\,\xi_k\,e_k(x,y)$$

其中：
- $\lambda_k, e_k(x,y)$ 为协方差核 $C(\mathbf{r}_1, \mathbf{r}_2) = \sigma_d^2 \exp(-|\mathbf{r}_1 - \mathbf{r}_2|/\lambda_c)$ 的第 $k$ 个特征值和特征函数（对矩形域 $[0,L]^2$ 可解析求解，或通过 Nyström 离散化数值求解）；
- $\xi_k \sim \mathcal{N}(0,1)$ 为独立标准正态随机变量；
- $N_{\text{KL}}$ 为截断项数，由目标方差解释率决定（典型取 $N_{\text{KL}}$ 使 $\sum_{k=1}^{N_{\text{KL}}} \lambda_k / \sum_{k=1}^\infty \lambda_k > 95\%$，对于指数核，特征值衰减为 $\lambda_k \propto 1/k^2$，$N_{\text{KL}} \sim 5\text{--}15$ 通常可满足）。

**与 PINN 的集成方式**（两种可选方案）：

| 方案 | 实现方式 | 适用场景 | 优缺点 |
|:---|:---|:---|:---|
| **方案 A: 训练时采样输入** | 将 KL 系数 $\xi_1,\dots,\xi_{N_{\text{KL}}}$ 作为网络辅助输入维度，即网络输入变为 $(x,y,z,t,\xi_1,\dots,\xi_{N_{\text{KL}}})$，每次训练迭代随机采样 $\xi_k \sim \mathcal{N}(0,1)$；**注意**：傅里叶特征编码仅作用于空间坐标 $(x,y,z)$，不作用于 $\xi_k$（$\xi_k$ 为标准正态标量，无需谱编码） | 工艺容差 Monte Carlo 分析（§12.1），需要在参数空间中批量预测 | 训练成本较高（参数空间增大），但训练完成后可单次前向覆盖任意厚度起伏实现 |
| **方案 B: 推理时随机场后处理** | 训练阶段使用固定名义厚度 $d_0$，推理时用 KL 展开生成 $M$ 个随机场实现，分别输入 PINN 获得 $M$ 个输出，统计给出置信区间 | 单点工况评估（固定设计但需评估厚度波动影响） | 训练成本低，但每次厚度变化需重新前向传播多次 |

**推荐策略**：研究阶段采用方案 B（快速原型验证），工程交付阶段采用方案 A（单次训练覆盖参数空间）。两种方案均无需改变 PINN 的损失函数结构——厚度起伏通过 $C_{\text{eq}}(x,y)$ 和 $V(x,y)$ 间接影响电场和界面约束。

### 器件各层材料物理性质表

| 层 | 材料 | 厚度 | 相对介电常数 $\varepsilon_r$ | 电导率 $\sigma$ (S/m) | 折射率 | 工艺方法 |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| 透明电极 | ITO | $100\text{--}200\,\text{nm}$ | — | $\sim 10^6$ | 1.8–2.0 | 磁控溅射 |
| 介电层 | $\text{SiN}_x$ / $\text{Al}_2\text{O}_3$ | $200\text{--}400\,\text{nm}$ | 6–9 | $10^{-12}\text{--}10^{-14}$ | 1.9–2.1 | PECVD / ALD |
| 疏水层 | AF 涂层 (如 Cytop / Teflon-AF) | $10\text{--}50\,\text{nm}$ | 1.9–2.1 | $10^{-14}\text{--}10^{-16}$ | 1.3–1.4 | 旋涂 / 浸涂 |
| 油相 | 十二烷 / 硅油 | $10\text{--}30\,\mu\text{m}$ | 2.0–2.5 | $< 10^{-12}$ | 1.4–1.5 | 填充注入 |
| 水相 | 去离子水 + 电解质 | $20\text{--}70\,\mu\text{m}$ | 78–80 | $10^{-4}\text{--}10^{-2}$ | 1.33 | 填充注入 |
| 盖板 | 玻璃 / PET | $0.5\text{--}1\,\text{mm}$ | 4–7 | $< 10^{-14}$ | 1.5–1.7 | 封装键合 |

该表为 PINN 模型中各层材料参数提供了参考取值范围，实际建模时建议以器件实测值为准。

---

# 4 流体动力学控制方程

## 4.1 弱可压缩连续性方程

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = -\frac{1}{c_{\text{art}}^2}\frac{Dp}{Dt} \tag{1}$$

其中 $c_{\text{art}}$ 为人工声速（弱可压缩近似下取大值以逼近不可压缩），等号右端项允许密度随压力的微弱变化，避免完全不可压缩假设下 PINN 的压力-速度解耦困难。

## 4.2 三维 Navier-Stokes 动量方程

$$\rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \nabla \cdot (\mu \nabla \mathbf{u}) + \mathbf{F}_{\text{csf}} + \mathbf{F}_{\text{elec}} + \mathbf{F}_{\text{mar}} \tag{2}$$

其中：
- $\mathbf{F}_{\text{csf}}$：连续表面张力（CSF 模型），$\mathbf{F}_{\text{csf}} = \gamma \kappa \mathbf{n} \delta_\Gamma$
- $\mathbf{F}_{\text{elec}}$：电场体积力，$\mathbf{F}_{\text{elec}} = -\frac{1}{2}\varepsilon_0 \varepsilon_r |\nabla\varphi|^2 \nabla\varepsilon_r$（Korteweg-Helmholtz 力密度的简化形式；已略去电致伸缩项 $\frac{1}{2}\varepsilon_0 \nabla(|\nabla\varphi|^2 \rho\,\partial\varepsilon_r/\partial\rho)$，因其在弱可压缩/不可压缩近似下可并入修正压力梯度，不影响速度场动力学）
- $\mathbf{F}_{\text{mar}}$：Marangoni 界面切向力（参见 §5.4）。在数值实现时作为体积力 $\mathbf{F}_{\text{mar}} = \tau_M \delta_\Gamma$ 加入，其中 $\tau_M$ 由式(8)定义

$\frac{D}{Dt} = \frac{\partial}{\partial t} + \mathbf{u} \cdot \nabla$ 为物质导数。

### 流动状态与无量纲分析

电润湿像素内的流动特征可通过以下无量纲数表征，为方程简化提供依据：

| 无量纲数 | 定义 | 典型值 | 物理含义 |
|:---|:---:|:---:|:---|
| Reynolds 数 $Re$ | $\rho U L_c / \mu$ | $0.01\text{--}1$ | 惯性力/粘性力之比，微尺度下层流主导 |
| Capillary 数 $Ca$ | $\mu U / \gamma$ | $10^{-4}\text{--}10^{-2}$ | 粘性力/界面张力之比，界面张力主导 |
| Weber 数 $We$ | $\rho U^2 L_c / \gamma$ | $10^{-6}\text{--}10^{-3}$ | 惯性力/界面张力之比，可忽略惯性 |
| Bond 数 $Bo$ | $\Delta\rho g L_c^2 / \gamma$ | $10^{-3}\text{--}10^{-1}$ | 重力/界面张力之比，微尺度下重力可忽略 |

其中特征长度 $L_c \sim 10\text{--}100\,\mu\text{m}$，特征速度 $U \sim 10^{-4}\text{--}10^{-2}\,\text{m/s}$（由电润湿驱动）。上表表明，微像素流动处于 **Stokes 流近似区域**（$Re \ll 1$, $Ca \ll 1$），惯性项可忽略，粘性和界面张力是主导力。然而，本模型保留完整 NS 方程而非简化为 Stokes 方程，以兼容瞬态加速阶段（驱动电压切换瞬间）的局部惯性效应。

### 重力与浮力效应

式 (2) 中隐去了重力项 $\rho \mathbf{g}$（$\mathbf{g}$ 为重力加速度矢量），基于 Bond 数 $Bo \ll 1$ 的量级估计。但在以下工况下，重力/浮力效应需显式保留并加入动量方程：

**竖直放置像素**：当像素基板竖直放置（如手机竖屏）时，重力方向平行于壁面（$\mathbf{g} \perp \mathbf{n}_{\text{wall}}$），油-水密度差 $\Delta\rho = \rho_w - \rho_o \sim 100\text{--}200\,\text{kg/m}^3$ 产生的浮力驱动油膜向上漂移，破坏水平放置时的中心对称油膜形貌。此时有效 Bond 数为 $Bo_{\text{eff}} = \Delta\rho g L_c d_o / \gamma$，在 $L_c > 100\,\mu\text{m}$ 时可接近 $O(1)$。

**大像素/大油量工况**：当像素边长 $L > 500\,\mu\text{m}$ 或油膜厚度 $d_o > 50\,\mu\text{m}$ 时，$Bo$ 可能达到 $0.1\text{--}1$，重力对界面形貌的修正不再可忽略。重力倾向于使油-水界面趋于水平（最小势能原理），与电润湿驱动的界面变形相互竞争。

**含重力项的完整动量方程**：

$$\rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \nabla \cdot (\mu \nabla \mathbf{u}) + \mathbf{F}_{\text{csf}} + \mathbf{F}_{\text{elec}} + \mathbf{F}_{\text{mar}} + \rho \mathbf{g} \tag{2a}$$

在 PINN 训练中，重力项 $\rho \mathbf{g}$ 无需额外损失项——它直接进入 NS 残差 $\mathcal{L}_{\text{NS}}$。对于水平放置像素（$\mathbf{g} \parallel \mathbf{n}_{\text{wall}}$），重力仅贡献静水压力分量 $\rho g z$，可吸收进压力场 $p$ 的定义中（即以修正压力 $p^* = p - \rho g z$ 替代 $p$），从而形式上消除重力项。

### 动量方程中各力项的物理详述

**连续表面张力 $\mathbf{F}_{\text{csf}}$** 是两相界面上的面积力，通过 CSF (Continuum Surface Force) 模型转化为体积力作用于界面过渡区（$\phi \in [0.2, 0.8]$）。其方向沿界面法向，大小与界面张力 $\gamma$ 和局部曲率 $\kappa$ 成正比。在实际计算中，$\delta_\Gamma = |\nabla\phi|$ 将面积力平滑分布到界面附近薄层（厚度 $\sim 2\varepsilon_{\text{intf}}$，$\varepsilon_{\text{intf}}$ 为 VOF 界面半宽）。

**电场体积力 $\mathbf{F}_{\text{elec}}$** 源于介质极化应力，完整形式包含两部分：Korteweg-Helmholtz 力密度 $\frac{1}{2}\varepsilon_0 |\nabla\varphi|^2 \nabla\varepsilon_r$（作用于介电常数梯度区，即两相界面和介质层界面）和电致伸缩力 $\frac{1}{2}\varepsilon_0 \nabla(|\nabla\varphi|^2 \rho\,\partial\varepsilon_r/\partial\rho)$。式 (2) 中仅保留 Korteweg-Helmholtz 项——因为在弱可压缩近似下，电致伸缩项为某标量场的全梯度，可与压力梯度 $\nabla p$ 合并为修正压力梯度 $\nabla p^*$，不产生净体积力。这一处理与式 (2a) 中重力项通过修正压力吸收的逻辑一致。在电润湿像素中，$\mathbf{F}_{\text{elec}}$ 主要集中在油-水界面和疏水层-流体界面上。

**Marangoni 力 $\mathbf{F}_{\text{mar}}$** 作用于界面切向，仅当界面张力沿界面非均匀（由温度梯度或表面活性剂浓度梯度引起）时出现。Marangoni 效应的净效果是驱动界面流体从低张力区流向高张力区，可能对抗或增强电润湿驱动的铺展运动。

## 4.3 弱可压缩近似的理论与数值依据

两相流体物性通过 VOF 相场 $\phi$ 线性插值，同时引入温度依赖：

$$\begin{aligned}
\rho(\phi, T) &= \phi\,\rho_o(T) + (1-\phi)\,\rho_w(T) \\
\mu(\phi, T) &= \phi\,\mu_o(T) + (1-\phi)\,\mu_w(T)
\end{aligned} \tag{3}$$

其中 $\rho_o, \mu_o$ 为油相密度与动力粘度，$\rho_w, \mu_w$ 为水相密度与动力粘度，均为温度的函数（常见形式为 $\rho(T) = \rho_0[1 - \beta_T(T - T_0)]$，$\mu(T) = \mu_0 \exp(E_a/RT)$）。

## 4.4 壁面 Navier 滑移边界条件

在疏水壁面上，采用 Navier 滑移边界条件替代传统无滑移条件，以解决三相接触线的应力奇异性问题：

$$u_\tau = l_s \frac{\partial u_\tau}{\partial n} \tag{4}$$

其中 $u_\tau$ 为壁面切向速度，$l_s$ 为滑移长度（典型值 $l_s \sim 10\text{--}100\,\text{nm}$），$n$ 为壁面法向。

**弱可压缩近似的理论依据**：严格不可压缩条件 $\nabla \cdot \mathbf{u} = 0$ 在 PINN 中作为软约束时，压力场存在零模不确定性（压力仅以梯度出现，任意常数平移不改变方程残差），导致训练收敛困难。引入弱可压缩项 $c_{\text{art}}^{-2} D_t p$ 后：压力场通过声学时间尺度 $\tau_{\text{acoustic}} = L_c / c_{\text{art}}$ 与速度场解耦消除，数值上设置 $c_{\text{art}}$ 使 $\tau_{\text{acoustic}} \ll \tau_{\text{flow}}$（流动特征时间），保证密度波动 $\Delta\rho/\rho_0 \sim Ma^2 < 10^{-3}$（$Ma = U/c_{\text{art}}$ 为 Mach 数）。实践中取 $c_{\text{art}} \sim 10\text{--}100\,\text{m/s}$，对应 $Ma \sim 10^{-5}\text{--}10^{-3}$，弱可压缩近似对不可压缩流动的逼近误差可忽略。

**滑移边界的微观物理基础**：疏水表面上的 Navier 滑移条件源于固-液界面的分子尺度效应——疏水表面与水分子间的弱氢键作用（相对于体相水的强氢键网络）及界面水 depletion 效应导致界面附近形成低密度"气隙"层（depletion layer，厚度 $\sim 0.1\text{--}1\,\text{nm}$），该层内的水分子所受横向约束显著减弱，可沿表面切向滑动。滑移长度 $l_s$ 与该气隙层的厚度和疏水表面的本征接触角相关：$l_s$ 随接触角增大而增大（超疏水表面 $l_s$ 可达 $\mu\text{m}$ 量级）。引入滑移边界不仅消除了三相接触线的运动学奇点（传统无滑移边界下，接触线无法同时满足无滑移和界面运动），也使得 PINN 预测的接触线速度与分子动力学模拟和实验测量保持一致。

## 4.5 控制方程无量纲化

PINN 训练涉及空间尺度从 nm 到 mm、时间尺度从 $\mu\text{s}$ 到 ms 的跨数量级变量，直接在原始物理单位下训练会导致梯度范数相差 $10^6$ 倍以上，引起严重的优化病态。因此，所有控制方程在输入 PINN 前需进行无量纲化处理。

### 特征量选取

| 特征量 | 符号 | 定义 | 典型值 |
|:---|:---:|:---|:---:|
| 特征长度 | $L_c$ | 像素半宽 $L/2$ | $75\,\mu\text{m}$ |
| 特征速度 | $U_c$ | 电润湿驱动速度 $\varepsilon_0 \varepsilon_r V_0^2 / (\mu d_d)$ | $10^{-3}\text{--}10^{-2}\,\text{m/s}$ |
| 特征压力 | $p_c$ | 毛细压强 $\gamma / L_c$ | $\sim 200\,\text{Pa}$ |
| 特征时间 | $t_c$ | 对流时间 $L_c / U_c$ | $10^{-3}\text{--}10^{-1}\,\text{s}$ |
| 特征电势 | $\varphi_c$ | 驱动电压 $V_0$ | $5\text{--}30\,\text{V}$ |
| 特征电场 | $E_c$ | $V_0 / d_d$ | $\sim 10^7\,\text{V/m}$ |

定义无量纲变量（以上标 $*$ 表示）：

$$x^* = \frac{x}{L_c},\; y^* = \frac{y}{L_c},\; z^* = \frac{z}{L_c},\; t^* = \frac{t}{t_c},\; \mathbf{u}^* = \frac{\mathbf{u}}{U_c},\; p^* = \frac{p}{p_c},\; \varphi^* = \frac{\varphi}{\varphi_c}$$

### 无量纲化后的控制方程

**连续性方程**：

$$\frac{\partial \rho^*}{\partial t^*} + \nabla^* \cdot (\rho^* \mathbf{u}^*) = -\frac{1}{Ma^2} \frac{D^* p^*}{D^* t^*} \tag{1a}$$

其中 $Ma = U_c / c_{\text{art}}$ 为 Mach 数，$\rho^* = \rho/\rho_o$。

**Navier-Stokes 动量方程**（以毛细压强 $\gamma/L_c$ 为基准压力进行无量纲化）：

$$We\,\rho^* \frac{D^*\mathbf{u}^*}{D^*t^*} = -\nabla^* p^* + Ca\,\nabla^* \cdot (\mu^* \nabla^* \mathbf{u}^*) + \mathbf{F}_{\text{csf}}^* + \eta\,\mathbf{F}_{\text{elec}}^* + Ma_T\,\mathbf{F}_{\text{mar}}^* + Bo\,\rho^* \hat{\mathbf{g}} \tag{2b}$$

其中无量纲参数定义为：

| 无量纲数 | 定义 | 物理意义 |
|:---|:---:|:---|
| Weber 数 $We$ | $\rho_o U_c^2 L_c / \gamma$ | 惯性/毛细压强 |
| Capillary 数 $Ca$ | $\mu_o U_c / \gamma$ | 粘性/界面张力 |
| 电润湿数 $\eta$ | $\varepsilon_0 \varepsilon_r V_0^2 / (2 \gamma d_d)$ | 电场力/界面张力 |
| 热 Marangoni 数 $Ma_T$ | $(d\gamma/dT) \Delta T L_c / (\mu_o U_c)$ | 热毛细/粘性 |
| Bond 数 $Bo$ | $\Delta\rho g L_c^2 / \gamma$ | 重力/界面张力 |
| Mach 数 $Ma$ | $U_c / c_{\text{art}}$ | 流速/人工声速 |

**VOF 输运方程**（无量纲形式不变，因其齐次性）：

$$\frac{\partial \phi}{\partial t^*} + \mathbf{u}^* \cdot \nabla^* \phi = 0 \tag{5a}$$

**电场方程**：

$$\nabla^* \cdot \big((\sigma^* + \varepsilon^* \partial_{t^*}) \nabla^* \varphi^* \big) = 0 \tag{13a}$$

其中 $\sigma^* = \sigma t_c / (\varepsilon_0 \varepsilon_r)$ 为无量纲电导率。

### 无量纲化对 PINN 训练的数值意义

无量纲化后，所有物理变量及其空间导数均处于 $O(10^{-2})\text{--}O(10^2)$ 范围内，梯度范数差异从原始单位的 $O(10^6)$ 缩小至 $O(10^2)$。这带来三个直接好处：
1. 有效学习率在各损失项间均衡，减少了自适应权重调度的负担；
2. 网络参数的初始化可使用标准方案（如 Xavier），无需针对各输出通道分别缩放；
3. 傅里叶特征编码的频率参数 $\sigma$ 可直接以无量纲长度标定（$\sigma \sim 1\text{--}10$ 对应分辨 $0.1L_c\text{--}L_c$ 的空间特征）。

---

# 5 VOF 相场与界面力学连续模型

## 5.1 VOF 相场输运方程

两相界面运动由 VOF 无扩散纯对流输运方程描述：

$$\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = 0 \tag{5}$$

该方程刻画 $\phi$ 随流体速度 $\mathbf{u}$ 的对流输运，保证界面以物质速度运动。

**VOF 相场的物理本质**：式 (5) 表征的是界面处密度间断面的物质输运，其理论基础是两相流体的不可混溶性——界面两侧流体微元保持各自的物质身份。从统计力学角度看，$\phi$ 可解释为油相分子在空间点 $(x,y,z,t)$ 的占有概率，$0 < \phi < 1$ 的过渡区对应物理界面的有限厚度（$\sim 1\text{--}10\,\text{nm}$）。PINN 以连续可微神经网络表示 $\phi(x,y,z,t)$，其隐式正则性（神经网络固有的光滑性偏好）天然抑制了传统 VOF 方法中常见的界面非物理振荡。

**与 Diffuse Interface 方法的区别**：本模型采用 sharp-interface VOF 输运方程（纯对流无扩散），而非 Cahn-Hilliard 型的扩散界面模型。这一选择基于微像素的尺度特征——界面厚度（$\sim \text{nm}$）远小于像素尺寸（$\sim 100\,\mu\text{m}$），sharp-interface 近似足够精确。同时，避免了引入 Cahn-Hilliard 方程中四阶空间导数带来的 PINN 高阶求导计算负担和化学势参数的标定困难。

## 5.2 界面法向量与曲率正则化

为避免 $\nabla\phi$ 过小时数值不稳定，引入小量 $\delta$ 正则化（$\delta \sim 10^{-8}$）：

$$\mathbf{n} = \frac{\nabla\phi}{|\nabla\phi| + \delta}, \qquad \kappa = \nabla \cdot \mathbf{n} \tag{6}$$

其中 $\mathbf{n}$ 为界面单位法向量（由水相指向油相），$\kappa$ 为界面平均曲率（$\kappa > 0$ 表示界面凸向水相）。

## 5.3 界面法向应力连续条件（Young-Laplace 广义形式）

跨界面压力跃变由曲率引起的毛细压强和电场附加压强共同决定：

$$p_o - p_w = \gamma_{ow} \kappa + \Delta p_{\text{elec}} \tag{7}$$

其中 $\gamma_{ow}$ 为油-水界面张力（典型值 $\sim 0.015\text{--}0.05\,\text{N/m}$），$\Delta p_{\text{elec}}$ 为电场作用在界面上的附加法向压强（参见式 12）。

## 5.4 Marangoni 界面切向力

界面张力沿界面的非均匀分布产生切向 Marangoni 应力：

$$\tau_M = \frac{d\gamma_{ow}}{dT} \nabla_\parallel T + \frac{d\gamma_{ow}}{dC} \nabla_\parallel C \tag{8}$$

其中 $\nabla_\parallel = (\mathbf{I} - \mathbf{n}\mathbf{n}) \cdot \nabla$ 为界面梯度算子。第一项为热致 Marangoni 效应（热毛细对流），第二项为浓度致 Marangoni 效应。

在数值实现中，该切应力通过 $\mathbf{F}_{\text{mar}} = \tau_M \delta_\Gamma$ 转化为体积力源项加入动量方程。

### CSF 模型的数值实现要点

在 PINN 框架中实现 CSF 表面张力模型需注意以下数值细节：

- **界面 Dirac 函数的正则化**：$\delta_\Gamma = |\nabla\phi|$ 是界面的数值 Dirac 型分布函数。当界面过渡区太薄时（$|\nabla\phi|$ 极大但作用范围极窄），$\mathbf{F}_{\text{csf}}$ 会退化为近似点力，引起 PINN 的梯度病态。实践中通过限制 $|\nabla\phi| \leq |\nabla\phi|_{\max}$ 或适度增大界面过渡区宽度来缓解。
- **曲率计算的稳定性**：$\kappa = \nabla \cdot \mathbf{n}$ 涉及 $\phi$ 的二阶导数，在 PINN 自动微分中通过计算图传播。为使 $\kappa$ 场的光滑性足够，网络需具备充足的隐式正则化（通过谱偏置或傅里叶特征编码）。
- **与界面应力连续条件的自洽性**：CSF 体积力 $\mathbf{F}_{\text{csf}}$ 的散度应与 Young-Laplace 压力跃变（式 7）在界面法向上自洽——即 $\mathbf{n} \cdot (\nabla p + \mathbf{F}_{\text{csf}})$ 在界面上应趋近于零。这一自洽性可通过在 $\mathcal{L}_{\text{Intf}}$ 中显式添加交叉约束项来增强。

### Young-Laplace 方程的广义推导

式 (7) 的广义 Young-Laplace 方程可从界面应力张量跳跃条件严格推导。设界面应力张量 $\mathbf{T} = -p\mathbf{I} + \mu(\nabla\mathbf{u} + \nabla\mathbf{u}^T) + \mathbf{T}_{\text{Maxwell}}$（含 Maxwell 电磁应力张量 $\mathbf{T}_{\text{Maxwell}} = \varepsilon\mathbf{E}\mathbf{E} - \frac{1}{2}\varepsilon E^2\mathbf{I}$），跨界面法向应力连续条件给出：

$$\mathbf{n} \cdot [\mathbf{T}]_{w}^{o} \cdot \mathbf{n} = \gamma \kappa$$

其中 $[\cdot]_{w}^{o}$ 表示从水相到油相的跳跃量。展开 Maxwell 应力跳跃项即得式 (7) 中的电场附加压强 $\Delta p_{\text{elec}} = \frac{1}{2}\varepsilon_0 \varepsilon_r (E_n^2 - E_t^2)|_{w}^{o}$，其主导项在法向电场远大于切向电场时简化为式 (12) 的 Lippmann 形式。

---

# 6 电场-多层介质-导电方阻耦合模型

## 6.1 场强依赖非线性介电常数

介电层和疏水层的介电常数在强电场下表现出非线性极化饱和效应：

$$\varepsilon_d(E) = \varepsilon_{d0} + \alpha E^2, \qquad \varepsilon_f(E) = \varepsilon_{f0} + \beta E^2 \tag{9}$$

其中 $\varepsilon_{d0}, \varepsilon_{f0}$ 为低场介电常数，$\alpha, \beta$ 为非线性极化系数（二阶电光 Kerr 效应），$E = |\nabla\varphi|$ 为局部电场强度。

## 6.2 多层介质等效串联电容

介电层与疏水层串联构成等效单位面积电容：

$$C_{\text{eq}}(x,y) = \frac{\varepsilon_0}{\displaystyle\frac{d_d(x,y)}{\varepsilon_d(E)} + \frac{d_f(x,y)}{\varepsilon_f(E)}} \tag{10}$$

厚度 $d_d, d_f$ 的空间变化直接映射为电容的空间非均匀分布。

## 6.3 含非均匀方阻的面内电压衰减

ITO 电极的面内方阻 $R_s(x,y)$ 与介质等效电容 $C_{\text{eq}}$ 构成分布式 RC 传输线，造成电压从像素边缘向中心的指数衰减：

$$V(x,y) = V_0 \exp\!\left(-\frac{\sqrt{x^2 + y^2}}{\sqrt{R_s(x,y)\, C_{\text{eq}}(x,y)}}\,\right) \tag{11}$$

其中 $V_0$ 为像素边缘施加的驱动电压。方阻越大、电容越大，电压衰减越显著，像素边角与中心的电压差可达 $5\text{--}15\%$。

## 6.4 电润湿界面附加电场压强

由 Lippmann-Young 电润湿理论，介质层储存的静电能在界面上产生等效附加压强：

$$\Delta p_{\text{elec}} = \frac{1}{2} C_{\text{eq}} V^2(x,y) \tag{12}$$

该附加压强降低了油-水界面下方的有效界面张力，驱动三相接触线向外铺展。

## 6.5 介电层漏电流准静态电场方程

考虑介电层的有限电导率（漏电流），电场由含损耗的 Maxwell-Wagner 方程描述：

$$\nabla \cdot \big((\sigma + \varepsilon \partial_t) \nabla \varphi \big) = 0 \tag{13}$$

其中 $\sigma$ 为介质电导率（典型值 $\sigma \sim 10^{-12}\text{--}10^{-14}\,\text{S/m}$），$\varepsilon$ 为介电常数。$\partial_t \nabla\varphi$ 项描述介质极化对时变电场的动态响应。

### RC 传输线模型的详细推导

式 (11) 的电压衰减公式源于 ITO-介质-流体的分布式 RC 传输线方程。考虑 ITO 电极面内的电流连续性：

$$\nabla_s \cdot \left(\frac{1}{R_s(x,y)} \nabla_s V\right) = C_{\text{eq}}(x,y) \frac{\partial V}{\partial t}$$

其中 $\nabla_s$ 为面内梯度算子（仅作用于 $x,y$）。该方程描述：ITO 面内电流在方阻 $R_s$ 作用下产生欧姆压降 $\propto R_s$，同时电流通过介质电容 $C_{\text{eq}}$ 向流体域充电。在稳态极限下（$\partial_t V = 0$），圆对称近似（$r = \sqrt{x^2+y^2}$）给出解析解 $V(r) = V_0 \exp(-r / \sqrt{R_s C_{\text{eq}}})$，即式 (11) 的形式。

**频率依赖性**：当驱动电压为交流信号（频率 $f$）时，介质层的电容阻抗 $Z_C = 1/(j\omega C_{\text{eq}})$ 与方阻 $R_s$ 构成低通 RC 滤波器。截至频率 $f_c = 1/(2\pi R_s C_{\text{eq}} L^2)$ 决定了像素中心的电压跟随能力。对于典型参数（$R_s \sim 50\,\Omega/\square$，$C_{\text{eq}} \sim 10^{-4}\,\text{F/m}^2$，$L \sim 150\,\mu\text{m}$），$f_c \sim 10\,\text{MHz}$，远高于电润湿显示的典型驱动频率（$\sim 1\text{--}10\,\text{kHz}$），因此在准静态驱动下电压衰减效应占主导。

### 边缘电场畸变效应

像素边角处的几何不连续性导致局部电场集中（edge fringing effect），该区域的电场强度可比像素中心高 $10\text{--}30\%$。边缘电场畸变产生两个重要后果：一是边角处的 $\Delta p_{\text{elec}}$ 增强，导致接触线在边角处优先铺展（"角钉扎效应"）；二是边角处的介质层承受更高电场应力，成为介质击穿的薄弱点。

在 PINN 框架中，边缘电场效应通过求解含空间变化介电常数的完整电场方程（式 13）自然捕获，无需引入额外的经验边缘场修正因子。像素边角区域的采样点密度应适当提高（$2\times\text{--}3\times$ 内部密度），以精确分辨边缘场的空间变化。

### 电化学反应与水解阈值

驱动电压超过水的电化学稳定窗口（$\sim 1.23\,\text{V}$）时，水在电极-电解质界面处发生电解反应：

$$\text{阳极: } 2\text{H}_2\text{O} \to \text{O}_2 \uparrow + 4\text{H}^+ + 4e^- \qquad \text{阴极: } 2\text{H}_2\text{O} + 2e^- \to \text{H}_2 \uparrow + 2\text{OH}^-$$

然而，在电润湿像素中，介质层（$\text{SiN}_x$ 或 $\text{Al}_2\text{O}_3$）将 ITO 电极与水相物理隔离，实际水解电压远高于热力学阈值——介质层承担了绝大部分电压降（电容分压原理）。水-介质界面的实际电位差为：

$$\Delta V_{\text{water}} \approx V_0 \cdot \frac{C_{\text{eq}}}{C_{\text{EDL}}} \ll V_0$$

其中 $C_{\text{EDL}}$ 为水侧双电层电容（$\sim 10^{-1}\text{--}10^{0}\,\text{F/m}^2$），$C_{\text{eq}}$ 为介质等效电容（$\sim 10^{-4}\text{--}10^{-3}\,\text{F/m}^2$）。二者之比 $C_{\text{eq}}/C_{\text{EDL}} \sim 10^{-4}\text{--}10^{-3}$，因此即使 $V_0 = 30\,\text{V}$，水层实际电压降仅 $\sim 3\text{--}30\,\text{mV}$，远低于水解阈值。

**水解风险场景**：
- 介质层存在针孔缺陷（pinhole）时，局部电流密度剧增，在该缺陷点触发水解产气；
- 交流驱动频率极低（$f < 1\,\text{Hz}$）或直流长时间保持时，介质层慢极化完成后水层分压逐渐升高；
- 高湿度环境下，水分子渗透进入介质层微裂纹，降低有效电阻率。

**对 PINN 模型的影响**：产生气泡会破坏 VOF 界面连续性，引入三相（油-水-气）复杂界面。当前模型假设无气相存在，若实验中出现明显气泡，需扩展为三相 VOF 模型或在水解阈值以上施加电压上限约束。

### 介电层击穿场强与安全裕度

介质层可承受的最大电场由本征击穿场强 $E_{\text{BD}}$ 决定：

| 介质材料 | $E_{\text{BD}}$ (MV/cm) | 典型工作场强 (MV/cm) | 安全系数 |
|:---|:---:|:---:|:---:|
| $\text{SiN}_x$ (PECVD) | 5–8 | 1–3 | 2–5× |
| $\text{Al}_2\text{O}_3$ (ALD) | 8–12 | 1–3 | 3–10× |
| AF 涂层 (Cytop) | 2–4 | 0.5–1.5 | 2–4× |

工作场强取 $E_{\text{work}} = V_0 / d_d$（$V_0 \sim 15\text{--}30\,\text{V}$，$d_d \sim 200\text{--}400\,\text{nm}$）。击穿风险最高的位置是像素边角边缘电场集中区——该处局部场强可比平均场强高 $30\%\text{--}50\%$，因此设计阶段需以边缘峰值场强（而非平均场强）评估击穿安全裕度。

**时间相关介电击穿 (TDDB)**：即使工作场强低于本征击穿阈值，长期直流应力下介质层可能发生时间相关介电击穿。其寿命遵循经验模型 $t_{\text{BD}} = t_0 \exp(-\gamma E)$（$\gamma$ 为场加速因子，$\sim 1\text{--}3\,\text{cm/MV}$）。PINN 模型在设计优化中可将 $E_{\text{peak}}$ 作为约束条件，确保器件寿命满足目标（如 $> 10^4$ 小时）。

---

# 7 固壁润湿边界与三相接触线模型

## 7.1 接触角滞后动态准则（CAH）

三相接触线在运动过程中，其微观接触角 $\theta$ 取决于运动方向，在前进角 $\theta_A$ 与后退角 $\theta_R$ 之间变化：

$$\mathbf{n} \cdot \mathbf{n}_{\text{wall}} =
\begin{cases}
\cos\theta_A & \text{接触线向外铺展（advancing）} \\[4pt]
\cos\theta_R & \text{接触线向内收缩（receding）}
\end{cases} \tag{14}$$

典型值：$\theta_A \approx 110^\circ\text{--}130^\circ$，$\theta_R \approx 70^\circ\text{--}90^\circ$，滞后角 $\Delta\theta = \theta_A - \theta_R \sim 20^\circ\text{--}40^\circ$。

接触线运动方向由 $\mathbf{u} \cdot \mathbf{n}_{\text{wall}}$ 的符号判定：正为铺展，负为收缩。

**数值平滑处理（可选）**：式 (14) 的分段常数逻辑在 $\mathbf{u} \cdot \mathbf{n}_{\text{wall}} = 0$ 处存在不连续跳跃。若训练中接触线运动方向频繁切换（如电压极性交变驱动），该不连续性可能导致接触线附近的梯度计算不稳定。数字上可采用平滑过渡函数替代分段常数：

$$\cos\theta = \cos\theta_0 + \frac{\Delta\theta}{2} \tanh\!\left(\beta\,\mathbf{u} \cdot \mathbf{n}_{\text{wall}}\right)$$

其中 $\theta_0 = (\theta_A + \theta_R)/2$ 为平均接触角，$\Delta\theta = \theta_A - \theta_R$ 为滞后幅值，$\beta$ 为过渡尺度因子（典型值 $\beta \sim 10^2\text{--}10^4\,\text{s/m}$，控制过渡区宽度——$\beta$ 越大越逼近分段常数）。该平滑形式在整个速度域内 $C^\infty$ 连续，消除了 CAH 模型的梯度间断点，在 PINN 训练中比原始分段形式更稳定。此为可选的数值策略，当使用分段 CAH 训练无稳定性问题时无需启用。

## 7.2 Wenzel 表面粗糙度等效修正

疏水层表面的微观粗糙度 $r$（实际表面积与投影面积之比）通过 Wenzel 模型放大表观接触角：

$$\cos\theta_m = r \cos\theta_{\text{ideal}} \tag{15}$$

其中 $r \geq 1$（典型值 $r \sim 1.1\text{--}1.5$），$\theta_m$ 为粗糙表面上的表观接触角，$\theta_{\text{ideal}}$ 为理想光滑表面上的本征接触角。该修正表明：粗糙度使疏水表面更疏水（$\theta$ 增大），亲水表面更亲水（$\theta$ 减小）。

### 分子动力学理论 (MKT) 动态接触角模型

接触角滞后本质上源于三相接触线的分子尺度动力学。根据 Blake-Haynes 分子动力学理论 (Molecular Kinetic Theory, MKT)，接触线的宏观运动速度 $U_{\text{cl}}$ 与微观接触角 $\theta$ 偏离平衡角 $\theta_0$ 的程度满足：

$$U_{\text{cl}} = 2\kappa_0 \lambda \sinh\!\left[\frac{\gamma (\cos\theta_0 - \cos\theta)}{2n k_B T}\right]$$

其中 $\kappa_0$ 为分子跳跃频率（$\sim 10^6\text{--}10^{10}\,\text{Hz}$），$\lambda$ 为分子跳跃步长（$\sim 0.5\text{--}2\,\text{nm}$，对应于固体表面吸附位点间距），$n$ 为单位面积吸附位点数（$n \sim 1/\lambda^2$）。当 $U_{\text{cl}} \to 0$ 时，$\theta$ 趋于平衡角 $\theta_0$；当 $|U_{\text{cl}}|$ 增大时，$\theta$ 偏离 $\theta_0$ 更多，表现为接触角滞后。

**量纲一致性确认**：$\sinh$ 函数的参数 $\gamma(\cos\theta_0 - \cos\theta)/(2n k_B T)$ 的量纲为 $[\text{J/m}^2] / ([\text{m}^{-2}] \cdot [\text{J}]) = 1$（无量纲），满足超越函数的输入要求。前置因子 $2\kappa_0 \lambda$ 的量纲为 $[\text{s}^{-1}] \cdot [\text{m}] = [\text{m/s}]$，与接触线速度 $U_{\text{cl}}$ 一致。

**MKT 与 CAH 的关系**：式 (14) 的 CAH 模型是 MKT 的工程简化——前进角 $\theta_A$ 和后退角 $\theta_R$ 分别对应 $U_{\text{cl}} \to +\infty$ 和 $U_{\text{cl}} \to -\infty$ 极限下的动态接触角（实际由表面化学异质性和粗糙度决定饱和值）。在 PINN 训练中，根据精度和计算成本需求，可选用简化的 CAH 模型（式 14）或完整的 MKT 动态接触角模型。

### 接触线钉扎与表面异质性

实际疏水表面存在化学异质性和物理粗糙度的空间分布，导致接触线在微纳尺度上的局部钉扎（pinning）现象。钉扎效应是接触角滞后的主要微观成因：

- **化学异质性钉扎**：疏水涂层中残留的亲水位点（如未完全氟化的区域）对水相具有强亲和力，形成高能吸附位点，接触线跨越这些位点时需克服额外的能量势垒。
- **拓扑钉扎**：表面微纳结构的几何锐边（如光刻胶残留的台阶边缘）对接触线产生几何约束，使接触线"锁定"在锐边处，表观接触角可在 Wenzel 允许范围内任意取值（Gibbs 不等式约束）。

在 PINN 建模中，表面异质性的统计效应通过接触角滞后范围 $[\theta_R, \theta_A]$ 参数化。如需更高分辨率的钉扎效应建模，可将壁面接触角设为空间坐标的函数 $\theta(x,y)$，并根据表面能分布的统计特征（AFM 或接触角测量数据）随机生成。

### Cassie-Baxter 复合润湿状态

当表面粗糙度足够高（$r > |\cos\theta_{\text{ideal}}|^{-1}$）时，液滴可能从 Wenzel 全润湿状态转变为 Cassie-Baxter 复合状态——液体不完全填充粗糙结构的凹槽，凹槽内截留空气（或油相），形成固-液和液-气复合界面。Cassie-Baxter 表观接触角为：

$$\cos\theta_{\text{CB}} = f_s \cos\theta_{\text{ideal}} - (1 - f_s)$$

其中 $f_s$ 为固-液接触面积分数（$0 < f_s < 1$）。在电润湿像素中，电压驱动下可能发生 Cassie-Baxter → Wenzel 的不可逆润湿转变（wetting transition），导致接触角突变和油膜行为异常。PINN 模型通过接触角的电压依赖性（式 12 结合式 14）隐式捕获这一转变的部分特征。

---

# 8 双电层/表面活性剂/热场耦合模型

## 8.1 Poisson-Boltzmann 双电层方程

固-液界面附近双电层中的静电势 $\psi$ 由 Poisson-Boltzmann 方程描述：

$$\nabla^2 \psi = \frac{2 e n_0}{\varepsilon} \sinh\!\left(\frac{e\psi}{k_B T}\right) \tag{16}$$

其中 $e$ 为基元电荷，$n_0$ 为体相离子数密度 (m$^{-3}$，区别于摩尔浓度 mol/L)，$k_B$ 为 Boltzmann 常数，$\varepsilon$ 为溶液介电常数。

**Debye 长度**：$\lambda_D = \sqrt{\frac{\varepsilon k_B T}{2e^2 n_0}}$，表征 EDL 的特征厚度（典型值 $\lambda_D \sim 1\text{--}100\,\text{nm}$，取决于离子浓度）。对于稀溶液（$n_0 \sim 10^{-4}\,\text{M}$），$\lambda_D \sim 30\,\text{nm}$；对于浓溶液（$n_0 \sim 0.1\,\text{M}$），$\lambda_D \sim 1\,\text{nm}$。

**Debye-Hückel 线性近似**：当 $e\psi \ll k_B T$（即 $\psi \ll 25\,\text{mV}$），式 (16) 可线性化为 $\nabla^2\psi = \kappa_D^2 \psi$，其中 $\kappa_D = 1/\lambda_D$。

**PINN 处理策略**：对于 EDL 与流动的耦合，通常在壁面附近采用加密采样点策略；EDL 内的电场体力 $\mathbf{F}_{\text{EDL}} = -\rho_e \nabla\psi$（$\rho_e$ 为净电荷密度）通过 PB 方程求解的 $\psi$ 场后处理获得，加入式 (2) 动量方程的源项。

## 8.2 表面活性剂浓度输运方程

表面活性剂（surfactant）在油-水界面上的浓度 $C$ 服从对流-扩散方程：

$$\frac{\partial C}{\partial t} + \mathbf{u} \cdot \nabla C = D \nabla^2 C \tag{17}$$

其中 $D$ 为表面活性剂的有效扩散系数（典型值 $D \sim 10^{-10}\text{--}10^{-9}\,\text{m}^2/\text{s}$）。该方程描述表面活性剂在界面上的迁移和浓度重新分布。

**边界条件**：在固壁上设定零通量条件 $\partial C/\partial n = 0$；在计算域边界采用周期性或 Dirichlet 条件。

## 8.3 界面张力浓度依赖关系

油-水界面张力随表面活性剂浓度的增加而线性降低（低浓度近似）：

$$\gamma_{ow}(C) = \gamma_0 - k_C C \tag{18}$$

其中 $\gamma_0$ 为无表面活性剂时的本征界面张力，$k_C$ 为界面张力浓度敏感系数。结合式 (8) 的浓度 Marangoni 项，形成表面活性剂驱动界面流动的自洽耦合。

## 8.4 非等温热传导方程

稳态对流-导热能量方程描述像素内的温度分布（忽略粘性耗散和焦耳热效应）：

$$\rho c_p (\mathbf{u} \cdot \nabla T) = k \nabla^2 T \tag{19}$$

其中 $c_p$ 为比热容（VOF 加权），$k$ 为热导率（VOF 加权）。温度场 $T$ 通过以下路径耦合到流体与界面：
- 物性温变：$\rho(T), \mu(T)$ 影响流场（式 3）
- 热 Marangoni 效应：$d\gamma/dT$ 产生界面切应力（式 8）
- 双电层方程：温度出现在 PB 方程的指数项中（式 16）

### 电水动力学 (Electrohydrodynamics, EHD) 耦合路径

电场通过以下四条路径影响流体运动，构成完整的 EHD 耦合机制：

1. **电润湿效应**（主导路径）：$\Delta p_{\text{elec}}$ 通过 Young-Laplace 方程改变界面力学平衡，驱动接触线运动 → 改变 $\phi$ 场 → 通过 CSF 改变流场。
2. **介电泳力**（体相路径）：非均匀电场对介电常数不同的两相流体施加的净体力，作用于界面附近，方向由 $\nabla\varepsilon_r$ 和 $\nabla|\mathbf{E}|^2$ 共同决定。
3. **电渗流**（EDL 路径）：EDL 中的净电荷在切向电场作用下产生 Coulomb 力 $\mathbf{F}_{\text{EDL}} = -\rho_e \nabla_\parallel \varphi$，驱动壁面附近的滑移流动。电渗流速度量级 $u_{\text{EOF}} \sim \varepsilon \zeta E_t / \mu$（Smoluchowski 公式，$\zeta$ 为壁面 zeta 电势）。
4. **Joule 热效应**：介质漏电流和流体欧姆损耗生热 $\dot{q}_J = \sigma |\nabla\varphi|^2$，通过温度场影响流体物性（$\mu(T), \rho(T)$）和界面张力（热 Marangoni），构成电-热-流三场耦合。

在电润湿像素的典型工况下，路径 1 的贡献远大于路径 2–4（约占净驱动力的 $> 90\%$）。但在高频率交流驱动（$f > 10\,\text{kHz}$）或高离子浓度溶液中，路径 3 和 4 的贡献不可忽略。

### EDL 与表面活性剂的界面竞争效应

疏水壁面上同时存在 EDL 电场和表面活性剂吸附层，两者对有效接触角产生相反的调控作用：EDL 电场通过降低固-水界面能促进润湿（接触角减小）；表面活性剂在固-液界面的吸附通常增大接触角（疏水尾链朝外排列）。在 PINN 模型中，两者的净效应通过以下自洽关系耦合：

$$\gamma_{sw}(\psi, C) - \gamma_{so}(\psi, C) = \gamma_{ow}(C) \cos\theta_{\text{eff}}$$

即固-水和固-油界面张力同时依赖于 EDL 电势 $\psi$（通过 Lippmann 充电效应 $\gamma_{sw}(\psi) = \gamma_{sw}^0 - \frac{1}{2}C_{\text{EDL}}\psi^2$，$C_{\text{EDL}}$ 为 EDL 电容）和表面活性剂浓度 $C$（通过 Langmuir 吸附等温线 $\gamma(C) = \gamma_0 + RT\Gamma_\infty \ln(1 - C/C_\infty)$）。$\theta_{\text{eff}}$ 为同时考虑两种效应的有效接触角。

### 非等温效应的相对重要性评估

温度变化的主要来源包括：环境温度波动、Joule 热（介质漏电流和 ITO 电阻损耗）、以及粘性耗散。对电润湿像素进行量级估计：
- Joule 热功率密度 $\dot{q}_J \sim \sigma E^2 \sim 10^{-14} \times (10^7)^2 \sim 1\,\text{W/m}^3$（介质层）
- 粘性耗散 $\Phi \sim \mu (U/d)^2 \sim 10^{-3} \times (10^{-3}/10^{-5})^2 \sim 10\,\text{W/m}^3$
- 二者的绝热温升在毫秒时间尺度内均 $< 0.1\,\text{K}$

因此，对常规直流或低频驱动，等温假设足够精确。温度场 $T$ 和非等温耦合项在本模型中被保留用于以下场景：(a) 高功率密度交流驱动（$f > 100\,\text{kHz}$），(b) 环境温度极端变化（$-20^\circ\text{C}$ 到 $+60^\circ\text{C}$ 的户外显示应用），(c) 激光或其他外部加热辅助的电润湿操控。

---

# 9 吉布斯自由能泛函与守恒约束

## 9.1 总吉布斯自由能泛函

系统总吉布斯自由能包含界面能、电场能、内能、熵贡献与压力功项：

$$G_{\text{total}} = \sum \gamma A - \frac{1}{2} C_{\text{eq}} V^2 A_{sw} + U - TS + pV \tag{20}$$

其中：
- $\sum \gamma A$：所有界面（固-油、固-水、油-水）的界面能总和
- $-\frac{1}{2}C_{\text{eq}}V^2 A_{sw}$：介电层储存的静电场能（$A_{sw}$ 为固-水接触面积）
- $U$：系统内能（含热能、化学能）
- $TS$：熵对自由能的贡献（$S$ 为熵）
- $pV$：压力-体积功（注：变分时 $\delta(pV) = p\delta V + V\delta p$。在本模型的等压环境中 $\delta p = 0$，而 $\delta V$ 已通过界面位移隐含包含于面积变化项 $\sum \gamma \delta A$ 中——界面运动导致的体积变化与接触面积变化由几何约束关联，因此 $pV$ 项的变分贡献已被界面能变分自然吸收，无需单独列出）

### Lippmann 电场能近似的适用条件

式 (20) 中电场能项 $-\frac{1}{2}C_{\text{eq}} V^2 A_{sw}$ 采用了 **Lippmann 近似的能量形式**，该近似的成立隐含以下假设：

1. **介质电容储能远大于 EDL 电容储能**：$C_{\text{eq}} \ll C_{\text{EDL}}$。在此条件下，施加的电压绝大部分降落在介质层上（$\Delta V_{\text{dielectric}} \approx V_0$），EDL 的充电对总静电能的贡献可忽略。对于典型器件（$C_{\text{eq}} \sim 10^{-4}\,\text{F/m}^2$，$C_{\text{EDL}} \sim 10^{-1}\,\text{F/m}^2$），该假设成立，EDL 储能相对误差 $\sim C_{\text{eq}}/C_{\text{EDL}} < 10^{-3}$。

2. **介质层为理想线性电容**：$C_{\text{eq}}$ 与电压无关（忽略式 (9) 的 Kerr 非线性）。当 Kerr 效应显著时（$E \gtrsim 50\,\text{MV/m}$），电场能泛函需修正为 $-\frac{1}{2} \int_0^V C_{\text{eq}}(V') V' dV'$。

3. **电场能与界面能可分离**：总自由能可写为界面能 + 电场能的加和形式，即忽略电场对界面张力的直接调制（电毛细效应），仅通过 $\Delta p_{\text{elec}}$ 的力学途径耦合。在电润湿体系中，这一假设因界面张力 $\gamma_{ow}$ 主要取决于分子间短程力而非长程静电力而成立。

**当 EDL 效应不可忽略时**（高离子浓度、低电压、极薄介质层 $d_d < 50\,\text{nm}$），总自由能需扩展为包含 EDL 充电贡献的修正形式：

$$G_{\text{total}} = \sum \gamma A - \frac{1}{2} C_{\text{eq}} V^2 A_{sw} - \frac{1}{2} C_{\text{EDL}} \psi_s^2 A_{sw} + U - TS + pV$$

其中 $\psi_s$ 为 Stern 层表面电势，$C_{\text{EDL}}$ 为 EDL 积分电容（含 Stern 层和扩散层串联贡献）。该扩展将自然导出修正的 Young-Lippmann 方程，在低电压区 ($V_0 < 5\,\text{V}$) 产生可测量的接触角饱和偏差。

**稳态判据**：系统达到平衡时，$G_{\text{total}}$ 取极小值，即 $\delta G_{\text{total}} = 0$。该变分条件等价于 Euler-Lagrange 方程组的解，对应稳态油膜形貌。

## 9.2 动态能量弛豫方程

系统向稳态演化的速率与当前自由能偏离平衡值的程度成正比（类梯度流弛豫）：

$$\frac{\partial G}{\partial t} = -\frac{1}{\tau} (G - G_{\min}) \tag{21}$$

其中 $\tau$ 为弛豫时间常数，由系统阻尼（粘性耗散与界面摩擦）决定。该方程作为 PINN 训练的附加约束，确保预测的动态演化路径符合热力学不可逆过程的耗散结构理论。

## 9.3 油相体积守恒约束

油相在像素腔体中不可挥发、不可压缩，其总体积严格守恒：

$$\iiint_{\Omega} \phi(x,y,z,t)\, dV = V_{\text{oil}} = \text{常数} \tag{22}$$

该约束在 PINN 中通过在采样点上施加离散化积分约束实现：随机采样点的 $\phi$ 值均值乘以域体积 $L^2 H$ 应保持不变。

### 变分推导与 Euler-Lagrange 方程

稳态油膜平衡形貌可通过总吉布斯自由能 $G_{\text{total}}$ 在体积守恒约束（式 22）下的约束变分原理严格导出。引入 Lagrange 乘子 $\lambda_p$（物理上等价于系统压力），构建增广泛函：

$$\tilde{G}[\phi, \mathbf{u}, \varphi] = G_{\text{total}} + \lambda_p \left(\iiint_\Omega \phi\, dV - V_{\text{oil}}\right)$$

稳态条件 $\delta \tilde{G} = 0$ 对 $\phi$ 的变分导出一阶必要条件——即广义 Young-Laplace 方程（式 7）的 Euler-Lagrange 形式：

$$\gamma \kappa + \Delta p_{\text{elec}} + \lambda_p = \text{const} \quad (\text{在界面上})$$

这表明 PINN 的界面力学约束 $\mathcal{L}_{\text{Intf}}$（式 23 中对应项）本质上是变分原理的强形式——自由能极小化等价于界面应力平衡。该等价性保证了 PINN 预测的稳态形貌同时满足力学平衡和热力学自洽性，避免了纯力学模型中可能出现的非物理稳态解（如局部极小值但非全局能量最低的 metastable 形貌）。

### 能量耗散率与 Onsager 变分原理

系统向平衡态演化的瞬态路径由 Onsager 变分原理约束：最小化 Rayleigh 耗散泛函（粘性耗散 + 界面摩擦耗散）与自由能变化率之和。即系统演化路径 $\partial_t \mathbf{X}$（$\mathbf{X}$ 为状态变量集合）满足：

$$\min_{\partial_t \mathbf{X}} \left\{ \dot{G}_{\text{total}} + \Phi_{\text{diss}}(\partial_t \mathbf{X}) \right\}$$

其中 $\Phi_{\text{diss}} = \frac{1}{2}\int_\Omega \mu |\nabla\mathbf{u} + \nabla\mathbf{u}^T|^2 dV + \frac{1}{2}\xi \oint_{\Gamma_{\text{cl}}} U_{\text{cl}}^2 dl$ 为总耗散泛函（第一项为体相粘性耗散，第二项为接触线摩擦耗散，$\xi$ 为接触线摩擦系数）。式 (21) 的动态弛豫方程可视为 Onsager 原理在单模态近似下的一维简化——系统以特征弛豫时间 $\tau = \Phi_{\text{diss}} / (\partial^2 G/\partial X^2)$ 指数逼近稳态。

这一变分结构对 PINN 训练具有重要数值意义：当瞬态训练数据稀疏时，Onsager 原理作为物理约束可有效规约网络学习方向，避免产生违反热力学第二定律的非物理演化路径（如自由能不降反升）。

---

# 10 四维 PINN 总损失函数构造

## 10.1 多物理场联合损失

总损失函数为各物理约束残差的加权和：

$$\begin{aligned}
\mathcal{L}_{\text{total}} = &\;\mathcal{L}_{\text{NS}} + \mathcal{L}_{\text{Cont}} + \mathcal{L}_{\text{VOF}} + \mathcal{L}_{\text{Intf}} \\
+ &\;\mathcal{L}_{\text{Elec}} + \mathcal{L}_{\text{EDL}} + \mathcal{L}_{\text{Heat}} + \mathcal{L}_{\text{Surf}} \\
+ &\;\mathcal{L}_{\text{Contact}} + \mathcal{L}_{\text{Gibbs}} + \mathcal{L}_{\text{Vol}} \tag{23}
\end{aligned}$$

| 损失项 | 物理约束 | 残差计算方式 |
|:---:|:---|:---|
| $\mathcal{L}_{\text{NS}}$ | Navier-Stokes 动量方程 | $\|\rho D_t\mathbf{u} + \nabla p - \nabla\cdot(\mu\nabla\mathbf{u}) - \mathbf{F}\|_2^2$ |
| $\mathcal{L}_{\text{Cont}}$ | 流体连续性方程 | $\|\partial_t\rho + \nabla\cdot(\rho\mathbf{u}) + c_{\text{art}}^{-2}D_t p\|_2^2$ |
| $\mathcal{L}_{\text{VOF}}$ | 相场输运方程 | $\|\partial_t\phi + \mathbf{u}\cdot\nabla\phi\|_2^2$ |
| $\mathcal{L}_{\text{Intf}}$ | 界面应力与几何约束 | $\|p_o - p_w - \gamma\kappa - \Delta p_{\text{elec}}\|_2^2 + \|\mathbf{n} - \nabla\phi/|\nabla\phi|\|_2^2$ |
| $\mathcal{L}_{\text{Elec}}$ | 电场与介电方程 | $\|\nabla\cdot((\sigma + \varepsilon\partial_t)\nabla\varphi)\|_2^2 + \|\Delta p_{\text{elec}} - \frac{1}{2}C_{\text{eq}}V^2\|_2^2$ |
| $\mathcal{L}_{\text{EDL}}$ | 双电层 PB 方程 | $\|\nabla^2\psi - \frac{2 e n_0}{\varepsilon}\sinh(e\psi/(k_B T))\|_2^2$ |
| $\mathcal{L}_{\text{Heat}}$ | 热传导方程 | $\|\rho c_p(\mathbf{u}\cdot\nabla T) - k\nabla^2 T\|_2^2$ |
| $\mathcal{L}_{\text{Surf}}$ | 表面活性剂输运 | $\|\partial_t C + \mathbf{u}\cdot\nabla C - D\nabla^2 C\|_2^2$ |
| $\mathcal{L}_{\text{Contact}}$ | 壁面接触角边界约束 | $\|\mathbf{n}\cdot\mathbf{n}_{\text{wall}} - \cos\theta_{A/R}\|_2^2$（在壁面采样点上） |
| $\mathcal{L}_{\text{Gibbs}}$ | 吉布斯自由能极小化 | $\|\partial_t G + \tau^{-1}(G - G_{\min})\|_2^2$ |
| $\mathcal{L}_{\text{Vol}}$ | 油相体积守恒 | $\|\frac{1}{N}\sum_i\phi_i - V_{\text{oil}}/L^2 H\|_2^2$ |

## 10.2 数值稳定附加约束

为保证 PINN 训练的数值稳定性和收敛性，引入以下正则化策略：

1. **界面曲率高斯正则化** — 对曲率场 $\kappa$ 施加空间平滑约束：$\mathcal{L}_{\kappa\text{-reg}} = \lambda_\kappa \|\nabla\kappa\|_2^2$，抑制界面上的虚假高频振荡。

2. **三相接触线奇点局部正则化** — 在接触线邻域内施加局部网格加密并降低物理约束权重，避免应力奇异性污染全局解。具体做法：在距接触线 $< 2\lambda_D$ 的区域内，动量方程残差权重按距离线性衰减。

3. **自适应损失权重调度** — 各物理损失项的权重 $\lambda_i$ 在训练过程中动态调整（详见下表），避免某一项梯度幅值过大而淹没其他物理约束。采用 NeSII（Normalized Squared Inverse Imbalance）自适应调度策略。

| 训练阶段 | 流体权重 | 界面权重 | 电场权重 | 热力学权重 |
|:---|:---:|:---:|:---:|:---:|
| 几何初始化 | 0.1 | 1.0 | 0.5 | 0.05 |
| 流动发展 | 1.0 | 1.0 | 1.0 | 0.3 |
| 全物理耦合 | 1.0 | 1.0 | 1.0 | 1.0 |

以下权重比可作为 NeSII 算法中各损失项初始权重 $\lambda_i(0)$ 的参考比例，训练开始后即由算法自动接管。

4. **梯度裁剪与 NaN 恢复** — 设置全局梯度范数上限 $g_{\max}$，当 $\|\nabla_\theta \mathcal{L}\|_2 > g_{\max}$ 时裁剪梯度。若检测到 NaN 损失值，自动回退到最近稳定 checkpoint 并降低学习率。

5. **硬约束边界编码** — 对 Dirichlet 型边界条件（如壁面无穿透 $\mathbf{u}\cdot\mathbf{n}=0$），采用距离函数映射将边界条件硬编码到网络输出中，替代软约束损失项，提高边界满足精度。

### 自适应权重调度算法详解（NeSII）

表中所列的阶段式权重调度是概念框架，实际训练中采用连续自适应的 **NeSII (Normalized Squared Inverse Imbalance)** 算法。该策略根据各损失项相对于自身历史水平的失衡程度动态分配权重：

$$\lambda_i \propto \frac{1}{(\mathcal{L}_i / \mathcal{L}_i^{\text{ref}})^2 + \epsilon}$$

其中 $\mathcal{L}_i^{\text{ref}}$ 为该损失项的参考值（取过去 $M$ 个 epoch 的移动平均），$\epsilon$ 为小量防止除零。当某项损失相对于其参考值异常增大时，NeSII 适当降低其权重，避免该项主导训练；当某项损失已收敛至接近其参考值时，权重恢复以保持约束。该策略对损失的自然尺度不敏感，在物理约束项众多（本模型 11 项）的场景下比基于梯度范数的 GradNorm 方法更稳定。

### 推荐网络架构

PINN 网络的推荐架构如下（以供参考）：

| 组件 | 规格 |
|:---|:---|
| 输入维度 | 4 $(x,y,z,t)$ |
| 隐藏层数 | 6–8 |
| 每层神经元 | 128–256 |
| 激活函数 | GELU / Swish（避免 ReLU 的二阶导数恒为零导致动量方程残差退化） |
| 傅里叶特征编码 | 对 $(x,y,z)$ 坐标施加 $\gamma(\mathbf{x}) = [\sin(2\pi B \mathbf{x}), \cos(2\pi B \mathbf{x})]$，$B \sim \mathcal{N}(0, \sigma^2)$，$\sigma \sim 1\text{--}10$，缓解谱偏置（spectral bias） |
| 输出分支 | 多任务头：流场分支 (u,v,w,p) + 相场分支 ($\phi$) + 辅助场分支 ($\varphi, T, \psi, C$) |
| 输出变换 | $\phi \to \sigma(\phi)$ (Sigmoid 保证 $\in [0,1]$)；$p \to p - p_{\text{ref}}$ (参考压力锚定) |

### 输出分支间的梯度隔离策略

网络采用"共享骨干 + 任务特定头"的部分共享架构，以平衡物理场间的耦合信息传递与独立表征能力：

| 分支 | 共享层数 | 独立层数 | 设计理由 |
|:---|:---:|:---:|:---|
| 流场分支 $(u,v,w,p)$ | 前 4–5 层 | 后 2–3 层 | 流场变量间存在强耦合（NS 方程内禀关联），使用专用头提取速度-压力相关性 |
| 相场分支 $(\phi)$ | 前 4–5 层 | 后 2–3 层 | $\phi$ 与 $\mathbf{u}$ 通过 VOF 输运方程和 CSF 力耦合，共享骨干保证耦合信息传递，独立头保证界面尖锐特征的专用表征能力 |
| 辅助场分支 $(\varphi, T, \psi, C)$ | 前 5–6 层 | 后 1–2 层 | 辅助物理场通过各自控制方程与主流场耦合，较深的共享层保证物理一致性，较浅的独立头减少参数量和过拟合风险 |

**梯度隔离的实现**：各分支独立头之间不设显式梯度阻断（gradient blocking），即流场损失 $\mathcal{L}_{\text{NS}}$ 的梯度可反向传播至共享骨干的全部参数，但不同任务头之间的梯度仅在共享层交汇。这等价于硬参数共享（hard parameter sharing）多任务学习框架——共享层学习跨物理场的通用时空表示，任务头学习场量特定的解码映射。

### 激活函数的导数连续性保证

本模型中所有控制方程的最高微分阶数分布如下：

| 方程 | 最高导数阶数 | 涉及的激活函数导数 |
|:---|:---:|:---|
| NS 动量方程 (2) | 二阶 ($\nabla^2 \mathbf{u}$) | 激活函数二阶导数 $\sigma''$ |
| VOF 输运 (5) | 一阶 | $\sigma'$ |
| 曲率计算 (6) | 二阶 ($\nabla \cdot \mathbf{n} = \nabla \cdot (\nabla\phi/|\nabla\phi|)$) | $\sigma''$（经 $\nabla\phi$ 的复合链式法则等效为三阶） |
| 电场方程 (13) | 二阶 ($\nabla^2\varphi$) | $\sigma''$ |
| PB 方程 (16) | 二阶 ($\nabla^2\psi$) | $\sigma''$ |

**曲率计算的特殊性**：式 (6) 中 $\kappa = \nabla \cdot (\nabla\phi/|\nabla\phi|)$ 在自动微分的计算图上等效涉及 $\phi$ 的三阶导数（$\nabla\phi \to \mathbf{n} \to \nabla\cdot\mathbf{n}$ 的梯度传播链），因此激活函数需至少具备**连续的三阶导数**以避免曲率场的虚假高频噪声。

GELU 和 Swish 满足此需求：两者均为 $C^\infty$ 函数（无限阶连续可微），任意阶导数值均存在且光滑。相比之下：
- ReLU：二阶导数为零（Dirac delta 脉冲），三阶导数在零点无定义 → **不适用于本模型**
- Tanh：$C^\infty$ 但存在大输入时的梯度饱和问题 → 可作为备选，需配合良好的权重初始化
- Sine（SIREN 激活）：$C^\infty$，在谱偏置缓解方面表现优异 → 可替代傅里叶特征编码，但对初始化极度敏感

### 训练收敛判据

PINN 训练的最终收敛由以下多层次判据联合判定：

1. **总损失平台**：$\mathcal{L}_{\text{total}}$ 在过去 $N_{\text{patience}}$ 个 epoch 内的相对下降 $< 10^{-4}$。
2. **物理残差阈值**：各项物理损失绝对值低于预设阈值（如 $\mathcal{L}_{\text{NS}} < 10^{-3}$，$\mathcal{L}_{\text{Cont}} < 10^{-4}$，$\mathcal{L}_{\text{VOF}} < 10^{-3}$）。
3. **守恒量检验**：独立于损失函数的物理量验证——油相体积守恒误差 $< 1\%$；总吉布斯自由能单调递减（训练过程中不应出现自由能反弹）。
4. **网格独立性**：在不同采样点密度下预测结果的差异 $< 2\%$（以界面位置、接触角等工程量为指标），确保解不依赖于特定采样分布。

### 硬约束边界编码的实现方法

对于壁面无穿透条件 $\mathbf{u} \cdot \mathbf{n}_{\text{wall}} = 0$，网络原始输出 $\hat{\mathbf{u}}$ 经距离函数变换获得物理速度 $\mathbf{u}$：

$$\mathbf{u}(x,y,z) = \hat{\mathbf{u}}(x,y,z) \cdot d_{\text{wall}}(z)$$

其中 $d_{\text{wall}}(z)$ 为到最近壁面的有符号距离函数（signed distance function），在壁面上满足 $d_{\text{wall}} = 0$ 且 $\partial_n d_{\text{wall}} \neq 0$。对于直壁几何，$d_{\text{wall}}(z) = \min(z - z_{\text{bottom}}, H - z)$。类似的硬约束编码可用于温度场的壁面恒温条件和电势的电极 Dirichlet 条件，显著减少了边界采样点的数量和相应的软约束损失权重调优开销。

### CAH 隐式耦合的迭代处理策略

接触角滞后模型（式 14）存在"鸡生蛋"式隐式耦合：接触角决定壁面附近界面法向 $\mathbf{n}$ → 通过 CSF 力影响速度 $\mathbf{u}$ → $\mathbf{u} \cdot \mathbf{n}_{\text{wall}}$ 的符号又反过来决定该处是前进角还是后退角。这种强非线性自指涉耦合是 PINN 训练的已知难点。推荐的解耦策略为 **延迟更新（delayed update）**：每 $N_{\text{freeze}}$ 个 epoch 使用当前网络预测的速度场冻结一次接触角类型（铺展/收缩）的判定，在冻结间隔内保持 CAH 逻辑不变，使损失函数在该窗口内成为速度场的显函数，从而稳定梯度下降。$N_{\text{freeze}}$ 的典型取值为 $50\text{--}200$ epoch，训练后期可逐渐减小以逼近自洽解。对于使用平滑 CAH 过渡函数（$\tanh$ 形式，见 §7.1）的情况，此问题大大缓解——$\tanh$ 的连续可微性使梯度可自然通过接触角判定传播，通常无需显式冻结策略。

---

# 11 模型验证策略与基准测试

模型的正确性和精度需通过多层次验证体系确认，涵盖从单元问题到完整器件仿真的递进验证链路。

## 11.1 验证层次体系

| 验证层级 | 验证对象 | 参考标准 | 验收准则 |
|:---|:---|:---|:---|
| L1: 单元验证 | 各物理方程独立残差 | 解析解/制造解 | 残差 $< 10^{-3}$ |
| L2: 子物理耦合 | NS+VOF 耦合、电场+界面耦合 | CFD 基准 (COMSOL/OpenFOAM) | 界面位置偏差 $< 5\%$ |
| L3: 完整器件 | 电润湿像素完整仿真 | 实验数据 (共聚焦/接触角仪) | 接触角偏差 $< 5^\circ$、响应时间偏差 $< 20\%$ |
| L4: 参数扫描 | 工艺参数敏感性和设计空间 | 实验趋势 + CFD 多点对照 | 趋势一致、最优设计点偏差 $< 10\%$ |

## 11.2 L1 级验证：解析解与制造解

### 稳态 Young-Laplace 液滴验证

在无外电场 ($V=0$)、无流动 ($\mathbf{u}=0$) 的静态极限下，模型应退化到经典 Young-Laplace 液滴平衡。对于置于疏水表面上的轴对称油滴，平衡形貌有解析解：

$$z(r) = z_0 + \frac{2\gamma}{\Delta\rho g} \left[\frac{1}{\sqrt{1 + (dz/dr)^2}} - \frac{\cos\theta}{r/R}\right]$$

在 $Bo \ll 1$ 极限下退化为球冠解 $z(r) = \sqrt{R^2 - r^2} - R\cos\theta$（$R = [3V_{\text{oil}}/(\pi(2-3\cos\theta+\cos^3\theta))]^{1/3}$）。PINN 预测的界面形貌与解析球冠解之间的 RMS 偏差应 $< 2\%$。

### 方法制造解 (Method of Manufactured Solutions, MMS)

对 NS-VOF 耦合系统构造满足连续性的人造解析解（例如 $\mathbf{u}_{\text{MMS}} = (\sin x \cos y \sin t, -\cos x \sin y \sin t, 0)$，$\phi_{\text{MMS}} = 0.5 + 0.5\tanh((z - z_0(x,y,t))/\varepsilon)$），代入控制方程反推所需的源项。将带源项的方程作为 PINN 训练约束，比较 PINN 预测与 MMS 解析解，量化各物理损失项的单独误差贡献。

## 11.3 L2 级验证：CFD 基准对比

选取 3–5 个代表性工况（不同像素尺寸、驱动电压、油量），使用商业 CFD 软件 (COMSOL Multiphysics / ANSYS Fluent) 或开源代码 (OpenFOAM / Basilisk) 进行独立仿真，提取以下对比指标：

| 对比指标 | 提取方法 | 允许偏差 |
|:---|:---|:---:|
| 稳态接触角 $\theta_{\text{steady}}$ | 界面与壁面交点处 $\phi = 0.5$ 轮廓的斜率 | $\Delta\theta < 5^\circ$ |
| 油膜铺展半径 $R_{\text{oil}}$ | $\phi = 0.5$ 等值面在 $z = d_d+d_f$ 平面上的最大径向距离 | $\Delta R / R < 5\%$ |
| 界面形貌 RMS 偏差 | 沿 $z$ 方向逐点比对的 $\phi = 0.5$ 位置差异的 RMS | $< 5\%$ 像素高度 |
| 开关响应时间 $\tau_{90}$ | 铺展半径达到稳态值 $90\%$ 所需时间 | $\Delta\tau / \tau < 20\%$ |
| 速度场峰值 $|\mathbf{u}|_{\max}$ | 流场全域最大速度幅值 | $\Delta u / u < 15\%$ |

对比时应确保 CFD 模型与 PINN 模型使用相同的材料参数、几何尺寸和边界条件设定（包括滑移长度和接触角模型）。CFD 网格需进行独立性验证（至少三级网格加密，结果差异 $< 2\%$）。

## 11.4 L3 级验证：实验数据对比

### 实验测量手段

| 测量技术 | 测量量 | 精度 | 适用阶段 |
|:---|:---|:---:|:---|
| 共聚焦激光扫描显微镜 (CLSM) | 3D 油膜形貌 ($\phi(x,y,z)$) | $z$ 分辨率 $\sim 0.1\,\mu\text{m}$ | 稳态形貌 |
| 接触角测量仪 (侧视) | 表观接触角 $\theta$、铺展半径 | $\pm 1^\circ$ | 稳态 + 动态 (高速相机) |
| 高速相机 (顶视) | 油膜铺展/收缩平面投影面积 | 帧率 $> 1000\,\text{fps}$ | 动态开关 |
| 阻抗分析仪 | 介质电容 $C_{\text{eq}}$、方阻 $R_s$ | $\pm 1\%$ | 器件标定 |
| 白光干涉仪 / AFM | 介电层/疏水层厚度 $d_d, d_f$、粗糙度 $r$ | nm 级 | 工艺参数标定 |

### 对比流程

1. **器件标定**：测量各层厚度、方阻、粗糙度，作为 PINN 模型输入参数（非拟合参数）；
2. **材料物性独立测量**：油/水密度、粘度、界面张力、介电常数、接触角（在无外电场时测量本征接触角 $\theta_0$）；
3. **稳态对比**：在不同 $V_0$ 下测量稳态油膜形貌，与 PINN 预测对比；
4. **瞬态对比**：在电压切换瞬态过程中测量油膜铺展/收缩的时间序列，验证 PINN 的动态预测；
5. **偏差归因分析**：若偏差超过验收准则，逐级排查——物理模型缺失项 → 材料参数不确定性 → 数值收敛性 → 工艺非均匀性。

## 11.5 验证指标体系

| 指标 | 定义 | L2 (vs CFD) | L3 (vs 实验) |
|:---|:---|:---:|:---:|
| 接触角误差 | $\|\theta_{\text{PINN}} - \theta_{\text{ref}}\|$ | $< 3^\circ$ | $< 5^\circ$ |
| 界面位置 RMSD | $\sqrt{\frac{1}{N}\sum (z_{\text{PINN}} - z_{\text{ref}})^2}$ | $< 3\% H$ | $< 5\% H$ |
| 铺展半径误差 | $\|R_{\text{PINN}} - R_{\text{ref}}\| / R_{\text{ref}}$ | $< 3\%$ | $< 5\%$ |
| 响应时间误差 | $\|\tau_{90,\text{PINN}} - \tau_{90,\text{ref}}\| / \tau_{90,\text{ref}}$ | $< 15\%$ | $< 25\%$ |
| 体积守恒误差 | $\|\int\phi\,dV - V_{\text{oil}}\| / V_{\text{oil}}$ | $< 1\%$ | N/A (非直接测量) |
| 自由能单调性 | $G(t)$ 是否单调递减 | 是 | 是 (定性) |

# 12 不确定性量化 (UQ) 框架

输入参数（厚度、方阻、界面张力、接触角等）存在测量不确定性，需量化其对预测结果的影响：

## 12.1 前向 UQ — Monte Carlo 传播

对 $N_{\text{MC}}$ 组随机采样参数（$\sim 10^3\text{--}10^4$），利用已训练 PINN 的快速推理能力批量预测，获得输出量的概率分布（均值、标准差、95% 置信区间）。对于工艺容差分析，重点关注输出分布的尾部分位数（如 $P_{95}$ 的最差铺展均匀性）。

## 12.2 反向 UQ — Bayesian 参数标定

当部分模型参数无法直接测量时（如有效滑移长度 $l_s$、接触线摩擦系数 $\xi$），利用 PINN 作为代理模型，以实验观测数据为似然，通过 MCMC (Markov Chain Monte Carlo) 或变分推断 (Variational Inference) 反推参数的后验分布。

## 12.3 Sobol 全局灵敏度分析

对各输入参数计算一阶和总效应 Sobol 指数：

$$S_i = \frac{\text{Var}_{X_i}[\mathbb{E}(Y|X_i)]}{\text{Var}(Y)}, \qquad S_i^T = 1 - \frac{\text{Var}_{\sim i}[\mathbb{E}(Y|X_{\sim i})]}{\text{Var}(Y)}$$

其中 $Y$ 为关键输出量（如稳态接触角、铺展半径、响应时间）。一阶指数 $S_i$ 量化参数 $X_i$ 单独贡献的方差比例，总效应指数 $S_i^T$ 包含与所有其他参数的交互效应。灵敏度排序结果直接指导工艺开发优先级——高灵敏度参数需严格控制，低灵敏度参数可适当放宽公差。

## 12.4 验证失败判定：异常指纹清单

当预测偏离参考值时，需区分"可接受的数值/测量误差"与"模型进入非物理区间的根本性失效"。以下异常指纹清单用于快速判定后者——出现任一模式，表明模型未收敛到物理可行解：

| 异常指纹 | 判定标准 | 物理上限/下限 | 当出现时指示的根因 |
|:---|:---|:---|:---|
| 接触角超界 | $\theta < 0^\circ$ 或 $\theta > 180^\circ$ | 理论上下限 $0^\circ$（完全润湿），上限 $180^\circ$（完全不润湿） | 接触角损失 $\mathcal{L}_{\text{Contact}}$ 权重不足、壁面 $\phi$ 梯度计算错误、CAH 逻辑在铺展/收缩判定处出错 |
| 相场非物理破碎 | 体相域 ($\phi$ 应接近 0 或 1) 内出现孤立 $\phi = 0.5$ 等高面的碎片 | 物理界面应是单连通拓扑（油-水单一界面） | VOF 输运残差过高、$\mathcal{L}_{\text{VOF}}$ 未被充分约束、界面曲率正则化过弱导致虚假高频振荡 |
| 自由能非单调递增 | $G(t_{n+1}) > G(t_n)$ 在无外电场变化的时段内出现 | 热力学第二定律：孤立系统自由能不可自发增加 | $\mathcal{L}_{\text{Gibbs}}$ 权重不足、动态弛豫时间 $\tau$ 设置过短、Onsager 耗散泛函与实际流场失配 |
| 速度场发散 | $t > t_{\text{switch}} + 2\tau_{\text{flow}}$ 后仍出现 $\|\mathbf{u}\|_{\max} > 5\,U_c$（$\tau_{\text{flow}} = L_c/U_c$ 为流动特征时间，$t_{\text{switch}}$ 为最近一次电压切换时刻） | 电压切换瞬间允许短暂脉冲式速度峰值（可达 $20\,U_c$），但在 $2\tau_{\text{flow}}$ 后应衰减回 $O(U_c)$ 量级 | NS 残差梯度爆炸、CSF 力在薄界面处出现点力奇异性、滑移边界未生效 |
| 相场超出 [0,1] | $\phi < -0.1$ 或 $\phi > 1.1$ | $\phi \in [0,1]$ 为体积分数的物理定义 | 输出 Sigmoid 变换被绕过、网络原始输出范围失控、VOF 输运方程未施加 $\phi$ 有界性约束 |
| 体积守恒漂移 | $\|V_{\text{oil}}(t) - V_{\text{oil}}(0)\| / V_{\text{oil}}(0) > 5\%$ | 油相不可压缩/不可挥发 | $\mathcal{L}_{\text{Vol}}$ 相对于其他损失项的权重衰减过快、采样点数量不足以精确积分 |
| 负压或异常高压 | $p < -p_c$ 或 $p > 10 p_c$（$p_c = \gamma/L_c$ 为毛细压强） | 物理压力范围：毛细压强量级 | 参考压力锚定失效、人工声速 $c_{\text{art}}$ 设置过低、边界条件施加方式错误 |
| 电势阶梯不连续 | 介质层-流体界面处 $\varphi$ 出现大于 $0.1 V_0$ 的跳跃 | 电势连续（Dirichlet 跨界面） | 多层介质界面处的电场方程采样点不足、介电常数梯度 $\nabla\varepsilon_r$ 未在界面处正确参与损失计算 |

**异常指纹的使用方法论**：在验证流程（§11.4 步骤 5 偏差归因分析）中，异常指纹检查应优先于定量误差分析——若任一异常指纹被触发，首先修正模型数值设置（权重、采样、架构）直至指纹消失，再进行定量偏差评估。这避免了在非物理解的基底上追求与实验的"看似一致"（模型的物理错误与参数过度调谐相互抵消的风险）。



# 13 模型局限性与适用条件

明确模型的简化假设、适用范围和已知不足，为模型使用者提供风险意识和改进方向的清晰图谱。

## 13.1 物理假设与适用范围边界

| 假设 | 适用条件 | 失效场景 | 失效后果 |
|:---|:---|:---|:---|
| Sharp-interface VOF ($\phi$ 纯对流无扩散) | 界面厚度 $\ll$ 像素尺寸 ($\varepsilon_{\text{intf}} / L < 10^{-3}$) | $d_o < 1\,\mu\text{m}$（超薄油膜）、近临界点（界面厚度发散） | 界面曲率精度下降、界面张力估算偏差 |
| 弱可压缩近似 | $Ma = U/c_{\text{art}} < 10^{-3}$ | 水锤效应、空化气泡溃灭 | 连续性方程残差异常、压力场虚假振荡 |
| 等温假设 | Joule 热 + 粘性耗散温升 $< 0.1\,\text{K}$ | 高频大功率交流 ($f > 100\,\text{kHz}$)、激光加热 | 忽略热 Marangoni 和温变物性 |
| 稀溶液 PB 方程 | 离子浓度 $< 0.1\,\text{M}$、$\psi < 25\,\text{mV}$ | 高浓度电解质、离子液体、多价离子 | 需改用 Stern 层模型或修正 PB (MPB) |
| 无气相假设 | $V_0$ 低于水解有效电压、介质无针孔 | 介质缺陷、高湿度、交流长期老化 | 气泡破坏界面连续性 |
| 层流假设 | $Re < 1$ (微尺度 Stokes 流区) | 大像素 ($L > 1\,\text{mm}$) + 低粘度油、交流高频惯性效应 | 需引入湍流模型或 DNS |
| Newtonian 流体 | 油和水均为牛顿流体 | 聚合物添加剂油、高浓度表面活性剂溶液 (viscoelastic) | 需本构方程替换 Newtonian 粘性应力 |
| 固壁无渗透 | 介质层/疏水层致密无孔隙 | 多孔介质层、渗透性涂层 | 需 Brinkman/Darcy 耦合模型 |
| 单像素隔离 | 相邻像素间无流体交换 | 像素间极窄间隙、大电压差驱动的流体串扰 | 需扩展为多像素耦合模型 |

### 假设的耦合性与依赖关系

上述 9 项假设并非独立并列，而是存在层级依赖结构。将其区分为"源假设"和"派生约束"有助于明确模型修改时的波及范围：

**源假设（3项）**——定义模型的物理框架基准，修改任一项将触发连锁反应：

| 源假设 | 涉及的核心方程 | 放宽时需联动修改的模块 |
|:---|:---|:---|
| Sharp-interface VOF ($\phi$ 纯对流) | 式 (5)(6)(7) | 界面输运方程、CSF 模型、接触角边界条件、$\mathcal{L}_{\text{VOF}}$ + $\mathcal{L}_{\text{Intf}}$ 损失项 |
| 无气相假设 | 式 (5) 为单相场 | VOF 方程扩展为两分量相场 $(\phi_o, \phi_w, \phi_g)$，同步修改 Gibbs 自由能泛函（式 20 需增加气-液界面能项）、界面应力连续条件（三界面 Young-Laplace）、$\mathcal{L}_{\text{Vol}}$ 需分别约束油相和气相体积 |
| 单像素隔离 | 计算域定义 §2.1 | 几何域扩展为多像素阵列（含像素间连通通道），边界条件需增加像素间界面条件 |

**派生约束（6项）**——由微尺度几何 ($L \sim 100\,\mu\text{m}$) 和典型驱动工况 ($V_0 \sim 15\text{--}30\,\text{V}$, $f \sim 1\,\text{kHz}$) 自然导出，可独立放宽而不影响 VOF 框架：

| 派生约束 | 物理根源 | 放宽时仅需修改的模块 | 对源假设的依赖 |
|:---|:---|:---|:---|
| 层流假设 | $Re \ll 1$（微尺度几何） | 动量方程 (2) 增加惯性项 → 无需额外修改 | 无 |
| 等温假设 | Joule 热 + 粘性耗散 $\ll 1\,\text{K}$ | 热方程 (19) + 温变物性 (3) → 无需修改流场框架 | 无 |
| 弱可压缩近似 | PINN 数值需求（非物理根源） | 连续性方程 (1) 右端项为零 → 需引入压力 Poisson 求解 | 无 |
| 稀溶液 PB 方程 | 低离子浓度工况 | EDL 方程 (16) 替换为 MPB/SCL 模型 → 仅影响 $\mathcal{L}_{\text{EDL}}$ | 无 |
| Newtonian 流体 | 油/水本构特性 | 动量方程 (2) 粘性应力替换为非线性本构（如 power-law）→ 仅影响 $\mathcal{L}_{\text{NS}}$ | 无 |
| 固壁无渗透 | 致密介质层工艺 | 壁面边界条件增加渗透通量 → 影响 $\mathcal{L}_{\text{Contact}}$ | 无（但若渗透导致气相进入，则触发无气相假设的放宽） |

**耦合性关键结论**：
- 引入非牛顿流体本构**无需改变** VOF 框架或界面力学模型；
- 放宽无气相假设（引入三相 VOF）**必须同步修改** Gibbs 自由能泛函（式 20）、界面应力连续条件（三界面 Young-Laplace）、体积守恒约束（式 22 需扩展为双守恒）、以及接触角边界条件（需定义三相接触线的铺展系数 $S = \gamma_{ow} - (\gamma_{og} + \gamma_{wg})$）；
- 源假设中 Sharp-interface VOF 的修改代价最高——若切换为 Diffuse Interface (Cahn-Hilliard) 模型，需重写式 (5) 为 $\partial_t \phi + \mathbf{u}\cdot\nabla\phi = \nabla\cdot(M\nabla\mu)$（其中 $\mu = \delta G/\delta\phi$ 为化学势，$M$ 为迁移率），并在损失函数中增加化学势梯度驱动的扩散项 $\mathcal{L}_{\text{CH}} = \|\partial_t\phi + \mathbf{u}\cdot\nabla\phi - \nabla\cdot(M\nabla\mu)\|_2^2$，同时引入四阶空间导数 $\nabla^4\phi$（经 $\nabla^2\mu$ 链式传播），对 PINN 的自动微分计算图影响最大。

## 13.2 数值方法的已知局限

**高阶导数精度退化**：PINN 通过自动微分计算物理残差中的二阶导数（如 $\nabla^2 \mathbf{u}$、$\nabla \cdot \mathbf{n}$）。随着网络深度增加和导数阶数升高，二阶导数的数值噪声逐渐累积——在界面附近（$\phi$ 梯度极大处）这一问题尤为突出。缓解策略包括：采用 Sobolev 训练（同时拟合函数值和导数值）、引入曲率平滑正则化 $\mathcal{L}_{\kappa\text{-reg}}$（式 23 已包含）、使用改进的 MLP 架构（如 modified MLP with Fourier features）。

**高维采样效率**：4D $(x,y,z,t)$ + 多输出通道导致训练采样点需求随维度指数增长。尽管 LHS + 自适应重采样（§2.4）部分缓解了这一问题，但对于长时间瞬态仿真（$T_{\text{end}} \gg \tau_{\text{flow}}$），时间轴上的采样稀疏可能导致 PINN 在电压切换瞬间的高频行为处欠拟合。因果训练策略（causal PINN，按时间顺序分阶段训练）是一种有效的改进方向。

**多物理场权重调优的脆弱性**：11 项物理损失项的权重平衡（§10.2）目前依赖于 NeSII 自适应算法，缺乏严格的数学最优性保证。在某些参数组合下，自适应权重可能出现振荡（某一项权重周期性地主导-衰减-再主导），延长收敛时间。基于多目标优化 Pareto 前沿的权重选择方法（如 Pareto MTL）是潜在的更优方案。

**无网格与无数据双重的内插外推风险**：PINN 在远离采样点或超出训练参数范围的区域缺乏内禀的误差估计能力。当工艺参数超出手册覆盖范围（例如 $d_d$ 比训练所用范围薄 $30\%$），PINN 可能给出看似平滑但物理错误的预测。应对策略：在训练时添加小幅度参数噪声增强鲁棒性；在推理时通过 ensemble 预测的方差作为 epistemic uncertainty 的代理指标。

## 13.3 未包含的物理效应

以下物理效应在特定条件下可能显著影响器件行为，但当前版本模型中未予考虑：

| 未包含效应 | 可能影响 | 计划引入方式 |
|:---|:---|:---|
| 介电层电荷注入与俘获 (charge trapping) | 长期直流应力下介质层空间电荷积累，改变有效 $V(x,y)$ | 引入电荷注入速率方程 + 空间电荷密度场 $\rho_t(x,y,z,t)$ |
| 油膜蒸发与冷凝 | 开放器件长期运行时油量递减、有效 $V_{\text{oil}}$ 不守恒 | 引入挥发性损失速率项修正 $\mathcal{L}_{\text{Vol}}$ |
| 光致效应 (photovoltaic/photothermal) | 透明显示器件中光照加热和光伏效应改变电场分布 | 在热方程和电场方程中添加光生源项 |
| 机械变形 (flexible substrate) | 柔性基板弯曲导致像素腔体几何变形和膜厚变化 | 将几何参数设为弯曲曲率的函数 |
| 非牛顿流变学 | 长链烷烃在极高剪切率 ($> 10^6\,\text{s}^{-1}$) 下的剪切稀化 | 替换 Newtonian 本构为 power-law 或 Carreau 模型 |
| 三相 VOF (油-水-气) | 气泡或干燥区域的出现 | 扩展 $\phi$ 为两分量相场 $(\phi_o, \phi_w, \phi_g = 1 - \phi_o - \phi_w)$ |

## 13.4 适用场景声明

基于上述假设和局限，本模型在以下场景中具有可靠的预测能力：

- 像素边长 $50\,\mu\text{m} \leq L \leq 500\,\mu\text{m}$，油膜厚度 $5\,\mu\text{m} \leq d_o \leq 50\,\mu\text{m}$
- 直流或低频交流驱动 ($f \leq 10\,\text{kHz}$)，驱动电压 $5\,\text{V} \leq V_0 \leq 40\,\text{V}$
- 水质为去离子水 + 低浓度简单电解质（如 KCl $< 0.01\,\text{M}$），无氧化还原活性物质
- 油相为惰性烷烃或硅油，不含光/热敏成分
- 环境温度 $0^\circ\text{C}\text{--}60^\circ\text{C}$，无冷凝/结冰
- 器件水平放置或 $Bo < 0.1$ 的倾斜角度

超出上述范围的工况，模型可作为定性趋势参考，但定量预测需谨慎使用，并建议进行补充验证实验。

---

# 14 模型完备性覆盖清单

| 类别 | 覆盖内容 | 对应方程 |
|:---|:---|:---:|
| 时空维度 | 四维时空 $(x,y,z,t)$ 全域求解 | §2.1 |
| 流体动力学 | 弱可压缩瞬态 Navier-Stokes 方程 | 式 (1)(2) |
| 界面力学 | VOF 两相界面捕捉 + 法向/切向应力连续 | 式 (5)–(8) |
| 工艺参数 | 介电层/疏水层厚度空间非均匀分布 $d_d(x,y), d_f(x,y)$ | §3 |
| 介电特性 | 介电常数场强非线性 $\varepsilon(E)$ + 介电漏电流 | 式 (9)(13) |
| 电极效应 | 导电层方阻非均匀 $R_s(x,y)$ + 像素边角边缘电场畸变 | 式 (11) |
| 润湿边界 | 接触角滞后（前进角/后退角）+ 疏水壁面滑移 | 式 (4)(14) |
| 表面粗糙度 | Wenzel 等效修正 | 式 (15) |
| 双电层 | 固液双电层 EDL 电势效应 (Poisson-Boltzmann) | 式 (16) |
| 表面活性剂 | 表面活性剂输运 + 界面张力调制 | 式 (17)(18) |
| Marangoni 流 | 热/浓度界面切向流 | 式 (8) |
| 非等温效应 | 流体物性温度依赖 + 对流-导热耦合 | 式 (3)(19) |
| 热力学 | 含熵焓修正吉布斯自由能动态弛豫 | 式 (20)(21) |
| 守恒约束 | 油膜体积守恒 + PINN 数值正则化收敛 | 式 (22)(23) |
| 重力/浮力 | 密度差浮力 + 放置方向依赖性 ($Bo$) | 式 (2a) |
| 电化学 | 水解阈值电压 + 针孔缺陷气泡风险 | §6.6 |
| 介电可靠性 | 介电击穿场强 + 时间相关击穿 (TDDB) | §6.6 |
| 数值方法 | 控制方程无量纲化 + 硬约束编码 + 自适应权重 | §4.5, §10.2 |

### 完备性清单的使用说明

该清单作为模型建设和论文评审的对照工具：每项覆盖内容应在论文正文或补充材料中有对应的方程编号、数值实现描述和验证算例。对于特定应用场景（如仅关心稳态形貌、或忽略热效应），可据此清单明确声明模型简化假设及其适用范围。

例如，若研究仅关注直流稳态油膜形貌，可声明忽略项：§8（除双电层对接触角的静贡献外）、式 (17)–(19)（表面活性剂输运和热传导）、式 (21)（动态弛豫），从而将总损失函数从 11 项简化为 7 项，显著降低训练成本。该清单亦可作为模型版本演进的追踪依据——后续版本的模型更新可对照此清单标注物理覆盖的增删改。

---

# 15 模型核心创新点总结

1. **首次建立电润湿像素 4D-XYZt 时空 PINN 无网格模型**，摆脱传统 CFD 网格对复杂移动界面追踪的束缚，实现单一网络对任意时空点的直接推理。这一突破使得"训练一次、任意推理"的计算范式成为可能——模型训练完成后，任意空间坐标和时刻的物理场均可 $O(1)$ 时间获得，无需重新剖分网格或迭代求解。应用价值包括：实时交互式像素设计、大规模阵列并行仿真（每个像素的几何参数可不同）、以及基于梯度的反问题求解（如根据目标油膜形貌自动优化驱动电压波形）。

2. **完整耦合流体-界面-电场-热场-润湿-热力学多物理场自洽闭环**，包括 11 项物理损失项和 5 项数值正则化约束，覆盖从原子级双电层（$\sim 1\,\text{nm}$）到器件级像素（$\sim 100\,\mu\text{m}$）的跨五个数量级尺度物理。所有物理效应在统一的神经网络框架中以损失函数项的形式并行施加，避免了传统分区耦合策略中界面数据传输的插值误差和迭代收敛问题。自洽性由损失函数中各约束项之间的交叉梯度关系保证——例如，电场损失 $\mathcal{L}_{\text{Elec}}$ 的梯度同时优化电势 $\varphi$ 和速度 $\mathbf{u}$（通过 $\mathbf{F}_{\text{elec}}$ 项），而 NS 损失 $\mathcal{L}_{\text{NS}}$ 的梯度同时优化 $\mathbf{u}$ 和 $\phi$（通过 CSF 项），形成全耦合的参数更新路径。

3. **直接纳入介电/疏水/油/水层厚度、导电层方阻等工艺参数**，建立从结构设计参数到油膜形貌的正向预测映射。这突破了传统仿真模型需要"每个工艺参数组合独立运行一次仿真"的限制——在 PINN 中，工艺参数可作为额外的网络输入维度，单次训练即可覆盖参数空间。具体可实现：
   - 工艺容差 Monte Carlo 分析：在厚度、方阻等参数的统计分布中采样，批量预测油膜形貌分布，评估工艺窗口裕度；
   - 灵敏度排序：通过 PINN 输出对工艺参数的梯度 $\partial\phi/\partial d_d$ 等，定量排序各工艺参数对器件性能的影响大小；
   - 设计优化：耦合梯度优化器，自动搜索使油膜铺展均匀性最优的结构参数组合。

4. **引入接触角滞后、壁面滑移、双电层、Marangoni 流等非理想物理效应**，显著提高模型对真实器件（含缺陷和不均匀性）的预测保真度。这些"非理想"效应在简化模型中常被忽略或使用经验修正系数近似，但它们在以下场景中起决定性作用：
   - 接触角滞后决定像素开关的阈值电压和响应迟滞，忽略 CAH 将导致阈值电压预测偏低 $20\text{--}50\%$；
   - 壁面滑移是三相接触线运动的必要条件，忽略滑移的模型在接触线附近产生非物理的无限大应力；
   - 表面活性剂痕量污染（$\sim \text{ppm}$ 级）即可显著降低油-水界面张力，影响长期运行的稳定性；
   - Marangoni 流在非均匀温度场（如局部光照加热）下可产生与电润湿驱动力量级相当的界面流动。

5. **以吉布斯自由能最小化 + 动态弛豫为核心判据**，可同时预测稳态油膜形貌与动态开关/残影演化，提供统一的能量-动力学双视角。该框架的优势在于：稳态解和瞬态路径受同一热力学原理约束，保证了物理自洽性——网络的稳态预测自动满足自由能极小化，瞬态预测自动沿 Onsager 耗散路径演化。在实际应用中，这一统一框架使得单一训练好的 PINN 模型可同时回答"最终油膜是什么形貌"（稳态）和"需要多长时间达到稳态"（动力学）两个核心工程问题，无需分别建立稳态和瞬态模型。

6. **模型可直接服务于**：器件结构优化（像素尺寸、各层厚度、材料选择的多目标 Pareto 优化）、阈值电压分析（预测不同几何和材料组合下的最小驱动电压）、残影抑制策略设计（模拟油膜在电压撤除后的回缩动力学，优化复位波形）、润湿机理研究（分离电场效应、热效应、表面活性剂效应对接触线运动的各自贡献）、工艺参数敏感性分析（识别关键工艺控制参数，指导工艺开发优先级）及可靠性评估（预测长期循环后的性能退化趋势）。相较于传统实验试错法，基于 PINN 的虚拟仿真可将器件开发周期从数月缩短至数天，并可在实验之前预筛选掉不可行的设计空间区域。

---

*文档版本 V2.5.1 (微调版) | 最后更新 2026-05-14*
