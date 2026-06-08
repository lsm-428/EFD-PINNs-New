# EFD3D 全面代码审核报告（第 3 轮）

## 架构概览

44 个 Python 文件，~12,700 行核心代码，8 个子包。
核心数据流：PhysicsConfig -> PHYSICS -> PhysicsConstraints -> Trainer -> TwoPhasePINN
最新 commit：33e6fbf (ruff), b7b3d5c (ac_mobility=1e-10), 6246b5f (test+tau_recovery)

---

## I. 严重问题（CRITICAL）

### C-1: V_T_base 在 JSON 中有两个不同值，HybridPredictor 读到错误的

文件：config/device_calibrated_physics.json
- materials.V_T_base = 3.0（校准后正确值）
- dynamics.V_T_base = 5.0（旧值）

参数流追踪（V_eff = V - V_T）：
  PhysicsConfig.from_json()      -> materials.V_T_base     -> 3.0 (正确)
  PHYSICS['V_T_base']           -> to_dict()              -> 3.0 (正确)
  HybridPredictor._load_config() -> dynamics.V_T_base     -> 5.0 (错误!)
  pinn_two_phase.py             -> PHYSICS['V_THRESHOLD'] -> 5.0 (错误!)
  hybrid_predictor.young_lippmann() -> params['V_THRESHOLD', 5.0] -> 5.0 (错误!)
  constraints.py                -> materials_params['V_T_BASE', 5.0] -> 取决于来源

影响：30V 时 V_eff 差 2V，EW 力差 ~17%。Stage 1 教师和 Stage 2 学生使用不同阈值电压。

修复：HybridPredictor 应读 materials.V_T_base；删除 dynamics.V_T_base；
      PHYSICS['V_THRESHOLD'] 应从 V_T_base 计算而非硬编码。

### C-2: tests/test_ew_residual.py 引用已删除方法

文件：tests/test_ew_residual.py
compute_electrowetting_residual 已从 constraints.py 移除（commit d83c63c）。
测试文件仍调用它（3 处），pytest 会报 AttributeError。

修复：删除或重写为验证 NS 方程 EW 体积力的测试。


---

## II. 重要问题（IMPORTANT）

### I-1: ac_mobility 在 JSON 中缺失

文件：config/device_calibrated_physics.json
PHYSICS 默认已统一为 1e-10（commit b7b3d5c），但 JSON 中无 ac_mobility 字段。

修复：在 JSON materials 中补充 ac_mobility: 1e-10。

### I-2: test_ew_source_v2.py eps_ac 默认值错误

文件：tests/test_ew_source_v2.py L17
  eps_ac = mp.get('ac_interface_width', 5e-6)
默认值应为 5e-7（0.5um），不是 5e-6（5um）。

修复：改为 eps_ac = mp.get('ac_interface_width', 5e-7)。

### I-3: test_pinn_complete.py 安全漏洞

文件：tests/test_pinn_complete.py L58
  torch.load(checkpoint_path, map_location=device, weights_only=False)
evaluate.py 已改为 weights_only=True，但测试文件仍用 weights_only=False。

修复：改为 weights_only=True。

### I-4: evaluate.py 修改全局 PHYSICS

文件：evaluate.py L68-70
直接修改 PHYSICS 全局字典，可能影响后续评估或训练。

修复：用局部变量替代。

### I-5: _get_default_materials_params() 回退值违反 R1 规则

文件：src/physics/constraints.py L52-103
含大量不属于电润湿系统的硬编码材料属性：
- youngs_modulus=210e9, poisson_ratio=0.3
- conductivity=5.5e7
- thermal_conductivity_water=0.6
- specific_heat_water=4186.0
等 30+ 个与电润湿无关的材料参数。

修复：清理回退函数，只保留电润湿系统所需的参数。

### I-6: test_physics_sanity.py 断言可能失败

文件：tests/test_physics_sanity.py L47-48
  assert abs(config.tau - 0.0119) < 1e-6
  assert abs(config.epsilon_r - 12.0) < 1e-6
PHYSICS 默认值分别是 0.005 和 3.28，只有 JSON 覆盖后才正确。

修复：使用 get_materials_params() 获取实际加载后的值。

### I-7: n_vertical_samples 三处不一致

- DataGenerator 代码默认值: 20
- v4.6-optimized.json 配置: 50
- vertical_phi_modeling.md 文档: 5

修复：统一文档和默认值。


### I-8: interface_weight 从 500 降到 50 风险较大

文件：config/v4.6-optimized.json
interface_weight 降 10 倍，同时 continuity/vof/ns 各升 4 倍。
界面可能过度扩散。

修复：密切监控界面清晰度，必要时提高到 100-200。

### I-9: HybridPredictor 中 gamma=0.048 与 JSON gamma=0.015 不一致

文件：src/predictors/hybrid_predictor.py L40
  "gamma": 0.048,  # 极性液体-气表面张力
JSON 中 gamma=0.015（有效宏观表面张力，从 CV 拟合）。
HybridPredictor._load_config() 从 dynamics_params 读取，
dynamics.gamma 不存在，所以始终使用默认值 0.048。

影响：HybridPredictor.young_lippmann() 使用 sigma（界面张力），
不是 gamma。但 surface_tension_recovery() 可能使用 gamma。
需要确认 gamma 在 HybridPredictor 中的实际用途。

修复：在 dynamics_params 中添加 gamma: 0.015。

### I-10: pinn_two_phase.py 中 _analytical_contact_angle 使用二阶阻尼公式

文件：src/models/pinn_two_phase.py L724-738
  omega_d = omega_0 * sqrt(1 - zeta^2)
  damping = zeta / sqrt(1 - zeta^2) if zeta < 1 else 1.0
  theta_t = theta_eq + (theta0 - theta_eq) * exp(-zeta*omega0*t) * (cos(omega_d*t) + damping*sin(omega_d*t))

zeta=1.0 时，sqrt(1-zeta^2)=0，omega_d=0，damping=1.0。
退化为：theta_t = theta_eq + (theta0 - theta_eq) * exp(-t/tau) * (cos(0) + 1.0*sin(0))
         = theta_eq + (theta0 - theta_eq) * exp(-t/tau)

这是一阶指数，但代码形式上是二阶欠阻尼的退化情况。
HybridPredictor.dynamic_response() 使用纯一阶指数。
两者在 zeta=1.0 时数学上等价，但代码形式不一致。

修复：在 _analytical_contact_angle 中添加 zeta=1.0 的特殊情况处理，提高可读性。


---

## III. 改进建议（SUGGESTIONS）

### S1: 补充测试用例
当前改动涉及核心采样策略和物理方程，但缺少测试覆盖。
建议：测试垂直采样分布、阶段转换行为、ew_scale 对残差量级的影响。

### S2: 清理 constraints.py 冗余约束
compute_volume_conservation_residual 返回的 volume_consistency 和 ink_potential_min
在 predictions 只有 5 列时始终返回零。建议清理。

### S3: 添加配置参数范围验证
n_vertical_samples 过大时可能导致 z_points 为空（当 h_ink_edge * 0.5 < 1e-6）。

### S4: 更新文档
vertical_phi_modeling.md 中的 n_vertical_samples=5 与实际代码不符。

### S5: 统一 random.seed 的使用
pinn_two_phase.py 中既用 random.seed()（Python random）又用 np.random.seed()（NumPy），
还手动 import random 在函数内部（L2895, L2958）。建议统一使用 np.random.seed()。

### S6: 考虑添加 __repr__ 到 PhysicsConfig
方便调试时查看当前配置状态。

### S7: flow_solver.py 有 2439 行但处于闲置状态
根据 rules.md R10，闲置模块暂时保留。但建议添加明显的 DEPRECATED 标记。

### S8: 检查 constraints.py 中 safe_compute_gradient 和 safe_compute_laplacian
有多个功能相似的方法：
- safe_compute_gradient (constraints.py L1289)
- _compute_laplacian (constraints.py L693)
- safe_compute_laplacian (constraints.py L1536)
- safe_compute_laplacian_spatial (constraints.py L1273)
建议统一为一个梯度计算工具（可能移到 src/utils/gradients.py）。
