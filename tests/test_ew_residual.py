"""
验证 EW 残差是否产生非零梯度
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physics.constraints import PhysicsConstraints
from src.config.physics_config import get_materials_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 PhysicsConstraints
params = get_materials_params()
pc = PhysicsConstraints(materials_params=params)

# 创建测试点：底面区域 (z=0)，高电压 (V=30)，油墨状态 (phi~1)
n_points = 1000
x_phys = torch.rand(n_points, 6, device=device, requires_grad=True)
x_phys.data[:, 0] *= 174e-6  # x: [0, Lx]
x_phys.data[:, 1] *= 174e-6  # y: [0, Ly]
x_phys.data[:, 2] = 0.0       # z = 0 (底面)
x_phys.data[:, 3] = 0.0       # V_from = 0
x_phys.data[:, 4] = 30.0      # V_to = 30V
x_phys.data[:, 5] = 0.02      # t_since = 20ms

# 创建模拟预测：phi ~ 1 (油墨覆盖底面)
predictions = torch.zeros(n_points, 5, device=device)
predictions[:, 0] = 0.0   # u
predictions[:, 1] = 0.0   # v
predictions[:, 2] = 0.0   # w
predictions[:, 3] = 0.0   # p
predictions[:, 4] = 0.9   # phi ~ 0.9 (油墨)

# 计算 EW 残差
residuals = pc.compute_electrowetting_residual(x_phys, predictions)
ew_residual = residuals['electrowetting']

print(f"EW residual shape: {ew_residual.shape}")
print(f"EW residual value: {ew_residual.item():.6e}")
print(f"EW residual requires_grad: {ew_residual.requires_grad}")

# 检查梯度
if ew_residual.requires_grad and ew_residual.item() > 0:
    ew_residual.backward()
    if x_phys.grad is not None:
        grad_norm = x_phys.grad.norm().item()
        print(f"Gradient norm w.r.t. x_phys: {grad_norm:.6e}")
        print(f"Gradient w.r.t. V_to (index 4): {x_phys.grad[:, 4].abs().mean().item():.6e}")
        print(f"Gradient w.r.t. z (index 2): {x_phys.grad[:, 2].abs().mean().item():.6e}")
    else:
        print("ERROR: No gradient computed!")
else:
    print("ERROR: EW residual is zero or has no grad!")

# 对比：phi=0 (极性液体覆盖底面)
predictions2 = predictions.clone()
predictions2[:, 4] = 0.1   # phi ~ 0.1 (极性液体)

residuals2 = pc.compute_electrowetting_residual(x_phys, predictions2)
ew_residual2 = residuals2['electrowetting']

print(f"\n对比 (phi=0.1):")
print(f"EW residual value: {ew_residual2.item():.6e}")
print(f"Ratio (phi=0.9 / phi=0.1): {ew_residual.item() / max(ew_residual2.item(), 1e-15):.2f}")

# 检查 p_ew 的量纲和数值
eps0 = 8.854e-12
eps_r = 3.28
eps_h = 1.934
d_d = 4e-7
d_h = 4e-7
C_open = 1.0 / (d_d / (eps0 * eps_r) + d_h / (eps0 * eps_h))
V_eff = 25.0  # 30V - 5V threshold
p_ew = 0.5 * C_open * V_eff**2

print(f"\n=== EW 物理量 ===")
print(f"C_open: {C_open:.4e} F/m²")
print(f"V_eff: {V_eff} V")
print(f"p_ew: {p_ew:.4e} N/m = {p_ew:.4e} J/m²")
print(f"  (对比: sigma = 0.02505 N/m)")

# 检查 NS 内部 EW 力
f_ew_magnitude = 0.5 * C_open * V_eff**2
h_ink = 3e-6
z_decay_at_0 = 1.0
z_decay_at_1um = torch.exp(torch.tensor(-1e-6 / h_ink)).item()
z_decay_at_3um = torch.exp(torch.tensor(-3e-6 / h_ink)).item()

print(f"\n=== NS 内部 EW 力 ===")
print(f"f_ew_magnitude: {f_ew_magnitude:.4e} N/m")
print(f"z_decay at z=0: {z_decay_at_0:.4f}")
print(f"z_decay at z=1μm: {z_decay_at_1um:.4f}")
print(f"z_decay at z=3μm: {z_decay_at_3um:.4f}")

# 检查 AC 迁移率
sigma_ac = 0.02505
eps_ac = 5e-6
M_ac = 1e-10
D_eff = M_ac * sigma_ac / eps_ac
mob = D_eff / eps_ac

print(f"\n=== AC 迁移率 ===")
print(f"M_ac: {M_ac:.4e} m³·s/kg")
print(f"D_eff: {D_eff:.4e} m²/s")
print(f"mob: {mob:.4e} m/s")
print(f"界面迁移时间 τ = eps/mob: {eps_ac/mob*1000:.4f} ms")
