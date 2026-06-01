"""验证 EW 源项的量级和梯度"""
import torch
import sys
sys.path.insert(0, '.')
from src.physics.constraints import PhysicsConstraints
from src.config.physics_config import get_materials_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

params = get_materials_params()
pc = PhysicsConstraints(params)

# 构造测试点：底面中心，高电压
n = 1000
Lx, Ly, Lz = 174e-6, 174e-6, 20e-6
h_ink = 3e-6

# 场景1：底面中心，V=30V（最大驱动力）
x_phys = torch.zeros(n, 6, device=device)
x_phys[:, 0] = Lx / 2  # x 中心
x_phys[:, 1] = Ly / 2  # y 中心
x_phys[:, 2] = torch.linspace(0, h_ink, n, device=device)  # z: 0~3μm
x_phys[:, 3] = 0.0   # V_from
x_phys[:, 4] = 30.0  # V_to（最大电压）
x_phys[:, 5] = 0.02  # t_since（稳态）
x_phys.requires_grad_(True)

# 构造预测值：模拟油墨平铺状态（phi=1 在底部，phi=0 在上部）
predictions = torch.zeros(n, 5, device=device)
predictions[:, 4] = 0.5 * (1 - torch.tanh((x_phys[:, 2] - h_ink) / 1e-6))  # phi: 底部~1, 上部~0

# 计算 EW 源项
eps0 = 8.854e-12
eps_r = params.get('relative_permittivity', 3.28)
eps_h = params.get('epsilon_hydrophobic', 1.934)
d_d = params.get('dielectric_thickness', 4e-7)
d_h = params.get('hydrophobic_thickness', 4e-7)
sigma_ac = params.get('surface_tension_polar_ink', 0.02505)
eps_ac = params.get('ac_interface_width', 5e-6)
M_ac = params.get('ac_mobility', 5e-11)

C_open = 1.0 / (d_d/(eps0*eps_r) + d_h/(eps0*eps_h))
V_eff = 30.0 - 5.0  # 25V
p_ew = 0.5 * C_open * V_eff**2

z = x_phys[:, 2]
z_decay = torch.exp(-z / h_ink)
phi = predictions[:, 4]

D_eff = M_ac * sigma_ac / eps_ac
mob = D_eff / eps_ac

ew_source = (p_ew * z_decay * phi / (sigma_ac * eps_ac)) * mob

print(f"\n=== EW 源项量级分析 ===")
print(f"C_open = {C_open:.4e} F/m²")
print(f"p_ew (V=30) = {p_ew:.4e} N/m")
print(f"D_eff = {D_eff:.4e} m²/s")
print(f"mob = {mob:.4e} m/s")
print(f"sigma*eps = {sigma_ac * eps_ac:.4e} N")
print(f"\nEW 源项统计:")
print(f"  max = {ew_source.max().item():.4e} [1/s]")
print(f"  min = {ew_source.min().item():.4e} [1/s]")
print(f"  mean = {ew_source.mean().item():.4e} [1/s]")
print(f"  std = {ew_source.std().item():.4e} [1/s]")

# 对比 advection 量级
print(f"\n=== 对比 ===")
print(f"EW 源项 mean: {ew_source.mean().item():.4e} [1/s]")
print(f"典型 advection 量级: ~1e-2 ~ 1e0 [1/s]")
print(f"EW/advection 比值: {ew_source.mean().item() / 0.1:.1f}")

# 检查梯度
print(f"\n=== 梯度检查 ===")
x_phys2 = x_phys.clone().detach().requires_grad_(True)
pred2 = predictions.clone().detach()

# 通过 AC 残差计算
from src.physics.constraints import PhysicsConstraints
pc2 = PhysicsConstraints(params)

# 手动计算 AC 残差
phi = pred2[:, 4]
u = v = w = torch.zeros(n, device=device)
p = torch.zeros(n, device=device)

# 计算梯度
g_phi = torch.autograd.grad(phi.sum(), x_phys2, create_graph=True, retain_graph=True)[0]
phi_x, phi_y, phi_z = g_phi[:, 0], g_phi[:, 1], g_phi[:, 2]
phi_t = torch.zeros(n, device=device)  # 稳态

advection = phi_t + u * phi_x + v * phi_y + w * phi_z

# 计算 lap_phi
lap_phi = torch.zeros(n, device=device)
for i in range(3):
    g2 = torch.autograd.grad(g_phi[:, i].sum(), x_phys2, create_graph=True, retain_graph=True)[0]
    lap_phi += g2[:, i]

f_prime = 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

ac_residual = advection - mob * (eps_ac * lap_phi - sigma_ac * f_prime / eps_ac)

# 加入 EW 源项
z_coord = x_phys2[:, 2]
z_decay2 = torch.exp(-z_coord / h_ink)
ew_source2 = (p_ew * z_decay2 * phi / (sigma_ac * eps_ac)) * mob
ac_residual_with_ew = ac_residual - ew_source2

print(f"AC 残差 (无 EW): mean={ac_residual.mean().item():.4e}, std={ac_residual.std().item():.4e}")
print(f"AC 残差 (有 EW): mean={ac_residual_with_ew.mean().item():.4e}, std={ac_residual_with_ew.std().item():.4e}")
print(f"EW 源项:         mean={ew_source2.mean().item():.4e}, std={ew_source2.std().item():.4e}")

# 检查 EW 源项对 V_to 的梯度
x_phys3 = x_phys.clone().detach().requires_grad_(True)
V_to = x_phys3[:, 4]
V_eff3 = torch.clamp(V_to - 5.0, min=0.0)
p_ew3 = 0.5 * C_open * V_eff3**2
phi3 = pred2[:, 4].clone()
z_decay3 = torch.exp(-x_phys3[:, 2] / h_ink)
ew3 = (p_ew3 * z_decay3 * phi3 / (sigma_ac * eps_ac)) * mob

grad_V = torch.autograd.grad(ew3.sum(), x_phys3, create_graph=False)[0]
print(f"\nEW 源项对 V_to 的梯度: {grad_V[:, 4].mean().item():.4e}")
print(f"EW 源项对 z 的梯度: {grad_V[:, 2].mean().item():.4e}")
