"""验证 EW 源项 v2 的量级"""
from src.config.physics_config import get_materials_params

mp = get_materials_params()

eps0 = 8.854e-12
eps_r = mp.get('relative_permittivity', 3.28)
eps_h = mp.get('epsilon_hydrophobic', 1.934)
d_d = mp.get('dielectric_thickness', 4e-7)
d_h = mp.get('hydrophobic_thickness', 4e-7)
h_ink = mp.get('ink_thickness', 3e-6)
eps_ink = mp.get('epsilon_ink', 4.0)
sigma_ac = mp.get('surface_tension_polar_ink', 0.02505)
eps_ac = mp.get('ac_interface_width', 5e-6)
M_ac = mp.get('ac_mobility', 5e-11)

C_open = 1.0 / (d_d/(eps0*eps_r) + d_h/(eps0*eps_h))
C_ink = 1.0 / (d_d/(eps0*eps_r) + d_h/(eps0*eps_h) + h_ink/(eps0*eps_ink))
delta_C = C_open - C_ink

V_eff = 25.0  # 30V - 5V

# EW 源项（无 z_decay，即 z=0）
ew_at_surface = M_ac * delta_C * V_eff**2 / eps_ac**2

print(f"C_open = {C_open:.4e} F/m²")
print(f"C_ink = {C_ink:.4e} F/m²")
print(f"delta_C = {delta_C:.4e} F/m²")
print(f"M_ac = {M_ac:.4e} m³·s/kg")
print(f"eps_ac = {eps_ac:.4e} m")
print(f"V_eff = {V_eff} V")
print(f"\nEW 源项（z=0）: {ew_at_surface:.4e} [1/s]")

# 典型 advection
print("\n典型 advection: ~0.01~1 [1/s]")
print(f"EW/advection 比值: {ew_at_surface / 0.1:.1f}")

# 检查是否合理
if 0.001 < ew_at_surface < 10:
    print("\n✓ EW 源项量级合理")
elif ew_at_surface > 10:
    print(f"\n⚠️ EW 源项太大 ({ew_at_surface:.4e})，需要减小 M_ac")
    suggested_M = M_ac * 10 / ew_at_surface
    print(f"   建议 M_ac = {suggested_M:.4e}")
else:
    print(f"\n⚠️ EW 源项太小 ({ew_at_surface:.4e})，需要增大 M_ac")
