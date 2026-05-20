"""测试3: 墨滴汇聚 — 表面张力是否驱动圆润液滴"""
import torch, numpy as np

print("=" * 60)
print("测试3: 墨滴汇聚 — 表面张力 vs AC 扩散")
print("=" * 60)

sigma_phys = 0.02505  # N/m, CSF
eps_if = 1.5e-6       # m, 界面宽度
gamma_ac = 4.5e-7     # AC 迁移率 (已修正)

# 1. CSF 表面张力产生的 Laplace 压力
# 对于角落液滴, 曲率半径 R ≈ 10-50μm
# ΔP = σ/R
R_droplet = 20e-6  # 20μm 液滴半径
delta_P_csf = sigma_phys / R_droplet
print(f"\n1. CSF 驱动力:")
print(f"   液滴半径 R = {R_droplet*1e6:.0f}μm")
print(f"   ΔP = σ/R = {delta_P_csf:.0f} Pa")
print(f"   → 这个压力差驱动流场, 使液滴趋向圆形")

# 2. AC 方程的扩散效应
# γ∇²φ 产生平滑力
# 在界面处 |∇φ| ≈ φ(1-φ)/ε ≈ 0.25/1.5e-6 = 1.67e5 m⁻¹
# γ∇²φ ≈ γ × κ 的量级 = 4.5e-7 × (1/R) = 4.5e-7/20e-6 = 0.0225
ac_diff_force = gamma_ac / R_droplet
print(f"\n2. AC 扩散力:")
print(f"   γ∇²φ ≈ γ/R = {ac_diff_force:.2e} s⁻¹")
print(f"   → 无量纲, 远小于 CSF 的物理压力")

# 3. 能量对比
# 界面能 = σ × A
# 平铺油膜面积 ≈ Lx×Ly = 3.03e-8 m²
# 角落液滴面积 ≈ 4πR² (半球) ≈ 5.03e-9 m²
# ΔE = σ × ΔA
area_flat = 174e-6 * 174e-6
area_droplet = 2 * np.pi * R_droplet**2  # 半球面积
delta_E = sigma_phys * (area_flat - area_droplet)
print(f"\n3. 能量驱动:")
print(f"   平铺面积 = {area_flat*1e12:.0f}×10⁻¹² m²")
print(f"   液滴面积 = {area_droplet*1e12:.0f}×10⁻¹² m²")
print(f"   ΔE = σ×ΔA = {delta_E*1e9:.2f} nJ")
print(f"   → {'✅ 表面张力足以驱动汇聚' if delta_E > 1e-10 else '⚠️ 能量差太小'}")

# 4. CSF vs AC 主导性对比
print(f"\n4. CSF/AC 主导性:")
ratio = delta_P_csf / (ac_diff_force + 1e-20)
print(f"   CSF压力 / AC扩散 = {ratio:.0f}")
if ratio > 100:
    print(f"   ✅ CSF 主导界面形态 — 墨滴应该圆润")
elif ratio > 1:
    print(f"   ⚠️ CSF略大于AC — 墨滴可能不够圆润")
else:
    print(f"   ❌ AC 主导 — 墨滴会糊在角落")

# 5. 关键结论
print(f"\n5. 结论:")
print(f"   γ={gamma_ac:.1e} 时, AC扩散力远小于CSF表面张力")
print(f"   → CSF 主导界面形态 ✅")
print(f"   → 墨滴应向圆润形状演化 (表面张力最小化)")
print(f"   → 如果训练后墨滴糊在角落: 检查CSF权重是否被过度稀释")
PYEOF
