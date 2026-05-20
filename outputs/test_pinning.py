"""测试1: 纯钉扎 — 接触线是否钉在边缘"""
import torch, numpy as np
from src.physics.constraints import PhysicsConstraints
from src.config import get_materials_params

pc = PhysicsConstraints(materials_params=get_materials_params())
Lx, Ly = 174e-6, 174e-6
theta_wall = 71.0
cos_71 = np.cos(np.radians(71.0))  # 0.326

print("=" * 60)
print("测试1: 纯钉扎 — z=0 边缘, 恒定 V=0")
print("=" * 60)

# 场景: V=0, 油膜平铺, 接触线在像素边缘
# 接触线处的实际 cosθ ≈ 0 (1D解, n̂=(0,0,1))
# target = cos(71°) ≈ 0.326
# Δcos = 0.326 — 小于钉扎阈值 H_pin=0.15? 不, 大于!
# 所以应该突破, 但钉扎力应该仍然很强

for scenario, desc, z_val, v_val in [
    ("边缘钉扎", "z=0, x≈0 边缘, 接触线被约束", 0.0, 0),
    ("边缘上方", "z=3μm, x≈0, 非钉扎区", 3e-6, 0),
    ("底面中心", "z=0, x=Lx/2, 非边缘", 0.0, 0),
]:
    batch = 100
    x = torch.ones(batch) * (1e-6 if "x≈0" in scenario else Lx/2)
    y = torch.rand(batch) * Ly
    z = torch.ones(batch) * z_val
    pts = torch.stack([x,y,z,torch.zeros(batch),torch.ones(batch)*v_val,torch.ones(batch)*0.04],dim=1)
    pts.requires_grad_(True)

    # 1D解: phi = 0.5*(1-tanh((z-3e-6)/1e-6))
    phi = 0.5*(1.0-torch.tanh((pts[:,2]-3e-6)/1e-6))
    pred = torch.zeros(batch, 5)
    pred[:, 4] = phi

    r = pc.compute_sidewall_contact_angle_residual(pts, pred)
    loss = r['sidewall_contact_angle'].mean().item()

    # 计算 cosθ_local 和 Δcos
    grad = torch.autograd.grad(phi.sum(), pts, create_graph=True)[0]
    n_x = grad[:,0] / (torch.norm(grad[:,:3],dim=1)+1e-10)
    local_cos = n_x.mean().item()  # x=0 wall → n̂_wall=(1,0), dot=n_x
    delta = abs(local_cos - cos_71)

    pinned = "✅ 钉扎" if delta < 0.15 else "⚡ 突破"
    print(f"  {scenario}: loss={loss:.4f}, cosθ={local_cos:.3f}, Δcos={delta:.3f} → {pinned}")

print("\n期望: z=0边缘应该被钉扎(高loss), z>0或非边缘正常")
PYEOF
