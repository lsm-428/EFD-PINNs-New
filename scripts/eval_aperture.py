#!/usr/bin/env python3
"""基于 φ=0.5 等值面的开口率评估"""

import json
import sys

import torch

sys.path.insert(0, "/home/scnu/Gitee/EFD3D")
from src.config import PHYSICS
from src.models.aperture_model import EnhancedApertureModel
from src.models.pinn_two_phase import TwoPhasePINN

Lx = PHYSICS["Lx"]
Ly = PHYSICS["Ly"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config/v4.6-optimized.json") as f:
    config = json.load(f)

# 加载模型
model = TwoPhasePINN(config).to(device)
checkpoint = torch.load("outputs/train/pinn_20260614_040430/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

aperture_model = EnhancedApertureModel(config_path="config/device_calibrated_physics.json")

print(f'模型 epoch: {checkpoint["epoch"]}')
print(f'loss: {checkpoint["best_loss"]:.4e}')
print()

t_steady = 0.020
n_grid = 80  # XY 网格分辨率

print("=== 基于 φ=0.5 等值面的开口率评估 ===")
print(f'{"V(V)":>6} | {"η_teacher":>10} | {"η_pinn":>10} | {"η_error":>10}')
print("-" * 50)

for V in [0, 5, 10, 15, 20, 25, 30]:
    _, eta_teacher = aperture_model.theta_eta_from_triad(V, V, t_steady)
    eta_teacher = float(eta_teacher)

    # 在 XY 平面上采样，找 φ=0.5 等值面
    x = torch.linspace(PHYSICS["Lx"] * 0.02, PHYSICS["Lx"] * 0.98, n_grid, device=device)
    y = torch.linspace(PHYSICS["Ly"] * 0.02, PHYSICS["Ly"] * 0.98, n_grid, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # 在多个 z 层采样，找 φ=0.5 的位置
    z_layers = torch.linspace(0, PHYSICS["h_ink"] * 1.5, 20, device=device)

    # 对每个 (x,y)，找 φ=0.5 的 z 位置
    # 如果 φ 从 1（底部油墨）降到 0（顶部水），φ=0.5 的位置就是界面高度
    phi_profile = torch.zeros(n_grid, n_grid, len(z_layers), device=device)

    for k, z_val in enumerate(z_layers):
        pts = torch.stack(
            [
                xx.flatten(),
                yy.flatten(),
                torch.full_like(xx.flatten(), z_val),
                torch.full_like(xx.flatten(), float(V)),
                torch.full_like(xx.flatten(), float(V)),
                torch.full_like(xx.flatten(), t_steady),
            ],
            dim=1,
        )
        with torch.no_grad():
            phi_profile[:, :, k] = model(pts)[:, 4].reshape(n_grid, n_grid)

    # 对每个 (x,y)，用线性插值找 φ=0.5 的 z 位置
    # 从底部（z=0, φ≈1）向上找第一个 φ<0.5 的位置
    interface_z = torch.zeros(n_grid, n_grid, device=device)
    for i in range(n_grid):
        for j in range(n_grid):
            phi_col = phi_profile[i, j, :]
            # 找 φ 从 >0.5 到 <0.5 的过渡
            above = phi_col > 0.5
            if above.all():
                # 全是油墨，界面在顶部以上
                interface_z[i, j] = z_layers[-1]
            elif not above.any():
                # 全是水，界面在底部以下
                interface_z[i, j] = 0.0
            else:
                # 找过渡点
                idx = torch.where(above[:-1] & ~above[1:])[0]
                if len(idx) > 0:
                    k = idx[0].item()
                    # 线性插值
                    phi1, phi2 = phi_col[k].item(), phi_col[k + 1].item()
                    z1, z2 = z_layers[k].item(), z_layers[k + 1].item()
                    interface_z[i, j] = z1 + (0.5 - phi1) / (phi2 - phi1) * (z2 - z1)
                else:
                    interface_z[i, j] = 0.0

    # 开口率：interface_z > 0 的区域中，φ=0.5 轮廓包围的中心区域
    # 更简单：在 z=0 平面，φ<0.5 的区域就是开口
    # 但用户说"找 φ=0.5 等值面"，应该看 XY 平面上 φ=0.5 的轮廓

    # 方法：在 z=0 平面，找 φ=0.5 的轮廓
    # 如果中心 φ<0.5（水），边缘 φ>1（油墨），则开口率 = 中心水区域面积 / 总面积
    pts_z0 = torch.stack(
        [
            xx.flatten(),
            yy.flatten(),
            torch.zeros(n_grid * n_grid, device=device),
            torch.full_like(xx.flatten(), float(V)),
            torch.full_like(xx.flatten(), float(V)),
            torch.full_like(xx.flatten(), t_steady),
        ],
        dim=1,
    )
    with torch.no_grad():
        phi_z0 = model(pts_z0)[:, 4].reshape(n_grid, n_grid)

    # 开口率：φ<0.5 的面积比例（中心是水）
    eta_pinn = float((phi_z0 < 0.5).float().mean())

    print(f"{V:>6.0f} | {eta_teacher:>10.3f} | {eta_pinn:>10.3f} | {abs(eta_pinn-eta_teacher):>10.3f}")

print()
print("=== φ=0.5 等值面可视化 (V=30V) ===")
# 找 φ=0.5 的轮廓
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# z=0 平面 phi 场
im0 = axes[0].imshow(
    phi_z0.cpu().numpy(), origin="lower", extent=[0, Lx * 1e6, 0, Ly * 1e6], cmap="RdYlBu_r", vmin=0, vmax=1
)
axes[0].set_title("z=0 平面 φ 场 (V=30V)")
axes[0].set_xlabel("x (μm)")
axes[0].set_ylabel("y (μm)")
plt.colorbar(im0, ax=axes[0])

# φ=0.5 轮廓
axes[1].contour(
    xx.cpu().numpy() * 1e6, yy.cpu().numpy() * 1e6, phi_z0.cpu().numpy(), levels=[0.5], colors="black", linewidths=2
)
axes[1].set_title("φ=0.5 等值面 (V=30V)")
axes[1].set_xlabel("x (μm)")
axes[1].set_ylabel("y (μm)")
axes[1].set_aspect("equal")
axes[1].set_xlim(0, Lx * 1e6)
axes[1].set_ylim(0, Ly * 1e6)

# 界面高度图
im2 = axes[2].imshow(interface_z.cpu().numpy() * 1e6, origin="lower", extent=[0, Lx * 1e6, 0, Ly * 1e6], cmap="viridis")
axes[2].set_title("界面高度 z_interface (μm)")
axes[2].set_xlabel("x (μm)")
axes[2].set_ylabel("y (μm)")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("outputs/train/pinn_20260614_040430/aperture_eval.png", dpi=150)
print("可视化已保存到 outputs/train/pinn_20260614_040430/aperture_eval.png")
