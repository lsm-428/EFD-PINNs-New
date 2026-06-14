#!/usr/bin/env python3
"""油膜行为深度审查 - best_model (epoch 4400)"""

import json
import sys

import torch

sys.path.insert(0, "/home/scnu/Gitee/EFD3D")
from src.config import PHYSICS
from src.models.aperture_model import EnhancedApertureModel
from src.models.pinn_two_phase import TwoPhasePINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config/v4.6-optimized.json") as f:
    config = json.load(f)

model = TwoPhasePINN(config).to(device)
checkpoint = torch.load("outputs/train/pinn_20260614_040430/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

aperture_model = EnhancedApertureModel(config_path="config/device_calibrated_physics.json")

print(f'模型 epoch: {checkpoint["epoch"]}')
print(f'loss: {checkpoint["best_loss"]:.4e}')
print()

t_steady = 0.020

print("--- 稳态评估 (t=20ms, z=0平面) ---")
print(f'{"V(V)":>6} | {"eta_teacher":>10} | {"eta_pinn":>10} | {"eta_error":>10} | {"phi_mid":>8}')
print("-" * 60)

for V in [0, 5, 10, 15, 20, 25, 30]:
    _, eta_teacher = aperture_model.theta_eta_from_triad(V, V, t_steady)
    eta_teacher = float(eta_teacher)

    n_grid = 50
    x = torch.linspace(PHYSICS["Lx"] * 0.05, PHYSICS["Lx"] * 0.95, n_grid, device=device)
    y = torch.linspace(PHYSICS["Ly"] * 0.05, PHYSICS["Ly"] * 0.95, n_grid, device=device)
    z = torch.zeros(n_grid, n_grid, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    pts = torch.stack(
        [
            xx.flatten(),
            yy.flatten(),
            z.flatten(),
            torch.full_like(xx.flatten(), float(V)),
            torch.full_like(xx.flatten(), float(V)),
            torch.full_like(xx.flatten(), t_steady),
        ],
        dim=1,
    )

    with torch.no_grad():
        phi_pred = model(pts)[:, 4].reshape(n_grid, n_grid)

    eta_pinn = float((phi_pred < 0.5).float().mean())
    cx, cy = n_grid // 2, n_grid // 2
    phi_mid = float(phi_pred[cx, cy])

    print(
        f"{V:>6.0f} | {eta_teacher:>10.3f} | {eta_pinn:>10.3f} | {abs(eta_pinn-eta_teacher):>10.3f} | {phi_mid:>8.4f}"
    )

# z 方向分布
print("\n--- z 方向分布 (V=30V, x=y=L/2) ---")
z_layers = [0, 1e-6, 2e-6, 3e-6, 3.5e-6, 5e-6, 10e-6]
cx, cy = PHYSICS["Lx"] / 2, PHYSICS["Ly"] / 2
for z_val in z_layers:
    pt = torch.tensor([[cx, cy, z_val, 30.0, 30.0, t_steady]], device=device)
    with torch.no_grad():
        phi_val = float(model(pt)[0, 4])
    print(f"  z={z_val*1e6:.1f}um: phi={phi_val:.4f}")

# 2D phi 场切片
print("\n--- z=0 平面 phi 场 (V=30V) ---")
n = 10
x = torch.linspace(PHYSICS["Lx"] * 0.1, PHYSICS["Lx"] * 0.9, n, device=device)
y = torch.linspace(PHYSICS["Ly"] * 0.1, PHYSICS["Ly"] * 0.9, n, device=device)
xx, yy = torch.meshgrid(x, y, indexing="ij")
pts = torch.stack(
    [
        xx.flatten(),
        yy.flatten(),
        torch.zeros(n * n, device=device),
        torch.full_like(xx.flatten(), 30.0),
        torch.full_like(xx.flatten(), 30.0),
        torch.full_like(xx.flatten(), t_steady),
    ],
    dim=1,
)
with torch.no_grad():
    phi_grid = model(pts)[:, 4].reshape(n, n).cpu().numpy()

print("  x\\y ", end="")
for j in range(n):
    print(f"{y[j].item()*1e6:>6.0f}", end="")
print()
for i in range(n):
    print(f"{x[i].item()*1e6:>6.0f}", end="")
    for j in range(n):
        print(f"{phi_grid[i,j]:>6.2f}", end="")
    print()
