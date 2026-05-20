#!/usr/bin/env python3
"""Final verification of ablation study metrics. Run once, print everything."""
import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.models.pinn_two_phase import TwoPhasePINN  # noqa: E402
from src.models.aperture_model import EnhancedApertureModel  # noqa: E402
from src.config import PHYSICS  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANALYTICAL_CONFIG = "/home/scnu/Gitee/EFD3D/config/v4.5-standard.json"

RUNS = [
    ("Full Model",       "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260317_235529"),
    ("no_continuity",    "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260319_120821"),
    ("no_vof",           "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260320_040329"),
    ("no_interface",     "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260320_202048"),
    ("single_stage",     "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260321_124610"),
    ("smaller_network",  "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260322_043947"),
]

stage1 = EnhancedApertureModel(config_path=ANALYTICAL_CONFIG)
_, eta_ref = stage1.theta_eta_from_triad(30.0, 30.0, 0.040)
print(f"Analytical η(30V) = {eta_ref:.4f}\n")

for name, run_dir in RUNS:
    ckpt = os.path.join(run_dir, "best_model.pth")
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    eval_physics = dict(PHYSICS)
    if "physics" in config:
        eval_physics.update(config["physics"])
    if "model" not in config:
        config["model"] = {}
    model = TwoPhasePINN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    Lx, Ly, Lz = eval_physics["Lx"], eval_physics["Ly"], eval_physics["Lz"]
    h_ink = eval_physics["h_ink"]
    v0 = Lx * Ly * h_ink
    v_domain = Lx * Ly * Lz

    # --- Volume error (same method as training) ---
    n_vol = 30000
    torch.manual_seed(42); np.random.seed(42)
    x = torch.rand(n_vol, device=device) * Lx
    y = torch.rand(n_vol, device=device) * Ly
    z = torch.rand(n_vol, device=device) * Lz
    xyz = torch.stack([x, y, z], dim=1)
    t_steady = np.random.uniform(0.020, 0.050, 3).tolist()
    t_on = np.random.uniform(0.002, 0.025, 5).tolist()
    t_off = np.random.uniform(0.002, 0.025, 4).tolist()
    tests = []
    for v in [0.0, 10.0, 20.0, 25.0, 30.0]:
        for t in t_steady:
            tests.append((float(v), float(v), float(t)))
    for t in t_on:
        tests.append((0.0, 30.0, float(t))); tests.append((0.0, 25.0, float(t)))
    for t in t_off:
        tests.append((30.0, 0.0, float(t))); tests.append((25.0, 0.0, float(t)))
    vol_errors = []
    for vf, vt, ts in tests:
        pts = torch.cat([xyz,
            torch.full((n_vol, 1), float(vf), device=device),
            torch.full((n_vol, 1), float(vt), device=device),
            torch.full((n_vol, 1), float(ts), device=device)], dim=1)
        with torch.no_grad():
            phi = model(pts)[:, 4]
            phi = torch.clamp(phi, 0.0, 1.0)
        v_curr = v_domain * phi.mean()
        rel_err = (v_curr - v0) / (v0 + 1e-12)
        vol_errors.append(abs(rel_err.item()) * 100)
    vol_mean = np.mean(vol_errors)

    # --- Aperture at z=0 (30V steady) ---
    n_ap = 96
    xa = np.linspace(0, Lx, n_ap); ya = np.linspace(0, Ly, n_ap)
    XA, YA = np.meshgrid(xa, ya)
    inp_ap = np.zeros((n_ap*n_ap, 6), dtype=np.float32)
    inp_ap[:,0]=XA.ravel(); inp_ap[:,1]=YA.ravel(); inp_ap[:,2]=0.0
    inp_ap[:,3]=30.0; inp_ap[:,4]=30.0; inp_ap[:,5]=0.040
    with torch.no_grad():
        phi_ap = model(torch.tensor(inp_ap, device=device))[:,4].cpu().numpy().reshape(n_ap, n_ap)
    m = 0.5*(1.0+np.tanh((0.3-phi_ap)/0.05))
    eta_pinn = float(np.mean(m))
    ap_err = abs(eta_pinn - eta_ref) * 100

    # --- Interface at z=h_ink/2 (30V steady, 128x128 grid) ---
    n_if = 128
    xi = np.linspace(0, Lx, n_if); yi = np.linspace(0, Ly, n_if)
    XI, YI = np.meshgrid(xi, yi)
    inp_if = np.zeros((n_if*n_if, 6), dtype=np.float32)
    inp_if[:,0]=XI.ravel(); inp_if[:,1]=YI.ravel(); inp_if[:,2]=h_ink/2
    inp_if[:,3]=30.0; inp_if[:,4]=30.0; inp_if[:,5]=0.040
    with torch.no_grad():
        phi_if = model(torch.tensor(inp_if, device=device))[:,4].cpu().numpy().reshape(n_if, n_if)
    phi_range = float(phi_if.max() - phi_if.min())
    sharp_residual = float(np.mean(phi_if * (1.0 - phi_if)))

    # Interface gradient (only in transition zone)
    gy, gx = np.gradient(phi_if, yi[1]-yi[0], xi[1]-xi[0])
    grad_mag = np.sqrt(gx**2 + gy**2)
    mask = (phi_if > 0.1) & (phi_if < 0.9)
    if_grad = float(np.mean(grad_mag[mask])*1e-6) if mask.sum() > 10 else 0.0
    if_max_grad = float(np.max(grad_mag[mask])*1e-6) if mask.sum() > 10 else 0.0

    # Radial transition width (8 angles)
    cx, cy = Lx/2, Ly/2
    r_max = min(Lx, Ly)/2*0.9
    n_r = 200; r = np.linspace(0, r_max, n_r)
    widths = []
    for angle in np.linspace(0, 2*np.pi, 9)[:8]:
        xv = np.clip(cx+r*np.cos(angle), 0, Lx)
        yv = np.clip(cy+r*np.sin(angle), 0, Ly)
        inp_r = np.zeros((n_r, 6), dtype=np.float32)
        inp_r[:,0]=xv; inp_r[:,1]=yv; inp_r[:,2]=h_ink/2
        inp_r[:,3]=30.0; inp_r[:,4]=30.0; inp_r[:,5]=0.040
        with torch.no_grad():
            phi_r = model(torch.tensor(inp_r, device=device))[:,4].cpu().numpy()
        lo=np.where(phi_r>0.1)[0]; hi=np.where(phi_r<0.9)[0]
        if len(lo)>0 and len(hi)>0 and r[hi[-1]]>r[lo[0]]:
            widths.append((r[hi[-1]]-r[lo[0]])*1e6)
    avg_width = np.mean(widths) if widths else float('nan')

    # --- Quality classification ---
    if phi_range < 0.3:
        if_quality = "Degraded"
    elif sharp_residual < 0.005 and (np.isnan(avg_width) or avg_width < 5.0):
        if_quality = "Sharp"
    elif sharp_residual < 0.06 and if_grad > 0.01:
        if_quality = "Moderate"
    else:
        if_quality = "Diffuse"

    # --- Physical validity ---
    if vol_mean < 1.0 and if_quality == "Sharp" and ap_err < 20:
        valid = "Yes"
    elif if_quality == "Degraded":
        valid = "No"
    elif if_quality == "Diffuse":
        valid = "No"
    else:
        valid = "Partial"

    # --- Best loss from log ---
    best_loss = None
    with open(os.path.join(run_dir, "training.log")) as f:
        for line in f:
            if "训练完成!" in line:
                m = re.search(r'最佳损失:\s*([\d.e+\-]+)', line)
                if m: best_loss = float(m.group(1))

    w_str = f"{avg_width:.1f}" if not np.isnan(avg_width) else "N/A"

    print(f"{'='*70}")
    print(f"  {name}")
    print(f"  Best Loss: {best_loss:.1f}")
    print(f"  Volume Error: {vol_mean:.2f}%")
    print(f"  Aperture 30V: η={eta_pinn:.4f} (ref={eta_ref:.4f}), Error={ap_err:.2f}pp")
    print(f"  Interface: φ_range={phi_range:.4f}, φ(1-φ)={sharp_residual:.5f}")
    print(f"             |∇φ|_if={if_grad:.4f}/μm, max|∇φ|={if_max_grad:.2f}/μm, width={w_str}μm")
    print(f"  Interface Quality: {if_quality}")
    print(f"  Physically Valid: {valid}")

# Final summary table
print(f"\n\n{'='*100}")
print(f"{'Variant':<18} {'Loss':>7} {'Vol.Err':>8} {'η_PINN':>8} {'η_Ref':>8} {'Ap.Err':>8} {'If.Qual':>10} {'Valid':>8}")
print(f"{'-'*100}")
