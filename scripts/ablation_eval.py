#!/usr/bin/env python3
"""Compute ablation study metrics: volume error, aperture, interface quality, dynamic response."""
import os
import sys
import re
import json
import torch
import numpy as np
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.models.pinn_two_phase import TwoPhasePINN  # noqa: E402
from src.models.aperture_model import EnhancedApertureModel  # noqa: E402
from src.config import PHYSICS  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ANALYTICAL_CONFIG = "/home/scnu/Gitee/EFD3D/config/v4.5-standard.json"

RUNS = {
    "Full Model":       "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260317_235529",
    "no_continuity":    "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260319_120821",
    "no_vof":           "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260320_040329",
    "no_interface":     "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260320_202048",
    "single_stage":     "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260321_124610",
    "smaller_network":  "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260322_043947",
}


def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    eval_physics = dict(PHYSICS)
    if "physics" in config:
        eval_physics.update(config["physics"])
    if "model" not in config:
        config["model"] = {}
    model = TwoPhasePINN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, eval_physics


def compute_volume_error(model, eval_physics):
    Lx, Ly, Lz = eval_physics["Lx"], eval_physics["Ly"], eval_physics["Lz"]
    h_ink = eval_physics["h_ink"]
    v0 = Lx * Ly * h_ink
    v_domain = Lx * Ly * Lz
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
        tests.append((0.0, 30.0, float(t)))
        tests.append((0.0, 25.0, float(t)))
    for t in t_off:
        tests.append((30.0, 0.0, float(t)))
        tests.append((25.0, 0.0, float(t)))
    errors = []
    for v_from, v_to, t_since in tests:
        pts = torch.cat([
            xyz,
            torch.full((n_vol, 1), float(v_from), device=device),
            torch.full((n_vol, 1), float(v_to), device=device),
            torch.full((n_vol, 1), float(t_since), device=device),
        ], dim=1)
        with torch.no_grad():
            phi = model(pts)[:, 4]
            phi = torch.clamp(phi, 0.0, 1.0)
        v_curr = v_domain * phi.mean()
        rel_error = (v_curr - v0) / (v0 + 1e-12)
        errors.append(abs(rel_error.item()) * 100)
    return float(np.mean(errors)), float(np.max(errors))


def compute_aperture_at(model, eval_physics, V_from, V_to, t_since):
    """Compute aperture ratio at given condition."""
    Lx, Ly = eval_physics["Lx"], eval_physics["Ly"]
    n = 96; phi0, eps = 0.3, 0.05
    x = np.linspace(0, Lx, n); y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y)
    inputs = np.zeros((n * n, 6), dtype=np.float32)
    inputs[:, 0] = X.ravel(); inputs[:, 1] = Y.ravel(); inputs[:, 2] = 0.0
    inputs[:, 3] = V_from; inputs[:, 4] = V_to; inputs[:, 5] = t_since
    with torch.no_grad():
        out = model(torch.tensor(inputs, device=device))
        phi = out[:, 4].cpu().numpy().reshape(n, n)
    m = 0.5 * (1.0 + np.tanh((phi0 - phi) / eps))
    return float(np.mean(m))


def compute_dynamic_response_rmse(model, eval_physics, stage1_model):
    """Compute RMSE of aperture ratio vs analytical during ON and OFF transients.

    ON: 0→30V, times from 0 to 25ms
    OFF: 30V→0V, times from 0 to 25ms
    """
    times = np.linspace(0.001, 0.025, 20)
    on_errors, off_errors = [], []

    # ON transient: 0→30V
    for t in times:
        eta_pinn = compute_aperture_at(model, eval_physics, 0.0, 30.0, t)
        _, eta_ref = stage1_model.theta_eta_from_triad(0.0, 30.0, t)
        on_errors.append((eta_pinn - float(eta_ref)) ** 2)

    # OFF transient: 30V→0V
    for t in times:
        eta_pinn = compute_aperture_at(model, eval_physics, 30.0, 0.0, t)
        _, eta_ref = stage1_model.theta_eta_from_triad(30.0, 0.0, t)
        off_errors.append((eta_pinn - float(eta_ref)) ** 2)

    rmse_on = float(np.sqrt(np.mean(on_errors)))
    rmse_off = float(np.sqrt(np.mean(off_errors)))
    rmse_total = float(np.sqrt(np.mean(on_errors + off_errors)))
    return rmse_on, rmse_off, rmse_total


def compute_interface_metrics(model, eval_physics):
    """Interface quality at steady state (30V, t=0.040s) on xy-plane at z=h_ink/2."""
    Lx, Ly = eval_physics["Lx"], eval_physics["Ly"]
    h_ink = eval_physics["h_ink"]
    z = h_ink / 2
    n = 128
    x = np.linspace(0, Lx, n); y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y); dx, dy = x[1] - x[0], y[1] - y[0]

    inputs = np.zeros((n * n, 6), dtype=np.float32)
    inputs[:, 0] = X.ravel(); inputs[:, 1] = Y.ravel(); inputs[:, 2] = z
    inputs[:, 3] = 30.0; inputs[:, 4] = 30.0; inputs[:, 5] = 0.040

    with torch.no_grad():
        out = model(torch.tensor(inputs, device=device))
        phi = out[:, 4].cpu().numpy().reshape(n, n)

    phi_range = float(phi.max() - phi.min())
    gy, gx = np.gradient(phi, dy, dx)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Interface region only
    mask = (phi > 0.1) & (phi < 0.9)
    if_grad = float(np.mean(grad_mag[mask])) if mask.sum() > 0 else 0.0
    if_max_grad = float(np.max(grad_mag[mask])) if mask.sum() > 0 else 0.0

    # Sharpening residual
    sharp_res = float(np.mean(phi * (1.0 - phi)))

    # Diagonal scans: corner → center (crosses water→oil interface)
    # At 30V, pixel center is water (φ≈0), corners are oil (φ≈1).
    # Scanning outward from center misses the interface; corner→center captures it.
    cx, cy = Lx / 2, Ly / 2
    corners = [(0.0, 0.0), (Lx, 0.0), (0.0, Ly), (Lx, Ly)]
    n_r = 200
    widths = []
    for cx_c, cy_c in corners:
        dist = np.sqrt((cx - cx_c) ** 2 + (cy - cy_c) ** 2)
        r_max_c = dist * 0.95
        r = np.linspace(0, r_max_c, n_r)
        dx_c = (cx - cx_c) / dist
        dy_c = (cy - cy_c) / dist
        xv = np.clip(cx_c + r * dx_c, 0, Lx)
        yv = np.clip(cy_c + r * dy_c, 0, Ly)
        inp = np.zeros((n_r, 6), dtype=np.float32)
        inp[:, 0] = xv; inp[:, 1] = yv; inp[:, 2] = z
        inp[:, 3] = 30.0; inp[:, 4] = 30.0; inp[:, 5] = 0.040
        with torch.no_grad():
            phi_r = model(torch.tensor(inp, device=device))[:, 4].cpu().numpy()
        # Corner is oil (φ≈1), center is water (φ≈0) → φ decreases along r
        oil_region = np.where(phi_r > 0.9)[0]    # oil near corner
        water_region = np.where(phi_r < 0.1)[0]  # water near center
        if len(oil_region) > 0 and len(water_region) > 0:
            oil_end = r[oil_region[-1]]   # last oil point
            water_start = r[water_region[0]]  # first water point
            if water_start > oil_end:
                widths.append((water_start - oil_end) * 1e6)

    avg_width = float(np.mean(widths)) if widths else float('nan')

    # Quality classification based on sharpening residual and transition width.
    # φ(1-φ) ≈ 0.25 * (interface_area / total_area); lower = sharper interface.
    # Thresholds calibrated against expected PINN interface widths in this domain.
    if phi_range < 0.3:
        quality = "Degraded"
    elif sharp_res < 0.008 and (np.isnan(avg_width) or avg_width < 5.0):
        quality = "Sharp"
    elif sharp_res < 0.05 and phi_range > 0.8:
        quality = "Moderate"
    else:
        quality = "Diffuse"

    return {
        "phi_range": phi_range,
        "if_grad_per_um": round(if_grad * 1e-6, 4),
        "if_max_grad_per_um": round(if_max_grad * 1e-6, 2),
        "sharpening_residual": round(sharp_res, 5),
        "transition_width_um": avg_width,
        "quality": quality,
    }


def get_best_loss(log_path):
    if not os.path.exists(log_path): return None
    with open(log_path) as f:
        for line in f:
            if "训练完成!" in line:
                m = re.search(r'最佳损失:\s*([\d.e+\-]+)', line)
                if m: return float(m.group(1))
    return None


def main():
    stage1 = EnhancedApertureModel(config_path=ANALYTICAL_CONFIG)
    _, analytical_ref = stage1.theta_eta_from_triad(30.0, 30.0, 0.040)
    print(f"Analytical ref at 30V: eta = {analytical_ref:.4f}")
    print("(reference from EnhancedApertureModel, aperture error < 10pp indicates good agreement)")

    results = {}
    for name, run_dir in RUNS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        ckpt = os.path.join(run_dir, "best_model.pth")
        if not os.path.exists(ckpt):
            print("  SKIP: no best_model.pth"); continue

        model, eval_physics = load_model(ckpt)

        # 1. Volume error
        vol_mean, vol_max = compute_volume_error(model, eval_physics)
        print(f"  Vol.Err: mean={vol_mean:.2f}%, max={vol_max:.2f}%")

        # 2. Steady-state aperture at 30V
        eta_ss = compute_aperture_at(model, eval_physics, 30.0, 30.0, 0.040)
        ap_err = abs(eta_ss - analytical_ref) * 100
        print(f"  Aperture 30V: PINN={eta_ss:.4f}, Ref={analytical_ref:.4f}, Err={ap_err:.2f}pp")

        # 3. Dynamic response RMSE (key for VOF assessment)
        rmse_on, rmse_off, rmse_total = compute_dynamic_response_rmse(
            model, eval_physics, stage1
        )
        print(f"  Dynamic RMSE: ON={rmse_on:.4f}, OFF={rmse_off:.4f}, Total={rmse_total:.4f}")

        # 4. Interface quality
        iface = compute_interface_metrics(model, eval_physics)
        w = f"{iface['transition_width_um']:.1f}" if not np.isnan(iface['transition_width_um']) else "N/A"
        print(f"  Interface: φ_range={iface['phi_range']:.4f}, width={w}um, "
              f"|∇φ|_if={iface['if_grad_per_um']:.4f}/um, φ(1-φ)={iface['sharpening_residual']:.5f}, "
              f"quality={iface['quality']}")

        best_loss = get_best_loss(os.path.join(run_dir, "training.log"))
        print(f"  Best loss: {best_loss}")

        # Physical validity: engineering-useful model
        # Requires: volume error < 1%, binary interface (Moderate+), aperture < 20pp off
        if vol_mean < 1.0 and iface["quality"] in ("Sharp", "Moderate") and ap_err < 20:
            valid = "Yes"
        elif iface["quality"] == "Degraded" or vol_mean > 5.0:
            valid = "No"
        else:
            valid = "Partial"

        results[name] = {
            "best_loss": best_loss,
            "vol_error_mean_pct": round(vol_mean, 2),
            "aperture_30v_pinn": round(eta_ss, 4),
            "aperture_30v_ref": round(analytical_ref, 4),
            "aperture_error_pp": round(ap_err, 2),
            "dynamic_rmse_on": round(rmse_on, 4),
            "dynamic_rmse_off": round(rmse_off, 4),
            "dynamic_rmse_total": round(rmse_total, 4),
            "interface_quality": iface["quality"],
            "transition_width_um": round(iface["transition_width_um"], 1) if not np.isnan(iface["transition_width_um"]) else None,
            "sharpening_residual": iface["sharpening_residual"],
            "physically_valid": valid,
        }

    # Final table
    print("\n\n" + "="*140)
    print("ABLATION STUDY — COMPLETE RESULTS")
    print("="*140)
    hdr = (f"{'Variant':<18} {'Loss':>8} {'Vol.Err':>8} {'Ap.Err%':>8} "
           f"{'RMSE_ON':>9} {'RMSE_Tot':>9} {'If.Qual':>10} {'W(um)':>8} {'φ(1-φ)':>9} {'Valid':>8}")
    print(hdr)
    print("-"*140)
    order = ["Full Model", "no_continuity", "no_vof", "no_interface", "single_stage", "smaller_network"]
    for name in order:
        r = results.get(name)
        if r is None: continue
        w = f"{r['transition_width_um']:.1f}" if r['transition_width_um'] is not None else "N/A"
        print(f"{name:<18} {r['best_loss']:>8.1f} {r['vol_error_mean_pct']:>7.2f}% {r['aperture_error_pp']:>7.2f}% "
              f"{r['dynamic_rmse_on']:>9.4f} {r['dynamic_rmse_total']:>9.4f} "
              f"{r['interface_quality']:>10} {w:>8} {r['sharpening_residual']:>9.5f} {r['physically_valid']:>8}")

    out_path = "/home/scnu/Gitee/EFD3D/scripts/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
