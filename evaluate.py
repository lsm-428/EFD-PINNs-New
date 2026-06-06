#!/usr/bin/env python3
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")  # Non-interactive backend
import argparse
import glob
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Any

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# =============================================================================
# IEEE Publication Standard Configuration (Added 2026-02-05)
# =============================================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
    }
)

# Standard panel labels for multi-panel figures
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# 导入模型和物理参数
from skimage import measure

from src.config import CONFIG_PATH, PHYSICS
from src.config.paths import OUTPUT_DIR
from src.models.aperture_model import EnhancedApertureModel
from src.models.pinn_two_phase import DEFAULT_CONFIG, TwoPhasePINN


class PINNEvaluator:
    """
    Professional PINN Evaluator for EWD Two-Phase Flow.
    Integrates metrics, physical field visualization, and performance benchmarking.
    """

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or str(CONFIG_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage1_model = EnhancedApertureModel(config_path=self.config_path)
        self.eta_max = PHYSICS.get("eta_max", 0.68)
        self.spatial_res = 64
        self.aperture_res = 96
        self.aperture_phi0 = 0.3
        self.aperture_eps = 0.05

        # Professional EWD Colormap: LightCyan (Polar Liquid, phi=0) -> Magenta (Ink, phi=1)
        self.ewd_cmap = LinearSegmentedColormap.from_list("EWD", ["#E0FFFF", "#FF00FF"])

    def load_model(self, checkpoint_path: str) -> tuple[TwoPhasePINN | None, dict[str, Any]]:
        """Load a trained model and its configuration."""
        try:
            # SECURITY FIX: Use weights_only=True to prevent arbitrary code execution
            # See: CERT VU#252619, PyTorch Security Advisory GHSA-53q9-r3pm-6pq6
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            config = checkpoint.get("config", DEFAULT_CONFIG)

            # 更新全局物理参数，确保评估与训练一致
            if "physics" in config:
                for k, v in config["physics"].items():
                    PHYSICS[k] = v
                lx_um = PHYSICS["Lx"] * 1e6
                lz_um = PHYSICS["Lz"] * 1e6
                print(f"ℹ️ Updated PHYSICS from checkpoint: Lx={lx_um:.0f}um, Lz={lz_um:.0f}um")

            # 重新初始化分析模型以匹配物理参数
            self.stage1_model = EnhancedApertureModel(config_path=self.config_path)

            if "model" not in config:
                config["model"] = {}

            model = TwoPhasePINN(config).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            eval_cfg = config.get("eval", {}) if isinstance(config.get("eval", {}), dict) else {}
            self.aperture_phi0 = float(eval_cfg.get("aperture_phi0", self.aperture_phi0))
            self.aperture_eps = float(eval_cfg.get("aperture_eps", self.aperture_eps))
            self.aperture_res = int(eval_cfg.get("aperture_res", self.aperture_res))
            return model, config
        except Exception as e:
            print(f"Error loading model from {checkpoint_path}: {e}")
            return None, {}

    def get_fields(
        self,
        model: TwoPhasePINN,
        V_to: float,
        t_since: float,
        V_from: float | None = None,
        plane: str = "xy",
        coord: float | None = None,
        spatial_res: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract physical fields (phi, u, v, w, p) from the model."""
        if V_from is None:
            V_from = V_to
        n = int(spatial_res or self.spatial_res)
        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]

        if plane == "xy":
            if coord is None:
                coord = PHYSICS["h_ink"] / 2
            x = np.linspace(0, Lx, n)
            y = np.linspace(0, Ly, n)
            X, Y = np.meshgrid(x, y)
            z = np.full_like(X, coord)
            grid = (X, Y)
        else:  # xz plane
            if coord is None:
                coord = Ly / 2
            x = np.linspace(0, Lx, n)
            z = np.linspace(0, Lz, n)
            X, Z = np.meshgrid(x, z)
            y = np.full_like(X, coord)
            grid = (X, Z)

        inputs = np.zeros((n * n, 6), dtype=np.float32)
        if plane == "xy":
            inputs[:, 0] = X.ravel()
            inputs[:, 1] = Y.ravel()
            inputs[:, 2] = z.ravel()
        else:  # xz
            inputs[:, 0] = X.ravel()
            inputs[:, 1] = y.ravel()
            inputs[:, 2] = Z.ravel()

        inputs[:, 3] = V_from
        inputs[:, 4] = V_to
        inputs[:, 5] = t_since

        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(inputs, device=self.device))
            # PERF OPTIMIZATION: Single CPU transfer, then slice on CPU
            out_np = out.cpu().numpy()
            phi_np = out_np[:, 4].reshape(n, n)

            # 诊断信息
            phi_min, phi_max = np.min(phi_np), np.max(phi_np)
            if phi_max - phi_min < 1e-3 and not (np.abs(t_since) < 1e-5 and np.abs(V_to) < 1e-5):
                # Only warn if not in a trivial state (e.g. t=0 where full ink is expected)
                msg = (
                    f"⚠️ Warning: Constant phi field in {plane} plane "
                    f"(min={phi_min:.4f}, max={phi_max:.4f}) at V={V_to}V, t={t_since}s"
                )
                print(msg)

            return {
                "u": out_np[:, 0].reshape(n, n),
                "v": out_np[:, 1].reshape(n, n),
                "w": out_np[:, 2].reshape(n, n),
                "p": out_np[:, 3].reshape(n, n),
                "phi": phi_np,
                "grid": grid,
            }

    def compute_aperture(
        self,
        model: TwoPhasePINN,
        V_to: float,
        t_since: float,
        V_from: float | None = None,
    ) -> float:
        """Calculate aperture ratio (eta) at z = 0 (hydrophobic surface).

        Use z=0 to match Stage 1 model's aperture definition.
        Stage 1 calculates aperture based on contact angle at z=0 surface.

        Note: This calculation assumes the model outputs a clear interface
        (φ ≈ 0 in water, φ ≈ 1 in ink). If the model outputs a constant field,
        the result will be unreliable. Use check_constant_field() to verify.
        """
        z_sample = 0.0  # Use z=0 to match Stage 1
        fields = self.get_fields(
            model,
            V_to,
            t_since,
            V_from,
            plane="xy",
            coord=z_sample,
            spatial_res=self.aperture_res,
        )
        phi = fields["phi"]

        # Check if the model outputs a constant field
        phi_std = np.std(phi)
        if phi_std < 0.05:
            # Constant field - aperture calculation is unreliable
            return float(np.nan)

        phi0 = float(self.aperture_phi0)
        eps = float(self.aperture_eps)
        if eps <= 0:
            return float(np.mean(phi < phi0))
        m = 0.5 * (1.0 + np.tanh((phi0 - phi) / eps))
        return float(np.mean(m))

    def check_constant_field(self, model: TwoPhasePINN, V: float, t: float) -> dict:
        """Check if the model outputs a constant φ field.

        Returns:
            dict with keys:
                - is_constant: bool
                - phi_mean: float
                - phi_std: float
                - message: str
        """
        z_sample = 0.0
        fields = self.get_fields(model, V, t, V, plane="xy", coord=z_sample, spatial_res=self.aperture_res)
        phi = fields["phi"]
        phi_mean = float(np.mean(phi))
        phi_std = float(np.std(phi))

        is_constant = phi_std < 0.05
        if is_constant:
            message = f"常数场 (φ={phi_mean:.3f}, std={phi_std:.4f})"
        else:
            message = f"正常场 (φ_mean={phi_mean:.3f}, std={phi_std:.3f})"

        return {
            "is_constant": is_constant,
            "phi_mean": phi_mean,
            "phi_std": phi_std,
            "message": message,
        }

    def plot_dashboard(self, model: TwoPhasePINN, output_path: str, model_name: str = "PINN"):
        """Generate a professional 4-panel dashboard for a single model."""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig)

        # 1. Top View (Phase + Velocity)
        ax1 = fig.add_subplot(gs[0, 0])
        f_xy = self.get_fields(model, 30.0, 0.02, 30.0, plane="xy")
        X, Y = f_xy["grid"]
        im1 = ax1.contourf(X * 1e6, Y * 1e6, f_xy["phi"], levels=20, cmap=self.ewd_cmap, vmin=0, vmax=1)
        ax1.contour(X * 1e6, Y * 1e6, f_xy["phi"], levels=[0.5], colors="k", linewidths=2)

        # 优化速度矢量显示：根据最大速度自动缩放
        skip = self.spatial_res // 16
        u, v = f_xy["u"], f_xy["v"]
        speed = np.sqrt(u**2 + v**2)
        max_speed = np.max(speed) if np.max(speed) > 1e-6 else 1.0
        ax1.quiver(
            X[::skip, ::skip] * 1e6,
            Y[::skip, ::skip] * 1e6,
            u[::skip, ::skip] / max_speed,
            v[::skip, ::skip] / max_speed,
            color="black",
            alpha=0.6,
            scale=20,
            width=0.005,
        )

        ax1.set_title(f"Top View (z={PHYSICS['h_ink'] * 1e6 / 2:.1f}μm, 30V)")
        ax1.set_xlabel("x (μm)")
        ax1.set_ylabel("y (μm)")
        plt.colorbar(im1, ax=ax1, label="φ")

        # 2. Side View (Phase + Velocity)
        ax2 = fig.add_subplot(gs[0, 1])
        f_xz = self.get_fields(model, 30.0, 0.02, 30.0, plane="xz")
        X, Z = f_xz["grid"]
        im2 = ax2.contourf(X * 1e6, Z * 1e6, f_xz["phi"], levels=20, cmap=self.ewd_cmap, vmin=0, vmax=1)
        ax2.contour(X * 1e6, Z * 1e6, f_xz["phi"], levels=[0.5], colors="k", linewidths=2)

        u, w = f_xz["u"], f_xz["w"]
        speed_xz = np.sqrt(u**2 + w**2)
        max_speed_xz = np.max(speed_xz) if np.max(speed_xz) > 1e-6 else 1.0
        ax2.quiver(
            X[::skip, ::skip] * 1e6,
            Z[::skip, ::skip] * 1e6,
            u[::skip, ::skip] / max_speed_xz,
            w[::skip, ::skip] / max_speed_xz,
            color="black",
            alpha=0.6,
            scale=20,
            width=0.005,
        )

        ax2.set_title("Side View (y=Ly/2, 30V)")
        ax2.set_xlabel("x (μm)")
        ax2.set_ylabel("z (μm)")
        plt.colorbar(im2, ax=ax2, label="φ")

        # 3. Dynamic Response (0->30->0V) — raw PINN vs Stage1
        ax3 = fig.add_subplot(gs[1, :2])
        t_max = PHYSICS.get("t_max", 0.05)
        times = np.linspace(0, t_max, 100)
        t_rise, t_fall = t_max * 0.1, t_max * 0.5  # 10% 处上升，50% 处下降

        pinn_etas = []
        s1_etas = []

        for t in times:
            if t < t_rise:
                Vf, Vt, ts = 0.0, 0.0, 0.0
                eta = self.compute_aperture(model, Vt, ts, Vf)
            elif t < t_fall:
                # ON Phase
                Vf, Vt, ts = 0.0, 30.0, t - t_rise
                eta = self.compute_aperture(model, Vt, ts, Vf)
            else:
                # OFF Phase
                Vf, Vt, ts = 30.0, 0.0, t - t_fall
                eta = self.compute_aperture(model, Vt, ts, Vf)

            eta = max(0.0, min(1.0, eta))
            pinn_etas.append(eta)

            _, s1_e = self.stage1_model.theta_eta_from_triad(Vf, Vt, ts)
            s1_etas.append(s1_e)

        ax3.plot(times * 1000, s1_etas, "k--", label="Analytical Ref", alpha=0.6)
        ax3.plot(times * 1000, pinn_etas, "r-", label="PINN Prediction", linewidth=2)
        ax3.axhline(self.eta_max, color="gray", linestyle=":", label=f"Limit ({self.eta_max})")
        ax3.set_title("Dynamic Step Response (0V → 30V → 0V)")
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Aperture Ratio")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.0)

        # 4. Steady State Sweep
        ax4 = fig.add_subplot(gs[1, 2])
        voltages = np.linspace(0, 30, 11)
        t_steady = t_max * 0.8  # 使用 80% 的时间作为稳态判定
        pinn_ss = [self.compute_aperture(model, V, t_steady, V) for V in voltages]
        s1_ss = [self.stage1_model.theta_eta_from_triad(V, V, t_steady)[1] for V in voltages]
        ax4.plot(voltages, s1_ss, "k--o", label="Analytical", alpha=0.6)
        ax4.plot(voltages, pinn_ss, "r-s", label="PINN")
        ax4.axhline(self.eta_max, color="gray", linestyle=":")
        ax4.set_title(f"Steady State Sweep (t={t_steady * 1000:.0f}ms)")
        ax4.set_xlabel("Voltage (V)")
        ax4.set_ylabel("Aperture Ratio")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.05, 1.0)

        plt.suptitle(f"Professional PINN Evaluation Dashboard - {model_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"✅ Dashboard saved: {output_path}")

    def plot_phi_grid(self, model: TwoPhasePINN, output_path: str):
        """Plot a 7x6 grid of phi fields for various voltages and times."""
        _fig, axes = plt.subplots(7, 6, figsize=(24, 28))
        h_ink = PHYSICS["h_ink"]
        z_sample = h_ink / 2

        row_configs = [
            (0, 0, [0.000, 0.005, 0.010, 0.015, 0.020, 0.040], "Initial OFF (0V)"),
            (0, 10, [0.000, 0.002, 0.005, 0.010, 0.015, 0.025], "ON: 0V→10V"),
            (10, 0, [0.001, 0.003, 0.005, 0.010, 0.015, 0.025], "OFF: 10V→0V"),
            (0, 20, [0.000, 0.002, 0.005, 0.010, 0.015, 0.025], "ON: 0V→20V"),
            (20, 0, [0.001, 0.003, 0.005, 0.010, 0.015, 0.025], "OFF: 20V→0V"),
            (0, 30, [0.000, 0.002, 0.005, 0.010, 0.015, 0.025], "ON: 0V→30V"),
            (30, 0, [0.001, 0.003, 0.005, 0.010, 0.015, 0.025], "OFF: 30V→0V"),
        ]

        for i, (V_from, V_to, times, row_label) in enumerate(row_configs):
            for j, t in enumerate(times):
                ax = axes[i, j]
                # t is t_since
                f = self.get_fields(model, V_to, t, V_from, plane="xy", coord=z_sample)
                X, Y = f["grid"]
                # 0V 静态行强制修正：φ > 0.99 视为 1.0，避免数值噪声导致的边缘毛刺
                if V_from == 0 and V_to == 0:
                    f["phi"] = np.where(f["phi"] > 0.99, 1.0, f["phi"])

                ax.contourf(
                    X * 1e6,
                    Y * 1e6,
                    f["phi"],
                    levels=20,
                    cmap=self.ewd_cmap,
                    vmin=0,
                    vmax=1,
                )
                ax.contour(
                    X * 1e6,
                    Y * 1e6,
                    f["phi"],
                    levels=[0.5],
                    colors="black",
                    linewidths=2,
                )
                ax.set_aspect("equal")

                eta0 = self.compute_aperture(model, V_to, t, V_from)
                # Color coding: Green for ON, Orange for OFF, Gray for Static
                color = "gray" if V_from == 0 and V_to == 0 else "green" if "ON" in row_label else "orange"

                # Explicitly label as t_since
                ax.set_title(
                    f"Δt={t * 1000:.1f}ms, η={eta0:.2f}",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )
                if j == 0:
                    ax.set_ylabel(f"{row_label}\ny (μm)", fontsize=9)
                if i == 6:
                    ax.set_xlabel("x (μm)", fontsize=8)

        plt.suptitle(
            f"Ink Layer Center View (z={z_sample * 1e6:.1f}μm): φ Field Evolution",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✅ Phi grid saved: {output_path}")

    def plot_interface_3d(
        self,
        model: TwoPhasePINN,
        output_path: str,
        V_to: float,
        t_since: float,
        V_from: float | None = None,
    ):
        """Plot 3D interface (phi=0.5 isosurface) using Marching Cubes."""
        if V_from is None:
            V_from = V_to
        n = 50  # 增加分辨率以获得更平滑的表面
        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]

        x = np.linspace(0, Lx, n)
        y = np.linspace(0, Ly, n)
        z = np.linspace(0, Lz, n)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        inputs = np.zeros((n * n * n, 6), dtype=np.float32)
        inputs[:, 0] = X.ravel()
        inputs[:, 1] = Y.ravel()
        inputs[:, 2] = Z.ravel()
        inputs[:, 3] = V_from
        inputs[:, 4] = V_to
        inputs[:, 5] = t_since

        with torch.no_grad():
            out = model(torch.tensor(inputs, device=self.device))
            # PERF OPTIMIZATION: Single CPU transfer for 3D data
            out_np = out.cpu().numpy()
            phi = out_np[:, 4].reshape(n, n, n)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # 使用 Marching Cubes 算法提取等值面
        try:
            # verts: 顶点坐标, faces: 三角面片索引
            # 注意：phi 可能是全1(油墨)或全0(介质)，此时没有0.5等值面，会抛出 RuntimeError
            if phi.min() > 0.5 or phi.max() < 0.5:
                pass  # 没有界面，无需绘制
            else:
                verts, faces, _normals, _values = measure.marching_cubes(
                    phi, level=0.5, spacing=(Lx / n, Ly / n, Lz / n)
                )

                # 绘制三角网格曲面
                # 添加 edgecolor='k' 和 linewidth=0.1 来勾勒网格线，增强立体感
                ax.plot_trisurf(
                    verts[:, 0] * 1e6,
                    verts[:, 1] * 1e6,
                    verts[:, 2] * 1e6,
                    triangles=faces,
                    cmap=self.ewd_cmap,
                    alpha=0.9,
                    edgecolor="k",
                    linewidth=0.1,
                    shade=True,
                )

                # 绘制在底面(z=0)和侧壁上的投影轮廓，以增强空间定位感
                # 1. 底面投影
                ax.tricontourf(
                    verts[:, 0] * 1e6,
                    verts[:, 1] * 1e6,
                    verts[:, 2] * 1e6,
                    triangles=faces,
                    zdir="z",
                    offset=0,
                    cmap=self.ewd_cmap,
                    alpha=0.3,
                )

                # 2. 侧壁投影 (x=Lx)
                ax.tricontourf(
                    verts[:, 0] * 1e6,
                    verts[:, 1] * 1e6,
                    verts[:, 2] * 1e6,
                    triangles=faces,
                    zdir="x",
                    offset=Lx * 1e6,
                    cmap=self.ewd_cmap,
                    alpha=0.1,
                )

        except Exception:
            # 静默处理（通常是因为没有界面，这是正常的物理状态）
            pass

        # 绘制像素框架线框
        # 底面
        xx, yy = np.meshgrid(np.linspace(0, Lx, 2) * 1e6, np.linspace(0, Ly, 2) * 1e6)
        ax.plot_surface(xx, yy, np.zeros_like(xx), color="gray", alpha=0.1)
        # 顶面线框
        ax.plot(
            [0, Lx * 1e6, Lx * 1e6, 0, 0],
            [0, 0, Ly * 1e6, Ly * 1e6, 0],
            [Lz * 1e6] * 5,
            "k--",
            linewidth=0.5,
            alpha=0.3,
        )
        # 垂直棱线
        for x_corner in [0, Lx * 1e6]:
            for y_corner in [0, Ly * 1e6]:
                ax.plot(
                    [x_corner, x_corner],
                    [y_corner, y_corner],
                    [0, Lz * 1e6],
                    "k--",
                    linewidth=0.5,
                    alpha=0.3,
                )

        time_ms = t_since * 1000
        title = f"3D Interface (φ=0.5) Evolution\nVoltage: {V_from}V → {V_to}V, Time: {time_ms:.1f}ms"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        ax.set_zlabel("z (μm)")
        ax.set_zlim(0, Lz * 1e6)

        # 调整视角以符合论文常用角度
        ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"✅ 3D interface saved: {output_path}")

    def plot_dynamic_response_curves(self, model: TwoPhasePINN, output_path: str):
        """Plot dynamic aperture response curves for multiple voltage cycles."""
        plt.figure(figsize=(12, 8))

        # 定义测试循环: (V_from, V_to, label)
        cycles = [
            (0.0, 0.0, "0V (Static)"),
            (0.0, 5.0, "5V"),
            (0.0, 10.0, "10V"),
            (0.0, 15.0, "15V"),
            (0.0, 20.0, "20V"),
            (0.0, 25.0, "25V"),
            (0.0, 30.0, "30V"),
        ]

        # Expanded color palette for more voltages
        colors = ["gray", "purple", "blue", "deepskyblue", "orange", "magenta", "red"]

        # 生成时间点: 0-50ms
        times = np.linspace(0, 0.050, 100)  # 50ms total
        t_switch = 0.025

        # 初始化 Stage1 解析模型 (使用已加载的 self.stage1_model)
        # stage1_model = self.stage1_model

        for (_v_start, v_target, label), color in zip(cycles, colors, strict=False):
            etas = []
            analytical_etas = []  # 解析解 (Stage 1)

            for t in times:
                # Raw PINN prediction — no post-processing
                if v_target == 0:  # Static 0V
                    eta = self.compute_aperture(model, 0.0, t, 0.0)
                elif t <= t_switch:
                    # ON Phase: 0 -> V_target
                    eta = self.compute_aperture(model, v_target, t, 0.0)
                else:
                    # OFF Phase: V_target -> 0
                    t_since_off = t - t_switch
                    eta = self.compute_aperture(model, 0.0, t_since_off, v_target)

                # Clip to physical range [0, 1]
                eta = max(0.0, min(1.0, eta))
                etas.append(eta)

                # Stage 1 analytical prediction
                if v_target == 0:
                    _, a_eta = self.stage1_model.theta_eta_from_triad(0, 0, t)
                elif t <= t_switch:
                    _, a_eta = self.stage1_model.theta_eta_from_triad(0, v_target, t)
                else:
                    t_since_off = t - t_switch
                    _, a_eta = self.stage1_model.theta_eta_from_triad(v_target, 0, t_since_off)

                analytical_etas.append(a_eta)

            # Plot raw PINN prediction (solid)
            plt.plot(times * 1000, etas, label=f"{label} (PINN)", color=color, linewidth=2)

            # Plot Stage 1 analytical (dashed)
            if v_target > 0:
                plt.plot(
                    times * 1000,
                    analytical_etas,
                    label=f"{label} (Stage1)",
                    color=color,
                    linestyle="--",
                    alpha=0.5,
                )

                # RMSE: raw PINN vs Stage1
                rmse = np.sqrt(np.mean((np.array(etas) - np.array(analytical_etas)) ** 2))
                print(f"Voltage {v_target}V: RMSE = {rmse:.4f}")

        plt.axvline(t_switch * 1000, color="k", linestyle="--", alpha=0.3, label="Switch OFF")

        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Aperture Ratio (η)", fontsize=12)
        plt.title("Dynamic Aperture Response: PINN vs Stage 1 Analytical", fontsize=14)
        plt.legend(loc="upper right", fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"✅ Dynamic response curves saved: {output_path}")

    def calculate_response_times(self, model: TwoPhasePINN, output_path: str):
        """Calculate and plot response time markers on dynamic curves."""
        # 1. 模拟动态响应曲线 (与 plot_dynamic_response_curves 类似，但专注于标记时间点)
        times = np.linspace(0, 0.050, 200)  # 50ms, 0.25ms step
        t_switch = 0.025

        voltages = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        colors = ["purple", "blue", "deepskyblue", "orange", "magenta", "red"]

        plt.figure(figsize=(12, 8))

        for V, color in zip(voltages, colors, strict=False):
            etas = []

            # Use raw PINN steady-state for target calculation
            eta_steady_on = self.compute_aperture(model, V, 0.050, V) if V > 0 else 0.0

            # t_on: time to reach 90% of steady-state
            target_on = 0.9 * eta_steady_on
            t_on = np.nan
            t_on_val = np.nan

            # t_off: time to drop to 10% of steady-state
            target_off = 0.1 * eta_steady_on
            t_off = np.nan
            t_off_val = np.nan

            for t in times:
                # Raw PINN prediction — no post-processing
                if t <= t_switch:
                    # ON Phase
                    eta = self.compute_aperture(model, V, t, 0.0)

                    if np.isnan(t_on) and eta >= target_on:
                        t_on = eta
                        t_on_val = t
                else:
                    # OFF Phase
                    t_since_off = t - t_switch
                    eta = self.compute_aperture(model, 0.0, t_since_off, V)

                    if np.isnan(t_off) and eta <= target_off:
                        t_off = eta
                        t_off_val = t_since_off

                # Clip to physical range [0, 1]
                eta = max(0.0, min(1.0, eta))
                etas.append(eta)

            # Plot curve
            plt.plot(times * 1000, etas, label=f"{int(V)}V", color=color, linewidth=2)

            # Mark t_on point
            if not np.isnan(t_on_val):
                plt.scatter(t_on_val * 1000, t_on, color=color, s=50, zorder=5)
                plt.text(
                    t_on_val * 1000,
                    t_on + 0.03,
                    f"t_on={t_on_val * 1000:.1f}ms",
                    color=color,
                    fontsize=9,
                    ha="center",
                    fontweight="bold",
                )

            # Mark t_off point
            if not np.isnan(t_off_val):
                abs_time = t_switch + t_off_val
                plt.scatter(abs_time * 1000, t_off, color=color, marker="s", s=50, zorder=5)
                plt.text(
                    abs_time * 1000,
                    t_off + 0.03,
                    f"t_off={t_off_val * 1000:.1f}ms",
                    color=color,
                    fontsize=9,
                    ha="center",
                    fontweight="bold",
                )

            t_on_ms = f"{t_on_val * 1000:.0f}" if not np.isnan(t_on_val) else ">25"
            t_off_ms = f"{t_off_val * 1000:.0f}" if not np.isnan(t_off_val) else ">25"
            print(f"Voltage {V}V: t_on={t_on_ms}ms, t_off={t_off_ms}ms")

        plt.axvline(t_switch * 1000, color="k", linestyle="--", alpha=0.3, label="Switch OFF")

        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Aperture Ratio (η)", fontsize=12)
        plt.title("Dynamic Response Time Analysis (90% ON / 10% OFF)", fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"✅ Response time analysis saved: {output_path}")

    def plot_mass_conservation(self, model: TwoPhasePINN, output_path: str):
        """Analyze mass conservation by integrating phi over the domain.

        Uses the SAME sampling method as training code for consistency:
        - Random sampling (30000 points) instead of fixed grid
        - Random time points (np.random.uniform) instead of fixed times
        - Same test cases as training code

        Tests three scenarios:
        1. ON process: V_from = 0, V_to = V (voltage rise)
        2. OFF process: V_from = V, V_to = 0 (voltage drop)
        3. Steady state: V_from = V, V_to = V (constant voltage)
        """
        Lx, Ly, Lz, h_ink = (
            PHYSICS["Lx"],
            PHYSICS["Ly"],
            PHYSICS["Lz"],
            PHYSICS["h_ink"],
        )
        v0 = Lx * Ly * h_ink
        v_domain = Lx * Ly * Lz

        print("\n=== Mass Conservation Analysis (Training Code Method) ===")
        print(f"Domain: {Lx * 1e6:.1f}x{Ly * 1e6:.1f}x{Lz * 1e6:.1f} um^3")
        print(f"Initial ink volume: {v0 * 1e18:.2f} um³")
        print()

        # Use the SAME sampling method as training code
        n_vol = 30000
        torch.manual_seed(42)
        np.random.seed(42)

        # Generate random sampling points (same as training code)
        x = torch.rand(n_vol, device=self.device) * Lx
        y = torch.rand(n_vol, device=self.device) * Ly
        z = torch.rand(n_vol, device=self.device) * Lz
        xyz = torch.stack([x, y, z], dim=1)

        # Generate random time points (same as training code)
        t_steady = np.random.uniform(0.020, 0.050, 3).tolist()
        t_on = np.random.uniform(0.002, 0.025, 5).tolist()
        t_off = np.random.uniform(0.002, 0.025, 4).tolist()

        # Build test cases (same as training code)
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

        print(f"Test cases: {len(tests)}")
        print()

        # Calculate volume conservation loss (same as training code)
        loss_vol = 0.0
        errors = []
        for v_from, v_to, t_since in tests:
            pts = torch.cat(
                [
                    xyz,
                    torch.full((n_vol, 1), float(v_from), device=self.device),
                    torch.full((n_vol, 1), float(v_to), device=self.device),
                    torch.full((n_vol, 1), float(t_since), device=self.device),
                ],
                dim=1,
            )

            with torch.no_grad():
                phi = model(pts)[:, 4]
                phi = torch.clamp(phi, 0.0, 1.0)

            v_curr = v_domain * phi.mean()
            rel_error = (v_curr - v0) / (v0 + 1e-12)
            loss_vol += rel_error**2
            errors.append(abs(rel_error.item()) * 100)

        # Calculate final loss (same as training code)
        base_weight = 2000.0
        stage_weight = 1.0
        final_vol = loss_vol * base_weight * stage_weight / len(tests)

        # Calculate statistics
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)

        print("=== Volume Conservation Results ===")
        print("Training log Vol value: 0.317 (Epoch 58000)")
        print(f"Calculated final_vol: {final_vol:.6f}")
        print()
        print("Volume error statistics:")
        print(f"  Average error: {avg_error:.2f}%")
        print(f"  Max error: {max_error:.2f}%")
        print(f"  Min error: {min_error:.2f}%")
        print()

        # Create a simple bar chart
        _fig, ax = plt.subplots(figsize=(8, 6))

        # Group errors by voltage
        voltage_errors = {}
        for i, (_v_from, v_to, _t_since) in enumerate(tests):
            v_key = f"{int(v_to)}V"
            if v_key not in voltage_errors:
                voltage_errors[v_key] = []
            voltage_errors[v_key].append(errors[i])

        # Calculate average error for each voltage
        voltages = []
        avg_errors = []
        for v_key in sorted(voltage_errors.keys(), key=lambda x: int(x.replace("V", ""))):
            voltages.append(v_key)
            avg_errors.append(np.mean(voltage_errors[v_key]))

        colors = ["gray", "blue", "orange", "red", "magenta"]
        ax.bar(voltages, avg_errors, color=colors[: len(voltages)], ec="black")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Average Volume Error (%)")
        ax.set_title("Volume Conservation by Voltage")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (_v, err) in enumerate(zip(voltages, avg_errors, strict=False)):
            ax.text(i, err + 0.5, f"{err:.1f}%", ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✅ Mass conservation plot saved: {output_path}")

    def plot_z_profile(self, model: TwoPhasePINN, output_path: str):
        """Analyze vertical phase field distribution (sharpness check)."""
        z = np.linspace(0, PHYSICS["Lz"], 100)
        center_x = PHYSICS["Lx"] / 2
        center_y = PHYSICS["Ly"] / 2

        inputs = np.zeros((100, 6), dtype=np.float32)
        inputs[:, 0] = center_x
        inputs[:, 1] = center_y
        inputs[:, 2] = z
        inputs[:, 3] = 30.0  # V_from
        inputs[:, 4] = 30.0  # V_to
        inputs[:, 5] = 0.02  # t_since (steady state)

        with torch.no_grad():
            out = model(torch.tensor(inputs, device=self.device))
            # PERF OPTIMIZATION: Single CPU transfer, then slice on CPU
            phi = out[:, 4].cpu().numpy()

        plt.figure(figsize=(8, 6))
        plt.plot(z * 1e6, phi, "b-", linewidth=2, label="PINN φ")
        plt.axhline(0.5, color="k", linestyle=":", alpha=0.5)
        plt.axhline(0.0, color="k", linestyle="-", alpha=0.2)
        plt.axhline(1.0, color="k", linestyle="-", alpha=0.2)

        plt.xlabel("z (μm)")
        plt.ylabel("Phase Field φ")
        plt.title("Vertical Phase Profile at Center (30V, Steady)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✅ Z-profile plot saved: {output_path}")

    # =============================================================================
    # Feature: Wall-Climb Detection, Steady-State eta(V), Critical Voltage
    # =============================================================================

    def plot_steady_state_eta(
        self,
        model: TwoPhasePINN,
        output_path: str,
        voltages: list[float] | None = None,
        t_steady: float = 0.040,
    ):
        """Plot steady-state aperture ratio eta vs voltage.

        This is THE key result: at each voltage, what is the final opening?
        V_th = voltage where eta starts rising from 0
        eta_sat = saturated aperture at high voltage
        """
        if voltages is None:
            voltages = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

        etas = []
        for V in voltages:
            eta = self.compute_aperture(model, V, t_steady, V)
            etas.append(eta)
            print(f"  V={V:5.1f}V  eta={eta:.4f}")

        etas = np.array(etas)

        # Find threshold voltage (eta > 0.05)
        v_threshold = None
        for V, eta_val in zip(voltages, etas, strict=False):
            if eta_val > 0.05:
                v_threshold = V
                break

        # Find saturation voltage (eta > 0.9 * eta_max)
        eta_max_val = etas[-1] if len(etas) > 0 else 0
        v_saturation = None
        for V, eta_val in zip(voltages, etas, strict=False):
            if eta_val > 0.9 * eta_max_val:
                v_saturation = V
                break

        # Plot
        _fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(voltages, etas, "b-o", linewidth=2, markersize=5, label="PINN eta(V)")

        # Mark threshold
        if v_threshold is not None:
            ax.axvline(
                v_threshold,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"V_th ~ {v_threshold:.0f}V (eta>0.05)",
            )
            idx = voltages.index(v_threshold)
            ax.scatter([v_threshold], [etas[idx]], color="red", s=80, zorder=5)

        # Mark saturation
        if v_saturation is not None:
            ax.axvline(
                v_saturation,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"V_sat ~ {v_saturation:.0f}V (eta>0.9*eta_max)",
            )

        # Mark wall-climb danger zone
        ax.axhspan(0.85, 1.05, alpha=0.1, color="red", label="Wall-climb risk (eta>0.85)")
        ax.axhline(0.85, color="red", linestyle=":", alpha=0.4)

        ax.set_xlabel("Voltage (V)", fontsize=12)
        ax.set_ylabel("Aperture Ratio eta", fontsize=12)
        ax.set_title("Steady-State Aperture vs Voltage (t=40ms)", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

        v_th_str = f"{v_threshold}V" if v_threshold else "NOT REACHED"
        v_sat_str = f"{v_saturation}V" if v_saturation else "NOT REACHED"
        print("\nSteady-State eta(V) Results:")
        print(f"   V_threshold (eta>0.05): {v_th_str}")
        print(f"   V_saturation (eta>0.9*eta_max): {v_sat_str}")
        print(f"   eta at 30V: {etas[-1]:.4f}")
        print(f"Steady-state eta(V) plot saved: {output_path}")

        return {
            "voltages": voltages,
            "etas": etas.tolist(),
            "v_threshold": v_threshold,
            "v_saturation": v_saturation,
        }

    def detect_wall_climb(
        self,
        model: TwoPhasePINN,
        output_path: str,
        voltages: list[float] | None = None,
        t_steady: float = 0.040,
    ):
        """Detect wall-climbing: oil film overflowing the pixel wall.

        Wall-climb = phi > 0 at the pixel boundary (r -> r_wall, z near Lz).
        In a well-behaved EWD pixel, oil should be confined within the pixel well.
        If phi is significant near the wall top, the oil is climbing out.

        Checks:
        1. phi at (r=r_wall, z=Lz) -- wall top corner, should be ~0
        2. phi along (r=r_wall, z=0..Lz) -- wall edge profile
        3. phi radial profile at z=0 -- should be confined within pixel radius
        """
        if voltages is None:
            voltages = [0, 5, 10, 15, 20, 25, 30]

        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
        _cx, cy = Lx / 2, Ly / 2

        n_r = 50
        n_z = 50

        results = {}

        _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # --- Panel 1: Wall-edge phi vs Voltage ---
        ax1 = axes[0]
        wall_phi_top = []
        wall_phi_mid = []

        for V in voltages:
            z_arr = np.linspace(0, Lz, n_z)
            inputs = np.zeros((n_z, 6), dtype=np.float32)
            inputs[:, 0] = Lx
            inputs[:, 1] = cy
            inputs[:, 2] = z_arr
            inputs[:, 3] = V
            inputs[:, 4] = V
            inputs[:, 5] = t_steady

            with torch.no_grad():
                out = model(torch.tensor(inputs, device=self.device))
                phi_wall = out[:, 4].cpu().numpy()

            wall_phi_top.append(float(phi_wall[-1]))
            wall_phi_mid.append(float(phi_wall[n_z // 2]))
            results[V] = {"wall_phi_z": phi_wall.tolist()}

            ax1.plot(z_arr * 1e6, phi_wall, label=f"{V}V", linewidth=1.5)

        ax1.set_xlabel("z (um)")
        ax1.set_ylabel("phi at wall edge")
        ax1.set_title("Phase Field Along Wall Edge")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0.5, color="k", linestyle=":", alpha=0.3)

        # --- Panel 2: Wall-top phi vs Voltage ---
        ax2 = axes[1]
        ax2.plot(voltages, wall_phi_top, "rs-", linewidth=2, markersize=6, label="phi at wall top (z=Lz)")
        ax2.plot(
            voltages,
            wall_phi_mid,
            "b^-",
            linewidth=2,
            markersize=6,
            label="phi at wall mid (z=Lz/2)",
        )
        ax2.axhline(0.1, color="orange", linestyle="--", alpha=0.7, label="Warning: phi>0.1")
        ax2.axhline(0.3, color="red", linestyle="--", alpha=0.7, label="Critical: phi>0.3 (climbing!)")
        ax2.set_xlabel("Voltage (V)")
        ax2.set_ylabel("phi at wall edge")
        ax2.set_title("Wall-Edge Phase vs Voltage")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        climb_voltages = [V for V, p in zip(voltages, wall_phi_top, strict=False) if p > 0.3]
        warn_voltages = [V for V, p in zip(voltages, wall_phi_top, strict=False) if p > 0.1]
        print("\nWall-Climb Detection Results:")
        print(f"   Wall-top phi>0.1 (warning) at: {warn_voltages}V")
        print(f"   Wall-top phi>0.3 (climbing!) at: {climb_voltages}V")
        if climb_voltages:
            print(f"   *** OIL CLIMBING DETECTED at V >= {min(climb_voltages)}V! ***")
        elif warn_voltages:
            print(f"   *** Wall leakage warning at V >= {min(warn_voltages)}V ***")
        else:
            print(f"   No wall-climb detected up to {max(voltages)}V")

        # --- Panel 3: Radial phi profile at z=0 for key voltages ---
        ax3 = axes[2]
        x_arr = np.linspace(0, Lx, n_r)
        for V in [0, 10, 20, 30]:
            inputs = np.zeros((n_r, 6), dtype=np.float32)
            inputs[:, 0] = x_arr
            inputs[:, 1] = cy
            inputs[:, 2] = 0.0
            inputs[:, 3] = V
            inputs[:, 4] = V
            inputs[:, 5] = t_steady

            with torch.no_grad():
                out = model(torch.tensor(inputs, device=self.device))
                phi_radial = out[:, 4].cpu().numpy()

            ax3.plot(x_arr * 1e6, phi_radial, label=f"{V}V", linewidth=2)

        ax3.axhline(
            self.aperture_phi0,
            color="k",
            linestyle=":",
            alpha=0.4,
            label=f"phi0={self.aperture_phi0}",
        )
        ax3.set_xlabel("x (um)")
        ax3.set_ylabel("phi at z=0")
        ax3.set_title("Radial Phase Profile at Substrate")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.suptitle("Wall-Climb & Oil Containment Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Wall-climb analysis saved: {output_path}")

        return results

    def find_critical_voltage(
        self,
        model: TwoPhasePINN,
        output_path: str,
        V_range: tuple[float, float] = (0, 35),
        V_step: float = 1.0,
        t_steady: float = 0.040,
    ):
        """Find critical voltages:
        - V_open (starts to open)
        - V_full (fully open)
        - V_climb (oil climbs wall)

        These are the three key voltages for an EWD pixel:
        - V_open: eta just becomes detectable (eta > 0.02)
        - V_full: eta reaches practical maximum (eta > 0.8)
        - V_climb: oil starts climbing the wall (wall-top phi > 0.3)
        """
        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
        _cx, cy = Lx / 2, Ly / 2

        voltages = np.arange(V_range[0], V_range[1] + V_step, V_step)
        etas = []
        wall_phis = []

        print("\nCritical Voltage Scan:")
        print(f"   Scanning V = {V_range[0]:.0f} -> {V_range[1]:.0f}V, step = {V_step:.0f}V")

        for V in voltages:
            eta = self.compute_aperture(model, float(V), t_steady, float(V))
            etas.append(eta)

            inputs = np.array([[Lx, cy, Lz, V, V, t_steady]], dtype=np.float32)
            with torch.no_grad():
                out = model(torch.tensor(inputs, device=self.device))
                phi_wall = float(out[0, 4].cpu().numpy())
            wall_phis.append(phi_wall)

            if V_step + 0.01 > V % 5:
                print(f"   V={V:5.1f}V  eta={eta:.4f}  phi_wall={phi_wall:.4f}")

        etas = np.array(etas)
        wall_phis = np.array(wall_phis)

        V_open = None
        V_full = None
        V_climb = None

        for V, eta_val in zip(voltages, etas, strict=False):
            if eta_val > 0.02 and V_open is None:
                V_open = V
            if eta_val > 0.80 and V_full is None:
                V_full = V
                break

        for V, phi_val in zip(voltages, wall_phis, strict=False):
            if phi_val > 0.3 and V_climb is None:
                V_climb = V
                break

        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(voltages, etas, "b-", linewidth=2, label="eta (aperture ratio)")
        ax1.axhline(0.02, color="gray", linestyle=":", alpha=0.5, label="eta=0.02 (open threshold)")
        ax1.axhline(0.80, color="gray", linestyle="--", alpha=0.5, label="eta=0.80 (full open)")
        if V_open is not None:
            ax1.axvline(V_open, color="green", linestyle="-.", alpha=0.7, label=f"V_open ~ {V_open:.0f}V")
        if V_full is not None:
            ax1.axvline(V_full, color="blue", linestyle="-.", alpha=0.7, label=f"V_full ~ {V_full:.0f}V")
        if V_climb is not None:
            ax1.axvline(V_climb, color="red", linestyle="-.", alpha=0.7, label=f"V_climb ~ {V_climb:.0f}V")
        ax1.set_ylabel("Aperture Ratio eta")
        ax1.set_title("Critical Voltage Analysis")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        ax2.plot(voltages, wall_phis, "r-", linewidth=2, label="phi at wall top")
        ax2.axhline(0.1, color="orange", linestyle="--", alpha=0.7, label="Warning (phi>0.1)")
        ax2.axhline(0.3, color="red", linestyle="--", alpha=0.7, label="Climbing! (phi>0.3)")
        if V_climb is not None:
            ax2.axvline(V_climb, color="red", linestyle="-.", alpha=0.7, label=f"V_climb ~ {V_climb:.0f}V")
        ax2.set_xlabel("Voltage (V)")
        ax2.set_ylabel("Wall-Top Phase phi")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

        v_open_str = f"{V_open:.1f}V" if V_open is not None else "NOT REACHED"
        v_full_str = f"{V_full:.1f}V" if V_full is not None else "NOT REACHED"
        v_climb_str = f"{V_climb:.1f}V" if V_climb is not None else "NO CLIMB"
        print("\nCritical Voltage Results:")
        print(f"   V_open  (eta>0.02): {v_open_str}")
        print(f"   V_full  (eta>0.80): {v_full_str}")
        print(f"   V_climb (phi_wall>0.3): {v_climb_str}")
        print(f"Critical voltage plot saved: {output_path}")

        return {
            "V_open": V_open,
            "V_full": V_full,
            "V_climb": V_climb,
            "voltages": voltages.tolist(),
            "etas": etas.tolist(),
            "wall_phis": wall_phis.tolist(),
        }

    # =============================================================================
    # Feature: Statistical Significance Test (Best vs Final Model)
    # =============================================================================

    def compute_aperture_samples(
        self,
        model: TwoPhasePINN,
        V_to: float,
        t_since: float,
        V_from: float | None = None,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        Compute aperture ratio with multiple spatial samples for statistical analysis.

        Returns:
            Array of aperture values from different spatial positions
        """
        if V_from is None:
            V_from = V_to

        # Fix seed for reproducible statistical tests
        np.random.seed(42)

        # Sample multiple positions
        aperture_values = []
        for _ in range(n_samples):
            # Random position within the domain
            x = np.random.uniform(0, PHYSICS["Lx"])
            y = np.random.uniform(0, PHYSICS["Ly"])
            z = PHYSICS["h_ink"] / 2  # Mid-plane

            inputs = np.array([[x, y, z, V_from, V_to, t_since]], dtype=np.float32)

            with torch.no_grad():
                out = model(torch.tensor(inputs, device=self.device))
                # PERF OPTIMIZATION: Transfer full tensor to CPU first, then index
                phi = out[0, 4].cpu().numpy()

            # Compute aperture at this position
            phi0 = self.aperture_phi0
            eps = self.aperture_eps
            eta = float(phi < phi0) if eps <= 0 else 0.5 * (1.0 + np.tanh((phi0 - phi) / eps))

            aperture_values.append(eta)

        return np.array(aperture_values)

    def compare_models_statistically(
        self,
        model1: TwoPhasePINN,
        model2: TwoPhasePINN,
        output_path: str,
        test_voltages: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Statistical comparison between two models (e.g., best vs final).

        Performs:
        1. Aperture ratio comparison at multiple voltages
        2. Paired t-test for significance
        3. Effect size calculation (Cohen's d)

        Args:
            model1: First model (e.g., best)
            model2: Second model (e.g., final)
            output_path: Path to save comparison plot
            test_voltages: List of voltages to test

        Returns:
            Dictionary with statistical results
        """
        if test_voltages is None:
            test_voltages = [10.0, 20.0, 30.0]

        print("\n=== Statistical Model Comparison ===")

        results = {
            "voltages": [],
            "paired_ttest": {},
            "effect_sizes": {},
            "summary": {},
        }

        _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Test at different voltages
        all_eta1 = []
        all_eta2 = []

        for i, V in enumerate(test_voltages):
            eta1 = self.compute_aperture_samples(model1, V, 0.02, V)
            eta2 = self.compute_aperture_samples(model2, V, 0.02, V)

            all_eta1.extend(eta1)
            all_eta2.extend(eta2)

            # Paired t-test
            from scipy import stats as scipy_stats

            t_stat, p_value = scipy_stats.ttest_rel(eta1, eta2)

            # Cohen's d effect size
            pooled_std = np.sqrt((np.std(eta1) ** 2 + np.std(eta2) ** 2) / 2)
            cohens_d = (np.mean(eta1) - np.mean(eta2)) / pooled_std if pooled_std > 0 else 0

            results["voltages"].append(
                {
                    "voltage": V,
                    "model1_mean": float(np.mean(eta1)),
                    "model1_std": float(np.std(eta1)),
                    "model2_mean": float(np.mean(eta2)),
                    "model2_std": float(np.std(eta2)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": float(cohens_d),
                }
            )

            # Plot comparison for this voltage
            ax = axes[i // 2, i % 2]
            positions = np.arange(len(eta1))
            ax.scatter(positions, eta1, alpha=0.5, label="Model 1 (Best)", s=20)
            ax.scatter(positions, eta2, alpha=0.5, label="Model 2 (Final)", s=20)
            ax.axhline(np.mean(eta1), color="blue", linestyle="--", alpha=0.7)
            ax.axhline(np.mean(eta2), color="orange", linestyle="--", alpha=0.7)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Aperture Ratio")
            ax.set_title(f"Voltage {V}V: p={p_value:.4f}, d={cohens_d:.3f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Overall statistics
        all_eta1 = np.array(all_eta1)
        all_eta2 = np.array(all_eta2)

        t_stat, p_value = scipy_stats.ttest_rel(all_eta1, all_eta2)
        pooled_std = np.sqrt((np.std(all_eta1) ** 2 + np.std(all_eta2) ** 2) / 2)
        cohens_d = (np.mean(all_eta1) - np.mean(all_eta2)) / pooled_std if pooled_std > 0 else 0

        results["summary"] = {
            "overall_t_statistic": float(t_stat),
            "overall_p_value": float(p_value),
            "overall_cohens_d": float(cohens_d),
            "significant_at_0.05": bool(p_value < 0.05),
            "significant_at_0.01": bool(p_value < 0.01),
        }

        # Interpretation
        if p_value < 0.01:
            interpretation = "Highly significant difference (p < 0.01)"
        elif p_value < 0.05:
            interpretation = "Significant difference (p < 0.05)"
        else:
            interpretation = "No significant difference (p >= 0.05)"

        if abs(cohens_d) < 0.2:
            effect_interpretation = "Negligible effect"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "Small effect"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"

        # Summary text
        summary_text = (
            f"Paired t-test: {interpretation}\n"
            f"Effect size: {effect_interpretation} (Cohen's d = {cohens_d:.3f})\n"
            f"Model 1 mean: {np.mean(all_eta1):.4f} ± {np.std(all_eta1):.4f}\n"
            f"Model 2 mean: {np.mean(all_eta2):.4f} ± {np.std(all_eta2):.4f}"
        )

        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.1,
            0.5,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="center",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.suptitle(
            "Statistical Model Comparison: Best vs Final",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✅ Statistical comparison saved: {output_path}")

        # Print summary
        print("\nPaired t-test Results:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print(f"  Interpretation: {interpretation}")
        print(f"  Effect size: {effect_interpretation}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Unified PINN Evaluation & Visualization Tool")
    parser.add_argument(
        "model_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to model directory (e.g., outputs_pinn_...)",
    )
    parser.add_argument("--compare", action="store_true", help="Compare last 2 models")
    parser.add_argument("--res", type=int, default=64, help="Spatial resolution for plots")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best",
        help="Checkpoint to load: best | latest | both | <path-to-pth>",
    )
    parser.add_argument(
        "--stat-test",
        action="store_true",
        help="Run statistical significance test between best and final models",
    )
    args = parser.parse_args()

    evaluator = PINNEvaluator()
    evaluator.spatial_res = args.res

    if args.stat_test:
        # Statistical significance test between best and final models
        if args.model_dir is None:
            legacy_dirs = glob.glob("outputs_pinn_*")
            new_dirs = glob.glob(str(OUTPUT_DIR / "train" / "pinn_*"))
            output_dirs = sorted(legacy_dirs + new_dirs, key=os.path.getmtime)
            if output_dirs:
                args.model_dir = output_dirs[-1]

        if not args.model_dir or not os.path.exists(args.model_dir):
            print("Error: Model directory not found for statistical test.")
            return

        best_ckpt = os.path.join(args.model_dir, "best_model.pth")
        final_ckpt = os.path.join(args.model_dir, "final_model.pth")

        if not os.path.exists(best_ckpt) or not os.path.exists(final_ckpt):
            print("Error: Both best_model.pth and final_model.pth are required for statistical test.")
            return

        print("Loading models for statistical comparison...")
        model1, _ = evaluator.load_model(best_ckpt)
        model2, _ = evaluator.load_model(final_ckpt)

        if model1 and model2:
            output_path = os.path.join(args.model_dir, "statistical_comparison.png")
            evaluator.compare_models_statistically(model1, model2, output_path)
            print(f"\n✅ Statistical test complete. Results saved to {output_path}")

    elif args.compare:
        # [2026-01-16] 兼容旧目录和新规范目录
        legacy_dirs = glob.glob("outputs_pinn_*")
        new_dirs = glob.glob(str(OUTPUT_DIR / "train" / "pinn_*"))
        output_dirs = sorted(legacy_dirs + new_dirs, key=os.path.getmtime)

        if len(output_dirs) < 2:
            print("Need at least 2 models for comparison.")
            return

        target_dirs = output_dirs[-2:]
        print(f"Comparing: {target_dirs[0]} vs {target_dirs[1]}")

        for d in target_dirs:
            ckpt = os.path.join(d, "best_model.pth")
            model, _ = evaluator.load_model(ckpt)
            if model:
                out = os.path.join(d, "pro_dashboard.png")
                evaluator.plot_dashboard(model, out, model_name=os.path.basename(d))
    else:
        # Single model evaluation (or multiple checkpoints in one dir)
        if args.model_dir is None:
            # [2026-01-16] 兼容旧目录和新规范目录
            legacy_dirs = glob.glob("outputs_pinn_*")
            new_dirs = glob.glob(str(OUTPUT_DIR / "train" / "pinn_*"))
            output_dirs = sorted(legacy_dirs + new_dirs, key=os.path.getmtime)

            if not output_dirs:
                print("No model directories found.")
                return
            args.model_dir = output_dirs[-1]
        elif not os.path.exists(args.model_dir):
            # Try to resolve relative to OUTPUT_DIR/train
            potential_path = OUTPUT_DIR / "train" / args.model_dir
            if potential_path.exists():
                args.model_dir = str(potential_path)

        print(f"Evaluating: {args.model_dir}")

        # Determine which checkpoints to evaluate
        ckpts_to_eval = []
        ckpts_to_eval = ["best", "final"] if args.ckpt in {"both", "all"} else [args.ckpt]

        for ckpt_type in ckpts_to_eval:
            if ckpt_type == "best":
                ckpt_path = os.path.join(args.model_dir, "best_model.pth")
                suffix = "best"
            elif ckpt_type == "latest":
                ckpt_path = os.path.join(args.model_dir, "latest_model.pth")
                suffix = "latest"
            elif ckpt_type == "final":
                ckpt_path = os.path.join(args.model_dir, "final_model.pth")
                suffix = "final"
            else:
                ckpt_path = ckpt_type
                suffix = "custom"

            if not os.path.exists(ckpt_path):
                print(f"⚠️ Checkpoint not found: {ckpt_path}, skipping...")
                continue

            print(f"Loading {ckpt_type} model from {ckpt_path}...")
            model, config = evaluator.load_model(ckpt_path)

            if model:
                # 1. Professional Dashboard
                out = os.path.join(args.model_dir, f"pro_dashboard_{suffix}.png")
                # Use version from config for reproducible rendering
                model_label = config.get("metadata", {}).get("version", "PINN")
                evaluator.plot_dashboard(model, out, model_name=model_label)

                # 2. Phi Grid Evolution
                grid_out = os.path.join(args.model_dir, f"phi_grid_evolution_{suffix}.png")
                evaluator.plot_phi_grid(model, grid_out)

                # 3. 3D Interface
                interface_out = os.path.join(args.model_dir, f"interface_3d_steady_{suffix}.png")
                evaluator.plot_interface_3d(model, interface_out, 30.0, 0.02, 30.0)

                # 4. Dynamic Response Curves (0-50ms)
                curves_out = os.path.join(args.model_dir, f"dynamic_curves_{suffix}.png")
                evaluator.plot_dynamic_response_curves(model, curves_out)

                # 5. Response Time Stats
                stats_out = os.path.join(args.model_dir, f"response_times_{suffix}.png")
                evaluator.calculate_response_times(model, stats_out)

                # 6. Mass Conservation
                mass_out = os.path.join(args.model_dir, f"mass_conservation_{suffix}.png")
                evaluator.plot_mass_conservation(model, mass_out)

                # 7b. Steady-State eta(V) Curve
                eta_v_out = os.path.join(args.model_dir, f"steady_state_eta_{suffix}.png")
                evaluator.plot_steady_state_eta(model, eta_v_out)

                # 7c. Wall-Climb Detection
                climb_out = os.path.join(args.model_dir, f"wall_climb_{suffix}.png")
                evaluator.detect_wall_climb(model, climb_out)

                # 7d. Critical Voltage Analysis
                critical_out = os.path.join(args.model_dir, f"critical_voltage_{suffix}.png")
                evaluator.find_critical_voltage(model, critical_out)

                # 7. Z-Profile
                z_out = os.path.join(args.model_dir, f"z_profile_{suffix}.png")
                evaluator.plot_z_profile(model, z_out)

        print(f"✅ Evaluation complete. Results saved in {args.model_dir}")


if __name__ == "__main__":
    main()
