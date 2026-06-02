"""共享梯度计算工具。

本模块提供跨 constraints.py 和 pinn_two_phase.py 复用的梯度辅助函数，
消除跨文件公式重复。

Author: EFD-PINNs Team
"""

import torch


def compute_gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """统一梯度计算辅助函数。

    对 y.sum() 关于 x 求 autograd，零值回退为 zeros_like(x)。

    Args:
        y: 标量张量 (取 .sum() 用于 autograd)
        x: 输入坐标张量 [batch, dims]

    Returns:
        梯度张量，形状与 x 相同。如果梯度为 None，返回全零张量。

    Notes:
        此函数在 constraints.py 和 pinn_two_phase.py 中原各定义一次，
        现提取为共享实现以消除重复。
        create_graph=True 和 retain_graph=True 用于高阶导数链。
    """
    g = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        return torch.zeros_like(x)
    return g


def gradient_magnitude(
    phi_x: torch.Tensor,
    phi_y: torch.Tensor,
    phi_z: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算相场梯度的幅值及其平方。

    |∇φ|² = φ_x² + φ_y² + φ_z² + ε
    |∇φ|  = sqrt(|∇φ|²)

    ε 项用于数值稳定性，避免除零。

    Args:
        phi_x, phi_y, phi_z: 相场 φ 在 x/y/z 方向的一阶偏导数
        eps: 数值稳定常数 (默认 1e-10)

    Returns:
        (grad_mag_sq, grad_mag) 元组
    """
    grad_mag_sq = phi_x**2 + phi_y**2 + phi_z**2 + eps
    grad_mag = torch.sqrt(grad_mag_sq)
    return grad_mag_sq, grad_mag


def mean_curvature_3d(
    phi_x: torch.Tensor,
    phi_y: torch.Tensor,
    phi_z: torch.Tensor,
    phi_xx: torch.Tensor,
    phi_yy: torch.Tensor,
    phi_zz: torch.Tensor,
    phi_xy: torch.Tensor,
    phi_xz: torch.Tensor,
    phi_yz: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """三维精确平均曲率 (3D exact mean curvature)。

    κ = -div(∇φ/|∇φ|)

    显式公式：
    κ = -[(φ_xx(φ_y²+φ_z²) + φ_yy(φ_x²+φ_z²) + φ_zz(φ_x²+φ_y²)
           - 2(φ_x φ_y φ_xy + φ_x φ_z φ_xz + φ_y φ_z φ_yz))]
         / (|∇φ|³ + ε)

    Args:
        phi_x, phi_y, phi_z: 一阶偏导数
        phi_xx, phi_yy, phi_zz: 二阶纯偏导数
        phi_xy, phi_xz, phi_yz: 二阶混合偏导数
        eps: 数值稳定常数 (默认 1e-10)

    Returns:
        曲率张量 κ，形状与 phi_x 相同
    """
    grad_mag_sq = phi_x**2 + phi_y**2 + phi_z**2 + eps
    grad_mag = torch.sqrt(grad_mag_sq)

    numerator = (
        phi_xx * (phi_y**2 + phi_z**2)
        + phi_yy * (phi_x**2 + phi_z**2)
        + phi_zz * (phi_x**2 + phi_y**2)
        - 2
        * (phi_x * phi_y * phi_xy + phi_x * phi_z * phi_xz + phi_y * phi_z * phi_yz)
    )

    kappa = -numerator / (grad_mag_sq * grad_mag + eps)
    return kappa
