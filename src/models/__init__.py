"""
模型定义模块

包含 PINN 模型和开口率模型定义

注意: LSTM-PINN 模型已移至 experimental/lstm_pinn/
"""

from .aperture_model import ApertureModel, EnhancedApertureModel
from .pinn_data_generator import DataGenerator, PhysicsBasedSampler
from .pinn_network import FourierFeature, TwoPhasePINN
from .pinn_physics_loss import PhysicsLoss
from .pinn_trainer_loop import Trainer

__all__ = [
    "ApertureModel",
    "DataGenerator",
    "EnhancedApertureModel",
    "FourierFeature",
    "PhysicsBasedSampler",
    "PhysicsLoss",
    "Trainer",
    "TwoPhasePINN",
]
