"""
EFD-PINNs 源代码包

Physics-Informed Neural Networks for Electrowetting Display Dynamics
"""

from . import models, physics, predictors, solvers, training, utils

__version__ = "1.0.0"
__all__ = [
    "models",
    "physics",
    "predictors",
    "solvers",
    "training",
    "utils",
]
