"""Dashboard module for EFD3D visualization and interaction.

This module provides dashboard components for interactive exploration
and visualization of EFD3D simulation results.
"""

from src.dashboard.datastore import DataStore
from src.dashboard.model_manager import (
    ModelManager,
    clear_default_manager,
    get_default_manager,
)
from src.dashboard.training_output_analyzer import (
    LossDataParser,
    MetricsParser,
    ModelInfo,
    ModelLoader,
    TrainingConfigParser,
    TrainingOutputAnalyzer,
    TrainingOutputScanner,
    TrainingRunInfo,
)

__all__ = [
    "DataStore",
    "LossDataParser",
    "MetricsParser",
    "ModelInfo",
    "ModelLoader",
    "ModelManager",
    "TrainingConfigParser",
    "TrainingOutputAnalyzer",
    "TrainingOutputScanner",
    "TrainingRunInfo",
    "clear_default_manager",
    "get_default_manager",
]
