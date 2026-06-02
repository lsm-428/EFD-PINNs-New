# src/config/__init__.py
"""EFD3D 配置模块"""

from .paths import (
    CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    OUTPUT_DIR,
    PROJECT_ROOT,
    get_config_path,
    get_output_dir,
)
from .physics_config import (
    PHYSICS,
    PhysicsConfig,
    get_materials_params,
    get_physics_config,
)

__all__ = [
    "CONFIG_PATH",
    "DEFAULT_CONFIG_PATH",
    "OUTPUT_DIR",
    "PHYSICS",
    # 路径管理
    "PROJECT_ROOT",
    # 物理配置
    "PhysicsConfig",
    "get_config_path",
    "get_materials_params",
    "get_output_dir",
    "get_physics_config",
]
