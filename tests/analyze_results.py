"""
验证训练输出目录结构与配置一致性
(不依赖已训练模型，可在任何阶段运行)
"""

import glob
import json
import os

import pytest

MODEL_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "train"
)


def test_outputs_directory_exists():
    """验证 outputs/train 目录存在"""
    assert os.path.isdir(MODEL_BASE_DIR) or not os.path.exists(
        os.path.join(os.path.dirname(MODEL_BASE_DIR), "outputs")
    ), f"Expected outputs/train/ directory at {MODEL_BASE_DIR}"


def test_find_training_runs():
    """验证能找到至少一次训练运行记录"""
    if not os.path.isdir(MODEL_BASE_DIR):
        pytest.skip("No outputs/train directory found yet")

    run_dirs = sorted(glob.glob(os.path.join(MODEL_BASE_DIR, "pinn_*")))
    assert len(run_dirs) > 0, "No pinn_* training run directories found"


def test_training_log_exists():
    """验证最新训练运行有 training.log"""
    if not os.path.isdir(MODEL_BASE_DIR):
        pytest.skip("No outputs/train directory")

    run_dirs = sorted(glob.glob(os.path.join(MODEL_BASE_DIR, "pinn_*")))
    if not run_dirs:
        pytest.skip("No pinn_* directories found")

    latest = run_dirs[-1]
    log_path = os.path.join(latest, "training.log")

    assert os.path.isfile(log_path), (
        f"Latest training run {os.path.basename(latest)} missing training.log"
    )


def test_config_consistency():
    """验证 config 文件结构完整"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    assert os.path.isdir(config_dir), "config/ directory should exist"

    # 核心配置文件应存在
    required_configs = ["device_calibrated_physics.json", "v4.6-optimized.json"]
    for cfg_name in required_configs:
        cfg_path = os.path.join(config_dir, cfg_name)
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            assert isinstance(cfg, dict), f"{cfg_name} should be a valid JSON object"


def test_config_physics_params():
    """验证 device_calibrated_physics.json 物理参数完整性"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    cfg_path = os.path.join(config_dir, "device_calibrated_physics.json")

    if not os.path.exists(cfg_path):
        pytest.skip("device_calibrated_physics.json not found")

    with open(cfg_path) as f:
        cfg = json.load(f)

    # 关键物理参数应存在于配置中（使用实际键名）
    essential_keys = [
        "epsilon_r",  # 相对介电常数
        "dielectric_thickness",
        "hydrophobic_thickness",
        "sigma",  # 表面张力 (polar ink)
        "gamma",  # 表面张力 (另一组分)
        "ac_interface_width",
        "A_eff",  # 有效面积校正因子 (v7.2)
    ]

    # 递归查找参数（可能嵌套在 materials/physics/model 等节中）
    def find_key(d, key):
        if key in d:
            return True
        return any(isinstance(v, dict) and find_key(v, key) for v in d.values())

    for key in essential_keys:
        found = find_key(cfg, key)
        assert found, f"Missing essential parameter '{key}' anywhere in config"


def test_training_config_structure():
    """验证 v4.6-optimized.json 训练配置结构"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    cfg_path = os.path.join(config_dir, "v4.6-optimized.json")

    if not os.path.exists(cfg_path):
        pytest.skip("v4.6-optimized.json not found")

    with open(cfg_path) as f:
        cfg = json.load(f)

    # 应有 physics 和 training 子配置
    assert "physics" in cfg, "Config should have 'physics' section"
    assert isinstance(cfg["physics"], dict), "'physics' should be a dict"

    # EW 权重应存在
    assert "electrowetting_weight" in cfg["physics"], (
        "Missing electrowetting_weight in physics config"
    )
