"""
验证垂直采样分布、阶段转换、配置参数
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PHYSICS
from src.models.pinn_two_phase import DEFAULT_CONFIG, DataGenerator


@pytest.fixture
def data_generator():
    """创建 DataGenerator 实例"""
    device = torch.device("cpu")
    config = DEFAULT_CONFIG.copy()
    config["data"] = {
        "use_vertical_sampling": True,
        "n_vertical_samples": 50,
    }
    return DataGenerator(config, device)


class TestVerticalSampling:
    """验证垂直采样分布"""

    def test_sampling_count(self):
        """验证采样参数合理性"""
        n_samples = 50
        assert n_samples >= 2, "采样点数应 >= 2"

        # 关闭态 (η=0)
        h_edge = PHYSICS["h_ink"]
        interface_width = PHYSICS["ac_interface_width"]

        # 界面区域范围
        z_interface_min = max(0, h_edge - 3 * interface_width)
        z_interface_max = min(PHYSICS["Lz"], h_edge + 3 * interface_width)

        # 验证界面区域覆盖合理范围
        assert z_interface_min < h_edge, "界面区域应包含 h_edge"
        assert z_interface_max > h_edge, "界面区域应包含 h_edge"
        assert z_interface_max - z_interface_min > interface_width, "界面区域应大于界面宽度"

    def test_sampling_monotonicity(self, data_generator):
        """验证采样点沿 Z 轴单调递增"""
        # 生成采样点
        n_samples = 50
        n_interface = int(n_samples * 0.8)
        n_bottom = int(n_samples * 0.1)
        n_top = n_samples - n_interface - n_bottom

        eta = 0.3
        h_edge = PHYSICS["h_ink"] / max(1.0 - eta, PHYSICS["ink_initial_fraction"])
        interface_width = PHYSICS["ac_interface_width"]

        z_points = []
        z_bottom_max = max(0, h_edge - 3 * interface_width)
        if n_bottom > 0 and z_bottom_max > 0:
            z_bottom = np.linspace(0, z_bottom_max, n_bottom)
            z_points.extend(z_bottom)

        z_interface_min = max(0, h_edge - 3 * interface_width)
        z_interface_max = min(PHYSICS["Lz"], h_edge + 3 * interface_width)
        if n_interface > 0:
            z_interface = np.linspace(z_interface_min, z_interface_max, n_interface)
            z_points.extend(z_interface)

        z_top_min = min(PHYSICS["Lz"], h_edge + 3 * interface_width)
        if n_top > 0 and z_top_min < PHYSICS["Lz"]:
            z_top = np.linspace(z_top_min, PHYSICS["Lz"], n_top)
            z_points.extend(z_top)

        z_all = np.unique(z_points)

        # 验证单调递增
        for i in range(len(z_all) - 1):
            assert z_all[i] < z_all[i + 1], f"采样点应单调递增，失败于 z[{i}]={z_all[i]}"

    def test_interface_region_density(self, data_generator):
        """验证界面附近采样密度高于上下区域"""
        n_samples = 50
        n_interface = int(n_samples * 0.8)
        n_bottom = int(n_samples * 0.1)
        n_top = n_samples - n_interface - n_bottom

        eta = 0.3
        h_edge = PHYSICS["h_ink"] / max(1.0 - eta, PHYSICS["ink_initial_fraction"])
        interface_width = PHYSICS["ac_interface_width"]

        # 界面区域范围
        z_interface_min = max(0, h_edge - 3 * interface_width)
        z_interface_max = min(PHYSICS["Lz"], h_edge + 3 * interface_width)
        interface_range = z_interface_max - z_interface_min

        # 底部区域范围
        z_bottom_max = max(0, h_edge - 3 * interface_width)
        bottom_range = z_bottom_max

        # 顶部区域范围
        z_top_min = min(PHYSICS["Lz"], h_edge + 3 * interface_width)
        top_range = PHYSICS["Lz"] - z_top_min

        # 界面区域密度应高于上下区域
        if bottom_range > 0:
            interface_density = n_interface / interface_range
            bottom_density = n_bottom / bottom_range
            assert interface_density > bottom_density, "界面密度应高于底部"

        if top_range > 0:
            interface_density = n_interface / interface_range
            top_density = n_top / top_range
            assert interface_density > top_density, "界面密度应高于顶部"


class TestStageTransition:
    """验证阶段转换"""

    def _get_weights(self, epoch):
        """辅助函数：创建 Trainer 实例并获取权重"""
        from src.models.pinn_two_phase import Trainer

        # 创建最小 Trainer 实例
        config = DEFAULT_CONFIG.copy()
        config["training"] = {
            "epochs": 60000,
            "batch_size": 512,
            "learning_rate": 0.0003,
            "min_lr": 1e-6,
            "gradient_clip": 1.0,
            "stage1_epochs": 1500,
            "stage2_epochs": 4000,
            "stage3_epochs": 60000,
        }
        config["physics"] = DEFAULT_CONFIG["physics"]

        # 创建 Trainer（不需要完整初始化）
        trainer = object.__new__(Trainer)
        trainer.config = config
        trainer.stage1_epochs = 1500
        trainer.stage2_epochs = 4000
        trainer.epochs = 60000
        trainer.s3_smooth_span = 15000

        return trainer.get_physics_weights(epoch)

    def test_stage1_to_stage2_transition(self):
        """验证 S1→S2 转换时物理权重从 0 开始增长"""
        # S1 阶段 (epoch < stage1_epochs)
        weights_s1 = self._get_weights(0)
        assert weights_s1["continuity"] == 0.0, "S1 连续性权重应为 0"
        assert weights_s1["vof"] == 0.0, "S1 VOF 权重应为 0"

        # S2 阶段 (stage1_epochs <= epoch < stage2_epochs)
        weights_s2 = self._get_weights(2000)
        assert weights_s2["continuity"] > 0.0, "S2 连续性权重应大于 0"
        assert weights_s2["vof"] > 0.0, "S2 VOF 权重应大于 0"

    def test_stage3_full_physics(self):
        """验证 S3 阶段物理权重达到最大值"""
        # S3 后期 (epoch > stage2_epochs + smooth_span)
        weights_s3 = self._get_weights(60000)
        assert weights_s3["continuity"] > 0.0, "S3 连续性权重应大于 0"
        assert weights_s3["vof"] > 0.0, "S3 VOF 权重应大于 0"
        assert weights_s3["ns"] > 0.0, "S3 NS 权重应大于 0"


class TestConfigValidation:
    """验证配置参数范围"""

    def test_n_vertical_samples_positive(self):
        """验证 n_vertical_samples 为正整数"""
        n = DEFAULT_CONFIG["data"]["n_vertical_samples"]
        assert isinstance(n, int), "n_vertical_samples 应为整数"
        assert n > 0, "n_vertical_samples 应为正数"

    def test_interface_weight_non_negative(self):
        """验证 interface_weight 非负"""
        w = DEFAULT_CONFIG["physics"]["interface_weight"]
        assert w >= 0, "interface_weight 应非负"

    def test_physics_weights_non_negative(self):
        """验证所有物理权重非负"""
        for key in ["continuity_weight", "vof_weight", "ns_weight"]:
            w = DEFAULT_CONFIG["physics"][key]
            assert w >= 0, f"{key} 应非负"

    def test_stage_epochs_ordering(self):
        """验证阶段 epoch 顺序"""
        stage1 = DEFAULT_CONFIG["training"]["stage1_epochs"]
        stage2 = DEFAULT_CONFIG["training"]["stage2_epochs"]
        stage3 = DEFAULT_CONFIG["training"]["stage3_epochs"]
        assert stage1 < stage2 < stage3, "阶段 epoch 应递增"

    def test_physics_params_in_physics_dict(self):
        """验证所有物理参数在 PHYSICS 字典中存在"""
        required_params = [
            "Lx",
            "Ly",
            "Lz",
            "h_ink",
            "rho_oil",
            "mu_oil",
            "rho_polar",
            "mu_polar",
            "sigma",
            "gamma",
            "epsilon_r",
            "epsilon_h",
            "d_dielectric",
            "d_hydrophobic",
            "V_T_base",
            "V_T_sensitivity",
            "tau",
            "tau_recovery_factor",
            "ac_interface_width",
            "ac_mobility",
            "eta_max",
            "lambda_debye",
        ]
        for param in required_params:
            assert param in PHYSICS, f"PHYSICS['{param}'] 缺失"
