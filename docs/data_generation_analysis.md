# 数据生成逻辑完整梳理

## 文件架构

### pinn_two_phase.py::DataGenerator (3186行)

核心数据生成器，支持两种采样策略：
- **uniform**（默认）：直接在 DataGenerator 中实现
- **physics_based**：委托给 PhysicsBasedSampler

主要方法：
- generate_all_data() (L1146) — 总入口
- target_phi_3d() (L877) — phi 目标值计算
- _sample_point_by_eta() (L1082) — 空间自适应采样
- _center_opening_phi() (L993) — 中心开口模式
- _corner_blob_phi() (L1053) — 四角液滴模式
- sample_continuous_times() — Beta 时间采样
- get_opening_rate() (L851) — eta 计算
- get_contact_angle() (L759) — theta 计算
