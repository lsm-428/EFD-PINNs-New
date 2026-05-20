# 🌊 EFD3D Dashboard 快速启动指南

## 🚀 一键启动

```bash
# 启动Dashboard (默认端口8501)
uv run streamlit run scripts/dashboard.py

# 指定端口启动
uv run streamlit run scripts/dashboard.py --server.port 8502
```

打开浏览器访问: **http://localhost:8501**

## 📋 功能速查表

### 📊 2D场分析
- **用途**: 查看任意平面的流场分布
- **操作**: 选择平面 → 设置电压 → 点击"Render Slice"
- **输出**: 速度场、压力场、相场分布

### 🧊 3D体积视图
- **用途**: 3D界面重建和可视化
- **操作**: 设置电压参数 → 调节分辨率 → 点击"Generate 3D Model"
- **输出**: 3D等值面模型 (φ=0.5)

### 📈 瞬态响应
- **用途**: 分析动态电压响应
- **操作**: 选择波形类型 → 设置参数 → 点击"Simulate Trajectory"
- **输出**: 时域响应曲线

### 🩺 物理诊断
- **用途**: 验证物理约束
- **操作**: 设置检查参数 → 点击"Check Volume" 或 "Compute Residual Map"
- **输出**: 质量守恒值、PDE残差图

### 📊 训练输出分析
- **用途**: 分析训练过程和结果
- **操作**: 自动扫描outputs/train目录
- **输出**: 训练曲线、性能指标、模型对比

## ⚙️ 常用参数设置

### 电压参数
- **范围**: -30V 到 +30V
- **典型值**: 0V → 30V (阶跃响应)
- **时间范围**: 0-100ms

### 分辨率设置
- **2D切片**: 64 (快) / 128 (平衡) / 256 (高质)
- **3D体积**: 30-80 (密度)

### 切片位置
- **XY平面**: Z高度 (0-20μm)
- **XZ平面**: Y位置 (0-174μm)
- **YZ平面**: X位置 (0-174μm)

## 🎯 典型使用场景

### 场景1: 阶跃响应分析
1. 切换到 **📈 瞬态响应** 标签
2. 选择 "Step Function"
3. 设置: 延迟=0.1ms, 幅度=30V
4. 点击 "Simulate Trajectory"

### 场景2: 3D界面可视化
1. 切换到 **🧊 3D体积视图** 标签
2. 设置: V_from=0V, V_to=30V, 时间=1ms
3. 分辨率设为40
4. 点击 "Generate 3D Model"

### 场景3: 物理约束验证
1. 切换到 **🩺 物理诊断** 标签
2. 设置检查参数 (时间=0.5ms, 电压=30V)
3. 点击 "Check Volume" 验证质量守恒
4. 点击 "Compute Residual Map" 查看PDE残差

### 场景4: 2D流场分析
1. 切换到 **📊 2D场分析** 标签
2. 选择 "Step Response" 模式
3. 设置: V_prev=0V, V_curr=30V, 时间=0.5ms
4. 选择 "xy (Top View)" 平面
5. 点击 "Render Slice"

## 🔧 故障排除

### 端口被占用
```bash
# 检查端口占用
lsof -ti:8501

# 使用其他端口启动
uv run streamlit run scripts/dashboard.py --server.port 8502
```

### 模型加载失败
- 确认 `outputs/train/` 目录包含 `.pth` 文件
- 检查模型文件完整性
- 尝试使用CPU模式: 在侧边栏选择 "cpu"

### 性能问题
- 降低分辨率设置
- 使用CPU模式 (如果没有GPU)
- 关闭不必要的浏览器标签

## 📁 文件位置

- **Dashboard脚本**: `scripts/dashboard.py`
- **模型文件**: `outputs/train/pinn_*/`
- **配置文件**: `config/v4.5-standard.json`
- **源代码**: `src/dashboard/`

## 📞 获取帮助

### 检查功能完整性
```bash
# 运行功能测试
uv run python simple_dashboard_test.py

# 查看详细报告
cat DASHBOARD_FUNCTIONALITY_SUMMARY.md
```

### 常用命令
```bash
# 查看Streamlit版本
uv run streamlit --version

# 查看可用模型
uv run python -c "from pathlib import Path; print([p.name for p in Path('outputs/train').glob('pinn_*')])"

# 清理Streamlit缓存
rm -rf ~/.streamlit
```

## 🎓 快速学习路径

1. **第1步**: 启动Dashboard，浏览各个标签页
2. **第2步**: 在 **📊 2D场分析** 中尝试不同切片平面
3. **第3步**: 在 **🧊 3D体积视图** 中生成3D模型
4. **第4步**: 在 **📈 瞬态响应** 中分析动态行为
5. **第5步**: 在 **🩺 物理诊断** 中验证物理约束

## 📊 性能预期

| 操作 | 时间 | 说明 |
|------|------|------|
| 模型加载 | <1s | 首次启动时 |
| 2D场预测 | 5-50ms | 取决于分辨率 |
| 3D体积重建 | 0.01-5s | 取决于网格密度 |
| 图表渲染 | 实时 | 交互式 |

---

**版本**: v2.0 (2026年4月12日)
**状态**: ✅ 功能完整，生产就绪
