# EFD3D Scripts Guide

**最后更新**: 2026-04-13
**版本**: v4.5
**脚本总数**: 2 个

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Current Scripts](#current-scripts)
4. [Dashboard Usage](#dashboard-usage)
5. [Visualizer 3D Module](#visualizer-3d-module)
6. [Removed Scripts](#removed-scripts)
7. [Usage Patterns](#usage-patterns)
8. [Output Reference](#output-reference)

---

## 概述

EFD3D 脚本模块已经过精简，大多数 CLI 工具已集成到交互式 Streamlit 仪表板中，为所有分析任务提供统一接口。

### 当前脚本组织结构

```
scripts/
├── dashboard.py           # Streamlit 仪表板（主入口点）
└── run_ablation.sh        # 消融研究脚本
```

### 主要特性

- **Streamlit 仪表板**: 交互式 Web UI，整合了大部分分析工具
- **自动检测**: 工具自动检测最新模型（当未指定路径时）
- **出版物质量**: 通过笔记本脚本提供 IEEE 标准图形
- **3D 可视化**: 集成在 Streamlit 仪表板中

---

## Quick Start

### Basic Setup

```bash
# Activate environment
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows

# Navigate to project root
cd /home/scnu/Gitee/EFD3D
```

### Recommended Workflow

```bash
# Launch Dashboard (unified interface for all analysis)
streamlit run scripts/dashboard.py
```

The Dashboard provides access to:
- Training log analysis
- Model comparison
- Flow field visualization
- Performance benchmarking
- Stage 1 analytical demos

---

## Current Scripts

### 1. `dashboard.py` - Interactive Streamlit Dashboard

**Purpose**: Interactive web-based visualization and analysis interface

**File**: `scripts/dashboard.py`
**Type**: Main entry point (Streamlit app)

#### Features

- Real-time model loading and inference
- Interactive parameter controls (voltage, time, z-plane)
- Live visualization updates
- Model comparison tools
- Downloadable figures

#### Tab Structure

| Tab | Functionality |
|-----|--------------|
| 🔬 Field Analysis | Flow field visualization (φ, velocity, pressure) |
| ⏱️ Performance | Performance benchmarking and profiling |
| 🔄 Compare | Model comparison tools |
| 📐 Stage 1 | Stage 1 analytical model demos |
| 📊 Training | Training log analysis and visualization |

#### Usage

```bash
# Standard launch (recommended)
streamlit run scripts/dashboard.py

# Custom port and host
streamlit run scripts/dashboard.py --port 8502 --server.address 0.0.0.0

# Headless mode (for remote access)
streamlit run scripts/dashboard.py --server.headless true
```

---

### 2. 3D Visualization

3D visualization functionality is now integrated into the Streamlit Dashboard, providing interactive 3D rendering and analysis capabilities.

---

## Removed Scripts

The following scripts have been removed as their functionality is now available in the Dashboard or moved to the notebooks directory:

| Removed Script | Replacement |
|----------------|------------|
| `benchmark.py` | Dashboard tab: ⏱️ Performance |
| `compare.py` | Dashboard tab: 🔄 Compare |
| `analyze_log.py` | Dashboard tab: 📊 Training |
| `analyze_flow_field.py` | Dashboard tab: 🔬 Field Analysis |
| `stage1_demo.py` | Dashboard tab: 📐 Stage 1 |
| `generate.py` | (Removed - use Dashboard export features) |
| `ieee_figures.py` | (Removed - use Dashboard export features) |
| `verify_parameters.py` | (Simple utility, no replacement) |
| `test_all.py` | `uv run pytest tests/` |
| `cli_utils.py` | (Inlined into scripts) |
| `constants.py` | (Inlined into scripts) |

### Migration

For all removed scripts, use the Dashboard as the primary interface:

```bash
streamlit run scripts/dashboard.py
```

For publication figure generation, use the Dashboard export features (available in the relevant tabs).

---

## Usage Patterns

### Pattern 1: Interactive Analysis

```bash
# Launch Dashboard for all analysis tasks
streamlit run scripts/dashboard.py
```

Navigate to the appropriate tab:
- 🔬 Field Analysis: Flow field visualization
- ⏱️ Performance: Benchmarking
- 🔄 Compare: Model comparison
- 📐 Stage 1: Analytical model demos
- 📊 Training: Training log analysis

### Pattern 2: 3D Visualization

```bash
# Use Dashboard for 3D visualization
streamlit run scripts/dashboard.py
```

Navigate to the "🧊 3D体积视图" tab for interactive 3D visualization.

### Pattern 3: Publication Figure Generation

```bash
# Generate publication figures using Dashboard export features
streamlit run scripts/dashboard.py
# Navigate to the desired tab and use the export/download options
```

### Pattern 4: Training Workflow

```bash
# 1. Train model
uv run train_two_phase.py --config config/v4.5-standard.json

# 2. Analyze via Dashboard
streamlit run scripts/dashboard.py  # Use 📊 Training tab

# 3. Generate publication figures using Dashboard export features
streamlit run scripts/dashboard.py  # Export from desired tabs
```

---

## Output Reference

### Dashboard Outputs

The Dashboard generates interactive visualizations in the browser. You can download figures as PNG from the UI.

### 3D Visualization Output

```
outputs/visualizations/   # visualizer_3d module
├── aperture_3d.png
├── phi_iso_surface.png
└── velocity_vectors_3d.png
```

### Using --output with Dashboard

The Dashboard provides export options in each tab for saving visualizations to custom locations.

---

*For the most current usage, run `streamlit run scripts/dashboard.py --help`.*

**Related Documentation**:
- [Usage Guide](usage.md) - General usage and quick start
- [Visualization Guide](visualization_guide.md) - Detailed visualization options
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [API Reference](../api/README.md) - Programmatic API documentation
- [Project Core Documentation](../../CLAUDE.md) - Developer guide and knowledge base
