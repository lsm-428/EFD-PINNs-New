# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚀 Quick Start Commands

### Training
```bash
# Train with recommended configuration
uv run train_two_phase.py --config config/v4.5-standard.json

# Resume training from checkpoint
uv run train_two_phase.py --config config/v4.5-standard.json --resume_from outputs/train/pinn_YYYYMMDD_HHMMSS/best_model.pth
```

### Evaluation & Visualization
```bash
# Full evaluation (generates dashboard, phi grid, 3D interface, dynamic curves, etc.)
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/

# Evaluate specific checkpoint (best, latest, final, both, or path-to-pth)
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/ --ckpt best

# Compare last 2 trained models
uv run evaluate.py --compare

# Statistical significance test (best vs final model)
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/ --stat-test

# Launch interactive Streamlit dashboard
uv run scripts/dashboard.py

# Run ablation study
uv run scripts/run_ablation.sh

# Launch TensorBoard for training monitoring
tensorboard --logdir outputs/train/pinn_YYYYMMDD_HHMMSS/runs/
```

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/test_pinn_complete.py -v

# Run tests with detailed traceback
uv run pytest tests/ -v --tb=short

# Generate coverage report
uv run pytest tests/ --cov=src --cov-report=html
```

---

## 🏗️ Project Architecture

EFD3D is a Physics-Informed Neural Network (PINN) framework for 3D two-phase flow simulation in electrowetting applications (electronic paper displays).

### Core Architecture
- **Two-Stage Design**: Stage 1 (analytical contact angle model, `EnhancedApertureModel`) + Stage 2 (PINN for full flow field, `TwoPhasePINN`)
- **6D Triad Input**: `[x, y, z, V_from, V_to, t_since]` — enables arbitrary voltage sequence simulation from a single model
- **Physics Constraints**: Navier-Stokes equations + VOF interface tracking + electrowetting forces + mass conservation
- **Progressive Training**: 3-stage curriculum with dynamic loss weight scheduling:
  - **Stage 1** (Geometry, ~1500 epochs): Learn basic phase distribution from analytical contact angle model; active losses: `L_interface`, `L_data`
  - **Stage 2** (Kinematics, ~4000 epochs): Introduce velocity field and mass conservation; adds `L_continuity`, `L_vof`
  - **Stage 3** (Full Physics, ~60000 epochs): Full Navier-Stokes + electrowetting forces; adds `L_ns_momentum`, `L_electrowetting`, `L_volume`

### Source Layout
```
src/
├── models/                       # Neural network models
│   ├── pinn_two_phase.py         # TwoPhasePINN + DataGenerator + Trainer (main module)
│   └── aperture_model.py         # EnhancedApertureModel (Stage 1 analytical model)
├── data/
│   └── physics_sampling.py       # PhysicsBasedSampler: voltage/time/spatial sampling
├── physics/
│   └── constraints.py            # PhysicsConstraints (NS, VOF, continuity, electrowetting)
├── training/
│   ├── scheduler.py              # DynamicPhysicsWeightScheduler (adaptive loss balancing)
│   ├── stabilizer.py             # TrainingStabilizer (NaN recovery, gradient clipping)
│   └── components.py             # Shared training utilities
├── config/
│   ├── __init__.py               # Exports PHYSICS dict, PhysicsConfig, path helpers
│   ├── physics_config.py         # Type-safe physics configuration
│   └── paths.py                  # PROJECT_ROOT, CONFIG_PATH, OUTPUT_DIR (env-overridable)
├── dashboard/                    # Streamlit dashboard engine (16 modules)
│   ├── app.py                    # Main dashboard app
│   ├── datastore.py              # Shared data state
│   ├── model_manager.py          # Model loading/caching
│   ├── inference.py              # Inference engine
│   ├── plotter.py                # Visualization routines
│   ├── benchmark_panel.py        # Performance benchmarking
│   ├── compare_panel.py          # Multi-model comparison
│   ├── stage1_panel.py           # Stage 1 diagnostics
│   ├── training_output_analyzer.py # Training log analysis
│   ├── reports/                  # HTML/Markdown report generation
│   └── monitor/                  # Real-time training log watching
├── predictors/
│   ├── hybrid_predictor.py       # Stage 1 + Stage 2 integration
│   └── pinn_aperture.py          # PINN-based aperture prediction
├── solvers/
│   └── flow_solver.py            # Traditional CFD solver (for verification)
└── utils/
    ├── model_utils.py            # Checkpoint loading with architecture mismatch handling
    └── logging_config.py         # Unified logging (EFD_LOG_LEVEL env var)
```

### Experimental Modules (`experimental/`)
- **levelset/** — Level Set method alternative to VOF for interface tracking
- **end_to_end/** — End-to-end training without Stage 1 dependency
- **lstm_pinn/** — LSTM + PINN hybrid for temporal dynamics
- **test/** — Experimental tests (not part of main test suite)

### Entry Points
- `train_two_phase.py` — delegates to `src.models.pinn_two_phase.main()`
- `evaluate.py` — `PINNEvaluator` class: 7 visualization types + statistical comparison
- `scripts/dashboard.py` — lightweight launcher for `src.dashboard.app`

### Key Classes
- `TwoPhasePINN` (nn.Module) — dual-branch MLP: velocity branch + phase branch; key methods: `forward()`, `forward_triplet()`
- `DataGenerator` — generates training points (interior, boundary, interface, collocation); houses `generate_all_data()`
- `PhysicsLoss` — orchestrates computation of all physics loss components
- `Trainer` — data generation, physics loss, optimization loop, checkpointing; key method: `train()`
- `PhysicsConstraints` — computes NS residuals, VOF transport, continuity, electrowetting forces, volume conservation
- `PhysicsBasedSampler` — physics-informed voltage/time/spatial sampling in `src/data/physics_sampling.py`
- `DynamicPhysicsWeightScheduler` — ramps physics loss weights during training stages
- `TrainingStabilizer` — detects NaN, clips gradients, restores stable checkpoints

---

## ⚙️ Configuration System

### Environment Variables
| Variable | Effect | Default |
|---|---|---|
| `EFD_CONFIG_PATH` | Override physics config path | `config/device_calibrated_physics.json` |
| `EFD_OUTPUT_DIR` | Override output directory | `outputs/` |
| `EFD_LOG_LEVEL` | Logging level | `INFO` |
| `EFD_LOG_FILE` | Log file path | (stderr only) |
| `EFD_LOG_VERBOSE` | Verbose format (`1/0, true/false`) | `0` |

### Two config types (don't confuse them)

**Physics defaults** — loaded automatically from `config/device_calibrated_physics.json` (set by `src/config/paths.py` as `DEFAULT_CONFIG_PATH`). Override with `EFD_CONFIG_PATH` env var. Contains device geometry, material properties, initial conditions.

**Training config** — passed explicitly via `--config`. Primary: `config/v4.5-standard.json`. Contains epochs, learning rate, batch sizes, physics loss weights, training stages.

```python
from src.config import PHYSICS, get_config_path

# Physics parameters (from device_calibrated_physics.json)
Lx = PHYSICS["Lx"]          # Pixel width: 174μm
dielectric_thickness = PHYSICS["dielectric_thickness"]  # 400nm
gamma = PHYSICS["gamma"]    # Surface tension: 0.015 N/m
theta0 = PHYSICS["theta0"]  # Initial contact angle: 120°

# Path helpers
from src.config.paths import PROJECT_ROOT, OUTPUT_DIR, get_output_dir
```

### Available config files
```
config/
├── v4.5-standard.json              # Recommended training config (proven convergence)
├── v4.5-physics-sampling.json      # Physics-informed sampling variant
├── device_calibrated_physics.json  # Default physics calibration
└── ablation/                       # Ablation study configs
    ├── no_continuity.json
    ├── no_interface.json
    ├── no_vof.json
    ├── single_stage.json
    └── smaller_network.json
```

---

## 🧪 Testing

### Test Organization
```
tests/
├── test_pinn_complete.py              # End-to-end PINN training pipeline
├── test_physics_sanity.py             # Physics constraint validation
├── test_vof_transport.py              # VOF advection equation verification
├── test_vof_3d.py                     # 3D VOF implementation
├── test_vof_sensitivity.py            # VOF parameter sensitivity
├── test_hybrid_predictor.py           # Stage 1 + Stage 2 integration
├── test_enhanced_aperture_properties.py # Stage 1 analytical model checks
├── test_curvature_computation.py      # Interface curvature calculations
├── test_dynamic_weights.py            # Loss weight scheduler behavior
├── test_flow_solver_properties.py     # CFD solver verification
├── test_model_dimensions.py           # Model architecture validation
├── test_two_phase_data_generator.py   # Training data generation
├── test_3d_visualization_properties.py # 3D visualization checks
├── test_scripts_framework.py          # Script entry point tests
└── test_code_changes.py               # Code modification tracking
```

### Test Patterns
- **Physics validation**: governing equation residuals < 1e-3
- **Conservation laws**: mass/volume conservation error < 1%
- **Boundary conditions**: no-slip walls, interface continuity
- **Stage integration**: Stage 1 → Stage 2 compatibility

---

## 📊 Key Workflows

### Training Pipeline
1. Select training config in `config/`
2. Run `uv run train_two_phase.py --config config/v4.5-standard.json`
3. Monitor: TensorBoard (`tensorboard --logdir outputs/train/pinn_*/runs/`) or dashboard
4. Evaluate: `uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/`
5. Verify: `uv run pytest tests/test_pinn_complete.py -v`

### Debugging
- **Training instability**: Check `TrainingStabilizer` logs for NaN recovery events
- **Physics violations**: Run `test_physics_sanity.py` for equation-level diagnostics
- **Interface quality**: Validate VOF transport with `test_vof_transport.py`
- **Performance**: Use dashboard benchmark panel or TensorBoard GPU metrics

### Model Development Flow
- Physics changes → `src/physics/constraints.py`
- Architecture changes → `src/models/pinn_two_phase.py` (TwoPhasePINN class)
- Training logic → `src/training/` (scheduler, stabilizer, components)
- Configuration → update relevant JSON in `config/`
- Tests → add/modify in `tests/`

---

## 🛠️ Development Environment

- **Package manager**: `uv` (see `pyproject.toml`)
- **Python**: 3.12–3.13
- **GPU**: CUDA 11.8, PyTorch 2.7.1
- **Linting**: `ruff` (line-length 88)
- **Formatting**: `black` (line-length 88)
- **Testing**: `pytest` + `hypothesis`
- **Dependency groups**: `dev`, `testing`, `monitoring`, `web`, `full`, `dashboard`

### Output Structure
```
outputs/train/pinn_YYYYMMDD_HHMMSS/
├── best_model.pth               # Best model weights
├── best_model_epoch_XXXXX.pth   # Epoch-specific checkpoints
├── final_model.pth              # Final epoch weights
├── training.log                 # Training progress log
├── runs/                        # TensorBoard event files
├── pro_dashboard_best.png       # 4-panel evaluation dashboard
├── phi_grid_evolution_best.png  # 7×6 voltage/time grid
├── interface_3d_steady_best.png # 3D interface isosurface
├── dynamic_curves_best.png      # Dynamic response curves
├── response_times_best.png      # Response time analysis
├── mass_conservation_best.png   # Volume conservation check
├── z_profile_best.png           # Vertical phase profile
├── training_curve.png           # Loss curves
└── config.json                  # Training configuration snapshot
```

---

## 📚 Documentation References

Key docs for deeper context:
- **Quick Start**: `docs/guides/quickstart.md`
- **Physics & Device Guide**: `docs/guides/physics_and_device_guide.md`
- **Configuration Guide**: `docs/guides/configuration_guide.md`
- **Training Guide**: `docs/guides/training_guide.md`
- **API Reference**: `docs/api/README.md`
- **Architecture**: `docs/architecture/system_design.md`
- **Project Overview**: `docs/PROJECT_OVERVIEW.md`

---

## ⌨️ Slash Commands

Quick shortcuts for the most frequent workflows. All commands use `uv run`.

| Command | Action |
|---|---|
| `/train [config]` | Train with config (default: `config/v4.5-standard.json`) |
| `/eval <target>` | Evaluate: `best`, `latest`, `final`, `compare`, or a checkpoint path |
| `/test [scope]` | Run tests: `quick` (skip slow), `all`, or specify a test file |
| `/lint` | Check formatting (`ruff check && black --check`) |
| `/fmt` | Auto-fix formatting (`ruff check --fix && black`) |
| `/monitor` | Launch Streamlit dashboard |
| `/validate-config` | Check config files for missing keys |
| `/clean-outputs` | List or remove stale training output directories |

---

*Last updated: 2026-05-19 | Version: v4.5*
