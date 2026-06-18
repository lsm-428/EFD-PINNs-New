# Config Directory

## Active Configs

| File | Version | Purpose |
|------|---------|---------|
| `device_calibrated_physics.json` | v7.2 | Current main config, Route 1 calibrated parameters |
| `v4.6-optimized.json` | v4.6 | Optimized training config (unified gradient + Interface annealing) |

## Ablation Configs (`ablation/`)

Single-physics-constraint ablation experiments:

- `no_continuity.json` — Disable continuity equation
- `no_interface.json` — Disable interface tension
- `no_vof.json` — Disable VOF transport
- `single_stage.json` — Single-stage training
- `smaller_network.json` — Smaller network

## Archive (`archive/`)

Historical and deprecated configs, kept for reference only.
