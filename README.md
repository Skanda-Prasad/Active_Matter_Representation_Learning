# Active Matter JEPA

Physics-aware representation learning for active-matter simulations using spatial ViT-JEPA and a three-level hierarchical JEPA.

This repo is organized around the cleaned training code from `train_plz_work (1).py` and the production H-JEPA config from `vit_h_jepa_multiscale_a100 (1).yaml`. The older experiment folders in the workspace are treated as background and are ignored by default for GitHub.

## What Is Included

- `active_matter_jepa.train`: importable training entry point with dataset loading, models, VICReg/SigReg losses, checkpointing, validation, and frozen-probe evaluation.
- `configs/vit_h_jepa_multiscale_a100.yaml`: production three-level H-JEPA configuration for A100 training.
- `configs/local/`: small CPU smoke configs for validating the code path before launching a cluster job.
- `analytics/`: reusable EDA, dataset inspection, plotting, and feature-baseline scripts.
- `notebooks/`: exploratory notebooks from the original analysis workspace.
- `reports/`: final markdown data analytics report.
- `scripts/`: shell entry points for local smoke tests and A100 runs.
- `tests/`: lightweight import/model-construction checks.

## Repository Layout

```text
.
|-- .github/workflows/ci.yml
|-- analytics/
|   |-- analyze_active_matter.py
|   |-- feature_baselines.py
|   |-- inspect_dataset.py
|   |-- inspect_hdf5.py
|   |-- utils_io.py
|   |-- utils_plots.py
|   `-- utils_stats.py
|-- configs/
|   |-- vit_h_jepa_multiscale_a100.yaml
|   `-- local/
|       |-- h_jepa_multiscale_cpu_smoke.yaml
|       `-- physics_vit_jepa_cpu_smoke.yaml
|-- notebooks/
|-- reports/
|-- scripts/
|   |-- smoke_test.sh
|   `-- train_hjepa_a100.sh
|-- src/
|   `-- active_matter_jepa/
|       |-- __init__.py
|       |-- __main__.py
|       `-- train.py
|-- tests/
|   `-- test_package_import.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Setup

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

For CUDA clusters, install the PyTorch build that matches the node image if your environment does not already provide it.

## Data Layout

Point `--data_root` at either a flat directory of HDF5 files or a split directory:

```text
data_root/
|-- train/*.hdf5
|-- valid/*.hdf5
`-- test/*.hdf5
```

The loader looks for concentration, velocity, `D`, and `E` fields, plus scalar `alpha` and `zeta` values. It supports the key layouts used in the current active-matter files, including `t0_fields/concentration`, `t1_fields/velocity`, `t2_fields/D`, and `t2_fields/E`.

## Run

Local smoke test:

```bash
bash scripts/smoke_test.sh
```

Production H-JEPA run:

```bash
DATA_ROOT=/scratch/$USER/data/active_matter \
OUTPUT_DIR=/scratch/$USER/am_jepa/h_jepa_multiscale \
bash scripts/train_hjepa_a100.sh
```

Direct Python entry point:

```bash
am-jepa-train \
  --config configs/vit_h_jepa_multiscale_a100.yaml \
  --data_root /scratch/$USER/data/active_matter \
  --output_dir /scratch/$USER/am_jepa/h_jepa_multiscale
```

Two-GPU launch with PyTorch DDP:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m active_matter_jepa.train \
  --config configs/vit_h_jepa_multiscale_a100.yaml \
  --data_root /scratch/$USER/data/active_matter \
  --output_dir /scratch/$USER/am_jepa/h_jepa_multiscale
```

Add `--wandb_mode disabled` for offline or no-W&B runs. Add `--no-resume` when intentionally changing architecture or training config.

## Analytics

Run a quick EDA pass:

```bash
python analytics/analyze_active_matter.py \
  --data_root /path/to/active_matter \
  --out_dir runs/forensics_quick \
  --quick_mode \
  --run_baselines
```

Run hand-crafted feature baselines:

```bash
python analytics/feature_baselines.py \
  --data_root /path/to/active_matter \
  --out_dir runs/baseline_quick \
  --quick_mode
```

The final markdown report is in `reports/active_matter_data_analytics_report.md`. It summarizes the original workspace outputs; generated image/table artifacts are not committed by default.

## Outputs

Each run writes to `output_dir`:

- `config_resolved.json`: fully resolved runtime config.
- `trajectory_split.csv` and `window_index.csv`: generated train/valid/test sampling metadata.
- `train_channel_stats.npz`: normalization statistics.
- `training_history.csv`: epoch-level metrics.
- `checkpoints/best.ckpt` and `checkpoints/last.ckpt`.
- `frozen_probe_results.csv` and `representation_diagnostics.csv`.
- `final_summary.json`: run summary for downstream reporting.

## Development Checks

```bash
python -m compileall src
pytest
```

The `.gitignore` excludes local data, checkpoints, W&B runs, generated analysis outputs, and legacy scratch folders so `git add .` stays focused on the package, configs, scripts, tests, and docs.

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the same compile and test checks on pushes and pull requests.
