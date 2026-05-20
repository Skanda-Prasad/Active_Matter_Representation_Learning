# Analytics And EDA

This directory contains the reusable data analytics code from the original workspace.
Generated CSVs, plots, checkpoints, and HDF5 data are intentionally not committed.

## Main Scripts

- `analyze_active_matter.py`: full local analytics pipeline for HDF5 files, summary tables, plots, and optional baseline regressions.
- `feature_baselines.py`: grouped cross-validation baselines for predicting `alpha` and `zeta` from hand-crafted features.
- `inspect_dataset.py`: dataset inventory and split inspection.
- `inspect_hdf5.py`: low-level HDF5 structure inspection.
- `eda.py`: earlier standalone EDA script kept for reproducibility.
- `utils_io.py`, `utils_plots.py`, `utils_stats.py`: shared analytics helpers.

## Example Commands

Quick EDA pass:

```bash
python analytics/analyze_active_matter.py \
  --data_root /path/to/active_matter \
  --out_dir runs/forensics_quick \
  --quick_mode \
  --run_baselines
```

Feature baselines only:

```bash
python analytics/feature_baselines.py \
  --data_root /path/to/active_matter \
  --out_dir runs/baseline_quick \
  --quick_mode
```

Inspect one dataset root:

```bash
python analytics/inspect_dataset.py --data_root /path/to/active_matter
```

Outputs should go under `runs/` or another ignored local directory.

