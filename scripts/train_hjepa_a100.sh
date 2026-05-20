#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${repo_root}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${repo_root}/.cache/matplotlib}"

python -m active_matter_jepa.train \
  --config "${repo_root}/configs/vit_h_jepa_multiscale_a100.yaml" \
  --data_root "${DATA_ROOT:-/scratch/${USER}/data/active_matter}" \
  --output_dir "${OUTPUT_DIR:-/scratch/${USER}/am_jepa/h_jepa_multiscale}" \
  --wandb_mode "${WANDB_MODE:-online}"
