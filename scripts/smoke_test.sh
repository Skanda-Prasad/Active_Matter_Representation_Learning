#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${repo_root}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${repo_root}/.cache/matplotlib}"

python -m active_matter_jepa.train \
  --config "${repo_root}/configs/local/h_jepa_multiscale_cpu_smoke.yaml" \
  --data_root "${DATA_ROOT:-${repo_root}/data}" \
  --output_dir "${OUTPUT_DIR:-${repo_root}/runs/h_jepa_multiscale_cpu_smoke}" \
  --wandb_mode disabled \
  --smoke_test \
  --no-resume
