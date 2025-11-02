# File: scripts/dev_env.sh
#!/usr/bin/env bash
# Development env bootstrap for tsl_stack

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TSL_BASE="${REPO_ROOT}/nf_auto_runs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"   # 空ならCPU
export TSL_ENABLE_MLFLOW="${TSL_ENABLE_MLFLOW:-0}"

echo "[dev_env] REPO_ROOT=${REPO_ROOT}"
echo "[dev_env] PYTHONPATH=${PYTHONPATH}"
echo "[dev_env] TSL_BASE=${TSL_BASE}"
echo "[dev_env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[dev_env] TSL_ENABLE_MLFLOW=${TSL_ENABLE_MLFLOW}"

# handy shortcuts
alias tsl-help='python -m tsl.cli.tsl --help'
alias tsl-diagnose='python -m tsl.cli.tsl diagnose --base "${TSL_BASE}"'
alias tsl-clean-dry='python -m tsl.cli.tsl workspace-clean --all --dry-run'
alias tsl-clean-all='python -m tsl.cli.tsl workspace-clean --all -y'
alias tsl-run-smoke='python -m tsl.cli.tsl run-auto --data-csv "${REPO_ROOT}/gpu_smoke.csv" --num-samples 1 --save-model'
alias tsl-test='pytest -q'
