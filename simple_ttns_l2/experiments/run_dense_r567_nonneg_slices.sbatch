#!/bin/bash
# 生成 R5/R7 nonneg 完整 marginal 切片图（5×20 网格）
#SBATCH --job-name=r567_nonneg_slices
#SBATCH --output=logs/dense_r567_nonneg_slices_%j.out
#SBATCH --error=logs/dense_r567_nonneg_slices_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail
source ~/conda_start.sh
conda activate ttns
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export JAX_ENABLE_X64=1
unset PYTHONPATH

cd "${REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p logs
echo "[nonneg_slices] start=$(date)"
python3 -m simple_ttns_l2.experiments.plot_dense_dag_nonneg_slices --seed 0 --refined-only
echo "[nonneg_slices] done=$(date)"
