#!/bin/bash
# dense_dag_r567 补救方案轻量测试（marginal_l2 + 非负 MLE），目标 <30min
#SBATCH --job-name=dense_r567_remedy
#SBATCH --output=logs/dense_r567_remedy_%j.out
#SBATCH --error=logs/dense_r567_remedy_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:25:00

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

echo "[remedy] host=$(hostname) start=$(date)"
python3 -m simple_ttns_l2.experiments.dense_dag_r567_remedy_tests --seed 0 \
  --skip-r7-baseline --skip-marginal \
  --marginal-json simple_ttns_l2/reports/dense_dag_r567_remedy_marginal_partial.json
echo "[remedy] done=$(date)"
