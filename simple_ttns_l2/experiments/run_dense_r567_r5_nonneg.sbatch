#!/bin/bash
# R5 非负解析链补测（合并已有 remedy 结果）
#SBATCH --job-name=r567_r5_nonneg
#SBATCH --output=logs/dense_r567_r5_nonneg_%j.out
#SBATCH --error=logs/dense_r567_r5_nonneg_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
source ~/conda_start.sh
conda activate ttns
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export JAX_ENABLE_X64=1
unset PYTHONPATH
cd "${REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p logs
echo "[R5 nonneg] start=$(date)"
python3 -m simple_ttns_l2.experiments.dense_dag_r567_remedy_tests \
  --seed 0 --only-r5-nonneg \
  --merge-json simple_ttns_l2/reports/dense_dag_r567_remedy_metrics.json
echo "[R5 nonneg] done=$(date)"
