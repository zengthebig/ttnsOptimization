#!/bin/bash
# =====================================================================
# UCI 论文级五数据集基准（README Table 3 超参，单 seed=0）
#   POWER / GAS / HEPMASS / MINIBOONE / BSDS300
#   TTDE(链, 论文 rank) vs TTNSDE(MI 树, 独立 r_ttns)
#
# 用法（在 TTDE-dev 仓库根目录）:
#   mkdir -p logs
#   sbatch simple_ttns_l2/experiments/run_uci_paper.sbatch
#
# 覆盖:
#   DATA_DIR=/path/to/data REPO=/path/to/TTDE-dev sbatch ...
#   QUICK=1 sbatch ...          # 冒烟（极小 steps，非正式结果）
#   R_TTNS=2 sbatch ...         # 全局覆盖树 rank
#   R_TTNS_BSDS300=2 sbatch ... # 单数据集覆盖
#   ARRAY=0,2 sbatch ...        # 只跑部分 array 任务（需改 --array 或用 --array=0,2）
#
# 产物:
#   simple_ttns_l2/reports/uci_ttde_vs_ttns_metrics_paper_<ds>_seed0.json
#   simple_ttns_l2/reports/uci_<ds>_ttde_vs_ttns_bars_paper_<ds>_seed0.png
#   simple_ttns_l2/reports/uci_<ds>_ttde_vs_ttns_params_paper_<ds>_seed0.pkl
#   logs/uci_paper_%A_%a.out|.err
# =====================================================================
#SBATCH --job-name=uci_paper
#SBATCH --output=logs/uci_paper_%A_%a.out
#SBATCH --error=logs/uci_paper_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-4
### ------- 按集群取消注释并填写 -------
### #SBATCH --partition=<你的分区名>
### #SBATCH --account=<你的账号/项目>
### #SBATCH --qos=<你的qos>

set -euo pipefail

DATASETS=(power gas hepmass miniboone bsds300)
DS="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"

# 统一按 GAS/BSDS 最坏情形申请 7 天 / 128G；短任务会提前结束。
echo "[resource] dataset=${DS} requested mem=128G time=7-00:00:00"

# ---- 1. 环境 --------------------------------------------------------
source ~/conda_start.sh
conda activate ttns

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_ENABLE_X64=1
unset PYTHONPATH

# ---- 2. 仓库 --------------------------------------------------------
# 默认：本脚本所在仓库（TTDE-dev）；可用 REPO=... 覆盖
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO:-${DEFAULT_REPO}}"
mkdir -p logs simple_ttns_l2/reports

DATA_DIR="${DATA_DIR:-/home/sbzeng/2_1/research/datasets/data}"
SEED="${SEED:-0}"
OUT_TAG="paper_${DS}_seed${SEED}"

echo "[uci-paper] repo=$(pwd) dataset=${DS} seed=${SEED} data_dir=${DATA_DIR}"
echo "[uci-paper] hostname=$(hostname) date=$(date -Is)"

PY="${PYTHON:-python}"
# conda env 的 python
if command -v python >/dev/null 2>&1; then
  PY=python
fi

CMD=(
  "${PY}" -m simple_ttns_l2.experiments.uci_ttde_vs_ttns
  --preset paper
  --dataset "${DS}"
  --data-dir "${DATA_DIR}"
  --seed "${SEED}"
  --out-tag "${OUT_TAG}"
)

if [[ "${QUICK:-0}" == "1" ]]; then
  CMD+=(--quick)
  echo "[uci-paper] QUICK=1 smoke mode"
fi

echo "[uci-paper] cmd: ${CMD[*]}"
"${CMD[@]}"
echo "[uci-paper] done dataset=${DS} date=$(date -Is)"
