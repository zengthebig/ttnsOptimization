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
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
### ------- 按集群取消注释并填写 -------
### #SBATCH --partition=<你的分区名>
### #SBATCH --account=<你的账号/项目>
### #SBATCH --qos=<你的qos>

set -euo pipefail

DATASETS=(power gas hepmass miniboone bsds300)
DS="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"

# 资源说明：本集群 QOS 有 MaxCpuMinutes 限制，7天×16核会被拒。
# 默认 2天×8核×64G；GAS/BSDS 若超时需提高 time 或续跑（见报告）。
echo "[resource] dataset=${DS} cpus=8 mem=64G time=2-00:00:00"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_ENABLE_X64=1
unset PYTHONPATH

# ---- 仓库 --------------------------------------------------------
# sbatch 会把脚本拷到 spool，不可用 BASH_SOURCE 反推仓库根。
# 必须在仓库根目录提交：cd /path/to/TTDE-dev && sbatch ...
cd "${REPO:-${SLURM_SUBMIT_DIR}}"
mkdir -p logs simple_ttns_l2/reports
pwd
ls -la simple_ttns_l2/experiments/uci_ttde_vs_ttns.py

DATA_DIR="${DATA_DIR:-/home/sbzeng/2_1/research/datasets/data}"
SEED="${SEED:-0}"
OUT_TAG="paper_${DS}_seed${SEED}"

echo "[uci-paper] repo=$(pwd) dataset=${DS} seed=${SEED} data_dir=${DATA_DIR}"
echo "[uci-paper] hostname=$(hostname) date=$(date -Is)"

# 优先用 conda env 绝对路径，避免计算节点 module/conda.sh 异常
PY="${PYTHON:-/home/sbzeng/.conda/envs/ttns/bin/python}"
if [[ ! -x "${PY}" ]]; then
  source ~/conda_start.sh || true
  conda activate ttns || true
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
