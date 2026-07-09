#!/bin/bash
# =====================================================================
# dense_dag_r567_three_way 的 SLURM 提交脚本
#   跨簇 DAG(100节点) × 分层 TTNS(R5/R7) × immediate 分块
# 用法:
#   sbatch simple_ttns_l2/experiments/run_dense_r567.sbatch
#   # 覆盖 seeds:  SEEDS=0,1,2,3,4 sbatch ... run_dense_r567.sbatch
#   # 冒烟:        QUICK=1 SEEDS=0 sbatch ... run_dense_r567.sbatch
# 产物:
#   simple_ttns_l2/reports/dense_dag_r567_report_zh.md   (中文报告, 每 seed 增量写)
#   simple_ttns_l2/reports/dense_dag_r567_three_way_metrics.json
# =====================================================================
#SBATCH --job-name=dense_r567
#SBATCH --output=logs/dense_r567_%j.out
#SBATCH --error=logs/dense_r567_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # JAX/XLA 是 CPU 多线程, 16 核较合适(可调 8~32)
#SBATCH --mem=32G                   # 实测占用不高, 32G 富余
#SBATCH --time=02:00:00             # 3 seed 估 ~15-20 分钟, 给 2 小时缓冲
### ------- 下面几行按你的集群取消注释并填写 -------
### #SBATCH --partition=<你的分区名>      # 例: cpu / normal / batch
### #SBATCH --account=<你的账号/项目>     # 若集群要求 account
### #SBATCH --qos=<你的qos>              # 若集群要求 qos

set -euo pipefail

# ---- 1. 环境激活 (★ 按你的集群改这一段 ★) --------------------------
# 如果用 module:
#   module purge
#   module load python/3.11        # 或集群上带 jax 的 python
# 如果用 conda:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate <你带 jax 的环境名>
# 什么都不配则用系统默认 python3(需自带 jax + numpy + 可选 matplotlib)
source ~/conda_start.sh
conda activate ttns
# ---- 2. 线程数与 SLURM 分配一致, 避免超额占核 --------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_ENABLE_X64=1            # 项目要求 float64 (脚本内也会设, 双保险)
unset PYTHONPATH                   # 等价于项目约定的 `env -u PYTHONPATH`

# ---- 3. 进仓库 + 切分支 + 拉最新 ---------------------------------
#   REPO 默认取 sbatch 提交时所在目录; 也可 REPO=/path/to/repo sbatch ...
cd "${REPO:-$SLURM_SUBMIT_DIR}"
# ★ 若计算节点无外网, 把下面 3 行 git 操作改到登录节点先做, 这里删掉 ★
git fetch origin
git checkout worktree-dense-r567-compare
git pull --ff-only origin worktree-dense-r567-compare

mkdir -p logs

# ---- 4. 运行 ----------------------------------------------------
SEEDS="${SEEDS:-0,1,2}"
EXTRA=""
[ "${QUICK:-0}" = "1" ] && EXTRA="--quick"
echo "[dense_r567] host=$(hostname) cpus=${OMP_NUM_THREADS} seeds=${SEEDS} extra='${EXTRA}'"
echo "[dense_r567] start: $(date)"

srun python3 -m simple_ttns_l2.experiments.dense_dag_r567_three_way --seeds "${SEEDS}" ${EXTRA}

echo "[dense_r567] done: $(date)"
echo "[dense_r567] report -> simple_ttns_l2/reports/dense_dag_r567_report_zh.md"
