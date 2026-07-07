#!/usr/bin/env bash
# =============================================================================
# run_full_scale_study.sh — 分层 DAG × TTNS 全因子大规模研究(编排脚本)
#
# 目的:把本轮零散验证的所有维度并成一个系统研究——延迟分布 × R6网格分辨率 ×
#      预算 × 图规模 × 方法 × 多seed,统一指标(joint_LL / oracle gap / corr_fro /
#      边缘 W1 + ISE峰形误差)。
#
# !!! 本脚本仅供大内存机器运行,不要在开发机跑(会 OOM / swap 抖动)。
#     设计为可断点续跑(每个 run 独立 checkpoint,重跑自动跳过已完成)。
#
# 依赖的 python 参数化入口(需先小改现有脚本,加 argparse/env 读取):
#   simple_ttns_l2/experiments/full_scale_runner.py   —— 单点 run(读下列 FS_* 环境变量)
#   现有 budget_sweep_layered.py / plot_slice_refined.py 的 CFG 逻辑可复用进该 runner。
#   单点 run 内部:构图→采数据→拟合 L0→按 METHODS 跑 R5/R6/R7(+基线)→算全指标→落 JSON。
#
# 用法:
#   RESULTS_DIR=/data/ttns_study bash run_full_scale_study.sh            # 全量
#   PHASES="1 2" bash run_full_scale_study.sh                            # 只跑指定阶段
#   DRY_RUN=1 bash run_full_scale_study.sh                               # 只打印将执行的 run,不跑
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- 全局配置
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/simple_ttns_l2/reports/full_scale}"
LOG_DIR="$RESULTS_DIR/logs"
CKPT_DIR="$RESULTS_DIR/ckpt"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CKPT_DIR"

MAX_PARALLEL="${MAX_PARALLEL:-1}"      # 并发 run 数(JAX+大网格吃内存,谨慎>1)
SEEDS="${SEEDS:-0 1 2 3 4}"            # 结论点 seed 数(≥5)
PHASES="${PHASES:-1 2 3 4 5 6}"        # 要跑的阶段
DRY_RUN="${DRY_RUN:-0}"
PY="env -u PYTHONPATH python3 -u -m simple_ttns_l2.experiments.full_scale_runner"

cd "$REPO_ROOT"

# ---------------------------------------------------------------- 单点 run 封装
# 每个 run 由一组 FS_* 环境变量唯一确定 → 派生 run_id → 独立日志+checkpoint+跳过。
run_one() {
  local tag="$1"; shift
  # 其余参数形如 KEY=VAL,导出为 FS_KEY 供 runner 读取
  local kvs=("$@")
  local run_id="$tag"
  for kv in "${kvs[@]}"; do run_id="${run_id}__${kv//=/-}"; done
  run_id="${run_id// /_}"
  local out_json="$CKPT_DIR/${run_id}.json"
  local log_file="$LOG_DIR/${run_id}.log"

  if [[ -f "$out_json" ]]; then
    echo "[skip] $run_id (已完成)"; return 0
  fi
  # 组装环境变量
  local envs=()
  for kv in "${kvs[@]}"; do envs+=("FS_${kv%%=*}=${kv#*=}"); done
  envs+=("FS_OUT_JSON=$out_json")

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry] ${envs[*]} $PY"; return 0
  fi
  echo "[run] $run_id → $log_file"
  # shellcheck disable=SC2086
  env "${envs[@]}" $PY > "$log_file" 2>&1 || {
    echo "[FAIL] $run_id (见 $log_file)"; return 1; }
}

# 简单并发闸门(最多 MAX_PARALLEL 个后台 run)
wait_slot() { while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do sleep 5; done; }

# ---------------------------------------------------------------- 阶段 1: 延迟分布轴
# 固定中等图(7层[3,3])+中预算, 只换延迟分布, R5/R6/R7 三方 + 基线。
# 看重尾程度如何改变方法排名(本轮已见 log-skew-normal 放大 R5 短板)。
phase1() {
  echo "=== Phase 1: 延迟分布轴 ==="
  local DELAYS=(uniform logskewnorm lognormal gamma)
  for d in "${DELAYS[@]}"; do
    wait_slot
    run_one "p1_delay" \
      DELAY="$d" N_LAYERS=7 CLUSTERS="3,3" RANK=16 M=24 N_FIT=20000 N_S_JOINT=44 \
      METHODS="R5,R6,R7,baselines" SEED=0 &
  done
  wait
}

# ---------------------------------------------------------------- 阶段 2: 预算轴(每延迟)
# rank × m × n_fit 阶梯(逐旋钮抬升)。均匀 + log-skew-normal 各一遍。
phase2() {
  echo "=== Phase 2: 预算轴 ==="
  local DELAYS=(uniform logskewnorm)
  local RANKS=(8 16 24 32)
  local MS=(24 48)
  local NFITS=(20000 40000 80000)
  for d in "${DELAYS[@]}"; do
    for r in "${RANKS[@]}"; do
      wait_slot
      run_one "p2_rank" DELAY="$d" N_LAYERS=7 CLUSTERS="3,3" RANK="$r" M=24 N_FIT=20000 \
        N_S_JOINT=44 METHODS="R5,R6,R7" SEED=0 &
    done
    for m in "${MS[@]}"; do
      wait_slot
      run_one "p2_m" DELAY="$d" N_LAYERS=7 CLUSTERS="3,3" RANK=16 M="$m" N_FIT=20000 \
        N_S_JOINT=44 METHODS="R5,R6,R7" SEED=0 &
    done
    for nf in "${NFITS[@]}"; do
      wait_slot
      run_one "p2_nfit" DELAY="$d" N_LAYERS=7 CLUSTERS="3,3" RANK=8 M=24 N_FIT="$nf" \
        N_S_JOINT=44 METHODS="R7" SEED=0 &      # n_fit 只影响 R7
    done
  done
  wait
}

# ---------------------------------------------------------------- 阶段 3: R6 网格分辨率轴(关键)
# 直接验证"R6 边缘峰形差是粗 G^K 网格抹峰"假设: n_s_joint 阶梯,看 R6 边缘 ISE 是否回落。
phase3() {
  echo "=== Phase 3: R6 网格分辨率轴(验证抹峰假设) ==="
  local DELAYS=(uniform logskewnorm)
  local GRIDS=(36 44 60 80)
  for d in "${DELAYS[@]}"; do
    for g in "${GRIDS[@]}"; do
      wait_slot
      run_one "p3_nsjoint" DELAY="$d" N_LAYERS=7 CLUSTERS="3,3" RANK=16 M=24 N_FIT=20000 \
        N_S_JOINT="$g" METHODS="R6" SEED=0 &
    done
  done
  wait
}

# ---------------------------------------------------------------- 阶段 4: 图规模轴
# 层数 × 簇大小 —— 深度退化 + 大块(注意 R6 的 G^K 仅小块 K<=4 可行)。
phase4() {
  echo "=== Phase 4: 图规模轴 ==="
  local LAYERS=(5 7 10 14)
  local CLUS=("3,3" "4,4" "5,5,5")   # K=3/4/5; K>=5 时 R6 跳过(G^K 爆),仅 R5/R7
  local d=logskewnorm
  for L in "${LAYERS[@]}"; do
    for c in "${CLUS[@]}"; do
      local kmax; kmax=$(echo "$c" | tr ',' '\n' | sort -n | tail -1)
      local methods="R5,R7"; (( kmax <= 4 )) && methods="R5,R6,R7"
      wait_slot
      run_one "p4_graph" DELAY="$d" N_LAYERS="$L" CLUSTERS="$c" RANK=16 M=24 N_FIT=20000 \
        N_S_JOINT=44 METHODS="$methods" SEED=0 &
    done
  done
  wait
}

# ---------------------------------------------------------------- 阶段 5: 多 seed 结论点
# 对关键配置(基线延迟 + log-skew-normal, 甜点预算)跑 ≥5 seed 报均值/方差。
phase5() {
  echo "=== Phase 5: 多 seed 结论点 ==="
  for d in uniform logskewnorm; do
    for s in $SEEDS; do
      wait_slot
      run_one "p5_seed" DELAY="$d" N_LAYERS=7 CLUSTERS="3,3" RANK=16 M=24 N_FIT=40000 \
        N_S_JOINT=60 METHODS="R5,R6,R7,baselines" SEED="$s" &
    done
  done
  wait
}

# ---------------------------------------------------------------- 阶段 6: 汇总 + 出图
# 扫描 CKPT_DIR 全部 JSON,拼总表 + 出跨轴对比图(延迟×方法、网格×ISE、深度×误差等)。
phase6() {
  echo "=== Phase 6: 汇总 + 出图 ==="
  [[ "$DRY_RUN" == "1" ]] && { echo "[dry] aggregate $CKPT_DIR/*.json"; return 0; }
  env -u PYTHONPATH python3 -u -m simple_ttns_l2.experiments.full_scale_aggregate \
    --ckpt-dir "$CKPT_DIR" --out-dir "$RESULTS_DIR" \
    > "$LOG_DIR/aggregate.log" 2>&1
  echo "汇总完成 → $RESULTS_DIR (总表 + 图)"
}

# ---------------------------------------------------------------- 主流程
echo "结果目录: $RESULTS_DIR   并发: $MAX_PARALLEL   seeds: $SEEDS   阶段: $PHASES"
t0=$SECONDS
for p in $PHASES; do
  case "$p" in
    1) phase1;; 2) phase2;; 3) phase3;; 4) phase4;; 5) phase5;; 6) phase6;;
    *) echo "未知阶段 $p";;
  esac
done
echo "全部完成,用时 $(( SECONDS - t0 ))s。结果在 $RESULTS_DIR"

# =============================================================================
# 资源与规模估计(供排期):
#   单点 run: R5 ~5min / R7 ~1min / R6(G^K, n_s_joint=44)~14min / n_s_joint=80 ~更久。
#   Phase1 ~1h, Phase2 ~5h, Phase3 ~3h, Phase4 ~4h(大图/深层更久), Phase5 ~6h。
#   全量(单并发)≈ 20-25h;建议大内存(≥64G)+ MAX_PARALLEL=2~4 压到 ~6-8h。
#   R6 的 G^K 网格是内存/时间主瓶颈,K>=5 或 n_s_joint>=80 显著变重。
#
# 待接线(本脚本不含,属后续实现):
#   full_scale_runner.py   —— 读 FS_* 环境变量的单点 runner(复用现有 CFG/拟合/指标代码)
#   full_scale_aggregate.py—— 扫 JSON 拼表 + 出跨轴对比图
# =============================================================================
