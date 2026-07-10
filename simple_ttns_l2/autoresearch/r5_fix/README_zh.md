# R5 解析链修复 — Autoresearch 脚手架

> **状态**：进行中 · **基准任务**：`dense_dag_r567` · **对照上限**：R7 sampled（同图、同分块、同 seed）

---

## 1. 任务目标

修复 **R5 全解析链**（`fit_analytic_chain`）在跨簇 dense DAG 上后层（尤其 L3–L4）的严重失效：

| 现象 | 基线 R5 (audit) | 目标 |
|---|---|---|
| L4 `joint_LL` | **−13.0** | ≥ **+1.0**（逼近 R7 ≈ +1.05） |
| L4 `nonpos_rate` | **≈57%** | **0%**（或 ≤1% 且采样 clip 可忽略） |
| L4 `std_ratio` (model/GT) | **≈0.35**（欠分散） | **0.85–1.15** |
| L4 最强相关对样本 $r$ | 结构丢失 | 与 GT $r$ 同号且 \|Δr\| < 0.3 |
| 传播方式 | 保持 **解析 UpperForest**，不偷换为 R7 采样链 | 除非 attempt 明确标注 hybrid |

**根因假设（待逐条验证/否定）**：

1. 块内 **Chow-Liu 树投影** $p_{\text{tree}}$ 丢失跨父相关 → 误差逐层累积。
2. **linear TTNS** 在 L2+ 出现大量负密度 → `clip(f≥0)` 采样导致边缘变尖、相关塌缩。
3. **immediate 分块** 块间独立，无法表达跨块相关（结构限制 vs 拟合目标问题需分离）。
4. L0 仍 linear，负区从 L0 传入 UpperForest CDF。
5. 解析网格 `n_s / n_s_pair` 或 `s_max` 递推不足 → 上层 CDF 偏差。

**不在本 autoresearch 范围内**：改 DAG 规格、改 global TT/TTNS、改 R7 本身。

---

## 2. 评测方式

### 2.1 固定 benchmark

- **图**：`dense_dag_r567` — 100 节点 · 5 层 · 簇 `[4,4,4,4,4]` · fanin=3 · rotate_cross · `block_mode=immediate`
- **配置**：`simple_ttns_l2/experiments/dense_dag_r567_three_way.CFG`
- **数据**：`n_total=24000`，70/30 train/test，`seed=0`（除非 attempt 另有说明）
- **L0**：与基线一致用 `fit_layer_forest`（linear），除非 attempt 显式改 L0

### 2.2 必报数值指标（每层 + 汇总）

由 `run_attempt.py` 写入 `artifacts/<attempt_id>/metrics.json`：

| 指标 | 方向 | 说明 |
|---|---|---|
| `joint_ll` | ↑ | 留出 test 上该层联合对数密度均值 |
| `nonpos_rate` | ↓ | 负密度占比 |
| `std_ratio_mean` | →1 | 模型采样边缘 std / GT std（均值） |
| `std_ratio_min` | →1 | 最欠分散节点 |
| `corr_fro_norm` | ↓ | L1–L4 归一化相关 Frobenius 误差（refined 图同口径） |
| `fit_seconds` | 记录 | 拟合耗时 |

**主判据（L4）**：`joint_ll` 与 `nonpos_rate`；**辅判据**：`std_ratio_mean`、`corr_fro_norm`、L1 是否不伤。

### 2.3 必产物图像（每次 attempt **同时**输出，缺一不可）

| 文件 | 内容 |
|---|---|
| `artifacts/<id>/marginal_slices.png` | 5×20 全节点一维边缘直方图（GT 黑 vs 本 attempt） |
| `artifacts/<id>/slice_refined.png` | 三块精细图：top-3 边缘+残差 / LL·corr vs 深度 / L1·L4 最强相关对散点 |

缺少任一 PNG → **attempt 无效**，不得 commit。

### 2.4 对照基线（只读，不重复跑）

| 标签 | 来源 |
|---|---|
| R5 λ=0 | `reports/dense_dag_r567_slices_audit.json` |
| R7 sampled | 同上 |
| R5 nonneg / marginal sweep | `reports/dense_dag_r567_remedy_metrics.json` |

---

## 3. 工作流（每个 attempt）

```bash
# 1. 跑 attempt（自动写 metrics + 两张图 + 更新 PROGRESS.md）
conda activate ttns
cd /home/sbzeng/2_1/research/TTDE
python -m simple_ttns_l2.autoresearch.r5_fix.run_attempt \
  --attempt-id 001 \
  --name "简短描述" \
  --variant baseline

# 2. 检查产物
ls simple_ttns_l2/autoresearch/r5_fix/artifacts/001/

# 3. git commit（必须）
./simple_ttns_l2/autoresearch/r5_fix/finish_attempt.sh 001
```

**规则**：

1. 一次 attempt = 一个独立假设 = **一次 git commit**。
2. commit 信息格式：`autoresearch(r5): <id> <name>`，正文附 L4 指标与图路径。
3. 代码改动与 `artifacts/<id>/`、`PROGRESS.md` 同 commit。
4. 失败 attempt 也要 commit（标注 ❌），避免重复踩坑。

---

## 4. 待做事项（Backlog）

优先级从上到下；完成一项后在 `PROGRESS.md` 标记并 commit。

| ID | 假设 / 改动 | variant 关键字 | 状态 |
|---|---|---|---|
| — | **基线复现** R5 λ=0（对齐 audit） | `baseline` | ⬜ 待跑 |
| A1 | L0 改为 nonneg MLE，L1+ 仍解析 L2 | `l0_nonneg` | ⬜ |
| A2 | 全链 nonneg + 解析 L2（已 preliminary：相关仍差） | `nonneg` | ✅ 见 remedy |
| A3 | `marginal_l2_weight` ∈ {0.3, 1.0} | `marginal_l2` | ✅ 否定 |
| A4 | 块内改 **joint 解析目标**（R6 口径，K≤4 块） | `joint_block` | ⬜ |
| A5 | `block_mode=source`（更大块，测 immediate 是否过碎） | `block_source` | ⬜ |
| A6 | 增大 `n_s, n_s_pair, an_steps` | `finer_grid` | ⬜ |
| A7 | 块内 **MLE**（nonneg）+ 解析 UpperForest 传播 | `analytic_mle` | ✅ 008 达标 |
| A8 | hybrid：L1+ 用 R7 采样传播，但评测仍称 R5-fix 对照 | `hybrid_sample_prop` | ✅ 007 达标（对照） |
| A9 | 块内 Chow-Liu 改 **最大生成森林** / 更高 rank | `rank16` / `forest_cl` | ⬜ |

> 新假设：复制 `attempts/_template.json` → `attempts/<id>.json`，在 `run_attempt.py` 注册或传 `--params`。

---

## 5. 进度

详见 **[PROGRESS.md](./PROGRESS.md)**（由 `run_attempt.py` 自动追加行）。

### 目录结构

```
simple_ttns_l2/autoresearch/r5_fix/
├── README_zh.md          # 本文件
├── PROGRESS.md           # 进度表
├── run_attempt.py        # 统一 runner
├── finish_attempt.sh     # commit 助手
├── attempts/             # 可选 per-attempt JSON 参数
│   └── _template.json
└── artifacts/
    └── <attempt_id>/
        ├── metrics.json
        ├── manifest.json
        ├── marginal_slices.png    # 必
        └── slice_refined.png      # 必
```

### 已有外部结论（计入背景，非本脚手架 commit）

- `marginal_l2_weight`：**不能**消除负区，L2+ 更差 → 否定 A3。
- R5 nonneg（A2）：`nonpos=0` 但 L4 LL≈2.45、相关 r≈0.01 → 非负 alone 不够。
- R7 nonneg：L4 LL≈7.07 — **性能上限参考**，非 R5 解析传播。

---

## 6. 扩展 runner

在 `run_attempt.py` 的 `VARIANTS` 注册新 fit 函数：

```python
VARIANTS["my_variant"] = FitSpec(
    fit_fn=my_fit_fn,
    color="#...",
    description="一句话假设",
)
```

`my_fit_fn(forest0, spec, params, key, s_max0, cfg) -> Dict[int, list]` 与 `fit_analytic_chain` 签名一致。
