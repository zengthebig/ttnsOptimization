# Dev 实验汇总：证据与结论

> **分支**：`dev`（独立工作树 `../TTDE-dev`，基于最新 `origin/main`）
> **目的**：把各实验 worktree 的**有效代码 + 最终指标/参数快照/展示图 + 结论报告**收拢到同一可审计分支；排除缓存、Slurm raw log、W&B、checkpoint 与失效中间图。
> **证据标签**（全文统一）：
>
> | 标签 | 含义 |
> |---|---|
> | **已验证** | 有复现入口 + 最终 JSON/图，结论可直接引用 |
> | **单 seed 初证** | 结果方向明确，但仅单 seed / 未达论文级配置 |
> | **未运行脚本** | 代码/编排已入库，大规模实验尚未执行 |
> | **已淘汰路线** | 负结果可追溯，不再作为主改进杠杆 |

**不要把不同配置下的数字混为同一实验。** 详细指标以各小节链接的 JSON/报告为准。

---

## 0. 纳入来源一览

| 主题 | 源分支 / 提交 | 集成方式 | 本分支关键提交 |
|---|---|---|---|
| UCI 基准 | `worktree-uci-ttns-benchmark` @ `d19a787` → `1dfeb03` | cherry-pick | `22070eb`、`9a84177` |
| Theta 重参数化 | `worktree-theta-reparam` @ `dab8f56` | cherry-pick | `1eb106e` |
| Budget / R6 / delay | `worktree-budget-sweep-layered` @ `05055cd`…`5bb081a` | cherry-pick | 至 `8e5bc63` |
| Dense R5/R6/R7 + R5 修复 | `worktree-dense-r567-compare` @ `5ff2a7d` | **选择性导入**（非整体 cherry-pick） | `6156cae` |
| 历史 main 快照 | 本地 `main@019a9e7` | **不合并**（含 `.venv`/大量 raw） | 仅作归档引用 |

共享文件手工合并：

- `simple_ttns_l2/analytic_tree_fit.py`：dense 的 `block_mode` / `marginal_l2_weight` / `normalize_every` / R7 超参 + budget 的 R6 完整联合链 + `edge_hi_eff`/`node_hi_eff`（任意 delay）。
- `program_progress.md`：以 UCI 最新事实为主，并补 dense / budget / theta / 本汇总入口。
- **未改** 仅供人类维护的 `Program.md`。

---

## 1. UCI 基准（POWER / GAS / HEPMASS）

| 项 | 内容 |
|---|---|
| **状态** | **单 seed 初证**（配置方向正；多 seed / 论文级容量未完成） |
| **源提交** | `1dfeb03`（本地 `9a84177`） |
| **数据** | MAF UCI：POWER(6D)、GAS(8D)、HEPMASS(21D) |
| **模型** | 平方混合 TTNSDE（MI 树，恒等排列）vs 平方混合 TTDE（链，随机排列）；等参数量 `match_params` |
| **Basis / 训练** | B-spline $q=2$，$m=128$，$n_\mathrm{comps}=8$，seed=$0$，steps=$5000$，train-cap=$40000$ |
| **关键修复** | ① 非链 mixture 强制恒等排列；② validation NLL 用 finite mean；③ HEPMASS 需 canonical 初始化稳定 |

### 主指标（TTNSDE − TTDE，test_LL）

| 数据集 | Δtest_LL | TTNSDE | TTDE |
|---|---|---|---|
| POWER | **+0.0708** | 0.1076 | 0.0368 |
| GAS | **+0.2622** | 1.8521 | 1.5898 |
| HEPMASS | **+1.0577** | −23.8042 | −24.8620 |

### 有效结论

- 修复后，三个数据集上**树混合均优于等参数量链混合**（单 seed）。
- GAS 修复前曾落后链约 −0.41；修复后反超 +0.26（净改善约 +0.67 nat）。
- HEPMASS 旧负结果因 `val_nll=inf` 失效；finite-val 修复后领先 +1.06 nat。

### 已知限制

- 仍为**单 seed**；绝对 LL 未达论文官方 TTDE（POWER 0.46 / GAS 8.93 / HEPMASS −21.34）。
- 早期小容量三方（m=96 / HEPMASS 线性 TTNS 等）数字**不可与 §5.2.4 大配置混读**。

### 最终产物

- 指标：`uci_ttde_vs_ttns_metrics_ncomps8fix.json`、`uci_ttde_vs_ttns_metrics_ncomps8fix_power.json`、`uci_ttde_vs_ttns_metrics_ncomps8fix_finite_hepmass.json`
- 展示图：`uci_power_ttde_vs_ttns_slices_ncomps8fix_displaynorm.png`、`uci_gas_ttde_vs_ttns_slices_ncomps8fix_displaynorm.png`；bars 同 tag
- 参数快照：`uci_power_ttde_vs_ttns_params_ncomps8fix_slices.pkl`、`uci_gas_ttde_vs_ttns_params_ncomps8fix_slices.pkl`
- 复现：`env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.uci_ttde_vs_ttns --dataset <power|gas|hepmass> --n-comps 8 --m 128 ...`
- 进度条目：`program_progress.md` §5.2.4

---

## 2. Dense DAG R5 / R6 / R7 与 R5 非负修复

| 项 | 内容 |
|---|---|
| **状态** | **已验证**（合成 dense 图；最终展示与项目报告齐备）；autoresearch 单次 attempt 为探索记录 |
| **源提交** | `5ff2a7d`（精选路径，见提交 `6156cae`） |
| **目标图** | 100 节点密连接多层 DAG（5 层 ×20 节点，簇内+跨簇父；max-plus delay） |
| **方法** | R5 树投影解析 L2 / R6 完整联合 / R7 采样 L2；非负 core→raw² + 快 lr 解析链 |

### 主结论

1. **线性 R5 在深层负区严重**（L4 `nonpos` 高、joint_LL 崩），不是解析传播公式错，而是块内拟合/符号问题。
2. **`marginal_l2_weight` 不能治负区**，大 λ 更差（已淘汰作主修复）。
3. **非负解析 L2 + 足够 lr/步数**可恢复深层：attempt `010` L4 `joint_ll≈6.23`，`nonpos=0`，std 比接近 1；项目报告推荐该配置。
4. R7 非负 MLE 对照亦强（L4≈7.07），用于定位能力上界，但终局仍偏好纯解析。

### 最终产物

- 项目报告：[`ttns_multilayer_dag_project_report_zh.md`](ttns_multilayer_dag_project_report_zh.md)
- 三方 / remedy：`dense_dag_r567_three_way_metrics.json`、`dense_dag_r567_remedy_metrics.json` + 对应 `*_report_zh.md` / PNG
- 展示：`project_final_solution.png`、`dense_dag_r567_slice_refined.png`、`dense_dag_r567_r5_nonneg_slices.png` 等
- Autoresearch：`simple_ttns_l2/autoresearch/r5_fix/`（attempts + `artifacts/*/manifest.json|metrics.json` + `target_audit_l4.json`）
- 复现入口：`experiments/dense_dag_r567_three_way.py`、`dense_dag_r567_remedy_tests.py`、`run_dense_r567*.sbatch`、`autoresearch/r5_fix/run_attempt.py`

### 明确排除（未入库）

- `__pycache__` / Slurm `logs/`
- 各 attempt 的逐次 `marginal_slices.png` / `slice_refined.png`
- 大体积 raw debug：`target_audit_fit_quality.json`、`l2_fit_debug_*`、`dense_dag_r567_slices_audit.json` 等

---

## 3. Budget sweep / R6 / delay 泛化

| 项 | 内容 |
|---|---|
| **状态** | 预算扫描与 R6 小块参照 **已验证**；**全因子编排脚本未运行** |
| **源提交** | 至 `5bb081a`（本地末 `8e5bc63`） |
| **设置** | clustered `[2,3,4,5,6]`×5 层（100 节点）；R5/R7 预算阶梯；R6 小块参照；log-skew delay |

### 主结论

- 三旋钮分攻误差轴：rank→R7 相关，m→R5 密度，n_fit→R7 深层 LL。
- 密度已达 oracle 约 **95–98%/维**；R7 相关误差约 **0.02–0.05**/对。
- 深层非死墙：L4 joint_LL 随 n_fit 20k→80k 从 0.03→1.23 单调拉起。
- R6 证明 corr 残差来自**目标口径**而非容量：加大 rank 不改树投影 fro，换完整联合才压低。
- Delay 已推广到任意分布（`edge_hi_eff`/`node_hi_eff`）。

### 最终产物

- 报告：[`budget_sweep_layered_report_zh.md`](budget_sweep_layered_report_zh.md)
- JSON：`budget_sweep_layered_metrics.json`、`budget_r6_block_ref_metrics.json`、`budget_baselines_ll_metrics.json`
- 图：`budget_sweep_layered.png`、`budget_sweep_error_viz.png`、`budget_sweep_slice_refined*.png`
- 未运行：`experiments/run_full_scale_study.sh`（仅脚本入库）

---

## 4. Theta 重参数化（负结果）

| 项 | 内容 |
|---|---|
| **状态** | **已淘汰路线**（正确性 PASS，效果负） |
| **源提交** | `dab8f56`（本地 `1eb106e`） |
| **做法** | 线性 L2 下 core θ → identity / θ² / exp θ（强制有效核非负） |

### 主指标（4D 双峰，3 seed，val_l2 越低越好）

| transform | mean val_l2 |
|---|---|
| identity（基线） | **−0.19728** |
| square | −0.19346 |
| exp | −0.12648 |

### 结论

在既有**线性 L2**框架内做核符号重参数化**不是改进杠杆**；非负性应走 `p=ψ²/Z` MLE（或 dense 已验证的非负解析 L2 块），而非在 L2 目标上硬加符号约束。

### 产物

- [`reparam_theta_report_zh.md`](reparam_theta_report_zh.md)
- `simple_ttns_l2/reparam.py`、`experiments/reparam_check.py`

---

## 5. 历史 main 快照（归档，未合并）

本地 `main@019a9e7` 含 `.venv` 与大量 raw 产物，**故意不并入 `dev`**。若需 BSDS300 等仅存在于该快照的结论，应单独抽取 summary JSON/报告，禁止整树合并。当前 UCI 主结论已被 `1dfeb03` 覆盖；BSDS300 仍标记为数据未就绪（见 `program_progress.md` §5.2）。

---

## 6. 跨实验总表（勿混读数字）

| 实验 | 证据标签 | 一句话 |
|---|---|---|
| UCI mixture 修复后 | 单 seed 初证 | POWER/GAS/HEPMASS 上 TTNSDE 均胜等参 TTDE |
| Dense 非负解析 L2 | 已验证 | 复杂多层 DAG 上 L4 LL≈6.23，负密度清零 |
| Budget 阶梯 | 已验证 | R5/R7 互补；深层靠 n_fit；误差已小 |
| 全因子 study 脚本 | 未运行脚本 | 编排已就绪，结果待跑 |
| θ→θ²/exp（线性 L2） | 已淘汰路线 | 正确但无效/更差 |
| marginal_l2 大 λ | 已淘汰路线 | 加重负区与崩溃 |

---

## 7. 复现与验证入口（本分支）

```bash
# 工作树
cd ../TTDE-dev   # 或本仓库 checkout 的 dev

# UCI TTNS 相关单测 / L2 目标 smoke（集成验证用）
export PYTHONPATH=$PWD/TTNSDE
python -m unittest TTNSDE/test/test_ttns_opt_unittest.py -v
env -u PYTHONPATH python3 -m unittest simple_ttns_l2/tests/test_ttns_l2_objective_unittest.py -v

# Theta 正确性
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.reparam_check
```

更完整命令见各专题报告与 `program_progress.md` §8。

---

*生成：`dev` 实验汇总集成（相对 `origin/main` 新建，不改写各实验分支历史）。*
