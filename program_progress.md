# program_progress.md — TTNS 项目进度交接（单 agent 基线）

> **定位**：这是"指给一个 agent 就能开干"的基线文档。整合项目目标、可选模型、baseline、benchmark、成功标准、允许修改范围与当前进度。
> **维护**：由人类编辑迭代。深度细节查 `Program.md`（人维护，单层 TTNS 算子/测试/早期实验）与 `ALGORITHM_zh.md`（多层 DAG × TTNS 算法总纲，含 R1–R9 路线表、公式、已验证结论与负结果）。
> **语言/公式**：正文中文；公式用 `$` 定界；标识符/命令/路径保持英文。

---

## 1. 项目目标

**终极目标**：用 **TTNS（树结构张量网络）** 近似高维密度函数，并在两条轴线上证明 TTNS 的优势：

1. **高维分布 density estimation**：在原始 TTDE（张量列车）代码库上扩展，把权重张量从链（TT）推广到树（TTNS）。
2. **DAG 上的层间传播**：支持 DAG 结构密度近似；终局为**多层 DAG**（拓扑分层、层内无父子、层间向下传播）。

**模型定义（用户定型，见 `clarify.md` / `ALGORITHM_zh.md` §0）**：

- 每一层（含所有下游层）都表示为一个**单父 TTNS 森林**（层内按相关性/MI 分块、每块一棵 chow-liu 树 TTNS）。
- 层与层之间用**层间传递 merge**（原 max-plus 术语）连接：下层节点 $x_v=\max_{u\in\mathrm{pa}(v)}(x_u+e_{uv})+d_v$，edge delay $e$、node delay $d$ 已知、独立、连续均匀。
- 逐层构建下游层：先按 DAG 拓扑 + MI 构建该层 TTNS 结构，再用 merge 的解析表达优化（全程无采样，CDF 域解析）。

**核心已验证结论**：结构感知的分层 TTNS 远优于把全联合硬拟合的全局 TT / 全局 TTNS / 全局 TTDE（参数少数量级，joint_LL / 边缘 / 相关全面领先）。

---

## 2. 当前进度总览

### 2.1 已完成（核心）

- **TTNS 核心算子**：内积、rank-1 求值、二次型、加减；快速/稳定两套等价（差异 ~1e-13）。9 节点深树 dense 对照误差 ~1e-16；3-child 扇出树 6/6 PASS。**无核心算子 bug**。
- **单层三拓扑系统实验**（L2 路径，3 seed）：Chow–Liu vs chain **56.9%**（7D）/ **64.4%**（8D）；balanced vs chain 49.7%；随机树 matched vs chain 42.6%。`init_noise=0` 时 chain TTNS ≡ pure TT。
- **多父 DAG 地基**：`dag_ttns.py` 真多父 core（greedy einsum）+ 5 组 20 项 dense 验证 PASS；diamond DAG vs 最优树 +11.4%，3 层 +29.3%，4 层/20 节点 +239%。
- **层间传播 pipeline 打通**：`dag_pipeline.py` 逐层前向采样；方案 A（采样+重拟合）/ B（CDF 解析）/ copula 采样全部落地。
- **全解析链主模型**（`analytic_tree_fit.py`）：DAG 结构分块（共享祖先）+ 块内解析 MI 树 + 解析 L2 优化，无采样、无跨层累积。R5（树投影）/ R6（完整联合）/ R7（采样求 L2）三落地。
- **全局 vs 分层系统对比**：基础 `[4,4,4]` 与复杂双峰 `[6,6,6,6]` 24 节点，分层全面碾压全局 TT/TTNS（参数少 3 个量级）。
- **五模型逐层对比**（`per_layer_all_methods.py`）：全联合 28 维，分层 joint_LL **23.10** vs TTDE 19.35 vs global_TTNS 7.34 vs global_TT 3.75。
- **全局 TTDE：TT(链) vs TTNS(MI 树)**：同 rank MI 树 ↑似然（test_LL 12.52→14.34）；**等参数量下 MI 树 TTNS 仍胜 +1.87**（TT 提到 rank=64 纯过拟合）。
- **加速**：hub 边降 rank（速度 ∝ rank^(fanout+1)）：hub_rank=12 → 3.5× 几乎无损；rank=8 → 12×。3-child 融合 einsum → junction 训练 3.2×。

### 2.2 尚未系统完成

- **R6 矩张量 $O(m^K)$ 交叉项**（高优先，待实现）：把 $\mathbb E_{p_Y}[q]=\langle C,T\rangle$ 一次性预计算矩张量，每步 $O(m^K)$（约快 15×），确定性传播 + 完整联合两头都占（仅限小块）。
- **每块改平方（非负）参数化** $p=\psi^2/Z$（最大改进杠杆）：理论已备（`squared_ttns_theory_zh.md`），待在单 3 节点块验证 + 打通全链；平方 TTNS 树采样器待补。
- **TTNS MLE 在真实 UCI 数据上的多 seed 系统基准**（POWER/GAS 已本地可用，见 §5）。
- **更大规模**（30+ 维、多块层）纯逐层森林链（方案 A）多 seed 统计；深层多块重拟合稳定化（偶发 val_l2 发散）。
- 旧笔记中 `TTNSDE/scripts/smoke_train_ttns.py` 若缺失需重建。

---

## 3. 可选模型

### 3.1 单层（层内）拓扑 — 三选一

| 拓扑 | 说明 | 现状 |
|------|------|------|
| **Chow–Liu** | 数据估互信息取最大生成树；**树类数据驱动最优** | ✅ 主线，vs chain 56.9%/64.4% |
| **balanced** | 固定堆式平衡二叉树；仅 rank-1 初始化 | ✅ vs chain 49.7% |
| **chain** | 固定链；等价 pure TT | ✅ 基线/对照 |

### 3.2 多层（层间）传播路线 — R1–R9

骨架统一：**真值数据 → 拟合 $L_0$ 数据森林 → 逐层向下生成下游层** + 并列全局黑盒基线。差异在 (a) 下游层拟合目标口径、(b) 传播方式。

| 路线 | 入口代码 | 目标/口径 | 传播 | 定位/结论 |
|---|---|---|---|---|
| **R1** 闭式核连乘 | `maxplus_cond_logdensity` | 完整联合(已知核) | 闭式条件核 | 分层因子化**性能上界**；下游非 TTNS，仅参考 |
| **R2** 方案A 采样链 | `propagate_layer`+`fit_layer_forest` | 完整联合(样本) | 采样+merge | **成链最优**；抗跨层累积，有 MC 噪声 |
| **R3** 方案B 单步解析 | `maxplus_cdf(_forest)` | 边缘/配对统计 | 确定性收缩 | **单步统计最准**/诊断用 |
| **R4** 方案B copula 链 | `sample_layer_copula` | 边缘+全相关 | 收缩+高斯copula | 保 pairwise，高斯近似 |
| **R5** 全解析链-树投影 | `fit_analytic_chain` | 树投影 $p_{tree}$ | 确定性收缩 | ★**主模型默认**；深层 LL 稳、corr 差 |
| **R6** 全解析链-完整联合 | `analytic_block_target_joint` | 完整联合 $p_Y$ | 确定性收缩 | corr 好 + 无 MC 累积，但 $O(G^K)$ 维度灾难 |
| **R7** 采样求 L2 链 | `fit_sampled_chain` | 完整联合(样本) | 采样+merge | corr 好、可扩块，深层 LL 略降 |
| **R8** 全局 TT/TTNS | `fit_flat` | 全联合(线性 L2) | — | 基线，远逊分层 |
| **R9** 全局 TTDE | `fit_ttde_tt`/`fit_ttde_ttns` | 全联合(平方 MLE) | — | 最强全局黑盒基线 |

> **R5/R6/R7 是同一"每层单父 TTNS 森林"模型的三种拟合落地**（结构相同，只差目标口径与传播方式）。取舍见 `ALGORITHM_zh.md` §6.6：要相关/联合几何用 R7（或 R6 小块）；要深层逐点密度用 R5；两头都占走 R6 矩张量版（待实现）。

### 3.3 参数化变体

- **线性 TTNS**（$q_\theta$，L2 目标 $L=\int q^2-2\mathbb E[q]$）：当前 R2/R5/R7/R8 用。
- **平方 TTNS**（$p=\psi^2/Z$，MLE）：`PAsTTNSSqrOpt`（R9 TTDE 用）；理论已证仍是张量网络，Scheme B 传播化为二次型树收缩（闭式、无采样）。**最大改进杠杆，待落地**。

---

## 4. Baseline 模型

| Baseline | 参数化 | 结构 | 训练 | 用途 |
|---|---|---|---|---|
| **global_TT** | 线性 TTNS | 全局 chain（=pure TT） | L2 | 线性黑盒下界 |
| **global_TTNS** | 线性 TTNS | 全全局 chow-liu 树 | L2 | 线性黑盒 |
| **global_TTDE** | 平方 TT | 全局 chain | MLE | **最强全局黑盒** |
| **TTDE TTNS(MI 树)** | 平方 TTNS | 全局 chow-liu 树 | MLE | 拓扑收益对照（等参数量仍胜 TT +1.87） |
| **pure TT** | — | chain | — | chain TTNS 的 `init_noise=0` 等价物（sanity） |
| Gaussian | — | — | 拟合训练集均值方差 | README 报告的弱基线 |

**参数量公式**（`ttde_ttns_vs_tt.py` 已实现 `match_params`）：TT $P=(d-2)mr^2+2mr$；TTNS $P=m\sum_v r^{\deg(v)}$。MI 树含高度数 hub → 参数暴涨；公平对比须等参数量对齐。

---

## 5. 测试 benchmark 数据集

### 5.1 合成（主线，已系统使用）

| 数据集 | 维度/规模 | 生成 | 主测点 |
|---|---|---|---|
| fork DAG 目标 | 6/7/8D | `experiments/dag_target.py` | 单层 chow-liu vs chain 切片 IAE |
| banded 阶梯 `[7,7,7,7]` | 28D, 4 层, fanin=3, wrap | `build_layered_spec` | 全联合分层 vs 全局（主战场） |
| clustered `[3,3]×3` | 18D, 3 层 | `build_clustered_spec` | 结构分块非平凡；R5/R6/R7 对比 |
| 双峰/复杂源 `[6,6,6,6]` | 24D | `compare_global_vs_layered_plot.bimodal_sources` | 多峰 + 环形依赖，分层优势放大 |
| 随机递归树 | 6D | `random_tree_target.py` | matched vs chain（M1.2） |
| diamond / 3 层 / 4 层多父 DAG | 4–20D | `dag_pipeline` | 多父 core 表达力 vs 树 |

源分布库：均匀、双峰高斯混合、Beta 偏态、非对称双峰、尖峰+均匀噪声。延迟 $e,d\sim U(0,0.3)$（均匀保证 Scheme B 解析精确）。

### 5.2 真实 UCI（MAF 套件，README §UCI）

| 数据集 | 维度 | 本地可用 | 状态 |
|---|---|---|---|
| **POWER** | 6 | ✅ `data/data/power/data.npy`（125M） | ✅ 已跑 TTDE/线性TTNS/平方TTNS 三方基准 |
| **GAS** | 8 | ✅ `data/data/gas/ethylene_CO.pickle`（168M） | ✅ 已跑三方基准 |
| HEPMASS | 21 | ❌ 未下载 | README 给下载指引 |
| MINIBOONE | 43 | ❌ 未下载 | — |
| BSDS300 | 64 | ❌ 未下载 | — |

> `data/data/` 下 `mnist`（空）、`cifar10`（仅 2 个 batch）不完整，非当前主线。**POWER/GAS 已完成 TTNS MLE 真实数据三方基准**（`uci_ttde_vs_ttns.py`）。数据根目录传 `--data-dir data/data`，加载器按 `root/<name>/...` 取文件。

#### 5.2.1 UCI 三方基准结果（2026-07-03，B-spline q=2 m=48，等参数量对齐 match_params，1000 步）

入口：`env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.uci_ttde_vs_ttns --dataset power|gas`。
同基、同 Chow–Liu 树（`estimate_chow_liu_tree(n_bins=16, root=0)`）、平方 TT(链) 用 `match_params` 反解 rank 与 TTNS 等参数量。test_LL = 留出集平均对数密度；线性模型 `q` 已 $\int q{=}1$，负值点 clip 到 1e-12 并报 `nonpos_rate`（线性参数化固有缺陷）；平方模型 log_p 在密度近零点可下溢为 $-\infty$，取 finite 均值并报非正率。切片密度 2D 积分 ≈1，平方 TT 边缘 vs `ttde_block_logp` 交叉检验 max|Δ|≈1e-16（机器精度）。

| 数据集 | 模型 | 参数化/拓扑 | test_LL↑ | train_LL | params | nonpos |
|---|---|---|---|---|---|---|
| POWER(6D) | global_TTDE | 平方TT(链) | **−0.550** | −0.417 | 63,936 | 0.000 |
| POWER | global_TTNS | 线性TTNS(MI树) | −2.297 | −2.257 | 118,944 | 0.026 |
| POWER | global_TTNSDE | 平方TTNS(MI树) | **−0.526** | −0.422 | 65,088 | 0.000 |
| GAS(8D) | global_TTDE | 平方TT(链) | **−4.435** | −4.439 | 7,680 | 0.000 |
| GAS | global_TTNS | 线性TTNS(MI树) | −7.962 | −8.012 | 112,320 | 0.077 |
| GAS | global_TTNSDE | 平方TTNS(MI树) | **−4.453** | −4.463 | 6,720 | 0.000 |

**UCI 结论**：① **平方参数化是决定性杠杆**——线性→平方提升 +1.77(POWER)/+3.51(GAS) nats；线性 TTNS 因 $q$ 可负、2.6%~7.7% 点密度≤0，test_LL 大幅落后。② **树拓扑收益在等参数量下基本消失**——平方 TT(链) vs 平方 TTNS(MI 树) 仅 +0.024(POWER)/−0.018(GAS)，不显著。③ 真实数据上**平方 TTNS(MI 树) ≈ 平方 TT(链)**，不像合成 DAG 那样树结构占优；推测因 UCI 经标准化+去相关预处理后变量间近树依赖弱，且 m=48 控时配置下链式已足够。图：`uci_ttde_vs_ttns_bars.png`、`uci_{power,gas}_ttde_vs_ttns_slices.png`。

### 5.3 评测指标

| 指标 | 含义 | 用途 |
|---|---|---|
| `joint_loglik` / `test_LL` | 留出集平均联合对数密度 | 跨模型主指标（↑） |
| `w1_marg` | 逐节点边缘 Wasserstein-1 均值 | 采样质量（↓） |
| `corr_fro` | 相关矩阵 Frobenius 误差 vs 真值 | 相关结构还原（↓） |
| 2D 切片 IAE | 边际积分绝对误差 | 单层树实验主指标 |
| `val_l2` / `val_nll` | 训练目标 | 训练监控（勿跨参数化比） |
| `final_integral` | $\int q$ | L2 路径应 ≈1 |

---

## 6. 成功标准

### 6.1 单层（树形收益，`prompt_phase2.md` §10）

- [x] **树形**：3 seed 下 balanced（或随机树）目标，匹配 TTNS mean IAE 比 chain **稳定低 ≥20%**（实测 49.7% / 42.6%）。
- [x] **多父 DAG**：联结树/chow-liu TTNS 在至少 1 个含双父依赖的切片对上 IAE **显著低于** chain（3 seed 平均 26.9% / 56.9%，关键切片 (2,6) 优势最大）。
- [x] **数据驱动拓扑**：Chow–Liu 自动估树 vs chain 3 seed 平均 56.9%，淘汰手工 junction。
- [x] **正确性**：`validate_ttns_opt.py` 全程 PASS（dense 误差 ≤1e-8，二次型可放宽 1e-5）。

### 6.2 多层（DAG 层间传播，`ALGORITHM_zh.md` §6）

- [x] **分层 TTNS ≫ 全局 TT/TTNS/TTDE**（全联合口径）：
  - 基础 `[4,4,4]`：分层 joint_LL **7.23**（64 参数）vs global_TTNS 3.90（79k）vs global_TT −1.35（117k）。
  - 复杂 `[6,6,6,6]`：分层 **20.69**（144 参数）vs global_TTNS 2.88 vs global_TT 1.81。
  - 28 维全联合：分层 **23.10** vs TTDE 19.35 vs global_TTNS 7.34 vs global_TT 3.75；corr_fro 0.297 vs 2.01 vs 9.67 vs 9.79。
- [x] **TTDE TT vs TTNS(MI 树)**：等参数量下 MI 树 TTNS 仍胜 **+1.87** test_LL（TT rank=64 纯过拟合）。
- [x] **R5/R6/R7 取舍已刻画**：完整联合目标（R6/R7）把块内非树相关 `corr_fro` 0.260→0.05/0.06（质变）；R7 链级每层 corr_fro 砍半但深层 LL 因 MC 累积略降。

### 6.3 待达成（下一步成功标准）

- [ ] **R6 矩张量版**：确定性传播 + 完整联合，无 MC 累积、$O(m^K)$（≤15× 加速），在小块上同时拿到 R5 的深层 LL 稳 + R7 的 corr 好。
- [ ] **平方 TTNS 块**：单 3 节点块平方 LL 推向 TTDE 水平 → 扩 `UpperForest` 二次型传播打通全链。
- [ ] **真实 UCI（POWER/GAS）多 seed 系统基准**：✅ 单 seed 三方对比已完成（见 §5.2.1），待补 ≥3 seed 均值/方差 + 论文级 m(256/512)/rank 配置复现 TTDE 报告值。初步结论：平方参数化主导（+1.77/+3.51 nats），树拓扑等参数量下基本持平。
- [ ] **大图稳定性**：30+ 维多块层方案 A 链多 seed 统计；深层 refit 不发散。

---

## 7. 允许修改的范围与硬约束

### 7.1 允许修改

- **TTNS 相关代码**：`TTNSDE/ttde/ttns/ttns_opt.py`（算子，改后必跑 §8 测试）、`simple_ttns_l2/` 下模型/传播/采样/分块代码。
- **实验脚本**：`simple_ttns_l2/experiments/` 与 `simple_ttns_l2/tests/`。
- **结果文档**：`simple_ttns_l2/reports/*_report_zh.md` + `*_metrics.json`（正文中文）。
- **agent 维护的流水账**：`prompt_phase2.md` checklist、`ALGORITHM_zh.md`（算法总纲，可更新结论/负结果）。

### 7.2 硬约束（不可违反）

- **除非用户明确要求，不修改 `Program.md`**（人维护交接文档）。
- **禁止使用 super-node**：不得把多个节点打包成单个多态张量核；补块内环上非树边相关只能在**单父 TTNS 框架内**另想办法（更优树/边选择、采样端 copula）。
- **模型侧只用单父 TTNS**（已淘汰早期多父 core `dag_ttns.py` 作主模型；保留作诊断/对照）。
- **每轮有效改动一个 commit**：说明修改内容、理论效果、测试/实验结果。
- **公平对比默认 `init_noise=0`**；结论性实验 ≥3 seed 报告均值/方差。
- **改 `ttns_opt.py` 前后必跑 §8 正确性测试**。

### 7.3 勿提交

大数据集、wandb 密钥、`workdir` 检查点。

---

## 8. 环境与正确性验证

### 8.1 环境

- Python ≥3.9；`pyproject.toml` 列 `jax==0.4.8`，但 macOS arm64 实际用 `jax==0.4.20`/`jaxlib==0.4.20`、`flax==0.6.11`、`scipy==1.11.4`。数值检查开 float64：`jax.config.update("jax_enable_x64", True)`。
- **两个 `ttde` 包**（根目录 `ttde/` 与 `TTNSDE/ttde/`）：TTNS 工作必须用 `TTNSDE/ttde/`。

### 8.2 导入规则（重要，导错会静默跑旧代码）

| 场景 | 命令 |
|------|------|
| **L2 实验脚本**（`simple_ttns_l2/`） | `env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.<name>`（脚本内已 `sys.path.insert`） |
| **仅 TTNS 算子测试** | `export PYTHONPATH=$PWD/TTNSDE` 后跑 `validate_ttns_opt.py` |

### 8.3 必做正确性（改 `ttns_opt.py` 后）

```bash
# 算子 dense 对照
export PYTHONPATH=$PWD/TTNSDE && python3 TTNSDE/validate_ttns_opt.py
# L2 目标 + smoke
env -u PYTHONPATH python3 -m unittest simple_ttns_l2/tests/test_ttns_l2_objective_unittest.py -v
env -u PYTHONPATH python3 -m unittest simple_ttns_l2/tests/test_train_l2_smoke_unittest.py -v
# TTNS 数学单测
PYTHONPATH=TTNSDE python -m unittest TTNSDE/test/test_ttns_opt_unittest.py -v
```

通过标准：dense 误差 ≤1e-8（病态二次型可放宽 1e-5）；新拓扑训练前用 `TTNS.full_tensor` 加 dense 往返测试。

### 8.4 主要实验命令

```bash
# 全局 TT vs 全局 TTNS vs 分层 TTNS（出图）
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.compare_global_vs_layered_plot
# 五模型逐层对比（默认 clustered [3,3]×3 小配置）
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.per_layer_all_methods
# 全局 TTDE：TT(链) vs TTNS(MI 树)（TTNS 树 rank 勿超 8，hub 核 ~rank^deg 会爆）
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.ttde_ttns_vs_tt
# R5(tree) vs R6(joint) vs R7(samp) 单块对比
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.joint_vs_tree_block
# 链级 R5 vs R7
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.sampled_vs_analytic_chain
```

---

## 9. 关键文件索引

```
TTNSDE/ttde/ttns/ttns_opt.py          # TTNSOpt + 收缩算子（改后必测）
TTNSDE/ttde/tt/tensors.py             # TTNS dense 参考 + full_tensor
TTNSDE/ttde/score/models/opt_for_tree_data.py  # PAsTTNSSqrOpt (MLE)
TTNSDE/ttde/train.py                  # --model-type ttns
TTNSDE/validate_ttns_opt.py           # dense 对照
simple_ttns_l2/
  dag_pipeline.py                     # MultiLayerSpec / build_layered_spec / build_clustered_spec / sample_joint
  maxplus_pipeline.py                 # DelayParams / ground_truth_samplers / propagate_layer(方案A)
  layered_forest.py                   # 层内MI分块 + chow-liu森林 + 条件核 + sample_forest
  analytic_tree_fit.py                # ★ 全解析链(主模型)：structural_blocks + _pair_mi + R5/R6/R7
  maxplus_cdf(_forest).py             # 方案B CDF域解析（marginal/pair/Hoeffding + block_joint_cdf）
  ttns_sampler.py                     # 单父TTNS条件inverse-CDF采样
  objective.py / train_l2.py          # L2目标、基、CLI
  chow_liu.py                         # 数据驱动树拓扑
  dag_ttns.py                         # 真多父core（诊断/对照，非主模型）
  experiments/                        # 基准实验脚本（见 ALGORITHM_zh.md §7）
  reports/                            # 所有图/JSON/中文报告
  tests/                              # 单测
Program.md / ALGORITHM_zh.md / clarify.md / squared_ttns_theory_zh.md  # 交接/算法/模型/理论文档
```

---

## 10. 下一步优先级

1. **R6 矩张量 $O(m^K)$ 交叉项**（高优先）：确定性 + 完整联合 + 无 MC 累积，小块两头都占。实现见 `ALGORITHM_zh.md` §3.5（把 `_cross_term_fn_joint` 网格换基索引）。
2. **平方 TTNS 单块验证**：在 3 节点块验证平方 LL 是否推向 TTDE，再扩 `UpperForest` 二次型传播。补平方 TTNS 树采样器。
3. **真实 UCI 基准升级**：POWER/GAS 单 seed 三方已跑（§5.2.1）；待补 ① ≥3 seed 均值/方差 ② 论文级 m=256/512、rank=16/32 全配置复现 TTDE 报告 test_LL(POWER 0.46 / GAS 8.93) ③ 度数受限树验证 UCI 上树拓扑是否真能持平/胜链。
4. **大图稳定性**：30+ 维多块层方案 A 链多 seed；深层 refit 退火/正则防发散。
5. **TTDE TTNS 度数受限树**：限制 hub 孩子数压参数量，看拓扑收益是否保持。

---

*创建：2026-07-03 — 整合 `Program.md` + `ALGORITHM_zh.md` + `prompt_phase1/2.md` + `README.md` 为单 agent 基线交接文档。*
