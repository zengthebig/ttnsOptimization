# program_progress.md — TTNS 项目进度交接（单 agent 基线）

> **定位**：这是"指给一个 agent 就能开干"的基线文档。整合项目目标、可选模型、baseline、benchmark、成功标准、允许修改范围与当前进度。
> **维护**：由人类编辑迭代。深度细节查 `Program.md`（人维护，单层 TTNS 算子/测试/早期实验）与 `ALGORITHM_zh.md`（多层 DAG × TTNS 算法总纲，含 R1–R9 路线表、公式、已验证结论与负结果）。
> **语言/公式**：正文中文；公式用 `$` 定界；标识符/命令/路径保持英文。
> **Dev 汇总入口**：跨 worktree 的有效实验证据与结论统一见 [`simple_ttns_l2/reports/dev_experiment_summary_zh.md`](simple_ttns_l2/reports/dev_experiment_summary_zh.md)（含「已验证 / 单 seed 初证 / 未运行脚本 / 已淘汰路线」四级标签）。**勿把不同配置下的数字混为同一实验。**

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
- **UCI 混合 TTNSDE 排列修复**（§5.2.4）：`n_comps=8,m=128` 单 seed 下 POWER/GAS/HEPMASS 相对等参 TTDE 分别为 **+0.0708 / +0.2622 / +1.0577**（绝对 LL 未达论文配置）。
- **Dense R5/R6/R7 + 非负解析修复**：100 节点密连接 DAG 上，非负 core→raw² + 快 lr 使 L4 平均 LL≈**6.23**、`nonpos=0`；项目报告见 `ttns_multilayer_dag_project_report_zh.md`。
- **Budget 阶梯 + delay 泛化**：R5/R7 预算扫描与 R6 小块参照已落地；层间 delay 支持任意分布（log-skew）；全因子编排脚本已入库但**尚未运行**。
- **Theta 重参数化负结果**：线性 L2 下 θ→θ²/exp 正确但效果变差（已淘汰，见 `reparam_theta_report_zh.md`）。

### 2.2 尚未系统完成

- **R6 矩张量 $O(m^K)$ 交叉项**（高优先，待实现）：把 $\mathbb E_{p_Y}[q]=\langle C,T\rangle$ 一次性预计算矩张量，每步 $O(m^K)$（约快 15×），确定性传播 + 完整联合两头都占（仅限小块）。
- **平方 TTNS 全链传播** $p=\psi^2/Z$：dense 上非负解析 L2 块已打通；理论见 `squared_ttns_theory_zh.md`。待补平方 TTNS 树采样器，并系统对比「非负解析 L2」vs「平方 MLE」全链。
- **TTNS MLE 在真实 UCI 数据上的多 seed 系统基准**（POWER/GAS/HEPMASS 单 seed 已有，见 §5.2.4；待 ≥3 seed + 论文级容量）。
- **全因子大规模 study**（`run_full_scale_study.sh`）：脚本已入库，结果未跑。
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
- **非负解析 L2 块**（core→raw²，仍用解析 L2）：dense R5 修复主方案；深层负区可清零（见项目报告）。**不是**在线性 L2 上对 θ 做 square/exp 变换。
- **平方 TTNS**（$p=\psi^2/Z$，MLE）：`PAsTTNSSqrOpt`（R9 TTDE / UCI mixture 用）；理论已证仍是张量网络。UCI 侧已用于混合基准；分层全链平方传播仍待系统化。
- **已淘汰**：线性 L2 框架内 θ→θ²/exp 重参数化（正确性 PASS，val_l2 变差；`reparam_theta_report_zh.md`）。

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
| **POWER** | 6 | ✅ `/home/sbzeng/2_1/research/datasets/data/power/` | ✅ 已跑小/中配置；**论文级 Table 3 五数据集单 seed 任务见 §5.2.5** |
| **GAS** | 8 | ✅ `.../gas/` | 同上 |
| **HEPMASS** | 21 | ✅ `.../hepmass/` | 同上 |
| MINIBOONE | 43 | ✅ `.../miniboone/` | 论文级任务已编排（§5.2.5），结果待回填 |
| BSDS300 | 64 | ✅ `.../BSDS300/` | 论文级任务已编排（§5.2.5），结果待回填 |

> 正式数据根目录：`/home/sbzeng/2_1/research/datasets/data`（`--data-dir`）。仓库内 `data/data/` 可能不完整，以该绝对路径为准。

#### 5.2.1 UCI 三方基准结果（2026-07-03 更新，B-spline q=2 **m=96**，等参数量对齐 match_params，**2000 步**，单 seed=0）

入口：`env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.uci_ttde_vs_ttns --dataset both --m 96 --r-ttns 6 --steps 2000 --data-dir <绝对路径>/data/data`。
同基、同 Chow–Liu 树（`estimate_chow_liu_tree(n_bins=16, root=0)`）、平方 TT(链) 用 `match_params` 反解 rank 与平方 TTNS 等参数量。test_LL = 留出集平均对数密度；线性模型 `q` 已 $\int q{=}1$，负值点 clip 到 1e-12 并报 `nonpos_rate`；平方模型 log_p 在密度近零点可下溢为 $-\infty$，取 finite 均值并报非正率。切片密度 2D 积分 ≈1，平方 TT 边缘 vs `ttde_block_logp` 交叉检验 max|Δ|≈8e-17(POWER)/4e-16(GAS)（机器精度）。

| 数据集 | 模型 | 参数化/拓扑 | test_LL↑ | train_LL | params | rank | nonpos |
|---|---|---|---|---|---|---|---|
| POWER(6D) | global_TTDE | 平方TT(链) | **0.051** | 0.272 | 127,872 | 18 | 0.000 |
| POWER | global_TTNS | 线性TTNS(MI树) | −1.208 | −1.150 | 64,320 | 5 | 0.046 |
| POWER | global_TTNSDE | 平方TTNS(MI树) | **0.078** | 0.242 | 130,176 | 6 | 0.000 |
| GAS(8D) | global_TTDE | 平方TT(链) | **−1.283** | −1.278 | 38,400 | 8 | 0.000 |
| GAS | global_TTNS | 线性TTNS(MI树) | −21.970 | −21.995 | 103,680 | 9 | 0.742 |
| GAS | global_TTNSDE | 平方TTNS(MI树) | **−1.155** | −1.151 | 36,288 | 6 | 0.000 |

**容量对比（m=48 旧 → m=96 新）**：加大容量后两个数据集的平方模型 test_LL 都大幅上升——POWER TTDE −0.550→**0.051**、TTNSDE −0.526→**0.078**；GAS TTDE −4.435→**−1.283**、TTNSDE −4.453→**−1.155**。这证实了旧结论的主要疑点「欠训练」确实存在：m=48 时模型远未拟合到位。POWER 已接近论文报告值（TTDE ~0.46），**GAS 仍差约 10 nats**（论文 ~8.93），说明 m=96 对 GAS 仍不够，结论解读需谨慎。

**UCI 结论（更新）**：

① **平方参数化在 LL 口径上稳健且大幅领先，但「线性 vs 平方」不是干净的参数化消融**——线性 TTNS 用 L2 目标、平方模型用 MLE，二者训练目标不同；且线性 L2 在此容量下**优化发散**（POWER `train_l2` 冲到千级、GAS 冲到千万级，GAS 留出集 **74% 点密度≤0**）。故 +1.29(POWER)/+20.8(GAS) 的巨大差距主要反映**线性 L2 的不稳定/不可靠**，不能全部归因于参数化本身。可靠表述：**平方+MLE 稳健，线性+L2 在真实高分辨率基上易崩**。

② **等参数量下树拓扑相对链只有微弱且一致的正收益**——平方 TT(链) vs 平方 TTNS(MI 树)：POWER **+0.027**、GAS **+0.128**（注意 GAS 树的参数 36,288 反而**少于**链的 38,400 仍胜）。相比 m=48 旧值（+0.024/−0.018），加大容量后两个数据集都转为树略优，GAS 的边际更明显。但差距仍 <0.13 nat。

③ **总体仍是「树 ≈ 链」**，未见合成 DAG 上那种树结构大幅占优；且以下**局限未消除**，不足以下一般性结论：(a) **单 seed**，±0.13 nat 可能落在种子噪声内；(b) **POWER 的 Chow–Liu 树退化为星形**（`deg=[4,1,1,1,2,1]`，hub 核 $\sim m r^4$ 吃掉绝大多数参数），并非有代表性的树；(c) **GAS 仍欠拟合**（离论文 ~10 nats）。「UCI 去相关预处理导致近树依赖弱」仍只是**推测**，未量化树相对链的 MI 增益。图/数据（rank1 初始化，保留版）：`uci_{power,gas}_ttde_vs_ttns_*_rank1init.png`、`uci_ttde_vs_ttns_metrics_rank1init.json`。

#### 5.2.2 TTNSDE 初始化对比：canonical(EM) vs rank1（2026-07-03，m=96，仅换 TTNSDE 初始化）

新增 `--ttns-init {canonical,rank1}`（`fit_ttde_ttns` 非链拓扑初始化：`canonical` 用 `CanonicalRankK` 走 EM 估 R 个逐维边际分量再叠成 bond-R TTNS；`rank1` 为旧 `Rank1Only`）。同大参数（m=96, r_ttns=6, steps=2000, seed=0），**只改 TTNSDE 的初始化**，TTDE/线性 TTNS 不变。入口：`... uci_ttde_vs_ttns --dataset both --m 96 --r-ttns 6 --steps 2000 --ttns-init canonical --data-dir <abs>/data/data`。

| 数据集 | TTNSDE 初始化 | test_LL↑ | train_LL | init+训练耗时 | vs 同数据集 TTDE(链) |
|---|---|---|---|---|---|
| POWER | rank1（原，保留） | **0.078** | 0.242 | 35s | +0.027 |
| POWER | canonical（新，EM） | **−0.021** | 0.063 | 167s | −0.086 |
| GAS | rank1（原，保留） | **−1.155** | −1.151 | 20s | +0.128 |
| GAS | canonical（新，EM） | **−1.163** | −1.159 | 201s | +0.120 |

**初始化结论（诚实负结果）**：在这两个数据集、单 seed、m=96 下，**canonical(EM) 初始化并未提升 TTNSDE 的 test_LL，反而略降**——POWER −0.099（0.078→−0.021，明显）、GAS −0.008（基本持平）；且 EM 初始化使 TTNSDE 训练耗时大增（POWER 35s→167s、GAS 20s→201s）。需注意**存在 run-to-run 方差**：同配置、同 seed 的 TTDE(链) 在两次运行间 test_LL 也有 ~0.013 抖动（POWER 0.051→0.064），故 GAS 的 −0.008 落在噪声内、不可判读；POWER 的 −0.099 超出该噪声，倾向于「canonical 在此设置下没帮助」。要定论仍需 ≥3 seed。文件：默认名（`uci_ttde_vs_ttns_metrics.json`、`uci_{power,gas}_ttde_vs_ttns_*.png`）= canonical 结果；`*_rank1init.*` = 保留的原 rank1 结果。

#### 5.2.3 HEPMASS 三方基准（2026-07-03，21D，m=64，r_ttns=3，1000 步，单 seed=0）

入口：`env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.uci_ttde_vs_ttns --dataset hepmass --m 64 --r-ttns 3 --steps 1000 --train-cap 40000 --ttns-init <init> --data-dir <abs>/data/data`。21 维、受控容量（fit_train=30000，monitor_val=2000）以控时长。Chow–Liu/MI 树最大度仅 4（无 hub 爆炸），`match_params` 反解平方 TT(链) rank=4。等参数量对齐：平方 TTNS(MI树) r=3 → 25,152 参数 vs 平方 TT(链) r=4 → 19,968。

| 模型 | 参数化/拓扑 | test_LL↑ | train_LL | params | nonpos | sec |
|---|---|---|---|---|---|---|
| global_TTDE | 平方TT(链) | **−25.79** | −25.60 | 19,968 | 0.000 | 161 |
| global_TTNS | 线性TTNS(MI树) | **−25.44** | −25.39 | 68,352 | 0.020 | 41 |
| global_TTNSDE | 平方TTNS(MI树) | **−26.25** | −26.10 | 25,152 | 0.000 | 119 |

Δtest_LL：TTNS−TTDE **+0.36**、TTNSDE−TTDE **−0.46**、TTNSDE−TTNS −0.82。

**初始化数值稳定性（重要）**：TTNSDE 用 `rank1` 初始化在 HEPMASS 上**直接发散为 NaN**（step 300 起 train_nll=nan，nonpos=1.000）；换 `canonical`(EM 预热) 后正常收敛（nonpos=0）。这与 POWER/GAS（rank1 够用、canonical 无优势甚至略慢）相反——**21 维平方参数化更易崩，canonical 的 EM 预热在此成为必需**。故 §5.2.2「canonical 无用」的结论仅限低维 POWER/GAS，不可外推到更高维。

**HEPMASS 结论（本小节小容量/单分量口径，已被 §5.2.4 部分覆盖）**：在该配置下，**单分量**平方 TTNSDE 未超过 TTDE 链（−0.46 nat）。唯一有 LL 优势的是线性 L2 版 TTNS（+0.36），但代价是 **3.4× 参数**（68k vs 20k）且 **2% 留出点密度为负**。**混合** TTNSDE（`n_comps=8`）修复后的最新结论见 §5.2.4，勿与本表数字混读。

#### 5.2.4 修复混合 TTNSDE 排列后的 UCI 基准（2026-07-05，m=128，n_comps=8，单 seed=0）

背景：原 TTDE mixture 代码会为每个分量生成一个随机变量排列。这个机制对 TT(链)合理，因为随机排列后仍是一条合法链；但对数据估计的 Chow-Liu/MI 树不合理，因为固定的 MI 树边会被套到随机变量对上，导致除第 0 个恒等排列分量外，其余分量退化成随机树。已修复 `PAsTTNSSqrOpt.create`：非链树拓扑在 `n_comps>1` 时强制所有分量使用恒等排列，使每个分量都使用同一棵真 MI 树；混合多样性由初始化噪声和训练动态提供。TTDE 链仍保留随机排列，基线不变。

入口示例：`env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.uci_ttde_vs_ttns --dataset <power|gas|hepmass> --n-comps 8 --m 128 --r-ttns 6 --steps 5000 --train-cap 40000 --out-tag ncomps8fix_<dataset> --data-dir <abs>/data/data`。平方 TT(链) 仍用 `match_params` 反解 rank 与 TTNSDE 近似等参数量；`n_comps>1` 时跳过线性 TTNS 与切片图（当前边缘 helper 仅支持单分量）。

| 数据集 | 模型 | 参数化/拓扑 | test_LL↑ | train_LL | params | rank | 备注 |
|---|---|---|---|---|---|---|---|
| POWER(6D) | global_TTDE | 平方TT(链, 随机排列 mixture) | 0.0368 | 0.6328 | 1,363,968 | 18 | 早停 2700 |
| POWER | global_TTNSDE | 平方TTNS(MI树, 恒等排列 mixture) | **0.1076** | 0.4808 | 1,388,544 | 6 | 早停 3000 |
| GAS(8D) | global_TTDE | 平方TT(链, 随机排列 mixture) | 1.5898 | 1.9234 | 409,600 | 8 | 跑满 5000 |
| GAS | global_TTNSDE | 平方TTNS(MI树, 恒等排列 mixture) | **1.8521** | 2.1322 | 387,072 | 6 | 跑满 5000 |
| HEPMASS(21D) | global_TTDE | 平方TT(链, 随机排列 mixture) | −24.8620 | −17.9986 | 5,013,504 | 16 | finite-val 修复后重跑，早停 2700 |
| HEPMASS | global_TTNSDE | 平方TTNS(MI树, 恒等排列 mixture) | **−23.8042** | −20.6789 | 4,859,904 | 6 | finite-val 修复后重跑，早停 2700 |

Δtest_LL（TTNSDE − TTDE）：POWER **+0.0708**，GAS **+0.2622**，HEPMASS **+1.0577**。其中 GAS 的修复最关键：修复前同配置下 TTNSDE 为 1.1827、落后链 −0.4069；修复后升到 1.8521、反超链 +0.2622，净改善约 **+0.669 nat**。HEPMASS 的修复同样关键：旧训练监控因 0.05% 验证点 `log_p=-inf` 使 `val_nll=inf`，best params 未更新，旧值无效；改用 finite mean 监控后，TTNSDE 从表观落后变为领先 **+1.0577 nat**。

**当前结论**：真实 UCI 上，修复后的混合 TTNSDE 在 POWER/GAS/HEPMASS 三个数据集上均优于等参数量链式 TTDE。2026-07-06 debug 确认：HEPMASS 的验证集/测试集有极少数点落在由 `tr_fit` 建出的 B-spline 支撑外（测试约 0.032%），导致 `log_p=-inf`；旧 `_train_mle` 用普通均值计算 `val_nll`，因此全程 `val_nll=inf`，best params 从未更新，旧 HEPMASS 负结果无效。已修复训练监控为 finite mean 并打印 `val_nonfinite`，且若全程没有 finite 验证值则返回最后训练参数。修复后 HEPMASS 大配置日志中 `val_nonfinite=0.0005`，`val_nll` 全程 finite，最终 TTNSDE **−23.8042** vs TTDE **−24.8620**。绝对 LL 仍未对齐论文官方 TTDE（POWER 0.46、GAS 8.93、HEPMASS −21.34），但 HEPMASS 距论文 TTDE 的差距已由旧无效链结果约 3.82 nat 缩小到 TTNSDE 约 2.46 nat。

2026-07-06 补充：已实现 `n_comps>1` 的 mixture 2D 切片图（按各分量归一化常数 $Z_c$ 加权，TTDE 分量处理随机排列，TTNSDE 分量使用恒等排列 MI 树），并保存参数快照：`uci_power_ttde_vs_ttns_params_ncomps8fix_slices.pkl`、`uci_gas_ttde_vs_ttns_params_ncomps8fix_slices.pkl`。第一版 raw 全范围粗网格图在 GAS 强相关切片上出现网格积分 2.3–2.9，debug 后确认是尖峰密度 + 粗网格 aliasing，而非模型归一化错误：将切片窗口改为逐维 0.5%–99.5% quantile、网格加密，并对有限窗口内密度做 display-normalization 后，raw window integral 恢复正常。最终展示图：`uci_power_ttde_vs_ttns_slices_ncomps8fix_displaynorm.png`（raw win∫：0.971/0.965、0.972/0.970、0.979/0.977），`uci_gas_ttde_vs_ttns_slices_ncomps8fix_displaynorm.png`（raw win∫：0.946/0.942、0.996/0.996、0.971/0.969）。旧 `*_slices_ncomps8fix_slices.png` 仅保留作 raw/debug，不作为最终展示。

#### 5.2.5 论文级五数据集单 seed 基准（编排中，2026-07-23）

**证据标签**：单 seed 初证（任务已提交，结果待回填）。协议与产物见 [`simple_ttns_l2/reports/uci_paper_benchmark_report_zh.md`](simple_ttns_l2/reports/uci_paper_benchmark_report_zh.md)。

- 入口：`env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.uci_ttde_vs_ttns --preset paper --dataset <ds> --data-dir /home/sbzeng/2_1/research/datasets/data --seed 0`
- 提交：`sbatch simple_ttns_l2/experiments/run_uci_paper.sbatch`（阵列 0–4 对应 POWER/GAS/HEPMASS/MINIBOONE/BSDS300）
- 活跃 job：`785458`（0 POWER / 2 HEPMASS RUNNING）、`785461`（3 MINIBOONE RUNNING；4 BSDS PENDING）、`785463`（1 GAS 重提，pandas pickle 修复后排队）
- 口径：TTDE 用 README Table 3 的 m/rank/n_comps/batch/steps；TTNSDE 同超参但独立 `r_ttns`；关闭 `match_params` 与 early stop；全量训练数据。
- 勿与 §5.2.4 的 `m=128,n_comps=8` 数字混读。

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
- [ ] **平方 / 非负分层全链对照**：dense 非负解析 L2 已打通；待系统对比平方 MLE 全链 + 树采样器。
- [ ] **真实 UCI 多 seed / 论文级容量**：✅ §5.2.4 中配单 seed；§5.2.5 论文 Table 3 五数据集单 seed 已编排（结果待回填）。其后补 ≥3 seed。
- [ ] **全因子 study 实际运行**：`run_full_scale_study.sh` 已入库，结果待回填。
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
2. **非负/平方分层全链对照**：以 dense 最终非负解析配置为锚，补平方 TTNS 树采样器与 `UpperForest` 二次型传播，系统对比解析 L2 vs 平方 MLE。
3. **真实 UCI 基准升级**：§5.2.4 单 seed 已正；§5.2.5 论文级五数据集单 seed 已编排（`run_uci_paper.sbatch`）。待回填结果后，再补 ≥3 seed。
4. **运行全因子 study**（`run_full_scale_study.sh`）并回填证据标签。
5. **大图稳定性**：30+ 维多块层方案 A 链多 seed；深层 refit 退火/正则防发散。

跨实验证据总表与产物链接：[`simple_ttns_l2/reports/dev_experiment_summary_zh.md`](simple_ttns_l2/reports/dev_experiment_summary_zh.md)。

---

*创建：2026-07-03 — 整合 `Program.md` + `ALGORITHM_zh.md` + `prompt_phase1/2.md` + `README.md` 为单 agent 基线交接文档。*
*更新：2026-07-23 — `dev` 分支汇总 UCI / dense / budget / theta；见 `dev_experiment_summary_zh.md`。*
