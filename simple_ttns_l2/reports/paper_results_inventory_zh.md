# 可用于论文的实验结果清单

> 本清单基于 `dev` 分支提交 `3b0b827`，整理日期为 2026-07-24。本文只引用已提交的 JSON、报告、图片和代码，不引用未提交日志、checkpoint、pickle 内容或集群实时状态。`Program.md` 未被修改。
>
> 证据级别描述产物的可审计完整度，不等同于统计强度。“已验证”表示有复现入口、最终 JSON 或图、以及可恢复的完整配置；若实验仍只有 `seed=0`，本文会在“统计范围”中明确标注，且不会据此声称统计显著性。

## 1. 执行摘要

当前最适合进入论文正文的证据有三组。第一组是合成数据上的结构收益：7D 和 8D fork DAG 的三 seed 实验中，Chow–Liu TTNS 相对 chain TTNS 的关键二维切片 IAE 分别平均降低 56.9% 和 64.4%；随机递归树三 seed 实验中，匹配拓扑相对 chain 的 IAE 平均降低 42.6%。第二组是已知 DAG 结构和 delay kernel 条件下的分层建模收益：12D、24D 和 28D 实验均表明结构感知的 layered 方法在联合对数似然、边缘误差和相关误差上优于全局 TT、全局 TTNS 和平方 TTDE，但这些结果必须写明结构和层间核是已知的，而且目前主要是单 seed。第三组是 100 节点 dense DAG 上的 R5 修复：非负解析 L2 版本在 `seed=0` 下把 L4 `joint_ll` 从线性 R5 的约 $-13$ 提高到 6.2313，并把 L1–L4 的 `nonpos_rate` 降为 0；该结果有完整 attempt 配置和 JSON，但仍需多 seed 确认稳定性。

真实数据方面，已经完成的是小于论文协议容量的 UCI 混合模型单 seed 结果。配置为 `q=2, m=128, n_comps=8, seed=0, train_cap=40000, steps=5000`，并使用 `match_params=true`。在 POWER、GAS 和 HEPMASS 上，TTNSDE 相对 TTDE 的 `test_LL` 差值分别为 +0.0708、+0.2622 和 +1.0577。它们可以作为可复现的单 seed 初证，但不能写成统计显著或论文级五数据集结论。

五数据集论文级 UCI 协议是另一项实验。它使用 `n_comps=32`、全量训练数据、论文 TTDE rank、禁用 early stop，并覆盖 POWER、GAS、HEPMASS、MINIBOONE 和 BSDS300。仓库中尚无任何 `uci_ttde_vs_ttns_metrics_paper_<ds>_seed0.json`，因此该项只能归为“协议已准备、结果未完成”。报告中记录的 Slurm job 状态只是 2026-07-23 的快照，不是已提交实验结果，不能用于论文表格。

Budget sweep 清楚刻画了 `rank`、`m` 和 `n_fit` 的不同作用，但全扫描只有 `seed=0`。Theta 重参数化是可信的负结果：在 4D 双峰、3 seed 的小实验中，线性 L2 的 `identity` 优于 `square` 和 `exp`；由于没有独立结果 JSON 或图，它适合放在消融或附录，而不应作为主结果。

## 2. 证据级别总表

| 证据级别 | 实验 | 统计范围 | 论文用途 |
|---|---|---|---|
| 已验证 | 7D/8D fork DAG：Chow–Liu、balanced、chain | 3 seed | 主文中的拓扑匹配收益 |
| 已验证 | 6D 随机递归树：matched TTNS vs chain | 3 seed | 主文或附录中的外部结构复核 |
| 已验证 | CDF 解析传播 vs 采样传播 | 单次配置 | 方法动机或机制消融 |
| 已验证 | 12D/24D/28D layered vs global TT/TTNS/TTDE | 单 seed 或单次配置 | 已知结构和核条件下的主实验；不得声称统计显著 |
| 已验证 | 100 节点 dense DAG 的 R5-linear、R7-linear、R5-nonneg、R7-nonneg 及固定链对照 | `seed=0` | 大图可行性、失效诊断和修复结果；需要多 seed |
| 已验证 | 100 节点 budget sweep、R6 小块参照、全局基线 | `seed=0` | 预算敏感性和 R5/R6/R7 机制消融；需要多 seed |
| 单 seed 初证 | UCI POWER/GAS/HEPMASS，`m=128, n_comps=8` | `seed=0` | 真实数据初证；不能替代论文级五数据集结果 |
| 单 seed 初证 | UCI HEPMASS 单分量小容量三方对比 | `seed=0` | 负例、初始化和容量边界；不可与混合结果合并 |
| 单 seed 初证 | 18D 全局平方 TTNS(MI tree) vs TT(chain) | `seed=0` | 拓扑收益的补充证据；参数匹配不精确且树计算代价高 |
| 协议已准备、结果未完成 | UCI 五数据集 `--preset paper` | 计划为 `seed=0` | 等待 JSON 后才能进入结果表；之后仍需至少 3 seed |
| 协议已准备、结果未完成 | `run_full_scale_study.sh` 全因子实验 | 未运行 | 投稿前确认性交叉实验 |
| 协议已准备、结果未完成 | R6 矩张量 $O(m^K)$ 版本、平方 TTNS 分层全链 | 尚未实现或未系统运行 | 不能形成当前论文论断 |
| 负结果或仅用于消融 | 线性 L2 下 $\theta\to\theta^2$ 或 $\exp(\theta)$ | 3 seed 小实验 | 参数化消融或附录 |
| 负结果或仅用于消融 | Dense R5 的 `marginal_l2_weight=0.3/1.0` | `seed=0` | 说明边缘惩罚不能修复负区 |
| 负结果或仅用于消融 | Dense 线性 R5/R7 后层欠分散与负密度 | `seed=0` | 失效模式和修复必要性 |
| 负结果或仅用于消融 | UCI 小容量初始化和单分量 HEPMASS | `seed=0` | 初始化稳定性边界；部分报告与 JSON 命名不一致，需谨慎 |
| 负结果或仅用于消融 | 早期多父 core：diamond、3 层和 20 节点 deep DAG | `seed=0` | 结构表达力诊断；该模型已退出主线 |

## 3. 论文主张与证据映射

| 可支持的论文主张 | 直接证据 | 限定语句 | 当前不能支持的扩展 |
|---|---|---|---|
| 当目标依赖接近树结构时，匹配或数据驱动 TTNS 比 chain TTNS 更准确。 | `dag_chow_liu_vs_chain_multiseed_metrics.json`、`dag_chow_liu_vs_chain_8d_multiseed_metrics.json`、`random_tree_matched_vs_chain_multiseed_metrics.json` 及对应报告。 | 结论限于指定合成目标、B-spline L2 训练和 3 个 seed。 | 不能据此声称任意真实数据或任意树结构都显著优于链。 |
| 在已知 DAG 结构和 delay kernel 时，结构感知的 layered 因子化可用更少学习参数优于全局黑盒。 | `global_vs_layered_metrics.json`、`global_vs_layered_complex_metrics.json`、`per_layer_all_methods_metrics.json` 及对应图片和报告。 | 必须说明层间条件核和图结构为 oracle/已知；结果主要为单次配置。 | 不能声称在未知图或未知 kernel 的纯密度估计任务上同样领先。 |
| CDF 域解析传播可避免有符号密度采样截断造成的相关损失。 | `maxplus_cdf_vs_sampling_metrics.json` 和 `maxplus_cdf_vs_sampling_report_zh.md`。 | 在 `[4,4,4]`、`fanin=2`、上层负质量比例约 0.005 的配置中，`corr_fro` 为 0.0463 vs 0.1102。 | 不能声称所有分布和规模下解析传播都更快或更准。 |
| R5 的 dense 后层崩溃主要可由非负解析 L2 和合适优化尺度修复。 | `autoresearch/r5_fix/artifacts/010/manifest.json`、`metrics.json`、`target_audit_summary_zh.md` 和 `project_final_solution.png`。 | `seed=0`；最终配置继承 `dense_dag_r567_three_way.CFG`，并覆盖 `an_lr=0.001, an_steps=3000`。 | 不能声称已证明多 seed 稳定性、全局最优或优于所有 R7 变体。 |
| R5、R6、R7 的目标口径具有互补性。 | `budget_sweep_layered_metrics.json` 和 `budget_r6_block_ref_metrics.json`。 | R6 证据只来自 3 维小块；budget sweep 为 `seed=0`。 | 不能声称 R6 已在 100 节点全链上验证，也不能声称单个预算点同时最优。 |
| 修复后的混合 TTNSDE 在三个 UCI 数据集的单 seed 中优于等参数量 TTDE。 | 三个 `uci_ttde_vs_ttns_metrics_ncomps8fix*.json`。 | 配置为 `m=128, n_comps=8, seed=0`，且绝对 LL 未对齐论文 TTDE。 | 不能声称统计显著、五数据集全面领先或已达到论文级容量。 |
| 在线性 L2 目标中直接约束 core 符号不是有效改进杠杆。 | `reparam_theta_report_zh.md` 和 `experiments/reparam_check.py`。 | 只在 4D 双峰、3 seed 的小配置中成立。 | 不能外推为平方密度 $p=\psi^2/Z$ 或非负解析块本身无效。 |

## 4. 主实验结果

### 4.1 单层拓扑收益

#### 4.1.1 7D 和 8D fork DAG

实验目的是检验从样本估计的 Chow–Liu 树能否比固定 chain 更好地表示双父节点依赖。7D 配置为 `q=2, m=56, rank=16, batch_sz=128, lr=0.001, train_steps=400, n_train=8000, n_val=4000, n_test=4000`；seed 为 313、2602 和 20260227。8D 使用同一训练配置和相同三个 seed。对比方法是 Chow–Liu TTNS、balanced TTNS 和 chain TTNS，主指标为关键二维切片 IAE，方向为越低越好。

7D 的 Chow–Liu 树在三个 seed 中均命中 6/7 条生成 DAG 边，Chow–Liu 相对 chain 的关键切片 IAE 改善分别为 55.9%、64.6% 和 50.1%，跨 seed 平均为 56.9%。8D 中命中 7/8 条边，改善分别为 65.7%、63.6% 和 63.8%，跨 seed 平均为 64.4%。这些数值来自 `dag_chow_liu_vs_chain_multiseed_report_zh.md`、`dag_chow_liu_vs_chain_8d_multiseed_report_zh.md` 以及同名 `*_metrics.json`。

复现入口为 `simple_ttns_l2/experiments/fit_dag_chow_liu_vs_chain.py` 及其多 seed 调用逻辑。可用于论文的论断是：在这些 fork DAG 合成目标上，数据驱动的树拓扑在三个 seed 中一致降低关键切片 IAE。尚不能支持真实数据上的统计显著性，也不能把“命中边数”解释为完整因果结构恢复。

#### 4.1.2 随机递归树

实验使用 6D 随机递归树，配置为 `q=2, m=64, rank=16, batch_sz=256, lr=0.001, train_steps=400, n_train=8000, n_val=4000, n_test=4000`，seed 为 313、2602 和 20260227。匹配目标 parent 的 TTNS 相对 chain 的关键切片 IAE 改善分别为 36.7%、41.6% 和 49.5%，跨 seed 平均为 42.6%。来源是 `random_tree_matched_vs_chain_multiseed_metrics.json` 和 `random_tree_matched_vs_chain_multiseed_report_zh.md`，复现入口为 `experiments/fit_matched_vs_chain_random_tree.py` 与 `fit_matched_vs_chain_random_tree_multiseed.py`。

该结果可以作为“拓扑匹配收益不只出现在单一 fork DAG”这一补充论据。它不能支持对所有随机树分布的总体概率论断，因为只使用了三个生成 seed 和一组训练超参数。

### 4.2 已知多层 DAG 上的 layered 方法

12D 基础实验使用 `layer_sizes=[4,4,4]`、`fanin=2`、`n_total=40000` 和均匀 delay $U[0,0.25]$。`layered_TTNS` 的学习参数量为 4,672，`joint_LL=7.231`、`W1_marg=0.0053`、`corr_fro=0.128`；平方 TTDE 的相应数值为 64,640、5.630、0.0064 和 1.043。来源是 `global_vs_layered_metrics.json`、`global_vs_layered_report_zh.md`、`global_vs_layered_overview.png` 和 `global_vs_layered_marginals.png`，复现入口为 `experiments/compare_global_vs_layered_plot.py`。

24D 复杂实验使用 `layer_sizes=[6,6,6,6]`、双峰源、`fanin=2`、`n_total=50000`、`m=24` 和均匀 delay $U[0,0.25]$。`layered_TTNS` 使用 19,728 个学习参数，`joint_LL=20.689`、`W1_marg=0.0049`、`corr_fro=0.297`；平方 TTDE 使用 171,936 个参数，指标为 15.822、0.0051 和 2.974。来源是 `global_vs_layered_complex_metrics.json`、对应报告和两张 `global_vs_layered_complex_*.png`，复现入口同上。

28D 五模型实验使用 `layer_sizes=[7,7,7,7]`、`fanin=3`、`wrap=true`、`n_total=40000`、`q=2, m=24` 和异质多峰/偏态源。完整 28D 联合口径下，`layered_joint` 的 `joint_LL=23.10`、`corr_fro=0.297`、`W1_marg=0.0055`；`global_TTDE` 为 19.35、2.01 和 0.0060。来源是 `per_layer_all_methods_metrics.json`、`per_layer_all_methods_report_zh.md` 和 `per_layer_all_methods_fulljoint.png`，复现入口为 `experiments/per_layer_all_methods.py`。

以上三项共同支持的论文论断必须写成：当 DAG 结构和 max-plus delay kernel 已知时，按结构因子化的 layered 方法在这些合成基准中比全局黑盒更参数高效。不能删去“结构和 kernel 已知”这一条件。28D 报告还显示在逐层边缘口径上 `global_TTDE` 的 LL 更高；因此不能选择性地只报告完整联合表而不解释评测口径差异。

### 4.3 CDF 解析传播机制实验

实验使用 `layer_sizes=[4,4,4]`、`fanin=2`、`n=15000` 和均匀 edge/node delay $U[0,0.25]$，比较方案 A 的采样传播和方案 B 的 CDF 域解析传播。上层有符号 TTNS 的平均负质量比例约为 0.005。解析方案的 `mean_abs_err=0.0032`、`std_abs_err=0.0096`、`corr_fro=0.0463`；采样方案为 0.0036、0.0093 和 0.1102。来源是 `maxplus_cdf_vs_sampling_metrics.json` 和 `maxplus_cdf_vs_sampling_report_zh.md`，复现入口为 `experiments/fit_maxplus_cdf_vs_sampling.py`。

该实验支持“采样截断会损失相关结构，而解析积分在这一配置下避免了该损失”的机制解释。它不直接支持复杂度、墙钟速度或任意非均匀 delay 下的普适优势。

### 4.4 补充结构实验与早期多父 core 诊断

18D 全局平方模型实验使用 clustered `[3,3]x3`、`m=24` 和 `seed=0`，比较 TT chain rank 8、近参数量 TT chain rank 64 与 TTNS MI-tree rank 8。三者 `test_LL` 分别为 12.52、12.47 和 14.34，参数量分别为 24,960、1,575,936 和 1,596,096。来源是 `simple_ttns_l2/reports/ttde_ttns_vs_tt_metrics.json` 和 `ttde_ttns_vs_tt.png`，复现入口为 `simple_ttns_l2/experiments/ttde_ttns_vs_tt.py`。它可以作为平方 MLE 下树拓扑收益的单 seed 补充，但近参数量 chain 与 tree 仍相差约 1.3%，且 rank 64 chain 的训练 LL 更高、测试 LL 更低，说明过拟合。该项不能支持统计显著性。

早期多父 core 实验直接把 moral graph 的多父连接编码进 `dag_ttns.py`，它们只用于回答“树缺边是否会造成表达损失”，不是当前单父 TTNS 主方法。Diamond 4D 实验使用 `q=2, m=16, rank=6, steps=400, n_train=20000, n_val=8000, seed=0`，多父 DAG 的 `best_val_l2=-19.1007`，最优树为 -17.1505，报告的相对改善为 11.4%。3 层 5 节点实验使用 `m=16, rank=6, steps=400, n_train=30000, n_val=10000, seed=0`，多父 DAG 为 -97.1440，最优树为 -75.1263，改善 29.3%。4 层 20 节点实验使用 `[5,5,5,5]`、`m=10, rank=6, steps=600, n_train=40000, n_val=12000, seed=0`，近参数量下多父 DAG 为 -703207.9，最优树为 -207451.1，报告改善 239.0%。来源分别是 `diamond_dag_vs_tree_metrics.json`、`multilayer_dag_vs_tree_metrics.json`、`deep_dag_vs_tree_metrics.json` 及对应报告；复现入口为 `simple_ttns_l2/experiments/fit_diamond_dag_vs_tree.py`、`fit_multilayer_dag_vs_tree.py` 和 `fit_deep_dag_vs_tree.py`。

这些早期结果的 `val_l2` 量级随维度和密度尺度剧烈变化，不能跨实验比较，也不能与 LL 指标混用。尤其 20 节点结果的绝对 L2 数值很大，且只有 `seed=0`。由于当前项目硬约束是单父 TTNS、禁止 super-node，多父 core 结果只能放在相关工作、动机或附录诊断中，不能冒充 R5/R6/R7 主方法结果。

## 5. R5 主方法与 R6/R7 对照

### 5.1 名称和变体

本文统一使用以下名称，以避免报告中的简称混淆。

- `R5-linear` 表示 `dense_dag_r567_three_way_metrics.json` 中的 `R5_tree`，即线性、有符号的解析树投影 L2。
- `R5-nonneg` 表示 core 采用 `raw**2` 的非负解析树投影 L2。最终推荐配置是 autoresearch attempt 010。
- `R6-joint` 表示完整联合目标的解析 L2。当前只有 3 维小块参照，没有 100 节点全链结果。
- `R7-linear` 表示 `R7_sampled`，即从完整联合目标采样后拟合的线性 L2 链。
- `R7-nonneg` 表示补救实验中的非负 MLE 链。它与 `R7-linear` 不是同一训练目标或参数化。

### 5.2 Dense 100 节点基准配置

Dense 基准有 5 层，每层 20 个节点，簇为 `[4,4,4,4,4]`，`fanin=3`、`cross_fanin=2`、`rotate_cross=true`，共有 272 条边。delay 为 `src~U[0,1]`、edge/node delay 为 $U[0,0.3]$。数据和模型配置为 `n_total=24000, n_sample=8000, n_fit=20000, q=2, m=24, rank=8, block_mode=immediate, seed=0`。完整基础配置保存在 `dense_dag_r567_three_way_metrics.json` 和 `experiments/dense_dag_r567_three_way.py` 的 `CFG` 中。

### 5.3 原始 R5-linear 与 R7-linear

`R5-linear` 的 L0–L4 `joint_LL` 为 22.693、5.272、5.212、1.968 和 -13.015；`R7-linear` 为 22.693、5.838、5.108、3.111 和 1.049。对应 `corr_fro` 在 L4 为 5.768 和 4.901，参数量为 193,248 和 183,840，单次用时为 246.5 秒和 234.8 秒。数值来自 `dense_dag_r567_three_way_metrics.json`，展示图为 `dense_dag_r567_results.png` 和 `dense_dag_r567_slice_refined.png`，复现入口为 `experiments/dense_dag_r567_three_way.py` 和 `run_dense_r567.sbatch`。

切片审计使用同一 `seed=0`、`n_plot=5000` 和 `grid=400`，但重新计算得到 `R5-linear` L4 `joint_LL=-12.585`、`nonpos_rate=0.571`，`R7-linear` L4 `joint_LL=1.049`、`nonpos_rate=0.134`。因此正文表格应以主 JSON 的 -13.015 为主，负密度诊断引用审计的 -12.585/0.571，并明确这是审计重评值，不能把两者写成同一次评估。审计还显示 R7 L4 平均标准差比为 0.722，20 个节点中 19 个低于 0.8，说明欠分散是真实现象而不是绘图 bug。来源是 `dense_dag_r567_slices_audit_zh.md` 和 `dense_dag_r567_remedy_metrics.json`。

### 5.4 R5-nonneg 最终修复

最终 attempt 010 继承上述 Dense `CFG`，使用 `variant=nonneg`，并覆盖 `an_lr=0.001, an_steps=3000`。L1–L4 的 `joint_ll` 为 6.7897、7.1279、6.9413 和 6.2313，`nonpos_rate` 均为 0；L4 的平均标准差比为 0.9516，`corr_fro=0.2310`。主数值来源是 `simple_ttns_l2/autoresearch/r5_fix/artifacts/010/metrics.json`，配置覆盖来源是同目录 `manifest.json`，完整基础配置来源是 `simple_ttns_l2/experiments/dense_dag_r567_three_way.py`。最终展示图是 `simple_ttns_l2/reports/project_final_solution.png`，复现入口为 `simple_ttns_l2/autoresearch/r5_fix/run_attempt.py --config simple_ttns_l2/autoresearch/r5_fix/attempts/010_nonneg_l2_fast_lr_steps3000.json`。

固定链对照中，rank 8 chain 的参数量约为 80,352、L4 LL 为 5.95；rank 13 chain 的参数量约为 205,152、L4 LL 为 6.15；Chow–Liu tree rank 8 的参数量约为 193,248、L4 LL 为 6.23。数值来自 `ttns_multilayer_dag_project_report_zh.md`，rank 13 的机器可读结果在 `autoresearch/r5_fix/artifacts/012/metrics.json`。它支持“近参数量对齐后树结构仍有小幅收益”，但差距很小且只有 `seed=0`，不能写成显著优势。

`R5-nonneg` 可以支持纯解析路线在该 100 节点基准上可行，也可以支持原始失败主要与非负参数化下学习率过小有关。它不能支持多 seed 稳定性，也不能支持“R5 总体优于 R7”，因为 `R7-nonneg` 在补救实验中的 L4 LL 为 7.068，高于 attempt 010 的 6.231，而且两者训练目标不同。

### 5.5 R6 与 R7 的边界

R6 小块参照使用 clustered `[3,3]` 的 3 维块，对比同一树结构下的 `tree(R5)`、`joint(R6)` 和 `samp(R7)`。在非树相关明显的 block 0 中，rank 8 的 `corr_fro` 分别为 0.2607、0.0484 和 0.0601；rank 16 分别为 0.2602、0.0446 和 0.0614。来源是 `budget_r6_block_ref_metrics.json`，复现入口为 `experiments/budget_r6_block_ref.py`。这支持相关残差主要来自树投影目标口径，而不是简单容量不足。

该 R6 证据只覆盖 3 维小块。`dense_dag_r567_three_way_metrics.json` 明确记录 `with_r6=false`，所以仓库不能支持“R6 已完成 100 节点 dense 全链验证”或“R6 在大图上优于 R5/R7”。

`R7-nonneg` 的补救实验在 L4 得到 `joint_LL=7.0684`、`nonpos_rate=0.00056`、平均标准差比 0.9706；来源是 `dense_dag_r567_remedy_metrics.json` 和 `dense_dag_r567_remedy_report_zh.md`，复现入口为 `experiments/dense_dag_r567_remedy_tests.py`。它适合作为能力上界和定位对照，但由于它采用非负 MLE 且包含采样，不能作为“纯解析 R5”的同目标公平胜负表。

## 6. Budget sweep

### 6.1 实验设置

Budget sweep 使用 100 节点、5 层的 clustered `[2,3,4,5,6]` 结构，每层 20 个节点，`fanin=2`，delay 为 `src~U[0,1]` 和 edge/node $U[0,0.3]$。基础配置为 `n_total=24000, n_sample=8000, n_fit=20000, q=2, m=24, rank=8, an_lr=0.003, an_steps=700, seed=0`。完整配置和所有点保存在 `budget_sweep_layered_metrics.json`，报告为 `budget_sweep_layered_report_zh.md`，复现入口为 `experiments/budget_sweep_layered.py`。

### 6.2 核心结果

下游 L1–L4 平均值显示，rank 从 8 增至 16 时，R7 的平均 `corr_fro` 从 0.829 降至 0.624，而 R5 约维持在 1.11；rank 24 没有继续改善 R7。把 `m` 从 24 增至 48 时，R5 相对 oracle 的平均 LL gap 从 0.874 降至 0.753，但相关误差基本不变。把 R7 的 `n_fit` 从 20k 增至 40k 和 80k 时，L4 `joint_LL` 从 0.0328 增至 0.7158 和 1.2289。所有数值均来自 `budget_sweep_layered_metrics.json`。

R5 `m=48` 的逐维 gap 从 L1 的约 0.017 nat 增至 L4 的约 0.051 nat。报告把 $\exp(-\mathrm{gap}/20)$ 换算为约 98.3% 到 95.0% 的逐维似然比；论文中应写“相对 oracle 的逐维指数化 LL 比”，不要笼统写成“恢复了 95–98% 的密度”。R7 rank 16 的每对相关系数 RMS 误差由 L1 的约 0.021 增至 L4 的约 0.047，计算口径为 `corr_fro/sqrt(380)`。

等预算全局基线以 rank 24 的 576,480 参数预算为参照。L1–L4 中，分层 R5 `m=48` 的 LL 为 5.67、3.49、2.96、2.60；`global_TTDE` 为 5.06、3.17、2.68、2.50；`global_TT` 和 `global_TTNS` 在 L2–L4 为负。来源是 `budget_baselines_ll_metrics.json` 与 budget 报告。参数并非严格相等：分层 R5 为 139k–576k，`global_TTDE` 为 941,760，因此论文应称“按预设预算构造的全局基线”，不应称为逐模型精确等参数量。

### 6.3 可支持和不能支持的论断

该组实验可以支持三个旋钮作用于不同误差轴，以及 R7 深层 LL 可随 `n_fit` 增大而改善。由于所有扫描点只有 `seed=0`，不能声称 rank 16 是统计稳定的最优甜点，也不能把 L4 的单调序列解释为一般收敛率。`rank24` 和 `nfit80k` 曾触发内存压力的说明来自报告，但仓库没有资源监控 JSON，因此只能作为工程备注。

`experiments/run_full_scale_study.sh` 已提交但尚未产生全因子结果。它属于“协议已准备、结果未完成”，不能把单旋钮扫描组合成一个从未实际运行的 `rank16+m48+nfit80k` 结果。

## 7. Theta 重参数化消融

实验目的，是在线性 L2 TTNS 中比较 `identity`、`square` 和 `exp` 三种 core 变换。代码使用 4D 双峰合成数据，每个 seed 有 512 个训练样本和 512 个验证样本，`q=2, m=12, rank=4, steps=400, lr=0.005, noise=0.05`，seed 为 0、1 和 2。指标是最优 `val_l2`，方向为越低越好。

`identity`、`square` 和 `exp` 的 mean `val_l2` 分别为 -0.19728、-0.19346 和 -0.12648，标准差分别为 0.00114、0.00770 和 0.00741。来源是 `reparam_theta_report_zh.md`，完整可执行配置在 `experiments/reparam_check.py`，实现位于 `simple_ttns_l2/reparam.py`。正确性检查包括变换后的 core 误差不超过 $10^{-12}$、归一化后积分为 1，以及 identity 路径单测不变。

该实验支持“在线性 L2 目标中逐元素强制 core 非负没有改善这一小规模任务”。它不能支持“非负密度参数化总体无效”，因为 Dense 的 `R5-nonneg` 和 UCI 的平方 MLE 使用不同目标与不同模型解释。该项没有独立 JSON 或最终图片，因此证据级别归入“负结果或仅用于消融”，建议在投稿前补一个机器可读 JSON，避免报告成为唯一数值载体。

## 8. UCI 真实数据结果

### 8.1 已完成的小容量混合模型单 seed 结果

三项已完成结果使用 MAF UCI 数据，TTDE 是平方 TT 链混合，TTNSDE 是平方 TTNS MI-tree 混合。共同配置由三个最终 JSON 记录：`q=2, m=128, r_ttns=6, n_comps=8, seed=0, train_cap=40000, ttde_n_train=30000, steps=5000, ttde_steps=5000, batch_sz=512, lr=0.002, train_noise=0.001, ttns_init=canonical, match_params=true`。TTDE 链的 mixture 保留随机排列，TTNSDE 的非链 mixture 使用恒等排列。

| 数据集 | 维度 | TTDE `test_LL` | TTNSDE `test_LL` | 差值 | TTDE/TTNSDE 参数量 | 来源 |
|---|---:|---:|---:|---:|---:|---|
| POWER | 6 | 0.0368 | 0.1076 | +0.0708 | 1,363,968 / 1,388,544 | `uci_ttde_vs_ttns_metrics_ncomps8fix_power.json` |
| GAS | 8 | 1.5898 | 1.8521 | +0.2622 | 409,600 / 387,072 | `uci_ttde_vs_ttns_metrics_ncomps8fix.json` |
| HEPMASS | 21 | -24.8620 | -23.8042 | +1.0577 | 5,013,504 / 4,859,904 | `uci_ttde_vs_ttns_metrics_ncomps8fix_finite_hepmass.json` |

复现入口为 `experiments/uci_ttde_vs_ttns.py`。最终 bars 图包括 `uci_power_ttde_vs_ttns_bars_ncomps8fix_power.png`、`uci_gas_ttde_vs_ttns_bars_ncomps8fix.png` 和 `uci_hepmass_ttde_vs_ttns_bars_ncomps8fix_finite_hepmass.png`。最终二维展示图只覆盖 POWER 和 GAS，分别为 `uci_power_ttde_vs_ttns_slices_ncomps8fix_displaynorm.png` 和 `uci_gas_ttde_vs_ttns_slices_ncomps8fix_displaynorm.png`。这些图使用有限窗口 display-normalization；LL 表不使用该展示归一化。

这组三个差值只能描述为单 seed 观察结果。绝对 LL 未达到报告引用的论文 TTDE 值 POWER 0.46、GAS 8.93 和 HEPMASS -21.34，因此它们不能证明达到原论文复现水平。JSON 中平方模型的 `nonpos_rate` 在 POWER/HEPMASS 分别约为 0.00014/0.00032；结合代码和报告，这一字段实际上包含支撑外或非有限 `log_p` 的比例，不应解释为线性密度为负的比例。

### 8.2 已完成的 HEPMASS 单分量小容量结果

单独的 HEPMASS 三方实验使用 `q=2, m=64, r_ttns=3, steps=1000, train_cap=40000, seed=0, n_comps=1`。TTDE、线性 TTNS 和平方 TTNSDE 的 `test_LL` 分别为 -25.7928、-25.4350 和 -26.2545；参数量分别为 19,968、68,352 和 25,152。来源是当前 `uci_ttde_vs_ttns_metrics.json` 和 `program_progress.md` §5.2.3，复现入口仍为 `experiments/uci_ttde_vs_ttns.py`。

该配置中平方 TTNSDE 落后 TTDE 0.4617 nat，而线性 TTNS 领先 TTDE 0.3578 nat，但线性 TTNS 使用约 3.4 倍参数且 `nonpos_rate=0.0197`。它是单分量、小容量、单 seed 负例，不能与 `m=128, n_comps=8` 的混合结果做平均、拼表或配置继承。

### 8.3 尚未完成的五数据集论文级结果

论文级协议由 `uci_paper_benchmark_report_zh.md`、`experiments/uci_ttde_vs_ttns.py --preset paper` 和 `experiments/run_uci_paper.sbatch` 定义。共同项为 `q=2, n_comps=32, em_steps=10, init_noise=0.01, lr=0.001`、全量训练数据和 `ttde_patience=0`。各数据集配置如下。

| 数据集 | m | TTDE rank | TTNSDE 默认 rank | batch | steps | train_noise |
|---|---:|---:|---:|---:|---:|---:|
| POWER | 256 | 16 | 6 | 8192 | 10000 | 0.01 |
| GAS | 512 | 32 | 4 | 1024 | 100000 | 0.01 |
| HEPMASS | 128 | 32 | 3 | 2048 | 10000 | 0.01 |
| MINIBOONE | 64 | 32 | 2 | 1024 | 10000 | 0.08 |
| BSDS300 | 256 | 16 | 2 | 512 | 100000 | 0.01 |

截至提交 `3b0b827`，仓库中不存在预期的 `uci_ttde_vs_ttns_metrics_paper_<ds>_seed0.json`。因此五行结果表全部为空，本清单把它归为“协议已准备、结果未完成”，而不是“单 seed 初证”。`uci_paper_benchmark_report_zh.md` 中 2026-07-23 的 RUNNING/PENDING 描述只能证明任务曾被编排，不能证明结果已经完成。

小容量实验与论文级实验的配置、指标和文件名必须完全分开。不得把小容量的 +0.0708/+0.2622/+1.0577 填入论文级表格，不得把论文级的 `m/rank/n_comps/batch/steps` 写成已完成 JSON 的配置，也不得把 MINIBOONE/BSDS300 写成已有结果。即使五数据集的 `seed=0` 全部完成，它们仍只构成单 seed 初证；投稿级统计结论仍需预先固定协议后补至少 3 seed。

## 9. 负结果与适用边界

### 9.1 Dense 边缘惩罚失败

在 Dense 基准中，`R5 marginal_l2_weight=0.3` 的 L4 `joint_LL=-84.489`、`nonpos_rate=1.0`，权重 1.0 时为 -68.861 和 1.0；权重 0 的审计重评为 -12.585 和 0.571。来源是 `dense_dag_r567_remedy_metrics.json`。因此可以否定“简单增大边缘 L2 权重能够修复负区”这一候选路线，但结果只有 `seed=0`，更合适的表述是“在该基准和这两个权重上失败”。

### 9.2 线性 R5/R7 的负密度和欠分散

`R5-linear` 在 L4 有约 57.1% 的非正密度点，`R7-linear` 的 L4 平均标准差比为 0.722。切片审计已排除列接线错误和主要直方图截断伪影。该负结果支持引入非负拟合和同时报告 LL、`nonpos_rate`、标准差比及相关误差，不能仅凭一维切片评价模型。

### 9.3 UCI 初始化结果的证据限制

`program_progress.md` §5.2.2 报告了 POWER/GAS 的 `m=96, steps=2000` 初始化比较，但当前 `uci_ttde_vs_ttns_metrics_rank1init.json` 实际包含的是 POWER `m=24, steps=200, ttns_init=canonical` 的 quick 配置，而不是报告描述的 m=96 rank1 两数据集结果。当前 `uci_ttde_vs_ttns_metrics.json` 也只包含 HEPMASS `m=64`，与该小节所称“默认名为 POWER/GAS canonical 结果”不一致。因此 m=96 初始化数值只能视为报告级历史记录，不能作为有匹配最终 JSON 的主表证据。投稿前应从相应提交历史恢复或重跑为带唯一 `out_tag` 的 JSON；在此之前不要引用其精确差值作为正式结论。

HEPMASS 小容量报告还记录 rank1 初始化出现 NaN、canonical 初始化稳定，但最终 JSON 只保存 canonical 结果，没有 rank1 失败的机器可读训练轨迹。因此该点可作为工程边界说明，不应量化成严格失败率。

### 9.4 公平性边界

Layered 模型使用已知 DAG 和已知 delay kernel，而全局 TT/TTNS/TTDE 从样本拟合全联合。这是“利用正确结构先验的系统比较”，不是信息条件完全相同的黑盒模型比较。UCI `match_params=true` 只做近似参数匹配，例如 POWER 两模型参数为 1,363,968 和 1,388,544，不能写成逐参数严格相等。Budget 全局 TTDE 的参数量 941,760 高于名义预算 576,480，也不能写成精确等预算。

## 10. 报告间不一致与处理规则

| 不一致 | 证据 | 本清单的处理 |
|---|---|---|
| Dense 原始 R5 L4 LL 有 -13.015 和 -12.585 两个值。 | 主 JSON/主报告为 -13.015；same-run 切片审计重评为 -12.585。 | 主结果引用 -13.015；负密度审计引用 -12.585，并明确评估来源不同。 |
| `dense_dag_r567_report_zh.md` 称切片图由 `plot_dense_dag_slices.py` 生成。 | `dense_dag_r567_slices_audit_zh.md` 说明当前 PNG 实际来自 `dense_dag_r567_three_way.py --plot` 的 same-run artifact。 | 图的复现入口以审计报告为准。 |
| “R5”有时指线性树投影，有时指最终非负解析 L2。 | `dense_dag_r567_three_way_metrics.json` 与 autoresearch attempt 010。 | 正文强制使用 `R5-linear` 和 `R5-nonneg`。 |
| “R7”有时指线性 sampled L2，有时指非负 MLE。 | `dense_dag_r567_three_way_metrics.json` 与 `dense_dag_r567_remedy_metrics.json`。 | 正文强制使用 `R7-linear` 和 `R7-nonneg`，不直接比较为同目标胜负。 |
| 五数据集协议报告的证据标签写成“单 seed 初证”，但结果表为空。 | `uci_paper_benchmark_report_zh.md` 中无结果，仓库无 `*_paper_*_seed0.json`。 | 归类为“协议已准备、结果未完成”。 |
| UCI m=96 初始化报告与当前 JSON 内容不一致。 | `program_progress.md` §5.2.2 对比 `uci_ttde_vs_ttns_metrics_rank1init.json` 和默认 JSON。 | 只作为报告级历史负结果，不列入有完整 JSON 的主表。 |
| UCI JSON 中平方模型字段名为 `nonpos_rate`。 | POWER/HEPMASS 的 TTDE 和 TTNSDE 给出完全相同的小比例，报告解释为支撑外 `log_p=-inf`。 | 解释为非有限/支撑外比例，不解释为平方密度为负。 |
| Budget 报告写“密度达到 oracle 的 95–98%”。 | 原始量是每维 LL gap 0.017–0.051 nat。 | 改写为指数化逐维 LL 比约 98.3%–95.0%。 |
| `ttde_ttns_vs_tt_metrics.json` 中“matched” TT rank 64 参数仍与 TTNS rank 8 略有差异，且 test LL 12.47 低于 rank 8 的 12.52。 | JSON 中参数为 1,575,936 vs 1,596,096。 | 只支持大幅增大 chain 容量未消除该单 seed 差距，不写成严格等参数或统计结论。 |

## 11. 图表素材清单

### 11.1 建议主文使用

| 图或表 | 文件 | 用途 | 注意事项 |
|---|---|---|---|
| Dense 最终解析方案 | `project_final_solution.png` | 展示 R5-nonneg 的 L4 边缘、误差和最强相关对 | 标注 `seed=0` 和 attempt 010 配置。 |
| Dense 原始 R5/R7 对照 | `dense_dag_r567_results.png`、`dense_dag_r567_slice_refined.png` | 展示线性 R5 后层崩溃及 R7 对照 | LL 主值使用主 JSON；切片图入口使用 three-way `--plot`。 |
| Budget 趋势 | `budget_sweep_layered.png` | rank、m、n_fit 单旋钮趋势 | 标注所有点为 `seed=0`，不要画置信区间。 |
| Budget 误差诊断 | `budget_sweep_error_viz.png` | oracle gap 和 corr 误差 | 解释逐维指数化 LL 比。 |
| 12D layered 总览 | `global_vs_layered_overview.png` | 基础结构先验收益 | 明示已知 DAG/kernel。 |
| 24D complex 总览 | `global_vs_layered_complex_overview.png` | 多峰和更深图结果 | 明示单次配置。 |
| 28D 完整联合 | `per_layer_all_methods_fulljoint.png` | layered vs 全局方法的主口径 | 同时在正文解释逐层边缘表。 |
| UCI POWER/GAS bars | `uci_power_ttde_vs_ttns_bars_ncomps8fix_power.png`、`uci_gas_ttde_vs_ttns_bars_ncomps8fix.png` | 单 seed 真实数据初证 | 不与 paper preset 混用。 |
| UCI POWER/GAS 2D slices | `uci_power_ttde_vs_ttns_slices_ncomps8fix_displaynorm.png`、`uci_gas_ttde_vs_ttns_slices_ncomps8fix_displaynorm.png` | 定性展示 | 图是有限窗口 display-normalization，不能读作原始积分。 |

### 11.2 建议附录或消融使用

`project_original_metrics.png` 和 `project_original_details.png` 适合展示修复前失效；`project_fixed_topology_control.png` 适合固定链近参数量对照；`dense_dag_r567_remedy_metrics.png` 适合比较 `marginal_l2_weight`、R5-nonneg 和 R7-nonneg；`budget_sweep_slice_refined.png` 与 `budget_sweep_slice_refined_uniform.png` 适合边缘诊断；`per_layer_all_methods_metrics.png` 和 `per_layer_all_methods_marginals.png` 适合解释完整联合与逐层边缘口径的差别。

Theta 重参数化目前没有独立图。Fork DAG 和随机树的主结果报告中有完整数表和 JSON，但本清单未发现专门的多 seed 汇总图；投稿前可只从已提交 JSON 重新绘制，不需要重跑实验。

## 12. 复现入口

| 实验 | 入口 | 主要结果文件 |
|---|---|---|
| 7D/8D Chow–Liu vs chain | `experiments/fit_dag_chow_liu_vs_chain.py` | `dag_chow_liu_vs_chain*_multiseed_metrics.json` |
| 随机树 matched vs chain | `experiments/fit_matched_vs_chain_random_tree_multiseed.py` | `random_tree_matched_vs_chain_multiseed_metrics.json` |
| Layered vs global | `experiments/compare_global_vs_layered_plot.py` | `global_vs_layered*_metrics.json` |
| 28D 五模型 | `experiments/per_layer_all_methods.py` | `per_layer_all_methods_metrics.json` |
| CDF vs sampling | `experiments/fit_maxplus_cdf_vs_sampling.py` | `maxplus_cdf_vs_sampling_metrics.json` |
| Dense R5/R7 原始对照 | `experiments/dense_dag_r567_three_way.py` | `dense_dag_r567_three_way_metrics.json` |
| Dense 补救 | `experiments/dense_dag_r567_remedy_tests.py` | `dense_dag_r567_remedy_metrics.json` |
| Dense 最终 R5-nonneg | `simple_ttns_l2/autoresearch/r5_fix/run_attempt.py` 与 attempt 010 JSON | `simple_ttns_l2/autoresearch/r5_fix/artifacts/010/{manifest,metrics}.json` |
| Budget sweep | `experiments/budget_sweep_layered.py` | `budget_sweep_layered_metrics.json` |
| R6 小块 | `experiments/budget_r6_block_ref.py` | `budget_r6_block_ref_metrics.json` |
| Budget 全局基线 | `experiments/budget_baselines_ll.py` | `budget_baselines_ll_metrics.json` |
| Theta 消融 | `experiments/reparam_check.py` | `reparam_theta_report_zh.md` |
| UCI 小容量 | `experiments/uci_ttde_vs_ttns.py` | `uci_ttde_vs_ttns_metrics_ncomps8fix*.json` |
| UCI paper preset | `experiments/uci_ttde_vs_ttns.py --preset paper`、`experiments/run_uci_paper.sbatch` | 预期 `uci_ttde_vs_ttns_metrics_paper_<ds>_seed0.json`，当前不存在 |
| 全因子 study | `experiments/run_full_scale_study.sh` | 当前无最终结果 |

复现 L2 实验时应使用 `env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.<name>`，以避免导入仓库中的另一个 `ttde` 包。本文没有执行任何新的大型实验。

## 13. 缺失实验与投稿前优先级

1. 最高优先级是等待或重新提交五数据集 paper preset，并逐数据集保存唯一 tag 的 JSON。完成 `seed=0` 后先核对绝对 TTDE LL 与论文值的差距，再冻结协议补至少 3 seed；在此之前不要填正式 UCI 主表。
2. 第二优先级是对 Dense attempt 010、固定链 rank 13 和一个明确选定的 R7 对照补至少 3 seed。应预先固定 `m/rank/n_total/n_fit/an_lr/an_steps`，同时报告 L4 LL、`nonpos_rate`、标准差比、`corr_fro` 和墙钟时间。
3. 第三优先级是对 budget 的关键点而不是全部点补多 seed。建议至少复核 base、rank 16、m48 和 nfit80k，并报告均值和标准差；这样才能判断 rank 16 的甜点和 L4 单调改善是否稳定。
4. 第四优先级是把 Theta 消融结果保存为 JSON，并从已有结果生成图。该工作不需要大型实验；如果无法恢复逐 seed 数值，则运行同一小脚本即可，但不得与本次整理提交混合。
5. 第五优先级是运行已提交的全因子 study，或者在论文中明确删去任何暗示 `rank16+m48+nfit80k` 已被验证的表述。
6. R6 矩张量版和平方 TTNS 分层全链属于方法扩展。若论文主张需要“确定性且完整联合”或“R5/R7 两头都占”，它们是必需实验；若当前论文只主张树投影解析路线的可行性，则应把它们列为未来工作，而不是未完成结果。
7. 投稿前应统一 UCI 输出 tag，修复 `metrics_rank1init.json` 和默认 `metrics.json` 与报告描述不一致的问题。优先采用新增唯一文件，不应覆盖现有历史 JSON。

## 14. 最终可写入论文的保守结论

现有证据最稳妥的叙述是：三 seed 合成实验显示，匹配或数据驱动的树拓扑在树状和 fork DAG 依赖上持续优于链拓扑；在图结构和 delay kernel 已知的多层合成 DAG 中，结构感知的 layered 因子化以更少学习参数优于全局 TT、TTNS 和 TTDE；在 100 节点 dense DAG 的单 seed 基准中，非负解析 L2 与更合适的优化尺度修复了线性 R5 的后层负密度崩溃；在三个 UCI 数据集的小于论文协议容量的单 seed 混合实验中，TTNSDE 的 `test_LL` 均高于近似等参数量 TTDE，但这一结果尚不构成多 seed 或五数据集论文级结论。

当前不能写入摘要或结论的内容包括：五数据集 paper preset 已完成、TTNSDE 在 UCI 上统计显著优于 TTDE、R6 已在 100 节点全链验证、组合最大预算已经实际运行、或 Dense R5 的修复已证明跨 seed 稳定。以上主张均缺少对应的已提交最终 JSON。
