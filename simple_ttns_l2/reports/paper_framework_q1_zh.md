# 面向 SCI/SCIE 一区投稿的论文框架

> 本框架基于 `dev` 分支现有算法、理论说明和 `simple_ttns_l2/reports/paper_results_inventory_zh.md`。它是一份可直接扩写为英文论文的结构设计，不代表当前证据已经达到一区录用条件，也不承诺投稿结果。
>
> “一区”可能指 JCR Q1，也可能指中国科学院分区一区。正式选刊时必须按投稿当年的具体学科分类核对；本文先按照两者中较严格的完整性要求设计论文。

## 1. 当前判断

仓库中原先没有论文稿、LaTeX 骨架或论文提纲。现有材料已经能够支撑一篇完整论文的核心故事，但当前版本更接近“可形成论文初稿”，尚不是“一区可投终稿”。

现有工作的最强主线不是单纯宣称 TTNS 比 TT 更强，而是：针对已知多层 DAG 上的 max-plus 随机传播，提出结构感知的分层 TTNS 密度表示和 CDF 域解析传播方法；通过非负解析 L2 修复深层传播中的负密度失效；并用 R5、R6、R7 统一解释树投影、完整联合目标和采样近似之间的精度—计算取舍。

一区投稿前至少还需要完成以下四项关键补强。

1. 五数据集论文级 UCI 实验必须产生最终 JSON，并至少补 3 个 seed；若计算资源允许，建议使用 5 个 seed。
2. Dense 100 节点最终 R5-nonneg、固定链近参数量对照和选定 R7 对照必须补多 seed。
3. 真实数据实验需要加入同领域现代密度估计基线，或严格说明为何论文只比较 TTDE 系方法。仅比较 TTDE 与 TTNSDE 通常不足以支撑宽泛的“先进密度估计方法”论断。
4. 理论部分必须从当前推导整理成正式的定义、命题、定理、证明和复杂度结论，并把“已证明”和“经验观察”分开。

## 2. 推荐论文定位

### 2.1 一句话问题定义

本文研究如何在具有已知层间依赖和 max-plus 随机传播机制的多层 DAG 上，以可扩展、非负且结构感知的张量网络近似逐层和全联合密度。

### 2.2 一句话方法定义

本文以每层 Chow–Liu TTNS forest 表示层内联合密度，在 CDF 域解析收缩层间 max-plus kernel，并以非负解析 L2 拟合下游 TTNS，从而避免逐层 Monte Carlo 采样和有符号密度截断造成的误差累积。

### 2.3 一句话结果定义

现有证据表明，结构匹配 TTNS 在三 seed 合成树/DAG 目标上稳定优于 chain；结构感知的 layered 模型在已知 DAG/kernel 的合成基准上优于全局 TT/TTNS/TTDE；非负解析 L2 在单 seed 的 100 节点 dense DAG 上修复了线性 R5 的深层负密度崩溃；小容量 UCI 单 seed 结果显示 TTNSDE 相对 TTDE 为正，但论文级多 seed结果尚未完成。

### 2.4 论文类型

建议定位为“方法论文 + 理论推导 + 系统实验”，而不是单纯应用论文或纯 benchmark 论文。主要创新应落在结构化密度传播算法和可解析性上，UCI 结果用于证明方法不局限于合成数据。

## 3. 备选题目

### 3.1 推荐英文题目

**Structure-Aware Tensor-Tree Density Propagation on Multilayer DAGs**

副标题可选：**Analytic Max-Plus Contractions with Nonnegative L2 Fitting**

### 3.2 更强调算法的英文题目

**Analytic Density Propagation with Tree Tensor Networks for Multilayer Max-Plus DAGs**

### 3.3 更强调稳定性的英文题目

**Stable Nonnegative Tensor-Tree Density Estimation and Analytic Propagation on Deep DAGs**

### 3.4 中文工作题目

**面向多层 max-plus DAG 的结构感知树张量网络解析密度传播方法**

题目中暂不使用 “state-of-the-art”、“optimal” 或 “universal”，因为现有证据不能支持这些措辞。

## 4. 核心研究问题

全文应围绕四个明确问题展开。

1. 树结构张量网络相对链式张量列车，能否更有效地表示树状或近树状依赖？
2. max-plus 层间传播能否在 TTNS 上解析计算，而不依赖逐层采样？
3. 解析传播进入深层后，如何避免线性 L2 TTNS 的负密度、欠分散和相关塌缩？
4. 树投影 R5、完整联合 R6 和采样近似 R7 在精度、相关恢复和计算成本上如何取舍？

UCI 实验回答的是第五个外部有效性问题：结构化平方 TTNSDE 在标准真实密度估计数据上，能否稳定达到或超过近参数量链式 TTDE？

## 5. 建议贡献点

最终摘要和引言建议保留 3 至 4 个贡献点，不能超过现有理论和实验的实际覆盖范围。

### 5.1 贡献一：分层 TTNS 表示

提出一种面向多层 DAG 的结构感知密度表示。每层按依赖关系分块，并用 Chow–Liu tree TTNS forest 表示层内密度；层间条件结构由已知 max-plus kernel 连接。该设计把全局高维密度估计分解为可控的局部树张量收缩。

需要说明的边界是：当前主实验假定 DAG 结构和 delay kernel 已知。不能在贡献中暗示同时完成了图结构学习和 kernel 估计。

### 5.2 贡献二：CDF 域解析传播

推导 max-plus 传播下的一维 marginal 和二维 pair CDF 的可分离期望，并利用 TTNS 收缩解析计算这些量。与采样传播相比，该方法避免对有符号密度进行截断，并在现有机制实验中降低相关矩阵误差。

正式论文需要给出清晰的输入、输出、收缩顺序和复杂度，而不能只给公式直觉。

### 5.3 贡献三：非负解析 L2 拟合

提出或系统化使用 core `raw**2` 的非负解析 L2 拟合，使模型密度非负，同时保留解析交叉项和 TTNS 收缩。Dense 100 节点实验显示，该方法配合合适的优化尺度可以消除下游层非正密度并恢复 LL 和相关结构。

必须把该方法与以下两项区分：线性 L2 下对已有 core 做一般 `theta->theta^2/exp(theta)` 重参数化的负结果，以及平方密度 $p=\psi^2/Z$ 的 MLE 模型。三者的目标函数和数学解释不同。

### 5.4 贡献四：R5/R6/R7 统一比较

在同一单父 TTNS forest 结构下，系统比较树投影解析目标 R5、完整联合解析目标 R6 和采样求 L2 的 R7，并通过 budget sweep 分离 rank、基函数数目 `m` 和拟合样本数 `n_fit` 对密度、相关和深层稳定性的影响。

该贡献目前的限制是 R6 只在小块验证。若投稿时仍没有大规模 R6，贡献表述必须限定为“小块上的完整联合参照”。

## 6. 摘要框架

英文摘要建议控制在 180 至 230 词，分为五个逻辑句群。

1. 背景句：高维密度在多层 DAG 上传播时同时面临结构错配、维度增长和 Monte Carlo 误差累积。
2. 缺口句：链式 TT 难以匹配树状依赖，而逐层采样会在有符号近似下损失相关结构。
3. 方法句：提出 layered TTNS forest、CDF 域解析 max-plus contraction 和非负解析 L2 fitting。
4. 结果句：只填入最终冻结后的多 seed 结果，至少包括一个结构收益数字、一个 Dense 数字、一个真实数据汇总数字和一个计算成本数字。
5. 结论句：说明方法适用于已知 DAG/kernel 的结构化密度传播，不把结论扩展到未知图的一般密度估计。

当前不可直接放入摘要的数字包括 UCI paper preset 的任何空结果、多 seed 尚未完成的 Dense 显著性结论，以及尚未运行的最大组合预算结果。

## 7. 正文结构

建议全文正文约 8,000 至 10,000 个英文单词，不含参考文献和补充材料。具体页数应服从目标期刊模板。

### 7.1 Introduction

#### 第一段：问题背景

介绍高维联合密度在可靠性传播、网络延迟、风险传递或结构化随机系统中的作用。说明多层 DAG 中的下游变量由共享父节点和随机 delay 共同决定，导致层内相关和跨层依赖快速复杂化。

#### 第二段：现有方法局限

说明全局黑盒密度模型忽略已知图结构，参数和样本效率较低；链式 TT 偏置于线性顺序；逐层 Monte Carlo 传播会产生采样噪声，并可能在有符号密度近似中引入截断误差。

#### 第三段：核心思想

提出每层 TTNS forest、解析 CDF contraction 和非负解析 L2 的总体思想。这里应配一张概念图，而不是立即进入所有 R5/R6/R7 细节。

#### 第四段：主要结果

按“结构收益—解析传播—Dense 稳定性—真实数据”顺序总结结果。所有单 seed 数字必须显式标注，最终投稿前替换为均值和标准差或置信区间。

#### 第五段：贡献列表

使用 3 至 4 个项目列出第 5 节的贡献。贡献必须是方法或知识增量，不能把“实现了代码”单独列为贡献。

### 7.2 Related Work

建议分为四个子节。

1. Tensor Train、TTDE 和 tensor-network density estimation。
2. Tree tensor networks、hierarchical Tucker 和结构化张量分解。
3. Graphical models、Chow–Liu tree 和结构化密度估计。
4. 不确定性传播、max-plus systems、CDF propagation 和 Monte Carlo alternatives。

相关工作必须回答三个审稿问题：TTNS 密度估计相对 hierarchical Tucker 有什么新意；解析 max-plus 传播相对已有可靠性/排队网络算法有什么差异；非负解析 L2 相对 nonnegative tensor factorization 和平方 MLE 有什么区别。

正式写作前需要做系统文献检索并更新引用。本框架不预填可能过时或未经核对的文献条目。

### 7.3 Problem Formulation

#### 7.3.1 Multilayer DAG

定义 $G=(V,E)$、层划分 $V=\cup_{\ell=0}^{L}V_\ell$、父节点集合 $\mathrm{pa}(v)$，并明确层内无有向边、边只从上一层或更上层指向下层的具体假设。

#### 7.3.2 Max-plus stochastic transition

定义

$$
y_v=\max_{u\in\mathrm{pa}(v)}(x_u+e_{uv})+d_v,
$$

说明 edge delay $e_{uv}$ 和 node delay $d_v$ 的独立性、支持区间和已知性假设。

#### 7.3.3 Learning objective

区分三个目标：逐层密度近似、完整全联合密度和采样质量。定义 `joint_LL`、`corr_fro`、`W1_marg`、二维切片 IAE、`nonpos_rate` 和参数量。必须统一指标方向。

### 7.4 Tree Tensor Network Density Model

#### 7.4.1 TTNS representation

定义基函数向量 $b_v(x_v)$、树拓扑和 TTNS core，写出

$$
q_\theta(x)=\left\langle T_\theta,\bigotimes_{v=1}^{d}b_v(x_v)\right\rangle.
$$

说明 chain 是 TTNS 的特例，给出 TT 与 TTNS 参数量公式，并讨论高阶 hub 的计算代价。

#### 7.4.2 Data-driven topology

说明如何估计 pairwise mutual information、构建 Chow–Liu maximum spanning tree、选择 root 和形成 forest block。必须区分用于选树的 mutual information 与用于评估的 Pearson correlation。

#### 7.4.3 Linear, nonnegative and squared variants

并列表述线性 L2 TTNS、core `raw**2` 非负解析 L2 和平方 TTNSDE $p=\psi^2/Z$。建议用一张表列出非负性、目标函数、传播方式和计算代价。

### 7.5 Analytic Density Propagation

#### 7.5.1 Marginal CDF contraction

从

$$
F_v(s)=\mathbb E\left[\prod_{u\in\mathrm{pa}(v)}F_e(s-x_u)\right]
$$

推导可分离函数在 TTNS 上的收缩。明确离散网格误差来自哪里。

#### 7.5.2 Pair CDF and dependency reconstruction

推导两个下游变量的 pair CDF，说明共享父节点如何产生相关，并解释如何从 marginal/pair 统计构造下游 Chow–Liu tree target。

#### 7.5.3 Tree-target L2 contraction

正式给出交叉项 $\mathbb E_{p_{tree}}[q_\theta]$ 的树消息传递和平方项 $\int q_\theta^2$ 的 Gram contraction。算法框应列出从叶到根的消息和复杂度。

#### 7.5.4 Nonnegative analytic fitting

定义 $C_i=R_i^2$，说明非负 B-spline 基下 $q_\theta(x)\ge0$。解释为何该参数化改变优化尺度，以及归一化步骤如何保持积分为 1。

#### 7.5.5 R5, R6 and R7

给出统一流程图：R5 使用树投影目标并确定性传播；R6 使用完整联合目标并进行高维网格或矩张量收缩；R7 从完整联合目标采样并用 Monte Carlo 近似 L2。表中比较目标偏差、Monte Carlo 方差、复杂度和适用块大小。

### 7.6 Theoretical Analysis

理论部分建议至少包含以下正式结果。

#### 命题一：可分离函数的 TTNS 期望可解析收缩

证明对 $f(x)=\prod_v f_v(x_v)$，$\mathbb E_{q_\theta}[f(x)]$ 可转化为基函数投影后的 TTNS contraction。

#### 定理一：max-plus marginal 和 pair CDF 的解析可计算性

在 delay 独立且 CDF 已知的条件下，证明下游 marginal/pair CDF 可由上层 TTNS 的有限张量收缩得到。应明确“解析”是指不使用 Monte Carlo，而数值实现仍可能使用一维或二维网格积分。

#### 命题二：非负性与归一化

在基函数非负且 core 元素非负时，证明 $q_\theta(x)\ge0$；在积分有限且非零时，归一化后为合法密度。

#### 定理二或复杂度命题：计算复杂度

给出 marginal、pair、R5 tree contraction、R6 joint contraction 和 R7 sampling 的时间与内存复杂度，变量至少包括维度 $d$、块大小 $K$、基函数数 $m$、rank $r$、网格数 $G$ 和样本数 $n_{fit}$。

#### 定理三：数值误差分解

`maxplus_ttns_cdf_theorem_zh.md` 已给出局部投影的多线性扰动界、node-delay
离散的 Wasserstein-1 界，以及由 TTNS 建模误差、局部积分误差、node-delay
求积误差、插值误差和尾部误差组成的单层 CDF 总误差分解。正式论文应保留其
明确假设，并说明 Markov kernel contraction、Kantorovich–Rubinstein bound
和一般多线性扰动界是标准工具；可主张的是它们与本文 TTNS max-plus 收缩流程的
组合，而不是这些标准不等式本身。

当前 `squared_ttns_theory_zh.md` 可作为理论整理的起点，但正式论文必须压缩为与主方法直接相关的定理，避免把尚未系统实验的平方全链写成已完成贡献。

### 7.7 Experimental Protocol

#### 7.7.1 Research questions

实验章节开始先列出 RQ1 至 RQ5，并让每个表或图明确回答其中一个问题。

- RQ1：拓扑匹配是否改善密度近似？
- RQ2：解析传播是否优于采样传播？
- RQ3：非负解析 L2 是否修复深层失效？
- RQ4：R5/R6/R7 和预算旋钮如何取舍？
- RQ5：方法能否迁移到标准真实数据？

#### 7.7.2 Datasets

合成数据包括 7D/8D fork DAG、6D 随机树、12D/24D/28D layered DAG 和 100 节点 dense DAG。真实数据包括 POWER、GAS、HEPMASS、MINIBOONE 和 BSDS300。每个数据集必须报告维度、样本划分、预处理和是否使用已知结构/kernel。

#### 7.7.3 Baselines

最低基线集合应包括 global TT、global TTNS、TTDE chain、TTNSDE MI-tree、layered sampling、R5、R6 小块和 R7。真实 UCI 若要面向广泛密度估计期刊，建议再加入至少两类现代 flow 或 autoregressive baseline；若无法运行，应引用严格同预处理和同数据划分的公开结果，并清楚标注“reported result”而非本地复现。

#### 7.7.4 Fairness

分别报告参数量、训练步数、训练样本数、wall-clock、硬件和峰值内存。把“近参数量”、“同训练预算”和“同结构先验”分开，不使用含糊的 “fair comparison”。

#### 7.7.5 Statistics

主结论建议使用至少 5 seed；资源受限时最低为 3 seed。报告 mean、standard deviation 和 95% bootstrap confidence interval。对同 seed 配对实验可报告配对差值，但样本数过小时不要滥用显著性检验。所有超参数选择规则应在看 test set 前固定。

### 7.8 Results

#### 7.8.1 Topology matters

主表报告 7D、8D 和随机树三 seed IAE，配一张拓扑示意图和一张相对改善图。现有可引用平均改善为 56.9%、64.4% 和 42.6%，来源见实验清单。

#### 7.8.2 Layered structure vs global models

报告 12D、24D 和 28D 的 `joint_LL`、`W1_marg`、`corr_fro`、参数量和时间。正文必须强调结构和 delay kernel 已知，并同时解释完整联合与逐层边缘的评测口径差异。

#### 7.8.3 Dense 100-node propagation

以 R5-linear、R5-nonneg、固定 chain 和一个预先选定的 R7 对照为主表。主指标应包括逐层 LL、`nonpos_rate`、标准差比和 `corr_fro`。投稿版用多 seed 聚合替换当前 attempt 010 的单 seed 数字。

#### 7.8.4 Budget and scalability

报告 rank、`m`、`n_fit` 的作用、训练时间、内存和随层数/节点数的 scaling。不要把未运行的 `rank16+m48+nfit80k` 写成已有结果。

#### 7.8.5 UCI benchmark

主表严格使用 paper preset 的最终 JSON。每行报告 TTDE、TTNSDE、差值、参数量、训练时间和 seed 聚合。小容量 `m=128,n_comps=8` 结果可放在附录或作为开发阶段证据，不能与 paper preset 混表。

### 7.9 Ablation Studies

建议至少包含六项消融。

1. Chow–Liu、balanced 和 chain 拓扑。
2. R5-linear 与 R5-nonneg。
3. R5、R6 小块和 R7 的目标口径。
4. `rank`、`m` 和 `n_fit` 的独立 budget sweep。
5. `marginal_l2_weight` 负结果。
6. 线性 L2 下 `identity/square/exp` 的 theta 重参数化负结果。

如果篇幅允许，再加入 learned topology 与 oracle topology、不同 delay 分布、不同层数/节点数和不同 `an_lr` 的稳定性消融。负结果应进入正文或补充材料，不能只保留成功配置。

### 7.10 Discussion and Limitations

建议主动说明以下限制。

- 主 synthetic benchmark 假定 DAG 结构和 delay kernel 已知。
- Chow–Liu tree 只能保留树宽为 1 的依赖，环状或高阶依赖会产生结构偏差。
- 高度数 hub 的 TTNS core 随 rank 指数增长。
- R6 完整联合目标存在 $O(G^K)$ 或相关的维度灾难，目前只适合小块。
- 现有 Dense 和 budget 证据仍需多 seed。
- UCI 的绝对 LL 尚未完成 paper preset 对齐。
- CDF 网格带来离散化误差，当前没有完整误差上界。

主动说明限制通常比让审稿人指出更有利，但每个限制后应给出可行缓解方向。

### 7.11 Conclusion

结论只回答四个研究问题，不引入新结果。最后一句可指向未知图/kernel 联合学习、矩张量 R6 和自适应 forest 分块等未来工作。

## 8. 主文图表设计

建议主文控制在 7 至 9 张图、4 至 6 张表，其余进入补充材料。

### Figure 1：方法总览

展示 multilayer DAG、每层 TTNS forest、解析 CDF propagation 和下游 nonnegative fitting。该图应是新绘制的论文级矢量图，而不是直接使用实验调试图。

### Figure 2：TT、TTNS 与 layered forest 的结构差异

用同一小型依赖图展示 chain 缺边、Chow–Liu tree 和分层 block forest，解释结构归纳偏置。

### Figure 3：解析传播算法

展示 marginal/pair CDF 的消息收缩，配合算法框和符号说明。

### Figure 4：三 seed 拓扑收益

汇总 7D、8D 和随机树的 IAE，显示每个 seed 的点和均值，而不是只画均值柱状图。

### Figure 5：Layered vs global

使用 `global_vs_layered_complex_overview.png` 的信息重新绘制统一风格版本，并同时显示参数量与三项误差。

### Figure 6：Dense 逐层稳定性

重新绘制 R5-linear、R5-nonneg、chain 和 R7 的 L0–L4 LL、`nonpos_rate` 和 `corr_fro`。当前 `project_final_solution.png` 可作为素材，但投稿版应加入多 seed 阴影。

### Figure 7：Budget trade-off

基于 `budget_sweep_layered_metrics.json` 展示 rank、`m` 和 `n_fit` 的独立影响。多 seed 完成后加入误差条。

### Figure 8：UCI 主结果

只使用 paper preset 多 seed 结果。可以是每数据集 TTNSDE−TTDE 差值和置信区间，也可以用表代替图。

### Table 1：符号和方法对照

列出 global TT、global TTNS、TTDE、TTNSDE、R5、R6 和 R7 的结构、参数化、训练目标、是否解析、是否非负和复杂度。

### Table 2：合成数据配置

统一列出维度、层数、节点数、fanin、delay、样本数、`m`、rank 和 seed。

### Table 3：主 synthetic 结果

合并拓扑实验和 layered/global 的核心指标，详细逐层结果放补充材料。

### Table 4：Dense 100 节点结果

报告多 seed mean±std、95% CI、参数量、时间和内存。

### Table 5：UCI paper benchmark

五数据集、至少 3 seed，区分本地复现和文献报告值。

## 9. 补充材料结构

补充材料建议包含以下内容。

1. 所有定理的完整证明和额外引理。
2. TTNS contraction 的伪代码和实现细节。
3. 全部数据生成参数和随机 seed。
4. 逐层完整指标表、所有 marginal slices 和相关热图。
5. Theta、`marginal_l2_weight`、初始化失败和 finite validation 修复等负结果。
6. UCI 预处理、支持区间、nonfinite rate 和超参数。
7. 计算环境、硬件、wall-clock 和内存。
8. 代码与数据可用性声明，以及从 JSON 生成论文表图的脚本。

## 10. 一区审稿风险清单

### 风险一：创新点被认为是已知技术组合

缓解方式是明确指出新的数学接口：max-plus CDF 的可分离函数如何与 TTNS contraction 对接，以及非负解析 L2 如何在不进行逐层采样的条件下材料化下游密度。需要用定理、算法和复杂度支撑，而不是只用实验命名 R5。

### 风险二：比较条件不公平

Layered 方法使用已知图和 kernel，而全局基线没有这些先验。论文必须把目标定义为“已知结构下的密度传播”，并增加一个同样使用结构先验的合理 baseline，或者清楚解释全局基线回答的是参数效率问题。

### 风险三：真实数据不足

当前小容量 UCI 单 seed 不足以支撑一区主结果。必须完成 paper preset、多 seed 和现代基线；如果真实数据仍明显欠拟合，应把论文定位收窄为 structured stochastic propagation，而不是 general-purpose density estimation。

### 风险四：单 seed 和选择性报告

Dense、budget 和 layered 结果目前主要是单 seed。投稿前应固定超参数和 seed 列表，报告所有完成运行和失败情况，并让表图由最终 JSON 自动生成。

### 风险五：理论与实现脱节

`squared_ttns_theory_zh.md` 中的平方全链解析性尚未完全对应系统实验。正文只保留已实现且验证的理论；其余放讨论或未来工作。

### 风险六：过度声称

禁止使用以下未被现有证据支持的表述：统计显著优于 TTDE、在所有五个 UCI 数据集领先、R6 已扩展到 100 节点、组合最大预算已验证、未知图上同样有效、或方法达到 state of the art。

## 11. 投稿前最低完成标准

只有同时满足以下条件，才建议进入一区期刊投稿流程。

- 论文级 UCI 五数据集结果全部完成，并有唯一命名的最终 JSON。
- 至少三个关键实验族具有 3 至 5 seed：拓扑、Dense、UCI；budget 至少复核关键点。
- 加入现代真实数据基线，或将题目和摘要明确收窄到结构化传播。
- 理论部分至少包含两个正式命题、一个核心定理和复杂度分析，并完成逐条证明核查。
- 所有主表数字能从已提交 JSON 自动生成，图中不手工抄数。
- 报告参数量、时间、硬件和内存，并解释 oracle 结构/kernel 的信息优势。
- 完成方法消融、失败模式、限制和复现说明。
- 英文稿经过领域术语、数学符号、统计报告和母语表达四轮检查。

## 12. 推荐写作顺序

1. 先冻结 Problem Formulation、方法符号和 R5/R6/R7 命名。
2. 再写 Method 和 Theoretical Analysis，确保每个贡献有公式或算法载体。
3. 从现有 JSON 自动生成 provisional 表图，缺失结果保留 `TBD`，禁止估算。
4. 完成 Dense 和 UCI 多 seed 后冻结主结果表。
5. 根据最终证据写 Introduction、Abstract 和 Conclusion。
6. 最后选择目标一区期刊，并按其 scope、篇幅、图表和开放数据政策调整稿件。

摘要和引言应最后写，因为它们必须准确反映最终完成的证据，而不是项目计划。

## 13. 与仓库证据的对应

| 论文部分 | 当前主要来源 |
|---|---|
| 问题和算法总览 | `ALGORITHM_zh.md`、`clarify.md` |
| TTNS L2 和实现 | `simple_ttns_l2/README.md`、`simple_ttns_l2/objective.py`、`simple_ttns_l2/analytic_tree_fit.py` |
| 平方/非负理论 | `squared_ttns_theory_zh.md`、`simple_ttns_l2/reparam.py` |
| 证据、数字和边界 | `simple_ttns_l2/reports/paper_results_inventory_zh.md` |
| Dense 主结果 | `simple_ttns_l2/autoresearch/r5_fix/artifacts/010/`、`simple_ttns_l2/reports/ttns_multilayer_dag_project_report_zh.md` |
| Budget 和 R6/R7 | `simple_ttns_l2/reports/budget_sweep_layered_metrics.json`、`budget_r6_block_ref_metrics.json` |
| UCI | `simple_ttns_l2/reports/uci_ttde_vs_ttns_metrics*.json`、`uci_paper_benchmark_report_zh.md` |

## 14. 最终建议

建议以“多层 DAG 上的结构化解析密度传播”为论文中心，以 TTNS 相对 TT 的拓扑收益作为基础，以 R5-nonneg 的 Dense 修复作为方法成败的关键证据，以 UCI 作为外部有效性验证。不要把论文中心改成“在所有通用密度估计任务上击败 TTDE”，因为当前真实数据证据尚不足，而且最有辨识度的创新是图结构与解析传播的结合。

如果投稿前能完成多 seed、现代基线、复杂度定理和自动化复现链，这一工作具备冲击 SCI/SCIE 一区期刊的论文形态；如果这些项目未完成，更稳妥的定位是较窄的结构化概率建模或张量网络方法期刊，而不是宽泛的顶级机器学习/统计学习期刊。
