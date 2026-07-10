# TTNS 多层 DAG 密度估计项目报告

## 摘要

本项目的目标是把张量列车（Tensor Train, TT）密度估计推广到树结构张量网络（Tree Tensor Network State, TTNS），再进一步用于多层 DAG 的密度传播。简单说，我们希望用一棵树或一组树来表示高维联合密度，并让这种表示能够沿着 DAG 的层间因果结构向下传播。

目前项目已经完成三件关键事情：

1. 在单层合成目标上验证 TTNS 比链式 TT 更适合表达树状或 DAG 状依赖。
2. 用 Chow-Liu 方法从数据中自动学习层内树结构，避免手工指定拓扑。
3. 在多层 dense DAG 上实现并修复了全解析传播链。最终 attempt 010 证明：不需要逐层 sampled propagation，纯解析 `UpperForest` 传播加非负解析 L2 拟合也可以达到目标。

最终 dense DAG 修复结果为：L4 `joint_LL=6.23`，`nonpos_rate=0`，`std_ratio=0.952`，最强相关 model `r=+0.534` vs GT `+0.758`。这说明多层解析 TTNS 路线是可行的。

## 1. Algorithm

### 1.1 用 TTNS 表示密度

TTNS 用树结构连接多个一维基函数。给定样本 $x=(x_1,\dots,x_d)$，模型密度写成

$$
q_\theta(x)=\langle T_\theta,\bigotimes_{k=1}^d b_k(x_k)\rangle .
$$

这里 $b_k(x_k)$ 是第 $k$ 个变量上的样条基，$T_\theta$ 是树结构张量网络。相比普通 TT 链，TTNS 可以让变量按照树连接；如果目标分布的依赖也接近树结构，TTNS 通常比链更省参数、更准确。

训练主要有两条路线：

- L2 路线：最小化 $\int q_\theta^2 - 2\mathbb{E}_{p}[q_\theta]$。
- MLE 路线：最大化样本似然，并用平方或非负参数化保证密度非负。

本项目中多层 DAG 的最终修复使用的是非负解析 L2：core 写成 $C_i=R_i^2$，从而保证块内密度非负。

### 1.2 层内树结构

单层高维密度的关键问题是：树怎么选？

项目比较了三种结构：

- `chain`：固定链结构，也就是传统 TT。
- `balanced`：固定平衡树。
- `Chow-Liu`：从训练样本估计两两互信息，再取最大生成树。

实验表明，Chow-Liu 是目前最稳的选择。它能从数据中自动找到主要依赖边，不需要人工猜拓扑。

### 1.3 多层 DAG 传播

多层 DAG 中，节点按层排列。上一层变量 $x_u$ 通过 max-plus 机制生成下一层变量 $y_v$：

$$
y_v=\max_{u\in \mathrm{pa}(v)}(x_u+e_{uv})+d_v .
$$

其中 $e_{uv}$ 是 edge delay，$d_v$ 是 node delay。项目采用的解析传播思路是：不从上一层模型采样，而是在 CDF 域计算下一层分布。

对一个下层节点 $v$，其 CDF 可写成

$$
F_v(s)=\mathbb{E}\left[\prod_{u\in \mathrm{pa}(v)}F_e(s-x_u)\right],
$$

再卷积 node delay。由于括号内是对每个父变量的可分离函数，这个期望可以通过上一层 TTNS forest 的张量收缩解析计算。两个下层节点的 pair CDF 也可以用同样方式计算。

这就是 `UpperForest` 的作用：把上一层 TTNS forest 当成上层分布，用解析 CDF 传播到下一层。

### 1.4 每层如何拟合

每一层不是用一棵大树表示全部 20 个节点，而是按 DAG 结构拆成若干块。每个块内构造一个树目标：

1. 对块内每个节点计算解析 marginal。
2. 对块内每对节点计算解析 pair CDF。
3. 用 pair 信息选一棵块内树。
4. 构造树目标

$$
p_{\mathrm{tree}}(y)=p_r(y_r)\prod_{v\ne r}p(y_v\mid y_{\mathrm{pa}(v)}).
$$

然后用 TTNS 拟合这个树目标。最终推荐做法是：

- 使用非负 core：$C_i=R_i^2$。
- 使用解析 L2 目标。
- 使用较大的学习率 `an_lr=1e-3`。
- 训练 3000 步。

初始化时先用每个变量的一维边缘做 rank-1 初始化。因此初始模型的边缘大致正确，但相关基本为零；训练的主要任务就是把块内相关结构学出来。

### 1.5 树边上的收缩公式

这一节解释一个关键点：解析 L2 拟合为什么不用采样。我们要最小化

$$
L(\theta)=\int q_\theta(y)^2\,dy-2\mathbb{E}_{p_{\mathrm{tree}}}[q_\theta(y)].
$$

右边有两项。第一项是模型自己和自己的内积 $\int q_\theta^2$；第二项是目标分布和模型的交叉项 $\mathbb{E}_{p_{\mathrm{tree}}}[q_\theta]$。两项都能在树上从叶子往根收缩。

先看交叉项。块内树目标是

$$
p_{\mathrm{tree}}(y)
=
p_r(y_r)\prod_{v\ne r}p(y_v\mid y_{\mathrm{pa}(v)}),
$$

其中 $r$ 是树根，$\mathrm{pa}(v)$ 是 $v$ 在块内树上的父节点。TTNS 模型可理解为：每个变量 $y_v$ 先过一维基函数 $b_{v,i}(y_v)$，然后所有节点的 core 通过树边上的隐藏指标 $\alpha$ 连起来。

如果把树从根 $r$ 定向到叶子，则每个非根节点 $v$ 有一条连向父节点的边。记这条边上的隐藏指标为 $\alpha_v$。节点 $v$ 的 core 写作

$$
C_v(\alpha_v,i_v,\{\alpha_c:c\in \mathrm{ch}(v)\}),
$$

其中 $i_v$ 是物理基函数指标，$\mathrm{ch}(v)$ 是 $v$ 的子节点集合。根节点没有父边，所以根的 $\alpha_r$ 固定为 0。

目标是计算

$$
\mathbb{E}_{p_{\mathrm{tree}}}[q_\theta]
=
\int q_\theta(y)\,p_{\mathrm{tree}}(y)\,dy.
$$

直接做这个积分是高维积分。收缩算法的想法是：先把每个子树都压缩成一个“消息”，再把消息交给父节点。消息 $M_{v\to u}$ 表示“以 $v$ 为根的整棵子树已经积分掉，只留下父变量 $y_u$ 和边指标 $\alpha_v$”。

如果 $v$ 是叶子，它没有子节点，消息就是

$$
M_{v\to u}(y_u,\alpha_v)
=
\int
\sum_{i_v}
C_v(\alpha_v,i_v)\,b_{v,i_v}(y_v)\,
p(y_v\mid y_u)\,dy_v.
$$

这句话的含义很朴素：把叶子变量 $y_v$ 积掉，只留下它对父节点 $u$ 的影响。

如果 $v$ 不是叶子，就先接收所有孩子 $c$ 传来的消息 $M_{c\to v}$。在给定 $y_v$ 和 $\alpha_v$ 时，节点 $v$ 连同所有孩子子树的贡献是

$$
G_v(y_v,\alpha_v)
=
\sum_{i_v,\{\alpha_c\}}
C_v(\alpha_v,i_v,\{\alpha_c\})
b_{v,i_v}(y_v)
\prod_{c\in \mathrm{ch}(v)}
M_{c\to v}(y_v,\alpha_c).
$$

然后把 $y_v$ 也通过条件密度 $p(y_v\mid y_u)$ 积掉，得到传给父节点 $u$ 的消息：

$$
M_{v\to u}(y_u,\alpha_v)
=
\int
G_v(y_v,\alpha_v)\,p(y_v\mid y_u)\,dy_v.
$$

这样从叶子一路传到根。到根节点时，没有父节点了，只剩根变量 $y_r$。根节点先把所有孩子消息合并：

$$
G_r(y_r,0)
=
\sum_{i_r,\{\alpha_c\}}
C_r(0,i_r,\{\alpha_c\})
b_{r,i_r}(y_r)
\prod_{c\in \mathrm{ch}(r)}
M_{c\to r}(y_r,\alpha_c).
$$

最后用根边缘密度 $p_r(y_r)$ 做一维积分：

$$
\mathbb{E}_{p_{\mathrm{tree}}}[q_\theta]
=
\int G_r(y_r,0)\,p_r(y_r)\,dy_r.
$$

实际代码不是连续积分，而是在网格 $s_1,\dots,s_G$ 上求和。记

$$
B_v[g,i]=b_{v,i}(s_g),
$$

条件核矩阵记为

$$
P_{v\mid u}[g,h]\approx p(y_v=s_g\mid y_u=s_h).
$$

那么内部节点先算

$$
G_v[g,\alpha_v]
=
\sum_{i_v,\{\alpha_c\}}
C_v(\alpha_v,i_v,\{\alpha_c\})
B_v[g,i_v]
\prod_{c\in \mathrm{ch}(v)}
M_{c\to v}[g,\alpha_c],
$$

再把它乘上条件核矩阵，得到

$$
M_{v\to u}[h,\alpha_v]
\approx
\sum_{g=1}^{G}
G_v[g,\alpha_v]\,P_{v\mid u}[g,h]\,\Delta s.
$$

这就是 `_cross_term_fn` 的主要计算。每条边只传一个矩阵 $M_{v\to u}$，矩阵大小是 $G\times R$，其中 $G$ 是网格点数，$R$ 是树边 rank。

现在看第一项 $\int q_\theta^2$。它和上面很像，只是现在没有目标密度 $p_{\mathrm{tree}}$，而是模型 $q_\theta$ 和自己相乘。可以想象有两份相同的 TTNS：一份指标不加撇，另一份指标加撇。每个节点的物理变量 $y_v$ 被积分掉，得到基函数的 Gram 矩阵：

$$
H_v(i,j)=\int b_{v,i}(y)b_{v,j}(y)\,dy.
$$

叶子节点传给父节点的是一个二次型消息：

$$
S_{v\to u}(\alpha_v,\alpha_v')
=
\sum_{i,j}
C_v(\alpha_v,i)
C_v(\alpha_v',j)
H_v(i,j).
$$

内部节点也一样，只是要乘上所有孩子的二次型消息：

$$
S_{v\to u}(\alpha_v,\alpha_v')
=
\sum_{\substack{i,j\\ \{\alpha_c,\alpha_c'\}}}
C_v(\alpha_v,i,\{\alpha_c\})
C_v(\alpha_v',j,\{\alpha_c'\})
H_v(i,j)
\prod_{c\in \mathrm{ch}(v)}
S_{c\to v}(\alpha_c,\alpha_c').
$$

一路收到根后，根节点没有父边，根指标固定为 0。最后得到的标量就是

$$
\int q_\theta(y)^2\,dy.
$$

所以整个解析 L2 没有 Monte Carlo 采样：交叉项用 $M$ 消息沿树边收缩，平方项用 $S$ 消息沿树边收缩。优化时只需要对这些收缩结果反向传播即可。

## 2. Numerical Example

### 2.1 单层 DAG：Chow-Liu 优于 chain

早期实验先验证 TTNS 的基本价值。在 7 维 fork DAG 目标上，Chow-Liu TTNS 能自动命中 6/7 条 DAG 真边。三 seed 聚合结果显示：

- Chow-Liu vs chain 的关键切片 IAE 平均提升 **56.9%**。
- 单 seed 20260227 中，关键切片 mean IAE：Chow-Liu `0.3469`，chain `0.6953`。

在 8 维 fork DAG 目标上，Chow-Liu 命中 7/8 条 DAG 真边，效果更明显：

- Chow-Liu vs chain 的跨 seed 平均提升 **64.4%**。
- 单 seed 20260227 中，关键切片 mean IAE：Chow-Liu `0.2484`，chain `0.6858`。

这个阶段说明：如果目标分布存在树状或近树状依赖，TTNS 明显优于普通链式 TT；并且树结构最好从数据中学习，而不是固定为 chain 或 balanced。

### 2.2 解析传播优于直接采样传播的场景

在 max-plus 传播实验中，同一个上一层 TTNS 可以用两种方式传播到下一层：

- 方案 A：从上一层 TTNS 采样，再做 max-plus。
- 方案 B：用 CDF 域解析传播。

在一个三层小实验中，解析传播的相关误差更小：

- 采样传播 `corr_fro=0.1102`
- 解析传播 `corr_fro=0.0463`

原因是上一层 TTNS 若有少量负密度，采样时必须截断负值，这会削弱相关；解析传播是线性积分，不需要截断。

这一步给出了 R5 全解析链的动机：如果我们能解析传播，就应尽量避免逐层采样带来的截断误差。

### 2.3 dense DAG 上的原始问题

最终难点是 `dense_dag_r567`。这是一个 100 节点、5 层、每层 20 维的跨簇 dense DAG。后层存在较强跨簇相关。

最初的 R5 解析链在 L4 严重失效：

- L4 `joint_LL=-13.02`
- L4 `nonpos_rate≈0.57`
- L4 `std_ratio≈0.35` 或出现局部欠分散
- 最强相关对基本塌缩

R7 sampled 链可以作为参考上限，原始 audit 中 L4 `joint_LL≈+1.05`，后续非负 sampled/hybrid 版本 attempt 007 达到 L4 `joint_LL=7.07`，最强相关 model `r=+0.663` vs GT `+0.758`。

但 sampled/hybrid 不是我们最终想要的 R5 解析链，因为它把层间传播换成了采样传播。真正目标是修复解析 R5。

### 2.4 定位 R5 失败原因

我们先排查解析 target 是否错误。对 L1 的第一个 block 做直接比较：

- truth 最强相关约 `r≈0.69`
- 从上一层 forest 采样再 max-plus 传播得到 `r≈0.69`
- `UpperForest.pair_cdf` 解析得到 `r≈0.69`

三者一致，说明解析传播和块内 target 构造没有明显错误。

然后比较拟合器。rank-1 初始化几乎没有相关：

- target/truth 相关约 `r≈0.690`
- 初始化采样相关约 `r≈0.002`

默认低学习率 `an_lr=3e-5` 下，即使训练很多步，相关仍然很低：

- 训练后相关约 `r≈0.025`

但把学习率提高后，同一个目标、同一个初始化、同一个解析 L2 公式可以学出相关：

- `an_lr=1e-3`，1500 步：`r≈0.641`
- `an_lr=2e-3`，1500 步：`r≈0.663`

解析 L2 目标本身也没有坏。对从同一个 target 采样后用 MLE 得到的模型，有：

- `cross_analytic=33.018`
- `cross_mc=33.016`

解析交叉项和 Monte Carlo 估计几乎一致。因此问题不是公式错，而是非负参数化 $C_i=R_i^2$ 下，原学习率太小，相关通道增长太慢。

### 2.5 最终修复结果

attempt 009 把非负解析 L2 的学习率提高到 `an_lr=1e-3`，训练 1500 步，已经达标：

- L4 `joint_LL=5.89`
- L4 `nonpos_rate=0`
- L4 `std_ratio=0.937`
- L4 最强相关 model `r=+0.513` vs GT `+0.758`

attempt 010 在 009 基础上把训练步数增加到 3000，作为最终推荐：

- L4 `joint_LL=6.23`
- L4 `nonpos_rate=0`
- L4 `std_ratio=0.952`
- L4 `corr_fro_norm=0.231`
- L4 最强相关 model `r=+0.534` vs GT `+0.758`

010 相比 009 有小幅提升，训练后段 loss 已接近平台。因此推荐最终配置为：

- `variant=nonneg`
- `an_lr=1e-3`
- `an_steps=3000`
- 保持解析 `UpperForest` 传播
- 保持非负 core 参数化

## 3. 坑点

### 3.1 不能只看边缘

一些失败模型的一维边缘看起来并不差，`std_ratio` 也可能接近 1，但块内相关已经塌缩。dense DAG 的主要难点是联合结构，不是单个节点的边缘形状。因此必须同时看最强相关对和 `corr_fro_norm`。

### 3.2 target 错误和 fit 失败容易混淆

最终模型相关塌缩时，很容易怀疑解析传播公式错了。但 L1 block 诊断说明：truth、sampled propagation、analytic pair CDF 三者相关一致。也就是说 target 是对的，早期失败主要发生在拟合器。

### 3.3 非负参数化会改变优化尺度

非负 core 写成 $C_i=R_i^2$ 后，梯度尺度和线性 core 不一样。rank-1 初始化附近，相关结构依赖非 rank-1 通道；如果学习率太小，这些通道几乎长不起来。原始 `an_lr=3e-5` 就是这个问题。

### 3.4 sampled 方法不是最终答案，但能帮助定位问题

007 sampled/hybrid 效果最好，但它不是纯 R5 解析链。它的意义在于说明 TTNS 结构本身有能力表达相关。008 进一步说明，同一个解析 target 如果改用 MLE 拟合，也能学好。最后 009 和 010 证明，只要解析 L2 的优化配置正确，纯解析链也能达标。

### 3.5 `layer.corr` 的名字容易误导

`analytic_block_target(..., use_mi=True)` 中的 `layer.corr` 实际保存的是选树用的互信息权重矩阵，不是 Pearson 相关矩阵。直接拿它做相关惩罚是不对的。

### 3.6 导包路径必须小心

仓库里有两个 `ttde` 包：根目录的原始 TTDE，以及 `TTNSDE/ttde`。TTNS 相关实验必须使用 `TTNSDE/ttde`。如果导错包，可能不会立刻报错，但会运行旧代码。

### 3.7 性能瓶颈来自树结构本身

树节点如果有多个子节点，收缩比 chain 更贵。早期 junction 或 Chow-Liu 拓扑慢，并不一定是 bug，而是高扇出树的计算代价。后续通过 3-child einsum 和 hub 边降 rank 可以缓解，但不能指望树结构和 chain 一样快。

## 结论

本项目证明了三点。

第一，TTNS 相比链式 TT 更适合表达树状和 DAG 状依赖；Chow-Liu 可以从数据中自动学习有效树结构。

第二，多层 DAG 中的 max-plus 传播可以在 CDF 域解析完成，避免逐层采样截断带来的相关损失。

第三，dense DAG 上 R5 原始失效不是解析传播路线不可行，而是非负解析 L2 拟合的优化配置太保守。提高学习率并给足训练预算后，纯解析 R5 达到目标。

因此，当前推荐方案是：层内用结构块 TTNS forest，层间用 `UpperForest` 解析传播，块内用非负解析 L2 拟合，并采用 attempt 010 的训练配置。
