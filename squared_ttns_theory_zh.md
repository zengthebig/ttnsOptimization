# 平方（非负）TTNS 在分层 max-plus 框架下的解析性：完备理论推导

> **目的**：证明把每层/每块的密度从**线性 TTNS**（$p=q$）换成**平方 TTNS**（$p=\psi^2/Z$，天然非负）后，
> clarify.md 的「层间传递 merge」CDF 域**解析传播**（Scheme B）与**解析拟合**（解析 L2）**仍然成立**，
> 只是把「线性收缩」替换为「二次型收缩」、bond 维平方。给出严格命题、证明、复杂度与实现清单。
>
> **约定**：正文中文；公式用 `$` 定界；标识符/库名/文件名英文。与 `ALGORITHM_zh.md`（主纲）、`clarify.md`（模型定义）配套。

---

## 0. 结论摘要

设一层内一个**结构块**的变量为 $x=(x_1,\dots,x_K)$，每维基 $\{b^{(k)}_i\}_{i=1}^m$，块内为单父树（parent 函数 $\mathrm{pa}$，根 $\rho$）。

1. **线性 TTNS** $q(x)$ 关于每维基向量 $b^{(k)}(x_k)$ 是**多重线性**的 → 任意「每维一维权函数之积」的积分可**逐腿替换**为一维积分向量后做树收缩（**可分离引理**，§1）。这正是 Scheme B 传播成立的根。
2. **平方 TTNS** $\psi(x)^2$ 关于每维**外积** $B^{(k)}(x_k)=b^{(k)}(x_k)b^{(k)}(x_k)^\top\in\mathbb R^{m\times m}$ 是多重线性的 → $\psi^2$ **仍是一个张量网络**（bond 翻倍为 $r^2$、每维「基」变成 $m^2$ 个乘积 $b_ib_{i'}$）（**命题 3.1**）。
3. 因此**可分离引理对 $\psi^2$ 照样适用**：Scheme B 需要的一切量——归一化 $Z$、marginal CDF、pair CDF、矩——都化为**二次型树收缩**（quadratic form），**闭式、无采样**（**定理 4.1**）。每条腿的一维「向量」升级为 $m\times m$ **矩阵** $M_{ii'}=\int b_ib_{i'}w$。
4. **平方块的拟合**也能保持解析：L2 目标 $\int(\psi^2-p_{\text{tree}})^2$ 的**交叉项**是二次型树消息传递、**自项** $\int\psi^4$ 是四次收缩（bond $r^4$），二者皆闭式；小块（$K=3$）下 $r^4$ 可算 → **仍无采样**（§5，Option A）。
5. **代价**：bond 维 $r\to r^2$（传播）、$r^4$（自项）；高度数节点核 $\sim r^{2\deg}$ 放大（§6.5 rank16 编译爆炸即此），故**只在小块上安全**——恰好我们按 DAG 结构分块、块很小。
6. **附带收益**：$\psi^2\ge0$ 恒成立，消除线性 TTNS 密度取负（`npos` 率）的问题。

---

## 1. 记号与线性 TTNS 的可分离性

### 1.1 树 TTNS 的定义

块内单父树 $T=(V,E)$，$|V|=K$，根 $\rho$，每节点 $k$ 有子集合 $\mathrm{ch}(k)$、度 $\deg(k)=|\mathrm{ch}(k)|+\mathbb 1[k\ne\rho]$。给每条树边配一个 bond 维 $r$。节点 $k$ 的**核张量** $G^{(k)}$ 的指标为：

- 一个物理指标 $i_k\in\{1,\dots,m\}$（基）；
- 一个父腿 bond $\alpha_{k}\in\{1,\dots,r\}$（$k\ne\rho$）；
- 每个孩子 $c\in\mathrm{ch}(k)$ 一个 bond $\beta_{kc}\in\{1,\dots,r\}$。

定义节点的**赋值核** $A^{(k)}(x_k)$：把物理指标与基收缩，
$$A^{(k)}(x_k)_{\alpha_k,\,\{\beta_{kc}\}}=\sum_{i=1}^m G^{(k)}_{\alpha_k,\,i,\,\{\beta_{kc}\}}\,b^{(k)}_i(x_k).$$

线性 TTNS 函数 = 沿树把相邻核共享 bond 收缩：
$$
q(x)=\sum_{\{\alpha,\beta\}}\ \prod_{k\in V} A^{(k)}(x_k)_{\alpha_k,\{\beta_{kc}\}}\Big|_{\text{相邻边 bond 相等}}.
\tag{1.1}
$$

### 1.2 多重线性

**观察**：固定其它维时，$q(x)$ 关于第 $k$ 维的基向量 $b^{(k)}(x_k)=(b^{(k)}_1,\dots,b^{(k)}_m)^\top$ 是**线性**的（因为 $x_k$ 只经 $A^{(k)}$ 进入，且 $A^{(k)}$ 对 $b^{(k)}(x_k)$ 线性）。故 $q$ 对 $\{b^{(k)}(x_k)\}_k$ 联合**多重线性**。

### 1.3 可分离引理（Scheme B 的根基）

> **引理 1.1（可分离积分）**  设 $q$ 为 (1.1) 的树 TTNS，$\{w_k\}_{k\in V}$ 为任意一维权函数。则
> $$
> \int_{\mathbb R^K} q(x)\prod_{k}w_k(x_k)\,dx
> =\sum_{\{\alpha,\beta\}}\prod_{k}\widetilde A^{(k)}_{\alpha_k,\{\beta_{kc}\}},\qquad
> \widetilde A^{(k)}_{\alpha_k,\{\beta_{kc}\}}=\sum_i G^{(k)}_{\alpha_k,i,\{\beta_{kc}\}}\underbrace{\int b^{(k)}_i(x_k)\,w_k(x_k)\,dx_k}_{=:~m^{(k)}_i}.
> \tag{1.2}
> $$
> 即：**把每条腿的基向量替换为一维积分向量 $m^{(k)}=\big(\int b_i w_k\big)_i$，再做同样的树收缩。**

**证明**  由 (1.1)，$q(x)\prod_k w_k(x_k)$ 是形如 $\sum_{\{\alpha,\beta\}}\prod_k\big(\sum_i G^{(k)}_{\dots i\dots}b^{(k)}_i(x_k)\big)w_k(x_k)$ 的有限和。积分与有限和交换；由 Fubini，$\int\prod_k(\cdot)_k\,dx=\prod_k\int(\cdot)_k\,dx_k$，因为每个因子只依赖 $x_k$。逐维积分只作用在 $b^{(k)}_i(x_k)w_k(x_k)$ 上，得 $m^{(k)}_i$。代回即 (1.2)。$\qquad\blacksquare$

**特例**：$w_k\equiv1$ 时 $m^{(k)}_i=\int b_i$（基积分向量），对应「把该维积掉」。

---

## 2. 层间传递 merge 的 CDF 公式（回顾）

节点 $v$（下层）由父集 $\mathrm{pa}(v)$（上层）经 merge 生成：
$$x_v=\max_{u\in\mathrm{pa}(v)}\big(x_u+e_{uv}\big)+d_v,\quad e_{uv}\sim U[e_{lo},e_{hi}],\ d_v\sim U[n_{lo},n_{hi}],\ \text{独立}.$$

记 $F_e(t)=\mathrm{clip}\!\big((t-e_{lo})/(e_{hi}-e_{lo}),0,1\big)$。给定上层取值 $x$ 与 $d_v$，
$$\Pr\big[\max_u(x_u+e_{uv})\le t\mid x\big]=\prod_{u\in\mathrm{pa}(v)}\Pr[e_{uv}\le t-x_u]=\prod_{u}F_e(t-x_u).$$
对上层密度 $p_X$ 取期望，再对节点延迟 $d_v$ 卷积：
$$
F_v(s)=\frac{1}{n_{hi}-n_{lo}}\int_{n_{lo}}^{n_{hi}}\underbrace{\mathbb E_{x\sim p_X}\!\Big[\prod_{u\in\mathrm{pa}(v)}F_e(s-d-x_u)\Big]}_{=:~\Phi_v(s-d)}\,dd.
\tag{2.1}
$$

**核心量** $\Phi_v(t)=\mathbb E_{x\sim p_X}\big[\prod_{u}F_e(t-x_u)\big]$：它是「每维一维权函数 $w_u(x_u)=F_e(t-x_u)$（父）、$w=1$（非父）之积」对上层密度的期望。$d_v$ 卷积（外层积分）与 $x$ 无关，代码见 `_delay_convolve_1d`。配对 CDF $F_{vw}(s,t)$ 同理，两个节点各放 $F_e$ 权、共享父腿放 $F_e(s-\cdot)F_e(t-\cdot)$。

**线性上层**：$p_X=q$，由引理 1.1，$\Phi_v(t)$ = 每父腿放向量 $\int b_i F_e(t-x)\,dx$、非父腿放 $\int b_i$ 的树收缩。这就是现有 `UpperModel/UpperForest` 的实现。

---

## 3. 平方参数化：$\psi^2$ 仍是张量网络

设 $\psi$ 为 (1.1) 的树 TTNS（核记 $G^{(k)}$、bond $r$），密度
$$p(x)=\frac{\psi(x)^2}{Z},\qquad Z=\int\psi(x)^2\,dx.$$

### 3.1 doubled network（命题）

> **命题 3.1**  $\psi(x)^2$ 是一个 bond 维为 $r^2$、每维物理指标为**基乘积** $\{b_i b_{i'}\}_{i,i'=1}^{m}$（$m^2$ 个）的树 TTNS；等价地，它对每维的**外积特征** $B^{(k)}(x_k)=b^{(k)}(x_k)b^{(k)}(x_k)^\top$ 多重线性。

**证明**  $\psi(x)^2=\psi(x)\cdot\psi(x)$。取两份独立副本，副本 1 的 bond 记 $(\alpha,\beta)$、副本 2 记 $(\alpha',\beta')$。定义**doubled 核**
$$\mathcal G^{(k)}_{(\alpha_k,\alpha'_k),\,(i,i'),\,\{(\beta_{kc},\beta'_{kc})\}}:=G^{(k)}_{\alpha_k,i,\{\beta_{kc}\}}\,G^{(k)}_{\alpha'_k,i',\{\beta'_{kc}\}}.$$
其父/子 bond 为**有序对**（维 $r^2$），物理指标为对 $(i,i')$（维 $m^2$）。则
$$
\psi(x)^2=\sum_{\{(\alpha,\alpha'),(\beta,\beta')\}}\prod_k\Big(\sum_{i,i'}\mathcal G^{(k)}_{(\alpha_k,\alpha'_k),(i,i'),\{\cdots\}}\,b^{(k)}_i(x_k)\,b^{(k)}_{i'}(x_k)\Big),
\tag{3.1}
$$
即 (1.1) 形式，只是 bond$\to r^2$、每维基$\to\{b_ib_{i'}\}$。而 $b_i(x_k)b_{i'}(x_k)=B^{(k)}(x_k)_{ii'}$，故对 $B^{(k)}(x_k)$ 多重线性。$\qquad\blacksquare$

**要点**：平方**没有破坏张量网络结构与可分离性**，只是把「线性/向量」升级为「二次型/矩阵」。

---

## 4. 定理：Scheme B 全部量对平方 TTNS 闭式

> **定理 4.1（平方上层的解析传播）**  设上层块密度 $p_X=\psi^2/Z$。则 §2 中所有量均为 $\psi$ 的**二次型树收缩**，闭式可算：
> 1. **归一化** $Z=\mathrm{QF}\big(\psi;\{\Gamma^{(k)}\}_k\big)$，其中 gram $\Gamma^{(k)}_{ii'}=\int b^{(k)}_i b^{(k)}_{i'}$。
> 2. **核心量** $\Phi_v(t)=\dfrac1Z\,\mathrm{QF}\big(\psi;\{M^{(u)}(t)\}_{u\in\mathrm{pa}(v)},\{\Gamma^{(k)}\}_{k\notin\mathrm{pa}(v)}\big)$，其中
>    $$M^{(u)}(t)_{ii'}=\int b^{(u)}_i(x)\,b^{(u)}_{i'}(x)\,F_e(t-x)\,dx.$$
>    再由 (2.1) 对 $d_v$ 卷积得 $F_v$。
> 3. **配对** $F_{vw}(s,t)$：把 $v$ 独占父放 $M^{(u)}(s)$、$w$ 独占父放 $M^{(u)}(t)$、共享父放 $\int b_ib_{i'}F_e(s-x)F_e(t-x)dx$，其余 $\Gamma$，同样二次型收缩 + 两维各自 $d$ 卷积。
> 4. **矩**（$\mathbb E[x_v],\mathrm{Var}$）由 $F_v$ 数值积分（同现有 `moments_from_marginal`）。
>
> 这里 $\mathrm{QF}(\psi;\{W_k\}):=\big\langle\psi,\ (\bigotimes_k \hat W_k)\,\psi\big\rangle$ 为以每节点矩阵 $W_k$ 作用于该维基系数的**二次型**（即 `ttns_opt.normalized_quadratic_form_ttns` 的节点矩阵版）。

**证明**  由命题 3.1，$\psi^2$ 是树 TTNS（doubled）。对 §2 的核心量 $\Phi_v(t)=\int\psi(x)^2\prod_u F_e(t-x_u)\prod_{k\notin\mathrm{pa}}1\,dx/Z$ 应用引理 1.1（作用于 doubled network）：每维的「一维积分」现在作用在物理对 $(i,i')$ 上，得
$$\int b_i(x_k)b_{i'}(x_k)\,w_k(x_k)\,dx_k=\big(M^{(k)}\big)_{ii'},$$
父腿 $w_u=F_e(t-\cdot)\Rightarrow M^{(u)}(t)$，非父 $w=1\Rightarrow\Gamma^{(k)}$。把 doubled 核的物理对 $(i,i')$ 与 $M^{(k)}$ 收缩，等价于「原 $\psi$ 的两份副本各出一个物理指标、被 $M^{(k)}$ 的两个下标分别接住」，即二次型 $\langle\psi,(\otimes_k\hat M^{(k)})\psi\rangle$。$Z$ 取 $w_k\equiv1$（全 $\Gamma$）。配对与矩同理。所有收缩为有限维张量收缩，闭式。$\qquad\blacksquare$

**与线性对照**

| 每条腿贡献 | 线性 TTNS | 平方 TTNS |
|---|---|---|
| 一般权 $w$ | 向量 $\int b_i\,w\in\mathbb R^m$ | 矩阵 $\int b_ib_{i'}\,w\in\mathbb R^{m\times m}$ |
| 父腿（marginal） | $\int b_i F_e(s-x)$ | $M^{(u)}(s)=\int b_ib_{i'}F_e(s-x)$ |
| 非父腿 | $\int b_i$ | $\Gamma=\int b_ib_{i'}$ |
| 收缩类型 | 线性 eval（bond $r$） | 二次型 QF（bond $r^2$） |

**块间独立**：一层为若干独立块之积 $\prod_b\psi_b^2/Z_b$，期望按块因子分解（同 `UpperForest`），每块用上面的二次型收缩。故**成链传播全程解析、逐块**。

---

## 5. 平方块的解析拟合

目标：块的树密度（由定理 4.1 的 marginal/pair 得到，同现 `analytic_block_target`）
$$p_{\text{tree}}(y)=p_\rho(y_\rho)\prod_{k\ne\rho}p\big(y_k\mid y_{\mathrm{pa}(k)}\big).$$
要用 $\hat p=\psi^2/Z$ 逼近它。两条路：

### 5.1 Option A：解析 L2（无采样，推荐用于小块）

$$
\mathcal L(\psi)=\int\big(\psi^2-p_{\text{tree}}\big)^2\,dy=\underbrace{\int\psi^4}_{\text{自项}}-2\underbrace{\int\psi^2\,p_{\text{tree}}}_{\text{交叉项}}+\underbrace{\int p_{\text{tree}}^2}_{\text{常数}}.
\tag{5.1}
$$

- **交叉项** $\int\psi^2 p_{\text{tree}}=\mathbb E_{p_{\text{tree}}}[\psi^2]$：$\psi^2$（doubled 树）与 $p_{\text{tree}}$（同变量的树密度）在同一棵树上，做**树消息传递**：叶到根，节点 $k$ 的消息把「$p_{\text{tree}}$ 的条件核 $p(y_k\mid y_{\mathrm{pa}(k)})$」与「doubled 核在网格上的取值」沿 $y_k$ 积分。这是现有 `_cross_term_fn`（线性）到**二次型**的直接推广（每维用 $b_ib_{i'}$ 而非 $b_i$）。闭式。
- **自项** $\int\psi^4$：四次自收缩 $=\mathrm{QF}(\psi^2;\{\Gamma\})$ 的「再平方」，每维需四阶基张量 $T^{(k)}_{ijkl}=\int b_ib_jb_kb_l$（$m^4$，一次性预计算），bond $r^4$。对块 $K=3$、$r\le8$：$r^4=4096$，完全可算。
- **优化**：对核做 Adam 梯度下降（$q=\psi^2$ 天然非负，**无正定约束**）。$\mathcal L$ 是核的多项式，可微。

> Option A 使「层间解析、无跨层采样、无误差累积」这条主线**完整保留**，代价是 $r^4$ 自项（小块可控）。

### 5.2 Option B：块内 MLE（对解析目标采样）

对 $p_{\text{tree}}$（3 节点树，逆 CDF 极易采样）抽 $\{y^{(n)}\}_{n=1}^N$，最大化
$$\sum_n\log\hat p(y^{(n)})=\sum_n\Big(\log\psi(y^{(n)})^2-\log Z\Big),\quad Z=\mathrm{QF}(\psi;\{\Gamma\}).$$
这就是 TTDE 目标限制到块，「数据」来自**已知解析目标**。含采样，但**仅块内、来自解析树密度**，不跨层、不累积误差；比 $r^4$ 便宜。

> **为何 MLE 需采样而 L2 不需**：KL/MLE 含 $\log\psi^2$，对基**非可分离**（$\log$ 破坏乘积→积分分解），故只能蒙特卡洛；L2 只含 $\psi^2,\psi^4$ 的**多项式**，可分离引理适用 → 全解析。

---

## 6. 成链一致性

设第 $l$ 层已材料化为平方块森林 $\{\psi_{l,b}^2/Z_{l,b}\}_b$。推进到第 $l{+}1$ 层：

1. 以该森林为上层，用**定理 4.1 的二次型收缩**算下一层各节点 marginal/pair CDF（`UpperForest` 的二次型版）。
2. 按 DAG 结构分块（`structural_blocks`）+ 解析 MI 建块内树（`analytic_block_target`）。
3. 每块用 §5 拟合出 $\psi_{l+1,b}$。

全链每一步都是闭式二次型收缩，**与现有线性链结构同构**，仅把「线性 eval」换成「QF」。

---

## 7. 复杂度与非负性

- **bond 维**：传播 $r\to r^2$；自项 $\int\psi^4$ 为 $r^4$。
- **节点核**：度 $\deg$ 的节点，线性核 $\sim m\,r^{\deg}$，平方后二次型收缩的局部代价 $\sim m^2\,r^{2\deg}$ →**指数被翻倍**。这解释了 §6.5 中 MI 树（含度=5 枢纽）在 rank16 下 $r^{2\cdot5}$ 编译爆炸。**推论：平方只在低度数小块上安全**——与「按 DAG 结构分块、块小」天然契合；**禁用 super-node**（用户硬约束）也正好避免人为造高度数核。
- **每维矩阵/张量**：一维「向量」升为 $m\times m$（传播）与 $m^4$（自项，预计算一次）。
- **非负性**：$\psi^2\ge0$ 恒成立，$\hat p$ 是合法密度，消除线性 TTNS 的 `npos`（密度取负）问题，采样/对数密度更稳。

---

## 8. 与代码的对应 + 实现清单

| 数学对象 | 现有（线性） | 平方版落点 |
|---|---|---|
| 每腿一维积分 | 向量 $\int b_i w$ | 矩阵 $M_{ii'}=\int b_ib_{i'}w$ |
| 收缩 | `UpperModel._contract`（线性） | 二次型：复用 `ttns_opt.normalized_quadratic_form_ttns` 思路，节点矩阵 = $M^{(k)}$ |
| 归一化 $Z$ | 基积分连乘 | 全 $\Gamma$ 的 QF |
| marginal/pair CDF | `UpperModel.proj_single/_contract` | 换成 QF + $M^{(u)}(s)$ |
| 拟合交叉项 | `_cross_term_fn`（线性树消息） | doubled 树消息（每维 $b_ib_{i'}$） |
| 拟合自项 | `quadratic_form_ttns`（$\int q^2$，$r^2$） | $\int\psi^4$（$r^4$，四阶基张量 $T_{ijkl}$） |

**最小实现步骤**：
1. 预计算每维 $\Gamma=\int bb^\top$、$M(s)=\int bb^\top F_e(s-\cdot)$（网格）、$T=\int b_ib_jb_kb_l$。
2. 写块内平方 L2（Option A）：`_cross_term_sqr`（二次型树消息）+ `quartic_self_ttns`（$\int\psi^4$）+ Adam。**先在单个 3 节点块上验证** LL 是否推向 TTDE 水平。
3. 若有效，再把 `UpperForest` 扩出二次型传播路径（`UpperForestSqr`），使**平方块能作为下一层上层**，打通全链。

---

## 9. 局限与注意

- **$r^4$ 与高度数**：仅小块可行；大树/高度数枢纽会爆（§7）。工程上须对块大小/度数设上限或对 rank 设上限。
- **L2 非凸**：(5.1) 对 $\psi$ 非凸（与 TTDE 同），需良好初始化（可用 rank-1 目标边缘的 $\sqrt{\cdot}$，参考库内 `fix_nonsqrt_init`）与梯度裁剪。
- **数值**：$M(s)$、$T$ 在样条基上解析或高精度数值积分；网格分辨率影响 CDF 精度（同现有 `n_s/n_s_pair`）。
- **采样**：平方 TTNS 的**树条件采样器**仍需实现（§6.5 未决项）；在拿到平方块后，块内 3 维可用二次型条件 CDF 逆采样（TTDE 链采样的树推广）。

---

*创建：2026-07-02 — 证明平方（非负）TTNS 与 clarify.md 的层间传递 merge 解析传播/拟合相容：$\psi^2$ 仍是张量网络，Scheme B 全量化为二次型树收缩（闭式），拟合可保持无采样（L2）；代价为 bond 维平方、仅小块安全。配套 `ALGORITHM_zh.md` §3.4/§6.5。*
