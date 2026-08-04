# TTNS 上的 max-plus 随机传播：联合 CDF 收缩定理

## 0. 文档目的

本文只讨论一个数学问题：当上游随机向量的联合密度由 Tree Tensor Network State（TTNS）表示时，如何计算经过 max-plus 随机映射后下游随机向量的联合累积分布函数。

全文先定义所有对象，再陈述和证明定理。连续积分公式与代码中的网格、求积、插值和截断近似严格分开。

本文不声称 TTNS 密度估计、Chow–Liu tree、tensor-network marginalization 或 max-plus 概率公式本身是首次提出。可能具有新颖性的对象，是这些结构的特定结合：连续 TTNS 密度、具有共享父节点的多输出 max-plus 随机映射，以及通过一维局部投影和树收缩计算下游联合 CDF。

## 1. 记号

对正整数 $n$，记

$$
[n]=\{1,2,\ldots,n\}.
$$

设上游随机向量为

$$
X=(X_1,\ldots,X_d)\in\mathbb R^d.
$$

其联合概率密度记为 $p_X$。

对每个坐标 $k\in[d]$，选取 $m_k$ 个 Borel 可测且绝对可积的一维基函数

$$
b_{k,1},\ldots,b_{k,m_k}:\mathbb R\longrightarrow\mathbb R.
$$

记

$$
b_k(x_k)
=
\bigl(b_{k,1}(x_k),\ldots,b_{k,m_k}(x_k)\bigr)^\top.
$$

## 2. TTNS 系数张量

### 2.1 树结构

设

$$
T=(V,E)
$$

是一棵树，其中

$$
V=[d].
$$

树的每个顶点对应上游随机变量的一个坐标。对每条树边 $e\in E$，指定一个正整数 $r_e$，称为该边的 bond dimension。

对每条边 $e$，引入 bond index

$$
\alpha_e\in[r_e].
$$

### 2.2 节点 core

对每个顶点 $k\in V$，记与 $k$ 相邻的边集合为

$$
\delta(k)=\{e\in E:e\text{ 与 }k\text{ 相邻}\}.
$$

顶点 $k$ 的 TTNS core 记为

$$
G^{(k)}.
$$

它包含一个物理指标

$$
i_k\in[m_k]
$$

以及所有相邻树边的 bond indices

$$
(\alpha_e)_{e\in\delta(k)}.
$$

因此可写成

$$
G^{(k)}_{i_k,(\alpha_e)_{e\in\delta(k)}}.
$$

### 2.3 系数张量

定义 $d$ 阶系数张量 $A$：

$$
A_{i_1,\ldots,i_d}
=
\sum_{(\alpha_e)_{e\in E}}
\prod_{k=1}^{d}
G^{(k)}_{i_k,(\alpha_e)_{e\in\delta(k)}}.
\tag{2.1}
$$

式 (2.1) 的含义是：对所有树边的 bond indices 求和，而每个节点 core 贡献一个因子。

### 2.4 TTNS 密度

假设上游密度具有有限基展开

$$
p_X(x_1,\ldots,x_d)
=
\sum_{i_1=1}^{m_1}\cdots\sum_{i_d=1}^{m_d}
A_{i_1,\ldots,i_d}
\prod_{k=1}^{d}b_{k,i_k}(x_k).
\tag{2.2}
$$

并假设

$$
p_X(x)\ge0
\quad\text{对几乎处处的 }x\in\mathbb R^d,
\tag{2.3}
$$

以及

$$
\int_{\mathbb R^d}p_X(x)\,dx=1.
\tag{2.4}
$$

条件 (2.3)–(2.4) 保证 $p_X$ 是合法概率密度。若它们不成立，后面的代数收缩恒等式仍可能成立，但所得函数不能保证是概率 CDF。

## 3. TTNS 的多线性收缩函数

给定向量

$$
z_k=(z_{k,1},\ldots,z_{k,m_k})^\top\in\mathbb R^{m_k},
\qquad k\in[d],
$$

定义

$$
\mathcal C_T(z_1,\ldots,z_d)
=
\sum_{i_1=1}^{m_1}\cdots\sum_{i_d=1}^{m_d}
A_{i_1,\ldots,i_d}
\prod_{k=1}^{d}z_{k,i_k}.
\tag{3.1}
$$

将 (2.1) 代入 (3.1)，可得

$$
\mathcal C_T(z_1,\ldots,z_d)
=
\sum_{(\alpha_e)_{e\in E}}
\prod_{k=1}^{d}
\left(
\sum_{i_k=1}^{m_k}
G^{(k)}_{i_k,(\alpha_e)_{e\in\delta(k)}}z_{k,i_k}
\right).
\tag{3.2}
$$

因此，$\mathcal C_T$ 可以直接沿树收缩计算，而不需要显式构造完整系数张量 $A$。

## 4. 可分离积分引理

### 引理 1

设 $w_k:\mathbb R\to\mathbb R$ 为可测函数，并假设下面的积分绝对收敛。定义

$$
z_{k,i}
=
\int_{\mathbb R}b_{k,i}(x)w_k(x)\,dx.
\tag{4.1}
$$

则

$$
\int_{\mathbb R^d}
p_X(x_1,\ldots,x_d)
\prod_{k=1}^{d}w_k(x_k)
\,dx_1\cdots dx_d
=
\mathcal C_T(z_1,\ldots,z_d).
\tag{4.2}
$$

### 证明

将 (2.2) 代入左侧：

$$
\begin{aligned}
&\int_{\mathbb R^d}
p_X(x)
\prod_{k=1}^{d}w_k(x_k)\,dx\\
&=
\int_{\mathbb R^d}
\sum_{i_1,\ldots,i_d}
A_{i_1,\ldots,i_d}
\prod_{k=1}^{d}
b_{k,i_k}(x_k)w_k(x_k)
\,dx.
\end{aligned}
$$

因为求和是有限和，可以交换求和与积分。由绝对可积性和 Fubini 定理，得到

$$
\begin{aligned}
&\int_{\mathbb R^d}
p_X(x)
\prod_{k=1}^{d}w_k(x_k)\,dx\\
&=
\sum_{i_1,\ldots,i_d}
A_{i_1,\ldots,i_d}
\prod_{k=1}^{d}
\left(
\int_{\mathbb R}
b_{k,i_k}(x_k)w_k(x_k)\,dx_k
\right).
\end{aligned}
$$

根据 (4.1)，右侧等于

$$
\sum_{i_1,\ldots,i_d}
A_{i_1,\ldots,i_d}
\prod_{k=1}^{d}z_{k,i_k}
=
\mathcal C_T(z_1,\ldots,z_d).
$$

证毕。

## 5. 多输出 max-plus 随机映射

设有 $K$ 个下游随机变量

$$
Y=(Y_1,\ldots,Y_K).
$$

对每个下游坐标 $a\in[K]$，指定非空父节点集合

$$
P_a\subseteq[d].
$$

定义

$$
Y_a
=
\max_{k\in P_a}
\bigl(X_k+E_{ka}\bigr)
+D_a.
\tag{5.1}
$$

这里：

- $E_{ka}$ 是从 $X_k$ 到 $Y_a$ 的 edge delay；
- $D_a$ 是 $Y_a$ 的 node delay。

作如下独立性假设。

### 假设 A

1. 随机变量族

   $$
   \{E_{ka}:a\in[K],\ k\in P_a\}
   $$

   中的所有随机变量相互独立。
2. 随机向量 $X$、完整 edge-delay 随机变量族和随机向量 $D$ 相互独立。

假设 A 不要求 $D_1,\ldots,D_K$ 相互独立。只有在实现中希望把对 $D$ 的
$K$ 维外层期望分解为 $K$ 个一维求积时，才额外使用以下条件。

### 假设 B

随机变量 $D_1,\ldots,D_K$ 相互独立。

记 $E_{ka}$ 的 CDF 为

$$
F_{ka}(t)=P(E_{ka}\le t).
\tag{5.2}
$$

node delay 向量记为

$$
D=(D_1,\ldots,D_K).
$$

## 6. 条件联合 CDF 引理

### 引理 2

因为 $\mathbb R^d$ 和 $\mathbb R^K$ 是标准 Borel 空间，给定 $(X,D)$ 的正则
条件分布存在。在假设 A 下，对任意

$$
y=(y_1,\ldots,y_K)\in\mathbb R^K,
$$

，可以选择一个正则条件概率版本，使下式对 $P_{(X,D)}$ 几乎处处的
$(x,d)\in\mathbb R^d\times\mathbb R^K$ 成立：

$$
P(Y_1\le y_1,\ldots,Y_K\le y_K\mid X=x,D=d)
=
\prod_{a=1}^{K}
\prod_{k\in P_a}
F_{ka}(y_a-d_a-x_k).
\tag{6.1}
$$

### 证明

由 (5.1)，事件 $Y_a\le y_a$ 等价于

$$
X_k+E_{ka}+D_a\le y_a
\quad\text{对所有 }k\in P_a.
$$

在条件 $X=x,D=d$ 下，它等价于

$$
E_{ka}\le y_a-d_a-x_k
\quad\text{对所有 }k\in P_a.
$$

因此

$$
\begin{aligned}
&\{Y_1\le y_1,\ldots,Y_K\le y_K\}\\
&=
\bigcap_{a=1}^{K}
\bigcap_{k\in P_a}
\{E_{ka}\le y_a-d_a-x_k\}.
\end{aligned}
$$

由全部 edge delays 的相互独立性，条件概率等于各事件概率之积，即得到 (6.1)。

证毕。

## 7. TTNS 上的 max-plus 联合 CDF 收缩定理

### 定理 1

假设：

1. 上游密度 $p_X$ 满足 (2.2)–(2.4)；
2. 假设 A 成立。

由于每个 $b_{k,i}$ 绝对可积，且 CDF 乘积取值于 $[0,1]$，下面的一维局部
投影自动绝对收敛。

对固定的 $y\in\mathbb R^K$ 和 $d\in\mathbb R^K$，对每个上游坐标 $k\in[d]$ 和每个基函数指标 $i\in[m_k]$，定义

$$
M_{k,i}(y,d)
=
\int_{\mathbb R}
b_{k,i}(x)
\prod_{\substack{a\in[K]\\k\in P_a}}
F_{ka}(y_a-d_a-x)
\,dx.
\tag{7.1}
$$

如果 $k$ 不属于任何 $P_a$，则 (7.1) 中的空乘积定义为 1，所以

$$
M_{k,i}(y,d)
=
\int_{\mathbb R}b_{k,i}(x)\,dx.
\tag{7.2}
$$

记

$$
M_k(y,d)
=
\bigl(M_{k,1}(y,d),\ldots,M_{k,m_k}(y,d)\bigr)^\top.
$$

则下游随机向量 $Y$ 的联合 CDF

$$
F_Y(y)
=
P(Y_1\le y_1,\ldots,Y_K\le y_K)
$$

满足

$$
F_Y(y)
=
\mathbb E_D
\left[
\mathcal C_T
\bigl(
M_1(y,D),\ldots,M_d(y,D)
\bigr)
\right].
\tag{7.3}
$$

如果 $D$ 具有联合密度 $f_D$，则等价地

$$
F_Y(y)
=
\int_{\mathbb R^K}
\mathcal C_T
\bigl(
M_1(y,d),\ldots,M_d(y,d)
\bigr)
f_D(d)\,dd.
\tag{7.4}
$$

若进一步满足假设 B，且各 $D_a$ 都具有密度 $f_{D_a}$，则

$$
f_D(d)=\prod_{a=1}^{K}f_{D_a}(d_a)
$$

成立。定理 1 本身只需要 $D$ 的联合概率分布，不需要假设 B，也不要求 $D$
具有密度。

### 证明

由全期望公式，

$$
F_Y(y)
=
\mathbb E_D
\left[
P(Y_1\le y_1,\ldots,Y_K\le y_K\mid D)
\right].
\tag{7.5}
$$

固定 $D=d$。再次使用全期望公式，并应用引理 2：

$$
\begin{aligned}
&P(Y_1\le y_1,\ldots,Y_K\le y_K\mid D=d)\\
&=
\mathbb E_X
\left[
\prod_{a=1}^{K}
\prod_{k\in P_a}
F_{ka}(y_a-d_a-X_k)
\right].
\end{aligned}
\tag{7.6}
$$

按照上游坐标 $k$ 重新组织乘积：

$$
\prod_{a=1}^{K}
\prod_{k\in P_a}
F_{ka}(y_a-d_a-X_k)
=
\prod_{k=1}^{d}
\left[
\prod_{\substack{a\in[K]\\k\in P_a}}
F_{ka}(y_a-d_a-X_k)
\right].
\tag{7.7}
$$

对每个 $k\in[d]$，定义

$$
w_k(x;y,d)
=
\prod_{\substack{a\in[K]\\k\in P_a}}
F_{ka}(y_a-d_a-x).
\tag{7.8}
$$

于是 (7.6) 变成

$$
P(Y_1\le y_1,\ldots,Y_K\le y_K\mid D=d)
=
\int_{\mathbb R^d}
p_X(x)
\prod_{k=1}^{d}w_k(x_k;y,d)
\,dx.
\tag{7.9}
$$

对 (7.9) 应用引理 1。根据 (4.1)，对应的局部投影向量恰好是 (7.1) 定义的 $M_k(y,d)$。因此

$$
P(Y_1\le y_1,\ldots,Y_K\le y_K\mid D=d)
=
\mathcal C_T
\bigl(M_1(y,d),\ldots,M_d(y,d)\bigr).
\tag{7.10}
$$

将 (7.10) 代回 (7.5)，得到 (7.3)。若 $D$ 具有联合密度，则按照期望的积分定义得到 (7.4)。

证毕。

## 8. 推论

### 推论 1：单个下游节点

当 $K=1$ 时，记下游变量为

$$
Y=\max_{k\in P}(X_k+E_k)+D.
$$

对 $k\in P$，定义

$$
M_{k,i}(y,d)
=
\int b_{k,i}(x)F_{E_k}(y-d-x)\,dx.
$$

对 $k\notin P$，定义

$$
M_{k,i}=\int b_{k,i}(x)\,dx.
$$

则

$$
F_Y(y)
=
\mathbb E_D
\left[
\mathcal C_T(M_1(y,D),\ldots,M_d(y,D))
\right].
\tag{8.1}
$$

### 推论 2：两个具有共享父节点的下游节点

设 $K=2$。如果某个上游坐标 $k$ 同时属于 $P_1$ 和 $P_2$，则其局部投影为

$$
\begin{aligned}
M_{k,i}(y,d)
=
\int b_{k,i}(x)
&F_{k1}(y_1-d_1-x)\\
&\cdot F_{k2}(y_2-d_2-x)
\,dx.
\end{aligned}
\tag{8.2}
$$

因此，共享父节点不会破坏变量可分离性。它只使该父坐标的一维权函数变成两个 edge CDF 的乘积。

### 推论 3：独立 TTNS forest

设坐标集合 $[d]$ 被划分为互不相交的块

$$
B_1,\ldots,B_J,
$$

且模型密度具有乘积形式

$$
p_X(x)=\prod_{j=1}^{J}p_j(x_{B_j}).
\tag{8.3}
$$

假设每个 $p_j$ 分别由一棵 TTNS 表示。则定理 1 中的条件联合 CDF 收缩分解为各块收缩的乘积：

$$
\mathcal C_{mathrm{forest}}(M_1,\ldots,M_d)
=
\prod_{j=1}^{J}
\mathcal C_{T_j}\bigl((M_k)_{k\in B_j}\bigr).
\tag{8.4}
$$

这里的精确性是相对于模型 (8.3) 而言。若真实上游密度不能按块独立分解，则从真实密度到 (8.3) 还存在额外的模型误差。

## 9. 平方 TTNS 推论

### 9.1 平方密度

设

$$
\psi(x)
=
\sum_{i_1,\ldots,i_d}
B_{i_1,\ldots,i_d}
\prod_{k=1}^{d}b_{k,i_k}(x_k),
\tag{9.1}
$$

其中系数张量 $B$ 具有 TTNS 表示。定义

$$
Z=\int_{\mathbb R^d}\psi(x)^2\,dx,
$$

并假设

$$
0<Z<\infty.
$$

平方 TTNS 密度定义为

$$
p_X(x)=\frac{\psi(x)^2}{Z}.
\tag{9.2}
$$

### 9.2 二次型收缩

对每个坐标 $k$，给定矩阵

$$
W_k\in\mathbb R^{m_k\times m_k},
$$

定义

$$
\mathcal Q_T(W_1,\ldots,W_d)
=
\sum_{i_1,\ldots,i_d}
\sum_{j_1,\ldots,j_d}
B_{i_1,\ldots,i_d}
B_{j_1,\ldots,j_d}
\prod_{k=1}^{d}(W_k)_{i_k,j_k}.
\tag{9.3}
$$

因为 $B$ 具有 TTNS 表示，$\mathcal Q_T$ 可以通过两份 TTNS 的 doubled contraction 计算。每条原 bond index 被替换为一对 bond indices，因此边 $e$ 的 doubled bond dimension 为 $r_e^2$。

### 推论 4：平方 TTNS 的 max-plus 联合 CDF

在定理 1 的 delay 假设下，定义

$$
(W_k(y,d))_{i,j}
=
\int_{\mathbb R}
b_{k,i}(x)b_{k,j}(x)
\prod_{\substack{a\in[K]\\k\in P_a}}
F_{ka}(y_a-d_a-x)
\,dx.
\tag{9.4}
$$

再定义 Gram matrix

$$
(\Gamma_k)_{i,j}
=
\int_{\mathbb R}b_{k,i}(x)b_{k,j}(x)\,dx.
\tag{9.5}
$$

当 $k$ 不属于任何父节点集合时，$W_k(y,d)=\Gamma_k$。并且

$$
Z=\mathcal Q_T(\Gamma_1,\ldots,\Gamma_d).
\tag{9.6}
$$

下游联合 CDF 满足

$$
F_Y(y)
=
\frac{1}{Z}
\mathbb E_D
\left[
\mathcal Q_T
\bigl(W_1(y,D),\ldots,W_d(y,D)\bigr)
\right].
\tag{9.7}
$$

### 证明

将 $p_X=\psi^2/Z$ 代入定理 1 的证明。展开 $\psi^2$ 后，每个坐标上出现基函数乘积

$$
b_{k,i}(x)b_{k,j}(x).
$$

对应的一维局部积分即为 (9.4)。对全部坐标收缩后得到 (9.3)，再除以归一化常数 $Z$，即得 (9.7)。

证毕。

## 10. 稳定性命题

这一节讨论的是概率传播算子本身，不依赖 TTNS。

### 10.1 Total variation distance

对两个概率测度 $P$ 和 $Q$，定义

$$
\|P-Q\|_{\mathrm{TV}}
=
\sup_A|P(A)-Q(A)|,
\tag{10.1}
$$

其中上确界对所有可测集合 $A$ 取得。

### 命题 1：单步 max-plus 传播不放大 total variation error

设 $K$ 表示由 (5.1) 和固定 delay 分布定义的随机转移 kernel。若 $P$ 和 $Q$ 是两个上游概率分布，则

$$
\|PK-QK\|_{\mathrm{TV}}
\le
\|P-Q\|_{\mathrm{TV}}.
\tag{10.2}
$$

特别地，对任意 $y\in\mathbb R^K$，

$$
|F_{PK}(y)-F_{QK}(y)|
\le
\|P-Q\|_{\mathrm{TV}}.
\tag{10.3}
$$

### 证明

对任意下游可测集合 $A$，定义

$$
h_A(x)=K(x,A).
$$

因为 $K(x,A)$ 是条件概率，故

$$
0\le h_A(x)\le1.
$$

于是

$$
PK(A)-QK(A)
=
\int h_A(x)\,(P-Q)(dx).
$$

由 total variation 的变分表示，

$$
|PK(A)-QK(A)|
\le
\|P-Q\|_{\mathrm{TV}}.
$$

对 $A$ 取上确界，得到 (10.2)。令

$$
A=(-\infty,y_1]\times\cdots\times(-\infty,y_K],
$$

即可得到 (10.3)。

证毕。

### 命题 2：逐层材料化误差累积界

设真实逐层分布满足

$$
P_\ell=P_{\ell-1}K_\ell,
\qquad \ell=1,\ldots,L.
\tag{10.4}
$$

设近似算法先传播上一层近似分布 $Q_{\ell-1}$，再将传播结果材料化为 TTNS 分布 $Q_\ell$。假设第 $\ell$ 层的材料化误差满足

$$
\|Q_\ell-Q_{\ell-1}K_\ell\|_{\mathrm{TV}}
\le\varepsilon_\ell.
\tag{10.5}
$$

则

$$
\|Q_L-P_L\|_{\mathrm{TV}}
\le
\|Q_0-P_0\|_{\mathrm{TV}}
+
\sum_{\ell=1}^{L}\varepsilon_\ell.
\tag{10.6}
$$

### 证明

由三角不等式和命题 1，

$$
\begin{aligned}
\|Q_\ell-P_\ell\|_{\mathrm{TV}}
&\le
\|Q_\ell-Q_{\ell-1}K_\ell\|_{\mathrm{TV}}
+
\|Q_{\ell-1}K_\ell-P_{\ell-1}K_\ell\|_{\mathrm{TV}}\\
&\le
\varepsilon_\ell
+
\|Q_{\ell-1}-P_{\ell-1}\|_{\mathrm{TV}}.
\end{aligned}
$$

从 $\ell=1$ 到 $L$ 递推，即得 (10.6)。

证毕。

命题 1 是一般 Markov kernel 的经典 contraction 性质，不应被声明为本文首次发现。本文可以使用它解释 max-plus 解析传播和逐层 TTNS 材料化之间的误差关系。

### 命题 2A：单层新增误差的可加分解

命题 2 中的 $\varepsilon_\ell$ 必须只表示第 $\ell$ 层新引入的误差，不能把此前
各层已经存在的误差再次计入。为此定义

$$
\widetilde Q_\ell=Q_{\ell-1}K_\ell,
$$

即从上一层近似分布出发、使用精确 max-plus kernel 得到的传播分布。进一步设
$R_\ell^{\mathrm{num}}$ 是确定性数值离散后得到的概率分布，
$R_\ell^{\mathrm{struct}}$ 是将该分布限制到选定 forest、Chow--Liu tree 或
block 结构后得到的概率分布，$Q_\ell$ 是最终材料化的 TTNS 概率分布。定义

$$
\begin{aligned}
\eta_\ell^{\mathrm{num}}
&=
\|\widetilde Q_\ell-R_\ell^{\mathrm{num}}\|_{\mathrm{TV}},\\
\eta_\ell^{\mathrm{struct}}
&=
\|R_\ell^{\mathrm{num}}-R_\ell^{\mathrm{struct}}\|_{\mathrm{TV}},\\
\eta_\ell^{\mathrm{fit}}
&=
\|R_\ell^{\mathrm{struct}}-Q_\ell\|_{\mathrm{TV}}.
\end{aligned}
$$

则单层材料化误差满足

$$
\|Q_{\ell-1}K_\ell-Q_\ell\|_{\mathrm{TV}}
\le
\eta_\ell^{\mathrm{num}}
+
\eta_\ell^{\mathrm{struct}}
+
\eta_\ell^{\mathrm{fit}}.
$$

因而

$$
\boxed{
\|P_L-Q_L\|_{\mathrm{TV}}
\le
\|P_0-Q_0\|_{\mathrm{TV}}
+
\sum_{\ell=1}^{L}
\left(
\eta_\ell^{\mathrm{num}}
+
\eta_\ell^{\mathrm{struct}}
+
\eta_\ell^{\mathrm{fit}}
\right).
}
$$

特别地，对第 $L$ 层任意联合 CDF 查询点 $y$，

$$
\boxed{
|F_{P_L}(y)-F_{Q_L}(y)|
\le
\|P_0-Q_0\|_{\mathrm{TV}}
+
\sum_{\ell=1}^{L}
\left(
\eta_\ell^{\mathrm{num}}
+
\eta_\ell^{\mathrm{struct}}
+
\eta_\ell^{\mathrm{fit}}
\right).
}
$$

### 证明

前三项之间的界由两次三角不等式直接得到。将所得单层界代入命题 2 的
$\varepsilon_\ell$，即可得到多层 total variation 界。联合 CDF 对应一个下正交
矩形事件，其概率差不超过 total variation distance，因此得到最后一个不等式。

证毕。

这里要求 $\widetilde Q_\ell$、$R_\ell^{\mathrm{num}}$、
$R_\ell^{\mathrm{struct}}$ 和 $Q_\ell$ 都是定义在同一可测空间上的概率测度。
如果实验只在有限公共网格上归一化概率质量，则得到的是该离散模型上的精确 TV，
或连续模型的 grid-estimated TV；不能将它直接称为连续分布的严格 TV 上界。

### 命题 2B：max-plus 多层传播的 Wasserstein 非扩张界

在第 $\ell$ 层的输入和输出空间上均使用 $\ell_\infty$ metric。把该层的全部
edge delay 和 node delay 记为随机向量 $Z_\ell$。对固定 $z$，定义

$$
\Phi_{\ell,a}(x;z)
=
\max_{i\in P_{\ell,a}}
\{x_i+e_{ia}\}
+d_a.
$$

则

$$
\|\Phi_\ell(x;z)-\Phi_\ell(x';z)\|_\infty
\le
\|x-x'\|_\infty.
$$

若 $P$ 和 $Q$ 使用相同的 delay 分布，则对任意 $p\ge1$，

$$
W_p(PK_\ell,QK_\ell)
\le
W_p(P,Q).
$$

如果第 $\ell$ 层的 Wasserstein 材料化误差满足

$$
W_p(Q_{\ell-1}K_\ell,Q_\ell)
\le
\zeta_\ell,
$$

则

$$
\boxed{
W_p(P_L,Q_L)
\le
W_p(P_0,Q_0)
+
\sum_{\ell=1}^{L}\zeta_\ell.
}
$$

### 证明

对任意 $x,x'$ 和相同的 $z$，最大值的基本不等式给出

$$
\begin{aligned}
|\Phi_{\ell,a}(x;z)-\Phi_{\ell,a}(x';z)|
&\le
\max_{i\in P_{\ell,a}}|x_i-x_i'|\\
&\le
\|x-x'\|_\infty.
\end{aligned}
$$

再对输出坐标 $a$ 取最大值即可得到 $\Phi_\ell(\cdot;z)$ 的 $1$-Lipschitz
性质。取 $P,Q$ 的任意 coupling，并在两个传播副本中同步使用同一个
$Z_\ell$，输出 coupling 的 $p$ 阶代价不超过输入 coupling 的代价。对输入
coupling 取下确界，得到 Wasserstein contraction。最后使用三角不等式逐层递推，
即可得到材料化误差的可加界。

证毕。

该 Wasserstein 结论依赖相同 delay law 下的同步耦合。若真实和近似 delay law
不同，还必须加入相应的 delay coupling 误差。Wasserstein distance 也不能在没有
反集中或有界密度等附加条件时直接替代上面的联合 CDF uniform error。

### 命题 2C：Chow--Liu forest 结构损失的精确 KL 恒等式

设 $P$ 是 $Y=(Y_1,\ldots,Y_K)$ 的联合概率分布，所有后续 KL divergence、
mutual information 和 entropy expression 均有限。给定一棵有向 forest
$T$，每个非根节点 $v$ 只有一个树父节点
$\operatorname{pa}_T(v)$。使用 $P$ 的真实一维和树边二维边缘构造

$$
q_T(y)
=
\prod_{r\in\operatorname{root}(T)}p_r(y_r)
\prod_{v\notin\operatorname{root}(T)}
\frac{
p_{v,\operatorname{pa}_T(v)}
(y_v,y_{\operatorname{pa}_T(v)})
}{
p_{\operatorname{pa}_T(v)}
(y_{\operatorname{pa}_T(v)})
}.
$$

定义 total correlation

$$
\operatorname{TC}_P(Y_1,\ldots,Y_K)
=
D_{\mathrm{KL}}
\left(
P
\middle\|
\prod_{k=1}^{K}P_k
\right).
$$

则

$$
\boxed{
D_{\mathrm{KL}}(P\|Q_T)
=
\operatorname{TC}_P(Y_1,\ldots,Y_K)
-
\sum_{(u,v)\in E(T)}
I_P(Y_u;Y_v).
}
$$

进一步，取任何与 forest 方向相容的节点次序 $v_1,\ldots,v_K$。令

$$
A_j
=
\{v_1,\ldots,v_{j-1}\}
\setminus
\{\operatorname{pa}_T(v_j)\},
$$

其中根节点的树父集合为空，则

$$
\boxed{
D_{\mathrm{KL}}(P\|Q_T)
=
\sum_{j=1}^{K}
I_P
\left(
Y_{v_j};
Y_{A_j}
\mid
Y_{\operatorname{pa}_T(v_j)}
\right).
}
$$

当条件集合为空时，右侧按普通 mutual information 理解。特别地，对三节点树
$Y_1-Y_2-Y_3$，

$$
D_{\mathrm{KL}}(P\|Q_T)
=
I_P(Y_1;Y_3\mid Y_2).
$$

### 证明

对 $q_T$ 取对数并在 $P$ 下求期望。每个根节点贡献其一维边缘的负熵；每条
有向边贡献相应条件分布的负条件熵。整理各节点熵项得到

$$
D_{\mathrm{KL}}(P\|Q_T)
=
\sum_{k=1}^{K}H_P(Y_k)
-H_P(Y)
-
\sum_{(u,v)\in E(T)}I_P(Y_u;Y_v),
$$

即第一个恒等式。另一种证明使用 chain rule：

$$
p(y)
=
\prod_{j=1}^{K}
p(y_{v_j}\mid y_{v_1},\ldots,y_{v_{j-1}}),
$$

而 $q_T$ 在第 $j$ 个因子中只保留树父节点。逐项计算条件 KL，恰好得到

$$
I_P
\left(
Y_{v_j};
Y_{A_j}
\mid
Y_{\operatorname{pa}_T(v_j)}
\right).
$$

求和即得第二个恒等式。

证毕。

由 Pinsker inequality 立即得到

$$
\|P-Q_T\|_{\mathrm{TV}}
\le
\sqrt{
\frac12
\left[
\operatorname{TC}_P(Y)
-
\sum_{(u,v)\in E(T)}I_P(Y_u;Y_v)
\right]
}.
$$

因此，R5 tree projection 丢失的不是抽象的“高阶相关”，而是上述条件互信息之和。

### 推论 5：有限精度 MI 选树的 regret

设候选 forest 均有 $M$ 条边，真实 edge mutual information 为 $I_e$，数值估计
为 $\widehat I_e$，并满足

$$
\max_e|\widehat I_e-I_e|\le\epsilon.
$$

令 $T^\star$ 最大化真实权重 $\sum_{e\in T}I_e$，令 $\widehat T$ 最大化估计
权重 $\sum_{e\in T}\widehat I_e$。则

$$
\boxed{
\sum_{e\in T^\star}I_e
-
\sum_{e\in\widehat T}I_e
\le
2M\epsilon.
}
$$

因而

$$
D_{\mathrm{KL}}(P\|Q_{\widehat T})
\le
D_{\mathrm{KL}}(P\|Q_{T^\star})
+
2M\epsilon.
$$

### 证明

对任意含 $M$ 条边的 forest，真实权重与估计权重之差的绝对值不超过
$M\epsilon$。在 $T^\star$ 和 $\widehat T$ 上各使用一次该界，再使用
$\widehat T$ 的估计权重最优性，即得结论。

证毕。

### 命题 2D：TTNS rank 的谱下界与 tree-SVD 构造界

设离散联合概率表或平方可积联合密度系数形成 Hilbert tensor $P_T$。对 tree
边 $e$，切断 $e$ 后变量分成 $S_e$ 和 $S_e^c$，相应矩阵展开记为
$P_T^{(e)}$，奇异值记为

$$
\sigma_{e,1}\ge\sigma_{e,2}\ge\cdots.
$$

若 TTNS 在边 $e$ 上的 bond rank 不超过 $r_e$，则任何这样的 $Q$ 均满足

$$
\boxed{
\|P_T-Q\|_F
\ge
\max_{e\in E(T)}
\left(
\sum_{j>r_e}\sigma_{e,j}^2
\right)^{1/2}.
}
$$

若使用标准正交 TT-SVD，或使用满足嵌套正交投影条件的 tree-SVD，则存在一个
不要求非负的 TTNS $Q_{\mathrm{svd}}$ 满足

$$
\boxed{
\|P_T-Q_{\mathrm{svd}}\|_F
\le
\left(
\sum_{e\in E(T)}
\sum_{j>r_e}\sigma_{e,j}^2
\right)^{1/2}.
}
$$

### 证明

任意 TTNS 的边 $e$ bond rank 都是矩阵展开 $Q^{(e)}$ 的 rank 上界。
Eckart--Young theorem 因而给出每条边上的奇异值尾部下界；同时对全部边成立，
便得到这些下界的最大值。正交 tree-SVD 逐边投影时，各截断残差的平方范数按照
正交投影误差估计相加，从而得到第二个界。

证毕。

谱下界也适用于非负 TTNS，但 $Q_{\mathrm{svd}}$ 可能有负元素。因此第二个界是
无符号 TTNS 的构造界，不能直接当作非负 TTNS 的可达上界。实际非负拟合误差
与两个谱量的差距，可以用来诊断 nonnegativity 和优化引入的额外代价。

### 命题 2E：非负 tree conditional 材料化的三个可计算界

设两个概率分布具有相同的有向 forest 结构：

$$
p_T(y)
=
\prod_{r}p_r(y_r)
\prod_{v\notin\operatorname{root}(T)}
p_v(y_v\mid y_{\operatorname{pa}_T(v)}),
$$

$$
q_T(y)
=
\prod_{r}\widehat p_r(y_r)
\prod_{v\notin\operatorname{root}(T)}
\widehat p_v(y_v\mid y_{\operatorname{pa}_T(v)}).
$$

假设 $P_T\ll Q_T$。定义逐条件 TV 和

$$
\begin{aligned}
A_T
={}&
\sum_{r}\|P_r-\widehat P_r\|_{\mathrm{TV}}\\
&+
\sum_{v\notin\operatorname{root}(T)}
\mathbb E_{X\sim P_{\operatorname{pa}_T(v)}}
\left[
\left\|
P_v(\cdot\mid X)-\widehat P_v(\cdot\mid X)
\right\|_{\mathrm{TV}}
\right],
\end{aligned}
$$

以及 conditional cross-entropy excess

$$
\begin{aligned}
D_T
={}&
\sum_r D_{\mathrm{KL}}(P_r\|\widehat P_r)\\
&+
\sum_{v\notin\operatorname{root}(T)}
\mathbb E_{X\sim P_{\operatorname{pa}_T(v)}}
\left[
D_{\mathrm{KL}}
\left(
P_v(\cdot\mid X)
\middle\|
\widehat P_v(\cdot\mid X)
\right)
\right].
\end{aligned}
$$

若离散 joint table 共有 $n_T$ 个 cells，再定义

$$
F_T
=
\frac{\sqrt{n_T}}{2}\|p_T-q_T\|_F.
$$

则

$$
\boxed{
D_T
=
D_{\mathrm{KL}}(P_T\|Q_T),
\qquad
\|P_T-Q_T\|_{\mathrm{TV}}
\le
\min
\left\{
A_T,\,
\sqrt{\frac{D_T}{2}},\,
F_T
\right\}.
}
$$

### 证明

第一个界按从叶节点到根节点的顺序，逐个把 $p_v(\cdot\mid x)$ 替换为
$\widehat p_v(\cdot\mid x)$。替换某条边时，其父节点的边缘分布尚未因祖先
kernel 替换而改变；已经替换的后代 kernel 积分为 1，因此也不改变该父边缘。
该次替换的 joint TV 不超过题中对应的父边缘加权 conditional TV。最后替换各
根分布，并使用公共下游 Markov kernel 的 TV contraction。对全部 hybrid
distribution 使用三角不等式求和，得到 $A_T$。

由 forest factorization 和 KL chain rule，

$$
D_{\mathrm{KL}}(P_T\|Q_T)
=
\sum_r D_{\mathrm{KL}}(P_r\|\widehat P_r)
+
\sum_v
\mathbb E_{P_{\operatorname{pa}_T(v)}}
D_{\mathrm{KL}}
\left(
P_v(\cdot\mid X)
\middle\|
\widehat P_v(\cdot\mid X)
\right)
=D_T.
$$

Pinsker inequality 给出第二个界。最后，

$$
\|p_T-q_T\|_1
\le
\sqrt{n_T}\|p_T-q_T\|_F
$$

给出第三个界。

证毕。

如果每个离散 conditional matrix 具有非负 rank-$r$ 分解，则这些因子可以吸收
到相邻 tree cores，产生 bond rank 不超过 $r$ 的非负 TTNS。命题 2E 因而把每条
conditional matrix 的局部低 rank 误差直接接到整体 TTNS fitting error。
$D_T$ 还等于相对于真实 tree conditionals 的 cross-entropy excess。因此若训练
目标是 conditional cross-entropy，它可由“当前 loss 减去无限容量 tree
optimum”得到。若训练目标是 Frobenius/NMF loss，则必须在训练后另算 $D_T$，
不能把 Frobenius residual 当作 KL。

### 命题 2F：有限状态完整概率表上的 a posteriori certificate

设每层状态空间均有限，并令

$$
P_\ell=P_{\ell-1}K_\ell,
\qquad
R_\ell=Q_{\ell-1}K_\ell.
$$

这里 $R_\ell$ 由有限和精确计算，所以不存在 quadrature 项。对 $R_\ell$ 选择
forest $T_\ell$，并用它的真实根边缘与 tree-edge conditionals 构造
$R_\ell^{T_\ell}$。令非负 TTNS 材料化 $Q_\ell$ 与 $T_\ell$ 具有相同拓扑，
所有 conditional tables 已归一化，并满足
$R_\ell^{T_\ell}\ll Q_\ell$。

取与 $T_\ell$ 方向相容的次序 $v_1,\ldots,v_K$，并定义

$$
\mathcal A_{\ell,j}
=
\{v_1,\ldots,v_{j-1}\}
\setminus
\{\operatorname{pa}_{T_\ell}(v_j)\}.
$$

定义结构信息损失

$$
\begin{aligned}
S_\ell
&=
\operatorname{TC}_{R_\ell}(Y_1,\ldots,Y_K)
-
\sum_{(u,v)\in E(T_\ell)}
I_{R_\ell}(Y_u;Y_v)\\
&=
\sum_j
I_{R_\ell}
\left(
Y_{v_j};
Y_{\mathcal A_{\ell,j}}
\mid
Y_{\operatorname{pa}_{T_\ell}(v_j)}
\right),
\end{aligned}
$$

并定义拟合信息损失

$$
\begin{aligned}
D_\ell
={}&
\sum_r
D_{\mathrm{KL}}
\left(
(R_\ell^{T_\ell})_r
\middle\|
(Q_\ell)_r
\right)\\
&+
\sum_{v\notin\operatorname{root}(T_\ell)}
\mathbb E_{X\sim
(R_\ell^{T_\ell})_{\operatorname{pa}_{T_\ell}(v)}}
D_{\mathrm{KL}}
\left(
(R_\ell^{T_\ell})_v(\cdot\mid X)
\middle\|
(Q_\ell)_v(\cdot\mid X)
\right).
\end{aligned}
$$

把命题 2E 中逐根与逐边的 conditional TV 和记为 $A_\ell$，令 $n_\ell$
是该层 joint table 的 cell 数，并令

$$
b_\ell
=
\min
\left\{
A_\ell,\,
\sqrt{\frac{D_\ell}{2}},\,
\frac{\sqrt{n_\ell}}{2}
\|R_\ell^{T_\ell}-Q_\ell\|_F
\right\}.
$$

则

$$
\boxed{
\|P_L-Q_L\|_{\mathrm{TV}}
\le
\|P_0-Q_0\|_{\mathrm{TV}}
+
\sum_{\ell=1}^{L}
\left[
\sqrt{\frac{S_\ell}{2}}
+
b_\ell
\right].
}
$$

### 证明

Markov contraction、命题 2C、命题 2E 和三角不等式依次给出

$$
\begin{aligned}
\|P_\ell-Q_\ell\|_{\mathrm{TV}}
&\le
\|P_{\ell-1}K_\ell-Q_{\ell-1}K_\ell\|_{\mathrm{TV}}
+
\|R_\ell-R_\ell^{T_\ell}\|_{\mathrm{TV}}
+
\|R_\ell^{T_\ell}-Q_\ell\|_{\mathrm{TV}}\\
&\le
\|P_{\ell-1}-Q_{\ell-1}\|_{\mathrm{TV}}
+
\sqrt{\frac{S_\ell}{2}}
+
b_\ell.
\end{aligned}
$$

逐层展开即得结论。

证毕。

这个版本的右端没有未估计的“structure error”或“fit error”占位符，
但 $S_\ell,D_\ell,A_\ell$ 仍要从完整概率表事后计算。因此它是实现审计所用的
a posteriori certificate，不是只依赖模型参数的先验误差定理。
$S_\ell$ 由 marginal entropy 和 edge mutual information 直接计算；三节点
tree 中它就是唯一被删除的 conditional mutual information。$D_\ell$ 是逐根、
逐边 conditional KL 的加权和，也等于 cross-entropy excess。$A_\ell$ 逐列
比较 conditional tables，Frobenius 项直接比较两个 joint tensors。三者都由
材料化前后的离散表确定，并取其中最小者。

若 $P_0=Q_0$，初始项为零。若还希望利用严格收缩，可在有限状态空间枚举

$$
\alpha_\ell
=
\max_{x,x'}
\frac12
\sum_y
|K_\ell(y\mid x)-K_\ell(y\mid x')|.
$$

此时递推中的继承误差可乘以 $\alpha_\ell$，展开后产生相应的 coefficient
products。这不是未定义常数，而是有限 kernel table 上的有限最大值。但对支撑
相隔足够远的 max-plus 输入，$\alpha_\ell$ 常等于 1；本文四层审计正是这种
情形，所以不把事后观察到的约 $0.51$ 比值冒充先验 contraction coefficient。

### 定理 2：由 delay overlap、interaction、rank 和 quadrature 参数控制的先验界

对第 $\ell$ 层，设条件输出可以写成

$$
Y_\ell
=
H_\ell(X_{\ell-1},E_\ell)+D_\ell,
$$

其中

$$
H_{\ell,a}(x,e)
=
\max_{i\in P_{\ell,a}}
\{x_i+e_{ia}\},
$$

$E_\ell$ 收集 edge delays，$D_\ell$ 是与 $(X_{\ell-1},E_\ell)$ 独立的
node-delay 向量，具有密度 $g_\ell$。以下四组量都在传播和材料化之前由模型
假设给定。

第一，选择 edge-delay 事件 $\mathcal E_\ell$，使得

$$
\mathbb P(E_\ell\in\mathcal E_\ell)
\ge
1-\tau_\ell^{\mathrm{edge}}.
$$

设存在紧集 $\mathcal U_\ell$，使得对所有允许的输入 $x$ 和
$e\in\mathcal E_\ell$，

$$
H_\ell(x,e)\in\mathcal U_\ell.
$$

定义 node-delay 的公共平移重叠

$$
\beta_\ell
=
(1-\tau_\ell^{\mathrm{edge}})
\int_{\mathbb R^{K_\ell}}
\inf_{u\in\mathcal U_\ell}
g_\ell(y-u)\,dy,
\qquad
\rho_\ell=1-\beta_\ell.
$$

第二，取 $0\le\tau_\ell^{\mathrm{quad}}<1$。数值传播在一个 $m_\ell$ 维积分盒

$$
\mathcal B_\ell
=
\prod_{q=1}^{m_\ell}[a_{\ell,q},b_{\ell,q}]
$$

内进行，盒外概率质量不超过 $\tau_\ell^{\mathrm{quad}}$。对每个输出 cell
$z$，其盒内概率质量的 integrand 记为 $f_{\ell,z}(t)$。设使用各坐标步长
$h_{\ell,q}$ 的 tensor-product composite trapezoidal rule，并且

$$
B_{\ell,q}
=
\sup_z\sup_{t\in\mathcal B_\ell}
\left|
\frac{\partial^2 f_{\ell,z}(t)}
{\partial t_q^2}
\right|
<
\infty.
$$

令 $n_\ell$ 是输出 probability table 的 cell 数，并定义

$$
\nu_\ell
=
\tau_\ell^{\mathrm{quad}}
+
\frac{
n_\ell\operatorname{vol}(\mathcal B_\ell)
}{
12(1-\tau_\ell^{\mathrm{quad}})
}
\sum_{q=1}^{m_\ell}
h_{\ell,q}^2 B_{\ell,q}.
$$

第三，设精确数值传播分布存在一个 tree-structured candidate
$G_\ell$，并且

$$
\operatorname{osc}
\left(
\log\frac{dR_\ell^{\mathrm{num}}}{dG_\ell}
\right)
\le
\Omega_\ell.
$$

这里 $\operatorname{osc}(f)=\sup f-\inf f$。定义

$$
\mathfrak s(\Omega)
=
\min
\left\{
1,\,
\sqrt{\frac{\Omega}{2}},\,
\frac{e^\Omega-1}{\sqrt2}
\right\}.
$$

第四，在用于材料化的 compact output domain 上，对 Chow--Liu tree projection
的每个非根节点 $v$，设 scalar tree parent 的区间长度为
$D_{\ell,v}$。保留 root marginals 不变，并假设 tree projection 的
conditional law 满足

$$
\left\|
(R_\ell^{T_\ell})_v(\cdot\mid x)
-
(R_\ell^{T_\ell})_v(\cdot\mid x')
\right\|_{\mathrm{TV}}
\le
L_{\ell,v}|x-x'|^{s_{\ell,v}},
\qquad
0<s_{\ell,v}\le1.
$$

用 $r_{\ell,v}$ 个等长 parent bins 构造非负 separable conditional，并定义

$$
a_\ell
=
\sum_{v\notin\operatorname{root}(T_\ell)}
L_{\ell,v}
\left(
\frac{D_{\ell,v}}{r_{\ell,v}}
\right)^{s_{\ell,v}}.
$$

若 $Q_\ell$ 是上述数值传播、Chow--Liu projection 和非负 rank construction
依次得到的材料化分布，则

$$
\boxed{
\begin{aligned}
\|P_L-Q_L\|_{\mathrm{TV}}
\le{}&
\|P_0-Q_0\|_{\mathrm{TV}}
\prod_{\ell=1}^{L}\rho_\ell\\
&+
\sum_{\ell=1}^{L}
\left[
\nu_\ell+\mathfrak s(\Omega_\ell)+a_\ell
\right]
\prod_{j=\ell+1}^{L}\rho_j.
\end{aligned}
}
$$

### 证明

对任意输入 $x$，第 $\ell$ 层 transition density 满足

$$
\begin{aligned}
k_\ell(y\mid x)
&=
\mathbb E
\left[
g_\ell
\left(
y-H_\ell(x,E_\ell)
\right)
\right]\\
&\ge
(1-\tau_\ell^{\mathrm{edge}})
\inf_{u\in\mathcal U_\ell}g_\ell(y-u).
\end{aligned}
$$

右侧积分为 $\beta_\ell$，因此存在与 $x$ 无关的概率测度 $\lambda_\ell^0$，使

$$
K_\ell(x,\cdot)
=
\beta_\ell\lambda_\ell^0(\cdot)
+
(1-\beta_\ell)\widetilde K_\ell(x,\cdot).
$$

于是

$$
\|PK_\ell-QK_\ell\|_{\mathrm{TV}}
\le
\rho_\ell\|P-Q\|_{\mathrm{TV}}.
$$

对 quadrature，重复使用一维 composite trapezoidal error formula 和正权重
telescoping，得到每个 cell 的绝对误差不超过

$$
\frac{\operatorname{vol}(\mathcal B_\ell)}{12}
\sum_{q=1}^{m_\ell}h_{\ell,q}^2B_{\ell,q}.
$$

对 $n_\ell$ 个 cells 求和。若原始 quadrature masses 最后重新归一化，则

$$
\left\|p-\frac{\widehat p}{\sum_z\widehat p_z}\right\|_{\mathrm{TV}}
\le
\|p-\widehat p\|_1.
$$

盒内条件分布的 integrand 是原始 integrand 除以至少
$1-\tau_\ell^{\mathrm{quad}}$，所以 derivative bound 相应至多放大
$1/(1-\tau_\ell^{\mathrm{quad}})$。再加入盒外截断质量，得到 numerical TV
不超过 $\nu_\ell$。

对结构项，令

$$
Z_\ell
=
\frac{dR_\ell^{\mathrm{num}}}{dG_\ell}.
$$

由 $\mathbb E_{G_\ell}Z_\ell=1$ 和
$\operatorname{osc}(\log Z_\ell)\le\Omega_\ell$，可得

$$
e^{-\Omega_\ell}
\le
Z_\ell
\le
e^{\Omega_\ell}.
$$

因此

$$
D_{\mathrm{KL}}
(R_\ell^{\mathrm{num}}\|G_\ell)
\le
\min
\left\{
\Omega_\ell,\,
(e^{\Omega_\ell}-1)^2
\right\}.
$$

Chow--Liu projection 是全部 tree distributions 中的 KL information
projection，故其 KL 不大于上式。使用 Pinsker inequality，得到结构 TV
不超过 $\mathfrak s(\Omega_\ell)$。

对 rank 项，将 parent interval 分成 $r_{\ell,v}$ 个等长 bins，并在每个 bin
内选一个代表点 $c_j$。定义

$$
\widehat P_{\ell,v}(\cdot\mid x)
=
(R_\ell^{T_\ell})_v(\cdot\mid c_j),
\qquad
x\in B_j.
$$

这是 $r_{\ell,v}$ 个非负 parent indicators 与 child distributions 的和，
所以 conditional separation rank 不超过 $r_{\ell,v}$。Hölder 条件给出

$$
\sup_x
\left\|
(R_\ell^{T_\ell})_v(\cdot\mid x)
-
\widehat P_{\ell,v}(\cdot\mid x)
\right\|_{\mathrm{TV}}
\le
L_{\ell,v}
\left(
\frac{D_{\ell,v}}{r_{\ell,v}}
\right)^{s_{\ell,v}}.
$$

命题 2E 的 hybrid-kernel telescoping 因而给出整体非负 TTNS
materialization TV 不超过 $a_\ell$。每条 conditional separation 的
component index 可作为对应 tree edge 的 bond index，所以所得 joint
distribution 是各 edge bond rank 不超过 $r_{\ell,v}$ 的非负 TTNS。最后，
每层三角不等式给出

$$
\|P_\ell-Q_\ell\|_{\mathrm{TV}}
\le
\rho_\ell\|P_{\ell-1}-Q_{\ell-1}\|_{\mathrm{TV}}
+
\nu_\ell+\mathfrak s(\Omega_\ell)+a_\ell.
$$

展开递推即得结论。

证毕。

这个定理是先验 approximation theorem，而不是对当前非凸 R5 optimizer 的
convergence theorem。它证明存在一个满足给定 bond ranks 的显式非负 TTNS
construction。若实际训练器没有达到这个 construction 的误差，还必须增加独立
证明或认证的 optimization error；当前仓库尚无这部分理论。

这里的 $\Omega_\ell,L_{\ell,v},s_{\ell,v}$ 必须由模型类别、density bounds
或 interaction coefficients 在运行实验前统一控制。若它们只是从已经生成的
完整 joint table 中回算，定理在逻辑上就退化为命题 2F 的 a posteriori
certificate。compact output truncation 的尾概率也必须计入
$\tau_\ell^{\mathrm{quad}}$，不能静默丢弃。

结构假设也可以由非树 interaction potentials 推出。若

$$
\frac{dR_\ell^{\mathrm{num}}}{dG_\ell}(y)
\propto
\exp
\left\{
\sum_{A\notin T_\ell}V_{\ell,A}(y_A)
\right\},
$$

则可直接取

$$
\Omega_\ell
\le
\sum_{A\notin T_\ell}
\operatorname{osc}(V_{\ell,A}).
$$

因此结构项确实由被 tree 删除的 interaction strength 控制，而不是重新命名
后的未知 TV。

### 推论 6：Gaussian node delay 的闭式传播系数

若 $D_{\ell,a}$ 相互独立且
$D_{\ell,a}\sim\mathcal N(0,\sigma_{\ell,a}^2)$，并且

$$
\mathcal U_\ell
\subseteq
\prod_{a=1}^{K_\ell}
[u_{\ell,a}^-,u_{\ell,a}^+],
\qquad
\Delta_{\ell,a}=u_{\ell,a}^+-u_{\ell,a}^-,
$$

则

$$
\boxed{
\rho_\ell
\le
1-
(1-\tau_\ell^{\mathrm{edge}})
\prod_{a=1}^{K_\ell}
2\Phi
\left(
-\frac{\Delta_{\ell,a}}{2\sigma_{\ell,a}}
\right).
}
$$

这是因为一维等方差 Gaussian densities 在长度为 $\Delta$ 的全部平移之间的
公共重叠质量为 $2\Phi(-\Delta/(2\sigma))$，独立坐标的 rectangular lower
bound 相乘即可。

对 log-skew-normal node delay，虽然通常没有同样简短的闭式，但仍可由其已知
density 参数计算

$$
\int
\inf_{u\in[u^-,u^+]}
g_{\mathrm{LSN}}(y-u)\,dy.
$$

只要平移区间有界且 density 在其支撑内部为正，该量为正。若 node delay 为
零、edge-delay 截断后仍允许两个不同的 deterministic outputs，则公共重叠为
零，定理只能给出 $\rho_\ell=1$；此时不存在对全部输入分布都成立的严格 TV
收缩。

连续积分需要单独增加可证明的 quadrature 项。具体地，若局部 block 的每个
entry 都是

$$
M_i=\int_{a_i}^{b_i}g_i(x)\,dx,
$$

并使用步长 $h_i$ 的 composite trapezoidal rule，且
$g_i\in C^2([a_i,b_i])$，则

$$
|M_i-\widehat M_i|
\le
\frac{(b_i-a_i)h_i^2}{12}
\sup_{x\in[a_i,b_i]}|g_i''(x)|.
$$

把这个逐 entry 上界代入后文引理 3 的 telescoping formula，就得到完全由
$h_i$、积分区间、二阶导数上界和其余 block 的 sup norm 组成的 CDF 误差界。
若 delay CDF 具有 kink 或 atom，$C^2$ 假设不成立，便不能宣称该二阶界；此时
必须分段积分，或把高分辨率差异明确称为 grid-estimated numerical error。
本文 Uniform delay 的连续审计属于后一种情形，因此没有把经验网格差异写进
定理 2 的严格 TV 右端。

### 10.2 局部投影误差的多线性扰动界

定义系数张量的 entrywise one-norm：

$$
\|A\|_1
=
\sum_{i_1=1}^{m_1}\cdots\sum_{i_d=1}^{m_d}
|A_{i_1,\ldots,i_d}|.
\tag{10.7}
$$

对向量 $z\in\mathbb R^m$，记

$$
\|z\|_\infty=\max_i|z_i|.
\tag{10.8}
$$

### 引理 3：TTNS 多线性收缩的局部扰动界

给定两组局部向量

$$
z_k,\widehat z_k\in\mathbb R^{m_k},
\qquad k\in[d],
$$

定义

$$
\delta_k=\|z_k-\widehat z_k\|_\infty,
\qquad
\beta_k=\max\{\|z_k\|_\infty,\|\widehat z_k\|_\infty\}.
\tag{10.9}
$$

则

$$
\left|
\mathcal C_T(z_1,\ldots,z_d)
-
\mathcal C_T(\widehat z_1,\ldots,\widehat z_d)
\right|
\le
\|A\|_1
\sum_{k=1}^{d}
\delta_k
\prod_{\substack{j=1\\j\ne k}}^{d}\beta_j.
\tag{10.10}
$$

### 证明

由多线性得到 telescoping identity：

$$
\begin{aligned}
&
\mathcal C_T(z_1,\ldots,z_d)
-
\mathcal C_T(\widehat z_1,\ldots,\widehat z_d)\\
&=
\sum_{k=1}^{d}
\mathcal C_T
\left(
\widehat z_1,\ldots,\widehat z_{k-1},
z_k-\widehat z_k,
z_{k+1},\ldots,z_d
\right).
\end{aligned}
\tag{10.11}
$$

对第 $k$ 项展开 (3.1)，使用三角不等式，得到

$$
\begin{aligned}
&
\left|
\mathcal C_T
\left(
\widehat z_1,\ldots,\widehat z_{k-1},
z_k-\widehat z_k,
z_{k+1},\ldots,z_d
\right)
\right|\\
&\le
\|A\|_1
\delta_k
\prod_{j<k}\|\widehat z_j\|_\infty
\prod_{j>k}\|z_j\|_\infty\\
&\le
\|A\|_1
\delta_k
\prod_{j\ne k}\beta_j.
\end{aligned}
\tag{10.12}
$$

对 $k$ 求和即得 (10.10)。

证毕。

该界直接作用于定理 1。令 $z_k=M_k(y,d)$、
$\widehat z_k=\widehat M_k(y,d)$，则每个一维求积误差通过 $\delta_k$ 进入，
无需构造完整 $d$ 维积分。$\|A\|_1$ 是系数张量本身的范数，而不是某个具有
gauge 自由度的 core 范数；该界可能保守，但定义明确。

### 10.3 Node-delay 求积的 Wasserstein 界

在 $\mathbb R^K$ 上固定一个度量 $\rho$。对具有有限一阶矩的概率测度
$\mu$ 和 $\nu$，定义 Wasserstein-1 distance

$$
W_{1,\rho}(\mu,\nu)
=
\inf_{\pi\in\Pi(\mu,\nu)}
\int
\rho(d,\widehat d)
\,\pi(dd,d\widehat d),
\tag{10.13}
$$

其中 $\Pi(\mu,\nu)$ 是以 $\mu$ 和 $\nu$ 为两个边缘分布的全部 coupling 集合。

### 命题 3：外层 node-delay 离散误差

固定 $y\in\mathbb R^K$，定义

$$
g_y(d)
=
\mathcal C_T
\bigl(
M_1(y,d),\ldots,M_d(y,d)
\bigr).
\tag{10.14}
$$

设真实 node-delay 分布为 $\mu_D$，离散求积分布为 $\nu_D$。如果存在
$L_D(y)<\infty$，使

$$
|g_y(d)-g_y(\widehat d)|
\le
L_D(y)\rho(d,\widehat d)
\tag{10.15}
$$

对所有 $d,\widehat d$ 成立，则

$$
\left|
\int g_y\,d\mu_D
-
\int g_y\,d\nu_D
\right|
\le
L_D(y)W_{1,\rho}(\mu_D,\nu_D).
\tag{10.16}
$$

### 证明

对任意 $\pi\in\Pi(\mu_D,\nu_D)$，

$$
\begin{aligned}
\left|
\int g_y\,d\mu_D-\int g_y\,d\nu_D
\right|
&=
\left|
\int
\bigl(g_y(d)-g_y(\widehat d)\bigr)
\pi(dd,d\widehat d)
\right|\\
&\le
L_D(y)
\int\rho(d,\widehat d)\,\pi(dd,d\widehat d).
\end{aligned}
$$

对全部 coupling 取下确界即得 (10.16)。

证毕。

命题 3 是 Kantorovich–Rubinstein Lipschitz bound 的直接应用。它不要求
node delay 具有密度，也不预设求积是二阶。若希望得到
$O(n_d^{-2})$ 等具体阶数，必须另外证明 $g_y$ 的相应光滑性，并指定求积规则。

### 10.4 单层 CDF 的完整误差分解

设 $P$ 是真实上游分布，$Q$ 是由系数张量 $A$ 和基展开定义的 TTNS 概率分布，
$K_{\max}$ 是 (5.1) 所定义的 max-plus Markov kernel。记真实下游 CDF 为

$$
F_{P}(y)=(PK_{\max})((-\infty,y]).
\tag{10.17}
$$

这里

$$
(-\infty,y]
=
\prod_{a=1}^{K}(-\infty,y_a].
$$

令 $\mu_D$ 是真实 node-delay 分布，$\nu_D$ 是有限求积点及其权重所定义的离散
概率分布。对每个求积点 $d$，用数值局部投影
$\widehat M_k(y,d)$ 代替精确的 $M_k(y,d)$。最后在有限输出网格上计算并通过
插值或边界外推得到数值 CDF $\widehat F(y)$。

定义局部投影误差上界

$$
\delta_k(y)
\ge
\sup_{d\in\operatorname{supp}(\nu_D)}
\|M_k(y,d)-\widehat M_k(y,d)\|_\infty,
\tag{10.18}
$$

以及

$$
\beta_k(y)
\ge
\sup_{d\in\operatorname{supp}(\nu_D)}
\max\{
\|M_k(y,d)\|_\infty,
\|\widehat M_k(y,d)\|_\infty
\}.
\tag{10.19}
$$

将输出网格内部的插值误差记为 $\varepsilon_{\mathrm{interp}}(y)$，将边界外推
或有限支撑截断产生的误差记为 $\varepsilon_{\mathrm{tail}}(y)$。

### 定理 3：单层 max-plus CDF 总误差界

若命题 3 的 Lipschitz 条件成立，则

$$
\begin{aligned}
|F_P(y)-\widehat F(y)|
\le\;&
\|P-Q\|_{\mathrm{TV}}\\
&+
\|A\|_1
\sum_{k=1}^{d}
\delta_k(y)
\prod_{\substack{j=1\\j\ne k}}^{d}\beta_j(y)\\
&+
L_D(y)W_{1,\rho}(\mu_D,\nu_D)\\
&+
\varepsilon_{\mathrm{interp}}(y)
+
\varepsilon_{\mathrm{tail}}(y).
\end{aligned}
\tag{10.20}
$$

### 证明

在真实传播、TTNS 模型的精确传播、node-delay 离散传播、局部投影离散传播和
输出网格插值之间依次插入中间项并使用三角不等式。

第一项由命题 1 得到：

$$
\left|
(PK_{\max})((-\infty,y])
-
(QK_{\max})((-\infty,y])
\right|
\le
\|P-Q\|_{\mathrm{TV}}.
\tag{10.21}
$$

第二项对 $\nu_D$ 的每个求积点应用引理 3，再利用 (10.18)–(10.19)。
第三项由命题 3 得到。最后两个误差按照定义控制有限输出网格上的插值和边界
外推。合并各项即得 (10.20)。

证毕。

若输出 CDF 在某个网格单元上关于所选输出度量是 $L_Y$-Lipschitz，且多线性
插值所使用的所有顶点到查询点的距离不超过 $h$，则可取

$$
\varepsilon_{\mathrm{interp}}(y)\le L_Yh.
\tag{10.22}
$$

对于上边界 $u=(u_1,\ldots,u_K)$，把 $F(u)$ 外推为 1 所产生的误差满足

$$
1-F(u)
=
P\left(\bigcup_{a=1}^{K}\{Y_a>u_a\}\right)
\le
\sum_{a=1}^{K}P(Y_a>u_a).
\tag{10.23}
$$

因此 $\varepsilon_{\mathrm{tail}}$ 可以由各输出的一维尾概率控制，而不应仅凭
选择了一个较大的 `s_max` 就假定为 0。

若 $P$ 和 $Q$ 分别具有密度 $p$ 和 $q$，并且二者都支撑在 Lebesgue measure
有限的集合 $\Omega\subset\mathbb R^d$ 上，则由 Cauchy–Schwarz 不等式，

$$
\|P-Q\|_{\mathrm{TV}}
=
\frac12\|p-q\|_{L^1(\Omega)}
\le
\frac12\sqrt{|\Omega|}
\|p-q\|_{L^2(\Omega)}.
\tag{10.24}
$$

因此，若训练过程能够控制真实的 $L^2$ 密度误差，则 (10.24) 可以把它接入
(10.20)。经验 L2 objective 只有在另行建立泛化误差控制后，才能替代
$\|p-q\|_{L^2(\Omega)}$。

最后，令

$$
\Pi_{[0,1]}(z)=\min\{1,\max\{0,z\}\}.
$$

因为真实 CDF 值 $F\in[0,1]$，所以

$$
|\Pi_{[0,1]}(\widehat F)-F|
\le
|\widehat F-F|.
\tag{10.25}
$$

当前实现中的 elementwise clipping 不会增加相对于合法真实 CDF 的绝对误差，
因此不需要在 (10.20) 中再增加一个 clipping error；但频繁 clipping 仍应作为
数值诊断报告。

### 10.5 哪些部分是本文结果

- 定理 1 及其共享父节点、多输出和 TTNS forest 形式，是本文需要主张和检索
  新颖性的结构化结果。
- 引理 3 是一般有限多线性形式的扰动界；命题 1、命题 2、命题 2A 和命题 3
  分别使用 Markov kernel contraction、三角不等式和
  Kantorovich–Rubinstein bound；命题 2B 还使用 max-plus 映射在
  $\ell_\infty$ metric 下的 $1$-Lipschitz 性质。这些基础工具不应声称为首次
  提出。
- 命题 2C 是 Chow--Liu information projection 的经典 KL 恒等式及 chain-rule
  写法；命题 2D 使用经典 Eckart--Young 和 tree-SVD 截断界；命题 2E 使用
  KL chain rule、Pinsker、Cauchy--Schwarz 和 hybrid-kernel telescoping。
  这些组成部分同样不能单独主张为理论创新。
- 定理 3 的贡献是把这些标准工具按当前 TTNS max-plus 数值流程组合成可审计的
  单层误差预算；命题 2F 是完整概率表上的事后 certificate。定理 2 则给出
  参数级先验界，把传播、结构、rank 和 quadrature 分别写成 delay common
  overlap、非树 log-interaction oscillation、conditional TV-Hölder
  approximation rate 和显式求积常数。这里可能形成论文贡献的是这些量在
  TTNS max-plus 逐层材料化中的组合与证明；Doeblin、Pinsker、Chow--Liu 和
  Hölder partition approximation 等组成工具本身仍是经典结果。

平方 TTNS 可以把每个局部矩阵 $W_k$ 向量化，并把 doubled coefficient tensor
$B\otimes B/Z$ 视为新的有限多线性系数张量，从而直接应用引理 3 和定理 3。
该观察不改变误差分解的结构，只会把 $\|A\|_1$、$\delta_k$ 和 $\beta_k$
替换为 doubled network 对应的量。

## 11. 算术复杂度

本节只给固定查询点和理想精确一维积分下的基本算术上界，不包含编译、缓存、并行和稀疏性收益。

对节点 $k$，core 的元素数为

$$
m_k\prod_{e\in\delta(k)}r_e.
$$

在给定所有局部向量 $M_k$ 后，沿物理指标和树边完成一次收缩的算术量可由

$$
O\left(
\sum_{k=1}^{d}
m_k\prod_{e\in\delta(k)}r_e
\right)
\tag{11.1}
$$

控制。该表达式是按所有 core 大小求和得到的基本上界；实际 contraction order 可能改变常数。

如果每个一维积分用 $Q$ 个求积点计算，则构造一个固定 $(y,d)$ 对应的全部局部向量需要至多

$$
O\left(Q\sum_{k=1}^{d}m_k c_k\right)
\tag{11.2}
$$

次基本运算，其中

$$
c_k=|\{a\in[K]:k\in P_a\}|
$$

是上游节点 $k$ 同时作为多少个下游节点的父节点。若逐点直接计算 CDF 乘积，$c_k$ 进入常数；若复用中间结果，可进一步降低代价。

如果每个下游坐标使用含 $G$ 个点的网格，则完整 $K$ 维联合 CDF 表具有

$$
G^K
$$

个查询点。因此，即使每个点的 TTNS contraction 是可控的，完整联合表仍随 $K$ 指数增长。这解释了完整联合 R6 只适用于小块。

对于平方 TTNS，物理维由 $m_k$ 升至 $m_k^2$，每条 bond dimension 由 $r_e$ 升至 $r_e^2$。因此一次 doubled contraction 的基本上界相应变为

$$
O\left(
\sum_{k=1}^{d}
m_k^2
\prod_{e\in\delta(k)}r_e^2
\right).
\tag{11.3}
$$

这些上界还需要与实际实现的 contraction path 做逐项核对，投稿时应同时给出理论复杂度和实测 wall-clock/memory scaling。

## 12. “解析”的准确含义

定理 1 的结论是：原本的 $d$ 维积分可以精确化为

1. 每个上游坐标上的一维局部投影积分；
2. 一个有限 TTNS contraction；
3. 对 node delays 的外层期望。

因此，“解析”在本文中应解释为

> deterministic reduction to one-dimensional projections and finite tree-tensor contractions, without Monte Carlo sampling of the upstream distribution.

它不意味着所有一维积分都具有初等函数闭式表达。代码中仍然使用网格求积、node-delay quadrature 和插值。

## 13. 连续定理与当前实现的差别

理论公式 (7.3) 假设局部积分和 node-delay expectation 被精确计算。当前代码进行了以下数值近似：

1. 用有限 `q_grid` 近似 (7.1) 的一维积分；
2. 用有限 `n_d` 的分位点求积近似对 $D$ 的期望；
3. 在有限 `s_grid` 上计算 CDF；
4. 对平移后的 CDF 使用线性插值；
5. 在数值误差导致结果超出 $[0,1]$ 时进行 clipping；
6. moments 和 covariance 只在有限区间上积分。

因此，当前实现是定理 1 的确定性数值离散，而不是所有积分均精确的符号计算。

已经完成的轻量数值检查包括：

- 局部求积误差随 `q_grid` 的经验收敛；
- node-delay quadrature 随 `n_d` 的经验收敛；
- pair CDF 的边缘一致性；
- CDF 取值范围、单调性和输出排列一致性；
- 小维 full tensor、TTNS contraction 与直接积分的对照；
- log-skew-normal delay；
- 平方 TTNS doubled contraction。

投稿前仍需补充有限 `s_max` 的 tail truncation audit，以及与实际 contraction
path 对应的 wall-clock 和峰值内存 scaling。经验收敛曲线不能替代理论收敛阶。

其中第一轮轻量验证已经完成，来源为
`simple_ttns_l2/reports/maxplus_cdf_theorem_validation_metrics.json` 和
`simple_ttns_l2/reports/maxplus_cdf_theorem_validation_report_zh.md`，复现入口为
`simple_ttns_l2/experiments/validate_maxplus_cdf_theorem.py`。在一个三维、
rank-2、具有共享父节点的非负归一化 TTNS 构造上，TTNS pair CDF 与显式系数张量
的最大绝对差为 $5.55\times10^{-16}$，与选定点直接三维网格积分的最大绝对差为
$4.44\times10^{-16}$。相对于 `q_grid=2001` 数值参考，pair CDF 最大误差从
`q_grid=41` 时的 $2.08443\times10^{-3}$ 降至 `q_grid=321` 时的
$3.24116\times10^{-5}$。这些结果验证了当前实现与本定理公式在该构造上的一致性，
非零 uniform node delay 的 `n_d` 收敛以及 $K=3$ 完整联合 CDF 也已通过轻量
验证。其中 $K=3$ 收缩与显式系数张量公式的最大绝对差为
$3.33\times10^{-16}$。log-skew-normal edge/node delay 的确定性
`q_grid` 和 `n_d` 检查也呈稳定误差下降。平方 TTNS doubled contraction 的
归一化、marginal、pair 和直接三维积分检查全部通过；其中 doubled pair 与显式
平方系数张量的最大绝对差为 $3.44\times10^{-15}$。这些结果只验证实现路径，
不参与定理证明，也不等价于平方 TTNS 分层全链已经完成。

命题 2F 的轻量多层审计来源为
`simple_ttns_l2/reports/multilayer_error_analysis_metrics.json` 和
`simple_ttns_l2/reports/multilayer_error_analysis_zh.md`，复现入口为
`simple_ttns_l2/experiments/audit_multilayer_error_propagation.py`。连续单层
$K=3$ 构造上的 grid-estimated numerical、structural 和 fitting TV 分别为
$4.21143\times10^{-5}$、0.136266 和 0.040144；有限状态 $K=3,L=4$
构造中的 Markov contraction、单步递推和累积界检查全部通过。第 4 层实际全局
TV 为 0.226152，累积上界为 0.624913。连续单层结构 KL、`TC-edge MI` 和遗漏
条件互信息均为 0.0998766 nat，恒等式误差低于 $2\times10^{-15}$；rank-3
谱下界、无符号 TT-SVD 构造误差和非负材料化 Frobenius error 分别为
$8.82\times10^{-4}$、$1.1630\times10^{-3}$ 和
$1.5596\times10^{-3}$。连续 fitting conditional KL 为 0.0209111 nat，
最紧的可计算 fitting TV 界为 0.051810。命题 2F 的四层事后信息论
certificate 为
1.385602；由于它超过 TV 的自然上限，只能作为事后 certificate 失效的负面
诊断，不能验证定理 2 的先验 rate。有限状态第 2--4 层实际分布对的 TV 传播比约为
0.5083、0.5071 和 0.5203，而全局 Dobrushin coefficient 仍为 1。连续部分
不是严格连续 TV，有限状态部分也只是固定人工 Markov 模型；这些数值只验证误差
审计实现，不参与定理证明。

## 14. 与仓库实现的对应

| 数学对象 | 当前实现 |
|---|---|
| $F_{ka}$ | `simple_ttns_l2.maxplus_pipeline.edge_cdf` |
| node-delay expectation | `simple_ttns_l2.maxplus_pipeline.node_quadrature` |
| 单输出局部投影 | `UpperModel.proj_single` |
| 两输出共享父局部投影 | `UpperModel.proj_single_pair` |
| 任意多个输出共享父局部投影 | `UpperModel.proj_single_multi` |
| $\mathcal C_T$ | `UpperModel._contract` 调用 `batch_eval_rank1_ttns` |
| 单输出 CDF | `maxplus_cdf.marginal_cdf` |
| 两输出联合 CDF | `maxplus_cdf.pair_cdf` |
| 独立 forest 推论 | `maxplus_cdf_forest.UpperForest` |
| 小块完整联合 CDF | `maxplus_cdf_forest.block_joint_cdf` |
| 单层与多层误差审计 | `experiments/audit_multilayer_error_propagation.py` |

当前 `R5-nonneg` 是在有效 core 上保持非负的线性 TTNS 密度，因此定理 1 直接适用。平方密度 $p_X=\psi^2/Z$ 的推论 4 已有数学推导，但仓库尚未完成与线性 `UpperForest` 同等成熟的整链平方传播实现。

## 15. 新颖性边界

截至 2026-07-27 的初步检索，已经明确存在以下相邻工作。

1. Tensor-Train Density Estimation 已经证明 TT 密度可以计算 CDF、marginal、partition function 和进行采样。
2. Tree tensor network 已经被用于高维概率密度学习；也已有通过 Chow–Liu 选择 TTNS topology、给出 sample-complexity guarantee 的生成建模工作。
3. Tensor train 已经被用于随机有限体积、不确定性传播、概率密度离散计算和随机系统状态估计。
4. Tensor-network contraction 已经被用于工程网络可靠性计算。

本次检索没有发现与定理 1 完全相同的公开陈述，即没有发现同时包含下列全部要素的结果：

- 连续上游联合密度由 TTNS 基展开表示；
- 下游由具有共享父节点的多输出 max-plus 随机映射定义；
- edge 和 node delays 具有一般已知分布；
- 任意下游子集的联合 CDF 化为一维局部投影和原树收缩；
- 该结果用于逐层 TTNS density materialization。

但是，“没有检索到”不等于“证明从未有人做过”。正式投稿前仍应检索 MathSciNet、Web of Science、Scopus、Google Scholar、Crossref、arXiv 和目标应用领域的 max-plus/reliability/PERT 文献，并检查论文正文、附录和学位论文，而不只检查标题与摘要。

因此，目前可以使用的谨慎表述是：

> To the best of our knowledge, prior tensor-network density estimators do not provide the above joint-CDF contraction result for multilayer stochastic max-plus mappings with shared parents.

只有在系统文献综述完成后，才能决定是否保留 `to the best of our knowledge`。目前不能写“这是世界上首次”或“以前没有人做过”。

## 16. 结论

定理 1 的数学核心由三个事实组成：

1. max-plus 阈值事件在独立 edge delays 下转化为 edge CDF 的乘积；
2. 按上游变量重新分组后，该乘积是坐标可分离函数；
3. TTNS 密度对可分离函数的积分等于局部一维投影向量与系数 TTNS 的树收缩。

这三个步骤给出了完整、有限且无需上游 Monte Carlo 采样的联合 CDF 计算公式。平方 TTNS、独立 forest、marginal CDF、pair CDF 和小块完整联合 CDF 都是该一般定理的推论。
