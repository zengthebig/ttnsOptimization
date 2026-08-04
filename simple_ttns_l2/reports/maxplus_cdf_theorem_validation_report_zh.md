# TTNS max-plus 联合 CDF 收缩定理数值验证

## 1. 验证目的

本实验验证 `maxplus_ttns_cdf_theorem_zh.md` 中定理 1 对应的当前代码实现。实验只检查数学恒等式和数值离散，不训练模型，也不使用 Monte Carlo。

本实验回答以下问题。

1. TTNS 树收缩是否与显式构造完整系数张量后的收缩一致？
2. 当两个下游输出共享父节点时，pair CDF 中的共享父局部投影是否与直接三维积分一致？
3. 当前实现是否满足输出交换对称性、CDF 取值范围、单调性和边缘一致性？
4. 增大一维积分网格 `q_grid` 时，结果是否趋近于高分辨率参考值？
5. 非零 node delay 的外层求积是否随 `n_d` 增大而收敛？
6. 三个下游输出的完整联合 CDF 是否与显式系数张量公式一致？
7. log-skew-normal edge/node delay 下是否仍出现确定性数值收敛？
8. 平方 TTNS 的 doubled contraction 是否与显式平方系数张量一致？

机器可读结果为 `simple_ttns_l2/reports/maxplus_cdf_theorem_validation_metrics.json`，复现入口为 `simple_ttns_l2/experiments/validate_maxplus_cdf_theorem.py`。
最终 JSON 的 `status` 为 `pass`，22 项预设检查全部通过。

## 2. 数据与配置

实验构造一个三维上游随机向量

$$
X=(X_1,X_2,X_3).
$$

其密度由非负、归一化的 rank-2 TTNS 表示。TTNS 使用树

$$
\operatorname{parent}=(0,0,0),
$$

每个坐标使用 4 个一次 B-spline 基函数。密度由两个非负 rank-1 分量按权重 0.4 和 0.6 混合构造，因此不需要拟合步骤，并且根据精确基积分归一化。

两个下游输出的父节点集合分别为

$$
P_1=\{0,1\},
\qquad
P_2=\{1,2\}.
$$

所以坐标 1 是共享父节点。edge delay 使用

$$
E_{ka}\sim U[0,0.3],
$$

node delay 固定为 0。输出网格为 `[-1,2]` 上的 31 个等距点。

局部积分使用

$$
q_{\mathrm{grid}}\in\{41,81,161,321\},
$$

并以 `q_grid=2001` 作为本次数值收敛检查的高分辨率参考。该参考仍是数值结果，不是精确符号真值。

完整配置来自 `maxplus_cdf_theorem_validation_metrics.json` 的 `configuration` 字段。

## 3. 恒等式和合法性检查

以下数值取自 `q_grid=81` 的验证结果。

| 检查项 | 最大绝对误差或范围 | 判定 |
|---|---:|---|
| TTNS 密度归一化误差 | 0 | 通过 |
| TTNS marginal 与显式系数张量 | $3.33\times10^{-16}$ | 通过 |
| TTNS pair CDF 与显式系数张量 | $5.55\times10^{-16}$ | 通过 |
| pair CDF 与选定点直接三维网格积分 | $4.44\times10^{-16}$ | 通过 |
| 交换两个输出后的转置一致性 | 0 | 通过 |
| pair CDF 在第二坐标上取上边界后的 marginal 一致性 | $3.33\times10^{-16}$ | 通过 |
| marginal CDF 范围 | $[0,1-2.22\times10^{-16}]$ | 通过 |
| pair CDF 范围 | $[0,1-4.44\times10^{-16}]$ | 通过 |
| marginal 和 pair CDF 单调性最大违反量 | 0 | 通过 |

这里的“直接三维网格积分”是在相同上游网格和相同矩形求积规则下，显式构造

$$
p_X(x_1,x_2,x_3)
$$

后，对

$$
p_X(x)
F_{01}(s-x_1)
F_{11}(s-x_2)
F_{12}(t-x_2)
F_{22}(t-x_3)
$$

求和积分。它没有调用 TTNS contraction，因此可以检查共享父节点局部乘积和树收缩的实现是否接线正确。

## 4. `q_grid` 收敛结果

下表给出相对于 `q_grid=2001` 参考结果的最大绝对误差。数值全部来自 `maxplus_cdf_theorem_validation_metrics.json`。

| `q_grid` | marginal CDF 最大误差 | pair CDF 最大误差 |
|---:|---:|---:|
| 41 | $1.23984\times10^{-3}$ | $2.08443\times10^{-3}$ |
| 81 | $3.20458\times10^{-4}$ | $5.29753\times10^{-4}$ |
| 161 | $7.82616\times10^{-5}$ | $1.30577\times10^{-4}$ |
| 321 | $1.96576\times10^{-5}$ | $3.24116\times10^{-5}$ |

在该固定构造上，每次将 `q_grid` 约增加一倍，误差约缩小到前一次的四分之一。这个现象与本例中平滑分段多项式和所用网格规则有关。当前结果只支持“误差随 `q_grid` 增大而稳定下降”，尚不能据此宣称一般情形具有二阶收敛率。

运行时间字段也保存在 JSON 中，但受到 JAX 首次编译和缓存影响，且本实验没有独立预热或重复计时，因此不应把这些时间用于论文复杂度或速度比较。

## 5. 可以支持的论断

本实验可以支持以下有限论断。

1. 在该三维非负 rank-2 TTNS 构造上，当前 marginal 和 pair CDF 实现与定理 1 的显式系数张量公式在浮点精度内一致。
2. 对具有一个共享父节点的两个下游输出，共享父局部 CDF 乘积的实现与直接三维积分在相同离散规则下相符。
3. 在该配置上，当前结果满足 CDF 范围、坐标单调性、输出交换对称性和边缘一致性。
4. 在该配置上，局部积分误差随 `q_grid` 增大而稳定下降。
5. 对非对齐的 uniform node delay 区间 `[0.013,0.287]`，pair CDF 相对于
   `n_d=2048` 数值参考的最大绝对误差从 `n_d=2` 时的
   $5.75413\times10^{-3}$ 降至 `n_d=16` 时的 $4.32612\times10^{-6}$。
6. 在 $K=3$、每个上游变量同时被两个输出复用的构造上，`block_joint_cdf`
   与显式系数张量公式的最大绝对差为 $3.33\times10^{-16}$，三个坐标轴上的
   单调性违反量均为 0。

### 5.1 Node-delay quadrature 的结果边界

非零 node delay 验证固定使用 `q_grid=321` 和 31 点输出网格，并改变

$$
n_d\in\{2,4,8,16,32,64\}.
$$

pair CDF 相对于 `n_d=2048` 数值参考的最大绝对误差依次为

$$
5.75413\times10^{-3},
7.84252\times10^{-4},
5.68314\times10^{-4},
4.32612\times10^{-6},
4.32612\times10^{-6},
4.32612\times10^{-6}.
$$

`n_d=16` 之后的误差平台说明，在当前固定输出网格上，继续增加 node-delay
求积点已不再降低总差异；剩余误差需要与输出网格插值误差联合分析。因此这里只
声称该配置出现稳定收敛和离散误差平台，不声称一般收敛阶。

### 5.2 Log-skew-normal delay

重尾验证使用

$$
\exp(Z),
\qquad
Z\sim\operatorname{SkewNormal}(\xi=-2.12,\omega=0.45,\alpha=4),
$$

分别作为 edge delay 和 node delay。该实验不使用 Monte Carlo。

固定 `n_d=512`，并以 `q_grid=2001` 为数值参考时，pair CDF 最大绝对误差为：

| `q_grid` | pair CDF 最大误差 |
|---:|---:|
| 81 | $1.87035\times10^{-4}$ |
| 161 | $4.65685\times10^{-5}$ |
| 321 | $1.14172\times10^{-5}$ |
| 641 | $2.62933\times10^{-6}$ |

固定 `q_grid=641`，并以 `n_d=2048` 为数值参考时，pair CDF 最大绝对误差为：

| `n_d` | pair CDF 最大误差 |
|---:|---:|
| 8 | $5.19866\times10^{-3}$ |
| 16 | $2.77220\times10^{-3}$ |
| 32 | $1.46063\times10^{-3}$ |
| 64 | $8.31668\times10^{-4}$ |
| 128 | $3.76742\times10^{-4}$ |

两组结果都支持在该固定人工 TTNS 和给定 log-skew-normal 参数下，确定性数值误差
随分辨率增加而下降。它们仍不是一般收敛阶证明。

### 5.3 平方 TTNS

平方验证把同一个 rank-2 TTNS 解释为 amplitude $\psi$，并定义

$$
p_X(x)=\frac{\psi(x)^2}{Z}.
$$

归一化常数的 doubled contraction 结果为

$$
Z_{\mathrm{TTNS}}=0.37982440545349283,
$$

显式系数张量结果为

$$
Z_{\mathrm{dense}}=0.3798244054534935.
$$

两者绝对差为 $6.66\times10^{-16}$。在 `q_grid=81` 下：

| 检查项 | 最大绝对误差 |
|---|---:|
| doubled marginal 与显式平方张量 | $2.22\times10^{-15}$ |
| doubled pair CDF 与显式平方张量 | $3.44\times10^{-15}$ |
| pair CDF 与选定点直接三维积分 | $7.77\times10^{-16}$ |
| 交换两个输出后的转置一致性 | 0 |
| pair 上边界与 marginal 一致性 | $4.65170\times10^{-4}$ |

marginal 和 pair CDF 的范围均为 $[0,1]$，单调性最大违反量为 0。边缘一致性误差
不是 doubled contraction 误差，而主要来自有限 `q_grid` 及 pair 网格上的数值
积分。该结果验证了推论 4 的当前低维实现路径，但没有实现或验证平方 TTNS 的
逐层材料化和全链训练。

## 6. 尚不能支持的论断

本实验不能支持以下结论。

1. 它不能证明定理在数学上成立；数学证明仍以 `maxplus_ttns_cdf_theorem_zh.md` 为准，本实验只检查实现。
2. 它不能证明一般的数值收敛阶，因为只使用了一种基函数和一个人工 TTNS；
   uniform 与 log-skew-normal 两类结果都只是固定配置验证。
3. 它没有覆盖 lognormal、gamma 或其他重尾 delay。
4. $K=3$ 验证只使用 9 点等距网格、uniform edge delay 和零 node delay，
   不能证明任意 $K$ 下完整联合表具有可接受的计算成本。
5. 平方 TTNS 只验证了低维 doubled contraction，没有完成平方模型的分层全链。
6. 它没有进行统计实验，不能产生多 seed、显著性或泛化结论。
7. 它不能证明相关定理或算法此前无人提出。

## 7. 图表素材

- `maxplus_cdf_theorem_validation_convergence.png`：uniform 和
  log-skew-normal delay 下的 `q_grid`、`n_d` 经验离散误差。
- `maxplus_cdf_shared_parent_contraction.png`：共享父节点的一维局部投影和
  TTNS 收缩示意图。
- `maxplus_cdf_pair_validation_heatmaps.png`：pair CDF 高分辨率数值参考、
  `q_grid=81` 结果和 $\log_{10}$ 绝对误差。

三张图均由 `simple_ttns_l2/experiments/plot_maxplus_cdf_theorem_validation.py`
只读取最终 JSON 生成。

## 8. 后续边界

当前核心连续定理的线性 TTNS、共享父 pair、$K=3$ joint、非零 node delay、
log-skew-normal delay 和平方 doubled contraction 路径均已有小维实现证据。
下一阶段不应再增加相似的人工正确性例子，而应完成复杂度实测、尾部截断控制，
以及冻结配置后的 Dense/UCI 多 seed。

## 9. 复现

在仓库根目录运行：

```bash
python3 simple_ttns_l2/experiments/validate_maxplus_cdf_theorem.py
```

脚本会重写 `simple_ttns_l2/reports/maxplus_cdf_theorem_validation_metrics.json`。只有全部预设检查通过时，脚本才以成功状态退出。

验证通过后生成图像：

```bash
python3 simple_ttns_l2/experiments/plot_maxplus_cdf_theorem_validation.py
```
