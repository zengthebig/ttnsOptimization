# Chain TTNS vs Pure TT: Diagnosis Note (2026-02-26)

## Question
为什么 `chain TTNS` 和 `pure TT` 在 chain target 上的拟合结果会不一致？是否是实现 bug？

## Diagnosis Summary
结论：**不是核心算子/模型实现 bug**。不一致主要来自**初始化噪声注入方式在两种参数化下不等价**，导致在非凸目标上进入不同优化轨迹。

## Evidence

### 1) 表达与算子一致性检查（同一参数直接对照）
将同一个 `TTOpt` 映射到 `chain TTNS` 后，对照计算：
- 点值 `q(x)`
- 积分 `\int q`
- 二次型 `\int q^2`

结果在数值误差范围内一致（小规模 brute-force 也验证通过）。

### 2) 严格对齐训练（同数据、同超参、同 batch 序列）
配置：`n_dims=6, q=2, m=64, rank=16, batch=256, lr=1e-3, steps=120`

#### Case A: `init_noise=0`, `train_noise=0`
`chain TTNS` 与 `pure TT` 的 `val_l2` 日志逐点一致，最终值一致：
- `final_val_l2(chain_ttns) = -2.834431586871091e-05`
- `final_val_l2(pure_tt)    = -2.834431586871086e-05`

#### Case B: `init_noise=0`, `train_noise=1e-2`
两者仍逐点一致，最终值一致：
- `final_val_l2(chain_ttns) = -9.31052228465995e-05`
- `final_val_l2(pure_tt)    = -9.310522284659954e-05`

### 3) 默认噪声配置下会分叉
配置：`init_noise=1e-2`, `train_noise=1e-2`, `steps=300`（同数据同超参）
- `final_val_l2(chain_ttns) = 2.4249001464785613e-05`
- `final_val_l2(pure_tt)    = 1.3392389617330242e-05`

2D slice IAE（chain target）：
- pair `(0,1)`: `chain_ttns=4.293953`, `pure_tt=1.231067`
- pair `(0,3)`: `chain_ttns=2.836543`, `pure_tt=1.922629`
- pair `(2,5)`: `chain_ttns=1.001150`, `pure_tt=1.002008`

这说明默认初始化噪声下，优化轨迹确实会明显分叉。

## Root Cause
`init_noise` 目前是对“核心参数”直接加噪：
- `chain TTNS`: 逐 core（6 个 core）分别加噪
- `pure TT`: `first/inner/last` 3 组加噪

虽然两者表达能力等价，但这种噪声不是同一个“函数空间扰动”，在非凸优化里会放大成不同解。

## Is a Fix Necessary?
- **核心模型实现**：不需要修复（算子和训练公式在对齐条件下一致）。
- **对比实验协议**：建议修复，否则容易得到误导性结论。

## Recommended Fix (Experiment Fairness)
1. 对比不同参数化时，默认使用 `init_noise=0` 做主结论。
2. 如需随机初始化，改为“共享函数空间噪声”（例如先对 rank-1 向量加噪，再映射到 TT/TTNS），不要直接对 core 独立加噪。
3. 报告中至少给出 `>=3` 个 seed 的均值/方差，避免单次轨迹结论。
