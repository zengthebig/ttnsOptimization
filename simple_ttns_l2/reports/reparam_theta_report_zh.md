# core 参数重参数化 (θ→θ² / exp θ) 报告

**日期**：2026-07-03　**分支**：worktree-theta-reparam
**改动**：`simple_ttns_l2/reparam.py`（新）、`objective.py`（threaded `transform` kwarg）、`experiments/reparam_check.py`（新）

## 目标
在**线性 L2 TTNS**（R8 系）里，把 `ttns.cores` 每个存储参数 θ 在收缩前过一个逐元素变换 g：
`identity`（θ）/ `square`（θ²）/ `exp`（exp θ）。因 B-spline 基非负，θ²/exp 会强制**有效核非负 → q(x)≥0**，是 `p=ψ²/Z` 平方参数化的轻量近亲（仍保留 L2 目标与多线性收缩，只约束核符号）。

归一化需随变换调整（scale 作用在 g(θ) 输出上）：identity `θ·s`；square `θ·√s`；exp `θ+log s`。

## 正确性（全 PASS）
- `effective_ttns` 核 == g(θ) 逐元素（err ≤1e-12）。
- 变换感知归一化后 ∫q=1.0（identity/square/exp 均 1.0000000000）。
- `identity` 路径 bit-for-bit 不变：原 6 个单测（objective + smoke）全过。

## 小规模效果（4D 双峰合成，3 seed，val_l2 越低越好）
| transform | mean val_l2 | std |
|---|---|---|
| **identity（基线）** | **−0.19728** | 0.00114 |
| square (θ²) | −0.19346 | 0.00770 |
| exp (exp θ) | −0.12648 | 0.00741 |

**结论：无提升，反而变差。** square 略逊基线（−0.19346 vs −0.19728）且方差大 6×；exp 明显更差（−0.126）。

## 解释
- 线性 L2 TTNS 靠核的**正负相消**表达复杂/多峰密度；强制核非负削弱表达力（这正是平方模型改用 MLE 而非线性 L2 的原因——非负性收益要靠 `p=ψ²` 的目标切换兑现，而非在 L2 目标里硬加符号约束）。
- exp 额外带来病态：init 时 ∫q≈3.6e3、梯度尺度爆炸，优化更难。
- 即：**在既有线性 L2 框架内做核重参数化不是改进杠杆**；非负性要走文档已列的 `p=ψ²/Z` MLE 路线（§2.2「平方 TTNS 块」）才可能兑现收益。

## 复现
```bash
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.reparam_check
```
