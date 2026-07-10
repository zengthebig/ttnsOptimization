# R5 解析 target 与拟合质量诊断

本诊断用于回答：R5 失效到底是 `UpperForest.pair_cdf` / 解析传播 target 错了，还是解析 L2 拟合没有学进正确 target。

## 结论

当前证据支持：**解析 CDF 传播本身没有明显偏差；主要失效点在解析非负 L2 拟合阶段**。

- L1 中，真实 L1 最强相关约为 `0.685-0.698`；从 L0 forest 采样再 max-plus 传播得到 `0.685-0.705`；`UpperForest.pair_cdf` + Hoeffding 得到 `0.687-0.692`。三者一致，说明 L1 解析 target 是对的。
- 但同一个 L1 target 经 `_fit_analytic_ttns_nonneg` 拟合后，从拟合 TTNS 采样得到的相同相关只有 `0.020-0.049`（正式 `an_steps=5000`），几乎退化为独立。
- 提高 `init_noise` 到 `0.05` 的 L1 诊断仍只有 `-0.017-0.032`，因此不是简单初始化扰动不足。
- 逐层 audit 显示从 L2 开始，`sampled_prop` 与 `analytic` 仍基本一致，但二者都已低于 truth。这是因为上一层拟合后的 forest 已经丢失相关，而不是本层 `UpperForest.pair_cdf` 新引入了系统偏差。

## 关键数值

产物路径：

- `artifacts/target_audit_fit_quality.json`
- `artifacts/target_audit_l1_steps5000.json`
- `artifacts/target_audit_l1_noise005.json`
- `artifacts/target_audit_layerwise.json`

L1 正式步数检查（`an_steps=5000`, `rank=8`, `n_sample=5000`）：

| block | truth r | sampled_prop r | analytic r | fitted TTNS sample r |
|---:|---:|---:|---:|---:|
| 0 | +0.687 | +0.705 | +0.689 | +0.020 |
| 1 | +0.685 | +0.685 | +0.688 | +0.049 |
| 2 | +0.687 | +0.695 | +0.692 | +0.041 |
| 3 | +0.698 | +0.688 | +0.687 | +0.022 |

L4 逐层链尾检查（`an_steps=800` 的诊断链）：

| block | truth r | sampled_prop r | analytic r |
|---:|---:|---:|---:|
| 0 | +0.741 | +0.350 | +0.348 |
| 1 | +0.690 | +0.343 | +0.338 |
| 2 | +0.606 | +0.397 | +0.342 |
| 3 | +0.554 | +0.356 | +0.335 |

这里 `sampled_prop` 与 `analytic` 接近，说明 pair CDF 与同一上层 forest 的采样传播一致；它们共同低于 truth，是上层拟合后的 forest 已经丢失相关导致的。

## 代码层面观察

- `analytic_block_target(..., use_mi=True)` 中返回的 `layer.corr` 实际是树边权矩阵 `W`，默认存放 MI 权重，不是 Pearson 相关矩阵。因此早先 `nonneg_corr` 变体里的 `_corr_penalty_from_moments(..., target_corr=layer.corr)` 并没有惩罚到真实 Pearson 相关，001/002 不能作为“相关矩惩罚无效”的强证据。
- 007 `hybrid_sample_prop` 成功的关键不是 TTNS 树结构不同，而是训练口径不同：它用非负 MLE 在采样传播数据上拟合，能够保住层内相关。

## 下一步建议

若继续保留“解析 UpperForest 传播”原则，优先修正的是拟合器，而不是 target 构造：

1. 给解析非负 L2 加真正的 Pearson 相关惩罚，目标相关应从 `UpperForest.pair_cdf` 的 Hoeffding 协方差单独保存，而不是复用 MI 权重。
2. 或者改用解析 target 生成的树 CDF 条件采样样本，再用现有非负 MLE 拟合 TTNS。这样仍保留 UpperForest 解析传播生成 target，但绕开当前解析 L2 cross-term/平方参数化组合的相关塌缩。
3. 007 已满足项目指标，可作为当前 R5-fix 的成功收敛方案；若要求“全解析 L2 无采样训练”也达标，则需要新一轮针对 `_fit_analytic_ttns_nonneg` 的拟合器修复。
