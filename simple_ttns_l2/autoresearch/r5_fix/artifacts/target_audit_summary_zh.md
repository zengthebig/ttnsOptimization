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

## L2 拟合器定位更新

后续 `debug_l2_fit.py` 对 L1 block0 做了最小化诊断，产物为：

- `artifacts/l2_fit_debug_l1_block0.json`
- `artifacts/l2_fit_debug_lr_sweep_l1_block0.json`

结论更新如下：

- `_cross_term_fn` 的解析交叉项不是主 bug。对 `target_sample_mle` 模型，`cross_analytic=33.018`、`cross_mc=33.016`，二者几乎一致；该模型的解析 L2 loss 为 `-35.94`，显著优于当前低学习率解析 L2 模型的 `-11.46`。
- 线性解析 L2（允许负密度）在 L1 block0 能学到相关：`sample r≈+0.684`，说明树消息 L2 目标可以提供相关信号。
- 非负解析 L2 的失败主要来自优化配置：默认 `an_lr=3e-5` 时 L1 block0 只有 `r≈+0.025`；提高到 `lr=1e-3/2e-3` 后，1500 步即可达到 `r≈+0.641/+0.663`。
- 009 全链验证了这个定位：仅把非负解析 L2 的 `an_lr` 提高到 `1e-3`、`an_steps` 降到 `1500`，L4 达到 `joint_LL=5.89`、`nonpos=0`、`std_ratio=0.94`，最强相关 `r=+0.513` vs GT `+0.758`。

## 下一步建议

若继续保留“解析 UpperForest 传播”原则，当前推荐收敛到 009 的纯解析 L2 修复：

1. 将 dense R5-fix 的非负解析 L2 默认配置改为 `an_lr≈1e-3`、`an_steps≈1500`，并保留 008 的 `analytic_mle` 作为拟合器对照。
2. 如果后续跨 seed 或更大图不稳定，再考虑为 `_fit_analytic_ttns_nonneg` 加梯度裁剪或 warmup；目前证据不支持修改 `UpperForest.pair_cdf` 或 `analytic_block_target`。
3. 相关惩罚若继续保留，需要先修正目标口径：不能复用 `layer.corr` 的 MI 权重，应单独保存 Pearson 相关矩阵。
