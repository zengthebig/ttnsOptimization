# 原始 TTNSDE 训练方式 vs simple_ttns_l2

## 1. 实验设置

- 目标分布: `6 维 balanced tree Gaussian`。
- 模型拓扑: 两边都用 `balanced TTNS`。
- 原始模型: `TTNSDE/ttde` 的 `PAsTTNSSqrOpt + MLE(NLL)`。
- 当前模型: `simple_ttns_l2` 的 `TTNS + L2 objective`。
- 配置: `{"n_dims": 6, "q": 2, "m": 16, "rank": 2, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 100, "log_every": 5, "seed": 20260302, "n_train": 4000, "n_val": 2000, "n_test": 4000, "monitor_train_sz": 2000, "monitor_val_sz": 2000, "early_stop_patience_logs": 0, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 0, "early_stop_restore_best": false}`
- 切片图: `compare_original_ttns_vs_l2_simple.svg`

## 2. 训练指标

| model | objective | native_metric | val_native | finite_val_ll | finite_test_ll | val_nonpositive | test_nonpositive | total_time_sec |
|---|---|---|---:|---:|---:|---:|---:|---:|
| original_ttns | MLE / NLL | val_nll | 7.923234 | -7.923234 | -7.807370 | 0.0000 | 0.0002 | 7.901 |
| simple_ttns_l2 | L2 surrogate | val_l2 | -0.002308 | nan | nan | nan | nan | 1.526 |

说明: 两个 native metric 不同，不能直接横向比较。真正可比的是下面的共同外部指标: 2D marginal IAE。

## 3. 共同外部指标: 2D 切片 IAE

| pair | IAE_original_mle | IAE_simple_l2 | better |
|---:|---:|---:|---|
| (0,1) | 0.768685 | 0.700432 | simple_ttns_l2 |
| (0,2) | 0.764896 | 0.627428 | simple_ttns_l2 |
| (2,5) | 0.540749 | 0.627438 | original_ttns |

## 4. 聚合结论

- mean_IAE(original_ttns) = `0.691443`
- mean_IAE(simple_ttns_l2) = `0.651766`
- 这次在共同切片指标上，`simple TTNS(L2)` 更好。
- 时间对比: original=`7.901s`, simple_l2=`1.526s`。

## 5. 备注

- 原始模型是平方密度参数化，因此切片是通过二次型 marginal 计算出来的。
- 当前 simple 模型是直接线性密度参数化，因此切片是线性 marginal。
- 两者不是同一参数化族，所以这次实验比较的是“原始训练方式 vs 当前模型设定”的整体效果。
