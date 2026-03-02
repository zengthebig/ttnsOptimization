# 原始 TTNSDE 训练方式 vs simple_ttns_l2

## 1. 实验设置

- 目标分布: `6 维 complex balanced target`。
- 模型拓扑: 两边都用 `balanced TTNS`。
- 原始模型: `TTNSDE/ttde` 的 `PAsTTNSSqrOpt + MLE(NLL)`。
- 当前模型: `simple_ttns_l2` 的 `TTNS + L2 objective`。
- 配置: `{"n_dims": 6, "q": 2, "m": 24, "rank": 3, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 100, "log_every": 5, "seed": 20260302, "n_train": 5000, "n_val": 2500, "n_test": 5000, "monitor_train_sz": 2000, "monitor_val_sz": 2000, "early_stop_patience_logs": 0, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 0, "early_stop_restore_best": false}`
- 切片图: `compare_original_ttns_vs_l2_complex_balanced.svg`

## 2. 训练指标

| model | objective | native_metric | val_native | finite_val_ll | finite_test_ll | val_nonpositive | test_nonpositive | total_time_sec |
|---|---|---|---:|---:|---:|---:|---:|---:|
| original_ttns | MLE / NLL | val_nll | 6.517429 | -6.517429 | -6.484729 | 0.0000 | 0.0000 | 8.993 |
| simple_ttns_l2 | L2 surrogate | val_l2 | -0.018596 | nan | nan | nan | nan | 1.675 |

说明: 两个 native metric 不同，不能直接横向比较。真正可比的是下面的共同外部指标: 2D marginal IAE。

## 3. 共同外部指标: 2D 切片 IAE

| pair | IAE_original_mle | IAE_simple_l2 | better |
|---:|---:|---:|---|
| (0,1) | 0.814578 | 0.785346 | simple_ttns_l2 |
| (0,3) | 0.752210 | 0.696240 | simple_ttns_l2 |
| (2,5) | 0.925246 | 0.796428 | simple_ttns_l2 |

## 4. 聚合结论

- mean_IAE(original_ttns) = `0.830678`
- mean_IAE(simple_ttns_l2) = `0.759338`
- 这次在共同切片指标上，`simple TTNS(L2)` 更好。
- 时间对比: original=`8.993s`, simple_l2=`1.675s`。

## 5. 备注

- 原始模型是平方密度参数化，因此切片是通过二次型 marginal 计算出来的。
- 当前 simple 模型是直接线性密度参数化，因此切片是线性 marginal。
- 两者不是同一参数化族，所以这次实验比较的是“原始训练方式 vs 当前模型设定”的整体效果。
