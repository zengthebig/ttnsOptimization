# Complex Balanced Target 极限拟合对比报告

## 1. 实验目标

在同一个 `complex balanced target` 上，比较 4 个模型在高参数量配置下能拟合到多好。

- 配置: `{"n_dims": 6, "q": 2, "m": 96, "rank": 24, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 600, "log_every": 20, "seed": 20260227, "n_train": 10000, "n_val": 5000, "n_test": 5000, "monitor_train_sz": 4000, "monitor_val_sz": 4000, "early_stop_patience_logs": 0, "early_stop_min_delta": 1e-06, "early_stop_warmup_logs": 0, "early_stop_restore_best": true}`
- 切片图: `fit_limit_balanced_extreme_four_models.svg`

## 2. 训练结果

| model | objective | native_metric | val_native | aux_metric | total_time_sec |
|---|---|---|---:|---:|---:|
| original_ttns | MLE / NLL | val_nll | 3.873061 | finite_val_ll=-3.873061 | 225.274 |
| balanced_ttns | L2 | val_l2 | -0.065375 | best_val_l2=-0.063578 | 46.546 |
| chain_ttns | L2 | val_l2 | -0.047415 | best_val_l2=-0.046098 | 10.754 |
| tt | L2 | val_l2 | -0.046571 | best_val_l2=-0.045520 | 21.994 |

说明: `original_ttns` 的原生目标是 NLL，其余三个模型的原生目标是 L2，因此真正可比的是下面的统一切片指标。

## 3. 切片指标

| pair | noise_floor | IAE_original | IAE_balanced_ttns | IAE_chain_ttns | IAE_tt | ratio_original | ratio_balanced_ttns | ratio_chain_ttns | ratio_tt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (0,1) | 0.080325 | 0.179868 | 0.270794 | 0.379875 | 0.392252 | 2.239 | 3.371 | 4.729 | 4.883 |
| (0,3) | 0.080640 | 0.140003 | 0.216873 | 0.314420 | 0.312929 | 1.736 | 2.689 | 3.899 | 3.881 |
| (2,5) | 0.070031 | 0.148873 | 0.220423 | 0.492552 | 0.505860 | 2.126 | 3.148 | 7.033 | 7.223 |

## 4. 聚合结论

- mean_IAE: original_ttns=`0.156248`, balanced_ttns=`0.236030`, chain_ttns=`0.395616`, tt=`0.403680`。
- 最优模型: `original_ttns`。
