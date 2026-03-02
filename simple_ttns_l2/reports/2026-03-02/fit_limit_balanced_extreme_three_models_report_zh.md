# Balanced Tree 高复杂度三模型对比报告

## 1. 实验目标

在同一个 `balanced target` 上，比较 `balanced TTNS / chain TTNS / pure TT` 的拟合上限。

- 配置: `{"n_dims": 6, "q": 2, "m": 96, "rank": 24, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 600, "log_every": 20, "seed": 20260227, "n_train": 10000, "n_val": 5000, "n_test": 5000, "monitor_train_sz": 4000, "monitor_val_sz": 4000, "early_stop_patience_logs": 0, "early_stop_min_delta": 1e-06, "early_stop_warmup_logs": 0, "early_stop_restore_best": true}`
- 切片图: `fit_limit_balanced_extreme_three_models.svg`

## 2. 训练结果

| model | lr_schedule | final_lr | final_val_l2 | best_val_l2 | stop_step | best_step | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced_ttns | balanced_extreme_hold300_0.05x | 5.000e-05 | -0.065375 | -0.063578 | 600 | 260 | 1.000000 | 44.775 |
| chain_ttns | balanced_extreme_hold300_0.05x | 5.000e-05 | -0.047415 | -0.046098 | 600 | 520 | 1.000000 | 10.101 |
| tt | tt_delayed_cosine_hold240_0.1x | 1.000e-04 | -0.046571 | -0.045520 | 600 | 380 | 1.000000 | 21.795 |

## 3. 切片指标

| pair | noise_floor | IAE_balanced_ttns | IAE_chain_ttns | IAE_tt | ratio_balanced_ttns | ratio_chain_ttns | ratio_tt |
|---:|---:|---:|---:|---:|---:|---:|---:|
| (0,1) | 0.081382 | 0.275066 | 0.382658 | 0.390891 | 3.380 | 4.702 | 4.803 |
| (0,3) | 0.087077 | 0.224237 | 0.310555 | 0.310462 | 2.575 | 3.566 | 3.565 |
| (2,5) | 0.071140 | 0.222308 | 0.490852 | 0.502931 | 3.125 | 6.900 | 7.070 |

## 4. 聚合结论

- mean_IAE: balanced_ttns=`0.240537`, chain_ttns=`0.394688`, tt=`0.401428`.
- 最优模型: `balanced_ttns`。
