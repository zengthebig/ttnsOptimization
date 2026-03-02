# Balanced Tree 极限拟合报告（高复杂度）

## 实验目标

仅针对 `balanced target`，提升模型复杂度并评估 balanced TTNS 的拟合上限。

- Config: `{"n_dims": 6, "q": 2, "m": 96, "rank": 24, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 600, "log_every": 20, "seed": 20260227, "n_train": 10000, "n_val": 5000, "n_test": 5000, "monitor_train_sz": 4000, "monitor_val_sz": 4000, "early_stop_patience_logs": 0, "early_stop_min_delta": 1e-06, "early_stop_warmup_logs": 0, "early_stop_restore_best": true}`
- Figure: `fit_limit_balanced_extreme.svg`

## 训练摘要

| model | lr_schedule | final_lr | final_val_l2 | best_val_l2 | stop_step | best_step | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | balanced_extreme_hold300_0.05x | 5.000e-05 | -0.063324 | -0.066168 | 600 | 240 | 1.000000 | 626.533 |

## 切片拟合摘要

| pair | IAE(model,target) | IAE(target,target2) | ratio_to_floor | L2 |
|---:|---:|---:|---:|---:|
| (0,1) | 0.270545 | 0.081939 | 3.302 | 0.090220 |
| (0,3) | 0.214790 | 0.083161 | 2.583 | 0.068484 |
| (2,5) | 0.231293 | 0.073693 | 3.139 | 0.095888 |

- Mean IAE = `0.238876`, Mean ratio_to_floor = `3.008`.
