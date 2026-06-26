# InitNoise=0 三模型对比实验报告

## 1. 实验目的

在完全固定 `init_noise=0` 的条件下，对比三类模型在两类目标分布（balanced / chain）上的拟合能力：`balanced TTNS`、`chain TTNS`、`pure TT`。

## 2. 实验设置

- 配置: `{"n_dims": 6, "q": 2, "m": 64, "rank": 16, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.0, "train_steps": 300, "log_every": 10, "seed": 2602, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- 切片图: `fit_three_models_init0.svg`
- 指标: `L2`、2D 切片 `IAE`、`ratio_to_floor = IAE / noise_floor`

## 3. 训练结果对比

| target | model | lr_schedule | final_val_l2 | best_val_l2 | stopped_early | stop_step | best_step | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | balanced_ttns | delayed_cosine_hold120_0.1x | -0.019462 | -0.017958 | true | 230 | 170 | 1.000000 | 5.331 |
| balanced | chain_ttns | delayed_cosine_hold120_0.1x | -0.019462 | -0.017958 | true | 230 | 170 | 1.000000 | 2.539 |
| balanced | tt | tt_delayed_cosine_hold120_0.1x | -0.019462 | -0.017958 | true | 230 | 170 | 1.000000 | 3.268 |
| chain | balanced_ttns | delayed_cosine_hold120_0.1x | -0.000029 | -0.000029 | true | 100 | 10 | 1.000000 | 2.358 |
| chain | chain_ttns | delayed_cosine_hold120_0.1x | -0.000029 | -0.000029 | true | 100 | 10 | 1.000000 | 1.279 |
| chain | tt | tt_delayed_cosine_hold120_0.1x | -0.000029 | -0.000029 | true | 100 | 10 | 1.000000 | 1.800 |

## 4. 切片误差明细（2D）

| target | pair | noise_floor | IAE_balanced_ttns | IAE_chain_ttns | IAE_tt | ratio_balanced_ttns | ratio_chain_ttns | ratio_tt | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.074067 | 0.901692 | 0.901692 | 0.901692 | 12.174 | 12.174 | 12.174 | tt |
| balanced | (0,3) | 0.073391 | 0.796424 | 0.796424 | 0.796424 | 10.852 | 10.852 | 10.852 | chain_ttns |
| balanced | (2,5) | 0.065156 | 0.884286 | 0.884286 | 0.884286 | 13.572 | 13.572 | 13.572 | tt |
| chain | (0,1) | 0.077832 | 1.044078 | 1.044078 | 1.044078 | 13.414 | 13.414 | 13.414 | balanced_ttns |
| chain | (0,3) | 0.069113 | 0.693806 | 0.693806 | 0.693806 | 10.039 | 10.039 | 10.039 | balanced_ttns |
| chain | (2,5) | 0.037737 | 1.100388 | 1.100388 | 1.100388 | 29.159 | 29.159 | 29.159 | balanced_ttns |

## 5. 聚合统计（按 target 汇总）

| target | model | mean_IAE | mean_ratio_to_floor | mean_L2 |
|---|---:|---:|---:|---:|
| balanced | balanced_ttns | 0.860801 | 12.199 | 0.340071 |
| balanced | chain_ttns | 0.860801 | 12.199 | 0.340071 |
| balanced | tt | 0.860801 | 12.199 | 0.340071 |
| chain | balanced_ttns | 0.946091 | 17.538 | 0.338444 |
| chain | chain_ttns | 0.946091 | 17.538 | 0.338444 |
| chain | tt | 0.946091 | 17.538 | 0.338444 |

## 6. 结论（中文总结）

- target=`balanced`: 最优是 `chain_ttns`，其 mean_IAE=0.860801，相对第二名 `tt` 提升约 0.00% 。
- target=`chain`: 最优是 `balanced_ttns`，其 mean_IAE=0.946091，相对第二名 `chain_ttns` 提升约 0.00% 。

- 由于 `init_noise=0`，本实验重点反映模型结构与训练动力学差异，而不是初始化噪声导致的轨迹分叉。
