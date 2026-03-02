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
| balanced | balanced_ttns | delayed_cosine_hold120_0.1x | -0.018031 | -0.018766 | true | 240 | 180 | 1.000000 | 135.017 |
| balanced | chain_ttns | delayed_cosine_hold120_0.1x | -0.018031 | -0.018766 | true | 240 | 180 | 1.000000 | 103.466 |
| balanced | tt | tt_delayed_cosine_hold120_0.1x | -0.018031 | -0.018766 | true | 240 | 180 | 1.000000 | 7.519 |
| chain | balanced_ttns | delayed_cosine_hold120_0.1x | -0.000004 | -0.000004 | true | 100 | 10 | 1.000000 | 45.551 |
| chain | chain_ttns | delayed_cosine_hold120_0.1x | -0.000004 | -0.000004 | true | 100 | 10 | 1.000000 | 41.055 |
| chain | tt | tt_delayed_cosine_hold120_0.1x | -0.000004 | -0.000004 | true | 100 | 10 | 1.000000 | 3.977 |

## 4. 切片误差明细（2D）

| target | pair | noise_floor | IAE_balanced_ttns | IAE_chain_ttns | IAE_tt | ratio_balanced_ttns | ratio_chain_ttns | ratio_tt | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.073473 | 0.906705 | 0.906705 | 0.906705 | 12.341 | 12.341 | 12.341 | chain_ttns |
| balanced | (0,3) | 0.075725 | 0.792738 | 0.792738 | 0.792738 | 10.469 | 10.469 | 10.469 | balanced_ttns |
| balanced | (2,5) | 0.064394 | 0.893823 | 0.893823 | 0.893823 | 13.881 | 13.881 | 13.881 | chain_ttns |
| chain | (0,1) | 0.070605 | 1.042259 | 1.042259 | 1.042259 | 14.762 | 14.762 | 14.762 | balanced_ttns |
| chain | (0,3) | 0.070971 | 0.683333 | 0.683333 | 0.683333 | 9.628 | 9.628 | 9.628 | balanced_ttns |
| chain | (2,5) | 0.037404 | 1.025844 | 1.025844 | 1.025844 | 27.426 | 27.426 | 27.426 | balanced_ttns |

## 5. 聚合统计（按 target 汇总）

| target | model | mean_IAE | mean_ratio_to_floor | mean_L2 |
|---|---:|---:|---:|---:|
| balanced | balanced_ttns | 0.864422 | 12.230 | 0.343450 |
| balanced | chain_ttns | 0.864422 | 12.230 | 0.343450 |
| balanced | tt | 0.864422 | 12.230 | 0.343450 |
| chain | balanced_ttns | 0.917146 | 17.272 | 0.338804 |
| chain | chain_ttns | 0.917146 | 17.272 | 0.338804 |
| chain | tt | 0.917146 | 17.272 | 0.338804 |

## 6. 结论（中文总结）

- target=`balanced`: 最优是 `chain_ttns`，其 mean_IAE=0.864422，相对第二名 `balanced_ttns` 提升约 0.00% 。
- target=`chain`: 最优是 `balanced_ttns`，其 mean_IAE=0.917146，相对第二名 `chain_ttns` 提升约 0.00% 。

- 由于 `init_noise=0`，本实验重点反映模型结构与训练动力学差异，而不是初始化噪声导致的轨迹分叉。
