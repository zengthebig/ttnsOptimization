# Stable / Fast TTNS 对齐实验报告

## 1. 实验目的

验证 TTNS 的快速收缩实现是否改变训练结果，并量化训练时间收益。

- 目标分布：`balanced target (complex)`
- 模型结构：`balanced TTNS`
- 初始化：同一个 seed，同一份训练/验证/切片样本
- 配置：`{"n_dims": 6, "q": 2, "m": 96, "rank": 24, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 600, "log_every": 5, "seed": 20260227, "n_train": 10000, "n_val": 5000, "n_test": 5000, "monitor_train_sz": 4000, "monitor_val_sz": 4000, "early_stop_patience_logs": 0, "early_stop_min_delta": 1e-06, "early_stop_warmup_logs": 0, "early_stop_restore_best": true}`

## 2. 总结论

- 总训练时间：stable=`243.362s`，fast=`235.510s`，加速比=`1.033x`。
- 最终对齐评估（统一用 stable evaluator）：stable val_l2=`-0.065564`，fast val_l2=`-0.065564`，绝对差=`5.143108e-14`。
- 训练轨迹平均差异：mean |train_l2 diff|=`7.491728e-14`，mean |val_l2 diff|=`4.332683e-13`。

## 3. 终点指标

| variant | total_time_sec | best_step | best_val_l2(native) | final_val_l2_ref | final_integral_ref |
|---|---:|---:|---:|---:|---:|
| stable | 243.362 | 245 | -0.063851 | -0.065564 | 1.000000 |
| fast | 235.510 | 245 | -0.063851 | -0.065564 | 1.000000 |

## 4. 切片对比

- 曲线图：`stable_fast_balanced_curve.svg`
- 切片图：`stable_fast_balanced_slices.svg`
- 每 5 步详细历史：`stable_fast_balanced_history.csv`

| pair | IAE stable | IAE fast | winner |
|---:|---:|---:|---:|
| (0,1) | 0.273015 | 0.273015 | stable |
| (0,3) | 0.229514 | 0.229514 | stable |
| (2,5) | 0.224596 | 0.224596 | fast |

## 5. 解释

- 这次对比中，`stable` 和 `fast` 优化的数学目标完全相同，差异只来自浮点计算顺序变化。
- 如果最终 `val_l2_ref` 和切片 IAE 只出现很小差异，就说明加速版本没有实质改变训练结果。
- 如果差异明显，则说明数值路径变化已经足够大，影响了非凸优化轨迹。
