# 随机树目标：匹配 TTNS vs Chain TTNS 对比报告

## 目的

在**随机递归树**合成目标上，对比与目标 parent 一致的 TTNS（matched）与 chain TTNS 的 2D 切片 IAE。

## 配置

- 训练配置: `{"n_dims": 6, "q": 2, "m": 64, "rank": 16, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 20260227, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- 目标 parent: `[0, 0, 0, 1, 3, 1]`
- chain parent: `[0, 0, 1, 2, 3, 4]`

## 训练结果

| model | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|
| matched | -0.089709 | 1.000000 | 6.811 |
| chain | -0.060082 | 1.000000 | 3.316 |

## 切片 IAE

| pair | noise_floor | IAE_matched | IAE_chain |
|---|---:|---:|---:|
| (0,1) | 0.065401 | 0.236491 | 0.419629 |
| (0,3) | 0.089263 | 0.180806 | 0.337378 |
| (2,5) | 0.090901 | 0.143848 | 0.355086 |

## 切片聚合 ((0,1),(0,3),(2,5))

- mean IAE matched: **0.187048**
- mean IAE chain: **0.370698**
- matched vs chain 提升: **49.5%**

## 结论

- 匹配拓扑 TTNS 优于 chain **49.5%**，达到 M1.2 标准。

## 多 seed 聚合

| seed | target parent | matched | chain | m vs c |
|-----:|---|---:|---:|---:|
| 313 | `[0, 0, 1, 1, 2, 4]` | 0.229841 | 0.363298 | 36.7% |
| 2602 | `[0, 0, 0, 2, 2, 0]` | 0.218628 | 0.374643 | 41.6% |
| 20260227 | `[0, 0, 0, 1, 3, 1]` | 0.187048 | 0.370698 | 49.5% |

- 跨 seed 平均：matched vs chain **42.6%**
