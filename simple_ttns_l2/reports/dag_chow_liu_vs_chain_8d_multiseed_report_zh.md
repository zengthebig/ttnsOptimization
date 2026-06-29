# Fork DAG：数据驱动 Chow–Liu / Balanced / Chain TTNS 对比报告

## 目的

在含**双父节点**依赖（$x_2 \sim f(x_0, x_1)$）的合成 fork DAG 目标上，
用**数据驱动的 Chow–Liu 最大生成树**（从训练样本估互信息）替代手工联结树，
对比 Chow–Liu TTNS、balanced TTNS、chain TTNS 的 2D 切片 IAE 与训练耗时。

## 配置

- 训练配置: `{"n_dims": 8, "q": 2, "m": 56, "rank": 16, "batch_sz": 128, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 20260227, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- DAG 边: `[[0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [2, 7], [6, 7]]`
- Chow–Liu 估树（root=0, n_bins=16）parent: `[0, 2, 0, 2, 3, 4, 7, 2]`
- Chow–Liu 树边: `[[0, 2], [2, 7], [2, 1], [2, 3], [7, 6], [3, 4], [4, 5]]`
- 命中 DAG 真边: `[[0, 2], [1, 2], [2, 3], [2, 7], [3, 4], [4, 5], [6, 7]]`（7/8）
- balanced parent: `[0, 0, 0, 1, 1, 2, 2, 3]`
- chain parent: `[0, 0, 1, 2, 3, 4, 5, 6]`

## 训练结果

| model | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|
| chow_liu | -1.170559 | 1.000000 | 95.350 |
| balanced | -0.542724 | 1.000000 | 11.105 |
| chain | -0.436694 | 1.000000 | 3.205 |

## 切片 IAE

| pair | noise_floor | IAE_chow_liu | IAE_balanced | IAE_chain |
|---|---:|---:|---:|---:|
| (0,2) | 0.065298 | 0.257402 | 0.607706 | 0.675736 |
| (1,2) | 0.065477 | 0.265819 | 0.616537 | 0.640915 |
| (2,7) | 0.066714 | 0.311507 | 0.917583 | 1.067898 |
| (6,7) | 0.106336 | 0.159018 | 0.355359 | 0.358833 |
| (2,6) | 0.105346 | 0.150131 | 0.467307 | 0.484554 |
| (0,7) | 0.069870 | 0.213313 | 0.653091 | 0.699711 |

## 关键切片聚合 ((0,2),(1,2),(2,7),(6,7))

- mean IAE chow_liu: **0.248436**
- mean IAE balanced: **0.624296**
- mean IAE chain: **0.685845**
- chow_liu vs chain 提升: **63.8%**
- 最优模型: **chow_liu**

## 结论

- 数据驱动 Chow–Liu TTNS 在关键切片上优于 chain **63.8%**，达到 20% 标准。

## 多 seed 聚合

| seed | chow_liu | balanced | chain | cl vs c | 命中边 |
|-----:|---:|---:|---:|---:|---:|
| 313 | 0.252580 | 0.694867 | 0.737363 | 65.7% | 7/8 |
| 2602 | 0.270586 | 0.620427 | 0.743374 | 63.6% | 7/8 |
| 20260227 | 0.248436 | 0.624296 | 0.685845 | 63.8% | 7/8 |

- 跨 seed 平均：chow_liu vs chain **64.4%**
