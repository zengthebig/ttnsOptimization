# Fork DAG：数据驱动 Chow–Liu / Balanced / Chain TTNS 对比报告

## 目的

在含**双父节点**依赖（$x_2 \sim f(x_0, x_1)$）的合成 fork DAG 目标上，
用**数据驱动的 Chow–Liu 最大生成树**（从训练样本估互信息）替代手工联结树，
对比 Chow–Liu TTNS、balanced TTNS、chain TTNS 的 2D 切片 IAE 与训练耗时。

## 配置

- 训练配置: `{"n_dims": 7, "q": 2, "m": 56, "rank": 16, "batch_sz": 128, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 20260227, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- DAG 边: `[[0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6], [5, 6]]`
- Chow–Liu 估树（root=0, n_bins=16）parent: `[0, 2, 0, 2, 5, 6, 2]`
- Chow–Liu 树边: `[[0, 2], [2, 6], [2, 1], [2, 3], [6, 5], [5, 4]]`
- 命中 DAG 真边: `[[0, 2], [1, 2], [2, 3], [2, 6], [4, 5], [5, 6]]`（6/7）
- balanced parent: `[0, 0, 0, 1, 1, 2, 2]`
- chain parent: `[0, 0, 1, 2, 3, 4, 5]`

## 训练结果

| model | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|
| chow_liu | -1.023863 | 1.000000 | 69.638 |
| balanced | -0.939627 | 1.000000 | 14.498 |
| chain | -0.309858 | 1.000000 | 2.529 |

## 切片 IAE

| pair | noise_floor | IAE_chow_liu | IAE_balanced | IAE_chain |
|---|---:|---:|---:|---:|
| (0,2) | 0.064979 | 0.433400 | 0.406390 | 0.652756 |
| (1,2) | 0.065498 | 0.403203 | 0.399417 | 0.667711 |
| (2,6) | 0.062865 | 0.344586 | 0.352410 | 1.143786 |
| (5,6) | 0.102788 | 0.206334 | 0.229621 | 0.317010 |
| (2,5) | 0.105426 | 0.187923 | 0.237971 | 0.506122 |
| (0,6) | 0.065959 | 0.328519 | 0.306497 | 0.710090 |

## 关键切片聚合 ((0,2),(1,2),(2,6),(5,6))

- mean IAE chow_liu: **0.346881**
- mean IAE balanced: **0.346960**
- mean IAE chain: **0.695316**
- chow_liu vs chain 提升: **50.1%**
- 最优模型: **chow_liu**

## 结论

- 数据驱动 Chow–Liu TTNS 在关键切片上优于 chain **50.1%**，达到 20% 标准。

## 多 seed 聚合

| seed | chow_liu | balanced | chain | cl vs c | 命中边 |
|-----:|---:|---:|---:|---:|---:|
| 313 | 0.226964 | 0.278797 | 0.515022 | 55.9% | 6/7 |
| 2602 | 0.246670 | 0.296734 | 0.696439 | 64.6% | 6/7 |
| 20260227 | 0.346881 | 0.346960 | 0.695316 | 50.1% | 6/7 |

- 跨 seed 平均：chow_liu vs chain **56.9%**
