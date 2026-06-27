# Fork DAG：联结树 / Balanced / Chain TTNS 对比报告

## 目的

在含**双父节点**依赖（$x_2 \sim f(x_0, x_1)$）的合成 fork DAG 目标上，
对比联结树风格 TTNS、balanced TTNS 与 chain TTNS 的 2D 切片 IAE。

## 配置

- 训练配置: `{"n_dims": 7, "q": 2, "m": 56, "rank": 16, "batch_sz": 128, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 20260227, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- DAG 边: `[(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (5, 6)]`
- junction parent: `[0, 0, 1, 2, 2, 2, 5]`
- balanced parent: `[0, 0, 0, 1, 1, 2, 2]`
- chain parent: `[0, 0, 1, 2, 3, 4, 5]`

## 训练结果

| model | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|
| junction | -0.739501 | 1.000000 | 136.304 |
| balanced | -0.939627 | 1.000000 | 9.664 |
| chain | -0.309858 | 1.000000 | 1.648 |

## 切片 IAE

| pair | noise_floor | IAE_junction | IAE_balanced | IAE_chain |
|---|---:|---:|---:|---:|
| (0,2) | 0.064979 | 0.521661 | 0.406390 | 0.652756 |
| (1,2) | 0.065498 | 0.487582 | 0.399417 | 0.667711 |
| (2,6) | 0.062865 | 0.685808 | 0.352410 | 1.143786 |
| (5,6) | 0.102788 | 0.266185 | 0.229621 | 0.317010 |
| (2,5) | 0.105426 | 0.345436 | 0.237971 | 0.506122 |
| (0,6) | 0.065959 | 0.371283 | 0.306497 | 0.710090 |

## 关键切片聚合 ((0,2),(1,2),(2,6),(5,6))

- mean IAE junction: **0.490309**
- mean IAE balanced: **0.346960**
- mean IAE chain: **0.695316**
- junction vs chain 提升: **29.5%**
- balanced vs chain 提升: **50.1%**
- 最优模型: **balanced**

## 结论

- 联结树 TTNS 在关键切片上优于 chain **29.5%**，达到 Phase 2 M3 标准。

## 多 seed 聚合

| seed | junction | balanced | chain | j vs c |
|-----:|---:|---:|---:|---:|
| 313 | 0.387522 | 0.278797 | 0.515022 | 24.8% |
| 2602 | 0.511550 | 0.296734 | 0.696439 | 26.5% |
| 20260227 | 0.490309 | 0.346960 | 0.695316 | 29.5% |

- 跨 seed 平均：junction vs chain **26.9%**
