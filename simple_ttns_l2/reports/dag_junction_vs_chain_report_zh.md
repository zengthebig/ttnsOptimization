# Fork DAG：联结树 / Balanced / Chain TTNS 对比报告

## 目的

在含**双父节点**依赖（$x_2 \sim f(x_0, x_1)$）的合成 fork DAG 目标上，
对比联结树风格 TTNS、balanced TTNS 与 chain TTNS 的 2D 切片 IAE。

## 配置

- 训练配置: `{"n_dims": 7, "q": 2, "m": 56, "rank": 16, "batch_sz": 128, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 4242, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- DAG 边: `[(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (5, 6)]`
- junction parent: `[0, 0, 1, 2, 2, 2, 5]`
- balanced parent: `[0, 0, 0, 1, 1, 2, 2]`
- chain parent: `[0, 0, 1, 2, 3, 4, 5]`

## 训练结果

| model | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|
| junction | -0.738264 | 1.000000 | 137.484 |
| balanced | -0.854902 | 1.000000 | 9.517 |
| chain | -0.573832 | 1.000000 | 3.134 |

## 切片 IAE

| pair | noise_floor | IAE_junction | IAE_balanced | IAE_chain |
|---|---:|---:|---:|---:|
| (0,2) | 0.063099 | 0.487217 | 0.312418 | 0.520309 |
| (1,2) | 0.067282 | 0.519575 | 0.366833 | 0.504161 |
| (2,6) | 0.065014 | 0.708295 | 0.303723 | 0.870632 |
| (5,6) | 0.104298 | 0.280061 | 0.208782 | 0.234492 |
| (2,5) | 0.104974 | 0.382622 | 0.186321 | 0.392367 |
| (0,6) | 0.071137 | 0.349238 | 0.250063 | 0.653745 |

## 关键切片聚合 ((0,2),(1,2),(2,6),(5,6))

- mean IAE junction: **0.498787**
- mean IAE balanced: **0.297939**
- mean IAE chain: **0.532398**
- junction vs chain 提升: **6.3%**
- balanced vs chain 提升: **44.0%**
- 最优模型: **balanced**

## 结论

- 联结树相对 chain 提升 **6.3%**，未达 20% 阈值。
