# Fork DAG：联结树 TTNS vs Chain TTNS 实验报告

## 目的

在含**双父节点**依赖（$x_2 \sim f(x_0, x_1)$）的合成 fork DAG 目标上，
对比联结树风格 TTNS 与 chain TTNS 的 2D 切片 IAE。

## 配置

- 训练配置: `{"n_dims": 6, "q": 2, "m": 64, "rank": 16, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.0, "train_steps": 300, "log_every": 10, "seed": 4242, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- DAG 边: `[(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)]`
- 联结树 parent: `[0, 0, 0, 2, 2, 3]`
- chain parent: `[0, 0, 1, 2, 3, 4]`

## 训练结果

| model | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|
| junction | -0.034635 | 1.000000 | 4.634 |
| chain | -0.034317 | 1.000000 | 2.132 |

## 切片 IAE

| pair | noise_floor | IAE_junction | IAE_chain | ratio_junction | ratio_chain |
|---|---:|---:|---:|---:|---:|
| (0,1) | 0.066597 | 0.217461 | 0.222838 | 3.265 | 3.346 |
| (0,2) | 0.064780 | 0.905416 | 0.906745 | 13.977 | 13.997 |
| (1,2) | 0.067669 | 0.881082 | 0.886483 | 13.021 | 13.100 |
| (2,3) | 0.065903 | 1.075788 | 1.076809 | 16.324 | 16.339 |
| (3,5) | 0.104283 | 0.509329 | 0.510486 | 4.884 | 4.895 |

## 结论

- 全切片 mean IAE：junction=0.7178，chain=0.7207。
- 双父相关切片 ((0, 2), (1, 2), (0, 1)) mean IAE：junction=0.6680，chain=0.6720（相对提升 0.6%）。
- 优势未达 20%；可增大 rank/步数或调整联结树 parent。
