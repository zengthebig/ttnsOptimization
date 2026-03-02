# 2D Slice Report (complex_xlarge_adaptive)

- Config: `{"n_dims": 6, "q": 2, "m": 48, "rank": 10, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 100, "log_every": 5, "seed": 123, "n_train": 5000, "n_val": 2500, "n_test": 5000, "monitor_train_sz": 2000, "monitor_val_sz": 2000}`
- Figure: `topology_slice_complex_xlarge_adaptive.svg`

## Model Training Summary

| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|
| balanced | balanced | delayed_adaptive_hold40_factor0.5_pat2 | 1.000e-03 | -0.038620 | 1.000000 | 63.192 |
| balanced | chain | delayed_adaptive_hold40_factor0.5_pat2 | 1.000e-03 | -0.029799 | 1.000000 | 41.848 |
| chain | balanced | delayed_adaptive_hold40_factor0.5_pat2 | 3.125e-05 | 0.000030 | 1.000000 | 42.671 |
| chain | chain | delayed_adaptive_hold40_factor0.5_pat2 | 3.125e-05 | 0.000630 | 1.000000 | 41.978 |

## Slice Error Summary

| target_topology | pair | IAE balanced | IAE chain | winner |
|---|---:|---:|---:|---:|
| balanced | (0,1) | 0.571188 | 0.592673 | balanced |
| balanced | (0,3) | 0.403738 | 0.439145 | balanced |
| balanced | (2,5) | 0.518511 | 0.638513 | balanced |
| chain | (0,1) | 1.424471 | 2.909942 | balanced |
| chain | (0,3) | 0.698790 | 3.027998 | balanced |
| chain | (2,5) | 1.007133 | 1.032206 | balanced |
