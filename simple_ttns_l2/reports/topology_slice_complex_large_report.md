# 2D Slice Report (complex_large)

- Config: `{"n_dims": 6, "q": 2, "m": 36, "rank": 6, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 100, "log_every": 5, "seed": 123, "n_train": 5000, "n_val": 2500, "n_test": 5000, "monitor_train_sz": 2000, "monitor_val_sz": 2000}`
- Figure: `topology_slice_complex_large.svg`

## Model Training Summary

| target_topology | model_topology | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|
| balanced | balanced | -0.027712 | 1.000000 | 60.010 |
| balanced | chain | -0.025258 | 1.000000 | 40.714 |
| chain | balanced | 0.000009 | 1.000000 | 40.869 |
| chain | chain | 0.000041 | 1.000000 | 40.482 |

## Slice Error Summary

| target_topology | pair | IAE balanced | IAE chain | winner |
|---|---:|---:|---:|---:|
| balanced | (0,1) | 0.632007 | 0.619318 | chain |
| balanced | (0,3) | 0.485672 | 0.434199 | chain |
| balanced | (2,5) | 0.643610 | 0.649021 | balanced |
| chain | (0,1) | 1.620061 | 1.785582 | balanced |
| chain | (0,3) | 0.869469 | 1.599841 | balanced |
| chain | (2,5) | 1.011002 | 1.003383 | chain |
