# 2D Slice Report (complex)

- Config: `{"n_dims": 6, "q": 2, "m": 24, "rank": 3, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 100, "log_every": 5, "seed": 123, "n_train": 5000, "n_val": 2500, "n_test": 5000, "monitor_train_sz": 2000, "monitor_val_sz": 2000}`
- Figure: `topology_slice_complex.svg`

## Model Training Summary

| target_topology | model_topology | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|
| balanced | balanced | -0.016727 | 1.000000 | 59.576 |
| balanced | chain | -0.014706 | 1.000000 | 40.153 |
| chain | balanced | 0.000003 | 1.000000 | 40.686 |
| chain | chain | 0.000054 | 1.000000 | 40.413 |

## Slice Error Summary

| target_topology | pair | IAE balanced | IAE chain | winner |
|---|---:|---:|---:|---:|
| balanced | (0,1) | 0.733483 | 0.793893 | balanced |
| balanced | (0,3) | 0.636750 | 0.727380 | balanced |
| balanced | (2,5) | 0.771056 | 0.781880 | balanced |
| chain | (0,1) | 1.415416 | 8.523013 | balanced |
| chain | (0,3) | 1.152108 | 8.082762 | balanced |
| chain | (2,5) | 1.001212 | 1.000899 | chain |
