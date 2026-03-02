# 2D Slice Report (complex_xlarge_decay)

- Config: `{"n_dims": 6, "q": 2, "m": 48, "rank": 10, "batch_sz": 256, "lr": 0.0005, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 100, "log_every": 5, "seed": 123, "n_train": 5000, "n_val": 2500, "n_test": 5000, "monitor_train_sz": 2000, "monitor_val_sz": 2000}`
- Figure: `topology_slice_complex_xlarge_decay.svg`

## Model Training Summary

| target_topology | model_topology | lr_schedule | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|
| balanced | balanced | cosine_decay_0.1x | -0.005762 | 1.000000 | 62.751 |
| balanced | chain | cosine_decay_0.1x | -0.005115 | 1.000000 | 43.215 |
| chain | balanced | cosine_decay_0.1x | -0.000006 | 1.000000 | 42.192 |
| chain | chain | cosine_decay_0.1x | -0.000005 | 1.000000 | 42.221 |

## Slice Error Summary

| target_topology | pair | IAE balanced | IAE chain | winner |
|---|---:|---:|---:|---:|
| balanced | (0,1) | 0.904915 | 0.937506 | balanced |
| balanced | (0,3) | 0.761788 | 0.781313 | balanced |
| balanced | (2,5) | 0.917486 | 0.967064 | balanced |
| chain | (0,1) | 0.762067 | 0.767151 | balanced |
| chain | (0,3) | 0.663105 | 0.686632 | balanced |
| chain | (2,5) | 1.024587 | 1.023180 | chain |
