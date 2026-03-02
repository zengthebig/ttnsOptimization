# Pure TT On Chain Target Report

- Config: `{"n_dims": 6, "q": 2, "m": 64, "rank": 16, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 300, "log_every": 10, "seed": 910, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000}`
- Figure: `fit_tt_chain_target.svg`

## Training Summary

| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|
| chain | tt | tt_delayed_cosine_hold120_0.1x | 1.000e-04 | -0.001110 | 1.000000 | 10.349 |

## Slice Summary

| pair | noise_floor_IAE | IAE_TT | ratio_to_floor | L2 |
|---:|---:|---:|---:|---:|
| (0,1) | 0.074480 | 0.240575 | 3.230 | 0.081597 |
| (0,3) | 0.066823 | 0.151520 | 2.267 | 0.047795 |
| (2,5) | 0.036536 | 1.352086 | 37.007 | 0.483686 |
