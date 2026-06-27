# Fit Ceiling Two-Model Report

This compares balanced and chain models on exactly the same target data to check how far each can fit.

- Config: `{"n_dims": 6, "q": 2, "m": 64, "rank": 16, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 20260227, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- Figure: `fit_ceiling_two_models_complex.svg`

## Training Summary

| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|
| balanced | balanced | ceiling_delayed_cosine_hold160_0.1x | 4.335e-04 | -0.063854 | 1.000000 | 6.137 |
| balanced | chain | ceiling_delayed_cosine_hold160_0.1x | 1.000e-04 | -0.042647 | 1.000000 | 3.189 |
| chain | balanced | ceiling_delayed_cosine_hold160_0.1x | 1.000e-03 | 0.000175 | 1.000000 | 2.275 |
| chain | chain | ceiling_delayed_cosine_hold160_0.1x | 6.665e-04 | -0.001225 | 1.000000 | 2.387 |

## Slice Summary

| target_topology | pair | noise_floor_IAE | IAE_balanced | IAE_chain | ratio_balanced | ratio_chain | L2_balanced | L2_chain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.071667 | 0.267133 | 0.390036 | 3.727 | 5.442 | 0.093862 | 0.151111 |
| balanced | (0,3) | 0.072821 | 0.217792 | 0.334771 | 2.991 | 4.597 | 0.074292 | 0.111668 |
| balanced | (2,5) | 0.065464 | 0.211580 | 0.553031 | 3.232 | 8.448 | 0.079165 | 0.222607 |
| chain | (0,1) | 0.070033 | 1.042523 | 0.320519 | 14.886 | 4.577 | 0.325020 | 0.107997 |
| chain | (0,3) | 0.071013 | 0.732246 | 0.242591 | 10.312 | 3.416 | 0.233981 | 0.077451 |
| chain | (2,5) | 0.033748 | 1.277741 | 1.392463 | 37.861 | 41.260 | 0.485442 | 0.483795 |
