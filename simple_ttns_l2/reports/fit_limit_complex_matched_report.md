# Fit Limit Report

This experiment tests how well TTNS can fit the complex 6D target under high capacity and longer training.

- Config: `{"n_dims": 6, "q": 2, "m": 64, "rank": 16, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 777, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000}`
- Figure: `fit_limit_complex_matched.svg`

## Training Summary

| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|
| balanced | balanced | fitlimit_delayed_cosine_hold160_0.1x | 1.000e-04 | -0.065457 | 1.000000 | 204.320 |
| chain | chain | fitlimit_delayed_cosine_hold160_0.1x | 1.000e-04 | 0.000026 | 1.000000 | 164.005 |

## Slice Fit Summary

| target_topology | pair | IAE(model,target) | IAE(target,target2) | ratio_to_floor | L2 |
|---|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.236831 | 0.077710 | 3.048 | 0.082305 |
| balanced | (0,3) | 0.172654 | 0.073728 | 2.342 | 0.054932 |
| balanced | (2,5) | 0.194900 | 0.061924 | 3.147 | 0.075103 |
| chain | (0,1) | 1.886046 | 0.074916 | 25.175 | 0.779119 |
| chain | (0,3) | 2.212036 | 0.067116 | 32.959 | 0.809242 |
| chain | (2,5) | 1.004156 | 0.038269 | 26.240 | 0.494303 |
