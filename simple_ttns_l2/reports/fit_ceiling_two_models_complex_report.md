# Fit Ceiling Two-Model Report

This compares balanced and chain models on exactly the same target data to check how far each can fit.

- Config: `{"n_dims": 6, "q": 2, "m": 64, "rank": 16, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 313, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000, "early_stop_patience_logs": 6, "early_stop_min_delta": 0.0001, "early_stop_warmup_logs": 4, "early_stop_restore_best": true}`
- Figure: `fit_ceiling_two_models_complex.svg`

## Training Summary

| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|
| balanced | balanced | ceiling_delayed_cosine_hold160_0.1x | 1.000e-04 | -0.066379 | 1.000000 | 8.428 |
| balanced | chain | ceiling_delayed_cosine_hold160_0.1x | 1.000e-04 | -0.044429 | 1.000000 | 3.417 |
| chain | balanced | ceiling_delayed_cosine_hold160_0.1x | 1.000e-03 | 0.000024 | 1.000000 | 2.444 |
| chain | chain | ceiling_delayed_cosine_hold160_0.1x | 1.000e-03 | 0.000108 | 1.000000 | 1.334 |

## Slice Summary

| target_topology | pair | noise_floor_IAE | IAE_balanced | IAE_chain | ratio_balanced | ratio_chain | L2_balanced | L2_chain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.072461 | 0.228714 | 0.397018 | 3.156 | 5.479 | 0.082141 | 0.153501 |
| balanced | (0,3) | 0.076111 | 0.190172 | 0.327039 | 2.499 | 4.297 | 0.061474 | 0.114142 |
| balanced | (2,5) | 0.065147 | 0.213767 | 0.539017 | 3.281 | 8.274 | 0.084376 | 0.219278 |
| chain | (0,1) | 0.073530 | 1.034464 | 1.053096 | 14.069 | 14.322 | 0.326978 | 0.317661 |
| chain | (0,3) | 0.069018 | 0.737519 | 0.765510 | 10.686 | 11.091 | 0.238197 | 0.224475 |
| chain | (2,5) | 0.037044 | 1.059331 | 1.028275 | 28.597 | 27.758 | 0.493541 | 0.492353 |
