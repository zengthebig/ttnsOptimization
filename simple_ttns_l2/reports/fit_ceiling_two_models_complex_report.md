# Fit Ceiling Two-Model Report

This compares balanced and chain models on exactly the same target data to check how far each can fit.

- Config: `{"n_dims": 6, "q": 2, "m": 64, "rank": 16, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 400, "log_every": 10, "seed": 313, "n_train": 8000, "n_val": 4000, "n_test": 4000, "monitor_train_sz": 3000, "monitor_val_sz": 3000}`
- Figure: `fit_ceiling_two_models_complex.svg`

## Training Summary

| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|---:|
| balanced | balanced | ceiling_delayed_cosine_hold160_0.1x | 1.000e-04 | -0.070321 | 1.000000 | 211.611 |
| balanced | chain | ceiling_delayed_cosine_hold160_0.1x | 1.000e-04 | -0.049015 | 1.000000 | 171.596 |
| chain | balanced | ceiling_delayed_cosine_hold160_0.1x | 1.000e-04 | 0.000119 | 1.000000 | 189.295 |
| chain | chain | ceiling_delayed_cosine_hold160_0.1x | 1.000e-04 | 0.000010 | 1.000000 | 175.749 |

## Slice Summary

| target_topology | pair | noise_floor_IAE | IAE_balanced | IAE_chain | ratio_balanced | ratio_chain | L2_balanced | L2_chain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.073042 | 0.213462 | 0.437181 | 2.922 | 5.985 | 0.074755 | 0.169130 |
| balanced | (0,3) | 0.073189 | 0.159684 | 0.374487 | 2.182 | 5.117 | 0.051235 | 0.124906 |
| balanced | (2,5) | 0.063058 | 0.191600 | 0.555089 | 3.038 | 8.803 | 0.075089 | 0.229387 |
| chain | (0,1) | 0.073106 | 3.447802 | 3.276745 | 47.162 | 44.822 | 1.043130 | 0.840879 |
| chain | (0,3) | 0.069871 | 2.522658 | 1.358634 | 36.105 | 19.445 | 0.826577 | 0.392368 |
| chain | (2,5) | 0.038332 | 1.003289 | 1.001582 | 26.174 | 26.129 | 0.495310 | 0.495218 |
