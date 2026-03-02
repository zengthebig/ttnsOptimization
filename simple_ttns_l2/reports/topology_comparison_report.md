# Topology Comparison Report

- Date: 2026-02-26T17:52:25
- Config: `{"n_dims": 6, "q": 2, "m": 16, "rank": 2, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 100, "log_every": 5, "seed": 0, "n_train": 4000, "n_val": 2000, "n_test": 4000, "monitor_train_sz": 2000, "monitor_val_sz": 2000}`

## Final Metrics (lower L2 is better)

| target_topology | model_topology | final_val_l2 | final_test_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|
| balanced | balanced | -0.002866 | -0.002606 | 1.000000 | 61.376 |
| balanced | chain | -0.002241 | -0.001943 | 1.000000 | 42.255 |
| chain | balanced | -0.000138 | -0.000470 | 1.000000 | 40.258 |
| chain | chain | -0.001375 | -0.001585 | 1.000000 | 40.819 |

## Topology Preference Check

- Target `balanced`: balanced-model test L2 = `-0.002606`, chain-model test L2 = `-0.001943`. Conclusion: `balanced better`.
- Target `chain`: chain-model test L2 = `-0.001585`, balanced-model test L2 = `-0.000470`. Conclusion: `chain better`.

## Overall

Matched topology wins in both directions under this setup.
