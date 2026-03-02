# Topology Comparison Report

- Date: 2026-02-26T19:14:56
- Config: `{"n_dims": 6, "q": 2, "m": 24, "rank": 3, "batch_sz": 256, "lr": 0.001, "train_noise": 0.01, "init_noise": 0.01, "train_steps": 100, "log_every": 5, "seed": 123, "n_train": 5000, "n_val": 2500, "n_test": 5000, "monitor_train_sz": 2000, "monitor_val_sz": 2000}`

## Final Metrics (lower L2 is better)

| target_topology | model_topology | final_val_l2 | final_test_l2 | final_integral | total_time_sec |
|---|---:|---:|---:|---:|---:|
| balanced | balanced | -0.017499 | -0.016924 | 1.000000 | 63.681 |
| balanced | chain | -0.016534 | -0.016113 | 1.000000 | 41.848 |
| chain | balanced | -0.000031 | -0.000031 | 1.000000 | 40.773 |
| chain | chain | -0.000026 | -0.000027 | 1.000000 | 40.108 |

## Topology Preference Check

- Target `balanced`: balanced-model test L2 = `-0.016924`, chain-model test L2 = `-0.016113`. Conclusion: `balanced better`.
- Target `chain`: chain-model test L2 = `-0.000027`, balanced-model test L2 = `-0.000031`. Conclusion: `balanced better or tie`.

## Overall

Matched topology does not win in both directions under this setup; consider larger rank/steps or multi-seed averaging.
