# TTNS L2 Objective

This directory isolates the L2 training objective for a TTNS model:

q_theta(x) = <T_theta, tensor_k b_k(x_k)>

with

L(theta) = int q_theta(x)^2 dx - 2 E_data[q_theta(x)] + const.

Implemented utilities:

- rank-1 evaluation q_theta(x)
- integral int q_theta^2 via local Gram matrices
- Monte-Carlo estimate of E_data[q_theta]
- objective assembly for optimization
- post-step normalization by int q_theta

This is intentionally separated from the existing log-likelihood training path.

## Standalone training entry

Run from repository root:

```bash
python -m simple_ttns_l2.train_l2 \
  --dataset Power \
  --q 2 --m 64 --rank 16 \
  --ttns-topology balanced \
  --init-noise 0.01 \
  --batch-sz 512 --train-noise 0.01 --lr 0.001 --train-steps 1000 \
  --normalize-every 1 \
  --data-dir /path/to/data --work-dir /path/to/workdir
```

## Tests

```bash
python -m unittest simple_ttns_l2/tests/test_ttns_l2_objective_unittest.py -v
python -m unittest simple_ttns_l2/tests/test_train_l2_smoke_unittest.py -v
```
