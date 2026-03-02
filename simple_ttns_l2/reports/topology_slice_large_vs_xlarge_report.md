# Slice Comparison: complex_large vs complex_xlarge

- complex_large : m=36, rank=6
- complex_xlarge: m=48, rank=10

## Training (final_val_l2, lower better)

| target | model | large | xlarge | delta(xlarge-large) |
|---|---:|---:|---:|---:|
| balanced | balanced | -0.027712 | -0.038620 | -0.010907 |
| balanced | chain | -0.025258 | -0.029799 | -0.004541 |
| chain | balanced | 0.000009 | 0.000015 | +0.000007 |
| chain | chain | 0.000041 | 0.000014 | -0.000027 |

## 2D Slice IAE (lower better)

| target | pair | large balanced | xlarge balanced | delta | large chain | xlarge chain | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.632007 | 0.571188 | -0.060820 | 0.619318 | 0.592673 | -0.026645 |
| balanced | (0,3) | 0.485672 | 0.403738 | -0.081935 | 0.434199 | 0.439145 | +0.004946 |
| balanced | (2,5) | 0.643610 | 0.518511 | -0.125100 | 0.649021 | 0.638513 | -0.010508 |
| chain | (0,1) | 1.620061 | 1.151442 | -0.468620 | 1.785582 | 4.138202 | +2.352620 |
| chain | (0,3) | 0.869469 | 0.730359 | -0.139110 | 1.599841 | 4.172712 | +2.572872 |
| chain | (2,5) | 1.011002 | 1.003168 | -0.007834 | 1.003383 | 1.004930 | +0.001547 |
