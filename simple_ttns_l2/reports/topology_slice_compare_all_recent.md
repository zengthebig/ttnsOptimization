# Comparison: large vs xlarge variants

## Config

| run | m | rank | lr |
|---|---:|---:|---:|
| large | 36 | 6 | 1.000e-03 |
| xlarge | 48 | 10 | 1.000e-03 |
| xlarge_decay | 48 | 10 | 5.000e-04 |
| xlarge_adaptive | 48 | 10 | 1.000e-03 |

## Training final_val_l2 (lower better)

| target | model | large | xlarge | xlarge_decay | xlarge_adaptive |
|---|---:|---:|---:|---:|---:|
| balanced | balanced | -0.027712 | -0.038620 | -0.005762 | -0.038620 |
| balanced | chain | -0.025258 | -0.029799 | -0.005115 | -0.029799 |
| chain | balanced | 0.000009 | 0.000015 | -0.000006 | 0.000030 |
| chain | chain | 0.000041 | 0.000014 | -0.000005 | 0.000630 |

## Slice IAE for balanced model (lower better)

| target | pair | large | xlarge | xlarge_decay | xlarge_adaptive |
|---|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.632007 | 0.571188 | 0.904915 | 0.571188 |
| balanced | (0,3) | 0.485672 | 0.403738 | 0.761788 | 0.403738 |
| balanced | (2,5) | 0.643610 | 0.518511 | 0.917486 | 0.518511 |
| chain | (0,1) | 1.620061 | 1.151442 | 0.762067 | 1.424471 |
| chain | (0,3) | 0.869469 | 0.730359 | 0.663105 | 0.698790 |
| chain | (2,5) | 1.011002 | 1.003168 | 1.024587 | 1.007133 |

## Slice IAE for chain model (lower better)

| target | pair | large | xlarge | xlarge_decay | xlarge_adaptive |
|---|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.619318 | 0.592673 | 0.937506 | 0.592673 |
| balanced | (0,3) | 0.434199 | 0.439145 | 0.781313 | 0.439145 |
| balanced | (2,5) | 0.649021 | 0.638513 | 0.967064 | 0.638513 |
| chain | (0,1) | 1.785582 | 4.138202 | 0.767151 | 2.909942 |
| chain | (0,3) | 1.599841 | 4.172712 | 0.686632 | 3.027998 |
| chain | (2,5) | 1.003383 | 1.004930 | 1.023180 | 1.032206 |
