# Slice Comparison: xlarge constant-lr vs xlarge decay-lr

- constant: m=48, rank=10, lr=0.001
- decay   : m=48, rank=10, lr=0.0005 with cosine decay to 0.1x

## Training (final_val_l2, lower better)

| target | model | constant | decay | delta(decay-constant) |
|---|---:|---:|---:|---:|
| balanced | balanced | -0.038620 | -0.005762 | +0.032858 |
| balanced | chain | -0.029799 | -0.005115 | +0.024684 |
| chain | balanced | 0.000015 | -0.000006 | -0.000021 |
| chain | chain | 0.000014 | -0.000005 | -0.000019 |

## 2D Slice IAE (lower better)

| target | pair | constant balanced | decay balanced | delta | constant chain | decay chain | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.571188 | 0.904915 | +0.333727 | 0.592673 | 0.937506 | +0.344833 |
| balanced | (0,3) | 0.403738 | 0.761788 | +0.358050 | 0.439145 | 0.781313 | +0.342168 |
| balanced | (2,5) | 0.518511 | 0.917486 | +0.398975 | 0.638513 | 0.967064 | +0.328550 |
| chain | (0,1) | 1.151442 | 0.762067 | -0.389375 | 4.138202 | 0.767151 | -3.371051 |
| chain | (0,3) | 0.730359 | 0.663105 | -0.067255 | 4.172712 | 0.686632 | -3.486080 |
| chain | (2,5) | 1.003168 | 1.024587 | +0.021419 | 1.004930 | 1.023180 | +0.018250 |
