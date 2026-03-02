# Slice Comparison: complex vs complex_large

- baseline: m=24, rank=3
- larger  : m=36, rank=6

## Training (final_val_l2, lower better)

| target | model | baseline | larger | delta(larger-baseline) |
|---|---:|---:|---:|---:|
| balanced | balanced | -0.016727 | -0.027712 | -0.010986 |
| balanced | chain | -0.014706 | -0.025258 | -0.010552 |
| chain | balanced | 0.000003 | 0.000009 | +0.000006 |
| chain | chain | 0.000054 | 0.000041 | -0.000013 |

## 2D Slice IAE (lower better)

| target | pair | baseline balanced | larger balanced | delta | baseline chain | larger chain | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | (0,1) | 0.733483 | 0.632007 | -0.101475 | 0.793893 | 0.619318 | -0.174575 |
| balanced | (0,3) | 0.636750 | 0.485672 | -0.151078 | 0.727380 | 0.434199 | -0.293181 |
| balanced | (2,5) | 0.771056 | 0.643610 | -0.127445 | 0.781880 | 0.649021 | -0.132858 |
| chain | (0,1) | 1.415416 | 1.620061 | +0.204645 | 8.523013 | 1.785582 | -6.737430 |
| chain | (0,3) | 1.152108 | 0.869469 | -0.282639 | 8.082762 | 1.599841 | -6.482921 |
| chain | (2,5) | 1.001212 | 1.011002 | +0.009791 | 1.000899 | 1.003383 | +0.002485 |
