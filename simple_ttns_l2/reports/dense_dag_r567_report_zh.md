# dense_dag_r567 跨簇 DAG × 分层 TTNS(R5/R7)结果报告

- **图**: 100 节点 · 5 层 · 20 维/层 · 簇=[4, 4, 4, 4, 4] · fanin=3 · 边=272
- **跨簇**: rotate_cross=True · cross_fanin=2 (逐层旋转桥接，制造真实跨簇相关)
- **分块口径(模型)**: block_mode=`immediate` (下一层节点是否同 TTNS 只看上一层直接父，不追祖先源 → 每层块有界)
- **方法**: R5_tree, R7_sampled(R6/global 关闭) · seeds=[0, 1, 2] · init_noise=0.0
- **示意图**: `simple_ttns_l2/reports/dense_dag_r567_schematic.png`
- **结果图**: `simple_ttns_l2/reports/dense_dag_r567_results.png` (逐层 LL/corr_fro，R5 vs R7；用 `plot_dense_dag_results` 生成)
- **切片图**: `simple_ttns_l2/reports/dense_dag_r567_slices.png` (逐层逐节点边缘密度 GT vs R5 vs R7；用 `plot_dense_dag_slices` 生成)
- **精细切片图**: `simple_ttns_l2/reports/dense_dag_r567_slice_refined.png` (三块:逐层top-3边缘密度+残差带 / 误差vs深度 / 最强相关对散点；用 `plot_dense_slice_refined` 生成)

## 逐层 joint_LL@truth (↑ 越高越好)

| 层 K | R5_tree | R7_sampled |
|---|---|---|
| L0 (K=20) | +22.697±0.070 | +22.697±0.070 |
| L1 (K=20) | -1.082±0.073 | +2.114±0.104 |
| L2 (K=20) | -1.984±0.085 | -1.132±0.113 |
| L3 (K=20) | -2.431±0.126 | -5.063±0.205 |
| L4 (K=20) | -7.494±0.120 | -7.520±0.533 |

## 逐层 corr_fro vs truth (↓ 越低越好)

| 层 K | R5_tree | R7_sampled |
|---|---|---|
| L0 (K=20) | +0.331±0.008 | +0.323±0.007 |
| L1 (K=20) | +3.758±0.015 | +3.763±0.010 |
| L2 (K=20) | +5.407±0.026 | +5.402±0.018 |
| L3 (K=20) | +5.885±0.030 | +5.877±0.031 |
| L4 (K=20) | +6.185±0.030 | +6.193±0.047 |

## 学习参数量 / 平均单 seed 用时

| 方法 | 参数量 | 用时(s) |
|---|---|---|
| R5_tree | 180,704 | 59.3 |
| R7_sampled | 165,024 | 128.8 |
