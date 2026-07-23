# dense_dag_r567 跨簇 DAG × 分层 TTNS(R5/R7)结果报告

- **图**: 100 节点 · 5 层 · 20 维/层 · 簇=[4, 4, 4, 4, 4] · fanin=3 · 边=272
- **跨簇**: rotate_cross=True · cross_fanin=2 (逐层旋转桥接，制造真实跨簇相关)
- **分块口径(模型)**: block_mode=`immediate` (下一层节点是否同 TTNS 只看上一层直接父，不追祖先源 → 每层块有界)
- **方法**: R5_tree, R7_sampled(R6/global 关闭) · seeds=[0] · init_noise=0.001
- **示意图**: `simple_ttns_l2/reports/dense_dag_r567_schematic.png`
- **结果图**: `simple_ttns_l2/reports/dense_dag_r567_results.png` (逐层 LL/corr_fro，R5 vs R7；用 `plot_dense_dag_results` 生成)
- **切片图**: `simple_ttns_l2/reports/dense_dag_r567_slices.png` (逐层逐节点边缘密度 GT vs R5 vs R7；用 `plot_dense_dag_slices` 生成)
- **精细切片图**: `simple_ttns_l2/reports/dense_dag_r567_slice_refined.png` (三块:逐层top-3边缘密度+残差带 / 误差vs深度 / 最强相关对散点；用 `plot_dense_slice_refined` 生成)

## 逐层 joint_LL@truth (↑ 越高越好)

| 层 K | R5_tree | R7_sampled |
|---|---|---|
| L0 (K=20) | +22.693±0.000 | +22.693±0.000 |
| L1 (K=20) | +5.272±0.000 | +5.838±0.000 |
| L2 (K=20) | +5.212±0.000 | +5.108±0.000 |
| L3 (K=20) | +1.968±0.000 | +3.111±0.000 |
| L4 (K=20) | -13.015±0.000 | +1.049±0.000 |

## 逐层 corr_fro vs truth (↓ 越低越好)

| 层 K | R5_tree | R7_sampled |
|---|---|---|
| L0 (K=20) | +0.322±0.000 | +0.325±0.000 |
| L1 (K=20) | +2.151±0.000 | +1.682±0.000 |
| L2 (K=20) | +4.045±0.000 | +3.445±0.000 |
| L3 (K=20) | +5.118±0.000 | +4.402±0.000 |
| L4 (K=20) | +5.768±0.000 | +4.901±0.000 |

## 学习参数量 / 平均单 seed 用时

| 方法 | 参数量 | 用时(s) |
|---|---|---|
| R5_tree | 193,248 | 246.5 |
| R7_sampled | 183,840 | 234.8 |
