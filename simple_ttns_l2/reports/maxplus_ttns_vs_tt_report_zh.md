# 多层 max-plus DAG 全联合：TTNS（树）vs TT（链）

在 max-plus 多层 DAG 的**全联合**样本上，同等参数预算下用不同单父拓扑拟合 L2 密度。
TT_chain 即 Tensor Train（链）；TTNS_chowliu 为数据驱动最大互信息树（匹配真实依赖）。
指标为留出测试集 L2 目标 $\int q^2-2\,\mathbb{E}[q]$（越低越好）。

配置：`layer_sizes=[4, 4, 4], fanin=2, delay={'src_lo': 0.0, 'src_hi': 1.0, 'edge_lo': 0.0, 'edge_hi': 0.25, 'node_lo': 0.0, 'node_hi': 0.25}, n_total=40000, budget≈120000`

| 模型 | rank | n_params | val_l2 | test_l2 | 用时(s) |
|---|---|---|---|---|---|
| TT_chain | 27 | 117504 | -93.1873 | -117.3263 | 23.0 |
| TTNS_chowliu | 8 | 79488 | -417.4232 | -392.1785 | 15.0 |
| TTNS_balanced | 12 | 116352 | -154.9453 | -153.4867 | 19.6 |

TT(链) test_l2 = -117.3263；最优 TTNS（TTNS_chowliu）test_l2 = -392.1785，
**相对提升 = 234.3%**（L2 越低越好；>0 表示 TTNS 优于 TT）。

注：chow-liu 树含高 degree 节点，在同一 budget 下只能取较低 rank，实际仅用 79488 参数
（< 链的 117504），却仍大幅胜出——说明优势来自**拓扑结构匹配**真实依赖，而非参数量。
balanced 树后期 L2 出现发散，已由 early-stop 还原最优。
