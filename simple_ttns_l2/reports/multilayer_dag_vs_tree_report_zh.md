# 多层 DAG 传播 pipeline：多父 DAG TTNS vs 树基线

结构：L1={0,1} 源；L2={2,3} 各依赖 (0,1)（和/差）；L3={4} 依赖 (2,3)。
传播按拓扑序逐层采样（源 → delay 核 → 下层），全联合用多父 DAGTTNS 拟合。
moral graph 含环且边数(6)>树上限(4)，任何树至少缺 2 条必需边。
指标：验证集 L2 目标 $\int q^2-2\,\mathbb{E}[q]$（越低越好）。

配置：`{'q': 2, 'm': 16, 'rank': 6, 'lr': 0.002, 'steps': 400, 'batch_sz': 512, 'n_train': 30000, 'n_val': 10000, 'init_noise': 0.01, 'normalize_every': 1, 'log_every': 50, 'seed': 0}`

| 模型 | final_val_l2 | best_val_l2 | 用时(s) |
|---|---|---|---|
| dag_multilayer | -97.143981 | -97.143981 | 4.0 |
| tree_chain | -31.930266 | -31.930266 | 1.4 |
| tree_balanced | -56.875998 | -56.875998 | 1.8 |
| tree_hub@2 | -75.126259 | -75.126259 | 3.9 |
| tree_spine | -37.769708 | -37.769708 | 1.7 |

最优树 final_val_l2 = -75.126259，DAG multilayer = -97.143981，
**相对提升 = 29.3%**（L2 越低越好）。
