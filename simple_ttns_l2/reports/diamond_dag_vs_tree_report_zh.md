# Diamond（有环）拟合：多父 DAG TTNS vs 树基线

真分布：两层 DAG，下层 $x_2,x_3$ 各自同时依赖 $x_0,x_1$；moral graph 含 4-环，
任何树都至少缺一条必需边。指标为验证集 L2 目标 $\int q^2-2\,\mathbb{E}[q]$（越低越好）。

配置：`{'q': 2, 'm': 16, 'rank': 6, 'lr': 0.002, 'steps': 400, 'batch_sz': 512, 'n_train': 20000, 'n_val': 8000, 'init_noise': 0.01, 'normalize_every': 1, 'log_every': 50, 'seed': 0}`

| 模型 | final_val_l2 | best_val_l2 | 用时(s) |
|---|---|---|---|
| dag_diamond | -19.100676 | -19.100676 | 1.9 |
| tree_chain | -6.637081 | -6.637081 | 1.2 |
| tree_balanced | -12.635950 | -12.635950 | 1.5 |
| tree_star@2 | -17.150459 | -17.150459 | 1.3 |

最优树 final_val_l2 = -17.150459，DAG diamond = -19.100676，
**相对提升 = 11.4%**（L2 越低越好）。
