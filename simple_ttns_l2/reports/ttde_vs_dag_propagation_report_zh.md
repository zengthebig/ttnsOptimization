# TTDE 做法（平方树+MLE） vs 我们 DAG 传播做法（线性+L2）

同一个例子（4 层 / 20 节点 banded 多父、乘积交互核），同一把尺子。

- **TTDE 做法**：$q=f^2/Z$，MLE 训练（原版 TTDE 的平方密度方案）。
- **我们 DAG 传播**：线性多父 DAG $q=\langle T,b\rangle$，L2 目标训练。

原生指标不同（NLL vs L2），故统一用 **held-out 平均对数似然 LL**（越高越好）：
把模型看成密度 $v/Z$ 后算 $\mathbb{E}[\log(v/Z)]$。平方树 $v=f^2$ 恒正；
线性 DAG $v=q$ 可能为负，在 $v>0$ 的点上算 LL 并报告 `nonpositive_rate`（负值占比）。
**参数量对齐**：以 DAG 参数量为预算，给平方树取 ≤ 预算的最大 rank。

配置：`{'q': 2, 'm': 10, 'rank': 6, 'n_train': 40000, 'n_val': 12000, 'init_noise': 0.01, 'batch_sz': 1024, 'train_noise': 0.05, 'log_every': 50, 'seed': 0, 'dag_lr': 0.0005, 'dag_steps': 600, 'dag_normalize_every': 1, 'dag_lr_decay': True, 'tree_lr': 0.0003, 'tree_steps': 400, 'tree_grad_clip': 1.0, 'tree_early_stop': 4, 'mix_n_comp': 8, 'mix_lr': 0.0003, 'mix_steps': 400}`

| 模型 | 做法 | rank | 参数量 | 共同指标 val_LL↑ | 负值占比 | 原生指标 | 原生值 | 用时(s) |
|---|---|---|---|---|---|---|---|---|
| dag_linear_l2 | 我们DAG传播(线性+L2) | 6 | 89400 | 10.1122 | 0.0404 | val_l2 | -461624.0697 | 113.0 |
| square_tree_chain | TTDE(平方+MLE) | 22 | 87560 | 8.2311 | 0.0000 | val_nll | -8.2311 | 12.9 |
| square_tree_balanced | TTDE(平方+MLE) | 10 | 83000 | 9.2011 | 0.0000 | val_nll | -9.2011 | 11.2 |
| ttde_mix_chain | 完整TTDE(平方+8排列混合) | 7 | 71680 | 7.8087 | 0.0000 | val_nll | -7.8087 | 45.8 |
| ttde_mix_balanced | 完整TTDE(平方+8排列混合) | 5 | 88000 | 8.3964 | 0.0000 | val_nll | -8.3964 | 25.9 |

DAG 共同 LL = 10.1122，最优树（含完整 TTDE 混合）共同 LL = 9.2011，
**LL 差 = 0.9111 nats**（>0 表示线性 DAG 传播在同一尺子上更优）。
