# 非负重参数化（core 参数 $x\to x^2$ 或 $x\to e^x$）对 DAG TTNS 的影响

同一例子（4 层/20 节点 banded 多父、乘积交互核），同结构同 rank。

B-spline 基处处非负 + core 参数非负 → $q\ge0$ 处处成立 → 合法密度、负值占比为 0、
归一化只需一阶 $\int q$，DAG 可直接 MLE。代价：非负张量网络失去符号相消的表达力。

- `square`: $x^2$（$x=0$ 梯度消失、$\pm$ 对称）
- `exp`: $e^x$（处处正梯度、优化更顺；无稀疏 init，用随机小初始化）

共同尺子：held-out 平均 LL（越高越好）；负值占比越低越好。

配置：`{'q': 2, 'm': 10, 'rank': 6, 'n_train': 40000, 'n_val': 12000, 'init_noise': 0.01, 'batch_sz': 1024, 'train_noise': 0.05, 'log_every': 50, 'seed': 0, 'lr': 0.0005, 'steps': 600, 'grad_clip': 1.0, 'lr_decay': True, 'early_stop': 4, 'reparams': ['square', 'exp']}`

| 模型 | 变体 | 参数量 | val_LL↑ | 负值占比 | 用时(s) |
|---|---|---|---|---|---|
| dag_linear_l2 | 线性+L2 | 89400 | 10.1122 | 0.0404 | 107.9 |
| dag_square_l2 | 参数平方+L2 | 89400 | 6.0791 | 0.0000 | 70.5 |
| dag_square_mle | 参数平方+MLE | 89400 | 7.7195 | 0.0000 | 97.2 |
| dag_exp_l2 | 参数exp+L2 | 89400 | -3.9285 | 0.0000 | 117.3 |
| dag_exp_mle | 参数exp+MLE | 89400 | -2.0657 | 0.0000 | 85.7 |
