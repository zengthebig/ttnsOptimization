# Max-plus 多层 DAG 传播（方案 A：采样 + 逐层重拟合）

数据依赖为多层 DAG，模型侧仅用**单父 TTNS**；多来源聚合为 max-plus：
$$x_v=\max_{u\in\mathrm{pa}(v)}(x_u+e_{uv})+d_v.$$

传播逐层进行：从上一层拟合好的 TTNS 采样 → 施加 max-plus（采新 delay）→
重拟合下一层单父 TTNS。`pred` = 方案 A 传播样本，`ref` = 从真值上层样本传播
（传播可达上界，隔离 TTNS 拟合+采样误差）。指标越小越好。

配置：`layer_sizes=[4, 4, 4], fanin=2, delay={'src_lo': 0.0, 'src_hi': 1.0, 'edge_lo': 0.0, 'edge_hi': 0.4, 'node_lo': 0.0, 'node_hi': 0.4}`

`q=2, m=16, rank=8, n=15000, steps=800`

| 层 | 变量数 | 拟合val_l2 | W1_marg(pred) | W1_marg(ref) | corr_fro(pred) | corr_fro(ref) |
|---|---|---|---|---|---|---|
| 0(源) | 4 | -0.9461 | 0.0035 | - | - | - |
| 1 | 4 | -1.2508 | 0.0035 | 0.0025 | 0.0551 | 0.0328 |
| 2 | 4 | -1.4078 | 0.0071 | 0.0035 | 0.1187 | 0.0162 |

解读：`pred` 接近 `ref` 说明 TTNS 表示+采样在传播链上保真；`pred` 与 `ref` 的差距即
单父 TTNS 对各层联合密度的压缩损失 + 采样截断误差的累积。
