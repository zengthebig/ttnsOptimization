# 分层 TTNS（逐层森林 + max-plus 传播）vs 扁平 TT/TTNS

分层联合密度（框架设定：DAG 结构与 delay 核已知）：
$$p(x)=p_{\text{forest}}(L_0)\cdot\prod_{l\ge1}\prod_{v\in L_l}K_v(x_v\mid\mathrm{pa}(v)),$$
源层用层内分块森林（从数据学），$K_v$ 为已知 max-plus 条件核（闭式）。
扁平基线把全 12 节点当整体拟合 L2 密度。指标：留出集平均**联合对数密度**（越高越好）。

配置：`layer_sizes=[4, 4, 4], fanin=2, delay={'src_lo': 0.0, 'src_hi': 1.0, 'edge_lo': 0.0, 'edge_hi': 0.25, 'node_lo': 0.0, 'node_hi': 0.25}, n_total=40000, budget≈120000`

源层分块结果：`L0 blocks = [[0], [1], [2], [3]]`（按 MI 阈值 0.02）。

| 模型 | 学习参数量 | joint_loglik | 非正密度占比 |
|---|---|---|---|
| layered_forest+maxplus | 64 | 7.2313 | 0.0000 |
| TT_chain(rank=27) | 117504 | -9.6523 | 0.3695 |
| TTNS_chowliu(rank=8) | 79488 | 3.8050 | 0.0248 |

分层 joint_LL = 7.2313；最优扁平（TTNS_chowliu）= 3.8050，**差 = 3.4263 nats/样本**（>0 表示分层更优）。

说明：分层模型仅学习源层（参数远少于扁平模型），借助已知 DAG 结构与 max-plus 条件核
因子分解联合，在留出集上的联合对数密度显著高于扁平 TT/TTNS——扁平模型无法捕捉跨层
max-plus 依赖。这验证了**逐层建模（结构感知）**相对**整体扁平建模**的代表能力优势。
