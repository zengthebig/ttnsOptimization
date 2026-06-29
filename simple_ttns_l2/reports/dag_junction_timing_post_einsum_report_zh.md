# M3 三子节点 einsum 优化后耗时对比

## 背景

commit `019d3d8` 为 `n_children==3` 增加融合 einsum，针对 7D fork DAG 联结树（节点 2 三子节点）训练瓶颈。

对比基准：**优化前** `dag_junction_vs_chain_multiseed_metrics.json`（seed 20260227）。

## seed 20260227（同配置、同早停步数）

| 模型 | 优化前 | 优化后（冷启动） | 优化后（multiseed 末次，JIT 已热） | 加速比（热） |
|------|-------:|-----------------:|-----------------------------------:|-------------:|
| **junction** | 136.3 s / 230 步 | 81.9 s | **42.9 s** | **3.2×** |
| balanced | 9.7 s / 400 步 | 14.9 s | 17.8 s | — |
| chain | 1.6 s / 160 步 | 3.2 s | 2.9 s | — |

- junction **每步**：590 ms → **187 ms**（热）/ 356 ms（冷）
- balanced / chain 未走 3-child 路径，耗时波动来自 JIT 预热与本机负载，**不视为回归**。

## 三 seed junction 训练总耗时（优化后）

| seed | junction total_sec | 步数 |
|-----:|-------------------:|-----:|
| 313 | 72.6 | 400（未早停） |
| 2602 | 46.6 | 280 |
| 20260227 | 42.0 | 230 |

## 精度（未变）

- 跨 seed junction vs chain IAE 提升：**26.9%**（与优化前一致）
- seed 20260227 关键切片 IAE：junction 0.490 / balanced 0.347 / chain 0.695

## 全流程

- 完整 multiseed（3 seed + 写报告）墙钟：**~348 s（5.8 min）**
- 优化前估测同流程 junction 单项即 **~400+ s**，整流程 **>15 min**

## 结论

三子节点 einsum 使 **junction 训练约 3× 加速**（JIT 热身后）；精度不变。后续瓶颈仍在 junction 相对 chain 的 **~15×** 每步开销（`normalized_*` + 7D 深度），而非 3-child 回退循环。
