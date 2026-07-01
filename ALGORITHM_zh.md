# ALGORITHM_zh.md — 多层 DAG × TTNS 算法交接总纲

> **定位**：这是"读完即可接手"的**算法总纲**，覆盖当前多层 DAG × TTNS 主线的架构、算法、公式、关键文件、复现命令、已验证结论与负结果、未决方向。
> **与其它文档的关系**：`Program.md`（人维护，含早期单层 TTNS 算子/测试细节，截至 2026-06-27，多层部分已过时）；`prompt_phase2.md`（agent 维护的流水账 checklist）；`simple_ttns_l2/reports/*`（逐实验报告）。本文件是它们的**统一入口**。
> **语言/公式规则**：正文中文；公式用 `$` 定界；标识符/命令/路径保持英文。

---

## 0. 一句话概述

在**已知结构的多层 DAG**（节点分层、层间用已知 **max-plus 延迟核**传播）上做高维密度近似：把联合分布**逐层**建成**单父 TTNS 森林**（层内按相关性分块、每块一棵 chow-liu 树 TTNS），层间用 max-plus 传播。核心结论：**结构感知的分层 TTNS 远优于把全联合硬拟合的全局 TT / 全局 TTNS**（参数少三个数量级，joint_LL / 边缘 / 相关全面领先）。

---

## 1. 数据模型（真值生成过程）

### 1.1 多层 DAG 拓扑

- `MultiLayerSpec`（`simple_ttns_l2/dag_pipeline.py`）：节点分层 `layers`，有向边 `edges`（父在更上层）。
- `build_layered_spec(layer_sizes, fanin, wrap)`：banded 多父阶梯图——下层每个节点连上层 `fanin` 个相邻父，`wrap=True` 成环。相邻子节点**共享父** → 层内产生大量短环（树不可完全表达）。

### 1.2 max-plus 传播核（已知、固定）

节点 $v$ 由其父集 $\mathrm{pa}(v)$（都在上一层）生成：
$$x_v=\max_{u\in\mathrm{pa}(v)}\big(x_u+e_{uv}\big)+d_v,\qquad e_{uv}\sim U[\text{edge\_lo},\text{edge\_hi}],\ d_v\sim U[\text{node\_lo},\text{node\_hi}]$$
- edge delay $e$、node delay $d$ 已知、独立、连续（均匀）。参数见 `DelayParams`（`simple_ttns_l2/maxplus_pipeline.py`）。
- 源层 $L_0$ 无父，由**源分布**生成（默认 $U[\text{src\_lo},\text{src\_hi}]$；复杂实验可换双峰等）。

### 1.3 采样真值全联合

- `ground_truth_samplers(spec, params)` → `(sources, kernels)`；`dag_pipeline.sample_joint(spec, sources, kernels, key, n)` 按拓扑序逐层前向采样，返回 `[n, n_nodes]`（列 = 节点 id）。
- 复杂源示例：`compare_global_vs_layered_plot.bimodal_sources`（$0.5\,\mathcal N(0.25,\sigma)+0.5\,\mathcal N(0.85,\sigma)$）。

---

## 2. 模型：单父 TTNS 森林（每层）

> **硬约束（用户定型）**：模型侧**只用单父 TTNS**（已淘汰早期多父 core `dag_ttns.py`）；**每一层都表示为一个 TTNS 森林**。

### 2.1 单父 TTNS 与 L2 目标

- `TTNSOpt`（`TTNSDE/ttde/ttns/ttns_opt.py`）：仅含 cores，拓扑由 `parent: Sequence[int]` 传入（单父树）。
- 线性未归一化密度 $q_\theta(x)=\langle T_\theta,\ \bigotimes_k b_k(x_k)\rangle$，基为 B 样条 `SplineOnKnots`（`build_bases(x, q, m)`，`train_l2.py`）。
- L2 目标 $L(\theta)=\int q_\theta^2 - 2\,\mathbb{E}_{\text{data}}[q_\theta]$（`simple_ttns_l2/objective.py`、训练器 `experiments/fit_diamond_dag_vs_tree.train_tree_l2`，含 `grad_clip`、`train_noise`、早停、逐步积分归一化）。

### 2.2 层内分块森林

`simple_ttns_l2/layered_forest.py`：
- `layer_blocks(samples, mi_threshold)`：层内两两互信息（MI）阈值化建图 → 连通分量 = **互不相关的块**；块间近似独立 → 层密度 = 块密度乘积。
- `fit_layer_forest(...)`：每块用 chow-liu（`chow_liu.estimate_chow_liu_tree`）定树拓扑 + `train_tree_l2` 拟合 → 返回 `List[BlockModel]`（森林）。
- `forest_log_density`：层对数密度 = $\sum_\text{block}\log q_\text{block}$。
- `sample_forest`：各块用 `ttns_sampler.sample_ttns`（条件 inverse-CDF）采样后按列拼回。

### 2.3 层间条件核（分层模型的联合密度）

给定 DAG 与 max-plus 核已知，分层联合密度：
$$p(x)=p_{\text{forest}}(L_0)\cdot\prod_{l\ge1}\prod_{v\in L_l}K_v\big(x_v\mid\mathrm{pa}(v)\big)$$
- 只学源层森林 $p_{\text{forest}}(L_0)$；$K_v$ 是**已知** max-plus 条件核的闭式对数密度（`layered_forest.maxplus_cond_logdensity`）：
$$F_M(s)=\prod_{u}F_e(s-x_u),\quad K_v(y)=\frac{F_M(y-\text{node\_lo})-F_M(y-\text{node\_hi})}{\text{node\_hi}-\text{node\_lo}}.$$

---

## 3. 传播算法（把上层推进到下层）

三种"生成下一层"的方式，都要点：**下一层要么用 max-plus 现场生成样本、要么从 B 的解析统计采样，然后（如需每层是 TTNS）用 chow-liu 重拟合成森林**。

### 3.1 方案 A：采样 + max-plus + 重拟合（成链最优）

`simple_ttns_l2/maxplus_pipeline.py` + `experiments/fit_layered_forest_propagation.py`、`fit_maxplus_propagation.py`：
1. 从上层森林采样 `sample_forest`；
2. 逐样本做 max-plus `propagate_layer`（现采 $e,d$，取 max）→ 下层目标样本；
3. `fit_layer_forest` 重拟合下层森林；逐层向下。

**关键性质（"结构再生"）**：下层相关性由 max-plus 通过**共享父**现场生成，与上层森林表示是否精确无关 → 抗跨层累积。**成链场景 A 最优。**

### 3.2 方案 B：CDF 域解析收缩（单步统计最精确）

`simple_ttns_l2/maxplus_cdf.py`（单 TTNS）+ `maxplus_cdf_forest.py`（森林感知）：
- 恒等式：多源 max 在 CDF 域 = 乘积，$F_{m_v}(s)=\mathbb{E}_x[\prod_u F_e(s-x_u)]$，是对上层 TTNS 的**可分离收缩**（线性积分、无采样截断）。
- 配对联合 CDF 同理（共享父腿用 $F_e(s-x_u)F_e(t-x_u)$）；由 Hoeffding $\mathrm{Cov}=\iint[F_{vw}-F_vF_w]$ 得全相关矩阵。
- 块间独立 → 期望按块因子分解（`UpperForest`）。
- `propagate_layer_cdf_forest` 返回每层 marginal (E,Var) 与全相关矩阵。

**关键性质**：单步、给定准确上层时，B 的边缘/配对统计**比 A 更精确**（见 §6）。

### 3.3 从 B 采样成链：copula（正确做法）

`maxplus_cdf_forest.sample_layer_copula`：用 B 的**全相关矩阵 $C$ + 边缘 $F_v$** 做高斯 copula 采样 $z\sim N(0,C),\ x_v=F_v^{-1}(\Phi(z_v))$。保全部两两相关、精确边缘。
- **不要**用早期的 `sample_layer_from_cdf`（按 B 相关性建**树** + 配对 CDF 数值微分条件逆采样）：树会丢环上非树边，corr 暴涨（见 §6 负结果）。

---

## 4. 对照基线：全局扁平模型

同一份全联合数据上，把全部节点当整体拟合 L2 密度：
- **global_TT**：链式拓扑 `chain_parent(n)`（等价 pure TT）。
- **global_TTNS**：chow-liu 树拓扑 `estimate_chow_liu_tree(...).parent`。
- 采样用 `ttns_sampler.sample_ttns`。参数预算对齐见 `fit_deep_dag_vs_tree._pick_tree_rank / _count_params`。

---

## 5. 评估口径

| 指标 | 含义 | 用途 |
|---|---|---|
| `joint_loglik` | 留出集平均联合对数密度 | 越高越好；跨模型主指标 |
| `w1_marg` | 逐节点边缘 Wasserstein-1 均值 | 越低越好（采样质量） |
| `corr_fro` | 相关矩阵 Frobenius 误差 vs 真值 | 越低越好（相关结构还原） |
| `val_l2` | L2 目标 $\int q^2-2\mathbb E[q]$ | 训练监控（越低越好；勿跨参数化比） |
| 2D 切片 IAE | 边际积分绝对误差 | 单层树实验主指标（见 Program.md §4.1） |

---

## 6. 已验证结论与诚实负结果

### 6.1 分层 TTNS ≫ 全局 TT / 全局 TTNS（主结论，含图）

`experiments/compare_global_vs_layered_plot.py`。图见 `reports/global_vs_layered_*`。

| 设定 | 模型 | 学习参数 | joint_LL↑ | W1↓ | corr_fro↓ |
|---|---|---|---|---|---|
| **基础** `[4,4,4]` 均匀源 | 分层 TTNS | **64** | **7.23** | **0.0053** | **0.138** |
| | 全局 TTNS | 79488 | 3.90 | 0.0245 | 2.13 |
| | 全局 TT | 117504 | -1.35 | 0.0397 | 4.28 |
| **复杂** `[6,6,6,6]` 双峰源 | 分层 TTNS | **144** | **20.69** | **0.0052** | **0.369** |
| | 全局 TTNS | 169920 | 2.88 | 0.0377 | 7.39 |
| | 全局 TT | 191520 | 1.81 | 0.0415 | 7.44 |

分布越复杂（多峰 + 更深环形依赖），分层优势越大。全局扁平模型抹平多峰、相关矩阵几乎只剩对角线。

### 6.2 成链：方案 A > 方案 B（copula）> B（树采样，负结果）

`experiments/fit_layered_forest_chain_schemes.py`（`[4,4,4,4]`）。3 种子 best-tracking 平均 corr_fro：**A=0.242 < B-copula=0.388 < B-regen=0.755**。

- **单步**（给定准确上层，`fit_layered_forest_schemes.py`）：**B 比 A 准**（逐层平均 corr_fro A=0.085 vs B=0.050；相关上层 L1→L2 B=0.071 vs A=0.141）。L0→L1 诊断：B-解析 corr_fro=0.029 < A=0.039。
- **B 树采样是 bug 级选择**：把 B 全相关走 chow-liu 树 → 丢环上强边（(6,7) 0.35→0.056），corr_fro 0.484。改 copula 后回落到 0.059（≈解析）。
- **成链 A 仍最优**：因 A 用真实样本做 max-plus，保住 max 诱导的真依赖（非高斯、含高阶）；B-copula 只能用高斯 copula 复现两两相关。
- **regen（结构再生）失败并已回滚**：曾试"共享父+上层边缘+已知核"现算相关、忽略上层跨父相关；但上层是相关环，跨父相关重要，丢掉后反而最差（0.755）。

**定位**：**A 用于成链；B 用于单步精确统计 / 诊断。** 不要再尝试"B 采样→重拟合"去超过 A（结构上不划算）。

### 6.3 其它既有结论（单层，详见 Program.md §4.1）

- 单层拓扑用 **Chow–Liu**（数据驱动最优，vs chain 提升 56.9%/64.4%）；`init_noise=0` 时 chain TTNS ≡ pure TT；无核心算子 bug。

---

## 7. 关键文件索引

```
simple_ttns_l2/
  dag_pipeline.py                         # MultiLayerSpec / build_layered_spec / sample_joint
  maxplus_pipeline.py                     # DelayParams / ground_truth_samplers / propagate_layer(方案A)
  layered_forest.py                       # 层内MI分块 + 块内chow-liu森林 + 条件核对数密度 + sample_forest
  ttns_sampler.py                         # 单父TTNS条件inverse-CDF采样
  maxplus_cdf.py                          # 方案B：单TTNS的CDF域解析（marginal/pair/Hoeffding）
  maxplus_cdf_forest.py                   # 方案B森林版 + sample_layer_copula（成链正确采样）
  objective.py / train_l2.py              # L2目标、基、CLI
  chow_liu.py                             # 数据驱动树拓扑
  experiments/
    compare_global_vs_layered_plot.py     # ★ 全局TT vs 全局TTNS vs 分层TTNS + 图（含bimodal复杂源）
    fit_layered_vs_flat_tt.py             # 分层(源层森林+已知核) vs 扁平，联合LL
    fit_layered_forest_propagation.py     # 每层=TTNS森林 + max-plus链（方案A）
    fit_layered_forest_schemes.py         # 单步 A vs B
    fit_layered_forest_chain_schemes.py   # 成链 A vs B(copula)；含regen负结果记录
    fit_diamond_dag_vs_tree.py            # train_tree_l2（L2训练器）所在
    fit_deep_dag_vs_tree.py               # _count_params / _pick_tree_rank（参数对齐）
    plot_chain_slices.py                  # A/B链逐层边缘切片图
  reports/                                # 所有图/JSON/中文报告
TTNSDE/ttde/ttns/ttns_opt.py             # TTNSOpt + 收缩算子（改后必测，见Program.md §5）
```

---

## 8. 环境与复现命令

### 8.1 环境（重要）

- 仓库有**两个** `ttde` 包；L2 实验脚本内部已 `sys.path.insert`，运行时**必须** `env -u PYTHONPATH`，否则会导入根目录旧包。
- 建议开启 float64：脚本内已 `jax.config.update("jax_enable_x64", True)`。
- 作图依赖 `matplotlib`、`scipy`（`python3 -m pip install --user matplotlib scipy`）。

### 8.2 主要实验

```bash
# ★ 全局 TT vs 全局 TTNS vs 分层 TTNS（基础配置，出两张图）
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.compare_global_vs_layered_plot

# 复杂分布（双峰源 + 24 节点）：见脚本 main() 或用 -c 传 cfg(source_mode='bimodal', layer_sizes=[6,6,6,6])

# 成链 A vs B(copula)（[4,4,4,4]）
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.fit_layered_forest_chain_schemes

# 单步 A vs B
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.fit_layered_forest_schemes

# A/B 链逐层边缘切片图
env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.plot_chain_slices
```

### 8.3 正确性（改 `ttns_opt.py` 后必做）

```bash
export PYTHONPATH=$PWD/TTNSDE && python3 TTNSDE/validate_ttns_opt.py
env -u PYTHONPATH python3 -m unittest simple_ttns_l2/tests/test_ttns_l2_objective_unittest.py -v
```

---

## 9. 未决方向

- 纯逐层森林链（方案 A）在更大例子（30+ 维、多块层）上的多 seed 统计。
- 层内多块森林的深层稳定重拟合（深层 refit 偶发 `val_l2` 发散，可加退火/更强正则）。
- 放宽"单父"约束（双父 / copula 增强的层表示）以在**表示**层面少丢环相关——目前被硬约束限制。
- 复杂源 + 复杂核（非均匀 delay）时，分层闭式核需相应推广（当前闭式核假设均匀 $e,d$）。
- 与 UCI 真实数据的对接（若目标 DAG 结构可假定/学习）。

---

## 10. 约定

- 面向人类的回复/报告用中文；公式 `$` 定界。
- 每轮有效改动一个 commit；实验报告写入 `reports/`（中文正文 + metrics JSON）。
- **除非用户明确要求，不修改 `Program.md`**（人维护交接文档）。

---

*创建：2026-07-01 — 多层 DAG × TTNS 算法交接总纲首版（整合截至 2026-06-30 的架构、方案 A/B、copula、regen 负结果、全局 vs 分层对比与复杂分布实验）。*
