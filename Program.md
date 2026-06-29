# Program.md — TTNS 优化项目交接文档

> **仅由人类维护。** Agent 可阅读与引用，除非用户明确要求，否则不得修改本文档。里程碑、约定或开放问题变更时，由人类更新。

---

## 0. 语言与文档规则

- **所有面向人类的回复、说明、报告、交接文档必须使用中文。**
- 代码标识符（函数名、类名、路径、CLI 参数）、终端命令保持英文/原文。
- **Markdown 公式统一用 `$` 定界**：行内 `$...$`，行间 `$$...$$`；不用 `\(` `\)`、`\[` `\]`。
- 新建或更新项目文档（含 `Program.md`、实验报告 markdown）时，正文用中文；配置 JSON 中的键名可保持英文。
- 详细规则见仓库根目录 `Prompt.md` 与 `.cursor/rules/chinese-language.mdc`。

---

## 1. 研究目标

1. **用 TTNS（树结构张量网络）近似高维密度函数**，在原始 TTDE（张量列车）代码库上扩展。
2. **支持 DAG 结构的密度近似**。终局为**多层 DAG**（拓扑分层、层间传播）；当前代码仍以**单层树**为主（chain / balanced / Chow–Liu），联结树为过渡 MVP。
3. **每次迭代必须先验证 TTNS 正确性**，再信任新功能、重构或性能改动。见 §5。

### 1.1 模型架构（目标定义，2026-06）

**单层（层内）**——拓扑排序后同一层 $L_k$ 内的变量，**层内无父子边**；用一棵树连接层内变量：

| 拓扑 | 说明 |
|------|------|
| **Chow–Liu** | 从数据估计互信息，取最大生成树；**树类中数据驱动的最优选择** |
| **balanced** | 固定堆式平衡树 |
| **chain** | 固定链；等价于 pure TT |

**多层（层间）**——对 DAG 拓扑排序得 $L_1, L_2, \ldots$：

- **层内**无父子；**父子边只出现在相邻层之间**。
- 传播 **从 $L_1$ 向下**（$L_1 \to L_2 \to \cdots$），配合 **node delay** / **edge delay**（设计概念，**代码尚未实现**）。

**当前实现与目标的差距：**

| 目标 | 现状 |
|------|------|
| 单层三拓扑 CL / balanced / chain | **三者均已系统实验**（M2.2：Chow–Liu vs chain **56.9%**） |
| 多层 + 层间向下传播 | **未实现**；`ttns_opt.py` 为**整图一棵 `parent[]`、叶→根后序** |
| 联结树 junction | **过渡方案**：手工 spanning tree 近似 fork DAG（`junction_tree.py`） |

### 当前活跃问题（2026-06）

> 在多层 TTNS 进行传播时，计算缓慢，结果效果差，怀疑有 bug。

对 Agent 的解读：

- **「多层传播」**：`TTNSDE/ttde/ttns/ttns_opt.py` 中自底向上的 `node_message` 递归（叶 → 根）。深度随树高增加（例如 6 维平衡树深度约 3；单元测试用 9 节点树深度约 4）。
- **慢**：平衡/高分支树比链结构更贵；`normalized_*` 算子在每个节点做 log 归一化；子节点数 **≥3** 时曾回退到通用 `tensordot`（**2026-06 已补 3-child 融合 einsum，待提交**）。
- **效果差**：可能是**拓扑不匹配**、初始化弱或优化问题，不一定是算子 bug。已有诊断（§6）表明：在 9 节点深树上核心数学与 dense 参考一致；`init_noise=0` 时 chain-TTNS ≡ pure-TT。
- **疑似 bug**：视为**未证实**。第一步始终重跑 §5 验证；再分析传播深度/分支；再检查拓扑对齐与初始化协议。

---

## 2. 仓库结构

| 路径 | 作用 |
|------|------|
| `ttde/` | 原始 TTDE 论文代码（TT / MLE），基本未改 |
| **`TTNSDE/ttde/`** | **TTNS 主开发目录**。新算子、模型及 `train.py --model-type ttns` 均在此 |
| `simple_ttns_l2/` | 独立的 **L2 目标**训练与实验脚本，通过 `sys.path` 引用 `TTNSDE/ttde` |
| `TTNSDE/test/` | TTNS 数学与烟雾测试 |
| `simple_ttns_l2/tests/` | L2 目标测试 |
| `simple_ttns_l2/reports/` | 实验输出（metrics JSON、SVG、中文/英文报告） |
| `TTNSDE/TTNS_TRAINING_PROGRESS.md` | 早期迁移笔记（部分已被本文档取代） |
| `Prompt.md` | 项目级提示与规则（人类维护） |

### 关键导入规则

仓库内存在**两个** `ttde` 包（根目录 `ttde/` 与 `TTNSDE/ttde/`）。TTNS 工作**必须**使用：

```bash
export PYTHONPATH=/path/to/ttnsOptimization/TTNSDE
```

或沿用 `simple_ttns_l2/` 中的 `sys.path.insert` 写法。导错包会静默运行过时代码。

### 依赖

见 `pyproject.toml`：Python ≥3.9，`jax==0.4.8`，`flax`，`optax`，`opt-einsum` 等。参考 conda 环境：`/home/sbzeng/.conda/envs/ttns`。数值检查建议开启 float64：`jax.config.update("jax_enable_x64", True)`。

---

## 3. 当前 TTNS 模型

### 3.1 数学形式

线性未归一化密度（L2 路径）：

$$
q_\theta(x) = \langle T_\theta,\; \bigotimes_k b_k(x_k) \rangle
$$

MLE 路径（`PAsTTNSSqrOpt`）采用**平方**参数化 $q = \tilde{q}^2$ 与对数似然，与原始 TTDE 一致。

L2 训练目标（`simple_ttns_l2`）：

$$
L(\theta) = \int q_\theta(x)^2 \, dx - 2\,\mathbb{E}_{\text{data}}[q_\theta(x)] + \text{常数项}
$$

训练后可选**积分归一化**（缩放根 core），由 `--normalize-every` 控制。

### 3.2 数据结构

**`TTNSOpt`**（`TTNSDE/ttde/ttns/ttns_opt.py`）— 仅含 cores 的 Flax pytree；拓扑单独传入 `parent: Sequence[int]`。

Core $G_u$ 形状：

```
(r_parent, dim_u, r_child1, r_child2, ...)
```

- `parent[u]` = 父节点编号；`parent[root] == root`（或根为 `-1`）。
- 子节点顺序 = 满足 `parent[v] == u` 的节点按**编号升序**（见 `_children_from_parent`）。
- 默认各边统一秩 `r`；扩展见 `ttde/tt/tensors.py` 中 `TTNS.zeros(..., rank_spec=...)`。

Dense 参考收缩：**`TTNS.full_tensor`**（`TTNSDE/ttde/tt/tensors.py`），用于测试中与暴力计算对照。

### 3.3 支持的拓扑

| 名称 | `parent` 构造 | 说明 |
|------|--------------|------|
| `chain` | `chain_parent(n)` | 等价于 TT 链；支持 canonical/EM 初始化 |
| `balanced` | `balanced_parent(n)` | 堆式平衡二叉树；仅 rank-1 初始化 |
| Chow–Liu | `simple_ttns_l2/chow_liu.py` | 离散化数据上两两互信息的最大生成树 |
| 随机树 | `random_tree_target.py` | 合成实验用类 DAG 目标 |

辅助函数：`simple_ttns_l2/train_l2.py` 的 `make_parent`；`opt_for_tree_data.py` 的 `balanced_parent` / `chain_parent`。

### 3.4 核心算子（`ttns_opt.py`）

| 函数 | 用途 |
|------|------|
| `normalized_inner_product_ttns` | $\langle T_1, T_2 \rangle$，带 log 范数 |
| `normalized_eval_rank1_ttns` | 单点 $\langle T, \otimes_k v_k \rangle$ |
| `normalized_quadratic_form_ttns` | 可分离局部 Gram 矩阵的 $\langle T, A T \rangle$ |
| `eval_rank1_ttns` / `quadratic_form_ttns` | 无逐节点归一化的快速路径 |
| `batch_eval_rank1_ttns` | 批量 rank-1 评估（MC 期望） |
| `add_ttns` / `subtract_ttns` | 增秩合并 |

**传播模式**（所有求值器）：递归 `node_message(node)` 先访问子节点再在本节点收缩。`normalized_*` 变体在每个节点调用 `_lognorm_and_normalized` — 数学正确，但在深树上开销大。

### 3.5 训练入口

**MLE（原 TTDE 训练器）：**

```bash
PYTHONPATH=TTNSDE python -m ttde.train \
  --dataset power --q 2 --m 64 --rank 16 --n-comps 8 \
  --model-type ttns --ttns-topology balanced \
  --em-steps 5 --noise 0.01 --batch-sz 512 --train-noise 0.01 \
  --lr 0.001 --train-steps 1000 \
  --data-dir /path/to/data --work-dir /path/to/workdir
```

模型：`TTNSDE/ttde/score/models/opt_for_tree_data.py` 中的 `PAsTTNSSqrOpt`。

**L2（独立路径）：**

```bash
python -m simple_ttns_l2.train_l2 \
  --dataset Power --q 2 --m 64 --rank 16 \
  --ttns-topology balanced --init-noise 0.01 \
  --batch-sz 512 --train-noise 0.01 --lr 0.001 --train-steps 1000 \
  --normalize-every 1 \
  --data-dir /path/to/data --work-dir /path/to/workdir
```

---

## 4. 进度快照

### 4.1 实验结果快照（截至 2026-06-27，3 seed：313 / 2602 / 20260227）

主指标：**关键切片 mean IAE**；提升 $= (\text{IAE}_\text{chain} - \text{IAE}_\text{model}) / \text{IAE}_\text{chain}$。  
配置共性：6D/7D 合成目标，`init_noise=0.01`，`rank=16`，L2 训练。详见 `simple_ttns_l2/reports/*_multiseed_report_zh.md`。

| 实验 | 对比 | 聚合提升 | 达标（≥20%） | 报告 |
|------|------|----------|:---:|------|
| **M1.1** complex balanced 目标 | balanced TTNS vs chain | **49.7%** | ✅ | `fit_ceiling_two_models_complex_multiseed_report_zh.md` |
| **M1.2** 随机递归树目标 | matched TTNS vs chain | **42.6%** | ✅ | `random_tree_matched_vs_chain_multiseed_report_zh.md` |
| **M3** 7D fork DAG 目标 | junction TTNS vs chain | **26.9%** | ✅ | `dag_junction_vs_chain_multiseed_report_zh.md` |
| M3 同实验 | balanced TTNS vs chain | **50.1%**（单 seed 20260227） | ✅ | 同上 |
| **M2.2** 7D fork DAG 目标 | **Chow–Liu** TTNS vs chain | **56.9%** | ✅ | `dag_chow_liu_vs_chain_multiseed_report_zh.md` |
| **M2.2-8D** 8D fork DAG 目标 | **Chow–Liu** TTNS vs chain | **64.4%** | ✅ | `dag_chow_liu_vs_chain_8d_multiseed_report_zh.md` |
| **Sanity** `init_noise=0` | chain TTNS ≡ pure TT | 轨迹与 IAE **完全一致** | ✅ | `fit_three_models_init0_report_zh.md` |

**M2.2 要点（数据驱动 Chow–Liu，7D fork DAG）：**

- **无需手工拓扑**：从训练样本估互信息最大生成树，每 seed 自动**命中 6/7 条 DAG 真边**（含关键 $(0,2),(1,2),(2,6)$）。
- 关键切片 chow_liu vs chain：seed 313 **55.9%**、2602 **64.6%**、20260227 **50.1%**，**聚合 56.9%**（**远超手工 junction 的 26.9%**）。
- `val_l2` 三模型最优（-1.024 vs balanced -0.940 vs chain -0.310）；多数 seed IAE 也优于 balanced。
- **代价**：Chow–Liu 同样慢（~70 s），因数据驱动同样把 $x_2$ 选为高扇出 hub（挂 3 子节点）——印证 junction 的「慢」是 fork DAG 最优树的**结构内在属性**，非算子 bug。
- **维度趋势（7D→8D）**：Chow–Liu vs chain 从 56.9% 升到 **64.4%**；且 8D 下 **Chow–Liu 大幅碾压 balanced**（关键切片 IAE 0.25 vs 0.62，7D 时两者接近）——**固定拓扑在高维越来越失配，数据驱动的价值随维度放大**。每 seed 命中 7/8 DAG 真边。
- **结论**：单层应优先用 Chow–Liu，淘汰手工 junction。

**加速（hub 边降 rank，单 seed，`chow_liu_hub_rank_sweep_report_zh.md`）：**

速度瓶颈 $\propto \text{rank}^{(\text{fanout}+1)}$。只对 hub 节点 2 的 4 条边降 rank（非 hub 边保持 16）：

| hub_rank | ms/step | 提速 | val_l2 | 关键切片 IAE |
|---------:|--------:|-----:|-------:|-------------:|
| 16（原） | 180 | 1.0× | -0.984 | 0.260 |
| **12** | 52 | **3.5×** | -0.958 | 0.274 |
| 8 | 15 | **12×** | -0.804 | 0.386 |
| 4 | 4.2 | 43× | -0.742 | 0.432 |
| 2 | 2.7 | 68× | -0.444 | 0.598 |

- **甜点 hub_rank=12**：几乎免费 **3.5×**（IAE 0.274 vs 0.260）。
- hub_rank=8：**12×** 提速，IAE 0.386 仍优于 chain（~0.55），接近 balanced（0.347）。
- 支持按边 rank：`TTNSOpt.from_rank1_vectors(..., edge_ranks=)` → `init_ttns_from_rank1` → `_train_one_model(model_edge_ranks=)`。

**M3 要点（7D fork DAG，seed 20260227）：**

- 关键切片 IAE：junction **0.49** vs balanced **0.35** vs chain **0.70**（越低越好）。
- 双父切片 **(2,6)**：junction 0.69 vs chain **1.14**（优势最大）。
- **训练总耗时**（seed 20260227，3-child einsum 优化后 / 优化前）：

| 模型 | 优化后 | 步数 | 约每步 | 优化前 | 加速 |
|------|-------:|-----:|-------:|-------:|-----:|
| junction | **43 s** | 230 | ~187 ms | 136 s | **3.2×** |
| balanced | 18 s | 400 | ~44 ms | 9.7 s | — |
| chain | 2.9 s | 160 | ~18 ms | 1.6 s | — |

- junction 热启动后约 **3.2×** 加速（冷启动首次约 **1.7×**）；精度仍 **26.9%**。详见 `dag_junction_timing_post_einsum_report_zh.md`。

**M1.1 / M1.2 训练耗时（seed 20260227，6D）：**

| 实验 | 较快模型 | 较慢模型 | 倍率 |
|------|--------:|--------:|-----:|
| M1.1 balanced vs chain | chain 2.4 s / 260 步 | balanced 2.3 s / 100 步（早停） | ≈1× |
| M1.2 matched vs chain | chain 3.3 s / 400 步 | matched 6.8 s / 360 步 | **2.1×** |

6D balanced 典型稳态：**~16–20 ms/步**（每 10 步 log 含 train+val L2）；6D chain：**~8 ms/步**。

**算子正确性（`validate_ttns_opt.py`，2026-06-29）：**

| 算子 | 绝对误差（9 节点树） |
|------|---------------------|
| inner_product | $\sim 10^{-16}$ |
| rank1_eval | $\sim 10^{-16}$ |
| quadratic_form | $\sim 10^{-10}$ |

另含 **4 节点 3-child 扇出树** dense 对照（6/6 PASS）。

**尚未系统完成：**

- 多层拓扑分层 + node/edge delay 传播（终局架构）

### 已完成

- [x] TTNS 核心算子：内积、rank-1 求值、二次型（含快速变体）
- [x] **9 节点深树**上的 dense 对照单元测试（`parent = [0,0,0,1,1,2,2,4,6]`）
- [x] `train.py` 接入 TTNS（`--model-type ttns`，`--ttns-topology chain|balanced`）
- [x] `PAsTTNSSqrOpt` MLE 模型（混合分量 + 样条基）
- [x] `simple_ttns_l2` L2 训练循环、归一化、早停
- [x] 拓扑对比实验（chain vs balanced 目标/模型）
- [x] 对齐初始化下 chain TTNS ≡ pure TT（`init_noise=0`）
- [x] 快速 vs 稳定求值器等价性（`stable_fast_balanced_report_zh.md`）
- [x] Phase 2：M1.1 / M1.2 / M3 多 seed 达标（见 §4.1）
- [x] fork DAG 合成目标 + 联结树 MVP（`dag_target.py`、`junction_tree.py`）
- [x] **M2.2：数据驱动 Chow–Liu TTNS 系统实验**（vs chain **56.9%**，淘汰手工 junction）
- [x] `simple_ttns_l2/reports/` 下实验报告与 metrics JSON

### 未完成 / 部分完成

- [ ] 非链拓扑的 TTNS **canonical / EM 初始化**（balanced 仅 rank-1 引导）
- [ ] **多层 DAG**（拓扑分层、层间向下传播、node/edge delay）— 见 §1.1
- [x] M3 用 3-child einsum 修复后重跑 multiseed（junction **3.2×**，IAE 仍 **26.9%**）
- [ ] TTNS MLE 在真实 UCI 数据上的多 seed 系统基准
- [ ] 旧笔记中的 `TTNSDE/scripts/smoke_train_ttns.py` — 若缺失需重建

---

## 5. 必做：正确性验证

**修改 `ttns_opt.py`、拓扑辅助函数或目标函数组装前后，必须运行以下检查。**

### 5.1 TTNS 数学单元测试

```bash
PYTHONPATH=TTNSDE python -m unittest TTNSDE/test/test_ttns_opt_unittest.py -v
```

覆盖：内积、rank-1 求值、二次型、加减、链情形、5 步 MLE 烟雾测试。

### 5.2 独立 dense 验证脚本

```bash
PYTHONPATH=TTNSDE python TTNSDE/validate_ttns_opt.py
```

### 5.3 L2 目标测试

```bash
python -m unittest simple_ttns_l2/tests/test_ttns_l2_objective_unittest.py -v
python -m unittest simple_ttns_l2/tests/test_train_l2_smoke_unittest.py -v
```

### 5.4 跨参数化对照（仅 chain）

改动 chain 相关代码时，确认 chain TTNS 与 `TTOpt` 在以下方面一致：

- 相同参数下的 `q(x)`、`∫q`、`∫q²`
- `init_noise=0`、`train_noise=0` 时的训练轨迹

见 `simple_ttns_l2/reports/chain_ttns_vs_tt_diagnosis.md`。

### 5.5 「通过」标准

- 与 dense 的绝对/相对误差 ≤ `1e-8`（病态随机矩阵上二次型可放宽至 `1e-5`）
- 形状：`vectors` 为 `[n_dims, basis_dim]`；`matrices` 为 `[n_dims, m, m]`
- 新拓扑：训练前先用 `TTNS.full_tensor` 增加 dense 往返测试

---

## 6. 已知结论（不一定是 bug）

| 现象 | 可能原因 | 参考 |
|------|----------|------|
| 默认噪声下 chain TTNS ≠ pure TT | `init_noise` 按 core 注入方式不等价 | `chain_ttns_vs_tt_diagnosis.md` |
| chain 目标 + balanced 模型 → val_l2 ≈ 0 | 拓扑不匹配，模型无法表达目标依赖 | `topology_comparison_complex_report.md` |
| balanced 训练比 chain 慢 | 树更深、双子女节点、normalized 路径 | `topology_comparison_*.md`，`stable_fast_balanced_report_zh.md` |
| fast vs stable 差异仅 ~1e-13 | 浮点重排，非逻辑错误 | `stable_fast_balanced_report_zh.md` |
| MLE 与 L2 不可直接比 | 参数化不同（平方 vs 线性） | `compare_original_ttns_vs_l2_*_report_zh.md` |

---

## 7. 待 Agent 推进的工作

### 7.1 诊断多层传播缓慢

1. 在 balanced 6D、rank 16、basis 64 上对比 `normalized_*` 与快速 `eval_*` / `quadratic_form_*` 耗时。
2. 统计每步训练中 `node_message` 调用次数与张量形状。
3. 检查训练是否不必要地走稳定路径（MLE 的 `log_p` / `log_int_p` 需要 normalized 算子）。
4. 考虑：显式后序遍历的迭代调度，替代嵌套 Python 递归，以利于 JAX 缓存。

### 7.2 诊断深树 / 平衡树拟合差

1. 确认**目标拓扑与模型 拓扑**是否对齐。
2. 试 `init_noise=0`，排除初始化假象导致的「假 bug」。
3. 增大 rank / 步数；检查积分归一化（`final_integral` 应 ≈ 1）。
4. 在同一目标上用切片 IAE 比较 L2 与 MLE（`topology_slice_visualization.py`）。

### 7.3 若确认存在真实 bug

- 在 `TTNSDE/test/test_ttns_opt_unittest.py` 增加**最小可失败测试**（小树、显式期望值）。
- 在 `ttns_opt.py` 修复；除非用户要求，**不要**改 `Program.md`。
- 重跑 §5 全套测试。

### 7.4 面向多层 DAG 目标

- **单层**：Chow–Liu / balanced / chain 三拓扑对比（M2.2）；随机树 matched vs chain 已做（M1.2）。
- **过渡**：联结树 spanning tree 近似 fork DAG（M3 已完成，junction vs chain **+26.9%**）。
- **终局**：DAG 拓扑排序分层；层内无父子、层间向下传播；**node delay / edge delay**（待设计与实现）。

---

## 8. 关键文件索引

```
TTNSDE/ttde/ttns/ttns_opt.py          # TTNSOpt + 所有收缩算子
TTNSDE/ttde/tt/tensors.py             # TTNS dense 参考 + validate_tree
TTNSDE/ttde/score/models/opt_for_tree_data.py  # PAsTTNSSqrOpt (MLE)
TTNSDE/ttde/train.py                  # --model-type ttns
TTNSDE/test/test_ttns_opt_unittest.py
TTNSDE/validate_ttns_opt.py
simple_ttns_l2/objective.py           # L2 积分与求值封装
simple_ttns_l2/train_l2.py            # L2 训练 CLI
simple_ttns_l2/chow_liu.py            # 数据驱动树拓扑
simple_ttns_l2/experiments/           # 基准实验脚本
```

---

## 9. 实验约定

- **公平对比**：除非研究初始化敏感性，默认 `init_noise=0`；随机性结论至少报告 ≥3 个 seed 的均值/方差。
- **主指标**：L2 路径 → `val_l2`（越低越好）；MLE 路径 → `val_nll`。跨模型对比 → 二维边际 IAE（切片对）。
- **配置存档**：`simple_ttns_l2/reports/*_metrics.json` 与 markdown 报告头部。
- **勿提交**：大数据集、wandb 密钥、`workdir` 检查点。

---

## 10. 原始 TTDE 基线

根目录 `README.md` 记录张量列车密度估计（Novikov 等，UAI 2021）。TTNS 将权重张量从链（TT）推广到树。原论文复现使用 `python -m ttde.train`，默认 `--model-type tt`。

---

*最后更新：2026-06-27 — 增加 §1.1 架构定义、§4.1 实验结果快照。*
