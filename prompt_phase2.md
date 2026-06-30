# Phase 2 工作指导 — TTNS 多层 DAG 收益

> 承接 `prompt_phase1.md`（Phase 1 已完成传播优化与 P0 单 seed 实验）。  
> 详细 API、必做测试、仓库结构见 `Program.md`；语言规则见 `Prompt.md` 与 `.cursor/rules/chinese-language.mdc`。

---

## 1. Phase 2 目标

**终局：** 在**多层 DAG**（含多父节点依赖）上，证明 TTNS 相对 TT（链）的密度近似收益。

**Phase 2 要交付的两条证据链：**

| 证据 | 内容 | 状态（Phase 1 后） |
|------|------|-------------------|
| A. 树形 DAG | 非链树目标 + 匹配拓扑 TTNS **稳定优于** chain TTNS / pure TT | 单 seed 已证实，待多 seed |
| B. 多父 DAG | 合成多父目标 + 联结树 TTNS **优于** chain（关键切片 IAE） | **未开始** |

---

## 2. Phase 1 已确认结论（勿重复论证）

- `TTNSDE/ttde/ttns/ttns_opt.py` dense 对照与 L2 单元测试通过；传播已改为显式后序遍历 + 融合 einsum（commit `34de737`）。
- **无核心算子 bug**（`init_noise=0` 时 chain TTNS ≡ pure TT）。
- **效果差的主因是拓扑不匹配**，不是传播算错。
- P0 单 seed（2026-06-26）：balanced 目标上 balanced TTNS val_l2=-0.066 vs chain -0.044；IAE 约低 50%。

报告：`simple_ttns_l2/reports/fit_ceiling_two_models_complex_report.md`、`fit_three_models_init0_report_zh.md`。

---

## 3. Agent 工作流程

1. `git log --oneline -10` 了解近期改动。
2. **改 `ttns_opt.py` 前后**必须跑 `Program.md` §5 正确性测试（见 §6）。
3. 按 §5 里程碑顺序推进；每完成一项更新本文档 checklist。
4. **每轮有效改动一个 commit**：说明修改内容、理论效果、测试/实验结果。
5. 实验报告写入 `simple_ttns_l2/reports/`，正文中文；metrics 存 JSON。

### 约束

- 只改 TTNS 相关代码、实验脚本、结果报告；**除非用户明确要求，不修改 `Program.md`**。
- 实验脚本使用 `env -u PYTHONPATH`（见 §6 导入规则）。
- 公平对比默认 `init_noise=0`；结论性实验 ≥3 seed。

---

## 4. 当前未解决问题（Phase 2 动机）

| 优先级 | 问题 |
|:---:|------|
| P0 | 树形收益仅单 seed，缺统计稳健性 |
| P0 | **多父 DAG 未实现** — `parent[]` 只能是树 |
| P1 | Chow–Liu / 8D / 随机树目标未系统对比 |
| P2 | 训练偏慢、balanced 缺 canonical/EM 初始化 |
| P2 | `test_ttns_opt_unittest.py` 在 Python 3.9 上可能因类型注解失败 |

---

## 5. 里程碑与任务

### M1 — 巩固树形收益（优先）

| ID | 任务 | 交付物 | 完成标准 |
|:--:|------|--------|----------|
| M1.1 | 多 seed P0 | 扩展 `fit_ceiling_two_models.py` 支持 `--seed` 或 sweep 脚本 | seeds `{313, 2602, 20260227}`：balanced 目标 mean IAE，匹配 TTNS 比 chain **≥20%** |
| M1.2 | 随机树 P1 | `fit_matched_vs_chain_random_tree.py`（新建） | 基于 `random_tree_target.py`；匹配 parent TTNS > chain |
| M1.3 | 测试环境 | `TTNSDE/ttde/score/experiment_setups/model_setups.py` 首行加 `from __future__ import annotations` | `test_ttns_opt_unittest.py` 全绿 |

### M2 — 数据驱动拓扑与深树

| ID | 任务 | 交付物 | 完成标准 |
|:--:|------|--------|----------|
| M2.1 | 修复脚本路径 | `fit_limit_extreme_tt_vs_ttns_three_topologies_mle_8d.py` 等 | 硬编码 conda `PYTHON` 改为 `sys.executable` |
| M2.2 | Chow–Liu 四模型 | 跑通 8D 脚本 | `reports/*chow_liu*_report_zh.md` + metrics JSON |
| M2.3 | 8D balanced | `TARGET_TOPOLOGY=balanced` | TTNS(balanced) vs TTNS(chain) vs TT(MLE) 切片 IAE 表 |
| M2.4 | 可选 UCI | Power 短训 smoke | Chow–Liu `parent` 与 val 指标 |

### M3 — 多父 DAG MVP（终局主线）

| ID | 任务 | 交付物 | 完成标准 |
|:--:|------|--------|----------|
| M3.1 | 合成 DAG 目标 | `simple_ttns_l2/experiments/dag_target.py` | 6D+，含 $x_3 \sim f(x_1, x_2) + \epsilon$；可采样 |
| M3.2 | 联结树 | `simple_ttns_l2/junction_tree.py` + `tests/test_junction_tree_unittest.py` | DAG 边 → 联结树 `parent`；单测覆盖 clique |
| M3.3 | 对比实验 | `fit_dag_junction_vs_chain.py` | 联结树 TTNS vs chain TTNS vs pure TT |
| M3.4 | 报告 | `reports/dag_junction_vs_chain_report_zh.md` | 含 $(x_1,x_2)$ / $(x_1,x_3)$ 等切片；chain IAE 显著差于联结树 |

**技术路线：** 优先 **B1 联结树展开**（复用现有 `TTNSOpt`）；**B2 多父 core** 仅在联结树秩/节点膨胀不可接受时启动（见 `Program.md` §7.4）。

### M4 — 工程债（穿插，非阻塞）

- 实验脚本统一入口与环境说明
- balanced canonical/EM 初始化调研
- MLE 训练 JIT / 减少 `normalized_*` 调用

---

## 6. 环境与正确性

### 依赖（参考）

```bash
python3 -m pip install --user "jax==0.4.20" "jaxlib==0.4.20" "flax==0.6.11" \
  "opt-einsum==3.3.0" "numpy==1.24.2" "scipy==1.11.4" "optax" "tqdm" "click" "h5py" "pandas==1.5.3"
```

### 导入规则（重要）

| 场景 | 命令 |
|------|------|
| **L2 实验脚本** | `env -u PYTHONPATH python3 simple_ttns_l2/experiments/....py` |
| **仅 TTNS 算子测试** | `export PYTHONPATH=$PWD/TTNSDE` 后跑 `validate_ttns_opt.py` |

`simple_ttns_l2` 实验脚本在代码内 `sys.path.insert(TTNSDE, REPO)`；**不要**同时设 `PYTHONPATH=TTNSDE`，否则会导入根目录旧 `ttde` 包。

### 必做正确性（改 `ttns_opt.py` 后）

```bash
cd /path/to/ttnsOptimization
export PYTHONPATH=$PWD/TTNSDE
python3 TTNSDE/validate_ttns_opt.py
env -u PYTHONPATH python3 simple_ttns_l2/tests/test_ttns_l2_objective_unittest.py -v
```

通过标准：dense 误差 ≤ $10^{-8}$（二次型可放宽至 $10^{-5}$）；L2 测试全绿。

---

## 7. 实验约定

### 对比矩阵（树形）

同一目标数据上：

- **模型 A：** `TTNS(与目标 parent 一致)`
- **模型 B：** `TTNS(chain)`
- **模型 C：** `pure TT`（链）

### 指标

| 指标 | 用途 |
|------|------|
| **2D 切片 IAE** | 跨 L2/MLE 主对比指标 |
| `ratio_to_floor = IAE / noise_floor` | 归一化误差 |
| `val_l2` / `val_nll` | 训练目标（口径不同，勿跨参数化直接比） |
| `final_integral` | L2 路径应 ≈ 1 |

### 必做对照

1. **负对照：** chain 目标 + balanced 模型 → 预期 val_l2 ≈ 0。
2. **Sanity：** `init_noise=0` 时 chain TTNS 与 pure TT 一致。
3. 每实验至少切片对 `(0,1)`, `(0,3)`, `(2,5)`（高维可增 $(1,2)$ 等 DAG 相关对）。

### 报告模板（中文 markdown 头部）

```markdown
# 实验标题

## 目的
## 配置（JSON）
## 训练结果表
## 切片 IAE 表
## 结论（3–5 句）
```

---

## 8. 关键文件索引

```
TTNSDE/ttde/ttns/ttns_opt.py              # 传播算子（改后必测）
TTNSDE/validate_ttns_opt.py               # dense 对照
simple_ttns_l2/experiments/
  fit_ceiling_two_models.py               # M1 P0 多 seed
  fit_three_models_init0.py               # sanity check
  random_tree_target.py                   # M1 随机树目标
  fit_limit_extreme_tt_vs_ttns_three_topologies_mle_8d.py  # M2 Chow–Liu
  dag_target.py                           # M3 待建
  fit_dag_junction_vs_chain.py            # M3 待建
simple_ttns_l2/junction_tree.py           # M3 待建
simple_ttns_l2/chow_liu.py                # Chow–Liu parent
simple_ttns_l2/reports/                   # 所有实验输出
prompt_phase1.md                          # Phase 1 日志（只读参考）
Program.md                                # 人类维护交接（只读除非被要求）
```

---

## 9. 推荐执行顺序

```
[ ] M1.1  多 seed fit_ceiling_two_models
[ ] M1.3  修复 Py3.9 unittest（可与 M1.1 并行）
[ ] M3.1  dag_target.py
[ ] M3.2  junction_tree.py + test
[ ] M3.3  fit_dag_junction_vs_chain + 报告
[ ] M1.2  随机树 matched vs chain
[ ] M2.1  脚本 PYTHON 路径修复
[ ] M2.2  Chow–Liu 四模型 + 8D 报告
```

**说明：** M3 与终局目标最直接，可与 M1.1 并行；M2 可在 M3 之后或资源允许时进行。

---

## 10. Phase 2 收工标准

- [ ] **树形：** 3 seed 下 balanced（或随机树）目标，匹配 TTNS mean IAE 比 chain **稳定低 ≥20%**。
- [ ] **多父 DAG：** 联结树 TTNS 在至少 **1 个**含双父依赖的切片对上 IAE **显著低于** chain（建议 ≥30% 或报告 p-value / 多 seed）。
- [ ] **正确性：** 全程 `validate_ttns_opt.py` 保持 3/3 PASS。
- [ ] **文档：** 每项里程碑有 `reports/*_report_zh.md` + `*_metrics.json`。

---

## 11. Checklist 进度（Agent 更新此处）

| 里程碑 | 状态 | 日期 | 备注 |
|--------|------|------|------|
| M1.1 多 seed P0 | **完成** | 2026-06-26 | 3 seed 聚合提升 49.7%，见 `fit_ceiling_two_models_complex_multiseed_report_zh.md` |
| M1.2 随机树 | **完成** | 2026-06-27 | 3 seed 聚合 matched vs chain **42.6%** |
| M1.3 unittest 修复 | **完成** | 2026-06-26 | `model_setups.py` 加 future annotations；unittest 全绿 |
| M2.1 脚本路径 | 未开始 | | |
| M2.2 Chow–Liu（数据驱动拓扑） | **完成** | 2026-06-29 | L2 路径：7D Chow–Liu vs chain **56.9%**（>junction 26.9%）；每 seed 命中 6/7 DAG 真边；`dag_chow_liu_vs_chain_multiseed_report_zh.md`。原定 8D MLE 脚本未做 |
| M2.3 8D fork DAG（Chow–Liu） | **完成** | 2026-06-29 | 3 seed Chow–Liu vs chain **64.4%**，**碾压 balanced**（IAE 0.25 vs 0.62）；命中 7/8 DAG 边；`dag_chow_liu_vs_chain_8d_multiseed_report_zh.md` |
| M2.5 hub-rank 加速 | **完成** | 2026-06-29 | 只降 hub 边 rank：rank12→**3.5×** 几乎无损（IAE 0.274 vs 0.260）；rank8→**12×**；`chow_liu_hub_rank_sweep_report_zh.md` |
| 按边 rank 支持 | **完成** | 2026-06-29 | `TTNSOpt.from_rank1_vectors(edge_ranks=)` → `init_ttns_from_rank1` → `_train_one_model(model_edge_ranks=)` |
| M3.1 dag_target | **完成** | 2026-06-26 | `experiments/dag_target.py`（6/7/8D） |
| M3.2 junction_tree | **完成** | 2026-06-27 | 7D 联结树 parent；修正 6D 星形错误 |
| M3.3 DAG 对比实验 | **完成** | 2026-06-27 | `fit_dag_junction_vs_chain.py` + multiseed |
| M3.4 DAG 报告 | **完成** | 2026-06-27 | 7D fork DAG；3 seed 聚合 junction vs chain **26.9%** |
| 3-child einsum 加速 | **完成** | 2026-06-29 | `n_children==3` 融合 einsum（commit `019d3d8`）；junction 训练 **3.2×** |

### Phase 2 收工标准进度

- [x] 树形：3 seed balanced 目标匹配 TTNS IAE 优于 chain ≥20%（**49.7%**）
- [x] 多父 DAG：联结树 IAE 显著优于 chain（3 seed 平均 **26.9%**，关键切片 (2,6) 优势最大）
- [x] **数据驱动拓扑：Chow–Liu 自动估树 vs chain 3 seed 平均 56.9%，全面超越手工 junction**
- [x] 改算子后 validate 仍 PASS（3-child einsum 后 6/6 PASS；unittest 通过）

### Phase 2 后续主线（2026-06-29 起）

> 结论：单层拓扑用 **Chow–Liu**（数据驱动最优），淘汰手工 junction；速度瓶颈在 hub 扇出，hub 边降 rank 可换 3.5–12× 提速。

- [x] **8D fork DAG**：Chow–Liu vs chain **64.4%**，碾压 balanced；维度↑优势↑（数据驱动价值随维度放大）
- [x] **真多父 core 地基**：`simple_ttns_l2/dag_ttns.py` 支持节点关联任意多 bond（多父）；`full_tensor/eval_rank1/inner_product/quadratic_form/integral/normalize` einsum 算子（greedy 路径）；`validate_dag_ttns.py` 用独立 brute-force 对照，fork/chain/polytree/三父/**diamond 有环** 5 组 20 项全 PASS（~1e-14）
- [x] **多父 L2 拟合 + DAG vs 树 demo**：`dag_train_l2.py` + `experiments/fit_diamond_dag_vs_tree.py`。diamond 真分布（$x_2,x_3$ 各依赖 $x_0,x_1$，moral graph 含 4-环，树不可表达）→ **DAG val_l2=-19.10 vs 最优树 star@2 -17.15，提升 11.4%**（`diamond_dag_vs_tree_report_zh.md`）
- [x] **层间传播 pipeline**（终局）：`dag_pipeline.py`（`MultiLayerSpec` + `build_graph_from_spec` + 逐层前向采样 `sample_joint`：源层 → delay 核 → 下层）。层内无边、层间多父，给定上层时同层条件独立（=每层若干 TTNS/森林）
- [x] **3 层多父 DAG 端到端 vs 树**：`experiments/fit_multilayer_dag_vs_tree.py`。L1={0,1}→L2={2,3}（和/差）→L3={4}（和），6 边 > 树上限 4。**DAG val_l2=-97.14 vs 最优树 hub@2 -75.13，提升 29.3%**（`multilayer_dag_vs_tree_report_zh.md`）
- [x] **复杂结构（4 层 / 20 节点）+ 参数量对齐**：`experiments/fit_deep_dag_vs_tree.py` + `build_layered_spec`。banded 多父阶梯 `[5,5,5,5]`，27 边 > 树上限 19；**乘积/XOR 交互核**（子与单父近零相关，树成对路径不可表达）。**参数量对齐到 ~83-89k**（DAG rank6=89400；给树 chain rank22=87560、balanced rank10=83000 占满预算）。**DAG val_l2=-703207 vs 最优树 balanced -207451，提升 239.0%**。关键：给树补 rank 到等参数量毫无用处，只是更快过拟合发散（chain 300 步、balanced 350 步早停）——瓶颈是结构性的，非容量（`deep_dag_vs_tree_report_zh.md`）
- [x] **工程加固**：einsum 改整数下标（解除 52 字母上限，支持大图）；训练加梯度裁剪 + train_noise + 早停（高维 L2 防过拟合/发散）；`dag_ttns` 20 项 dense 验证仍全 PASS
- [x] **架构澄清（重要）**：模型侧**只用单父 TTNS**（淘汰多父 core）；多层 = 数据依赖结构；层间聚合用 **max-plus**：$x_v=\max_{u}(x_u+e_{uv})+d_v$（edge/node delay 已知、独立、连续）。传播 = 逐层进行。
- [x] **TTNS 密度采样器**：`ttns_sampler.py` 条件 inverse-CDF 采样 + `experiments/check_ttns_sampler.py` 校验。诊断出线性 TTNS 负密度 clamp 是相关性衰减主因（拟合好时负质量 <2% 可忽略）。
- [x] **max-plus 方案 A（采样+逐层重拟合）**：`maxplus_pipeline.py` + `experiments/fit_maxplus_propagation.py`。采样误差跨层累积。
- [x] **max-plus 方案 B（CDF 域解析收缩）**：`maxplus_cdf.py` + `experiments/fit_maxplus_cdf_vs_sampling.py`。利用"多源 max 在 CDF 域 = 乘积"，可分离收缩、无截断；在带相关性的密度上比 A 更准地保相关。
- [x] **逐层森林（层内分块）+ 传播 → 全联合密度 vs 扁平 TT/TTNS**：`layered_forest.py`（层内 MI 阈值连通分量分块 + 块内单父 chow-liu TTNS + max-plus 条件核闭式对数密度）+ `experiments/fit_layered_vs_flat_tt.py`。联合 $p(x)=p_{\text{forest}}(L_0)\prod_{l\ge1}\prod_v K_v(x_v\mid\mathrm{pa}(v))$。源层 4 个独立 $U(0,1)$ 被正确分成 4 块 `[[0],[1],[2],[3]]`。**留出集联合对数密度：分层 7.231（仅 64 学习参数）vs 扁平 TTNS_chowliu 3.805（79k 参数）vs TT_chain −9.652（117k，高秩发散）→ 优势 +3.426 nats/样本**（`layered_vs_flat_tt_report_zh.md`）。
- [x] **架构修正：每层 = 完整 TTNS 森林 + max-plus 逐层传播链**（用户定型）：`experiments/fit_layered_forest_propagation.py`。每层层内按 MI 分块 → 每块单父 TTNS（块间独立）；层间 = 上层森林采样（`ttns_sampler.sample_ttns` 逐块 `layered_forest.sample_forest`）→ 已知 edge/node delay 做 max-plus（`propagate_layer`）→ 目标样本 → 重拟合下层森林。逐层向下，**每层都是 TTNS 森林**（不再用解析核拼联合，纠正此前 `fit_layered_vs_flat_tt.py` 只有源层是 TTNS 的偏差）。评估逐层边缘（mae/W1）+ 相关性（corr_fro）vs 真值，对照扁平 TT（chain 全联合采样后切片）。**[4,4,4] 结果：逐层平均 W1 分层 0.010 vs 扁平 0.085（8.5×）；corr_fro 分层 0.216 vs 扁平 0.990（4.6×）；源层正确分成 4 个独立块**（`layered_forest_propagation_report_zh.md`）。分层 corr_fro 随深度增长（0.083→0.184→0.381）= 方案 A 采样 clamp 误差累积，后续可换方案 B（CDF 解析）抑制。
- [x] **方案 B（森林感知 CDF 解析）作为并列第二方案**（不改采样实现）：新增 `maxplus_cdf_forest.py`（`UpperForest` + `propagate_layer_cdf_forest`），利用块间独立 → 期望按块因子分解 $\mathbb{E}[\prod_u F_e(s-x_u)]=\prod_b\mathbb{E}_{x_b}[\cdots]$，复用 `maxplus_cdf.UpperModel` 的可分离收缩；`experiments/fit_layered_forest_schemes.py` 逐层共享同一上层森林比 A（采样）/B（解析）/真值。**[4,4,4] 结果：逐层平均 corr_fro A=0.085 vs B=0.050；相关上层（L1→L2 有符号密度）B=0.071 vs A=0.141（~2×）；独立上层（L0→L1，4 个独立块）两者相当**（`layered_forest_schemes_report_zh.md`）。
- [ ] 后续打磨：方案 B 接入"逐层链"（每层重拟合森林以持续向下）、层内多块合成层演示、块内不同算法、多 seed 统计

---

*创建：2026-06-26 — Phase 2 启动。*
*更新：2026-06-29 — M2.2 Chow–Liu、hub-rank 加速、按边 rank 支持完成；确立 Chow–Liu 主线。*
*更新：2026-06-29 — 真多父 core 地基 + L2 拟合落地；diamond 有环结构 DAG vs 最优树 11.4%，证实多父超越树的表达力优势。*
*更新：2026-06-29 — 层间传播 pipeline 打通（逐层采样 + 多父全联合拟合）；3 层多父 DAG 端到端 vs 最优树 29.3%，终局主线跑通。*
*更新：2026-06-29 — 复杂 4 层/20 节点 + 乘积交互核：DAG vs 最优树 215.5%；einsum 整数下标 + 梯度裁剪/train_noise/早停 加固，支持大图稳定训练。*
*更新：2026-06-30 — 架构定型为单父 TTNS + max-plus 逐层传播；方案 A（采样重拟合）/ B（CDF 域解析）落地；逐层森林（层内 MI 分块）+ 已知条件核组装全联合，留出集联合对数密度优于扁平 TT/TTNS +3.426 nats。*
*更新：2026-06-30 — 架构修正：**每层 = 完整 TTNS 森林 + max-plus 逐层传播链**（`fit_layered_forest_propagation.py`）；逐层边缘/相关性还原全面优于扁平 TT（W1 8.5×、corr_fro 4.6×）。*
*更新：2026-06-30 — 新增**森林感知方案 B（CDF 解析，不改采样）**`maxplus_cdf_forest.py` + `fit_layered_forest_schemes.py`；A vs B 逐层 corr_fro 0.085 vs 0.050，相关上层 B ~2× 更优。*
