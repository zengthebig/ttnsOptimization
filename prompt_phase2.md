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
- [ ] **层间传播 pipeline**（终局）：拓扑分层；上层 TTNS 采样 × 已知 delay 核 → 生成目标样本；下层多父 TTNS 拟合（方案 2）。地基已就位，待串成多层前向

---

*创建：2026-06-26 — Phase 2 启动。*
*更新：2026-06-29 — M2.2 Chow–Liu、hub-rank 加速、按边 rank 支持完成；确立 Chow–Liu 主线。*
*更新：2026-06-29 — 真多父 core 地基 + L2 拟合落地；diamond 有环结构 DAG vs 最优树 11.4%，证实多父超越树的表达力优势。*
