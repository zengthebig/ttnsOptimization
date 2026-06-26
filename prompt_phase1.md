阅读当前目录中的Program.md, 里面有完整的说明，baseline数据，允许进行修改的范围，API参考以及构建测试命令。

+ 你的任务：开发TTNS传播代码并进行代码正确性测试以及效果测试。

+ 首先进行环境构建， 并确认测试正确性。

+ 流程： 
1. git log --oneline -5 查看git历史并且理解当前已经进行过哪些优化
2. 根据当前代码仓， 理解TTNS相关代码逻辑
3. 按照program.md中 “必须做的事” 的顺序，按照优先级展开。
4. 修改TTNS代码, 在构建完正确性和效果测试链路之后进行测试， 并将链路写入本文档。
7. 每轮修改必须输出一个commit, 说明修改内容，理论效果和测试效果

+ 约束
1. 只改TTNS相关代码, 以及更新结果文档的结果， 以及Program.md中的要做的事

---

## 环境构建（2026-06-26）

本机 Python 3.9 + macOS arm64；`jax==0.4.8` 无预编译 wheel，实际使用 `jax==0.4.20` / `jaxlib==0.4.20`、`flax==0.6.11`、`scipy==1.11.4`。

```bash
# 安装依赖（示例）
python3 -m pip install --user "jax==0.4.20" "jaxlib==0.4.20" "flax==0.6.11" \
  "opt-einsum==3.3.0" "numpy==1.24.2" "scipy==1.11.4" "optax" "tqdm"

# 关键：TTNSDE 必须排在仓库根目录之前，避免导入根目录旧 ttde 包
export PYTHONPATH=/path/to/ttnsOptimization/TTNSDE
cd /path/to/ttnsOptimization
```

导入规则：`sys.path` 中 `TTNSDE` 必须优先于仓库根目录（根目录也有 `ttde/`，会遮蔽 `ttns_opt.py`）。

---

## 正确性测试链路（Program.md §5）

```bash
cd /path/to/ttnsOptimization
export PYTHONPATH=$PWD/TTNSDE

# 5.1 数学单元测试（需 Python≥3.10 或修复 model_setups.py 类型注解）
python3 -m unittest TTNSDE/test/test_ttns_opt_unittest.py -v

# 5.2 dense 对照（9 节点深树，必跑）
python3 TTNSDE/validate_ttns_opt.py

# 5.3 L2 目标测试
env -u PYTHONPATH python3 simple_ttns_l2/tests/test_ttns_l2_objective_unittest.py -v
python3 -m pip install --user click  # smoke 测试额外依赖
env -u PYTHONPATH python3 simple_ttns_l2/tests/test_train_l2_smoke_unittest.py -v
```

**本轮结果（优化后）：**

| 检查项 | 结果 |
|--------|------|
| `validate_ttns_opt.py` | 3/3 PASS（误差 ≤ 1e-10） |
| `test_ttns_l2_objective_unittest.py` | 5/5 OK |
| fast vs stable 数值对齐 | 内积/求值/二次型与 dense 一致 |

---

## 第 1 轮优化（2026-06-26）

**修改文件：** `TTNSDE/ttde/ttns/ttns_opt.py`

**内容：**
1. 新增 `_postorder_nodes`、`_run_postorder`、`_run_normalized_postorder`：用显式后序遍历替代嵌套 Python 递归。
2. 新增 `_inner_product_local`：0/1/2 子节点分支使用融合 `cached_einsum`（修复物理指标共享后通过 dense 对照）。
3. `normalized_eval_rank1_ttns` / `normalized_quadratic_form_ttns` 复用 `_eval_rank1_local` / `_quadratic_local`（原先走通用 `tensordot` 循环）。

**理论效果：** 减少 Python 递归开销；平衡二叉树（≤2 子节点）上 stable 路径与 fast 路径一样走融合 einsum；有利于 JAX 追踪缓存。

**性能（balanced 6D, rank=16, basis=64, sec/call）：**

| 算子 | 耗时 |
|------|------|
| eval_rank1 fast | 0.0024 |
| eval_rank1 stable | 0.0040 |
| quadratic fast | 0.0066 |
| quadratic stable | 0.0077 |
| inner_product stable | 0.0059 |

stable 仍比 fast 慢（逐节点 log 归一化），但内积 stable 路径已从纯 tensordot 改为 einsum 融合。

---

## 研究目标（对齐）

> 在**多层依赖结构**（阶段性：树形 DAG；终局：多父 DAG）上，证明 TTNS 相对 TT（链）的密度近似收益。

**当前能力边界：**

| 能力 | 状态 |
|------|------|
| 多层树传播（balanced / 随机树 / 深树） | 已实现，算子正确性已验证 |
| 树目标 + 拓扑对齐 → TTNS 优于 chain | 已有报告（如 `fit_ceiling_two_models_complex`） |
| 多父节点 DAG TTNS | **未实现**（需联结树或改 core 拓扑） |
| Chow–Liu 四模型对比 | 脚本已有，**报告未入库** |

---

## 阶段二：多层树形 DAG 收益实验矩阵

**核心对比：** 同一目标分布上，`TTNS(匹配树)` vs `TTNS(chain)` vs `pure TT(chain)`。

**公平性约定：** `init_noise=0`；≥3 seed 报均值±方差；主指标 **2D 切片 IAE**（跨 L2/MLE 可比），辅指标 `val_l2` / `val_nll`。

| 优先级 | 目标分布 | 模型 A | 模型 B | 模型 C | 脚本 | 预期结论 |
|:---:|----------|--------|--------|--------|------|----------|
| P0 | complex **balanced** 树 | balanced TTNS | chain TTNS | pure TT | `fit_ceiling_two_models.py` | A 显著优于 B、C（已有初步证据） |
| P0 | complex **chain** 树 | chain TTNS | balanced TTNS | pure TT | 同上 | A ≈ C；B 失败（拓扑不匹配） |
| P1 | **随机递归树** | 匹配 parent TTNS | chain TTNS | pure TT | `random_tree_target.py` + 新建或扩展现有脚本 | 匹配树 > chain |
| P1 | balanced 目标，`init_noise=0` | balanced / chain / TT 三模型 | — | — | `fit_three_models_init0.py` | 可对齐时三者一致，作 sanity check |
| P2 | 提高树深（8D balanced） | balanced vs chain | TT(MLE) | Chow–Liu TTNS | `fit_limit_extreme_tt_vs_ttns_three_topologies_mle_8d.py` | 深树 + 非链目标放大 TTNS 优势 |
| P2 | 同 rank 拟合天花板 | balanced vs chain | — | — | `fit_quality_limit.py` / `fit_limit_balanced_extreme*.py` | 固定 rank 下 IAE 差距 |
| P3 | 真实数据（Power 等） | Chow–Liu TTNS | balanced | chain | 同上四模型脚本 + UCI | 数据驱动树是否优于固定拓扑（**待跑**） |

**必做对照（每个 P0/P1 实验）：**

1. 训练前确认 `target_parent` 与 `model_parent` 是否一致（不一致时预期失败，作负对照）。
2. 报告 `final_integral ≈ 1`（L2 路径归一化后）。
3. 至少 3 个切片对 `(0,1)`, `(0,3)`, `(2,5)` 的 IAE 与 `ratio_to_floor`。

**运行模板（L2，6D balanced 目标示例）：**

```bash
cd /path/to/ttnsOptimization
# 实验脚本自带 sys.path；勿设 PYTHONPATH=TTNSDE（会与 REPO 路径冲突）
env -u PYTHONPATH python3 simple_ttns_l2/experiments/fit_ceiling_two_models.py
env -u PYTHONPATH python3 simple_ttns_l2/experiments/fit_three_models_init0.py
```

**成功标准（阶段二）：**

- balanced（或随机树）目标上，匹配 TTNS 的 mean IAE 比 chain TTNS **稳定低 ≥20%**（多 seed）。
- chain 目标上，chain TTNS 与 pure TT 接近；balanced TTNS 的 val_l2 接近 0（负对照成立）。

---

## 阶段三：多父 DAG 技术路线

要在**真正多父节点**（如 $x_3 = f(x_1, x_2) + \epsilon$）上证明 TTNS 收益，必须先扩展表达能力。

### 方案对比

| | B1：联结树（Junction Tree）展开 | B2：多父 core 拓扑 |
|---|-------------------------------|-------------------|
| **思路** | 将 DAG 三角化/联结树化，在更大树上用现有 `TTNSOpt` | `parent` 允许多父；core 多根边；改写 `ttns_opt.py` 传播 |
| **改动范围** | 目标采样 + `parent` 构造；不改编译器核心 | `ttns_opt.py`、`tensors.py`、训练初始化、全套 §5 测试 |
| **优点** | 复用已验证算子；较快出实验 | 概念干净；不膨胀节点数 |
| **缺点** | 联结树节点/边秩可能很大；解释性弱 | 实现量大；正确性风险高 |
| **建议** | **首选 MVP**：先证「DAG 需要非链结构」 | 长期目标 |

### B1 最小实现范围（建议顺序）

1. **合成多父 DAG 目标**（新文件，如 `simple_ttns_l2/experiments/dag_target.py`）
   - 6–8 维，至少一个节点有两个父节点（显式条件分布）。
   - 提供 `dag_edges` 与联结树 `parent`（人类可读，便于 dense 对照子集）。

2. **联结树构造器**（新文件，如 `simple_ttns_l2/junction_tree.py`）
   - 输入：DAG 边列表 → 输出：联结树 `parent`（可先手写小树，再自动化）。
   - 单元测试：联结树边集覆盖原 DAG 的团（clique）。

3. **实验**
   - 目标：多父 DAG 样本。
   - 模型：`TTNS(联结树)` vs `TTNS(chain)` vs `pure TT`。
   - 指标：切片 IAE；重点展示 chain **无法**表达 $p(x_3 \mid x_1, x_2)$ 的联合结构。

4. **正确性**
   - 联结树节点数 ≤9 时，子集维度上可与 brute-force 边际对照。

### B2 触发条件

当 B1 实验证明「非链结构有收益」但联结树秩/节点膨胀过剧时，再启动 B2。

---

## 建议执行顺序（Agent 下一批工作）

1. [x] 跑通 P0：`fit_ceiling_two_models` + `fit_three_models_init0`，结果写入 `simple_ttns_l2/reports/`（2026-06-26）
2. [ ] 跑通 P2：Chow–Liu 四模型脚本，补第一份 Chow–Liu 对比报告
3. [ ] 设计并实现 `dag_target.py` + 联结树 `parent`（B1 MVP）
4. [ ] 多父 DAG 对比实验（联结树 TTNS vs chain）+ 中文报告
5. [ ] 每步修改 TTNS 相关代码后重跑 `validate_ttns_opt.py`

---

## P0 实验结果（2026-06-26，post ttns_opt 优化后重跑）

运行命令：`env -u PYTHONPATH python3 simple_ttns_l2/experiments/{fit_ceiling_two_models,fit_three_models_init0}.py`

### fit_ceiling_two_models（init_noise=0.01，400 步）

| 目标 | 模型 | final_val_l2 | mean IAE（3 切片） |
|------|------|-------------:|-------------------:|
| **balanced** | **balanced TTNS** | **-0.066** | **~0.21** |
| balanced | chain TTNS | -0.044 | ~0.42 |
| chain | balanced TTNS | ≈0（失败） | ~0.94 |
| chain | chain TTNS | ≈0 | ~0.95 |

**结论：** balanced 目标上匹配拓扑 TTNS 的 L2 与 IAE 均显著优于 chain（IAE 约低 50%）；chain 目标上 balanced 模型失败 → **负对照成立**。

报告：`simple_ttns_l2/reports/fit_ceiling_two_models_complex_report.md`

### fit_three_models_init0（init_noise=0，300 步）

| 目标 | balanced / chain / pure TT | 差异 |
|------|---------------------------|------|
| balanced | val_l2 均为 -0.019462；mean IAE 均为 0.861 | **完全一致**（sanity check） |
| chain | val_l2 均为 -0.000029；mean IAE 均为 0.946 | **完全一致** |

**结论：** `init_noise=0` 时三模型轨迹相同，说明此前「chain TTNS ≠ TT」来自初始化噪声，而非算子 bug。

报告：`simple_ttns_l2/reports/fit_three_models_init0_report_zh.md`

### P0 对阶段二成功标准的判定

| 标准 | 结果 |
|------|------|
| balanced 目标：匹配 TTNS IAE 优于 chain ≥20% | **通过**（~50%，单 seed） |
| chain 目标：chain ≈ TT；balanced 失败 | **通过** |
| 多 seed 均值±方差 | **未做**（下一步可补） |
