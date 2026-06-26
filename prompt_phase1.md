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
