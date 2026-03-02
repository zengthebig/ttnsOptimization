# TTNS 训练迁移进度

## 范围
本次更新在 `TTNSDE/ttde` 下新增了一个**最小可用的 TTNS 训练路径**，同时保持原有 TT 训练路径不变。

## 已完成内容

### 1. TTNS 优化表示与核心收缩算子
- 新增文件：`TTNSDE/ttde/ttns/ttns_opt.py`
- 新数据结构：
  - `TTNSOpt(cores)`
- 新增工具函数：
  - `normalized_inner_product_ttns(...)`
  - `normalized_eval_rank1_ttns(...)`
  - `normalized_quadratic_form_ttns(...)`
- 目的：
  - 为密度模型提供树结构版本的内积、rank-1 评估和二次型计算能力，作为 TT 对应算子的替代。

### 2. TTNS 版评分模型
- 新增文件：`TTNSDE/ttde/score/models/opt_for_tree_data.py`
- 新增模型类：
  - `PAsTTNSOptBase`
  - `PAsTTNSSqrOpt`
- 说明：
  - 默认使用平衡二叉树的父子拓扑。
  - 保持与现有 trainer/loss 兼容的高层 API（`log_p`、`log_int_p`、`init_canonical`、`add_noise`）。
  - 当前 `init_canonical` 采用最小化 rank-1 启动路径（尚未接入 TTNS-EM）。

### 3. 实验配置接线
- 更新文件：`TTNSDE/ttde/score/experiment_setups/model_setups.py`
- 新增配置类：
  - `PAsTTNSSqrOpt`

### 4. 训练入口接线
- 更新文件：`TTNSDE/ttde/train.py`
- 新增命令行参数：
  - `--model-type [tt|ttns]`（默认：`tt`）
  - `--ttns-topology [chain|balanced]`（默认：`chain`）
- 行为：
  - `tt` -> 走原有 TT 模型路径
  - `ttns` -> 走新增 TTNS 模型路径
  - `ttns + chain` -> 与 TT 链结构对齐的初始化/训练路径
  - `ttns + balanced` -> 使用通用 rank-1 初始化（不走 TT-chain canonical 映射）

## 当前状态

## 已可用
- 原 TT 路径仍可用。
- TTNS 路径已接入 `train.py`。
- 新增 TTNS 模块已通过 Python 语法编译检查。
- 已增加 TTNS 数学算子单元测试：
  - `TTNSDE/test/test_ttns_opt_unittest.py`
- 已增加 TTNS 最小训练烟雾脚本：
  - `TTNSDE/scripts/smoke_train_ttns.py`

## 尚未完成
- 与 TT 版本等价的 TTNS canonical/EM 初始化。
- 在真实数据集上的性能调优与数值稳定性评估。
- 当前环境下的完整端到端运行验证（shell 运行环境缺少 `jax` 包）。

## 使用方式（依赖可用时）

TTNS 模型示例命令：

```bash
python -m ttde.train \
  --dataset power \
  --q 2 --m 64 --rank 8 --n-comps 4 \
  --model-type ttns \
  --em-steps 5 --noise 0.01 \
  --batch-sz 512 --train-noise 0.01 --lr 0.001 --train-steps 100 \
  --data-dir /path/to/data --work-dir /path/to/workdir
```

运行 TTNS 数学算子单元测试：

```bash
PYTHONPATH=TTNSDE python3 -m unittest TTNSDE/test/test_ttns_opt_unittest.py
```

运行 TTNS 最小训练 smoke test：

```bash
PYTHONPATH=TTNSDE python3 TTNSDE/scripts/smoke_train_ttns.py
```

## 设计取舍
- TT 与 TTNS 两条训练路径并存，以降低回归风险。
- 避免大范围改动 trainer；主要改动集中在模型层与 TTNS 数学算子层。
- 先交付最小可运行 TTNS 初始化方案，TTNS-EM 作为后续任务推进。

## 建议下一步
1. 增加 TTNS 专用初始化策略（canonical-like 或 tree-EM）。
2. 在真实数据集上运行短程训练，观察 `loss/log_int_p/nonpositive` 是否稳定。
3. 引入 TTNS 版本的 canonical/EM 初始化，替换当前 rank-1 bootstrap。

# 运行
/home/sbzeng/.conda/envs/ttns/bin/python -u ttde/train.py \
  --dataset power \
  --q 2 --m 64 --rank 16 --n-comps 8 \
  --model-type tt \
  --em-steps 0 --noise 0.01 \
  --batch-sz 512 --train-noise 0.01 --lr 0.001 --train-steps 20 \
  --data-dir /home/sbzeng/2_1/research/datasets/data/ --work-dir ~/workdir_ttnsde
  3


cd /home/sbzeng/2_1/research/TTDE/TTNSDE

/home/sbzeng/.conda/envs/ttns/bin/python -m ttde.train \
  --dataset power \
  --q 2 --m 64 --rank 16 --n-comps 8 \
  --model-type ttns \
  --ttns-topology chain \
  --em-steps 0 --noise 0.01 \
  --batch-sz 512 --train-noise 0.01 --lr 0.001 --train-steps 20 \
  --data-dir /home/sbzeng/2_1/research/datasets/data/ --work-dir ~/workdir_ttnsde
  3
