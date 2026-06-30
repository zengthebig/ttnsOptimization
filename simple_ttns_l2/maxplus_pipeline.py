"""Max-plus 多层 DAG 传播 pipeline（方案 A：采样 + 逐层重拟合）。

与用户澄清的架构对齐：
- 数据依赖是多层 DAG（节点分层，父都在上一层），但**模型侧只用单父 TTNS**。
- 多来源聚合用 **max-plus**：节点 $v$ 的取值
  $$x_v = \max_{u\in\mathrm{pa}(v)} (x_u + e_{uv}) + d_v,$$
  其中 edge delay $e_{uv}$、node delay $d_v$ 为已知、独立的连续随机变量。
- 传播 = 逐层进行：上一层用 TTNS（森林）表示其联合密度 → 从该 TTNS 采样 →
  对样本施加 max-plus（采新的 delay）→ 得到下一层样本 → 重新拟合下一层 TTNS。

本文件提供：
1. `DelayParams`：源/边/节点延迟分布参数（默认均匀分布）。
2. `ground_truth_samplers`：构造与 `dag_pipeline.sample_joint` 兼容的 source/kernel，
   用于生成"真值"全联合样本（祖先 max-plus 采样）。
3. `propagate_layer`：把上一层样本经 max-plus 推进到下一层（方案 A 的传播步）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from simple_ttns_l2.dag_pipeline import MultiLayerSpec


@dataclass(frozen=True)
class DelayParams:
    """各延迟分布参数（均匀分布，连续且相互独立）。"""

    src_lo: float = 0.0
    src_hi: float = 1.0
    edge_lo: float = 0.0
    edge_hi: float = 0.5
    node_lo: float = 0.0
    node_hi: float = 0.5


def _maxplus_step(parent_vals: np.ndarray, rng: np.random.Generator, params: DelayParams) -> np.ndarray:
    """对一个下层节点施加 max-plus：parent_vals[n,k] -> [n]。

    $x_v=\\max_u(x_u+e_{uv})+d_v$，每次调用抽取新的独立 edge/node delay。
    """
    n, k = parent_vals.shape
    e = rng.uniform(params.edge_lo, params.edge_hi, size=(n, k))
    income = parent_vals + e
    d = rng.uniform(params.node_lo, params.node_hi, size=n)
    return income.max(axis=1) + d


def ground_truth_samplers(
    spec: MultiLayerSpec, params: DelayParams
) -> Tuple[Dict[int, Callable], Dict[int, Callable]]:
    """构造与 `dag_pipeline.sample_joint` 兼容的 (sources, kernels)。

    sources[node](rng, n) -> [n]；kernels[node](parent_vals[n,k], rng, n) -> [n]。
    用于生成真值全联合样本（祖先 max-plus 采样）。
    """
    sources: Dict[int, Callable] = {}
    kernels: Dict[int, Callable] = {}
    for node in spec.topo_order:
        if not spec.parents(node):
            sources[node] = lambda rng, n: rng.uniform(params.src_lo, params.src_hi, size=n)
        else:
            kernels[node] = lambda parent_vals, rng, n: _maxplus_step(parent_vals, rng, params)
    return sources, kernels


def layer_columns(spec: MultiLayerSpec, li: int) -> Tuple[int, ...]:
    """第 li 层的节点 id 元组（即该层样本矩阵的列顺序）。"""
    return spec.layers[li]


def propagate_layer(
    spec: MultiLayerSpec,
    li: int,
    prev_layer_samples: np.ndarray,
    params: DelayParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """把第 li-1 层样本经 max-plus 推进到第 li 层。

    prev_layer_samples: [n, len(layers[li-1])]，列顺序 = layers[li-1]。
    返回 [n, len(layers[li])]，列顺序 = layers[li]。
    """
    if li < 1:
        raise ValueError("propagate_layer 仅用于 li>=1")
    prev = spec.layers[li - 1]
    cur = spec.layers[li]
    col = {node: i for i, node in enumerate(prev)}
    P = np.asarray(prev_layer_samples, dtype=float)
    n = P.shape[0]
    out = np.zeros((n, len(cur)), dtype=float)
    for j, node in enumerate(cur):
        ps = spec.parents(node)
        if any(p not in col for p in ps):
            raise ValueError(f"节点 {node} 的父 {ps} 不全在上一层 {prev}（仅支持父在相邻上层）")
        parent_vals = np.stack([P[:, col[p]] for p in ps], axis=1)  # [n, k]
        out[:, j] = _maxplus_step(parent_vals, rng, params)
    return out
