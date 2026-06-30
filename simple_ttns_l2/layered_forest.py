"""逐层 TTNS 森林：层内按相关性分块 + 块内单父 TTNS + max-plus 条件核。

架构（与用户澄清对齐）：
- 逐层构建；层间只有父子/复用关系（下层节点的父都在上层，可被多子复用）。
- 层内先按相关性分成若干**互不相关的块**（block）；块间近似独立 → 层密度 = 块密度乘积。
- 每块内单独用单父 TTNS（chow-liu）拟合（不同块可换不同算法，此处统一 chow-liu）。

提供：
1. `layer_blocks`：层内 MI 阈值化 → 连通分量 = 独立块。
2. `fit_layer_forest`：逐块拟合单父 TTNS，返回森林。
3. `forest_log_density`：层密度对数 = Σ_block log q_block（块独立）。
4. `maxplus_cond_logdensity`：已知 max-plus 条件核的闭式对数密度。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jax
import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[1]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.chow_liu import (
    discretize_samples_quantile,
    estimate_mutual_information_matrix,
    estimate_chow_liu_tree,
)
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1
from simple_ttns_l2.objective import (
    batch_basis_vectors_from_samples,
    batch_eval_q_ttns,
    normalize_ttns_by_integral,
)
from simple_ttns_l2.experiments.fit_diamond_dag_vs_tree import train_tree_l2
from simple_ttns_l2.maxplus_pipeline import DelayParams


def layer_blocks(samples: np.ndarray, mi_threshold: float = 0.02, n_bins: int = 16) -> List[List[int]]:
    """层内变量按 MI 阈值化建图取连通分量 = 互不相关的块（返回局部索引分组）。"""
    x = np.asarray(samples)
    d = x.shape[1]
    if d == 1:
        return [[0]]
    mi = estimate_mutual_information_matrix(discretize_samples_quantile(x, n_bins=n_bins))
    adj = [[] for _ in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            if mi[i, j] > mi_threshold:
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * d
    blocks: List[List[int]] = []
    for s in range(d):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        blocks.append(sorted(comp))
    return blocks


@dataclass
class BlockModel:
    local_vars: Tuple[int, ...]   # 在本层内的局部索引
    global_vars: Tuple[int, ...]  # 全局节点 id
    parent: Tuple[int, ...]
    ttns: object
    bases: object


def fit_layer_forest(
    layer_samples: jnp.ndarray,
    global_ids: Sequence[int],
    cfg: dict,
    key,
    label: str = "layer",
    mi_threshold: float = 0.02,
) -> List[BlockModel]:
    """层内分块 + 逐块拟合单父 TTNS。返回 BlockModel 列表。"""
    x = np.asarray(layer_samples)
    blocks = layer_blocks(x, mi_threshold=mi_threshold)
    forest: List[BlockModel] = []
    for bi, blk in enumerate(blocks):
        sub = jnp.asarray(x[:, blk])
        gids = tuple(int(global_ids[i]) for i in blk)
        if len(blk) == 1:
            parent = [0]
        else:
            parent = [int(p) for p in estimate_chow_liu_tree(np.asarray(sub), n_bins=16, root=0).parent]
        bases = build_bases(sub, cfg["q"], cfg["m"])
        gram = vmap(type(bases).l2_integral)(bases)
        basis_integrals = vmap(type(bases).integral)(bases)
        k_init, key = jax.random.split(key)
        split = int(0.8 * sub.shape[0])
        tr, val = sub[:split], sub[split:]
        t0 = init_ttns_from_rank1(k_init, bases, tr, parent, cfg["rank"], cfg["init_noise"])
        best, _ = train_tree_l2(
            t0, parent, bases, tr, val, gram, basis_integrals,
            key=k_init, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
            normalize_every=1, log_every=cfg["log_every"], label=f"{label}_blk{bi}",
            train_noise=cfg["train_noise"], early_stop_patience=cfg["early_stop_patience"],
        )
        best, _ = normalize_ttns_by_integral(best, basis_integrals, parent)
        forest.append(BlockModel(tuple(blk), gids, tuple(parent), best, bases))
    return forest


def forest_log_density(forest: List[BlockModel], x_layer: np.ndarray, eps: float = 1e-12):
    """层密度对数 = Σ_block log q_block(x_block)。返回 (loglik[n], nonpos_rate)。"""
    x = np.asarray(x_layer)
    n = x.shape[0]
    total = np.zeros(n)
    nonpos = np.zeros(n, dtype=bool)
    for bm in forest:
        xb = jnp.asarray(x[:, list(bm.local_vars)])
        bv = batch_basis_vectors_from_samples(bm.bases, xb)
        q = np.asarray(batch_eval_q_ttns(bm.ttns, bv, list(bm.parent)))
        nonpos |= q <= 0
        total += np.log(np.clip(q, eps, None))
    return total, float(nonpos.mean())


def maxplus_cond_logdensity(
    x_v: np.ndarray, parent_vals: np.ndarray, params: DelayParams, eps: float = 1e-12
) -> np.ndarray:
    r"""已知 max-plus 条件核的闭式对数密度。

    $x_v=\max_u(x_u+e_{uv})+d_v$，$e\sim U[\text{edge\_lo},\text{edge\_hi}]$，$d\sim U[\text{node\_lo},\text{node\_hi}]$。
    $F_M(s)=\prod_u F_e(s-x_u)$，$K_v(y)=\dfrac{F_M(y-\text{node\_lo})-F_M(y-\text{node\_hi})}{\text{node\_hi}-\text{node\_lo}}$。
    """
    pv = np.asarray(parent_vals)  # [n, k]
    el, eh = params.edge_lo, params.edge_hi
    nl, nh = params.node_lo, params.node_hi

    def F_M(s):  # s: [n]
        Fu = np.clip((s[:, None] - pv - el) / (eh - el), 0.0, 1.0)  # [n, k]
        return Fu.prod(axis=1)

    dens = (F_M(x_v - nl) - F_M(x_v - nh)) / (nh - nl)
    return np.log(np.clip(dens, eps, None))
