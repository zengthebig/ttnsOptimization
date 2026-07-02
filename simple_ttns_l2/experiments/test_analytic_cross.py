"""单元测试：解析交叉项 E_{p_tree}[q] 的树消息传递 vs 暴力网格积分(小 K)。

伪造随机树 + 随机条件核 + 随机 cores + 随机基取值，两法必须一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from ttde.ttns.ttns_opt import TTNSOpt, eval_rank1_ttns, _root_from_parent  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import _cross_term_fn  # noqa: E402


def brute_force(cores, parent, Bg, pcond, p_root, delta):
    """E_{p_tree}[q] = Σ_config q(config) p_tree(config) Δ^K。"""
    K = len(parent)
    G = Bg[0].shape[0]
    root = _root_from_parent(parent)
    total = 0.0
    import itertools
    for config in itertools.product(range(G), repeat=K):
        vectors = jnp.stack([Bg[v][config[v]] for v in range(K)])
        q = float(eval_rank1_ttns(TTNSOpt(tuple(cores)), vectors, parent))
        pt = float(p_root[config[root]])
        for v in range(K):
            if v == root:
                continue
            pt *= float(pcond[v][config[v], config[parent[v]]])
        total += q * pt * (delta ** K)
    return total


def make_case(parent, G=8, m=5, rank=3, seed=0):
    rng = np.random.default_rng(seed)
    K = len(parent)
    root = _root_from_parent(parent)
    from ttde.ttns.ttns_opt import _children_from_parent
    children = _children_from_parent(parent)
    cores = []
    for v in range(K):
        rp = 1 if v == root else rank
        rc = [rank for _ in children[v]]
        cores.append(jnp.asarray(rng.standard_normal((rp, m, *rc))))
    Bg = [jnp.asarray(rng.random((G, m))) for _ in range(K)]
    delta = 0.3
    p_root = jnp.asarray(rng.random(G)); p_root = p_root / (float(jnp.sum(p_root)) * delta)
    pcond = {}
    for v in range(K):
        if v == root:
            continue
        M = rng.random((G, G))
        M = M / (M.sum(axis=0, keepdims=True) * delta)  # 每列积分=1
        pcond[v] = jnp.asarray(M)
    return cores, Bg, pcond, p_root, delta


def run(parent, name):
    cores, Bg, pcond, p_root, delta = make_case(parent)
    cross = _cross_term_fn(parent, Bg, {k: v for k, v in pcond.items()}, p_root, delta)
    mp = float(cross(cores))
    bf = brute_force(cores, parent, Bg, pcond, p_root, delta)
    rel = abs(mp - bf) / (abs(bf) + 1e-12)
    print(f"[{name}] parent={parent}  msg_passing={mp:.6f}  brute={bf:.6f}  rel_err={rel:.2e}")
    assert rel < 1e-6, f"MISMATCH {name}"


if __name__ == "__main__":
    run([-1, 0], "chain-2")
    run([-1, 0, 1], "chain-3")
    run([-1, 0, 0], "star-3")
    run([-1, 0, 1, 1, 0], "tree-5")
    run([-1, 0, 0, 0, 0], "star-5")
    print("ALL PASS")
