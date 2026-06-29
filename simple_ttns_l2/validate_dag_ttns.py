"""多父 DAG TTNS 正确性验证：用独立的 brute-force 收缩对照 einsum 实现。

遵守仓库铁律：新算子必须有 dense 对照。这里 brute-force（纯 Python 嵌套求和，
完全不依赖 einsum）作为 ground truth，验证多父收缩、rank-1 求值、内积。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# 强制把 TTNSDE 放到 sys.path 最前，避免被仓库根目录的同名 `ttde`（缺 ttns_opt）遮蔽。
for p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from simple_ttns_l2.dag_ttns import (  # noqa: E402
    DAGTTNS,
    dag_eval_rank1,
    dag_full_tensor,
    dag_inner_product,
    dag_quadratic_form,
    make_dag_graph,
    random_dag_ttns,
)


def _brute_full(cores, graph) -> np.ndarray:
    """独立于 einsum 的稠密收缩：遍历所有 bond / 物理 index 组合累加。"""
    np_cores = [np.asarray(c) for c in cores]
    bond_dim = [0] * len(graph.edges)
    for node in range(graph.n):
        for k, ei in enumerate(graph.incident[node]):
            bond_dim[ei] = np_cores[node].shape[1 + k]

    out = np.zeros(tuple(graph.dims))
    for phys in itertools.product(*[range(d) for d in graph.dims]):
        total = 0.0
        for bond in itertools.product(*[range(b) for b in bond_dim]):
            prod = 1.0
            for node in range(graph.n):
                idx = [phys[node]] + [bond[ei] for ei in graph.incident[node]]
                prod *= np_cores[node][tuple(idx)]
            total += prod
        out[phys] = total
    return out


def _brute_q2(dense: np.ndarray, grams) -> float:
    """独立计算 ∫q² = Σ_p M[p] (Π_k G_k 作用于各轴后的 M)[p]。"""
    M = np.asarray(dense)
    M2 = M.copy()
    for k, G in enumerate(grams):
        M2 = np.tensordot(np.asarray(G), M2, axes=([1], [k]))
        M2 = np.moveaxis(M2, 0, k)
    return float(np.sum(M * M2))


def _rand_gram(key, d) -> np.ndarray:
    a = np.asarray(jax.random.normal(key, (d, d), dtype=jnp.float64))
    return a @ a.T


def _assert_close(name: str, got, expected, atol=1e-10, rtol=1e-10) -> bool:
    got = np.asarray(got)
    expected = np.asarray(expected)
    abs_err = float(np.max(np.abs(got - expected)))
    denom = float(np.max(np.abs(expected))) + 1e-30
    rel_err = abs_err / denom
    ok = abs_err <= atol + rtol * denom
    print(f"[{name}] abs_err={abs_err:.3e} rel_err={rel_err:.3e} status={'PASS' if ok else 'FAIL'}")
    return ok


def _check_case(name: str, dims, edges, edge_ranks, key) -> bool:
    print(f"\n=== {name}: dims={list(dims)} edges={list(edges)} ranks={edge_ranks} ===")
    graph = make_dag_graph(len(dims), dims, edges)
    k_t1, k_t2, k_v = jax.random.split(key, 3)
    t1 = random_dag_ttns(k_t1, graph, edge_ranks=edge_ranks, scale=0.7)
    t2 = random_dag_ttns(k_t2, graph, edge_ranks=edge_ranks, scale=0.7)

    dense1 = np.asarray(dag_full_tensor(t1, graph))
    dense2 = np.asarray(dag_full_tensor(t2, graph))
    brute1 = _brute_full(t1.cores, graph)

    ok = True
    ok &= _assert_close(f"{name} full_tensor", dense1, brute1)

    vecs = [jax.random.normal(k, (d,), dtype=jnp.float64) for k, d in
            zip(jax.random.split(k_v, len(dims)), dims)]
    got_eval = float(dag_eval_rank1(t1, graph, vecs))
    expected_eval = float(np.einsum(
        dense1, list(range(len(dims))),
        *sum(([np.asarray(v), [i]] for i, v in enumerate(vecs)), []),
        [],
    ))
    ok &= _assert_close(f"{name} eval_rank1", got_eval, expected_eval)

    got_ip = float(dag_inner_product(t1, t2, graph))
    expected_ip = float(np.sum(dense1 * dense2))
    ok &= _assert_close(f"{name} inner_product", got_ip, expected_ip)

    gram_keys = jax.random.split(jax.random.fold_in(k_v, 7), len(dims))
    grams = [_rand_gram(gk, d) for gk, d in zip(gram_keys, dims)]
    got_q2 = float(dag_quadratic_form(t1, graph, [jnp.asarray(g) for g in grams]))
    expected_q2 = _brute_q2(dense1, grams)
    ok &= _assert_close(f"{name} quadratic_form", got_q2, expected_q2)
    return ok


def main():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)
    results = []
    # A. fork v-structure：节点 2 双父（多父核心用例）
    results.append(_check_case("fork", [3, 3, 3], [(0, 2), (1, 2)], 2, keys[0]))
    # B. chain：退化为单父链（sanity）
    results.append(_check_case("chain", [2, 3, 2], [(0, 1), (1, 2)], 2, keys[1]))
    # C. polytree：多父 + 链尾 + 按边不同 rank
    results.append(_check_case("polytree", [2, 2, 3, 2], [(0, 2), (1, 2), (2, 3)], {0: 2, 1: 3, 2: 2}, keys[2]))
    # D. 三父节点：节点 3 连 0,1,2
    results.append(_check_case("three_parents", [2, 2, 2, 3], [(0, 3), (1, 3), (2, 3)], 2, keys[3]))
    # E. diamond（有环）：x2,x3 同时依赖 x0,x1 —— 无向图含环，树无法表达，验证 einsum 通用收缩正确
    results.append(_check_case("diamond_loopy", [2, 2, 2, 2], [(0, 2), (1, 2), (0, 3), (1, 3)], 2, keys[4]))

    n_pass = sum(results)
    print(f"\n全部验证：{n_pass}/{len(results)} 组通过。")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
