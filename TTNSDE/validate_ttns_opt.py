"""
验证 TTNSOpt 核心数学算子的正确性（与 dense 暴力计算对照）。

验证项：
1) normalized_inner_product_ttns
2) normalized_eval_rank1_ttns
3) normalized_quadratic_form_ttns

运行方式（在有 jax 的环境）：
    PYTHONPATH=TTNSDE python3 TTNSDE/validate_ttns_opt.py
"""

from __future__ import annotations

from itertools import product
from typing import List, Sequence, Tuple
from pathlib import Path
import sys

# Ensure this script imports TTNSDE/ttde, not the repository-root ttde package.
THIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_ROOT))

import jax
from jax import numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from ttde.tt.tensors import TTNS
from ttde.ttns.ttns_opt import (
    TTNSOpt,
    normalized_eval_rank1_ttns,
    normalized_inner_product_ttns,
    normalized_quadratic_form_ttns,
)


def parent_to_neighbors(parent: Sequence[int]) -> List[List[int]]:
    n = len(parent)
    neighbors = [[] for _ in range(n)]
    for node, p in enumerate(parent):
        if p == node or p == -1:
            continue
        neighbors[node].append(p)
        neighbors[p].append(node)
    return neighbors


def children_from_parent(parent: Sequence[int]) -> List[List[int]]:
    n = len(parent)
    children = [[] for _ in range(n)]
    for node, p in enumerate(parent):
        if p == node or p == -1:
            continue
        children[p].append(node)
    return children


def build_random_ttns_opt(
    key: jnp.ndarray,
    parent: Sequence[int],
    dims: Sequence[int],
    rank: int,
) -> TTNSOpt:
    children = children_from_parent(parent)
    root = next(i for i, p in enumerate(parent) if p == i or p == -1)
    keys = jax.random.split(key, len(parent))
    cores = []
    for node in range(len(parent)):
        r_parent = 1 if node == root else rank
        r_children = [rank for _ in children[node]]
        shape = (r_parent, int(dims[node]), *r_children)
        cores.append(jax.random.normal(keys[node], shape, dtype=jnp.float64))
    return TTNSOpt(tuple(cores))


def ttns_opt_to_dense(ttns: TTNSOpt, parent: Sequence[int]) -> jnp.ndarray:
    root = next(i for i, p in enumerate(parent) if p == i or p == -1)
    ttns_nonopt = TTNS(
        cores=list(ttns.cores),
        neighbors=parent_to_neighbors(parent),
        root=root,
        parent=list(parent),
    )
    return ttns_nonopt.full_tensor


def recover_scalar(normalized_value) -> jnp.ndarray:
    return normalized_value.value * jnp.exp(normalized_value.log_norm)


def dense_rank1_eval(tensor: jnp.ndarray, vectors: jnp.ndarray) -> jnp.ndarray:
    # sum_{i1,...,id} T[i1,...,id] * \prod_k v_k[i_k]
    total = 0.0
    dims = tensor.shape
    for multi_idx in product(*[range(s) for s in dims]):
        phi = 1.0
        for axis, idx in enumerate(multi_idx):
            phi *= vectors[axis, idx]
        total += tensor[multi_idx] * phi
    return total


def dense_quadratic_form(
    tensor: jnp.ndarray,
    matrices: jnp.ndarray,
) -> jnp.ndarray:
    # sum_{i,j} T[i] * ( \prod_k M_k[i_k, j_k] ) * T[j]
    total = 0.0
    dims = tensor.shape
    all_indices = list(product(*[range(s) for s in dims]))
    for idx_i in all_indices:
        t_i = tensor[idx_i]
        for idx_j in all_indices:
            a_ij = 1.0
            for axis, (ii, jj) in enumerate(zip(idx_i, idx_j)):
                a_ij *= matrices[axis, ii, jj]
            total += t_i * a_ij * tensor[idx_j]
    return total


def assert_close(name: str, got: jnp.ndarray, expected: jnp.ndarray, atol=1e-5, rtol=1e-5):
    abs_err = jnp.max(jnp.abs(got - expected))
    denom = jnp.maximum(jnp.max(jnp.abs(expected)), 1e-12)
    rel_err = abs_err / denom

    print(f"\n[{name}]")
    print(f"  got      = {got}")
    print(f"  expected = {expected}")
    print(f"  abs_err  = {abs_err}")
    print(f"  rel_err  = {rel_err}")
    print(f"  tol      = atol={atol}, rtol={rtol}")
    if (got > 0) and (expected > 0):
        log_got = jnp.log(got)
        log_expected = jnp.log(expected)
        log_abs_err = jnp.abs(log_got - log_expected)
        print(f"  log(got)      = {log_got}")
        print(f"  log(expected) = {log_expected}")
        print(f"  |log_diff|    = {log_abs_err}")

    if not jnp.allclose(got, expected, atol=atol, rtol=rtol):
        raise AssertionError(
            f"[{name}] mismatch: got={got}, expected={expected}, abs_err={abs_err}, rel_err={rel_err}"
        )
    print(f"  status   = PASS")


def main():
    key = jax.random.PRNGKey(0)

    # 更复杂的树结构（9 个节点，不均匀分支 + 更深层级）：
    # 0(root) -> {1,2}
    # 1 -> {3,4}
    # 2 -> {5,6}
    # 4 -> {7}
    # 6 -> {8}
    parent = [0, 0, 0, 1, 1, 2, 2, 4, 6]
    dims = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    rank = 2

    print("=== TTNS Validation Config ===")
    print(f"parent = {parent}")
    print(f"dims   = {dims}")
    print(f"rank   = {rank}")
    print(f"n_nodes= {len(parent)}")
    print("==============================")

    key_t1, key_t2, key_vec, key_mat = jax.random.split(key, 4)
    t1 = build_random_ttns_opt(key_t1, parent, dims, rank)
    t2 = build_random_ttns_opt(key_t2, parent, dims, rank)

    dense_t1 = ttns_opt_to_dense(t1, parent)
    dense_t2 = ttns_opt_to_dense(t2, parent)
    print(f"dense_t1 shape = {dense_t1.shape}")
    print(f"dense_t2 shape = {dense_t2.shape}")
    print(f"dense_t1 dtype = {dense_t1.dtype}")
    print(f"dense_t2 dtype = {dense_t2.dtype}")

    # 1) 内积
    normalized_ip = normalized_inner_product_ttns(t1, t2, parent)
    got_ip = recover_scalar(normalized_ip)
    expected_ip = jnp.sum(dense_t1 * dense_t2)
    assert_close("inner_product", got_ip, expected_ip)

    # 2) rank-1 评估
    vectors = jax.random.normal(key_vec, (len(dims), dims[0]), dtype=jnp.float64)

    normalized_eval = normalized_eval_rank1_ttns(t1, vectors, parent)
    got_eval = recover_scalar(normalized_eval)
    expected_eval = dense_rank1_eval(dense_t1, vectors)
    assert_close("rank1_eval", got_eval, expected_eval)

    # 3) 二次型 <T, A T>，A 为各维局部矩阵的 Kronecker 结构
    mats = []
    mat_keys = jax.random.split(key_mat, len(dims))
    for i, mk in enumerate(mat_keys):
        a = jax.random.normal(mk, (dims[i], dims[i]))
        a = a.astype(jnp.float64)
        mats.append(a.T @ a + 1e-2 * jnp.eye(dims[i]))  # SPD，减少数值病态
    matrices = jnp.stack(mats, axis=0)

    normalized_quad = normalized_quadratic_form_ttns(t1, matrices, parent)
    got_quad = recover_scalar(normalized_quad)
    expected_quad = dense_quadratic_form(dense_t1, matrices)
    assert_close("quadratic_form", got_quad, expected_quad, atol=3e-5, rtol=3e-5)

    validate_three_child_fanout(key)
    print("\n全部验证通过（6/6）。")


def validate_three_child_fanout(key: jnp.ndarray):
    """根节点 3 个子节点：覆盖 n_children==3 的融合 einsum 快路径。"""
    parent = [0, 0, 0, 0]
    dims = [3, 3, 3, 3]
    rank = 2

    print("\n=== TTNS 3-child fanout validation ===")
    print(f"parent = {parent}")
    print(f"dims   = {dims}")
    print(f"rank   = {rank}")

    key_t1, key_t2, key_vec, key_mat = jax.random.split(key, 4)
    t1 = build_random_ttns_opt(key_t1, parent, dims, rank)
    t2 = build_random_ttns_opt(key_t2, parent, dims, rank)

    dense_t1 = ttns_opt_to_dense(t1, parent)
    dense_t2 = ttns_opt_to_dense(t2, parent)

    normalized_ip = normalized_inner_product_ttns(t1, t2, parent)
    got_ip = recover_scalar(normalized_ip)
    expected_ip = jnp.sum(dense_t1 * dense_t2)
    assert_close("inner_product_3child", got_ip, expected_ip)

    vectors = jax.random.normal(key_vec, (len(dims), dims[0]), dtype=jnp.float64)
    normalized_eval = normalized_eval_rank1_ttns(t1, vectors, parent)
    got_eval = recover_scalar(normalized_eval)
    expected_eval = dense_rank1_eval(dense_t1, vectors)
    assert_close("rank1_eval_3child", got_eval, expected_eval)

    mats = []
    mat_keys = jax.random.split(key_mat, len(dims))
    for i, mk in enumerate(mat_keys):
        a = jax.random.normal(mk, (dims[i], dims[i]))
        a = a.astype(jnp.float64)
        mats.append(a.T @ a + 1e-2 * jnp.eye(dims[i]))
    matrices = jnp.stack(mats, axis=0)

    normalized_quad = normalized_quadratic_form_ttns(t1, matrices, parent)
    got_quad = recover_scalar(normalized_quad)
    expected_quad = dense_quadratic_form(dense_t1, matrices)
    assert_close("quadratic_form_3child", got_quad, expected_quad, atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    main()
