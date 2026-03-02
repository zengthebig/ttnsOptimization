from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path
from typing import List, Sequence

import jax
import numpy as np
from jax import config, numpy as jnp, value_and_grad

# Ensure this test imports local simple_ttns_l2 and TTNSDE/ttde.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from ttde.tt.tensors import TTNS
from ttde.ttns.ttns_opt import (
    TTNSOpt,
    eval_rank1_ttns,
    normalized_eval_rank1_ttns,
    quadratic_form_ttns,
    normalized_quadratic_form_ttns,
)
from simple_ttns_l2.objective import l2_objective_ttns, normalize_ttns_by_integral

config.update("jax_enable_x64", True)


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


def dense_rank1_eval(tensor: jnp.ndarray, vectors: jnp.ndarray) -> jnp.ndarray:
    tensor_np = np.asarray(tensor)
    vectors_np = np.asarray(vectors)
    total = 0.0
    dims = tensor_np.shape
    for multi_idx in product(*[range(s) for s in dims]):
        phi = 1.0
        for axis, idx in enumerate(multi_idx):
            phi *= vectors_np[axis, idx]
        total += tensor_np[multi_idx] * phi
    return jnp.asarray(total, dtype=jnp.float64)


def dense_quadratic_form(tensor: jnp.ndarray, matrices: jnp.ndarray) -> jnp.ndarray:
    tensor_np = np.asarray(tensor)
    matrices_np = np.asarray(matrices)
    total = 0.0
    dims = tensor_np.shape
    all_indices = list(product(*[range(s) for s in dims]))
    for idx_i in all_indices:
        t_i = tensor_np[idx_i]
        for idx_j in all_indices:
            a_ij = 1.0
            for axis, (ii, jj) in enumerate(zip(idx_i, idx_j)):
                a_ij *= matrices_np[axis, ii, jj]
            total += t_i * a_ij * tensor_np[idx_j]
    return jnp.asarray(total, dtype=jnp.float64)


def perturb_ttns_entry(ttns: TTNSOpt, node: int, index: tuple[int, ...], delta: float) -> TTNSOpt:
    cores = list(ttns.cores)
    cores[node] = cores[node].at[index].add(delta)
    return TTNSOpt(tuple(cores))


class TTNSL2ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.parent = [0, 0, 0, 1, 1, 2, 2]
        self.dims = [2, 2, 2, 2, 2, 2, 2]
        self.rank = 2
        key = jax.random.PRNGKey(7)
        k_t, k_b, k_g, k_i = jax.random.split(key, 4)

        self.ttns = build_random_ttns_opt(k_t, self.parent, self.dims, self.rank)
        self.dense_ttns = ttns_opt_to_dense(self.ttns, self.parent)

        self.batch_sz = 16
        self.basis_vectors_batch = jax.random.normal(
            k_b, (self.batch_sz, len(self.dims), self.dims[0]), dtype=jnp.float64
        )

        mats = []
        for mk in jax.random.split(k_g, len(self.dims)):
            a = jax.random.normal(mk, (self.dims[0], self.dims[0]), dtype=jnp.float64)
            mats.append(a.T @ a + 1e-2 * jnp.eye(self.dims[0], dtype=jnp.float64))
        self.gram_matrices = jnp.stack(mats, axis=0)

        self.basis_integrals = jax.random.normal(k_i, (len(self.dims), self.dims[0]), dtype=jnp.float64)

    def test_l2_objective_matches_dense_formula(self):
        got = l2_objective_ttns(
            self.ttns,
            self.basis_vectors_batch,
            self.gram_matrices,
            self.parent,
        )

        int_q2 = dense_quadratic_form(self.dense_ttns, self.gram_matrices)
        mc_q = jnp.mean(
            jnp.array(
                [dense_rank1_eval(self.dense_ttns, vectors) for vectors in np.asarray(self.basis_vectors_batch)]
            )
        )
        expected = int_q2 - 2.0 * mc_q

        self.assertTrue(
            bool(jnp.allclose(got, expected, rtol=1e-9, atol=1e-9)),
            msg=f"L2 mismatch: got={got}, expected={expected}",
        )

    def test_fast_and_normalized_eval_match(self):
        vectors = self.basis_vectors_batch[0]
        fast = eval_rank1_ttns(self.ttns, vectors, self.parent)
        stable_obj = normalized_eval_rank1_ttns(self.ttns, vectors, self.parent)
        stable = stable_obj.value * jnp.exp(stable_obj.log_norm)
        self.assertTrue(
            bool(jnp.allclose(fast, stable, rtol=1e-9, atol=1e-9)),
            msg=f"eval mismatch: fast={fast}, stable={stable}",
        )

    def test_fast_and_normalized_quadratic_match(self):
        fast = quadratic_form_ttns(self.ttns, self.gram_matrices, self.parent)
        stable_obj = normalized_quadratic_form_ttns(self.ttns, self.gram_matrices, self.parent)
        stable = stable_obj.value * jnp.exp(stable_obj.log_norm)
        self.assertTrue(
            bool(jnp.allclose(fast, stable, rtol=1e-9, atol=1e-9)),
            msg=f"quadratic mismatch: fast={fast}, stable={stable}",
        )

    def test_normalize_ttns_by_integral_matches_dense_scaling(self):
        z_dense = dense_rank1_eval(self.dense_ttns, self.basis_integrals)
        normalized_ttns, z = normalize_ttns_by_integral(
            self.ttns,
            self.basis_integrals,
            self.parent,
            eps=1e-12,
        )
        dense_norm = ttns_opt_to_dense(normalized_ttns, self.parent)
        z_norm_dense = dense_rank1_eval(dense_norm, self.basis_integrals)

        self.assertTrue(bool(jnp.allclose(z, z_dense, rtol=1e-9, atol=1e-9)))
        if bool(jnp.abs(z_dense) > 1e-8):
            self.assertTrue(bool(jnp.allclose(z_norm_dense, 1.0, rtol=1e-8, atol=1e-8)))

            test_vectors = self.basis_vectors_batch[0]
            q_before = dense_rank1_eval(self.dense_ttns, test_vectors)
            q_after = dense_rank1_eval(dense_norm, test_vectors)
            self.assertTrue(
                bool(jnp.allclose(q_after, q_before / z_dense, rtol=1e-8, atol=1e-8)),
                msg=f"normalized eval mismatch: before={q_before}, after={q_after}, z={z_dense}",
            )

    def test_l2_objective_autodiff_gradient_matches_finite_difference(self):
        node = 3
        index = (0, 0)
        eps = 1e-5

        def loss_fn(curr_ttns: TTNSOpt):
            return l2_objective_ttns(
                curr_ttns,
                self.basis_vectors_batch,
                self.gram_matrices,
                self.parent,
            )

        loss, grads = value_and_grad(loss_fn)(self.ttns)
        grad_auto = grads.cores[node][index]

        ttns_pos = perturb_ttns_entry(self.ttns, node, index, +eps)
        ttns_neg = perturb_ttns_entry(self.ttns, node, index, -eps)
        grad_fd = (loss_fn(ttns_pos) - loss_fn(ttns_neg)) / (2.0 * eps)

        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(jnp.isfinite(grad_auto)))
        self.assertTrue(bool(jnp.isfinite(grad_fd)))
        self.assertTrue(
            bool(jnp.allclose(grad_auto, grad_fd, rtol=1e-3, atol=1e-3)),
            msg=f"grad mismatch: autodiff={grad_auto}, finite_diff={grad_fd}",
        )


if __name__ == "__main__":
    unittest.main()
