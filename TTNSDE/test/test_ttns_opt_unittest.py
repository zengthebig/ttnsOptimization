from __future__ import annotations

import unittest
from itertools import product
from typing import List, Sequence
from pathlib import Path
import sys
import numpy as np

# Ensure this test imports TTNSDE/ttde, not the repository-root ttde package.
THIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_ROOT))

import jax
from jax import numpy as jnp
from jax import config
import optax
from jax import value_and_grad

from ttde.tt.tensors import TTNS
from ttde.ttns.ttns_opt import (
    TTNSOpt,
    add_ttns,
    normalized_eval_rank1_ttns,
    normalized_inner_product_ttns,
    normalized_quadratic_form_ttns,
    subtract_ttns,
)
from ttde.score.experiment_setups import model_setups
from ttde.tt.losses import LLLoss

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


def recover_scalar(normalized_value) -> jnp.ndarray:
    return normalized_value.value * jnp.exp(normalized_value.log_norm)


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


def dense_quadratic_form(
    tensor: jnp.ndarray,
    matrices: jnp.ndarray,
) -> jnp.ndarray:
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


class TTNSOptMathTests(unittest.TestCase):
    def setUp(self):
        self.parent = [0, 0, 0, 1, 1, 2, 2, 4, 6]
        self.dims = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.rank = 2
        key = jax.random.PRNGKey(0)
        k1, k2, kv, km = jax.random.split(key, 4)
        self.t1 = build_random_ttns_opt(k1, self.parent, self.dims, self.rank)
        self.t2 = build_random_ttns_opt(k2, self.parent, self.dims, self.rank)
        self.dense_t1 = ttns_opt_to_dense(self.t1, self.parent)
        self.dense_t2 = ttns_opt_to_dense(self.t2, self.parent)
        self.vectors = jax.random.normal(kv, (len(self.dims), self.dims[0]), dtype=jnp.float64)

        mats = []
        for i, mk in enumerate(jax.random.split(km, len(self.dims))):
            a = jax.random.normal(mk, (self.dims[i], self.dims[i]), dtype=jnp.float64)
            mats.append(a.T @ a + 1e-2 * jnp.eye(self.dims[i], dtype=jnp.float64))
        self.matrices = jnp.stack(mats, axis=0)
        self._printed_config = False

    def _print_config_once(self):
        if self._printed_config:
            return
        print("\n=== TTNS UnitTest Config ===")
        print(f"parent = {self.parent}")
        print(f"dims   = {self.dims}")
        print(f"rank   = {self.rank}")
        print(f"dense_t1.shape = {self.dense_t1.shape}, dtype={self.dense_t1.dtype}")
        print(f"dense_t2.shape = {self.dense_t2.shape}, dtype={self.dense_t2.dtype}")
        print("============================")
        self._printed_config = True

    def _assert_close_verbose(self, name: str, got: jnp.ndarray, expected: jnp.ndarray, rtol: float, atol: float):
        self._print_config_once()
        abs_err = jnp.max(jnp.abs(got - expected))
        denom = jnp.maximum(jnp.max(jnp.abs(expected)), 1e-12)
        rel_err = abs_err / denom
        print(f"\n[{name}]")
        print(f"  got      = {got}")
        print(f"  expected = {expected}")
        print(f"  abs_err  = {abs_err}")
        print(f"  rel_err  = {rel_err}")
        if (got > 0) and (expected > 0):
            print(f"  |log_diff| = {jnp.abs(jnp.log(got) - jnp.log(expected))}")
        print(f"  tol      = atol={atol}, rtol={rtol}")
        ok = bool(jnp.allclose(got, expected, rtol=rtol, atol=atol))
        print(f"  status   = {'PASS' if ok else 'FAIL'}")
        self.assertTrue(
            ok,
            msg=f"[{name}] mismatch: got={got}, expected={expected}, abs_err={abs_err}, rel_err={rel_err}",
        )

    def test_inner_product_matches_dense(self):
        got = recover_scalar(normalized_inner_product_ttns(self.t1, self.t2, self.parent))
        expected = jnp.sum(self.dense_t1 * self.dense_t2)
        self._assert_close_verbose("inner_product", got, expected, rtol=1e-9, atol=1e-9)

    def test_rank1_eval_matches_dense(self):
        got = recover_scalar(normalized_eval_rank1_ttns(self.t1, self.vectors, self.parent))
        expected = dense_rank1_eval(self.dense_t1, self.vectors)
        self._assert_close_verbose("rank1_eval", got, expected, rtol=1e-9, atol=1e-9)

    def test_quadratic_form_matches_dense(self):
        got = recover_scalar(normalized_quadratic_form_ttns(self.t1, self.matrices, self.parent))
        expected = dense_quadratic_form(self.dense_t1, self.matrices)
        self._assert_close_verbose("quadratic_form", got, expected, rtol=1e-8, atol=1e-8)

    def test_chain_tree_case(self):
        parent = [0, 0, 1, 2, 3]
        dims = [2, 2, 2, 2, 2]
        rank = 2
        key = jax.random.PRNGKey(11)
        t = build_random_ttns_opt(key, parent, dims, rank)
        dense = ttns_opt_to_dense(t, parent)
        vectors = jax.random.normal(jax.random.PRNGKey(12), (len(dims), 2), dtype=jnp.float64)
        got = recover_scalar(normalized_eval_rank1_ttns(t, vectors, parent))
        expected = dense_rank1_eval(dense, vectors)
        print("\n=== Chain Tree Case ===")
        print(f"parent = {parent}, dims = {dims}, rank = {rank}, dense.shape = {dense.shape}")
        self._assert_close_verbose("chain_rank1_eval", got, expected, rtol=1e-9, atol=1e-9)

    def test_add_matches_dense(self):
        t_sum = add_ttns(self.t1, self.t2, self.parent)
        got = ttns_opt_to_dense(t_sum, self.parent)
        expected = self.dense_t1 + self.dense_t2
        abs_err = jnp.max(jnp.abs(got - expected))
        self.assertTrue(
            bool(jnp.allclose(got, expected, rtol=1e-9, atol=1e-9)),
            msg=f"add mismatch: abs_err={abs_err}",
        )

    def test_subtract_matches_dense(self):
        t_diff = subtract_ttns(self.t1, self.t2, self.parent)
        got = ttns_opt_to_dense(t_diff, self.parent)
        expected = self.dense_t1 - self.dense_t2
        abs_err = jnp.max(jnp.abs(got - expected))
        self.assertTrue(
            bool(jnp.allclose(got, expected, rtol=1e-9, atol=1e-9)),
            msg=f"subtract mismatch: abs_err={abs_err}",
        )


class TTNSTrainSmokeTests(unittest.TestCase):
    def test_ttns_one_batch_five_steps(self):
        print("\n=== TTNS Train Smoke (5 steps) ===")
        key = jax.random.PRNGKey(123)
        k_data, k_model, k_init = jax.random.split(key, 3)

        n_samples, n_dims = 128, 6
        samples = jax.random.normal(k_data, (n_samples, n_dims), dtype=jnp.float64)
        batch = samples[:64]

        setup = model_setups.PAsTTNSSqrOpt(q=2, m=16, rank=2, n_comps=2)
        model = setup.create(k_model, samples)

        params = model.init(k_init)
        params = model.mutate(params, k_init, samples, n_steps=0, method=model.init_canonical)

        loss_fn = LLLoss()
        optimizer = optax.adam(learning_rate=1e-3)
        optim_state = optimizer.init(params)

        def wrapped_loss(p):
            return loss_fn(model, p, batch, batch_sz=64)

        losses = []
        params_before = params
        for step in range(5):
            loss_value, grads = value_and_grad(wrapped_loss)(params)
            updates, optim_state = optimizer.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)

            grad_leaves = jax.tree_util.tree_leaves(grads)
            max_grad = max(float(jnp.max(jnp.abs(g))) for g in grad_leaves)

            losses.append(float(loss_value))
            print(f"step={step:02d} loss={float(loss_value):.10f} max|grad|={max_grad:.6e}")
            self.assertTrue(bool(jnp.isfinite(loss_value)))
            self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves))

        param_delta_leaves = jax.tree_util.tree_map(
            lambda a, b: jnp.max(jnp.abs(a - b)),
            params_before,
            params,
        )
        max_param_delta = max(float(x) for x in jax.tree_util.tree_leaves(param_delta_leaves))
        print(f"max|param_delta|={max_param_delta:.6e}")

        self.assertGreater(max_param_delta, 0.0, "parameters did not update")
        self.assertLess(min(losses), losses[0], "loss never improved within 5 steps")


if __name__ == "__main__":
    unittest.main()
