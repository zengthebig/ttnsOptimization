from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jax
import optax
from jax import config, numpy as jnp, tree_util

# Ensure this test imports local simple_ttns_l2 and TTNSDE/ttde.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from ttde.score.models.opt_for_tree_data import chain_parent
from simple_ttns_l2.train_l2 import (
    build_bases,
    init_ttns_from_rank1,
    l2_train_step,
    l2_loss_on_batch,
)
from simple_ttns_l2.objective import integral_q_ttns, normalize_ttns_by_integral

config.update("jax_enable_x64", True)


class TrainL2SmokeTests(unittest.TestCase):
    def test_l2_step_and_normalization(self):
        key = jax.random.PRNGKey(2026)
        k_samples, k_init, k_batch, k_noise = jax.random.split(key, 4)

        n_samples, n_dims = 128, 5
        samples = jax.random.normal(k_samples, (n_samples, n_dims), dtype=jnp.float64)
        batch = jax.random.normal(k_batch, (32, n_dims), dtype=jnp.float64)

        bases = build_bases(samples, q=2, m=16)
        parent = chain_parent(n_dims).tolist()
        gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
        basis_integrals = jax.vmap(type(bases).integral)(bases)

        ttns = init_ttns_from_rank1(
            key=k_init,
            bases=bases,
            samples=samples,
            parent=parent,
            rank=2,
            noise=1e-2,
        )
        ttns, z0 = normalize_ttns_by_integral(ttns, basis_integrals, parent)
        self.assertTrue(bool(jnp.isfinite(z0)))

        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(ttns)

        loss0 = l2_loss_on_batch(ttns, bases, batch, parent, gram_matrices)
        ttns_new, opt_state_new, loss1 = l2_train_step(
            ttns=ttns,
            opt_state=opt_state,
            optimizer=optimizer,
            bases=bases,
            xs_batch=batch,
            parent=parent,
            gram_matrices=gram_matrices,
            key_noise=k_noise,
            train_noise=0.01,
        )
        ttns_new, _ = normalize_ttns_by_integral(ttns_new, basis_integrals, parent)
        z1 = integral_q_ttns(ttns_new, basis_integrals, parent)

        self.assertTrue(bool(jnp.isfinite(loss0)))
        self.assertTrue(bool(jnp.isfinite(loss1)))
        self.assertTrue(bool(jnp.allclose(z1, 1.0, rtol=1e-8, atol=1e-8)))

        delta_tree = tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), ttns, ttns_new)
        max_delta = max(float(x) for x in tree_util.tree_leaves(delta_tree))
        self.assertGreater(max_delta, 0.0, "TTNS parameters did not update")

        _ = opt_state_new  # silence linters in case of future static checks


if __name__ == "__main__":
    unittest.main()
