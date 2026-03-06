from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree
from ttde.score.experiment_setups.model_setups import PAsTTNSSqrOpt
from ttde.score.models.opt_for_tree_data import normalize_tree_parent


class ChowLiuTests(unittest.TestCase):
    def test_estimate_chow_liu_tree_returns_valid_parent(self):
        key = jax.random.PRNGKey(0)
        k0, k1, k2, k3 = jax.random.split(key, 4)
        n_samples = 2048

        x0 = jax.random.normal(k0, (n_samples,))
        x1 = x0 + 0.05 * jax.random.normal(k1, (n_samples,))
        x2 = x1 + 0.05 * jax.random.normal(k2, (n_samples,))
        x3 = x0 - x2 + 0.05 * jax.random.normal(k3, (n_samples,))
        samples = np.asarray(np.stack([x0, x1, x2, x3], axis=1), dtype=np.float64)

        result = estimate_chow_liu_tree(samples, n_bins=12, root=0)

        self.assertEqual(len(result.parent), 4)
        self.assertEqual(result.parent[0], 0)
        self.assertEqual(len(result.edges), 3)
        self.assertEqual(result.mutual_information.shape, (4, 4))
        for node, parent in enumerate(result.parent):
            if node == 0:
                continue
            self.assertNotEqual(node, parent)
            self.assertGreaterEqual(parent, 0)
            self.assertLess(parent, 4)

    def test_custom_tree_parent_reaches_model_setup(self):
        key = jax.random.PRNGKey(1)
        samples = jax.random.normal(key, (64, 5))
        requested_parent = (2, 0, 2, 2, 3)
        normalized_parent = tuple(int(x) for x in normalize_tree_parent(requested_parent, 5))

        model_setup = PAsTTNSSqrOpt(
            q=2,
            m=12,
            rank=2,
            n_comps=1,
            tree_topology="balanced",
            tree_parent=requested_parent,
        )
        model = model_setup.create(jax.random.PRNGKey(3), samples)

        actual_parent = tuple(int(x) for x in np.asarray(model.tree_parent))
        self.assertEqual(actual_parent, normalized_parent)


if __name__ == "__main__":
    unittest.main()
