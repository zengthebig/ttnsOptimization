from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree
from simple_ttns_l2.experiments.random_tree_target import (
    build_random_tree_target_spec,
    random_recursive_tree_parent,
    sample_random_tree_distribution,
)


class RandomTreeTargetTests(unittest.TestCase):
    def test_random_recursive_tree_parent_is_valid(self):
        parent = random_recursive_tree_parent(jax.random.PRNGKey(0), 8)
        self.assertEqual(parent[0], 0)
        self.assertEqual(len(parent), 8)
        for node in range(1, 8):
            self.assertGreaterEqual(parent[node], 0)
            self.assertLess(parent[node], node)

    def test_random_tree_target_sampling_and_chow_liu_are_finite(self):
        spec = build_random_tree_target_spec(jax.random.PRNGKey(1), 8)
        samples = sample_random_tree_distribution(jax.random.PRNGKey(2), 512, spec)
        self.assertEqual(samples.shape, (512, 8))
        self.assertTrue(bool(jnp_all_finite(samples)))

        cl = estimate_chow_liu_tree(np.asarray(samples), n_bins=12, root=0)
        self.assertEqual(len(cl.parent), 8)
        self.assertEqual(cl.parent[0], 0)
        self.assertEqual(len(cl.edges), 7)


def jnp_all_finite(x) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x))))


if __name__ == "__main__":
    unittest.main()
