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

from simple_ttns_l2.experiments.dag_target import (
    fork_dag_edges,
    sample_fork_dag_distribution,
    validate_samples_finite,
)
from simple_ttns_l2.junction_tree import (
    clique_covers_dag_fork,
    dag_edges_to_child_parents,
    fork_junction_parent,
    validate_tree_parent,
)


class JunctionTreeTests(unittest.TestCase):
    def test_fork_junction_parent_is_valid_tree(self):
        parent = fork_junction_parent(6)
        validate_tree_parent(parent)
        self.assertEqual(parent[0], 0)
        self.assertEqual(len(parent), 6)

    def test_fork_dag_has_two_parents_at_node_2(self):
        parents = dag_edges_to_child_parents(fork_dag_edges())
        self.assertEqual(parents[2], {0, 1})

    def test_clique_covers_fork_triple(self):
        self.assertTrue(clique_covers_dag_fork(fork_dag_edges(), {0, 1, 2}))
        self.assertFalse(clique_covers_dag_fork(fork_dag_edges(), {0, 1}))

    def test_fork_junction_parent_7d_is_valid_tree(self):
        parent = fork_junction_parent(7)
        validate_tree_parent(parent)
        self.assertEqual(len(parent), 7)

    def test_fork_dag_7d_samples_are_finite(self):
        key = jax.random.PRNGKey(2)
        samples = sample_fork_dag_distribution(key, 128, n_dims=7)
        validate_samples_finite(samples)
        self.assertEqual(samples.shape, (128, 7))


if __name__ == "__main__":
    unittest.main()
