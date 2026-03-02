from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from typing import Any

import jax
from jax import numpy as jnp

# Ensure this test imports TTNSDE/ttde, not repository-root ttde package.
THIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_ROOT))

from ttde.score.experiment_setups import init_setups
from ttde.score.models import opt_for_table_data, opt_for_tree_data
from ttde.tt import losses


COMMON_PROTOCOL_METHODS = (
    "unnormalized_log_p",
    "log_p",
    "log_int_p",
    "tt_log_sqr_norm",
    "init_rank_1",
    "init_canonical",
    "add_noise",
)


class DummyModel:
    def __init__(self):
        self.calls: list[tuple[str, Any]] = []
        self.params = {"params": {"dummy": jnp.array(1.0)}}

    def init(self, key: jnp.ndarray):
        self.calls.append(("init", key))
        return self.params

    def init_canonical(self, key: jnp.ndarray, samples: jnp.ndarray, n_steps: int):
        del key, samples, n_steps

    def add_noise(self, key: jnp.ndarray, noise: float):
        del key, noise

    def mutate(self, variables, *args, rngs=None, method=None, **kwargs):
        del rngs
        self.calls.append(("mutate", method.__name__, args, kwargs))
        return variables


class TrainingProtocolTests(unittest.TestCase):
    def test_tt_and_ttns_expose_common_protocol_methods(self):
        for cls in (opt_for_table_data.PAsTTSqrOpt, opt_for_tree_data.PAsTTNSSqrOpt):
            for method_name in COMMON_PROTOCOL_METHODS:
                with self.subTest(model_class=cls.__name__, method=method_name):
                    self.assertTrue(hasattr(cls, method_name), f"{cls.__name__} missing {method_name}")
                    self.assertTrue(callable(getattr(cls, method_name)), f"{cls.__name__}.{method_name} is not callable")

    def test_canonical_rankk_keeps_init_then_canonical_then_noise_order(self):
        model = DummyModel()
        setup = init_setups.CanonicalRankK(em_steps=7, noise=0.03)
        key = jax.random.PRNGKey(0)
        samples = jnp.ones((8, 3), dtype=jnp.float64)

        params = setup(model, key, samples)

        self.assertIs(params, model.params)
        self.assertEqual(model.calls[0][0], "init")
        self.assertEqual(model.calls[1][0], "mutate")
        self.assertEqual(model.calls[2][0], "mutate")

        first_mutate = model.calls[1]
        second_mutate = model.calls[2]

        self.assertEqual(first_mutate[1], "init_canonical")
        self.assertEqual(second_mutate[1], "add_noise")

        self.assertEqual(first_mutate[3]["n_steps"], 7)
        self.assertEqual(second_mutate[2][1], 0.03)
        self.assertTrue(jnp.array_equal(first_mutate[2][1], samples))

        init_key = model.calls[0][1]
        canonical_key = first_mutate[2][0]
        noise_key = second_mutate[2][0]
        self.assertFalse(bool(jnp.array_equal(init_key, canonical_key)))
        self.assertFalse(bool(jnp.array_equal(canonical_key, noise_key)))
        self.assertFalse(bool(jnp.array_equal(init_key, noise_key)))

    def test_llloss_uses_model_log_p_path(self):
        source = inspect.getsource(losses.LLLoss.__call__)
        compact = "".join(source.split())
        self.assertIn("method=model.log_p", compact)


if __name__ == "__main__":
    unittest.main()
