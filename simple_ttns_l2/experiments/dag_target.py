from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import jax
from jax import config, numpy as jnp

config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ForkDAGTargetSpec:
    """
    6 维 fork DAG 目标（多父节点依赖在节点 2）：

    - $x_0, x_1$：独立根节点（双模混合）
    - $x_2 = f(x_0, x_1) + \\epsilon$（**双父**）
    - $x_3, x_4$：依赖 $x_2$
    - $x_5$：依赖 $x_3, x_4$
    """

    n_dims: int = 6
    root_mix_prob: float = 0.5
    root_std: float = 0.7
    noise_2: float = 0.25
    noise_345: float = 0.22


def fork_dag_edges(n_dims: int = 6) -> Tuple[Tuple[int, int], ...]:
    """有向边 (parent, child)；节点 2 有两个父节点 0 和 1。"""
    if n_dims != 6:
        raise ValueError(f"fork DAG MVP 目前仅支持 n_dims=6，收到 {n_dims}")
    return (
        (0, 2),
        (1, 2),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
    )


def sample_fork_dag_distribution(
    key: jnp.ndarray,
    n_samples: int,
    spec: ForkDAGTargetSpec | None = None,
) -> jnp.ndarray:
    spec = spec or ForkDAGTargetSpec()
    n_dims = spec.n_dims
    x = jnp.zeros((n_samples, n_dims), dtype=jnp.float64)

    key, k0, k1, k2, k3, k4, k5 = jax.random.split(key, 7)

    def _root(key_a, key_b):
        z = jax.random.bernoulli(key_a, p=spec.root_mix_prob, shape=(n_samples,))
        g1 = jax.random.normal(key_a, (n_samples,), dtype=jnp.float64) * spec.root_std - 1.2
        g2 = jax.random.normal(key_b, (n_samples,), dtype=jnp.float64) * spec.root_std + 1.1
        return jnp.where(z, g1, g2)

    x = x.at[:, 0].set(_root(k0, k0))
    x = x.at[:, 1].set(_root(k1, k1))

    u0, u1 = x[:, 0], x[:, 1]
    mean2 = 0.45 * u0 + 0.40 * u1 + 0.12 * jnp.sin(1.8 * u0 * u1)
    eps2 = jax.random.normal(k2, (n_samples,), dtype=jnp.float64) * spec.noise_2
    x = x.at[:, 2].set(mean2 + eps2)

    u2 = x[:, 2]
    for node, kn, coef in ((3, k3, 0.62), (4, k4, -0.55)):
        mean = coef * u2 + 0.08 * (u2 ** 2 - 1.0)
        eps = jax.random.normal(kn, (n_samples,), dtype=jnp.float64) * spec.noise_345
        x = x.at[:, node].set(mean + eps)

    u3, u4 = x[:, 3], x[:, 4]
    mean5 = 0.50 * u3 + 0.48 * u4 + 0.06 * u3 * u4
    eps5 = jax.random.normal(k5, (n_samples,), dtype=jnp.float64) * spec.noise_345
    x = x.at[:, 5].set(mean5 + eps5)

    return x


def validate_samples_finite(samples: jnp.ndarray) -> None:
    if not jnp.all(jnp.isfinite(samples)):
        raise ValueError("fork DAG samples contain non-finite values")
