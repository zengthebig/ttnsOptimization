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
    """有向边 (parent, child)。6D MVP 或 8D 扩展版。"""
    if n_dims == 6:
        return (
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 5),
            (4, 5),
        )
    if n_dims == 7:
        return fork_dag_edges_7d()
    if n_dims == 8:
        return fork_dag_edges_8d()
    raise ValueError(f"fork DAG 目前支持 n_dims=6/7/8，收到 {n_dims}")


def fork_dag_edges_7d() -> Tuple[Tuple[int, int], ...]:
    """7 维： nuisance chain + 远端双父 x6。"""
    return (
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (2, 6),
        (5, 6),
    )


def fork_dag_edges_8d() -> Tuple[Tuple[int, int], ...]:
    """
    8 维 fork DAG： nuisance chain 拉长 chain 拓扑路径，联结树保持 x7–x2 短路径。

    - x2 ~ f(x0,x1)；x3..x6 为弱 nuisance chain
    - x7 ~ g(x2, x6)（双父，且 x2 与 x6 在 chain 上相距远）
    """
    return (
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (2, 7),
        (6, 7),
    )


@dataclass(frozen=True)
class ForkDAGTargetSpec7D:
    n_dims: int = 7
    root_mix_prob: float = 0.5
    root_std: float = 0.7
    noise_2: float = 0.22
    noise_nuisance: float = 0.18
    nuisance_coef: float = 0.06
    noise_6: float = 0.18


def sample_fork_dag_distribution(
    key: jnp.ndarray,
    n_samples: int,
    spec: ForkDAGTargetSpec | ForkDAGTargetSpec7D | ForkDAGTargetSpec8D | None = None,
    n_dims: int | None = None,
) -> jnp.ndarray:
    if n_dims == 8 or (spec is not None and getattr(spec, "n_dims", 6) == 8):
        return sample_fork_dag_8d(key, n_samples, spec if isinstance(spec, ForkDAGTargetSpec8D) else None)
    if n_dims == 7 or (spec is not None and getattr(spec, "n_dims", 6) == 7):
        return sample_fork_dag_7d(key, n_samples, spec if isinstance(spec, ForkDAGTargetSpec7D) else None)
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


def sample_fork_dag_7d(
    key: jnp.ndarray,
    n_samples: int,
    spec: ForkDAGTargetSpec7D | None = None,
) -> jnp.ndarray:
    spec = spec or ForkDAGTargetSpec7D()
    n_dims = spec.n_dims
    x = jnp.zeros((n_samples, n_dims), dtype=jnp.float64)

    key, k0, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 8)

    def _root(key_a, key_b):
        z = jax.random.bernoulli(key_a, p=spec.root_mix_prob, shape=(n_samples,))
        g1 = jax.random.normal(key_a, (n_samples,), dtype=jnp.float64) * spec.root_std - 1.2
        g2 = jax.random.normal(key_b, (n_samples,), dtype=jnp.float64) * spec.root_std + 1.1
        return jnp.where(z, g1, g2)

    x = x.at[:, 0].set(_root(k0, k0))
    x = x.at[:, 1].set(_root(k1, k1))

    u0, u1 = x[:, 0], x[:, 1]
    mean2 = 0.50 * u0 + 0.42 * u1 + 0.14 * jnp.sin(1.9 * u0 * u1)
    eps2 = jax.random.normal(k2, (n_samples,), dtype=jnp.float64) * spec.noise_2
    x = x.at[:, 2].set(mean2 + eps2)

    u2 = x[:, 2]
    keys_n = [k3, k4, k5]
    for idx, kn in enumerate(keys_n, start=3):
        parent_val = x[:, idx - 1]
        mean = spec.nuisance_coef * parent_val
        eps = jax.random.normal(kn, (n_samples,), dtype=jnp.float64) * spec.noise_nuisance
        x = x.at[:, idx].set(mean + eps)

    u5 = x[:, 5]
    mean6 = (
        0.58 * u2
        + 0.52 * u5
        + 0.16 * jnp.sin(2.1 * u2 * u5)
        + 0.08 * (u2 ** 2 - u5 ** 2)
    )
    eps6 = jax.random.normal(k6, (n_samples,), dtype=jnp.float64) * spec.noise_6
    x = x.at[:, 6].set(mean6 + eps6)

    return x


@dataclass(frozen=True)
class ForkDAGTargetSpec8D:
    """8 维 fork DAG： nuisance chain + 远端双父节点 x7。"""

    n_dims: int = 8
    root_mix_prob: float = 0.5
    root_std: float = 0.7
    noise_2: float = 0.22
    noise_nuisance: float = 0.18
    nuisance_coef: float = 0.12
    noise_7: float = 0.20


def sample_fork_dag_8d(
    key: jnp.ndarray,
    n_samples: int,
    spec: ForkDAGTargetSpec8D | None = None,
) -> jnp.ndarray:
    spec = spec or ForkDAGTargetSpec8D()
    n_dims = spec.n_dims
    x = jnp.zeros((n_samples, n_dims), dtype=jnp.float64)

    key, k0, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 9)

    def _root(key_a, key_b):
        z = jax.random.bernoulli(key_a, p=spec.root_mix_prob, shape=(n_samples,))
        g1 = jax.random.normal(key_a, (n_samples,), dtype=jnp.float64) * spec.root_std - 1.2
        g2 = jax.random.normal(key_b, (n_samples,), dtype=jnp.float64) * spec.root_std + 1.1
        return jnp.where(z, g1, g2)

    x = x.at[:, 0].set(_root(k0, k0))
    x = x.at[:, 1].set(_root(k1, k1))

    u0, u1 = x[:, 0], x[:, 1]
    mean2 = 0.50 * u0 + 0.42 * u1 + 0.14 * jnp.sin(1.9 * u0 * u1)
    eps2 = jax.random.normal(k2, (n_samples,), dtype=jnp.float64) * spec.noise_2
    x = x.at[:, 2].set(mean2 + eps2)

    u2 = x[:, 2]
    keys_n = [k3, k4, k5, k6]
    for idx, kn in enumerate(keys_n, start=3):
        parent_val = x[:, idx - 1]
        mean = spec.nuisance_coef * parent_val
        eps = jax.random.normal(kn, (n_samples,), dtype=jnp.float64) * spec.noise_nuisance
        x = x.at[:, idx].set(mean + eps)

    u6 = x[:, 6]
    mean7 = (
        0.58 * u2
        + 0.52 * u6
        + 0.16 * jnp.sin(2.1 * u2 * u6)
        + 0.08 * (u2 ** 2 - u6 ** 2)
    )
    eps7 = jax.random.normal(k7, (n_samples,), dtype=jnp.float64) * spec.noise_7
    x = x.at[:, 7].set(mean7 + eps7)

    return x


def validate_samples_finite(samples: jnp.ndarray) -> None:
    if not jnp.all(jnp.isfinite(samples)):
        raise ValueError("fork DAG samples contain non-finite values")
