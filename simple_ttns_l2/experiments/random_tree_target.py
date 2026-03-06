from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
from jax import numpy as jnp


@dataclass(frozen=True)
class RandomTreeTargetSpec:
    parent: tuple[int, ...]
    root_mix_prob: float
    root_mean_a: float
    root_mean_b: float
    root_std_a: float
    root_std_b: float
    linear_coef: tuple[float, ...]
    quad_coef: tuple[float, ...]
    sin_coef: tuple[float, ...]
    sin_freq: tuple[float, ...]
    bias: tuple[float, ...]
    sigma_base: tuple[float, ...]
    sigma_slope: tuple[float, ...]


def random_recursive_tree_parent(key: jnp.ndarray, n_dims: int) -> list[int]:
    if n_dims < 2:
        return [0]
    parent = [0]
    for node in range(1, n_dims):
        key, curr_key = jax.random.split(key)
        parent.append(int(jax.random.randint(curr_key, (), 0, node)))
    return parent


def build_random_tree_target_spec(key: jnp.ndarray, n_dims: int) -> RandomTreeTargetSpec:
    key, k_tree, k_root, k_lin, k_quad, k_sin, k_freq, k_bias, k_sigma0, k_sigma1 = jax.random.split(key, 10)
    parent = tuple(random_recursive_tree_parent(k_tree, n_dims))

    root_mix_prob = float(jax.random.uniform(k_root, (), minval=0.35, maxval=0.65))
    root_mean_a = float(jax.random.uniform(k_root, (), minval=-1.8, maxval=-0.7))
    root_mean_b = float(jax.random.uniform(k_root, (), minval=0.7, maxval=1.8))
    root_std_a = float(jax.random.uniform(k_root, (), minval=0.45, maxval=0.85))
    root_std_b = float(jax.random.uniform(k_root, (), minval=0.40, maxval=0.80))

    linear = tuple(float(x) for x in jax.random.uniform(k_lin, (n_dims,), minval=0.35, maxval=0.85))
    quad = tuple(float(x) for x in jax.random.uniform(k_quad, (n_dims,), minval=-0.28, maxval=0.28))
    sin = tuple(float(x) for x in jax.random.uniform(k_sin, (n_dims,), minval=0.05, maxval=0.22))
    freq = tuple(float(x) for x in jax.random.uniform(k_freq, (n_dims,), minval=1.4, maxval=3.1))
    bias = tuple(float(x) for x in jax.random.uniform(k_bias, (n_dims,), minval=-0.18, maxval=0.18))
    sigma0 = tuple(float(x) for x in jax.random.uniform(k_sigma0, (n_dims,), minval=0.16, maxval=0.38))
    sigma1 = tuple(float(x) for x in jax.random.uniform(k_sigma1, (n_dims,), minval=0.02, maxval=0.12))

    return RandomTreeTargetSpec(
        parent=parent,
        root_mix_prob=root_mix_prob,
        root_mean_a=root_mean_a,
        root_mean_b=root_mean_b,
        root_std_a=root_std_a,
        root_std_b=root_std_b,
        linear_coef=linear,
        quad_coef=quad,
        sin_coef=sin,
        sin_freq=freq,
        bias=bias,
        sigma_base=sigma0,
        sigma_slope=sigma1,
    )


def sample_random_tree_distribution(
    key: jnp.ndarray,
    n_samples: int,
    spec: RandomTreeTargetSpec,
) -> jnp.ndarray:
    n_dims = len(spec.parent)
    x = jnp.zeros((n_samples, n_dims), dtype=jnp.float64)

    root = 0
    key, k_bern, k_g1, k_g2 = jax.random.split(key, 4)
    z = jax.random.bernoulli(k_bern, p=spec.root_mix_prob, shape=(n_samples,))
    g1 = jax.random.normal(k_g1, (n_samples,), dtype=jnp.float64) * spec.root_std_a + spec.root_mean_a
    g2 = jax.random.normal(k_g2, (n_samples,), dtype=jnp.float64) * spec.root_std_b + spec.root_mean_b
    x = x.at[:, root].set(jnp.where(z, g1, g2))

    for node in range(1, n_dims):
        parent = spec.parent[node]
        key, k_eps = jax.random.split(key)
        u = x[:, parent]
        mean = (
            spec.linear_coef[node] * u
            + spec.quad_coef[node] * (u ** 2 - 1.0)
            + spec.sin_coef[node] * jnp.sin(spec.sin_freq[node] * u)
            + spec.bias[node]
        )
        sigma = spec.sigma_base[node] + spec.sigma_slope[node] * jnp.abs(u)
        eps = jax.random.normal(k_eps, (n_samples,), dtype=jnp.float64) * sigma
        x = x.at[:, node].set(mean + eps)

    return x
