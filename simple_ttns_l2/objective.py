from __future__ import annotations

import sys
from typing import Sequence, Tuple
from pathlib import Path

from jax import numpy as jnp, vmap

# Ensure this standalone package imports TTNSDE/ttde, not repository-root ttde.
REPO_ROOT = Path(__file__).resolve().parents[1]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from ttde.ttns.ttns_opt import (
    TTNSOpt,
    batch_eval_rank1_ttns,
    eval_rank1_ttns,
    quadratic_form_ttns,
    normalized_eval_rank1_ttns,
    normalized_quadratic_form_ttns,
)


def _recover_scalar(normalized_value) -> jnp.ndarray:
    return normalized_value.value * jnp.exp(normalized_value.log_norm)


def _root_from_parent(parent: Sequence[int]) -> int:
    return next(i for i, p in enumerate(parent) if p == i or p == -1)


def basis_vectors_from_sample(bases, x: jnp.ndarray) -> jnp.ndarray:
    """
    Build rank-1 basis vectors for one sample.
    Output shape: [n_dims, basis_dim].
    """
    return vmap(type(bases).__call__)(bases, x)


def batch_basis_vectors_from_samples(bases, xs: jnp.ndarray) -> jnp.ndarray:
    """
    Build rank-1 basis vectors for a batch.
    Output shape: [batch, n_dims, basis_dim].
    """
    return vmap(lambda one_x: basis_vectors_from_sample(bases, one_x))(xs)


def eval_q_ttns(
    ttns: TTNSOpt,
    basis_vectors: jnp.ndarray,
    parent: Sequence[int],
    stable: bool = False,
) -> jnp.ndarray:
    r"""
    q_theta(x) = <T_theta, \otimes_k b_k(x_k)>.
    """
    if stable:
        return _recover_scalar(normalized_eval_rank1_ttns(ttns, basis_vectors, parent))
    return eval_rank1_ttns(ttns, basis_vectors, parent)


def batch_eval_q_ttns(
    ttns: TTNSOpt,
    basis_vectors_batch: jnp.ndarray,
    parent: Sequence[int],
    stable: bool = False,
) -> jnp.ndarray:
    if not stable:
        return batch_eval_rank1_ttns(ttns, basis_vectors_batch, parent)
    return vmap(lambda vectors: eval_q_ttns(ttns, vectors, parent, stable=stable))(basis_vectors_batch)


def integral_q_ttns(
    ttns: TTNSOpt,
    basis_integrals: jnp.ndarray,
    parent: Sequence[int],
    stable: bool = False,
) -> jnp.ndarray:
    r"""
    \int q_theta(x) dx, where basis_integrals[k, i] = \int f_{k,i}(x_k) dx_k.
    """
    return eval_q_ttns(ttns, basis_integrals, parent, stable=stable)


def integral_q2_ttns(
    ttns: TTNSOpt,
    gram_matrices: jnp.ndarray,
    parent: Sequence[int],
    stable: bool = False,
) -> jnp.ndarray:
    r"""
    \int q_theta(x)^2 dx, where gram_matrices[k, i, j] = \int f_{k,i} f_{k,j}.
    """
    if stable:
        return _recover_scalar(normalized_quadratic_form_ttns(ttns, gram_matrices, parent))
    return quadratic_form_ttns(ttns, gram_matrices, parent)


def mc_expectation_q_ttns(
    ttns: TTNSOpt,
    basis_vectors_batch: jnp.ndarray,
    parent: Sequence[int],
    stable: bool = False,
) -> jnp.ndarray:
    return batch_eval_q_ttns(ttns, basis_vectors_batch, parent, stable=stable).mean()


def l2_objective_ttns(
    ttns: TTNSOpt,
    basis_vectors_batch: jnp.ndarray,
    gram_matrices: jnp.ndarray,
    parent: Sequence[int],
    stable: bool = False,
) -> jnp.ndarray:
    r"""
    L(theta) = \int q_theta^2 - 2 * E_data[q_theta].
    Constant term \int p^2 is dropped.
    """
    int_q2 = integral_q2_ttns(ttns, gram_matrices, parent, stable=stable)
    mc_q = mc_expectation_q_ttns(ttns, basis_vectors_batch, parent, stable=stable)
    return int_q2 - 2.0 * mc_q


def l2_objective_from_samples(
    ttns: TTNSOpt,
    bases,
    xs: jnp.ndarray,
    parent: Sequence[int],
) -> jnp.ndarray:
    basis_vectors_batch = batch_basis_vectors_from_samples(bases, xs)
    gram_matrices = vmap(type(bases).l2_integral)(bases)
    return l2_objective_ttns(ttns, basis_vectors_batch, gram_matrices, parent)


def normalize_ttns_by_integral(
    ttns: TTNSOpt,
    basis_integrals: jnp.ndarray,
    parent: Sequence[int],
    eps: float = 1e-12,
    stable: bool = False,
) -> Tuple[TTNSOpt, jnp.ndarray]:
    r"""
    Project to unit integral:
        q_new = q / \int q
    by scaling the root core.
    """
    z = integral_q_ttns(ttns, basis_integrals, parent, stable=stable)
    safe_z = jnp.where(jnp.abs(z) < eps, 1.0, z)
    scale = 1.0 / safe_z
    root = _root_from_parent(parent)
    cores = list(ttns.cores)
    cores[root] = cores[root] * scale
    return TTNSOpt(tuple(cores)), z
