from __future__ import annotations

"""Core-parameter reparameterization for TTNS.

The stored core entries theta are the optimization variables. Before the cores
enter any contraction we optionally map every entry through an elementwise
transform g:

    identity : g(theta) = theta        (baseline; numerically unchanged)
    square   : g(theta) = theta**2     (non-negative cores)
    exp      : g(theta) = exp(theta)   (strictly positive cores)

With non-negative B-spline bases, non-negative cores force q(x) >= 0 everywhere,
i.e. a non-negative density model. This is a lighter-weight relative of the
p = psi^2 / Z squared parameterization (see squared_ttns_theory_zh.md): it keeps
the L2 objective and multilinear contraction, only constraining the sign of the
effective cores.

Normalization (scaling q by a positive factor `scale` through the root core) must
account for the transform, since the root core is a raw parameter theta while the
scale acts on the effective output g(theta):

    identity : theta_root <- theta_root * scale
    square   : theta_root <- theta_root * sqrt(scale)
    exp      : theta_root <- theta_root + log(scale)
"""

from typing import Sequence

from jax import numpy as jnp

from ttde.ttns.ttns_opt import TTNSOpt


TRANSFORMS = ("identity", "square", "exp")


def apply_transform(name: str, core: jnp.ndarray) -> jnp.ndarray:
    if name == "identity":
        return core
    if name == "square":
        return core * core
    if name == "exp":
        return jnp.exp(core)
    raise ValueError(f"unknown core transform: {name!r} (choose one of {TRANSFORMS})")


def effective_ttns(ttns: TTNSOpt, name: str = "identity") -> TTNSOpt:
    """Return a TTNS whose cores are g(theta). For identity this reuses the same
    arrays, so the numerics are bit-for-bit identical to the untransformed path."""
    if name == "identity":
        return ttns
    return TTNSOpt(tuple(apply_transform(name, c) for c in ttns.cores))


def rescale_root_for_scale(name: str, root_core: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Adjust the raw root parameter so that the effective output is multiplied by
    `scale` (assumed positive)."""
    if name == "identity":
        return root_core * scale
    if name == "square":
        return root_core * jnp.sqrt(scale)
    if name == "exp":
        return root_core + jnp.log(scale)
    raise ValueError(f"unknown core transform: {name!r} (choose one of {TRANSFORMS})")
