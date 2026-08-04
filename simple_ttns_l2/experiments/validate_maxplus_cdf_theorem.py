"""Lightweight numerical validation for the TTNS max-plus CDF contraction theorem.

This script performs no model training and no large Monte Carlo experiment.  It
constructs a normalized, nonnegative three-dimensional TTNS density with rank 2
and compares:

1. TTNS contraction against the explicitly assembled coefficient tensor;
2. the pair CDF against direct three-dimensional grid integration;
3. output permutation, monotonicity, range, and marginal consistency;
4. the local quadrature result across increasing ``q_grid`` values.

The two outputs use parent sets {0, 1} and {1, 2}, so coordinate 1 is a shared
parent and exercises the nontrivial local product in the joint-CDF theorem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
for _path in (str(REPO_ROOT), str(TTNSDE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

jax.config.update("jax_enable_x64", True)

from ttde.tt.basis import SplineOnKnots  # noqa: E402
from ttde.ttns.ttns_opt import TTNSOpt, quadratic_form_ttns  # noqa: E402
from ttde.utils import tree_stack  # noqa: E402

from simple_ttns_l2.maxplus_cdf import UpperModel, marginal_cdf, pair_cdf  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import UpperForest, block_joint_cdf  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, edge_cdf  # noqa: E402
from simple_ttns_l2.layered_forest import BlockModel  # noqa: E402


PARENT = [0, 0, 0]
PARENTS_Y1 = [0, 1]
PARENTS_Y2 = [1, 2]
N_DIMS = 3
BASIS_DIM = 4
BASIS_DEGREE = 1
S_GRID = np.linspace(-1.0, 2.0, 31)
PARAMS = DelayParams(
    edge_lo=0.0,
    edge_hi=0.3,
    node_lo=0.0,
    node_hi=0.0,
    kind="uniform",
)


def _stack_identical_bases():
    one = SplineOnKnots.from_uniform_knots(0.0, 1.0, BASIS_DIM, BASIS_DEGREE)
    return tree_stack([one for _ in range(N_DIMS)])


def _construct_density():
    """Return bases, TTNS, and an independently assembled dense coefficient tensor."""
    bases = _stack_identical_bases()
    basis_integrals = np.asarray(jax.vmap(type(bases).integral)(bases), dtype=float)

    raw = np.asarray(
        [
            [
                [0.8, 1.5, 0.7, 0.3],
                [1.4, 0.4, 0.8, 1.1],
                [0.5, 1.2, 1.6, 0.4],
            ],
            [
                [0.3, 0.7, 1.4, 1.0],
                [0.6, 1.7, 0.5, 0.9],
                [1.5, 0.8, 0.4, 1.2],
            ],
        ],
        dtype=float,
    )
    vectors = raw.copy()
    for component in range(vectors.shape[0]):
        for dim in range(N_DIMS):
            normalizer = float(vectors[component, dim] @ basis_integrals[dim])
            vectors[component, dim] /= normalizer

    mixture_weights = np.asarray([0.4, 0.6])
    vectors[:, 0, :] *= mixture_weights[:, None]

    ttns = TTNSOpt.from_canonical_vectors(jnp.asarray(vectors), PARENT, rank=2)
    dense = np.einsum("ra,rb,rc->abc", vectors[:, 0], vectors[:, 1], vectors[:, 2])
    return bases, ttns, dense, basis_integrals


def _dense_marginal(
    upper: UpperModel,
    dense: np.ndarray,
    basis_integrals: np.ndarray,
) -> np.ndarray:
    m0 = upper.proj_single(0, S_GRID, PARAMS)
    m1 = upper.proj_single(1, S_GRID, PARAMS)
    return np.einsum("abc,sa,sb,c->s", dense, m0, m1, basis_integrals[2])


def _dense_pair(upper: UpperModel, dense: np.ndarray) -> np.ndarray:
    m0 = upper.proj_single(0, S_GRID, PARAMS)
    m1 = upper.proj_single_pair(1, S_GRID, S_GRID, PARAMS)
    m2 = upper.proj_single(2, S_GRID, PARAMS)
    return np.einsum("abc,sa,stb,tc->st", dense, m0, m1, m2)


def _direct_grid_pair_points(
    upper: UpperModel,
    dense: np.ndarray,
    pair_indices: list[tuple[int, int]],
) -> dict[str, float]:
    """Directly integrate p(x) times the edge-CDF factors on the full 3D grid."""
    b0, b1, b2 = upper.Bx
    density_grid = np.einsum("abc,ia,jb,kc->ijk", dense, b0, b1, b2)
    x0, x1, x2 = upper.xgrids
    volume = upper.dx[0] * upper.dx[1] * upper.dx[2]
    out: dict[str, float] = {}
    for s_idx, t_idx in pair_indices:
        s = float(S_GRID[s_idx])
        t = float(S_GRID[t_idx])
        w0 = edge_cdf(PARAMS, s - x0)
        w1 = edge_cdf(PARAMS, s - x1) * edge_cdf(PARAMS, t - x1)
        w2 = edge_cdf(PARAMS, t - x2)
        value = np.einsum("ijk,i,j,k->", density_grid, w0, w1, w2) * volume
        out[f"{s_idx},{t_idx}"] = float(value)
    return out


def _monotonicity_violation(values: np.ndarray, axis: int) -> float:
    return float(max(0.0, -np.min(np.diff(values, axis=axis))))


def _run_at_q_grid(
    q_grid: int,
    bases,
    ttns: TTNSOpt,
    dense: np.ndarray,
    basis_integrals: np.ndarray,
    direct_points: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    start = perf_counter()
    upper = UpperModel(ttns, bases, PARENT, q_grid=q_grid)
    marginal = marginal_cdf(upper, PARENTS_Y1, S_GRID, PARAMS, n_d=1)
    pair = pair_cdf(upper, PARENTS_Y1, PARENTS_Y2, S_GRID, S_GRID, PARAMS, n_d=1)
    swapped = pair_cdf(upper, PARENTS_Y2, PARENTS_Y1, S_GRID, S_GRID, PARAMS, n_d=1)

    dense_marginal = np.clip(_dense_marginal(upper, dense, basis_integrals), 0.0, 1.0)
    dense_pair = np.clip(_dense_pair(upper, dense), 0.0, 1.0)

    selected = [(8, 8), (8, 15), (15, 8), (15, 15), (15, 22), (22, 15), (22, 22)]
    direct = _direct_grid_pair_points(upper, dense, selected) if direct_points else {}
    direct_errors = {
        key: abs(float(pair[tuple(int(v) for v in key.split(","))]) - value)
        for key, value in direct.items()
    }

    result = {
        "q_grid": q_grid,
        "runtime_seconds": perf_counter() - start,
        "normalization_abs_error": abs(
            float(np.einsum("abc,a,b,c->", dense, *basis_integrals)) - 1.0
        ),
        "ttns_vs_dense_marginal_max_abs": float(np.max(np.abs(marginal - dense_marginal))),
        "ttns_vs_dense_pair_max_abs": float(np.max(np.abs(pair - dense_pair))),
        "direct_grid_selected_points": direct,
        "direct_grid_selected_max_abs": float(max(direct_errors.values(), default=0.0)),
        "output_permutation_max_abs": float(np.max(np.abs(pair - swapped.T))),
        "marginal_consistency_max_abs": float(np.max(np.abs(pair[:, -1] - marginal))),
        "marginal_range_min": float(np.min(marginal)),
        "marginal_range_max": float(np.max(marginal)),
        "pair_range_min": float(np.min(pair)),
        "pair_range_max": float(np.max(pair)),
        "marginal_monotonicity_violation": _monotonicity_violation(marginal, axis=0),
        "pair_axis0_monotonicity_violation": _monotonicity_violation(pair, axis=0),
        "pair_axis1_monotonicity_violation": _monotonicity_violation(pair, axis=1),
    }
    return result, marginal, pair


def _validate_node_delay_convergence(
    bases,
    ttns: TTNSOpt,
) -> dict[str, Any]:
    upper = UpperModel(ttns, bases, PARENT, q_grid=321)
    params = DelayParams(
        edge_lo=0.0,
        edge_hi=0.3,
        node_lo=0.013,
        node_hi=0.287,
        kind="uniform",
    )
    reference_n_d = 2048
    reference_marginal = marginal_cdf(
        upper, PARENTS_Y1, S_GRID, params, n_d=reference_n_d
    )
    reference_pair = pair_cdf(
        upper, PARENTS_Y1, PARENTS_Y2, S_GRID, S_GRID, params, n_d=reference_n_d
    )
    runs = []
    for n_d in [2, 4, 8, 16, 32, 64]:
        marginal = marginal_cdf(upper, PARENTS_Y1, S_GRID, params, n_d=n_d)
        pair = pair_cdf(
            upper, PARENTS_Y1, PARENTS_Y2, S_GRID, S_GRID, params, n_d=n_d
        )
        runs.append(
            {
                "n_d": n_d,
                "marginal_vs_n_d2048_max_abs": float(
                    np.max(np.abs(marginal - reference_marginal))
                ),
                "pair_vs_n_d2048_max_abs": float(
                    np.max(np.abs(pair - reference_pair))
                ),
                "marginal_monotonicity_violation": _monotonicity_violation(
                    marginal, axis=0
                ),
                "pair_axis0_monotonicity_violation": _monotonicity_violation(
                    pair, axis=0
                ),
                "pair_axis1_monotonicity_violation": _monotonicity_violation(
                    pair, axis=1
                ),
            }
        )
    return {
        "configuration": {
            "q_grid": 321,
            "node_delay": {"kind": "uniform", "lo": 0.013, "hi": 0.287},
            "n_d_values": [item["n_d"] for item in runs],
            "reference_n_d": reference_n_d,
        },
        "runs": runs,
        "converged": runs[-1]["pair_vs_n_d2048_max_abs"]
        < runs[0]["pair_vs_n_d2048_max_abs"],
    }


def _validate_three_output_joint(
    bases,
    ttns: TTNSOpt,
    dense: np.ndarray,
) -> dict[str, Any]:
    block = BlockModel(
        local_vars=(0, 1, 2),
        global_vars=(0, 1, 2),
        parent=tuple(PARENT),
        ttns=ttns,
        bases=bases,
    )
    forest = UpperForest([block], q_grid=81)
    joint_grid = np.linspace(-1.0, 2.0, 9)
    parents_list = [[0, 1], [1, 2], [0, 2]]
    got = block_joint_cdf(forest, parents_list, joint_grid, PARAMS, n_d=1)

    upper = forest.blocks[0]["um"]
    m0 = upper.proj_single_multi(0, [joint_grid, joint_grid], PARAMS)
    m1 = upper.proj_single_multi(1, [joint_grid, joint_grid], PARAMS)
    m2 = upper.proj_single_multi(2, [joint_grid, joint_grid], PARAMS)
    expected = np.einsum("abc,ika,ijb,jkc->ijk", dense, m0, m1, m2)
    expected = np.clip(expected, 0.0, 1.0)

    return {
        "configuration": {
            "q_grid": 81,
            "joint_grid_size": len(joint_grid),
            "parents_list": parents_list,
            "shared_parent_structure": {
                "0": [0, 2],
                "1": [0, 1],
                "2": [1, 2],
            },
            "node_delay": {"kind": "uniform", "lo": 0.0, "hi": 0.0},
        },
        "ttns_vs_dense_joint_max_abs": float(np.max(np.abs(got - expected))),
        "range_min": float(np.min(got)),
        "range_max": float(np.max(got)),
        "axis0_monotonicity_violation": _monotonicity_violation(got, axis=0),
        "axis1_monotonicity_violation": _monotonicity_violation(got, axis=1),
        "axis2_monotonicity_violation": _monotonicity_violation(got, axis=2),
    }


def _validate_logskew_delay(
    bases,
    ttns: TTNSOpt,
) -> dict[str, Any]:
    """Deterministic q-grid and node-quadrature convergence for log-skew delays."""
    params = DelayParams(
        kind="logskewnorm",
        e_xi=-2.12,
        e_omega=0.45,
        e_alpha=4.0,
        d_xi=-2.12,
        d_omega=0.45,
        d_alpha=4.0,
    )
    grid = np.linspace(-1.0, 2.5, 36)

    def evaluate(q_grid: int, n_d: int) -> tuple[np.ndarray, np.ndarray]:
        upper = UpperModel(ttns, bases, PARENT, q_grid=q_grid)
        marginal = marginal_cdf(upper, PARENTS_Y1, grid, params, n_d=n_d)
        pair = pair_cdf(
            upper, PARENTS_Y1, PARENTS_Y2, grid, grid, params, n_d=n_d
        )
        return marginal, pair

    q_values = [81, 161, 321, 641]
    q_reference = 2001
    q_fixed_n_d = 512
    q_ref_marginal, q_ref_pair = evaluate(q_reference, q_fixed_n_d)
    q_runs = []
    for q_grid in q_values:
        marginal, pair = evaluate(q_grid, q_fixed_n_d)
        q_runs.append(
            {
                "q_grid": q_grid,
                "marginal_max_abs": float(
                    np.max(np.abs(marginal - q_ref_marginal))
                ),
                "pair_max_abs": float(np.max(np.abs(pair - q_ref_pair))),
            }
        )

    n_values = [8, 16, 32, 64, 128]
    n_reference = 2048
    n_fixed_q_grid = 641
    n_ref_marginal, n_ref_pair = evaluate(n_fixed_q_grid, n_reference)
    n_runs = []
    for n_d in n_values:
        marginal, pair = evaluate(n_fixed_q_grid, n_d)
        n_runs.append(
            {
                "n_d": n_d,
                "marginal_max_abs": float(
                    np.max(np.abs(marginal - n_ref_marginal))
                ),
                "pair_max_abs": float(np.max(np.abs(pair - n_ref_pair))),
                "marginal_monotonicity_violation": _monotonicity_violation(
                    marginal, axis=0
                ),
                "pair_axis0_monotonicity_violation": _monotonicity_violation(
                    pair, axis=0
                ),
                "pair_axis1_monotonicity_violation": _monotonicity_violation(
                    pair, axis=1
                ),
            }
        )

    return {
        "configuration": {
            "edge_delay": {
                "kind": "logskewnorm",
                "xi": params.e_xi,
                "omega": params.e_omega,
                "alpha": params.e_alpha,
            },
            "node_delay": {
                "kind": "logskewnorm",
                "xi": params.d_xi,
                "omega": params.d_omega,
                "alpha": params.d_alpha,
            },
            "output_grid": {
                "min": float(grid[0]),
                "max": float(grid[-1]),
                "size": len(grid),
            },
            "q_grid_values": q_values,
            "q_reference": q_reference,
            "q_fixed_n_d": q_fixed_n_d,
            "n_d_values": n_values,
            "n_reference": n_reference,
            "n_fixed_q_grid": n_fixed_q_grid,
        },
        "q_grid_runs": q_runs,
        "n_d_runs": n_runs,
        "q_grid_converged": q_runs[-1]["pair_max_abs"] < q_runs[0]["pair_max_abs"],
        "n_d_converged": n_runs[-1]["pair_max_abs"] < n_runs[0]["pair_max_abs"],
    }


def _weighted_gram_single(
    upper: UpperModel,
    u: int,
    grid: np.ndarray,
    params: DelayParams,
) -> np.ndarray:
    x = upper.xgrids[u]
    basis = upper.Bx[u]
    weights = edge_cdf(params, grid[:, None] - x[None, :])
    return np.einsum("sq,qi,qj->sij", weights, basis, basis) * upper.dx[u]


def _weighted_gram_pair(
    upper: UpperModel,
    u: int,
    s_grid: np.ndarray,
    t_grid: np.ndarray,
    params: DelayParams,
) -> np.ndarray:
    x = upper.xgrids[u]
    basis = upper.Bx[u]
    fs = edge_cdf(params, s_grid[:, None] - x[None, :])
    ft = edge_cdf(params, t_grid[:, None] - x[None, :])
    return (
        np.einsum("sq,tq,qi,qj->stij", fs, ft, basis, basis) * upper.dx[u]
    )


def _quadratic_batch(
    ttns: TTNSOpt,
    matrices: np.ndarray,
) -> np.ndarray:
    leading_shape = matrices.shape[:-3]
    flat = matrices.reshape((-1,) + matrices.shape[-3:])
    values = jax.vmap(
        lambda local: quadratic_form_ttns(ttns, local, PARENT)
    )(jnp.asarray(flat))
    return np.asarray(values).reshape(leading_shape)


def _validate_squared_ttns(
    bases,
    amplitude: TTNSOpt,
    dense_amplitude: np.ndarray,
) -> dict[str, Any]:
    """Validate the doubled contraction for p=psi^2/Z without fitting."""
    upper = UpperModel(amplitude, bases, PARENT, q_grid=81)
    gram = np.asarray(jax.vmap(type(bases).l2_integral)(bases), dtype=float)
    z_ttns = float(quadratic_form_ttns(amplitude, jnp.asarray(gram), PARENT))
    z_dense = float(
        np.einsum(
            "abc,def,ad,be,cf->",
            dense_amplitude,
            dense_amplitude,
            gram[0],
            gram[1],
            gram[2],
        )
    )

    w0 = _weighted_gram_single(upper, 0, S_GRID, PARAMS)
    w1 = _weighted_gram_single(upper, 1, S_GRID, PARAMS)
    w2 = _weighted_gram_single(upper, 2, S_GRID, PARAMS)
    w1_pair = _weighted_gram_pair(upper, 1, S_GRID, S_GRID, PARAMS)

    marginal_matrices = np.broadcast_to(
        gram[None, :, :, :], (len(S_GRID),) + gram.shape
    ).copy()
    marginal_matrices[:, 0] = w0
    marginal_matrices[:, 1] = w1
    marginal = np.clip(_quadratic_batch(amplitude, marginal_matrices) / z_ttns, 0.0, 1.0)
    dense_marginal = np.clip(
        np.einsum(
            "abc,def,sad,sbe,cf->s",
            dense_amplitude,
            dense_amplitude,
            w0,
            w1,
            gram[2],
        )
        / z_dense,
        0.0,
        1.0,
    )

    pair_matrices = np.broadcast_to(
        gram[None, None, :, :, :],
        (len(S_GRID), len(S_GRID)) + gram.shape,
    ).copy()
    pair_matrices[:, :, 0] = w0[:, None, :, :]
    pair_matrices[:, :, 1] = w1_pair
    pair_matrices[:, :, 2] = w2[None, :, :, :]
    pair = np.clip(_quadratic_batch(amplitude, pair_matrices) / z_ttns, 0.0, 1.0)
    dense_pair = np.clip(
        np.einsum(
            "abc,def,sad,stbe,tcf->st",
            dense_amplitude,
            dense_amplitude,
            w0,
            w1_pair,
            w2,
        )
        / z_dense,
        0.0,
        1.0,
    )

    swapped_matrices = np.broadcast_to(
        gram[None, None, :, :, :],
        (len(S_GRID), len(S_GRID)) + gram.shape,
    ).copy()
    swapped_matrices[:, :, 0] = w0[None, :, :, :]
    swapped_matrices[:, :, 1] = w1_pair
    swapped_matrices[:, :, 2] = w2[:, None, :, :]
    swapped = np.clip(
        _quadratic_batch(amplitude, swapped_matrices) / z_ttns,
        0.0,
        1.0,
    )

    basis0, basis1, basis2 = upper.Bx
    psi_grid = np.einsum(
        "abc,ia,jb,kc->ijk",
        dense_amplitude,
        basis0,
        basis1,
        basis2,
    )
    density_grid = psi_grid**2 / z_dense
    x0, x1, x2 = upper.xgrids
    volume = upper.dx[0] * upper.dx[1] * upper.dx[2]
    selected = [(8, 8), (15, 15), (15, 22), (22, 15), (22, 22)]
    direct_errors = []
    for s_idx, t_idx in selected:
        s = float(S_GRID[s_idx])
        t = float(S_GRID[t_idx])
        factor0 = edge_cdf(PARAMS, s - x0)
        factor1 = edge_cdf(PARAMS, s - x1) * edge_cdf(PARAMS, t - x1)
        factor2 = edge_cdf(PARAMS, t - x2)
        direct = (
            np.einsum(
                "ijk,i,j,k->",
                density_grid,
                factor0,
                factor1,
                factor2,
            )
            * volume
        )
        direct_errors.append(abs(float(pair[s_idx, t_idx]) - float(direct)))

    return {
        "configuration": {
            "n_dims": N_DIMS,
            "basis_dim": BASIS_DIM,
            "basis_degree": BASIS_DEGREE,
            "amplitude_ttns_rank": 2,
            "q_grid": 81,
            "parents_y1": PARENTS_Y1,
            "parents_y2": PARENTS_Y2,
            "node_delay": {"kind": "uniform", "lo": 0.0, "hi": 0.0},
            "training_steps": 0,
            "monte_carlo_samples": 0,
        },
        "normalizer_ttns": z_ttns,
        "normalizer_dense": z_dense,
        "normalizer_abs_error": abs(z_ttns - z_dense),
        "ttns_vs_dense_marginal_max_abs": float(
            np.max(np.abs(marginal - dense_marginal))
        ),
        "ttns_vs_dense_pair_max_abs": float(np.max(np.abs(pair - dense_pair))),
        "direct_grid_selected_max_abs": float(max(direct_errors)),
        "output_permutation_max_abs": float(np.max(np.abs(pair - swapped.T))),
        "marginal_consistency_max_abs": float(
            np.max(np.abs(pair[:, -1] - marginal))
        ),
        "marginal_range_min": float(np.min(marginal)),
        "marginal_range_max": float(np.max(marginal)),
        "pair_range_min": float(np.min(pair)),
        "pair_range_max": float(np.max(pair)),
        "marginal_monotonicity_violation": _monotonicity_violation(
            marginal, axis=0
        ),
        "pair_axis0_monotonicity_violation": _monotonicity_violation(pair, axis=0),
        "pair_axis1_monotonicity_violation": _monotonicity_violation(pair, axis=1),
    }


def run_validation() -> dict[str, Any]:
    bases, ttns, dense, basis_integrals = _construct_density()
    q_values = [41, 81, 161, 321]
    reference_q = 2001

    reference, reference_marginal, reference_pair = _run_at_q_grid(
        reference_q,
        bases,
        ttns,
        dense,
        basis_integrals,
        direct_points=False,
    )

    runs = []
    audit_pair = None
    for q_grid in q_values:
        result, marginal, pair = _run_at_q_grid(
            q_grid,
            bases,
            ttns,
            dense,
            basis_integrals,
            direct_points=(q_grid == 81),
        )
        result["marginal_vs_q2001_max_abs"] = float(
            np.max(np.abs(marginal - reference_marginal))
        )
        result["pair_vs_q2001_max_abs"] = float(np.max(np.abs(pair - reference_pair)))
        runs.append(result)
        if q_grid == 81:
            audit_pair = pair.copy()

    tolerances = {
        "normalization_abs_error": 1e-12,
        "ttns_vs_dense_max_abs": 2e-12,
        "direct_grid_selected_max_abs": 2e-12,
        "output_permutation_max_abs": 2e-12,
        "marginal_consistency_max_abs": 2e-3,
        "range_tolerance": 1e-12,
        "monotonicity_violation": 2e-12,
    }
    audit_run = next(item for item in runs if item["q_grid"] == 81)
    node_delay = _validate_node_delay_convergence(bases, ttns)
    joint3 = _validate_three_output_joint(bases, ttns, dense)
    logskew = _validate_logskew_delay(bases, ttns)
    squared = _validate_squared_ttns(bases, ttns, dense)
    checks = {
        "normalization": audit_run["normalization_abs_error"]
        <= tolerances["normalization_abs_error"],
        "ttns_vs_dense_marginal": audit_run["ttns_vs_dense_marginal_max_abs"]
        <= tolerances["ttns_vs_dense_max_abs"],
        "ttns_vs_dense_pair": audit_run["ttns_vs_dense_pair_max_abs"]
        <= tolerances["ttns_vs_dense_max_abs"],
        "direct_grid_selected": audit_run["direct_grid_selected_max_abs"]
        <= tolerances["direct_grid_selected_max_abs"],
        "output_permutation": audit_run["output_permutation_max_abs"]
        <= tolerances["output_permutation_max_abs"],
        "marginal_consistency": audit_run["marginal_consistency_max_abs"]
        <= tolerances["marginal_consistency_max_abs"],
        "marginal_range": audit_run["marginal_range_min"] >= -tolerances["range_tolerance"]
        and audit_run["marginal_range_max"] <= 1.0 + tolerances["range_tolerance"],
        "pair_range": audit_run["pair_range_min"] >= -tolerances["range_tolerance"]
        and audit_run["pair_range_max"] <= 1.0 + tolerances["range_tolerance"],
        "monotonicity": max(
            audit_run["marginal_monotonicity_violation"],
            audit_run["pair_axis0_monotonicity_violation"],
            audit_run["pair_axis1_monotonicity_violation"],
        )
        <= tolerances["monotonicity_violation"],
        "quadrature_convergence": runs[-1]["pair_vs_q2001_max_abs"]
        < runs[0]["pair_vs_q2001_max_abs"],
        "node_delay_quadrature_convergence": node_delay["converged"],
        "three_output_joint": joint3["ttns_vs_dense_joint_max_abs"]
        <= tolerances["ttns_vs_dense_max_abs"],
        "three_output_joint_monotonicity": max(
            joint3["axis0_monotonicity_violation"],
            joint3["axis1_monotonicity_violation"],
            joint3["axis2_monotonicity_violation"],
        )
        <= tolerances["monotonicity_violation"],
        "logskew_q_grid_convergence": logskew["q_grid_converged"],
        "logskew_node_delay_convergence": logskew["n_d_converged"],
        "squared_normalizer": squared["normalizer_abs_error"]
        <= tolerances["ttns_vs_dense_max_abs"],
        "squared_ttns_vs_dense_marginal": squared[
            "ttns_vs_dense_marginal_max_abs"
        ]
        <= tolerances["ttns_vs_dense_max_abs"],
        "squared_ttns_vs_dense_pair": squared["ttns_vs_dense_pair_max_abs"]
        <= tolerances["ttns_vs_dense_max_abs"],
        "squared_direct_grid_selected": squared["direct_grid_selected_max_abs"]
        <= tolerances["direct_grid_selected_max_abs"],
        "squared_output_permutation": squared["output_permutation_max_abs"]
        <= tolerances["output_permutation_max_abs"],
        "squared_marginal_consistency": squared["marginal_consistency_max_abs"]
        <= tolerances["marginal_consistency_max_abs"],
        "squared_monotonicity": max(
            squared["marginal_monotonicity_violation"],
            squared["pair_axis0_monotonicity_violation"],
            squared["pair_axis1_monotonicity_violation"],
        )
        <= tolerances["monotonicity_violation"],
    }

    return {
        "experiment": "maxplus_cdf_theorem_validation",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration": {
            "n_dims": N_DIMS,
            "basis_dim": BASIS_DIM,
            "basis_degree": BASIS_DEGREE,
            "ttns_rank": 2,
            "parent": PARENT,
            "parents_y1": PARENTS_Y1,
            "parents_y2": PARENTS_Y2,
            "shared_parents": sorted(set(PARENTS_Y1) & set(PARENTS_Y2)),
            "edge_delay": {
                "kind": PARAMS.kind,
                "lo": PARAMS.edge_lo,
                "hi": PARAMS.edge_hi,
            },
            "node_delay": {
                "kind": PARAMS.kind,
                "lo": PARAMS.node_lo,
                "hi": PARAMS.node_hi,
            },
            "s_grid": {
                "min": float(S_GRID[0]),
                "max": float(S_GRID[-1]),
                "size": len(S_GRID),
            },
            "q_grid_values": q_values,
            "reference_q_grid": reference_q,
            "training_steps": 0,
            "monte_carlo_samples": 0,
        },
        "tolerances": tolerances,
        "checks": checks,
        "runs": runs,
        "reference": reference,
        "node_delay_convergence": node_delay,
        "three_output_joint": joint3,
        "logskew_delay": logskew,
        "squared_ttns": squared,
        "plot_data": {
            "s_grid": S_GRID.tolist(),
            "pair_reference_q2001": reference_pair.tolist(),
            "pair_q81": audit_pair.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "simple_ttns_l2"
        / "reports"
        / "maxplus_cdf_theorem_validation_metrics.json",
    )
    args = parser.parse_args()

    metrics = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if metrics["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
