"""Deterministic single-layer and multilayer max-plus error audit.

This script is deliberately small.  It runs no Monte Carlo simulation and no
large model training.  It has two complementary parts:

1. A continuous K=3, rank-2 mixture-of-products density is propagated through
   a shared-parent max-plus layer.  High- and default-resolution contractions,
   a Chow--Liu projection, and a nonnegative rank-limited tree materialization
   provide numerical, structural, and fitting error terms on one common grid.
2. A finite-state K=3, L=4 max-plus chain is propagated exactly.  At every
   layer the exact propagated approximation is projected to a Chow--Liu tree
   and rank-limited through nonnegative matrix factorizations.  Because every
   object is a normalized probability mass table, total variation distances
   and the multilayer recurrence are exact up to floating-point roundoff.

The finite-grid quantities in part 1 are grid-estimated/discretized errors;
they are not presented as rigorous continuous-distribution TV bounds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


K = 3
L = 4
OUTPUT_PARENTS = ((0, 1), (1, 2), (2, 0))
CONTINUOUS_OUTPUT_GRID = np.linspace(-0.10, 1.40, 31)
CONTINUOUS_Q_DEFAULT = 401
CONTINUOUS_Q_REFERENCE = 4001
FINITE_GRID_SIZE = 17
FINITE_EDGE_PMF = np.asarray([0.65, 0.35], dtype=float)
NMF_RANK = 3
NMF_STEPS = 600
NMF_SEED = 20260727
CONDITIONAL_PROBABILITY_FLOOR = 1e-12


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def _l2(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.sum((p - q) ** 2)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0.0
    if np.any(q[mask] <= 0.0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _entropy(pmf: np.ndarray) -> float:
    pmf = np.asarray(pmf, dtype=float)
    mask = pmf > 0.0
    return float(-np.sum(pmf[mask] * np.log(pmf[mask])))


def _normalize_pmf(raw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(raw, dtype=float)
    negative_mass = float(-np.sum(raw[raw < 0.0]))
    clipped = np.clip(raw, 0.0, None)
    total = float(np.sum(clipped))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"invalid probability mass total: {total}")
    return clipped / total, {
        "raw_sum": float(np.sum(raw)),
        "raw_min": float(np.min(raw)),
        "negative_mass_before_clipping": negative_mass,
        "normalization_abs_error_after": abs(float(np.sum(clipped / total)) - 1.0),
    }


def _cdf_to_pmf(cdf: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    pmf = np.asarray(cdf, dtype=float)
    for axis in range(pmf.ndim):
        pad = [(0, 0)] * pmf.ndim
        pad[axis] = (1, 0)
        pmf = np.diff(np.pad(pmf, pad, mode="constant"), axis=axis)
    return _normalize_pmf(pmf)


def _pmf_to_cdf(pmf: np.ndarray) -> np.ndarray:
    out = np.asarray(pmf, dtype=float)
    for axis in range(out.ndim):
        out = np.cumsum(out, axis=axis)
    return out


def _monotonicity_violation(cdf: np.ndarray) -> float:
    worst = 0.0
    for axis in range(cdf.ndim):
        worst = max(worst, float(max(0.0, -np.min(np.diff(cdf, axis=axis)))))
    return worst


def _marginal(pmf: np.ndarray, axis: int) -> np.ndarray:
    return np.sum(pmf, axis=tuple(a for a in range(pmf.ndim) if a != axis))


def _joint_marginal(pmf: np.ndarray, axes: Sequence[int]) -> np.ndarray:
    axes = tuple(int(a) for a in axes)
    kept_sorted = tuple(sorted(axes))
    summed = np.sum(pmf, axis=tuple(a for a in range(pmf.ndim) if a not in axes))
    if axes != kept_sorted:
        position = {axis: pos for pos, axis in enumerate(kept_sorted)}
        summed = np.transpose(summed, tuple(position[axis] for axis in axes))
    return summed


def _marginal_entropy(pmf: np.ndarray, axes: Sequence[int]) -> float:
    axes = tuple(sorted(set(int(a) for a in axes)))
    if not axes:
        return 0.0
    return _entropy(_joint_marginal(pmf, axes))


def _mutual_information(pair: np.ndarray) -> float:
    pair, _ = _normalize_pmf(pair)
    pa = np.sum(pair, axis=1, keepdims=True)
    pb = np.sum(pair, axis=0, keepdims=True)
    denom = pa * pb
    mask = (pair > 0.0) & (denom > 0.0)
    return float(np.sum(pair[mask] * np.log(pair[mask] / denom[mask])))


def _tree_order(parent: Sequence[int]) -> list[int]:
    root = list(parent).index(-1)
    children = [[] for _ in parent]
    for child, par in enumerate(parent):
        if par != -1:
            children[par].append(child)
    order = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        order.append(node)
        queue.extend(sorted(children[node]))
    return order


def _structural_information(
    pmf: np.ndarray,
    tree: np.ndarray,
    parent: Sequence[int],
    weights: np.ndarray,
) -> dict[str, Any]:
    total_correlation = sum(_marginal_entropy(pmf, (a,)) for a in range(pmf.ndim)) - _entropy(pmf)
    edge_mutual_information = sum(
        float(weights[child, par]) for child, par in enumerate(parent) if par != -1
    )
    kl_to_tree = _kl(pmf, tree)
    identity_value = total_correlation - edge_mutual_information

    order = _tree_order(parent)
    cmi_terms: list[dict[str, Any]] = []
    for position, node in enumerate(order):
        par = int(parent[node])
        if par == -1:
            continue
        previous = order[:position]
        omitted = [axis for axis in previous if axis != par]
        if omitted:
            cmi = (
                _marginal_entropy(pmf, (node, par))
                + _marginal_entropy(pmf, tuple(omitted + [par]))
                - _marginal_entropy(pmf, (par,))
                - _marginal_entropy(pmf, tuple([node] + omitted + [par]))
            )
        else:
            cmi = 0.0
        cmi_terms.append(
            {
                "node": int(node),
                "tree_parent": par,
                "omitted_previous_nodes": [int(v) for v in omitted],
                "conditional_mutual_information": float(max(cmi, 0.0)),
            }
        )
    cmi_sum = sum(term["conditional_mutual_information"] for term in cmi_terms)
    return {
        "kl_to_chow_liu_tree": kl_to_tree,
        "total_correlation": total_correlation,
        "selected_edge_mutual_information_sum": edge_mutual_information,
        "tc_minus_edge_mi": identity_value,
        "conditional_mutual_information_terms": cmi_terms,
        "conditional_mutual_information_sum": cmi_sum,
        "kl_identity_abs_error": abs(kl_to_tree - identity_value),
        "cmi_identity_abs_error": abs(kl_to_tree - cmi_sum),
        "pinsker_tv_upper_bound": math.sqrt(max(0.0, 0.5 * kl_to_tree)),
        "actual_structural_tv": _tv(pmf, tree),
    }


def _subtree_nodes(parent: Sequence[int], root: int) -> list[int]:
    children = [[] for _ in parent]
    for child, par in enumerate(parent):
        if par != -1:
            children[par].append(child)
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(children[node])
    return sorted(out)


def _tt_svd_three_node(target: np.ndarray, parent: Sequence[int], rank: int) -> np.ndarray:
    if target.ndim != 3:
        raise ValueError("the deterministic audit implements TT-SVD only for K=3")
    degree = [0] * 3
    for child, par in enumerate(parent):
        if par != -1:
            degree[child] += 1
            degree[par] += 1
    center = degree.index(2)
    endpoints = sorted(a for a in range(3) if a != center)
    order = [endpoints[0], center, endpoints[1]]
    tensor = np.transpose(target, order)
    n0, n1, n2 = tensor.shape
    u0, s0, vh0 = np.linalg.svd(tensor.reshape(n0, n1 * n2), full_matrices=False)
    r0 = min(rank, len(s0))
    core0 = u0[:, :r0]
    remainder = s0[:r0, None] * vh0[:r0, :]
    u1, s1, vh1 = np.linalg.svd(remainder.reshape(r0 * n1, n2), full_matrices=False)
    r1 = min(rank, len(s1))
    core1 = u1[:, :r1].reshape(r0, n1, r1)
    core2 = s1[:r1, None] * vh1[:r1, :]
    reconstructed = np.einsum("ia,ajb,bk->ijk", core0, core1, core2)
    inverse = np.argsort(order)
    return np.transpose(reconstructed, inverse)


def _rank_spectral_diagnostics(
    target: np.ndarray,
    fitted: np.ndarray,
    parent: Sequence[int],
    rank: int,
) -> dict[str, Any]:
    edge_rows = []
    tail_squares = []
    for child, par in enumerate(parent):
        if par == -1:
            continue
        left = _subtree_nodes(parent, child)
        right = [a for a in range(target.ndim) if a not in left]
        matrix = np.transpose(target, left + right).reshape(
            int(np.prod([target.shape[a] for a in left])),
            int(np.prod([target.shape[a] for a in right])),
        )
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        tail_square = float(np.sum(singular_values[rank:] ** 2))
        tail_squares.append(tail_square)
        edge_rows.append(
            {
                "edge": [int(par), int(child)],
                "cut_left": left,
                "cut_right": right,
                "singular_values": singular_values.tolist(),
                "rank_tail_frobenius": math.sqrt(tail_square),
                "rank_tail_energy": tail_square,
            }
        )
    lower = math.sqrt(max(tail_squares, default=0.0))
    quasi_upper = math.sqrt(sum(tail_squares))
    tt_svd = _tt_svd_three_node(target, parent, rank)
    tt_svd_error = _l2(target, tt_svd)
    fitted_error = _l2(target, fitted)
    return {
        "rank_limit": rank,
        "edge_cut_spectra": edge_rows,
        "best_possible_frobenius_lower_bound": lower,
        "tree_svd_tail_upper_bound": quasi_upper,
        "constructed_unconstrained_tt_svd_frobenius": tt_svd_error,
        "actual_nonnegative_materialization_frobenius": fitted_error,
        "nonnegative_materialization_minus_lower_bound": fitted_error - lower,
        "tt_svd_bound_violation": max(0.0, tt_svd_error - quasi_upper),
        "rank_lower_bound_violation": max(0.0, lower - fitted_error),
    }


def _maximum_spanning_tree(weights: np.ndarray, root: int = 0) -> list[int]:
    n = int(weights.shape[0])
    selected = {root}
    undirected: list[tuple[int, int]] = []
    while len(selected) < n:
        candidates = [
            (float(weights[u, v]), u, v)
            for u in selected
            for v in range(n)
            if v not in selected
        ]
        if not candidates:
            raise ValueError("failed to construct spanning tree")
        _, u_best, v_best = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        undirected.append((u_best, v_best))
        selected.add(v_best)

    adjacency: list[list[int]] = [[] for _ in range(n)]
    for u, v in undirected:
        adjacency[u].append(v)
        adjacency[v].append(u)
    parent = [-2] * n
    parent[root] = -1
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in sorted(adjacency[u]):
            if parent[v] == -2:
                parent[v] = u
                queue.append(v)
    return parent


def _tree_projection(pmf: np.ndarray) -> tuple[np.ndarray, list[int], dict[int, np.ndarray], np.ndarray]:
    n = pmf.ndim
    weights = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            weights[a, b] = weights[b, a] = _mutual_information(_joint_marginal(pmf, (a, b)))
    parent = _maximum_spanning_tree(weights)
    root = parent.index(-1)
    root_marginal = _marginal(pmf, root)
    conditionals: dict[int, np.ndarray] = {}
    for child, par in enumerate(parent):
        if par == -1:
            continue
        pair = _joint_marginal(pmf, (child, par))
        par_marginal = np.sum(pair, axis=0)
        conditional = np.divide(
            pair,
            par_marginal[None, :],
            out=np.full_like(pair, 1.0 / pair.shape[0]),
            where=par_marginal[None, :] > 1e-15,
        )
        conditional /= np.clip(np.sum(conditional, axis=0, keepdims=True), 1e-15, None)
        conditionals[child] = conditional
    tree = _assemble_tree_distribution(root_marginal, parent, conditionals)
    return tree, parent, conditionals, weights


def _assemble_tree_distribution(
    root_marginal: np.ndarray,
    parent: Sequence[int],
    conditionals: dict[int, np.ndarray],
) -> np.ndarray:
    n = len(parent)
    grid_size = len(root_marginal)
    coordinates = np.indices((grid_size,) * n)
    root = list(parent).index(-1)
    joint = root_marginal[coordinates[root]]
    for child, par in enumerate(parent):
        if par != -1:
            joint = joint * conditionals[child][coordinates[child], coordinates[par]]
    joint, _ = _normalize_pmf(joint)
    return joint


def _nmf_conditional(
    matrix: np.ndarray,
    parent_weights: np.ndarray,
    rank: int,
    rng: np.random.Generator,
    steps: int,
) -> tuple[np.ndarray, float]:
    rows, cols = matrix.shape
    sqrt_weights = np.sqrt(np.clip(parent_weights, 0.0, None))
    weighted_matrix = matrix * sqrt_weights[None, :]
    u = rng.uniform(0.2, 1.0, size=(rows, rank))
    v = rng.uniform(0.2, 1.0, size=(rank, cols))
    eps = 1e-15
    for _ in range(steps):
        v *= (u.T @ weighted_matrix) / np.clip((u.T @ u) @ v, eps, None)
        u *= (weighted_matrix @ v.T) / np.clip(u @ (v @ v.T), eps, None)
    weighted_approximation = np.clip(u @ v, 0.0, None)
    approximation = np.divide(
        weighted_approximation,
        sqrt_weights[None, :],
        out=np.full_like(weighted_approximation, 1.0 / rows),
        where=sqrt_weights[None, :] > 1e-12,
    )
    # A declared positive floor prevents floating-point underflow from creating
    # support mismatch and an infinite forward conditional KL.
    approximation = np.maximum(approximation, CONDITIONAL_PROBABILITY_FLOOR)
    approximation /= np.sum(approximation, axis=0, keepdims=True)
    residual = float(np.linalg.norm((matrix - approximation) * sqrt_weights[None, :]))
    return approximation, residual


def _rank_limited_tree(
    root_marginal: np.ndarray,
    parent: Sequence[int],
    conditionals: dict[int, np.ndarray],
    rank: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float], dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    approximations: dict[int, np.ndarray] = {}
    residuals: dict[str, float] = {}
    exact_tree = _assemble_tree_distribution(root_marginal, parent, conditionals)
    for child in sorted(conditionals):
        par = int(parent[child])
        approximations[child], residuals[str(child)] = _nmf_conditional(
            conditionals[child], _marginal(exact_tree, par), rank, rng, NMF_STEPS
        )
    return (
        _assemble_tree_distribution(root_marginal, parent, approximations),
        residuals,
        approximations,
    )


def _conditional_kernel_tv_bounds(
    exact_tree: np.ndarray,
    parent: Sequence[int],
    conditionals: dict[int, np.ndarray],
    approximations: dict[int, np.ndarray],
) -> dict[str, Any]:
    weighted_sum = 0.0
    supremum_sum = 0.0
    conditional_kl_sum = 0.0
    rows = []
    for child in sorted(conditionals):
        par = int(parent[child])
        column_tv = 0.5 * np.sum(
            np.abs(conditionals[child] - approximations[child]), axis=0
        )
        parent_marginal = _marginal(exact_tree, par)
        weighted = float(parent_marginal @ column_tv)
        supremum = float(np.max(column_tv))
        column_kl = np.asarray(
            [
                _kl(conditionals[child][:, x], approximations[child][:, x])
                for x in range(conditionals[child].shape[1])
            ]
        )
        expected_conditional_kl = float(parent_marginal @ column_kl)
        conditional_kl_sum += expected_conditional_kl
        weighted_sum += weighted
        supremum_sum += supremum
        rows.append(
            {
                "edge": [par, int(child)],
                "parent_weighted_conditional_tv": weighted,
                "supremum_conditional_tv": supremum,
                "parent_weighted_conditional_kl": expected_conditional_kl,
                "conditional_pinsker_tv_upper_bound": math.sqrt(
                    max(0.0, 0.5 * expected_conditional_kl)
                ),
            }
        )
    fitted = _assemble_tree_distribution(
        _marginal(exact_tree, list(parent).index(-1)),
        parent,
        approximations,
    )
    joint_fit_kl = _kl(exact_tree, fitted)
    joint_pinsker = math.sqrt(max(0.0, 0.5 * joint_fit_kl))
    actual = _tv(
        exact_tree,
        fitted,
    )
    joint_frobenius = _l2(exact_tree, fitted)
    cell_count = int(exact_tree.size)
    frobenius_to_tv = 0.5 * math.sqrt(cell_count) * joint_frobenius
    return {
        "per_edge": rows,
        "parent_weighted_sum_bound": weighted_sum,
        "supremum_sum_bound": supremum_sum,
        "conditional_kl_sum": conditional_kl_sum,
        "joint_fit_kl": joint_fit_kl,
        "factorized_kl_identity_abs_error": abs(joint_fit_kl - conditional_kl_sum),
        "joint_pinsker_tv_upper_bound": joint_pinsker,
        "cross_entropy_excess_nats": conditional_kl_sum,
        "joint_cell_count": cell_count,
        "joint_frobenius_error": joint_frobenius,
        "frobenius_to_tv_upper_bound": frobenius_to_tv,
        "actual_fit_tv": actual,
        "weighted_bound_violation": max(0.0, actual - weighted_sum),
        "supremum_bound_violation": max(0.0, actual - supremum_sum),
        "joint_pinsker_bound_violation": max(0.0, actual - joint_pinsker),
        "frobenius_to_tv_bound_violation": max(0.0, actual - frobenius_to_tv),
    }


def _beta_pdf(x: np.ndarray, alpha: int, beta: int) -> np.ndarray:
    coefficient = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
    return coefficient * np.power(x, alpha - 1) * np.power(1.0 - x, beta - 1)


def _uniform_cdf(values: np.ndarray, lo: float = 0.0, hi: float = 0.3) -> np.ndarray:
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _continuous_joint_cdf(q_grid: int, output_parents: Sequence[Sequence[int]]) -> np.ndarray:
    x = np.linspace(0.0, 1.0, q_grid)
    mixture_weights = np.asarray([0.45, 0.55])
    beta_parameters = (
        ((2, 5), (3, 3), (5, 2)),
        ((5, 2), (2, 4), (3, 2)),
    )
    local_factors: list[np.ndarray] = []
    for input_axis in range(K):
        children = [a for a, parents in enumerate(output_parents) if input_axis in parents]
        shape = (len(mixture_weights),) + (len(CONTINUOUS_OUTPUT_GRID),) * len(children)
        local = np.empty(shape, dtype=float)
        for component in range(len(mixture_weights)):
            pdf = _beta_pdf(x, *beta_parameters[component][input_axis])
            pdf /= np.trapz(pdf, x)
            product = np.ones((len(CONTINUOUS_OUTPUT_GRID),) * len(children) + (q_grid,))
            for position, output_axis in enumerate(children):
                factor = _uniform_cdf(CONTINUOUS_OUTPUT_GRID[:, None] - x[None, :])
                reshape = [1] * len(children) + [q_grid]
                reshape[position] = len(CONTINUOUS_OUTPUT_GRID)
                product *= factor.reshape(reshape)
            local[component] = np.trapz(product * pdf, x, axis=-1)
        local_factors.append(local)

    arguments: list[Any] = [mixture_weights, [0]]
    for input_axis, local in enumerate(local_factors):
        children = [a for a, parents in enumerate(output_parents) if input_axis in parents]
        arguments.extend([local, [0] + [1 + a for a in children]])
    arguments.append([1, 2, 3])
    return np.einsum(*arguments, optimize="greedy")


def _continuous_subset_cdf(q_grid: int, selected_outputs: Sequence[int]) -> np.ndarray:
    parents = [OUTPUT_PARENTS[a] for a in selected_outputs]
    x = np.linspace(0.0, 1.0, q_grid)
    mixture_weights = np.asarray([0.45, 0.55])
    beta_parameters = (
        ((2, 5), (3, 3), (5, 2)),
        ((5, 2), (2, 4), (3, 2)),
    )
    factors = []
    for input_axis in range(K):
        children = [a for a, ps in enumerate(parents) if input_axis in ps]
        shape = (2,) + (len(CONTINUOUS_OUTPUT_GRID),) * len(children)
        local = np.ones(shape)
        if children:
            for component in range(2):
                pdf = _beta_pdf(x, *beta_parameters[component][input_axis])
                pdf /= np.trapz(pdf, x)
                product = np.ones((len(CONTINUOUS_OUTPUT_GRID),) * len(children) + (q_grid,))
                for position in range(len(children)):
                    factor = _uniform_cdf(CONTINUOUS_OUTPUT_GRID[:, None] - x[None, :])
                    reshape = [1] * len(children) + [q_grid]
                    reshape[position] = len(CONTINUOUS_OUTPUT_GRID)
                    product *= factor.reshape(reshape)
                local[component] = np.trapz(product * pdf, x, axis=-1)
        factors.append((children, local))

    arguments: list[Any] = [mixture_weights, [0]]
    for children, local in factors:
        arguments.extend([local, [0] + [1 + a for a in children]])
    arguments.append([1 + a for a in range(len(selected_outputs))])
    return np.einsum(*arguments, optimize="greedy")


def _continuous_single_layer_audit() -> dict[str, Any]:
    start = perf_counter()
    cdf_reference = _continuous_joint_cdf(CONTINUOUS_Q_REFERENCE, OUTPUT_PARENTS)
    cdf_default = _continuous_joint_cdf(CONTINUOUS_Q_DEFAULT, OUTPUT_PARENTS)
    pmf_reference, ref_diag = _cdf_to_pmf(cdf_reference)
    pmf_default, default_diag = _cdf_to_pmf(cdf_default)
    tree, parent, conditionals, weights = _tree_projection(pmf_default)
    root = parent.index(-1)
    fitted, nmf_residuals, approximate_conditionals = _rank_limited_tree(
        _marginal(tree, root), parent, conditionals, NMF_RANK, NMF_SEED
    )
    structural_information = _structural_information(
        pmf_default, tree, parent, weights
    )
    rank_spectra = _rank_spectral_diagnostics(tree, fitted, parent, NMF_RANK)
    conditional_bounds = _conditional_kernel_tv_bounds(
        tree, parent, conditionals, approximate_conditionals
    )

    _, reference_parent, _, reference_weights = _tree_projection(pmf_reference)
    mi_estimation_error = float(np.max(np.abs(reference_weights - weights)))
    reference_optimal_weight = sum(
        float(reference_weights[child, par])
        for child, par in enumerate(reference_parent)
        if par != -1
    )
    selected_true_weight = sum(
        float(reference_weights[child, par])
        for child, par in enumerate(parent)
        if par != -1
    )
    selection_regret = max(0.0, reference_optimal_weight - selected_true_weight)
    selection_regret_bound = 2.0 * (K - 1) * mi_estimation_error

    numerical = _tv(pmf_reference, pmf_default)
    structural = _tv(pmf_default, tree)
    fit = _tv(tree, fitted)
    local_total = _tv(pmf_reference, fitted)
    sum_bound = numerical + structural + fit

    direct_marginals = [
        _continuous_subset_cdf(CONTINUOUS_Q_REFERENCE, (a,)) for a in range(K)
    ]
    direct_pairs = {
        f"{a},{b}": _continuous_subset_cdf(CONTINUOUS_Q_REFERENCE, (a, b))
        for a in range(K)
        for b in range(a + 1, K)
    }
    # Explicit indexing is clearer than repeated take calls for K=3.
    marginal_from_joint = [
        cdf_reference[:, -1, -1],
        cdf_reference[-1, :, -1],
        cdf_reference[-1, -1, :],
    ]
    pair_from_joint = {
        "0,1": cdf_reference[:, :, -1],
        "0,2": cdf_reference[:, -1, :],
        "1,2": cdf_reference[-1, :, :],
    }
    marginal_consistency = max(
        float(np.max(np.abs(marginal_from_joint[a] - direct_marginals[a]))) for a in range(K)
    )
    pair_consistency = max(
        float(np.max(np.abs(pair_from_joint[key] - direct_pairs[key]))) for key in direct_pairs
    )

    permuted_order = (1, 0, 2)
    permuted_parents = tuple(OUTPUT_PARENTS[a] for a in permuted_order)
    permuted = _continuous_joint_cdf(CONTINUOUS_Q_REFERENCE, permuted_parents)
    permutation_error = float(
        np.max(np.abs(permuted - np.transpose(cdf_reference, permuted_order)))
    )

    return {
        "configuration": {
            "K": K,
            "input_density": "two-component mixture of product Beta densities",
            "mixture_weights": [0.45, 0.55],
            "output_parents": [list(p) for p in OUTPUT_PARENTS],
            "edge_delay": {"kind": "uniform", "lo": 0.0, "hi": 0.3},
            "node_delay": "zero",
            "output_grid_points": len(CONTINUOUS_OUTPUT_GRID),
            "output_grid_range": [
                float(CONTINUOUS_OUTPUT_GRID[0]),
                float(CONTINUOUS_OUTPUT_GRID[-1]),
            ],
            "q_grid_default": CONTINUOUS_Q_DEFAULT,
            "q_grid_reference": CONTINUOUS_Q_REFERENCE,
            "tree_rank_limit": NMF_RANK,
            "materialization": (
                "nonnegative rank-limited conditional factorizations; these induce "
                "an explicit tree tensor network with bond rank at most the rank limit"
            ),
            "nmf_steps": NMF_STEPS,
            "conditional_probability_floor": CONDITIONAL_PROBABILITY_FLOOR,
            "seed": NMF_SEED,
        },
        "error_definition": (
            "TV between normalized cell-mass tables obtained from a common output CDF grid; "
            "this is a grid-estimated/discretized error, not a certified continuous TV bound"
        ),
        "tree_parent": parent,
        "tree_edge_weights_mutual_information": weights.tolist(),
        "quantitative_structure": {
            **structural_information,
            "reference_tree_parent": reference_parent,
            "max_edge_mi_estimation_abs_error": mi_estimation_error,
            "true_mi_weight_of_reference_optimal_tree": reference_optimal_weight,
            "true_mi_weight_of_selected_tree": selected_true_weight,
            "tree_selection_true_mi_regret": selection_regret,
            "tree_selection_regret_upper_bound": selection_regret_bound,
            "tree_selection_bound_violation": max(
                0.0, selection_regret - selection_regret_bound
            ),
        },
        "quantitative_rank": rank_spectra,
        "quantitative_nonnegative_fit": conditional_bounds,
        "fully_explicit_local_bound": {
            "numerical_grid_tv": numerical,
            "structural_kl_nats": structural_information["kl_to_chow_liu_tree"],
            "structural_pinsker_tv": structural_information["pinsker_tv_upper_bound"],
            "fit_conditional_kl_nats": conditional_bounds["conditional_kl_sum"],
            "fit_pinsker_tv": conditional_bounds["joint_pinsker_tv_upper_bound"],
            "fit_hybrid_conditional_tv": conditional_bounds[
                "parent_weighted_sum_bound"
            ],
            "fit_bound_used": min(
                conditional_bounds["joint_pinsker_tv_upper_bound"],
                conditional_bounds["parent_weighted_sum_bound"],
            ),
            "total_upper_bound": (
                numerical
                + structural_information["pinsker_tv_upper_bound"]
                + min(
                    conditional_bounds["joint_pinsker_tv_upper_bound"],
                    conditional_bounds["parent_weighted_sum_bound"],
                )
            ),
        },
        "errors": {
            "eta_numeric_grid_tv": numerical,
            "eta_structure_grid_tv": structural,
            "eta_fit_grid_tv": fit,
            "actual_total_grid_tv": local_total,
            "triangle_sum_bound": sum_bound,
            "triangle_slack": sum_bound - local_total,
            "actual_total_grid_l2": _l2(pmf_reference, fitted),
        },
        "diagnostics": {
            "reference_pmf": ref_diag,
            "default_pmf": default_diag,
            "reference_cdf_min": float(np.min(cdf_reference)),
            "reference_cdf_max": float(np.max(cdf_reference)),
            "reference_cdf_monotonicity_violation": _monotonicity_violation(cdf_reference),
            "marginal_consistency_max_abs": marginal_consistency,
            "pair_consistency_max_abs": pair_consistency,
            "output_permutation_max_abs": permutation_error,
            "nmf_conditional_frobenius_residuals": nmf_residuals,
        },
        "runtime_seconds": perf_counter() - start,
    }


def _finite_edge_cdf_matrix(grid_size: int) -> np.ndarray:
    y = np.arange(grid_size)[:, None]
    x = np.arange(grid_size)[None, :]
    threshold = y - x
    cdf = np.zeros_like(threshold, dtype=float)
    for delay, probability in enumerate(FINITE_EDGE_PMF):
        cdf += probability * (threshold >= delay)
    # The finite-state benchmark defines a saturated upper boundary:
    # min(max_i(X_i + E_i), grid_size - 1).  Therefore the final row is 1.
    cdf[-1, :] = 1.0
    return cdf


def _finite_propagate_cdf(
    pmf: np.ndarray,
    selected_outputs: Iterable[int] = range(K),
) -> np.ndarray:
    selected_outputs = tuple(int(a) for a in selected_outputs)
    edge_cdf = _finite_edge_cdf_matrix(pmf.shape[0])
    arguments: list[Any] = [pmf, list(range(K))]
    for local_output_axis, original_output_axis in enumerate(selected_outputs):
        output_label = K + local_output_axis
        for parent in OUTPUT_PARENTS[original_output_axis]:
            arguments.extend([edge_cdf, [output_label, parent]])
    arguments.append([K + a for a in range(len(selected_outputs))])
    return np.einsum(*arguments, optimize="greedy")


def _finite_propagate_pmf(pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    cdf = _finite_propagate_cdf(pmf)
    out, diag = _cdf_to_pmf(cdf)
    return out, cdf, diag


def _initial_finite_pmf() -> np.ndarray:
    grid = np.arange(FINITE_GRID_SIZE, dtype=float)
    components = []
    raw_components = (
        (
            np.exp(-0.8 * (grid - 1.0) ** 2),
            np.exp(-0.7 * (grid - 2.0) ** 2),
            np.exp(-0.9 * (grid - 1.5) ** 2),
        ),
        (
            np.exp(-0.9 * (grid - 3.5) ** 2),
            np.exp(-0.8 * (grid - 1.0) ** 2),
            np.exp(-0.7 * (grid - 3.0) ** 2),
        ),
    )
    for component in raw_components:
        normalized = [v / np.sum(v) for v in component]
        components.append(np.einsum("i,j,k->ijk", *normalized))
    pmf = 0.4 * components[0] + 0.6 * components[1]
    return pmf / np.sum(pmf)


def _finite_multilayer_audit() -> dict[str, Any]:
    start = perf_counter()
    p_true = _initial_finite_pmf()
    q_approx = p_true.copy()
    cumulative_bound = _tv(p_true, q_approx)
    quantitative_cumulative_bound = cumulative_bound
    layers = []
    maximum_check_violation = 0.0

    for layer in range(1, L + 1):
        inherited = _tv(p_true, q_approx)
        p_next, p_cdf, p_diag = _finite_propagate_pmf(p_true)
        propagated_q, qprop_cdf, qprop_diag = _finite_propagate_pmf(q_approx)
        propagated_error = _tv(p_next, propagated_q)

        tree, parent, conditionals, weights = _tree_projection(propagated_q)
        root = parent.index(-1)
        fitted, residuals, approximate_conditionals = _rank_limited_tree(
            _marginal(tree, root),
            parent,
            conditionals,
            NMF_RANK,
            NMF_SEED + layer,
        )
        structural_information = _structural_information(
            propagated_q, tree, parent, weights
        )
        rank_spectra = _rank_spectral_diagnostics(
            tree, fitted, parent, NMF_RANK
        )
        conditional_bounds = _conditional_kernel_tv_bounds(
            tree, parent, conditionals, approximate_conditionals
        )
        structure = _tv(propagated_q, tree)
        fit = _tv(tree, fitted)
        local_total = _tv(propagated_q, fitted)
        local_sum_bound = structure + fit
        actual = _tv(p_next, fitted)
        one_step_bound = inherited + local_total
        cumulative_bound += local_total
        fit_explicit_bound = min(
            conditional_bounds["joint_pinsker_tv_upper_bound"],
            conditional_bounds["parent_weighted_sum_bound"],
        )
        quantitative_local_bound = (
            structural_information["pinsker_tv_upper_bound"] + fit_explicit_bound
        )
        quantitative_one_step_bound = inherited + quantitative_local_bound
        sharpened_quantitative_bound = propagated_error + quantitative_local_bound
        quantitative_cumulative_bound += quantitative_local_bound
        observed_contraction_ratio = (
            propagated_error / inherited if inherited > 1e-15 else 0.0
        )

        recurrence_violation = max(0.0, actual - one_step_bound)
        contraction_violation = max(0.0, propagated_error - inherited)
        local_triangle_violation = max(0.0, local_total - local_sum_bound)
        cumulative_violation = max(0.0, actual - cumulative_bound)
        quantitative_one_step_violation = max(
            0.0, actual - quantitative_one_step_bound
        )
        sharpened_quantitative_violation = max(
            0.0, actual - sharpened_quantitative_bound
        )
        quantitative_cumulative_violation = max(
            0.0, actual - quantitative_cumulative_bound
        )
        maximum_check_violation = max(
            maximum_check_violation,
            recurrence_violation,
            contraction_violation,
            local_triangle_violation,
            cumulative_violation,
            structural_information["kl_identity_abs_error"],
            structural_information["cmi_identity_abs_error"],
            rank_spectra["tt_svd_bound_violation"],
            rank_spectra["rank_lower_bound_violation"],
            conditional_bounds["weighted_bound_violation"],
            conditional_bounds["supremum_bound_violation"],
            conditional_bounds["factorized_kl_identity_abs_error"],
            conditional_bounds["joint_pinsker_bound_violation"],
            conditional_bounds["frobenius_to_tv_bound_violation"],
            quantitative_one_step_violation,
            sharpened_quantitative_violation,
            quantitative_cumulative_violation,
        )

        direct_marginals = [
            _finite_propagate_cdf(q_approx, (a,)) for a in range(K)
        ]
        direct_pairs = {
            f"{a},{b}": _finite_propagate_cdf(q_approx, (a, b))
            for a in range(K)
            for b in range(a + 1, K)
        }
        marginal_from_joint = [
            qprop_cdf[:, -1, -1],
            qprop_cdf[-1, :, -1],
            qprop_cdf[-1, -1, :],
        ]
        pair_from_joint = {
            "0,1": qprop_cdf[:, :, -1],
            "0,2": qprop_cdf[:, -1, :],
            "1,2": qprop_cdf[-1, :, :],
        }
        marginal_consistency = max(
            float(np.max(np.abs(marginal_from_joint[a] - direct_marginals[a])))
            for a in range(K)
        )
        pair_consistency = max(
            float(np.max(np.abs(pair_from_joint[key] - direct_pairs[key])))
            for key in direct_pairs
        )

        permuted_parents = tuple(OUTPUT_PARENTS[a] for a in (1, 0, 2))
        # Reuse the generic finite contraction with a local parent override.
        arguments: list[Any] = [q_approx, list(range(K))]
        edge_cdf = _finite_edge_cdf_matrix(FINITE_GRID_SIZE)
        for output_axis, parents in enumerate(permuted_parents):
            for par in parents:
                arguments.extend([edge_cdf, [K + output_axis, par]])
        arguments.append([K, K + 1, K + 2])
        permuted = np.einsum(*arguments, optimize="greedy")
        permutation_error = float(
            np.max(np.abs(permuted - np.transpose(qprop_cdf, (1, 0, 2))))
        )

        layers.append(
            {
                "layer": layer,
                "tree_parent": parent,
                "tree_edge_weights_mutual_information": weights.tolist(),
                "quantitative_structure": structural_information,
                "quantitative_rank": rank_spectra,
                "quantitative_nonnegative_fit": conditional_bounds,
                "fully_explicit_local_bound": {
                    "numerical_tv": 0.0,
                    "numerical_reason": "finite-state max-plus propagation is evaluated by exact finite sums",
                    "structural_kl_nats": structural_information[
                        "kl_to_chow_liu_tree"
                    ],
                    "structural_pinsker_tv": structural_information[
                        "pinsker_tv_upper_bound"
                    ],
                    "fit_conditional_kl_nats": conditional_bounds[
                        "conditional_kl_sum"
                    ],
                    "fit_pinsker_tv": conditional_bounds[
                        "joint_pinsker_tv_upper_bound"
                    ],
                    "fit_hybrid_conditional_tv": conditional_bounds[
                        "parent_weighted_sum_bound"
                    ],
                    "fit_bound_used": fit_explicit_bound,
                    "local_upper_bound": quantitative_local_bound,
                },
                "errors": {
                    "inherited_tv": inherited,
                    "propagated_tv_before_materialization": propagated_error,
                    "eta_structure_tv": structure,
                    "eta_fit_tv": fit,
                    "eta_local_total_tv": local_total,
                    "eta_structure_plus_fit_bound": local_sum_bound,
                    "actual_global_tv": actual,
                    "one_step_bound": one_step_bound,
                    "cumulative_bound": cumulative_bound,
                    "observed_distribution_pair_contraction_ratio": observed_contraction_ratio,
                    "quantitative_local_upper_bound": quantitative_local_bound,
                    "quantitative_one_step_bound": quantitative_one_step_bound,
                    "sharpened_quantitative_bound_using_observed_propagated_tv": (
                        sharpened_quantitative_bound
                    ),
                    "quantitative_cumulative_bound": quantitative_cumulative_bound,
                    "recurrence_slack": one_step_bound - actual,
                    "cumulative_slack": cumulative_bound - actual,
                },
                "diagnostics": {
                    "true_pmf": p_diag,
                    "propagated_approx_pmf": qprop_diag,
                    "true_cdf_min": float(np.min(p_cdf)),
                    "true_cdf_max": float(np.max(p_cdf)),
                    "true_cdf_monotonicity_violation": _monotonicity_violation(p_cdf),
                    "approx_cdf_monotonicity_violation": _monotonicity_violation(qprop_cdf),
                    "marginal_consistency_max_abs": marginal_consistency,
                    "pair_consistency_max_abs": pair_consistency,
                    "output_permutation_max_abs": permutation_error,
                    "nmf_conditional_frobenius_residuals": residuals,
                },
                "check_violations": {
                    "markov_contraction": contraction_violation,
                    "local_triangle": local_triangle_violation,
                    "one_step_recurrence": recurrence_violation,
                    "cumulative_bound": cumulative_violation,
                    "quantitative_one_step_bound": quantitative_one_step_violation,
                    "sharpened_quantitative_bound": sharpened_quantitative_violation,
                    "quantitative_cumulative_bound": quantitative_cumulative_violation,
                },
            }
        )
        p_true = p_next
        q_approx = fitted

    return {
        "configuration": {
            "K": K,
            "L": L,
            "state_grid": list(range(FINITE_GRID_SIZE)),
            "initial_distribution": "two-component mixture of product discrete Gaussian profiles",
            "output_parents": [list(p) for p in OUTPUT_PARENTS],
            "edge_delay_pmf": FINITE_EDGE_PMF.tolist(),
            "upper_boundary": "saturated at state 16",
            "global_dobrushin_coefficient": 1.0,
            "global_dobrushin_witness": {
                "input_state_a": [0, 0, 0],
                "input_state_b": [16, 16, 16],
                "reason": "their conditional output supports are disjoint",
            },
            "node_delay": "zero",
            "tree_rank_limit": NMF_RANK,
            "materialization": (
                "nonnegative rank-limited conditional factorizations; these induce "
                "an explicit tree tensor network with bond rank at most the rank limit"
            ),
            "nmf_steps": NMF_STEPS,
            "conditional_probability_floor": CONDITIONAL_PROBABILITY_FLOOR,
            "seed_base": NMF_SEED,
        },
        "error_definition": "exact total variation between normalized finite-state probability mass tables",
        "layers": layers,
        "maximum_inequality_violation": maximum_check_violation,
        "runtime_seconds": perf_counter() - start,
    }


def _checks(single: dict[str, Any], multi: dict[str, Any]) -> list[dict[str, Any]]:
    tolerance = 5e-11
    checks = []

    def add(name: str, value: float, threshold: float, relation: str = "<=") -> None:
        if relation == "<=":
            passed = value <= threshold
        elif relation == ">":
            passed = value > threshold
        else:
            raise ValueError(relation)
        checks.append(
            {
                "name": name,
                "value": float(value),
                "relation": relation,
                "threshold": float(threshold),
                "passed": bool(passed),
            }
        )

    se = single["errors"]
    sd = single["diagnostics"]
    ss = single["quantitative_structure"]
    sr = single["quantitative_rank"]
    sf = single["quantitative_nonnegative_fit"]
    add("single_layer_triangle_inequality", se["actual_total_grid_tv"], se["triangle_sum_bound"] + tolerance)
    add("single_layer_reference_cdf_lower_range", -sd["reference_cdf_min"], tolerance)
    add("single_layer_reference_cdf_upper_range", sd["reference_cdf_max"], 1.0 + tolerance)
    add("single_layer_cdf_monotonicity", sd["reference_cdf_monotonicity_violation"], tolerance)
    add("single_layer_marginal_consistency", sd["marginal_consistency_max_abs"], tolerance)
    add("single_layer_pair_consistency", sd["pair_consistency_max_abs"], tolerance)
    add("single_layer_output_permutation", sd["output_permutation_max_abs"], tolerance)
    add("single_layer_normalization", sd["reference_pmf"]["normalization_abs_error_after"], tolerance)
    add(
        "single_layer_reference_negative_mass",
        sd["reference_pmf"]["negative_mass_before_clipping"],
        tolerance,
    )
    add(
        "single_layer_default_negative_mass",
        sd["default_pmf"]["negative_mass_before_clipping"],
        tolerance,
    )
    add("single_layer_structural_error_nonzero", se["eta_structure_grid_tv"], 1e-6, relation=">")
    add("single_layer_numeric_below_structural", se["eta_numeric_grid_tv"], se["eta_structure_grid_tv"])
    add("single_layer_structural_kl_identity", ss["kl_identity_abs_error"], tolerance)
    add("single_layer_structural_cmi_identity", ss["cmi_identity_abs_error"], tolerance)
    add(
        "single_layer_structural_pinsker",
        ss["actual_structural_tv"],
        ss["pinsker_tv_upper_bound"] + tolerance,
    )
    add(
        "single_layer_tree_selection_mi_regret",
        ss["tree_selection_true_mi_regret"],
        ss["tree_selection_regret_upper_bound"] + tolerance,
    )
    add("single_layer_tt_svd_tail_bound", sr["tt_svd_bound_violation"], tolerance)
    add("single_layer_rank_lower_bound", sr["rank_lower_bound_violation"], tolerance)
    add(
        "single_layer_weighted_conditional_fit_bound",
        sf["weighted_bound_violation"],
        tolerance,
    )
    add(
        "single_layer_supremum_conditional_fit_bound",
        sf["supremum_bound_violation"],
        tolerance,
    )
    add(
        "single_layer_factorized_fit_kl_identity",
        sf["factorized_kl_identity_abs_error"],
        tolerance,
    )
    add(
        "single_layer_joint_fit_pinsker",
        sf["joint_pinsker_bound_violation"],
        tolerance,
    )
    add(
        "single_layer_frobenius_to_tv",
        sf["frobenius_to_tv_bound_violation"],
        tolerance,
    )

    add("multilayer_all_inequalities", multi["maximum_inequality_violation"], tolerance)
    for row in multi["layers"]:
        layer = row["layer"]
        diag = row["diagnostics"]
        structure = row["quantitative_structure"]
        rank = row["quantitative_rank"]
        fit = row["quantitative_nonnegative_fit"]
        add(f"L{layer}_true_cdf_lower_range", -diag["true_cdf_min"], tolerance)
        add(f"L{layer}_true_cdf_upper_range", diag["true_cdf_max"], 1.0 + tolerance)
        add(f"L{layer}_true_cdf_monotonicity", diag["true_cdf_monotonicity_violation"], tolerance)
        add(f"L{layer}_approx_cdf_monotonicity", diag["approx_cdf_monotonicity_violation"], tolerance)
        add(f"L{layer}_marginal_consistency", diag["marginal_consistency_max_abs"], tolerance)
        add(f"L{layer}_pair_consistency", diag["pair_consistency_max_abs"], tolerance)
        add(f"L{layer}_output_permutation", diag["output_permutation_max_abs"], tolerance)
        add(
            f"L{layer}_true_normalization",
            diag["true_pmf"]["normalization_abs_error_after"],
            tolerance,
        )
        add(
            f"L{layer}_approx_normalization",
            diag["propagated_approx_pmf"]["normalization_abs_error_after"],
            tolerance,
        )
        add(
            f"L{layer}_true_negative_mass",
            diag["true_pmf"]["negative_mass_before_clipping"],
            tolerance,
        )
        add(
            f"L{layer}_approx_negative_mass",
            diag["propagated_approx_pmf"]["negative_mass_before_clipping"],
            tolerance,
        )
        add(f"L{layer}_structural_kl_identity", structure["kl_identity_abs_error"], tolerance)
        add(f"L{layer}_structural_cmi_identity", structure["cmi_identity_abs_error"], tolerance)
        add(
            f"L{layer}_structural_pinsker",
            structure["actual_structural_tv"],
            structure["pinsker_tv_upper_bound"] + tolerance,
        )
        add(f"L{layer}_tt_svd_tail_bound", rank["tt_svd_bound_violation"], tolerance)
        add(f"L{layer}_rank_lower_bound", rank["rank_lower_bound_violation"], tolerance)
        add(
            f"L{layer}_weighted_conditional_fit_bound",
            fit["weighted_bound_violation"],
            tolerance,
        )
        add(
            f"L{layer}_factorized_fit_kl_identity",
            fit["factorized_kl_identity_abs_error"],
            tolerance,
        )
        add(
            f"L{layer}_joint_fit_pinsker",
            fit["joint_pinsker_bound_violation"],
            tolerance,
        )
        add(
            f"L{layer}_frobenius_to_tv",
            fit["frobenius_to_tv_bound_violation"],
            tolerance,
        )
        add(
            f"L{layer}_observed_contraction_ratio",
            row["errors"]["observed_distribution_pair_contraction_ratio"],
            1.0 + tolerance,
        )
        add(
            f"L{layer}_quantitative_one_step_bound",
            row["check_violations"]["quantitative_one_step_bound"],
            tolerance,
        )
        add(
            f"L{layer}_quantitative_cumulative_bound",
            row["check_violations"]["quantitative_cumulative_bound"],
            tolerance,
        )
    return checks


def _plot_results(single: dict[str, Any], multi: dict[str, Any], output_dir: Path) -> list[str]:
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False})
    layers = np.asarray([row["layer"] for row in multi["layers"]])
    structure = np.asarray([row["errors"]["eta_structure_tv"] for row in multi["layers"]])
    fit = np.asarray([row["errors"]["eta_fit_tv"] for row in multi["layers"]])
    local_total = np.asarray([row["errors"]["eta_local_total_tv"] for row in multi["layers"]])

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(layers, structure, label="structural projection", color="#4C78A8")
    ax.bar(layers, fit, bottom=structure, label="rank-limited fit", color="#F58518")
    ax.plot(layers, local_total, "o-", color="#222222", linewidth=2, label="actual local TV")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total variation")
    ax.set_title("Per-layer materialization error decomposition (finite-state K=3)")
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    decomposition_path = output_dir / "multilayer_error_decomposition.png"
    fig.savefig(decomposition_path, dpi=220)
    plt.close(fig)

    actual = np.asarray([row["errors"]["actual_global_tv"] for row in multi["layers"]])
    one_step = np.asarray([row["errors"]["one_step_bound"] for row in multi["layers"]])
    cumulative = np.asarray([row["errors"]["cumulative_bound"] for row in multi["layers"]])
    propagated = np.asarray(
        [row["errors"]["propagated_tv_before_materialization"] for row in multi["layers"]]
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(layers, actual, "o-", linewidth=2.2, label="actual global TV")
    ax.plot(layers, propagated, "s--", linewidth=1.8, label="after exact propagation")
    ax.plot(layers, one_step, "^--", linewidth=1.8, label="one-step bound")
    ax.plot(layers, cumulative, "d-", linewidth=2.0, label="cumulative bound")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total variation")
    ax.set_title("Observed multilayer error and additive bounds")
    ax.set_xticks(layers)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    bound_path = output_dir / "multilayer_error_bound.png"
    fig.savefig(bound_path, dpi=220)
    plt.close(fig)

    retained_mi = np.asarray(
        [
            row["quantitative_structure"]["selected_edge_mutual_information_sum"]
            for row in multi["layers"]
        ]
    )
    residual_cmi = np.asarray(
        [
            row["quantitative_structure"]["conditional_mutual_information_sum"]
            for row in multi["layers"]
        ]
    )
    rank_lower = np.asarray(
        [
            row["quantitative_rank"]["best_possible_frobenius_lower_bound"]
            for row in multi["layers"]
        ]
    )
    rank_ttsvd = np.asarray(
        [
            row["quantitative_rank"]["constructed_unconstrained_tt_svd_frobenius"]
            for row in multi["layers"]
        ]
    )
    rank_upper = np.asarray(
        [
            row["quantitative_rank"]["tree_svd_tail_upper_bound"]
            for row in multi["layers"]
        ]
    )
    rank_nonnegative = np.asarray(
        [
            row["quantitative_rank"]["actual_nonnegative_materialization_frobenius"]
            for row in multi["layers"]
        ]
    )
    observed_alpha = np.asarray(
        [
            row["errors"]["observed_distribution_pair_contraction_ratio"]
            for row in multi["layers"]
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    axes[0].bar(layers, retained_mi, label="retained edge MI", color="#4C78A8")
    axes[0].bar(
        layers,
        residual_cmi,
        bottom=retained_mi,
        label="omitted CMI = structural KL",
        color="#E45756",
    )
    axes[0].set_title("Exact Chow–Liu KL identity")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Information (nats)")
    axes[0].set_xticks(layers)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(layers, rank_lower, "o-", label="spectral lower")
    axes[1].plot(layers, rank_ttsvd, "s-", label="constructed TT-SVD")
    axes[1].plot(layers, rank_upper, "^--", label="tree-SVD upper")
    axes[1].plot(layers, rank_nonnegative, "d-", label="nonnegative materialization")
    axes[1].set_title("Rank-3 approximation diagnostics")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Frobenius error")
    axes[1].set_xticks(layers)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(layers[1:], observed_alpha[1:], "o-", linewidth=2, label="observed pair ratio")
    axes[2].axhline(1.0, color="#E45756", linestyle="--", label="global Dobrushin coefficient")
    axes[2].set_ylim(0.0, 1.08)
    axes[2].set_title("Propagation contraction")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("TV ratio")
    axes[2].set_xticks(layers)
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    quantitative_path = output_dir / "multilayer_quantitative_diagnostics.png"
    fig.savefig(quantitative_path, dpi=220)
    plt.close(fig)

    def report_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    return [
        report_path(decomposition_path),
        report_path(bound_path),
        report_path(quantitative_path),
    ]


def run(output_dir: Path) -> dict[str, Any]:
    started = perf_counter()
    single = _continuous_single_layer_audit()
    multi = _finite_multilayer_audit()
    checks = _checks(single, multi)
    figures = _plot_results(single, multi, output_dir)
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "experiment": "deterministic_multilayer_ttns_maxplus_error_audit",
        "status": "pass" if passed else "fail",
        "scope": {
            "continuous_single_layer": "K=3 full joint on a common finite output grid",
            "finite_state_multilayer": "K=3, L=4 exact probability-mass propagation",
            "not_claimed": [
                "continuous high-dimensional TV certification",
                "statistical significance",
                "large-scale runtime scaling",
                "general quadrature convergence order",
            ],
        },
        "continuous_single_layer": single,
        "finite_state_multilayer": multi,
        "checks": checks,
        "summary": {
            "passed_checks": sum(item["passed"] for item in checks),
            "total_checks": len(checks),
            "all_checks_passed": passed,
            "figures": figures,
            "runtime_seconds": perf_counter() - started,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "simple_ttns_l2/reports/multilayer_error_analysis_metrics.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.output.parent)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{result['status'].upper()}: "
        f"{result['summary']['passed_checks']}/{result['summary']['total_checks']} checks; "
        f"wrote {args.output}"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
