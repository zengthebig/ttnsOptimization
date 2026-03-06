from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ChowLiuTree:
    parent: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    mutual_information: np.ndarray


def _quantile_bin_indices(values: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int32)
    inner_edges = edges[1:-1]
    return np.digitize(values, inner_edges, right=False).astype(np.int32)


def discretize_samples_quantile(samples: np.ndarray, n_bins: int = 16) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must have shape [n_samples, n_dims], got {x.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    return np.stack([_quantile_bin_indices(x[:, dim], n_bins) for dim in range(x.shape[1])], axis=1)


def estimate_mutual_information_matrix(discrete_samples: np.ndarray) -> np.ndarray:
    x = np.asarray(discrete_samples, dtype=np.int32)
    if x.ndim != 2:
        raise ValueError(f"discrete_samples must have shape [n_samples, n_dims], got {x.shape}")

    n_samples, n_dims = x.shape
    if n_samples == 0:
        raise ValueError("discrete_samples must contain at least one sample")

    mi = np.zeros((n_dims, n_dims), dtype=np.float64)
    cardinalities = [int(x[:, dim].max()) + 1 for dim in range(n_dims)]

    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            counts = np.zeros((cardinalities[i], cardinalities[j]), dtype=np.float64)
            np.add.at(counts, (x[:, i], x[:, j]), 1.0)
            p_ij = counts / float(n_samples)
            p_i = p_ij.sum(axis=1, keepdims=True)
            p_j = p_ij.sum(axis=0, keepdims=True)
            mask = p_ij > 0.0
            denom = p_i * p_j
            score = np.sum(p_ij[mask] * (np.log(p_ij[mask]) - np.log(denom[mask])))
            mi[i, j] = score
            mi[j, i] = score

    return mi


def _maximum_spanning_tree(weights: np.ndarray) -> list[tuple[int, int]]:
    n_dims = weights.shape[0]
    if weights.shape != (n_dims, n_dims):
        raise ValueError(f"weights must be square, got {weights.shape}")

    visited = np.zeros(n_dims, dtype=bool)
    visited[0] = True
    edges: list[tuple[int, int]] = []

    while len(edges) < n_dims - 1:
        best_u = -1
        best_v = -1
        best_w = -np.inf
        for u in range(n_dims):
            if not visited[u]:
                continue
            for v in range(n_dims):
                if visited[v] or u == v:
                    continue
                w = float(weights[u, v])
                if w > best_w:
                    best_w = w
                    best_u = u
                    best_v = v
        if best_u < 0 or best_v < 0:
            raise ValueError("failed to construct spanning tree from weights")
        visited[best_v] = True
        edges.append((best_u, best_v))

    return edges


def orient_tree(edges: Sequence[tuple[int, int]], n_dims: int, root: int = 0) -> tuple[int, ...]:
    if not (0 <= root < n_dims):
        raise ValueError(f"root must lie in [0, {n_dims}), got {root}")

    adjacency = [[] for _ in range(n_dims)]
    for u, v in edges:
        if not (0 <= u < n_dims and 0 <= v < n_dims):
            raise ValueError(f"edge {(u, v)} is out of bounds for n_dims={n_dims}")
        adjacency[u].append(v)
        adjacency[v].append(u)

    parent = [-2] * n_dims
    parent[root] = root
    queue = deque([root])
    visited = {root}

    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            queue.append(v)

    if len(visited) != n_dims:
        raise ValueError("edges do not form a connected tree")
    return tuple(parent)


def estimate_chow_liu_tree(samples: np.ndarray, n_bins: int = 16, root: int = 0) -> ChowLiuTree:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must have shape [n_samples, n_dims], got {x.shape}")

    discrete = discretize_samples_quantile(x, n_bins=n_bins)
    mi = estimate_mutual_information_matrix(discrete)
    edges = _maximum_spanning_tree(mi)
    parent = orient_tree(edges, n_dims=x.shape[1], root=root)
    return ChowLiuTree(parent=parent, edges=tuple(edges), mutual_information=mi)
