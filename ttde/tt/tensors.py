from __future__ import annotations

from typing import Sequence, List, Optional

import jax
from jax import numpy as jnp
from flax import struct


@struct.dataclass
class TTNS:
    """Tensor Train Network State (tree-structured).

    Dimension convention for each core ``G_k``:
        G_k[alpha_parent, i_k, alpha_child1, alpha_child2, ...]

    ``alpha_parent`` is the virtual dimension to the parent (size 1 for root),
    ``i_k`` is the physical dimension, and the remaining axes correspond to the
    children in the order given by ``neighbors[k]`` with the parent filtered out.
    """

    cores: List[jnp.ndarray]
    neighbors: List[List[int]]
    root: Optional[int] = None
    parent: Optional[List[int]] = None

    @property
    def n_nodes(self) -> int:
        return len(self.cores)

    @property
    def n_dims(self) -> int:
        return self.n_nodes

    def _build_parent(self, root: int) -> List[int]:
        parent = [-1] * self.n_nodes
        parent[root] = root
        stack = [root]
        while stack:
            node = stack.pop()
            for nbr in self.neighbors[node]:
                if parent[nbr] != -1:
                    continue
                parent[nbr] = node
                stack.append(nbr)
        return parent

    def _resolve_parent(self) -> List[int]:
        if self.parent is not None:
            assert len(self.parent) == self.n_nodes
            return list(self.parent)
        root = 0 if self.root is None else self.root
        return self._build_parent(root)

    def _resolve_root(self, parent: List[int]) -> int:
        if self.root is not None:
            return self.root
        if -1 in parent:
            return parent.index(-1)
        for idx, value in enumerate(parent):
            if idx == value:
                return idx
        return 0

    def validate_tree(self) -> None:
        n_nodes = self.n_nodes
        assert len(self.neighbors) == n_nodes

        for node, nbrs in enumerate(self.neighbors):
            assert node not in nbrs
            assert len(set(nbrs)) == len(nbrs)
            for nbr in nbrs:
                assert 0 <= nbr < n_nodes
                assert node in self.neighbors[nbr]

        parent = self._resolve_parent()
        root = self._resolve_root(parent)
        assert 0 <= root < n_nodes

        visited = set()
        stack = [(root, -1)]
        while stack:
            node, prev = stack.pop()
            if node in visited:
                raise AssertionError("Graph contains a cycle")
            visited.add(node)
            for nbr in self.neighbors[node]:
                if nbr == prev:
                    continue
                stack.append((nbr, node))
        assert len(visited) == n_nodes

        for node, core in enumerate(self.cores):
            degree = len(self.neighbors[node])
            assert core.ndim == 2 + degree

        children_by_node = []
        for node in range(n_nodes):
            node_parent = parent[node]
            children = [nbr for nbr in self.neighbors[node] if nbr != node_parent]
            children_by_node.append(children)

        for node, core in enumerate(self.cores):
            if node == root:
                assert core.shape[0] == 1
                continue
            parent_node = parent[node]
            parent_children = children_by_node[parent_node]
            child_index = parent_children.index(node)
            parent_axis = 2 + child_index
            assert core.shape[0] == self.cores[parent_node].shape[parent_axis]

    def postorder(self) -> List[int]:
        parent = self._resolve_parent()
        root = self._resolve_root(parent)
        children_by_node = []
        for node in range(self.n_nodes):
            node_parent = parent[node]
            children = [nbr for nbr in self.neighbors[node] if nbr != node_parent]
            children_by_node.append(children)

        order = []
        stack = [(root, 0)]
        while stack:
            node, idx = stack.pop()
            children = children_by_node[node]
            if idx < len(children):
                stack.append((node, idx + 1))
                stack.append((children[idx], 0))
            else:
                order.append(node)
        return order


@struct.dataclass
class TT:
    @classmethod
    def zeros(cls, dims: Sequence[int], rs: Sequence[int]) -> TT:
        assert len(dims) == len(rs) + 1

        rs = [1] + list(rs) + [1]
        cores = [jnp.zeros((rs[i], dim, rs[i + 1])) for i, dim in enumerate(dims)]

        return cls(cores)

    @classmethod
    def generate_random(cls, key: jnp.ndarray, dims: Sequence[int], rs: Sequence[int]) -> TT:
        assert len(dims) == len(rs) + 1

        rs = [1] + list(rs) + [1]
        keys = jax.random.split(key, len(dims))
        cores = [jax.random.normal(key, (rs[i], dim, rs[i + 1])) for i, (dim, key) in enumerate(zip(dims, keys))]

        return cls(cores)

    cores: List[jnp.ndarray]

    @property
    def n_dims(self):
        return len(self.cores)

    @property
    def full_tensor(self) -> jnp.ndarray:
        res = self.cores[0]
        for core in self.cores[1:]:
            res = jnp.einsum('...r,riR->...iR', res, core)
        return jnp.squeeze(res, (0, -1))

    def reverse(self) -> TT:
        return TT([transpose_core(core) for core in self.cores[::-1]])

    def astype(self, dtype: jnp.dtype) -> TT:
        return TT([core.astype(dtype) for core in self.cores])

    def __sub__(self, other: TT):
        return subtract(self, other)

    def as_ttns(self) -> TTNS:
        n_dims = self.n_dims
        neighbors = [[] for _ in range(n_dims)]
        for idx in range(n_dims):
            if idx > 0:
                neighbors[idx].append(idx - 1)
            if idx < n_dims - 1:
                neighbors[idx].append(idx + 1)
        return TTNS(self.cores, neighbors, root=0)


@struct.dataclass
class TTOperator:
    @classmethod
    def generate_random(
        cls, key: jnp.ndarray, dims_from: Sequence[int], dims_to: Sequence[int], rs: Sequence[int]
    ) -> TTOperator:
        n_dims = len(dims_from)

        assert len(dims_from) == n_dims
        assert len(dims_to) == n_dims
        assert len(rs) + 1 == n_dims

        rs = [1] + list(rs) + [1]
        keys = jax.random.split(key, n_dims)
        cores = [
            jax.random.normal(key, (rs[i], dim_from, dim_to, rs[i + 1]))
            for i, (dim_from, dim_to, key) in enumerate(zip(dims_from, dims_to, keys))
        ]

        return cls(cores)

    cores: List[jnp.ndarray]

    @property
    def full_operator(self) -> jnp.ndarray:
        res = self.cores[0]
        for core in self.cores[1:]:
            res = jnp.einsum('...r,rijR->...ijR', res, core)
        return jnp.squeeze(res, (0, -1))

    def reverse(self):
        # idk, what should I do with axes 1 and 2.
        return TTOperator([jnp.moveaxis(core, (0, 1, 2, 3), (3, 1, 2, 0)) for core in self.cores[::-1]])


def transpose_core(core: jnp.ndarray) -> jnp.ndarray:
    return jnp.moveaxis(core, (0, 1, 2), (2, 1, 0))


def subtract(lhs: TT, rhs: TT) -> TT:
    assert lhs.n_dims == rhs.n_dims

    if lhs.n_dims == 1:
        return TT([lhs.cores[0] - rhs.cores[0]])

    first = jnp.concatenate([lhs.cores[0], -rhs.cores[0]], axis=-1)
    last = jnp.concatenate([lhs.cores[-1], rhs.cores[-1]], axis=0)
    inner = [
        jnp.concatenate(
            [
                jnp.concatenate([c1, jnp.zeros((c1.shape[0], c1.shape[1], c2.shape[2]))], axis=-1),
                jnp.concatenate([jnp.zeros((c2.shape[0], c2.shape[1], c1.shape[2])), c2], axis=-1),
            ],
            axis=0,
        ) for c1, c2 in zip(lhs.cores[1:-1], rhs.cores[1:-1])
    ]

    return TT([first] + inner + [last])
