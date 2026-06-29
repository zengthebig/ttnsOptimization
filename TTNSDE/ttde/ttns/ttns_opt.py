from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import jax
from flax import struct
from jax import numpy as jnp

from ttde.tt.tt_opt import NormalizedValue
from ttde.utils import cached_einsum


def _children_from_parent(parent: Sequence[int]) -> List[List[int]]:
    n = len(parent)
    children = [[] for _ in range(n)]
    root = None
    for node, p in enumerate(parent):
        if p == node or p == -1:
            if root is not None:
                raise ValueError("parent must contain exactly one root")
            root = node
            continue
        children[p].append(node)
    if root is None:
        raise ValueError("parent must contain a root")
    return children


def _root_from_parent(parent: Sequence[int]) -> int:
    return next(i for i, p in enumerate(parent) if p == i or p == -1)


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _edge_dims_from_ttns(
    ttns: "TTNSOpt",
    parent: Sequence[int],
    children: Sequence[Sequence[int]],
    root: int,
) -> Dict[Tuple[int, int], int]:
    if len(ttns.cores) != len(parent):
        raise ValueError("ttns.cores length must match parent length")

    edge_dims: Dict[Tuple[int, int], int] = {}
    for node, core in enumerate(ttns.cores):
        expected_ndim = 2 + len(children[node])
        if core.ndim != expected_ndim:
            raise ValueError(
                f"core[{node}] has ndim={core.ndim}, expected {expected_ndim} "
                f"(1 parent axis + 1 physical axis + {len(children[node])} child axes)"
            )

        if node == root:
            if core.shape[0] != 1:
                raise ValueError(f"root core parent-rank must be 1, got {core.shape[0]}")
        else:
            p = parent[node]
            key = _edge_key(node, p)
            dim = int(core.shape[0])
            prev = edge_dims.get(key)
            if prev is not None and prev != dim:
                raise ValueError(f"inconsistent edge rank on edge {key}: {prev} vs {dim}")
            edge_dims[key] = dim

        for child_idx, child in enumerate(children[node]):
            key = _edge_key(node, child)
            dim = int(core.shape[2 + child_idx])
            prev = edge_dims.get(key)
            if prev is not None and prev != dim:
                raise ValueError(f"inconsistent edge rank on edge {key}: {prev} vs {dim}")
            edge_dims[key] = dim

    return edge_dims


def _lognorm_and_normalized(value: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sqr_norm = (value ** 2).sum()
    is_zero = sqr_norm == 0
    safe = jnp.where(is_zero, 1.0, sqr_norm)
    return jnp.where(is_zero, -jnp.inf, 0.5 * jnp.log(safe)), value / jnp.sqrt(safe)


def _sum_logs(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.where((a == -jnp.inf) | (b == -jnp.inf), -jnp.inf, a + b)


def _postorder_nodes(children: Sequence[Sequence[int]], root: int) -> List[int]:
    order: List[int] = []
    stack: List[Tuple[int, int]] = [(root, 0)]
    while stack:
        node, idx = stack.pop()
        node_children = children[node]
        if idx < len(node_children):
            stack.append((node, idx + 1))
            stack.append((node_children[idx], 0))
        else:
            order.append(node)
    return order


def _run_postorder(
    children: Sequence[Sequence[int]],
    root: int,
    compute,
) -> jnp.ndarray:
    messages: Dict[int, jnp.ndarray] = {}
    for node in _postorder_nodes(children, root):
        child_msgs = [messages[child] for child in children[node]]
        messages[node] = compute(node, child_msgs)
    return messages[root]


def _run_normalized_postorder(
    children: Sequence[Sequence[int]],
    root: int,
    compute,
) -> NormalizedValue:
    messages: Dict[int, jnp.ndarray] = {}
    subtree_log: Dict[int, jnp.ndarray] = {}
    for node in _postorder_nodes(children, root):
        acc_log = jnp.array(0.0)
        for child in children[node]:
            acc_log = _sum_logs(acc_log, subtree_log[child])
        msg = compute(node, [messages[child] for child in children[node]])
        log_norm, normalized = _lognorm_and_normalized(msg)
        subtree_log[node] = _sum_logs(acc_log, log_norm)
        messages[node] = normalized
    root_msg = messages[root]
    total_log = subtree_log[root]
    scalar_log, scalar_norm = _lognorm_and_normalized(root_msg.squeeze())
    total_log = _sum_logs(total_log, scalar_log)
    return NormalizedValue(value=scalar_norm, log_norm=total_log)


def _inner_product_local(
    core1: jnp.ndarray,
    core2: jnp.ndarray,
    child_mats: Sequence[jnp.ndarray],
) -> jnp.ndarray:
    n_children = len(child_mats)
    if n_children == 0:
        return cached_einsum("pi,qi->pq", core1, core2)
    if n_children == 1:
        return cached_einsum("pia,ab,qib->pq", core1, child_mats[0], core2)
    if n_children == 2:
        return cached_einsum("piab,ac,bd,qicd->pq", core1, child_mats[0], child_mats[1], core2)
    if n_children == 3:
        return cached_einsum(
            "piabc,ad,be,cf,qidef->pq",
            core1,
            child_mats[0],
            child_mats[1],
            child_mats[2],
            core2,
        )

    weighted = core1
    for child_mat in child_mats:
        weighted = jnp.tensordot(weighted, child_mat, axes=([2], [0]))
    child_axes = tuple(range(1, 2 + n_children))
    return jnp.tensordot(weighted, core2, axes=(child_axes, child_axes))


def _eval_rank1_local(core: jnp.ndarray, vec: jnp.ndarray, child_vecs: Sequence[jnp.ndarray]) -> jnp.ndarray:
    n_children = len(child_vecs)
    if n_children == 0:
        return cached_einsum("rd,d->r", core, vec)
    if n_children == 1:
        return cached_einsum("rda,d,a->r", core, vec, child_vecs[0])
    if n_children == 2:
        return cached_einsum("rdab,d,a,b->r", core, vec, child_vecs[0], child_vecs[1])
    if n_children == 3:
        return cached_einsum("rdabc,d,a,b,c->r", core, vec, child_vecs[0], child_vecs[1], child_vecs[2])

    weighted = jnp.tensordot(core, vec, axes=([1], [0]))
    for child_vec in child_vecs:
        weighted = jnp.tensordot(weighted, child_vec, axes=([1], [0]))
    return weighted


def _batch_eval_rank1_local(
    core: jnp.ndarray,
    vec_batch: jnp.ndarray,
    child_vecs_batch: Sequence[jnp.ndarray],
) -> jnp.ndarray:
    n_children = len(child_vecs_batch)
    if n_children == 0:
        return cached_einsum("rd,nd->nr", core, vec_batch)
    if n_children == 1:
        return cached_einsum("rda,nd,na->nr", core, vec_batch, child_vecs_batch[0])
    if n_children == 2:
        return cached_einsum("rdab,nd,na,nb->nr", core, vec_batch, child_vecs_batch[0], child_vecs_batch[1])
    if n_children == 3:
        return cached_einsum(
            "rdabc,nd,na,nb,nc->nr",
            core,
            vec_batch,
            child_vecs_batch[0],
            child_vecs_batch[1],
            child_vecs_batch[2],
        )

    weighted = jax.vmap(lambda vec: jnp.tensordot(core, vec, axes=([1], [0])))(vec_batch)
    for child_vec in child_vecs_batch:
        weighted = jax.vmap(lambda w, c: jnp.tensordot(w, c, axes=([1], [0])))(weighted, child_vec)
    return weighted


def _quadratic_local(
    core: jnp.ndarray,
    metric: jnp.ndarray,
    child_mats: Sequence[jnp.ndarray],
) -> jnp.ndarray:
    n_children = len(child_mats)
    if n_children == 0:
        return cached_einsum("pi,ij,qj->pq", core, metric, core)
    if n_children == 1:
        return cached_einsum("pia,ab,qjb,ij->pq", core, child_mats[0], core, metric)
    if n_children == 2:
        return cached_einsum("piab,ac,bd,qjcd,ij->pq", core, child_mats[0], child_mats[1], core, metric)
    if n_children == 3:
        return cached_einsum(
            "piabc,ad,be,cf,qjdef,ij->pq",
            core,
            child_mats[0],
            child_mats[1],
            child_mats[2],
            core,
            metric,
        )

    weighted = core
    for child_mat in child_mats:
        weighted = jnp.tensordot(weighted, child_mat, axes=([2], [0]))
    child_axes = tuple(range(2, 2 + n_children))
    tmp = jnp.tensordot(weighted, core, axes=(child_axes, child_axes))
    return jnp.tensordot(tmp, metric, axes=([1, 3], [0, 1]))


@struct.dataclass
class TTNSOpt:
    cores: tuple[jnp.ndarray, ...]

    @classmethod
    def zeros(cls, parent: Sequence[int], dims: Sequence[int], rank: int):
        children = _children_from_parent(parent)
        root = next(i for i, p in enumerate(parent) if p == i or p == -1)
        cores = []
        for node in range(len(parent)):
            r_parent = 1 if node == root else rank
            r_children = [rank for _ in children[node]]
            shape = (r_parent, int(dims[node]), *r_children)
            cores.append(jnp.zeros(shape))
        return cls(tuple(cores))

    @classmethod
    def from_rank1_vectors(
        cls,
        vectors: jnp.ndarray,
        parent: Sequence[int],
        rank: int,
        edge_ranks: Dict[Tuple[int, int], int] | None = None,
    ):
        """
        vectors shape: [n_dims, basis_dim]

        edge_ranks: 可选的按无向边 rank 覆盖，key 为 (min(u,v), max(u,v))；
        未指定的边回退到统一 `rank`。用于给高扇出枢纽边单独降秩。
        """
        children = _children_from_parent(parent)
        root = next(i for i, p in enumerate(parent) if p == i or p == -1)

        def edge_rank(u: int, v: int) -> int:
            if edge_ranks is None:
                return rank
            return int(edge_ranks.get(_edge_key(u, v), rank))

        cores = []
        for node in range(len(parent)):
            r_parent = 1 if node == root else edge_rank(node, parent[node])
            r_children = [edge_rank(node, c) for c in children[node]]
            core = jnp.zeros((r_parent, vectors.shape[1], *r_children))
            index = (0, slice(None), *([0] * len(r_children)))
            core = core.at[index].set(vectors[node])
            cores.append(core)
        return cls(tuple(cores))


def _combine_ttns(
    lhs: TTNSOpt,
    rhs: TTNSOpt,
    parent: Sequence[int],
    rhs_root_scale: float,
) -> TTNSOpt:
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)

    edge_dims_l = _edge_dims_from_ttns(lhs, parent, children, root)
    edge_dims_r = _edge_dims_from_ttns(rhs, parent, children, root)
    if edge_dims_l.keys() != edge_dims_r.keys():
        raise ValueError("lhs and rhs must share the same tree topology")

    edge_dims_out = {e: edge_dims_l[e] + edge_dims_r[e] for e in edge_dims_l.keys()}
    out_cores: List[jnp.ndarray] = []

    for node in range(len(parent)):
        core_l = lhs.cores[node]
        core_r = rhs.cores[node]

        if core_l.ndim != core_r.ndim:
            raise ValueError(f"core[{node}] ndim mismatch: {core_l.ndim} vs {core_r.ndim}")
        if core_l.shape[1] != core_r.shape[1]:
            raise ValueError(
                f"core[{node}] physical dim mismatch: {core_l.shape[1]} vs {core_r.shape[1]}"
            )

        if node == root:
            parent_dim_out = 1
            lhs_parent_slice = slice(0, 1)
            rhs_parent_slice = slice(0, 1)
        else:
            p = parent[node]
            key_parent = _edge_key(node, p)
            l_parent = edge_dims_l[key_parent]
            r_parent = edge_dims_r[key_parent]
            parent_dim_out = edge_dims_out[key_parent]
            lhs_parent_slice = slice(0, l_parent)
            rhs_parent_slice = slice(l_parent, l_parent + r_parent)

        child_dims_out = []
        lhs_child_slices = []
        rhs_child_slices = []
        for child_idx, child in enumerate(children[node]):
            key_child = _edge_key(node, child)
            l_child = edge_dims_l[key_child]
            r_child = edge_dims_r[key_child]
            if core_l.shape[2 + child_idx] != l_child or core_r.shape[2 + child_idx] != r_child:
                raise ValueError(
                    f"core[{node}] child-axis rank mismatch on edge {key_child}: "
                    f"lhs={core_l.shape[2 + child_idx]}, rhs={core_r.shape[2 + child_idx]}, "
                    f"expected lhs={l_child}, rhs={r_child}"
                )
            child_dims_out.append(edge_dims_out[key_child])
            lhs_child_slices.append(slice(0, l_child))
            rhs_child_slices.append(slice(l_child, l_child + r_child))

        out_shape = (parent_dim_out, core_l.shape[1], *child_dims_out)
        out_core = jnp.zeros(out_shape, dtype=jnp.result_type(core_l, core_r))
        out_core = out_core.at[(lhs_parent_slice, slice(None), *lhs_child_slices)].set(core_l)

        rhs_scale = rhs_root_scale if node == root else 1.0
        out_core = out_core.at[(rhs_parent_slice, slice(None), *rhs_child_slices)].set(
            core_r * rhs_scale
        )
        out_cores.append(out_core)

    return TTNSOpt(tuple(out_cores))


def add_ttns(lhs: TTNSOpt, rhs: TTNSOpt, parent: Sequence[int]) -> TTNSOpt:
    return _combine_ttns(lhs, rhs, parent, rhs_root_scale=1.0)


def subtract_ttns(lhs: TTNSOpt, rhs: TTNSOpt, parent: Sequence[int]) -> TTNSOpt:
    return _combine_ttns(lhs, rhs, parent, rhs_root_scale=-1.0)


def normalized_inner_product_ttns(
    ttns1: TTNSOpt,
    ttns2: TTNSOpt,
    parent: Sequence[int],
) -> NormalizedValue:
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)

    def compute(node: int, child_mats: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return _inner_product_local(ttns1.cores[node], ttns2.cores[node], child_mats)

    return _run_normalized_postorder(children, root, compute)


def normalized_eval_rank1_ttns(
    ttns: TTNSOpt,
    vectors: jnp.ndarray,
    parent: Sequence[int],
) -> NormalizedValue:
    """
    Evaluate <ttns, rank1(vectors)> where vectors shape is [n_dims, basis_dim].
    """
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)

    def compute(node: int, child_vecs: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return _eval_rank1_local(ttns.cores[node], vectors[node], child_vecs)

    return _run_normalized_postorder(children, root, compute)


def eval_rank1_ttns(
    ttns: TTNSOpt,
    vectors: jnp.ndarray,
    parent: Sequence[int],
) -> jnp.ndarray:
    """
    Fast evaluation of <ttns, rank1(vectors)> without per-node log-normalization.
    vectors shape: [n_dims, basis_dim].
    """
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)

    def compute(node: int, child_vecs: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return _eval_rank1_local(ttns.cores[node], vectors[node], child_vecs)

    return jnp.asarray(_run_postorder(children, root, compute).squeeze())


def batch_eval_rank1_ttns(
    ttns: TTNSOpt,
    vectors_batch: jnp.ndarray,
    parent: Sequence[int],
) -> jnp.ndarray:
    """
    Fast batched evaluation of <ttns, rank1(vectors)>.
    vectors_batch shape: [batch, n_dims, basis_dim].
    Returns shape: [batch].
    """
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)

    def compute(node: int, child_vecs_batch: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return _batch_eval_rank1_local(
            ttns.cores[node],
            vectors_batch[:, node, :],
            child_vecs_batch,
        )

    return jnp.asarray(_run_postorder(children, root, compute).squeeze(-1))


def normalized_quadratic_form_ttns(
    ttns: TTNSOpt,
    matrices: jnp.ndarray,
    parent: Sequence[int],
) -> NormalizedValue:
    """
    Evaluate <ttns, A ttns>, where A is separable with local matrices[u].
    matrices shape: [n_dims, basis_dim, basis_dim].
    """
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)

    def compute(node: int, child_mats: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return _quadratic_local(ttns.cores[node], matrices[node], child_mats)

    return _run_normalized_postorder(children, root, compute)


def quadratic_form_ttns(
    ttns: TTNSOpt,
    matrices: jnp.ndarray,
    parent: Sequence[int],
) -> jnp.ndarray:
    """
    Fast evaluation of <ttns, A ttns> where A is separable with local matrices[u].
    matrices shape: [n_dims, basis_dim, basis_dim].
    """
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)

    def compute(node: int, child_mats: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return _quadratic_local(ttns.cores[node], matrices[node], child_mats)

    return jnp.asarray(_run_postorder(children, root, compute).squeeze())
