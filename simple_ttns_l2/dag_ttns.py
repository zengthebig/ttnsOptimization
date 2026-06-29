"""多父（multi-parent）DAG TTNS —— 多层 DAG 架构的地基。

与单父树 `TTNSDE/ttde/ttns/ttns_opt.py` 不同，这里每个节点可关联任意多条 bond，
因此天然支持多父结构（如 fork v-structure：$x_2$ 同时连 $x_0, x_1$）。

约定与限制（MVP）：
- 无向图必须**无环（polytree）**，以保证可精确收缩并做 dense 对照验证。
- 每条无向 bond 一个 rank；节点 core 形状为 `(physical_dim, *bond_ranks)`，
  bond 轴顺序由 `DAGGraph.incident[node]` 给出（关联边的全局 edge index 顺序）。
- 收缩用 einsum 自动完成；MVP 重正确性，不追求性能。
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct

_SYMBOLS = string.ascii_letters  # 52 个可用 einsum 字母

EdgeRanks = Union[int, Dict[int, int]]


@dataclass(frozen=True)
class DAGGraph:
    """无向无环（polytree）张量网络的结构描述（非 pytree，仅元信息）。"""

    n: int
    dims: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]       # 无向 bond，列表序即全局 edge index
    incident: Tuple[Tuple[int, ...], ...]    # 每节点关联 edge index，顺序对应 core bond 轴


@struct.dataclass
class DAGTTNS:
    """仅含 cores 的 pytree；图结构由 DAGGraph 单独传入算子。"""

    cores: Tuple[jnp.ndarray, ...]


def make_dag_graph(n: int, dims: Sequence[int], edges: Sequence[Tuple[int, int]]) -> DAGGraph:
    incident: List[List[int]] = [[] for _ in range(n)]
    norm_edges: List[Tuple[int, int]] = []
    for ei, (u, v) in enumerate(edges):
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError(f"非法边 {(u, v)}（n={n}）")
        norm_edges.append((int(u), int(v)))
        incident[u].append(ei)
        incident[v].append(ei)
    return DAGGraph(
        n=n,
        dims=tuple(int(d) for d in dims),
        edges=tuple(norm_edges),
        incident=tuple(tuple(e) for e in incident),
    )


def _edge_rank(edge_ranks: EdgeRanks, ei: int, default: int) -> int:
    if isinstance(edge_ranks, int):
        return int(edge_ranks)
    return int(edge_ranks.get(ei, default))


def _core_shape(graph: DAGGraph, node: int, edge_ranks: EdgeRanks, default_rank: int) -> Tuple[int, ...]:
    bond_ranks = [_edge_rank(edge_ranks, ei, default_rank) for ei in graph.incident[node]]
    return (graph.dims[node], *bond_ranks)


def random_dag_ttns(
    key: jnp.ndarray,
    graph: DAGGraph,
    edge_ranks: EdgeRanks = 2,
    default_rank: int = 2,
    scale: float = 1.0,
    dtype=jnp.float64,
) -> DAGTTNS:
    keys = jax.random.split(key, graph.n)
    cores = []
    for node in range(graph.n):
        shape = _core_shape(graph, node, edge_ranks, default_rank)
        cores.append(jax.random.normal(keys[node], shape, dtype=dtype) * scale)
    return DAGTTNS(tuple(cores))


def dag_ttns_from_rank1(
    vectors: Sequence[jnp.ndarray],
    graph: DAGGraph,
    edge_ranks: EdgeRanks = 1,
    default_rank: int = 1,
) -> DAGTTNS:
    """把每个物理腿的 rank-1 向量放在所有 bond 的 index 0 上（其余补零）。"""
    cores = []
    for node in range(graph.n):
        shape = _core_shape(graph, node, edge_ranks, default_rank)
        core = jnp.zeros(shape, dtype=vectors[node].dtype)
        idx = (slice(None), *([0] * len(graph.incident[node])))
        core = core.at[idx].set(vectors[node])
        cores.append(core)
    return DAGTTNS(tuple(cores))


def _edge_letters(offset: int, n_edges: int) -> Dict[int, str]:
    return {ei: _SYMBOLS[offset + ei] for ei in range(n_edges)}


def _phys_letters(offset: int, n_nodes: int) -> Dict[int, str]:
    return {i: _SYMBOLS[offset + i] for i in range(n_nodes)}


def dag_full_tensor(ttns: DAGTTNS, graph: DAGGraph) -> jnp.ndarray:
    """收缩所有 bond，返回稠密张量，形状 = `tuple(dims)`。"""
    n_edges = len(graph.edges)
    edge_l = _edge_letters(0, n_edges)
    phys_l = _phys_letters(n_edges, graph.n)
    subs, operands = [], []
    for node in range(graph.n):
        subs.append(phys_l[node] + "".join(edge_l[ei] for ei in graph.incident[node]))
        operands.append(ttns.cores[node])
    expr = ",".join(subs) + "->" + "".join(phys_l[i] for i in range(graph.n))
    return jnp.einsum(expr, *operands, optimize="greedy")


def dag_eval_rank1(ttns: DAGTTNS, graph: DAGGraph, vectors: Sequence[jnp.ndarray]) -> jnp.ndarray:
    r"""计算 $\langle T, \otimes_k v_k \rangle$：每个物理腿与 `vectors[k]` 收缩后再缩所有 bond。"""
    n_edges = len(graph.edges)
    edge_l = _edge_letters(0, n_edges)
    phys_l = _phys_letters(n_edges, graph.n)
    subs, operands = [], []
    for node in range(graph.n):
        subs.append(phys_l[node] + "".join(edge_l[ei] for ei in graph.incident[node]))
        operands.append(ttns.cores[node])
        subs.append(phys_l[node])
        operands.append(vectors[node])
    expr = ",".join(subs) + "->"
    return jnp.einsum(expr, *operands, optimize="greedy")


def dag_inner_product(ttns1: DAGTTNS, ttns2: DAGTTNS, graph: DAGGraph) -> jnp.ndarray:
    r"""$\langle T_1, T_2 \rangle$：两网络共享物理腿、各用一套 bond 字母。"""
    n_edges = len(graph.edges)
    e1 = _edge_letters(0, n_edges)
    e2 = _edge_letters(n_edges, n_edges)
    phys_l = _phys_letters(2 * n_edges, graph.n)
    subs, operands = [], []
    for node in range(graph.n):
        subs.append(phys_l[node] + "".join(e1[ei] for ei in graph.incident[node]))
        operands.append(ttns1.cores[node])
        subs.append(phys_l[node] + "".join(e2[ei] for ei in graph.incident[node]))
        operands.append(ttns2.cores[node])
    expr = ",".join(subs) + "->"
    return jnp.einsum(expr, *operands, optimize="greedy")


def dag_quadratic_form(ttns: DAGTTNS, graph: DAGGraph, gram: Sequence[jnp.ndarray]) -> jnp.ndarray:
    r"""$\int q^2 = \sum_{p,p'} T[p] T[p'] \prod_k G_k[p_k, p'_k]$，`gram[k]` 形状 `(d_k, d_k)`。"""
    n_edges = len(graph.edges)
    e1 = _edge_letters(0, n_edges)
    e2 = _edge_letters(n_edges, n_edges)
    phys_a = _phys_letters(2 * n_edges, graph.n)
    phys_b = _phys_letters(2 * n_edges + graph.n, graph.n)
    subs, operands = [], []
    for node in range(graph.n):
        subs.append(phys_a[node] + "".join(e1[ei] for ei in graph.incident[node]))
        operands.append(ttns.cores[node])
        subs.append(phys_a[node] + phys_b[node])
        operands.append(gram[node])
        subs.append(phys_b[node] + "".join(e2[ei] for ei in graph.incident[node]))
        operands.append(ttns.cores[node])
    expr = ",".join(subs) + "->"
    return jnp.einsum(expr, *operands, optimize="greedy")


def dag_integral(ttns: DAGTTNS, graph: DAGGraph, basis_integrals: Sequence[jnp.ndarray]) -> jnp.ndarray:
    r"""$\int q = \langle T, \otimes_k I_k \rangle$，`basis_integrals[k]` 形状 `(d_k,)`。"""
    return dag_eval_rank1(ttns, graph, basis_integrals)


def dag_batch_eval_rank1(ttns: DAGTTNS, graph: DAGGraph, basis_vectors_batch: jnp.ndarray) -> jnp.ndarray:
    """对一批 rank-1 基向量求值，`basis_vectors_batch` 形状 `[batch, n_nodes, basis_dim]`。"""
    return jax.vmap(lambda vecs: dag_eval_rank1(ttns, graph, vecs))(basis_vectors_batch)


def dag_normalize_by_integral(
    ttns: DAGTTNS,
    graph: DAGGraph,
    basis_integrals: Sequence[jnp.ndarray],
    root: int = 0,
    eps: float = 1e-12,
) -> Tuple[DAGTTNS, jnp.ndarray]:
    r"""投影到单位积分 $q \leftarrow q / \int q$，通过缩放 `root` 节点的 core 实现。"""
    z = dag_integral(ttns, graph, basis_integrals)
    safe_z = jnp.where(jnp.abs(z) < eps, 1.0, z)
    cores = list(ttns.cores)
    cores[root] = cores[root] / safe_z
    return DAGTTNS(tuple(cores)), z
