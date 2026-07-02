"""多层 DAG 传播 pipeline：拓扑分层 + delay 核逐层采样 + 多父结构图构造。

与用户定义对齐：
- **层内无父子边，层间多父边**：节点按拓扑分层；每条边 $u\\to v$ 中 $u$ 在更上层。
- **delay 核** $K_v(x_v\\mid \\mathrm{pa}(v))$：下层节点仅依赖其上层父；给定上层时同层节点条件独立
  （即"每层由若干 TTNS / 森林表示"）。
- **传播 = 逐层前向采样**（源层 → 过核 → 下层），产物为全联合样本；随后用多父
  DAGTTNS 拟合（方案 2）。全联合图结构 = 所有层间边的无向并集（可含环，如 diamond）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import jax
from jax import numpy as jnp

from simple_ttns_l2.dag_ttns import DAGGraph, make_dag_graph

# 源采样器：(rng, n) -> [n] ；delay 核：(parent_vals[n, k], rng, n) -> [n]
SourceSampler = Callable[[np.random.Generator, int], np.ndarray]
DelayKernel = Callable[[np.ndarray, np.random.Generator, int], np.ndarray]


@dataclass(frozen=True)
class MultiLayerSpec:
    """多层 DAG 规格。节点标号 0..n-1，须与 `layers` 展开一致。"""

    layers: Tuple[Tuple[int, ...], ...]          # 拓扑分层
    edges: Tuple[Tuple[int, int], ...]           # 有向边 (parent, child)，parent 在更上层

    @property
    def n(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def topo_order(self) -> List[int]:
        return [node for layer in self.layers for node in layer]

    def parents(self, node: int) -> List[int]:
        return [u for (u, v) in self.edges if v == node]

    def validate(self) -> None:
        seen: List[int] = self.topo_order
        if sorted(seen) != list(range(self.n)):
            raise ValueError(f"层划分节点必须恰为 0..{self.n - 1}，实际 {sorted(seen)}")
        layer_of = {node: li for li, layer in enumerate(self.layers) for node in layer}
        for (u, v) in self.edges:
            if layer_of[u] >= layer_of[v]:
                raise ValueError(f"边 {(u, v)} 必须从上层指向下层（layer {layer_of[u]} -> {layer_of[v]}）")


def build_layered_spec(layer_sizes: Sequence[int], fanin: int = 2, wrap: bool = True) -> MultiLayerSpec:
    """构造分层 banded 多父 DAG：下层每个节点连上层 `fanin` 个相邻父（可 wrap 成环）。

    相邻下层节点共享父节点 → 产生大量短环（树不可表达），同时保持局部连接、treewidth 受控。
    """
    layers: List[Tuple[int, ...]] = []
    idx = 0
    for sz in layer_sizes:
        layers.append(tuple(range(idx, idx + sz)))
        idx += sz
    edges = set()
    for li in range(1, len(layers)):
        prev, cur = layers[li - 1], layers[li]
        psz = len(prev)
        for j, node in enumerate(cur):
            for f in range(fanin):
                pos = (j + f) % psz if wrap else min(j + f, psz - 1)
                edges.add((prev[pos], node))
    return MultiLayerSpec(tuple(layers), tuple(sorted(edges)))


def build_clustered_spec(
    n_layers: int, clusters: Sequence[int], fanin: int = 2, wrap: bool = True
) -> MultiLayerSpec:
    """构造**簇内相连、簇间独立**的多层 DAG（每层大小 = sum(clusters)）。

    每层被划成若干簇（大小由 `clusters` 给定，簇的划分逐层对齐）；层间边只在**同簇内**
    连接（每个子节点连同簇内 `fanin` 个相邻父，可 wrap）。不同簇不共享任何祖先源
    → `structural_blocks` 能精确裂成对应的块，且块内节点存在真实非树相关（环）。
    """
    sz = sum(clusters)
    layers: List[Tuple[int, ...]] = []
    idx = 0
    for _ in range(n_layers):
        layers.append(tuple(range(idx, idx + sz)))
        idx += sz
    edges = set()
    for li in range(1, n_layers):
        prev, cur = layers[li - 1], layers[li]
        off = 0
        for csz in clusters:
            pcl, ccl = prev[off:off + csz], cur[off:off + csz]
            for j, node in enumerate(ccl):
                for f in range(min(fanin, csz)):
                    pos = (j + f) % csz if wrap else min(j + f, csz - 1)
                    edges.add((pcl[pos], node))
            off += csz
    return MultiLayerSpec(tuple(layers), tuple(sorted(edges)))


def build_graph_from_spec(spec: MultiLayerSpec, basis_dim: int) -> DAGGraph:
    """全联合图 = 所有层间边的无向并集；每个物理维取 `basis_dim`。"""
    spec.validate()
    return make_dag_graph(spec.n, [basis_dim] * spec.n, [(u, v) for (u, v) in spec.edges])


def sample_joint(
    spec: MultiLayerSpec,
    sources: Dict[int, SourceSampler],
    kernels: Dict[int, DelayKernel],
    key: jnp.ndarray,
    n: int,
    clip: Tuple[float, float] = (1e-4, 1.0 - 1e-4),
) -> jnp.ndarray:
    """按拓扑序逐层前向采样全联合，返回 `[n, n_nodes]`（jnp，已裁剪到定义域）。"""
    spec.validate()
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    vals: Dict[int, np.ndarray] = {}
    for node in spec.topo_order:
        ps = spec.parents(node)
        if not ps:
            if node not in sources:
                raise ValueError(f"源节点 {node} 缺少 source sampler")
            vals[node] = np.asarray(sources[node](rng, n), dtype=float)
        else:
            if node not in kernels:
                raise ValueError(f"下层节点 {node} 缺少 delay kernel")
            parent_vals = np.stack([vals[p] for p in ps], axis=1)  # [n, k]
            vals[node] = np.asarray(kernels[node](parent_vals, rng, n), dtype=float)
    xs = np.stack([vals[i] for i in range(spec.n)], axis=1)
    return jnp.asarray(np.clip(xs, clip[0], clip[1]))
