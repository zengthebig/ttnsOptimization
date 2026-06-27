from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple


def fork_junction_parent(n_dims: int = 6) -> List[int]:
    """
    fork DAG 的联结树风格 spanning tree。

    6 节点：0–1–2 链前缀 + x3,x4 挂 x2（短路径覆盖 (2,3),(2,4)）。

    8 节点： nuisance 侧枝挂 x2，x7 挂 x6，x7→x2 仅 2 跳（chain 需 5 跳）。

    6D parent = [0, 0, 1, 2, 2, 4]
    7D parent = [0, 0, 1, 2, 2, 2, 5]
    8D parent = [0, 0, 1, 2, 2, 2, 2, 6]
    """
    if n_dims == 6:
        return [0, 0, 1, 2, 2, 4]
    if n_dims == 7:
        return [0, 0, 1, 2, 2, 2, 5]
    if n_dims == 8:
        return [0, 0, 1, 2, 2, 2, 2, 6]
    raise ValueError(f"fork_junction_parent 目前支持 n_dims=6/7/8，收到 {n_dims}")


def validate_tree_parent(parent: Sequence[int]) -> None:
    n = len(parent)
    roots = [i for i, p in enumerate(parent) if p == i or p == -1]
    if len(roots) != 1:
        raise ValueError(f"parent 必须恰有一个根，得到 roots={roots}")

    root = roots[0]
    children = [[] for _ in range(n)]
    for node, p in enumerate(parent):
        if node == root:
            continue
        if p < 0 or p >= n or p == node:
            raise ValueError(f"非法 parent[{node}]={p}")
        children[p].append(node)

    order: List[int] = []
    stack = [(root, 0)]
    while stack:
        node, idx = stack.pop()
        ch = children[node]
        if idx < len(ch):
            stack.append((node, idx + 1))
            stack.append((ch[idx], 0))
        else:
            order.append(node)

    if len(order) != n:
        raise ValueError("parent 不是连通无环树")


def dag_edges_to_child_parents(edges: Iterable[Tuple[int, int]]) -> dict[int, Set[int]]:
    parents: dict[int, Set[int]] = {}
    for u, v in edges:
        parents.setdefault(v, set()).add(u)
    return parents


def clique_covers_dag_fork(edges: Sequence[Tuple[int, int]], clique: Set[int]) -> bool:
    """检查 clique 是否包含 fork DAG 的关键双父节点 {0,1,2}。"""
    required = {0, 1, 2}
    return required.issubset(clique)
