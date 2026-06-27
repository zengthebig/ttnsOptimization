from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple


def fork_junction_parent(n_dims: int = 6) -> List[int]:
    """
    fork DAG 的联结树风格 spanning tree（6 节点 MVP，手写）。

    树形（parent 指向父节点，根 self-loop）::

        0 (root)
       / \\
      1   2
         / \\
        3   4
         \\ /
          5

    parent = [0, 0, 0, 2, 2, 3]

    相对 chain，节点 0/1/2 在根下汇聚，更短路径覆盖双父依赖 clique {0,1,2}。
    """
    if n_dims != 6:
        raise ValueError(f"fork_junction_parent 目前仅支持 n_dims=6，收到 {n_dims}")
    return [0, 0, 0, 2, 2, 3]


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
