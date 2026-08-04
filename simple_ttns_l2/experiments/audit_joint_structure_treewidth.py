"""联合结构图 G_S / G_S^+ 的树宽审计（确定性小图组合计算）。

对应论文稿 output/pdf/ttns_maxplus_layer_propagation_theory_zh.tex 中
定义 def:joint 与 §sec:twtable 的树宽表。本脚本是该表的唯一数值来源。

不是实验脚本：不读任何数据、不用随机数、不训练、不依赖 JAX。只做小图组合
计算，单线程数秒级完成，输出确定性 JSON。

def:joint（与稿中逐字对应）
--------------------------
    V_G = {xi_k}_{k=1..d}  ⊔  {alpha_e}_{e in E(T)}  ⊔  {upsilon_a}_{a in S}
    E_G = (i) 对每个上游 k：{xi_k} ∪ {alpha_e : e in delta_T(k)} 补成团
          (ii) 对每个 a in S、每个 k in P_a：边 {xi_k, upsilon_a}
    G_S^+ = G_S 中额外把 {upsilon_a}_{a in S} 补成团（材料化整张 G^s 表时）

树宽算法
--------
顶点数 <= EXACT_MAX_VERTICES 时用按顶点子集的动态规划给出**精确**树宽
（Held--Karp 型消元序 DP）；否则给出 min-fill 与 min-degree 两种消元启发式
的较小者，即**上界**。JSON 的 "kind" 字段区分二者。

用法
----
    python simple_ttns_l2/experiments/audit_joint_structure_treewidth.py

写出 simple_ttns_l2/reports/joint_structure_treewidth.json 并在 stdout
打印可直接对照论文表格的摘要。
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Graph = Dict[str, Set[str]]

# 顶点数不超过此值时计算精确树宽；超过时只给启发式上界。
EXACT_MAX_VERTICES = 21

OUT_JSON = (
    pathlib.Path(__file__).resolve().parents[1] / "reports" / "joint_structure_treewidth.json"
)


# --------------------------------------------------------------- 图基本操作
def add_clique(g: Graph, nodes: Iterable[str]) -> None:
    nodes = list(nodes)
    for u in nodes:
        g.setdefault(u, set())
    for u, v in itertools.combinations(nodes, 2):
        g[u].add(v)
        g[v].add(u)


def add_edge(g: Graph, u: str, v: str) -> None:
    g.setdefault(u, set()).add(v)
    g.setdefault(v, set()).add(u)


def copy_graph(g: Graph) -> Graph:
    return {u: set(nb) for u, nb in g.items()}


def edge_list(g: Graph) -> List[List[str]]:
    return sorted([sorted((u, v)) for u in g for v in g[u] if u < v])


# --------------------------------------------------------------- 上游树 T
def path_tree(d: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(d - 1)]


def star_tree(d: int) -> List[Tuple[int, int]]:
    return [(0, i) for i in range(1, d)]


def balanced_binary_tree(d: int) -> List[Tuple[int, int]]:
    return [((i - 1) // 2, i) for i in range(1, d)]


TREES = {
    "path": path_tree,
    "star": star_tree,
    "binary": balanced_binary_tree,
}


# --------------------------------------------------------------- 父集合族
def banded_parents(d: int, K: int, fanin: int = 2) -> Dict[int, List[int]]:
    """节点 a 的父集合为上一层中一段连续下标（允许 wrap）。"""
    return {a: sorted(((a + j) % d) for j in range(fanin)) for a in range(K)}


def dense_parents(d: int, K: int, fanin: int = 5, stride: int = 3) -> Dict[int, List[int]]:
    """长程稠密层：连续段父节点 + 一条跨越 stride*(fanin-1) 的长程边。

    完全由 (d, K, fanin, stride) 决定，无随机性。前 fanin-1 个父节点是连续
    段 a, a+1, ..., a+fanin-2（mod d），第 fanin 个父节点是长程边
    a + stride*(fanin-1) （mod d）。用作检验长程 reconvergence 对树宽上界
    影响的对照行。
    """
    out: Dict[int, List[int]] = {}
    for a in range(K):
        pa = {(a + j) % d for j in range(fanin - 1)}
        pa.add((a + stride * (fanin - 1)) % d)
        out[a] = sorted(pa)
    return out


def clustered_crossed_parents(
    clusters: Sequence[int] = (5, 5, 5, 5),
    fanin: int = 4,
    cross_pairs: Sequence[Tuple[int, int]] = ((0, 1), (2, 3)),
    cross_fanin: int = 1,
) -> Tuple[int, int, Dict[int, List[int]]]:
    """复刻 simple_ttns_l2/dag_pipeline.py:build_crossed_spec 的层内结构。

    只读复刻（不 import），因此本脚本不依赖仓库其他模块。上下两层同构、同尺寸，
    节点按簇分块；簇内 fan-in 为 fanin（簇内 wrap），另对 cross_pairs 中每个
    (a, b) 让下层簇 b 的节点额外从上层簇 a 取 cross_fanin 个父节点。

    返回 (d, K, parent_sets)。默认参数给出 d = K = 20、|P_a| <= 5。
    """
    offs: List[int] = []
    off = 0
    for csz in clusters:
        offs.append(off)
        off += csz
    d = off
    parents: Dict[int, Set[int]] = {a: set() for a in range(d)}
    for ci, csz in enumerate(clusters):
        o = offs[ci]
        for j in range(csz):
            node = o + j
            for f in range(min(fanin, csz)):
                parents[node].add(o + (j + f) % csz)
    for (ca, cb) in cross_pairs:
        oa, csa = offs[ca], clusters[ca]
        ob, csb = offs[cb], clusters[cb]
        for j in range(csb):
            node = ob + j
            for f in range(min(cross_fanin, csa)):
                parents[node].add(oa + (j + f) % csa)
    return d, d, {a: sorted(parents[a]) for a in range(d)}


# --------------------------------------------------------------- 联合结构图
def joint_structure_graph(
    d: int,
    tree_edges: Sequence[Tuple[int, int]],
    parent_sets: Dict[int, Sequence[int]],
    S: Sequence[int],
    plus: bool,
) -> Graph:
    """按 def:joint 构造 G_S（plus=True 时为 G_S^+）。"""
    g: Graph = {}
    for k in range(d):
        g.setdefault(f"xi{k}", set())
    inc: Dict[int, List[str]] = {k: [] for k in range(d)}
    for (u, v) in tree_edges:
        name = f"al{min(u, v)}_{max(u, v)}"
        g.setdefault(name, set())
        inc[u].append(name)
        inc[v].append(name)
    # (i) core 团
    for k in range(d):
        add_clique(g, [f"xi{k}"] + inc[k])
    # (ii) edge-CDF 因子
    for a in S:
        g.setdefault(f"up{a}", set())
        for k in parent_sets[a]:
            add_edge(g, f"xi{k}", f"up{a}")
    # G_S^+
    if plus:
        add_clique(g, [f"up{a}" for a in S])
    return g


# --------------------------------------------------------------- 树宽
def treewidth_upper_bound(g: Graph, heuristic: str = "min-fill") -> int:
    """min-fill / min-degree 消元启发式给出的树宽上界。确定性：按顶点名破平。"""
    h = copy_graph(g)
    width = 0
    while h:
        if heuristic == "min-degree":
            v = min(sorted(h), key=lambda x: len(h[x]))
        else:

            def fill(x: str) -> Tuple[int, int]:
                nb = h[x]
                miss = sum(1 for a, b in itertools.combinations(nb, 2) if b not in h[a])
                return (miss, len(nb))

            v = min(sorted(h), key=fill)
        nb = h[v]
        width = max(width, len(nb))
        for a, b in itertools.combinations(nb, 2):
            h[a].add(b)
            h[b].add(a)
        for u in nb:
            h[u].discard(v)
        del h[v]
    return width


def treewidth_exact(g: Graph) -> int:
    """按顶点子集的 DP：tw = min over elimination orders of max fill-in degree。

    状态 S = 已消元顶点集合；消元 S 之后顶点 v 的邻居 =
    {u ∉ S∪{v} : v 与 u 在 G[S∪{v,u}] 中经 S 内顶点连通}。
    """
    verts = sorted(g)
    n = len(verts)
    if n == 0:
        return 0
    idx = {v: i for i, v in enumerate(verts)}
    adj = [0] * n
    for v, nbs in g.items():
        for u in nbs:
            adj[idx[v]] |= 1 << idx[u]

    full = (1 << n) - 1

    def nbr_after(S: int, v: int) -> int:
        seen = 1 << v
        frontier = adj[v] & S
        seen |= frontier
        out = adj[v] & ~S & ~(1 << v)
        while frontier:
            nxt = 0
            m = frontier
            while m:
                b = m & -m
                i = b.bit_length() - 1
                m ^= b
                out |= adj[i] & ~S & ~(1 << v)
                nxt |= adj[i] & S & ~seen
            seen |= nxt
            frontier = nxt
        return out

    @lru_cache(maxsize=None)
    def best(S: int) -> int:
        if S == full:
            return 0
        res = n
        m = full & ~S
        while m:
            b = m & -m
            v = b.bit_length() - 1
            m ^= b
            w = bin(nbr_after(S, v)).count("1")
            if w >= res:
                continue
            res = min(res, max(w, best(S | b)))
        return res

    sys.setrecursionlimit(10000)
    return best(0)


def treewidth(g: Graph) -> Tuple[int, str, str]:
    """返回 (值, "exact"|"upper_bound", 算法名)。"""
    if len(g) <= EXACT_MAX_VERTICES:
        return treewidth_exact(g), "exact", f"subset-DP (|V|={len(g)})"
    mf = treewidth_upper_bound(g, "min-fill")
    md = treewidth_upper_bound(g, "min-degree")
    algo = "min-fill" if mf <= md else "min-degree"
    return min(mf, md), "upper_bound", f"min(min-fill={mf}, min-degree={md}) -> {algo}"


# --------------------------------------------------------------- 配置表
def make_configs() -> List[dict]:
    """论文 §sec:twtable 中每一行的完整配置。顺序即表中顺序。"""
    cfgs: List[dict] = []

    def add(name, tree, d, K, parents, s, note=""):
        S = list(range(s))
        cfgs.append(
            dict(
                topology_name=name,
                upstream_tree=tree,
                d=d,
                K=K,
                parent_sets={str(a): parents[a] for a in sorted(parents)},
                output_subset=S,
                note=note,
            )
        )

    # banded 层：|P_a| = 2，允许 wrap
    for (tree, K) in [("path", 4), ("path", 6), ("path", 7), ("star", 6), ("star", 7)]:
        add(f"banded K={K}", tree, K, K, banded_parents(K, K, fanin=2), K,
            "整表：S = 全部输出")
    add("banded K=6", "path", 6, 6, banded_parents(6, 6, fanin=2), 1, "单点边缘 CDF")
    add("banded K=6", "path", 6, 6, banded_parents(6, 6, fanin=2), 2, "pair CDF")

    # clustered + crossed 层：簇内 fan-in 4、跨簇 fan-in 1，|P_a| <= 5，d = K = 20
    dd, dK, cp = clustered_crossed_parents()
    for tree in ["path", "binary", "star"]:
        add("clustered+crossed (|P_a|<=5)", tree, dd, dK, cp, dK, "整表：S = 全部输出")
    add("clustered+crossed (|P_a|<=5)", "path", dd, dK, cp, 1, "单点边缘 CDF")
    add("clustered+crossed (|P_a|<=5)", "path", dd, dK, cp, 2, "pair CDF")

    # 长程稠密层对照：同样 |P_a| = 5，但含一条跨 stride 的长程边
    dp = dense_parents(20, 20, fanin=5, stride=3)
    add("long-range dense (|P_a|=5)", "path", 20, 20, dp, 20, "长程 reconvergence 对照")
    add("long-range dense (|P_a|=5)", "path", 20, 20, dp, 1, "长程对照，单点边缘")
    add("long-range dense (|P_a|=5)", "path", 20, 20, dp, 2, "长程对照，pair")
    return cfgs


def k_sweep_configs(kmax: int = 32) -> List[dict]:
    """banded + path 上 tw(G_S) 是否随 K 恒定的逐项核对。"""
    out = []
    for K in range(4, kmax + 1):
        out.append(
            dict(
                topology_name=f"banded K={K}",
                upstream_tree="path",
                d=K,
                K=K,
                parent_sets={str(a): p for a, p in banded_parents(K, K, 2).items()},
                output_subset=list(range(K)),
                note="K 扫描",
            )
        )
    return out


# --------------------------------------------------------------- 执行
def evaluate(cfg: dict, record_edges: bool = True) -> dict:
    d = cfg["d"]
    tree_edges = TREES[cfg["upstream_tree"]](d)
    parents = {int(a): p for a, p in cfg["parent_sets"].items()}
    S = cfg["output_subset"]

    g = joint_structure_graph(d, tree_edges, parents, S, plus=False)
    gp = joint_structure_graph(d, tree_edges, parents, S, plus=True)

    tw_g, kind_g, algo_g = treewidth(g)
    tw_gp, kind_gp, algo_gp = treewidth(gp)

    rec = dict(cfg)
    rec["upstream_tree_edges"] = [list(e) for e in tree_edges]
    rec["n_vertices_GS"] = len(g)
    rec["n_vertices_GS_plus"] = len(gp)
    rec["tw_GS"] = tw_g
    rec["tw_GS_kind"] = kind_g
    rec["tw_GS_algorithm"] = algo_g
    rec["tw_GS_plus"] = tw_gp
    rec["tw_GS_plus_kind"] = kind_gp
    rec["tw_GS_plus_algorithm"] = algo_gp
    # 稿中注 rem:twtight 的两侧界
    s = len(S)
    deg_max = max(sum(1 for e in tree_edges if k in e) for k in range(d)) if d > 1 else 0
    rec["deg_T_max"] = deg_max
    rec["bound_lower_s_minus_1"] = s - 1
    rec["bound_upper_degmax_plus_s"] = deg_max + s
    rec["bounds_respected"] = (s - 1) <= tw_gp <= deg_max + s
    if record_edges:
        rec["joint_graph_edges_GS"] = edge_list(g)
        rec["joint_graph_edges_GS_plus"] = edge_list(gp)
    return rec


def main() -> int:
    table_rows = [evaluate(c) for c in make_configs()]
    sweep_rows = [evaluate(c, record_edges=False) for c in k_sweep_configs(32)]

    print(f"{'层结构':<22}{'T':<8}{'d':>4}{'s':>4}"
          f"{'tw(G_S)':>10}{'tw(G_S+)':>11}  kind")
    print("-" * 78)
    for r in table_rows:
        k = "exact" if r["tw_GS_kind"] == "exact" else "upper"
        print(f"{r['topology_name']:<22}{r['upstream_tree']:<8}{r['d']:>4}"
              f"{len(r['output_subset']):>4}{r['tw_GS']:>10}{r['tw_GS_plus']:>11}  {k}")

    print()
    tws = sorted({r["tw_GS"] for r in sweep_rows})
    twps = {r["K"]: r["tw_GS_plus"] for r in sweep_rows}
    exact_k = [r["K"] for r in sweep_rows if r["tw_GS_kind"] == "exact"]
    print(f"K 扫描 (banded + path, K=4..32): tw(G_S) 取值集合 = {tws}")
    print(f"  其中精确计算的 K = {exact_k}，其余为启发式上界")
    print(f"  tw(G_S^+) == K 是否对所有 K 成立: "
          f"{all(v == k for k, v in twps.items())}")
    print(f"  tw(G_S^+) 取值: {[twps[k] for k in sorted(twps)]}")
    bad = [r["topology_name"] for r in table_rows + sweep_rows
           if not r["bounds_respected"]]
    print(f"  rem:twtight 两侧界被违反的配置: {bad if bad else '无'}")

    payload = dict(
        description="联合结构图 G_S / G_S^+ 的树宽审计；对应论文 §sec:twtable",
        definition="def:joint in output/pdf/ttns_maxplus_layer_propagation_theory_zh.tex",
        exact_max_vertices=EXACT_MAX_VERTICES,
        table_rows=table_rows,
        k_sweep=sweep_rows,
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
