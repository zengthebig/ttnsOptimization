"""方案 B（森林感知）：对**每层 TTNS 森林**做 max-plus 的 CDF 域解析传播。

`maxplus_cdf.py` 的 `UpperModel` 只吃单个 TTNS；但本项目里每层是**森林**（按相关性分成
若干互不相关的块，每块一个单父 TTNS）。利用块间独立 → 期望按块因子分解：

$$\mathbb{E}_x\Big[\prod_{u\in\mathrm{pa}(v)}F_e(s-x_u)\Big]
   =\prod_{b}\ \mathbb{E}_{x_b}\Big[\prod_{u\in\mathrm{pa}(v)\cap b}F_e(s-x_u)\Big],$$

每块的期望就是对该块 TTNS 的可分离收缩（复用 `UpperModel.proj_single/_contract`）。
配对联合 CDF 同理逐块因子分解。本模块**不修改采样实现，也不修改 `maxplus_cdf.py`**，
只复用其 `UpperModel` 与公共函数。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.maxplus_pipeline import DelayParams
from simple_ttns_l2.maxplus_cdf import (
    UpperModel,
    _delay_convolve_1d,
    moments_from_marginal,
    cov_hoeffding,
)
from simple_ttns_l2.layered_forest import BlockModel


class UpperForest:
    """上层 = TTNS 森林（多块）；每块封装成一个 `UpperModel`，按全局节点 id 路由父腿。"""

    def __init__(self, forest: List[BlockModel], q_grid: int = 400):
        self.blocks = []
        for bm in forest:
            um = UpperModel(bm.ttns, bm.bases, list(bm.parent), q_grid=q_grid)
            gid2local = {int(g): i for i, g in enumerate(bm.global_vars)}
            self.blocks.append({"um": um, "gid2local": gid2local, "gids": set(int(g) for g in bm.global_vars)})

    def marginal_cdf(self, parents: Sequence[int], s_grid: np.ndarray, params: DelayParams, n_d: int = 64) -> np.ndarray:
        """节点（全局父 id = `parents`）的 marginal CDF，[S]。块间独立 → 各块收缩之积。"""
        S = len(s_grid)
        F_m = np.ones(S)
        pset = set(int(p) for p in parents)
        for blk in self.blocks:
            blk_parents = pset & blk["gids"]
            if not blk_parents:
                continue
            um = blk["um"]
            legs = {blk["gid2local"][u]: um.proj_single(blk["gid2local"][u], s_grid, params) for u in blk_parents}
            F_m *= um._contract(legs, batch=S)
        F_m = np.clip(F_m, 0.0, 1.0)
        return _delay_convolve_1d(F_m, s_grid, params, n_d)

    def pair_cdf(self, parents_v: Sequence[int], parents_w: Sequence[int],
                 s_grid: np.ndarray, t_grid: np.ndarray, params: DelayParams, n_d: int = 24) -> np.ndarray:
        """配对联合 CDF $F_{vw}(s,t)$，[S,T]。逐块因子分解 + 各自 node delay 卷积。"""
        S, T = len(s_grid), len(t_grid)
        pv, pw = set(int(p) for p in parents_v), set(int(p) for p in parents_w)
        ss, tt = np.meshgrid(np.arange(S), np.arange(T), indexing="ij")
        ss, tt = ss.ravel(), tt.ravel()
        F_flat = np.ones(S * T)
        for blk in self.blocks:
            bv, bw = pv & blk["gids"], pw & blk["gids"]
            if not bv and not bw:
                continue
            um = blk["um"]
            g2l = blk["gid2local"]
            only_v, only_w, shared = bv - bw, bw - bv, bv & bw
            legs: Dict[int, np.ndarray] = {}
            for u in only_v:
                legs[g2l[u]] = um.proj_single(g2l[u], s_grid, params)[ss]
            for u in only_w:
                legs[g2l[u]] = um.proj_single(g2l[u], t_grid, params)[tt]
            for u in shared:
                legs[g2l[u]] = um.proj_single_pair(g2l[u], s_grid, t_grid, params).reshape(S * T, -1)
            F_flat *= um._contract(legs, batch=S * T)
        F_m = np.clip(F_flat.reshape(S, T), 0.0, 1.0)
        return _pair_delay_convolve(F_m, s_grid, t_grid, params, n_d)


def _pair_delay_convolve(F_m: np.ndarray, s_grid: np.ndarray, t_grid: np.ndarray,
                         params: DelayParams, n_d: int = 24) -> np.ndarray:
    """对 [S,T] 的配对 CDF 沿 s、t 各做独立 node delay 卷积（与 maxplus_cdf.pair_cdf 一致）。"""
    S, T = F_m.shape
    ds = np.linspace(params.node_lo, params.node_hi, n_d)
    tmp = np.zeros_like(F_m)
    for d in ds:
        idx = np.interp(s_grid - d, s_grid, np.arange(S), left=0, right=S - 1)
        lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, S - 1); fr = idx - lo
        tmp += (1 - fr)[:, None] * F_m[lo, :] + fr[:, None] * F_m[hi, :]
    tmp /= n_d
    F = np.zeros_like(F_m)
    for d in ds:
        idx = np.interp(t_grid - d, t_grid, np.arange(T), left=0, right=T - 1)
        lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, T - 1); fr = idx - lo
        F += (1 - fr)[None, :] * tmp[:, lo] + fr[None, :] * tmp[:, hi]
    return F / n_d


def _inv_cdf(F: np.ndarray, grid: np.ndarray, u: np.ndarray) -> np.ndarray:
    """一维逆 CDF 采样：F[S] 单调∈[0,1]，u[n] → 样本[n]（桶内线性插值）。"""
    S = len(grid)
    Fc = np.maximum.accumulate(np.clip(F, 0.0, 1.0))
    Fc = Fc / (Fc[-1] if Fc[-1] > 0 else 1.0)
    idx = np.clip(np.searchsorted(Fc, u), 1, S - 1)
    f0, f1 = Fc[idx - 1], Fc[idx]
    fr = np.where(f1 > f0, (u - f0) / (f1 - f0), 0.0)
    return grid[idx - 1] + fr * (grid[idx] - grid[idx - 1])


def _cond_sample(G: np.ndarray, grid: np.ndarray, tw: np.ndarray, u: np.ndarray) -> np.ndarray:
    """条件逆 CDF：G[S,T]=P(V<=s|W=t)，给定父值 tw[n]、均匀 u[n] → 子样本[n]。"""
    S, T = G.shape
    pos = np.interp(tw, grid, np.arange(T))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    fr = pos - lo
    g = (1 - fr)[None, :] * G[:, lo] + fr[None, :] * G[:, hi]  # [S, n]
    g = np.maximum.accumulate(np.clip(g, 0.0, 1.0), axis=0)
    last = g[-1:, :]
    g = g / np.where(last <= 0, 1.0, last)
    out = np.empty(len(u))
    for i in range(len(u)):
        col = g[:, i]
        j = int(np.clip(np.searchsorted(col, u[i]), 1, S - 1))
        f0, f1 = col[j - 1], col[j]
        frac = (u[i] - f0) / (f1 - f0) if f1 > f0 else 0.0
        out[i] = grid[j - 1] + frac * (grid[j] - grid[j - 1])
    return out


def _max_spanning_tree(weight: np.ndarray) -> List[int]:
    """对称权重矩阵的最大生成树（Prim），返回 parent[]（根 parent=-1，根=0）。"""
    K = weight.shape[0]
    in_tree = [False] * K
    parent = [-1] * K
    best = [-np.inf] * K
    best[0] = np.inf
    for _ in range(K):
        u = max((k for k in range(K) if not in_tree[k]), key=lambda k: best[k])
        in_tree[u] = True
        for v in range(K):
            if not in_tree[v] and weight[u, v] > best[v]:
                best[v] = weight[u, v]
                parent[v] = u
    parent[0] = -1
    return parent


def _preorder_tree(parent: List[int]) -> List[int]:
    K = len(parent)
    children = [[] for _ in range(K)]
    root = 0
    for v, p in enumerate(parent):
        if p == -1:
            root = v
        else:
            children[p].append(v)
    order, stack = [], [root]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(reversed(children[u]))
    return order


def sample_layer_from_cdf(
    upper: UpperForest, spec, li: int, params: DelayParams, key,
    n: int, s_max: float, n_s: int = 140,
) -> np.ndarray:
    """从方案 B 的解析 CDF 表示中采样第 li 层（合法 CDF → 无 clamp 截断）。

    层内按 B 的相关性建最大生成树（chow-liu），根用 marginal 逆 CDF、子节点用
    配对 CDF 的条件逆 CDF 采样。返回 [n, layer_size]（列序 = spec.layers[li]）。
    """
    import jax as _jax

    cur = list(spec.layers[li])
    parents = {v: list(spec.parents(v)) for v in cur}
    K = len(cur)
    s_grid = np.linspace(0.0, s_max, n_s)

    F = {v: upper.marginal_cdf(parents[v], s_grid, params) for v in cur}

    # 相关矩阵（Hoeffding）→ 最大生成树
    var = {v: max(moments_from_marginal(F[v], s_grid)[1], 1e-12) for v in cur}
    W = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            Fvw = upper.pair_cdf(parents[cur[a]], parents[cur[b]], s_grid, s_grid, params)
            cov = cov_hoeffding(Fvw, F[cur[a]], F[cur[b]], s_grid, s_grid)
            W[a, b] = W[b, a] = abs(cov) / np.sqrt(var[cur[a]] * var[cur[b]])
    tree_parent = _max_spanning_tree(W) if K > 1 else [-1]

    out = np.zeros((n, K))
    for v in _preorder_tree(tree_parent):
        key, k_u = _jax.random.split(key)
        u = np.asarray(_jax.random.uniform(k_u, (n,)))
        pa = tree_parent[v]
        if pa == -1:
            out[:, v] = _inv_cdf(F[cur[v]], s_grid, u)
        else:
            Fvw = upper.pair_cdf(parents[cur[v]], parents[cur[pa]], s_grid, s_grid, params)  # [S(v),T(pa)]
            fw = np.gradient(F[cur[pa]], s_grid)
            dFdt = np.gradient(Fvw, s_grid, axis=1)
            G = dFdt / np.clip(fw[None, :], 1e-9, None)
            out[:, v] = _cond_sample(G, s_grid, out[:, pa], u)
    return out


def sample_layer_copula(
    upper: UpperForest, spec, li: int, params: DelayParams, key,
    n: int, s_max: float, n_s: int = 200, n_s_pair: int = 100,
) -> np.ndarray:
    """从方案 B 的解析分布采样第 li 层，用**高斯 copula**保住全部两两相关 + 精确边缘。

    B 给出每节点 marginal CDF $F_v$ 与**全相关矩阵** $C$（Hoeffding，非树）。
    采样：$z\\sim N(0,C)$，$u_v=\\Phi(z_v)$，$x_v=F_v^{-1}(u_v)$。不丢非树边、无 clamp 截断。
    （相比树采样：树会丢环上非树边；copula 保全 pairwise。）
    """
    import jax
    from scipy.special import ndtr  # 标准正态 CDF

    cur = list(spec.layers[li])
    parents = {v: list(spec.parents(v)) for v in cur}
    K = len(cur)
    s_grid = np.linspace(0.0, s_max, n_s)
    F = [upper.marginal_cdf(parents[v], s_grid, params) for v in cur]

    C = propagate_layer_cdf_forest(upper, spec, li, params, s_max, n_s=n_s, n_s_pair=n_s_pair)["corr"]
    w, V = np.linalg.eigh(C)
    L = V @ np.diag(np.sqrt(np.clip(w, 1e-6, None)))
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    z = (L @ rng.standard_normal((K, n))).T  # [n, K]
    u = ndtr(z)
    return np.stack([_inv_cdf(F[j], s_grid, u[:, j]) for j in range(K)], axis=1)


def propagate_layer_cdf_forest(
    upper: UpperForest, spec, li: int, params: DelayParams,
    s_max: float, n_s: int = 120, n_s_pair: int = 60,
) -> Dict:
    """对第 li 层所有节点：森林解析算出各 marginal（E,Var）与相关矩阵。

    返回 {"nodes":[...], "mean":[...], "var":[...], "corr": [[...]]}。父用全局 id（UpperForest 路由）。
    """
    s_grid = np.linspace(0.0, s_max, n_s)
    cur = list(spec.layers[li])
    parents = {v: list(spec.parents(v)) for v in cur}

    F_marg = {v: upper.marginal_cdf(parents[v], s_grid, params) for v in cur}
    mean = np.array([moments_from_marginal(F_marg[v], s_grid)[0] for v in cur])
    var = np.array([moments_from_marginal(F_marg[v], s_grid)[1] for v in cur])

    sp = np.linspace(0.0, s_max, n_s_pair)
    F_marg_p = {v: upper.marginal_cdf(parents[v], sp, params) for v in cur}
    var_p = {v: moments_from_marginal(F_marg_p[v], sp)[1] for v in cur}
    K = len(cur)
    corr = np.eye(K)
    for a in range(K):
        for b in range(a + 1, K):
            Fvw = upper.pair_cdf(parents[cur[a]], parents[cur[b]], sp, sp, params)
            cov = cov_hoeffding(Fvw, F_marg_p[cur[a]], F_marg_p[cur[b]], sp, sp)
            corr[a, b] = corr[b, a] = cov / np.sqrt(var_p[cur[a]] * var_p[cur[b]])
    return {"nodes": cur, "mean": mean, "var": var, "corr": corr}
