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
