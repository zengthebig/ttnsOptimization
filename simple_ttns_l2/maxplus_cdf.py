"""方案 B：max-plus 多层 DAG 的 **CDF 域解析传播**（无采样截断）。

核心恒等式（multi-source max 在 CDF 域是乘积）：
$$F_{m_v}(s)=P\\big(\\max_{u\\in\\mathrm{pa}(v)}(x_u+e_{uv})\\le s\\big)
        =\\mathbb{E}_x\\Big[\\prod_{u}F_{e}(s-x_u)\\Big],$$
其中期望对上层联合密度（上层 TTNS）取，$\\prod_u F_e(s-x_u)$ 是**逐父腿可分离**的函数，
故该期望 = 对上层 TTNS 的**可分离收缩**（每条父腿放一个向量，其余腿放基积分）。
这是对（可能有符号的）密度做**线性积分**，因此精确、无截断——正是方案 A 采样的痛点所在。

加入 node delay：$F_{x_v}(t)=\\mathbb{E}_{d_v}[F_{m_v}(t-d_v)]$（卷积）。

配对联合 CDF 同样可分离：
$$F_{m,vw}(s,t)=\\mathbb{E}_x\\Big[\\prod_{u\\in\\mathrm{pa}(v)}F_e(s-x_u)\\prod_{u'\\in\\mathrm{pa}(w)}F_e(t-x_{u'})\\Big],$$
共享父 $u\\in\\mathrm{pa}(v)\\cap\\mathrm{pa}(w)$ 的腿放 $F_e(s-x_u)F_e(t-x_u)$（仍是单腿函数）。

由 marginal/配对 CDF 经 Hoeffding 公式得协方差：
$$\\mathrm{Cov}(x_v,x_w)=\\iint\\big[F_{vw}(s,t)-F_v(s)F_w(t)\\big]\\,ds\\,dt.$$
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[1]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from ttde.ttns.ttns_opt import TTNSOpt, batch_eval_rank1_ttns  # noqa: E402

from simple_ttns_l2.maxplus_pipeline import DelayParams
from simple_ttns_l2.ttns_sampler import _basis_eval_dim


def _uniform_cdf(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """U(lo,hi) 的 CDF，逐元素。"""
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)


class UpperModel:
    """封装上层 TTNS + bases + parent，预备每维的积分网格与基取值，供 CDF 域收缩。"""

    def __init__(self, ttns: TTNSOpt, bases, parent: Sequence[int], q_grid: int = 400):
        self.ttns = ttns
        self.bases = bases
        self.parent = list(parent)
        self.n_dims = len(self.parent)
        self.basis_dim = int(ttns.cores[0].shape[1]) if ttns.cores[0].ndim >= 2 else None
        self.basis_integrals = np.asarray(vmap(type(bases).integral)(bases))  # [n_dims, basis_dim]
        self.basis_dim = self.basis_integrals.shape[1]
        knots = np.asarray(bases.knots)
        self.xgrids: List[np.ndarray] = []
        self.Bx: List[np.ndarray] = []  # [Q, basis_dim] per dim
        self.dx: List[float] = []
        for d in range(self.n_dims):
            lo, hi = float(knots[d, 0]), float(knots[d, -1])
            g = np.linspace(lo, hi, q_grid)
            self.xgrids.append(g)
            self.Bx.append(np.asarray(_basis_eval_dim(bases, d, jnp.asarray(g))))
            self.dx.append(float((hi - lo) / (q_grid - 1)))
        self.support = (float(knots[:, 0].min()), float(knots[:, -1].max()))

    def proj_single(self, u: int, s_grid: np.ndarray, params: DelayParams) -> np.ndarray:
        r"""每条父腿向量 vec_u(s)[i]=$\int F_e(s-x)b_{u,i}(x)dx$，返回 [S, basis_dim]。"""
        x = self.xgrids[u]
        F = _uniform_cdf(s_grid[:, None] - x[None, :], params.edge_lo, params.edge_hi)  # [S,Q]
        return (F @ self.Bx[u]) * self.dx[u]

    def proj_single_pair(self, u: int, s_grid: np.ndarray, t_grid: np.ndarray, params: DelayParams) -> np.ndarray:
        r"""共享父腿向量 $\int F_e(s-x)F_e(t-x)b_i(x)dx$，返回 [S, T, basis_dim]。"""
        x = self.xgrids[u]
        Fs = _uniform_cdf(s_grid[:, None] - x[None, :], params.edge_lo, params.edge_hi)  # [S,Q]
        Ft = _uniform_cdf(t_grid[:, None] - x[None, :], params.edge_lo, params.edge_hi)  # [T,Q]
        # [S,T,Q] = Fs[:,None,:]*Ft[None,:,:]，再与 Bx[u] 收缩
        STQ = Fs[:, None, :] * Ft[None, :, :]
        return np.einsum("stq,qi->sti", STQ, self.Bx[u]) * self.dx[u]

    def _contract(self, leg_vectors: Dict[int, np.ndarray], batch: int) -> np.ndarray:
        """对上层 TTNS 收缩：未指定的腿用基积分，指定腿用给定向量（已 batch）。

        leg_vectors[k]: [batch, basis_dim]。返回 [batch]。
        """
        V = np.broadcast_to(self.basis_integrals[None, :, :], (batch, self.n_dims, self.basis_dim)).copy()
        for k, vec in leg_vectors.items():
            V[:, k, :] = vec
        return np.asarray(batch_eval_rank1_ttns(self.ttns, jnp.asarray(V), self.parent))


def _delay_convolve_1d(F_m: np.ndarray, s_grid: np.ndarray, params: DelayParams, n_d: int = 64) -> np.ndarray:
    """$F_v(t)=\\mathbb{E}_{d}[F_m(t-d)]$，d~U(node_lo,node_hi)。在 s_grid 上插值平均。"""
    ds = np.linspace(params.node_lo, params.node_hi, n_d)
    out = np.zeros_like(F_m)
    for d in ds:
        out += np.interp(s_grid - d, s_grid, F_m, left=0.0, right=1.0)
    return out / n_d


def marginal_cdf(
    upper: UpperModel, parents: Sequence[int], s_grid: np.ndarray, params: DelayParams, n_d: int = 64
) -> np.ndarray:
    """节点（父为 `parents`）的 marginal CDF $F_v$，在 s_grid 上返回 [S]。"""
    legs = {u: upper.proj_single(u, s_grid, params) for u in parents}
    F_m = upper._contract(legs, batch=len(s_grid))
    F_m = np.clip(F_m, 0.0, 1.0)  # 数值兜底（理论上单调∈[0,1]）
    return _delay_convolve_1d(F_m, s_grid, params, n_d)


def pair_cdf(
    upper: UpperModel,
    parents_v: Sequence[int],
    parents_w: Sequence[int],
    s_grid: np.ndarray,
    t_grid: np.ndarray,
    params: DelayParams,
    n_d: int = 24,
) -> np.ndarray:
    """配对联合 CDF $F_{vw}(s,t)$（含各自独立 node delay 卷积），返回 [S, T]。"""
    S, T = len(s_grid), len(t_grid)
    pv, pw = set(parents_v), set(parents_w)
    shared = pv & pw
    only_v = pv - pw
    only_w = pw - pv

    # 构造 [S*T, basis_dim] 的各腿向量
    legs: Dict[int, np.ndarray] = {}
    ss, tt = np.meshgrid(np.arange(S), np.arange(T), indexing="ij")
    ss, tt = ss.ravel(), tt.ravel()
    for u in only_v:
        vec = upper.proj_single(u, s_grid, params)  # [S, basis]
        legs[u] = vec[ss]
    for u in only_w:
        vec = upper.proj_single(u, t_grid, params)  # [T, basis]
        legs[u] = vec[tt]
    for u in shared:
        vec = upper.proj_single_pair(u, s_grid, t_grid, params)  # [S,T,basis]
        legs[u] = vec.reshape(S * T, -1)

    F_m = upper._contract(legs, batch=S * T).reshape(S, T)
    F_m = np.clip(F_m, 0.0, 1.0)

    # 对 s、t 两个方向分别做独立 node delay 卷积
    F = np.zeros_like(F_m)
    ds = np.linspace(params.node_lo, params.node_hi, n_d)
    # 先沿 s 卷积
    tmp = np.zeros_like(F_m)
    for d in ds:
        idx = np.interp(s_grid - d, s_grid, np.arange(S), left=0, right=S - 1)
        lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, S - 1); fr = idx - lo
        tmp += (1 - fr)[:, None] * F_m[lo, :] + fr[:, None] * F_m[hi, :]
    tmp /= n_d
    # 再沿 t 卷积
    for d in ds:
        idx = np.interp(t_grid - d, t_grid, np.arange(T), left=0, right=T - 1)
        lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, T - 1); fr = idx - lo
        F += (1 - fr)[None, :] * tmp[:, lo] + fr[None, :] * tmp[:, hi]
    F /= n_d
    return F


def moments_from_marginal(F_v: np.ndarray, s_grid: np.ndarray) -> Tuple[float, float]:
    """由 marginal CDF 求 E[X], Var[X]（X>=0，∫_0^∞(1-F)）。"""
    one_minus = np.clip(1.0 - F_v, 0.0, 1.0)
    ex = float(np.trapz(one_minus, s_grid))
    ex2 = float(np.trapz(2.0 * s_grid * one_minus, s_grid))
    return ex, max(ex2 - ex * ex, 1e-12)


def cov_hoeffding(F_vw: np.ndarray, F_v: np.ndarray, F_w: np.ndarray,
                  s_grid: np.ndarray, t_grid: np.ndarray) -> float:
    r"""Hoeffding：$\mathrm{Cov}=\iint[F_{vw}(s,t)-F_v(s)F_w(t)]\,ds\,dt$。"""
    integrand = F_vw - np.outer(F_v, F_w)
    inner = np.trapz(integrand, t_grid, axis=1)
    return float(np.trapz(inner, s_grid))


def propagate_layer_cdf(
    upper: UpperModel,
    spec,
    li: int,
    params: DelayParams,
    s_max: float,
    n_s: int = 120,
    n_s_pair: int = 60,
) -> Dict:
    """对第 li 层所有节点：解析算出各 marginal（E,Var）与相关矩阵。

    返回 {"nodes":[...], "mean":[...], "var":[...], "corr": [[...]]}。
    """
    s_lo = 0.0
    s_grid = np.linspace(s_lo, s_max, n_s)
    cur = list(spec.layers[li])
    # 上层全局 id -> UpperModel 局部腿索引
    col = {node: i for i, node in enumerate(spec.layers[li - 1])}
    parents = {v: [col[u] for u in spec.parents(v)] for v in cur}

    F_marg = {v: marginal_cdf(upper, parents[v], s_grid, params) for v in cur}
    mean = np.array([moments_from_marginal(F_marg[v], s_grid)[0] for v in cur])
    var = np.array([moments_from_marginal(F_marg[v], s_grid)[1] for v in cur])

    # 相关矩阵（配对 CDF 用较粗网格）
    sp = np.linspace(s_lo, s_max, n_s_pair)
    F_marg_p = {v: marginal_cdf(upper, parents[v], sp, params) for v in cur}
    var_p = {v: moments_from_marginal(F_marg_p[v], sp)[1] for v in cur}
    K = len(cur)
    corr = np.eye(K)
    for a in range(K):
        for b in range(a + 1, K):
            va, vb = cur[a], cur[b]
            Fvw = pair_cdf(upper, parents[va], parents[vb], sp, sp, params)
            cov = cov_hoeffding(Fvw, F_marg_p[va], F_marg_p[vb], sp, sp)
            corr[a, b] = corr[b, a] = cov / np.sqrt(var_p[va] * var_p[vb])
    return {"nodes": cur, "mean": mean, "var": var, "corr": corr}
