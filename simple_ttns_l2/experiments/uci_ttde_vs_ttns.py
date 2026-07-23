"""真实 UCI (POWER / GAS) 三方对比：全局 TTDE vs 全局线性 TTNS vs 全局平方 TTNS。

约束（prompt_benchmark.md）：不创建新模型，只用已实现模型；确认数学正确性。

三个模型（同 B-spline 基、同 q/m、同 Chow–Liu 树，仅参数化 / 拓扑不同）：
  1. global_TTDE   : 平方 TT (链)        p=q^2/Z  MLE   — fit_ttde_tt
  2. global_TTNS   : 线性 TTNS (MI 树)   q, L2=∫q²-2E[q], 归一化 ∫q=1 — fit_flat
  3. global_TTNSDE : 平方 TTNS (MI 树)   p=q^2/Z  MLE   — fit_ttde_ttns

公平性：
  - 基 (q,m) 与 knots 全部由同一份 train_x 决定 → 三模型基完全相同
    (create_space_uniform_knots 对 (xs,m,q) 确定性，三模型都用 train_x 建基)。
  - Chow–Liu 树从 train_x 估一次，线性 TTNS 与平方 TTNS 共用。
  - 线性/平方 TTNS 同树同 rank；平方 TT(链) 用 match_params 把 rank 反解到与 TTNS 等参数量。
  - test_LL：平方模型 log_p=2log|q|-logZ (合法归一化对数密度)；线性模型 log(clip(q,1e-12))
    (q 已 ∫=1，q>0 处即合法密度)，并单独报 nonpos_rate (q≤0 占比) —— 这是线性参数化的固有缺陷。

为控时，q/m/rank/steps 为"降低预算"配置（远小于论文 m=256），仅用于三模型横向对比。
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Sequence

import jax
import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ttde.datasets.power import _POWER  # noqa: E402
from ttde.datasets.gas import _GAS  # noqa: E402
from ttde.datasets.hepmass import _HEPMASS  # noqa: E402
from ttde.score.models.opt_for_tree_data import chain_parent  # noqa: E402
from ttde.ttns.ttns_opt import (  # noqa: E402
    TTNSOpt, quadratic_form_ttns,
)

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.train_l2 import build_bases  # noqa: E402
from simple_ttns_l2.objective import (  # noqa: E402
    batch_eval_q_ttns, batch_basis_vectors_from_samples, normalize_ttns_by_integral,
)
from simple_ttns_l2.experiments.fit_layered_vs_flat_tt import fit_flat, flat_joint_loglik  # noqa: E402
from simple_ttns_l2.experiments.fit_diamond_dag_vs_tree import train_tree_l2  # noqa: E402
from simple_ttns_l2.experiments.ttde_tt_baseline import (  # noqa: E402
    fit_ttde_tt, fit_ttde_ttns, ttde_logp,
)
from simple_ttns_l2.experiments.ttde_ttns_vs_tt import (  # noqa: E402
    ttns_params, tt_params, tree_degrees, tt_rank_for_params,
)

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def _tree_take_first_axis(tree, idx):
    return jax.tree_util.tree_map(lambda arr: arr[idx], tree)


def _tree_perm_first_axis(tree, perm):
    perm = np.asarray(perm, dtype=np.int64)
    return jax.tree_util.tree_map(lambda arr: arr[perm], tree)


def _tt_envs_from_cores_and_bases(cores, bases):
    d = len(cores)
    gram = np.asarray(jax.vmap(type(bases).l2_integral)(bases))
    env = [None] * d
    env[d - 1] = np.ones((1, 1))
    for t in range(d - 2, -1, -1):
        G = cores[t + 1]
        env[t] = np.einsum("ij,ais,sl,bjl->ab", gram[t + 1], G, env[t + 1], G)
    lenv = [None] * (d + 1)
    lenv[0] = np.ones((1, 1))
    for t in range(1, d + 1):
        G = cores[t - 1]
        lenv[t] = np.einsum("ac,aib,ij,cjd->bd", lenv[t - 1], G, gram[t - 1], G)
    Z = float(lenv[d][0, 0])
    return gram, env, lenv, Z


# ------------------------------------------------------------------ 数据

def load_uci(name: str, data_dir: Path):
    """返回 (train_x, val_x, test_x) float64 numpy。直接用 _POWER/_GAS/_HEPMASS 拿 test 划分。"""
    root = Path(data_dir)
    if name.lower() == "power":
        d = _POWER(root)
    elif name.lower() == "gas":
        d = _GAS(root)
    elif name.lower() == "hepmass":
        d = _HEPMASS(root)
    else:
        raise ValueError(name)
    trn = np.asarray(d.trn.x, dtype=np.float64)
    val = np.asarray(d.val.x, dtype=np.float64)
    tst = np.asarray(d.tst.x, dtype=np.float64)
    return trn, val, tst


# ------------------------------------------------------------------ 平方 TT 链的 2D 边缘 (任意 i<j)

def _eval_pair_marginal_sq_tt_on_grid(
    cores, gram, lenv, env, Z, bases, dim_i, dim_j, xi_centers, xj_centers,
    chunk_size: int = 128,
) -> np.ndarray:
    """平方 TT(链) p=q²/Z 在 (dim_i, dim_j) 上的 2D 边缘密度网格。

    把所有 ≠ {i,j} 的维用 gram 积掉，i,j 维在网格上取 b(x)b(x)ᵀ。
    cores[k]: [r_k, m, r_{k+1}], r_0=r_d=1。lenv[t]@bond t, env[t]@bond t+1 (env[d-1]=1)。
    """
    d = len(cores)
    i, j = int(dim_i), int(dim_j)
    if i > j:
        i, j = j, i
    base_dims = [jax.tree_util.tree_map(lambda arr, t=t: arr[t], bases) for t in range(d)]
    basis_call = type(bases).__call__
    xi = jnp.asarray(xi_centers, dtype=jnp.float64)
    xj = jnp.asarray(xj_centers, dtype=jnp.float64)
    gx, gy = jnp.meshgrid(xi, xj, indexing="ij")
    pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)  # [G, 2]  (xi, xj)

    def A_at(t, ys):
        """A_t(x) = Σ_p b_p(x) C_t[r_left, p, r_right]  →  [n, r_left, r_right] = G_t(x_t)。"""
        bt = vmap(lambda y, bd=base_dims[t]: basis_call(bd, y))(ys)  # [n, m]
        C = cores[t]                                                 # [r_left, m, r_right]
        return jnp.einsum("np,ipq->niq", bt, C)                      # [n, r_left, r_right]

    def fold_gram(M_left, t_lo, t_hi):
        """从 bond t_lo 的方阵 M 出发，把 dims [t_lo..t_hi-1] 用 gram 积掉，返回 bond t_hi 的方阵。

        G_k = Σ_p b_p(x) C_k[.,p,.]，∫ b_p b_q gram[p,q]。
        ∫ G_kᵀ M G_k dx_k = C_k[.,p,.]ᵀ M C_k[.,q,.] gram[p,q] → bond k+1 方阵。
        """
        M = M_left  # [r_{t_lo}, r_{t_lo}]
        for k in range(t_lo, t_hi):
            C = cores[k]  # [r_k, m, r_{k+1}]
            tmp = jnp.einsum("ab,bpc->apc", M, C)          # [r_k, m, r_{k+1}]
            M = jnp.einsum("apc,pq,aqd->cd", tmp, gram[k], C)  # [r_{k+1}, r_{k+1}]
        return M

    vals_parts = []
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start:start + chunk_size]
        Ai = A_at(i, chunk[:, 0])  # [n, r_i, r_{i+1}]
        Aj = A_at(j, chunk[:, 1])  # [n, r_j, r_{j+1}]
        # 开 i：bond i → bond i+1。M1[n,r_{i+1},r_{i+1}] = A_i[n,.,a]ᵀ lenv[.,.] A_i[n,.,b]
        L = lenv[i].astype(Ai.dtype)
        M = jnp.einsum("npa,pq,nqb->nab", Ai, L, Ai)
        # 中：bond i+1 → bond j，fold gram over [i+1, j)  → [n, r_j, r_j]
        M = vmap(lambda m: fold_gram(m, i + 1, j))(M)
        # 开 j：bond j → bond j+1。M3[n,r_{j+1},r_{j+1}] = A_j[n,.,a]ᵀ M[n,.,.] A_j[n,.,b]
        M = jnp.einsum("npa,npq,nqb->nab", Aj, M, Aj)
        # 右：env[j] @ bond j+1
        E = env[j].astype(M.dtype)
        p = jnp.einsum("nab,ab->n", M, E)  # [n]
        vals_parts.append(p / Z)

    vals = jnp.concatenate(vals_parts, axis=0)
    return np.asarray(vals).reshape((len(xi_centers), len(xj_centers)))


# ------------------------------------------------------------------ 线性 TTNS 的 2D 边缘

def _eval_pair_marginal_linear_ttns_on_grid(
    ttns, bases, parent, dim_i, dim_j, xi_centers, xj_centers,
) -> np.ndarray:
    """线性 TTNS q (∫q=1) 在 (i,j) 的 2D 边缘：非 {i,j} 维用基积分向量替换。"""
    n_dims = len(parent)
    basis_int = np.asarray(vmap(type(bases).integral)(bases))  # [d, m]
    m = basis_int.shape[1]
    xi = jnp.asarray(xi_centers, dtype=jnp.float64)
    xj = jnp.asarray(xj_centers, dtype=jnp.float64)
    gx, gy = jnp.meshgrid(xi, xj, indexing="ij")
    pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
    base_dims = [jax.tree_util.tree_map(lambda arr, t=t: arr[t], bases) for t in range(n_dims)]
    basis_call = type(bases).__call__

    vals_parts = []
    for start in range(0, pts.shape[0], 256):
        chunk = pts[start:start + 256]
        n = chunk.shape[0]
        V = np.broadcast_to(basis_int[None], (n, n_dims, m)).copy()
        bi = np.asarray(vmap(lambda y: basis_call(base_dims[dim_i], y))(chunk[:, 0]))
        bj = np.asarray(vmap(lambda y: basis_call(base_dims[dim_j], y))(chunk[:, 1]))
        V[:, dim_i, :] = bi
        V[:, dim_j, :] = bj
        q = np.asarray(batch_eval_q_ttns(ttns, jnp.asarray(V), list(parent)))
        vals_parts.append(q)
    vals = np.concatenate(vals_parts, axis=0)
    return np.asarray(vals).reshape((len(xi_centers), len(xj_centers)))


# ------------------------------------------------------------------ 平方 TTNS 的 2D 边缘 (复用 helper 的实现)

def _eval_pair_marginal_sq_ttns_on_grid(
    ttns, parent, bases, gram_matrices, dim_i, dim_j, xi_centers, xj_centers,
    chunk_size: int = 128,
) -> np.ndarray:
    """平方 TTNS p=q²/Z 在 (i,j) 上的 2D 边缘。n_comps=1 → perm=arange，直接用原序。"""
    xi = jnp.asarray(xi_centers, dtype=jnp.float64)
    xj = jnp.asarray(xj_centers, dtype=jnp.float64)
    gx, gy = jnp.meshgrid(xi, xj, indexing="ij")
    pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

    base_mats = jnp.asarray(gram_matrices, dtype=jnp.float64)
    basis_call = type(bases).__call__
    basis_i = jax.tree_util.tree_map(lambda arr: arr[dim_i], bases)
    basis_j = jax.tree_util.tree_map(lambda arr: arr[dim_j], bases)
    z = quadratic_form_ttns(ttns, base_mats, parent)
    quad = jax.jit(lambda mats_batch: jax.vmap(lambda mb: quadratic_form_ttns(ttns, mb, parent))(mats_batch))

    vals_parts = []
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start:start + chunk_size]
        vi = jax.vmap(lambda point: basis_call(basis_i, point[0]))(chunk)
        vj = jax.vmap(lambda point: basis_call(basis_j, point[1]))(chunk)
        mats_batch = jnp.broadcast_to(base_mats, (chunk.shape[0],) + base_mats.shape)
        mats_batch = mats_batch.at[:, dim_i].set(jax.vmap(jnp.outer)(vi, vi))
        mats_batch = mats_batch.at[:, dim_j].set(jax.vmap(jnp.outer)(vj, vj))
        vals_parts.append(quad(mats_batch) / z)
    vals = jnp.concatenate(vals_parts, axis=0)
    return np.asarray(vals).reshape((len(xi_centers), len(xj_centers)))


def _eval_pair_marginal_sq_tt_mixture_on_grid(
    params, model, dim_i, dim_j, xi_centers, xj_centers, chunk_size: int = 128,
) -> np.ndarray:
    """平方 TT mixture 在原始维度 (dim_i, dim_j) 上的 2D 边缘。

    TTDE mixture 每个分量有自己的变量排列 perm：core 位置 k 消费原始变量 perm[k]。
    单分量边缘需要在 permuted 坐标里找 dim_i/dim_j 的位置，并用 permuted bases/Gram
    保持 core 位置与基函数维度一致。整体密度为 Σ_c ψ_c² / Σ_c Z_c。
    """
    tt = params["tt"]["tt"]
    first = np.asarray(tt.first)    # [C, 1, m, r]
    inner = np.asarray(tt.inner)    # [C, d-2, r, m, r]
    last = np.asarray(tt.last)      # [C, r, m, 1]
    perms = np.asarray(model.permutations)
    n_comp = first.shape[0]
    weighted, Zs = [], []
    for c in range(n_comp):
        perm = perms[c]
        pos_i = int(np.where(perm == dim_i)[0][0])
        pos_j = int(np.where(perm == dim_j)[0][0])
        bases_c = _tree_perm_first_axis(model.bases, perm)
        cores = [first[c]] + [inner[c, k] for k in range(inner.shape[1])] + [last[c]]
        gram_c, env_c, lenv_c, Z_c = _tt_envs_from_cores_and_bases(cores, bases_c)
        if pos_i < pos_j:
            d_c = _eval_pair_marginal_sq_tt_on_grid(
                cores, gram_c, lenv_c, env_c, Z_c, bases_c,
                pos_i, pos_j, xi_centers, xj_centers, chunk_size=chunk_size,
            )
        else:
            # helper 的第一个 grid 对应较小的位置；反序时先按 (j,i) 算，再转回 (i,j)。
            d_c = _eval_pair_marginal_sq_tt_on_grid(
                cores, gram_c, lenv_c, env_c, Z_c, bases_c,
                pos_j, pos_i, xj_centers, xi_centers, chunk_size=chunk_size,
            ).T
        weighted.append(d_c * Z_c)
        Zs.append(Z_c)
    return np.sum(weighted, axis=0) / max(float(np.sum(Zs)), 1e-300)


def _eval_pair_marginal_sq_ttns_mixture_on_grid(
    params, model, parent, dim_i, dim_j, xi_centers, xj_centers, chunk_size: int = 128,
) -> np.ndarray:
    """平方 TTNS mixture 在原始维度 (dim_i, dim_j) 上的 2D 边缘。

    修复后的非链 TTNSDE mixture 强制所有分量用恒等排列，因此每个分量共享同一棵 MI 树。
    整体密度同样为 Σ_c ψ_c² / Σ_c Z_c。
    """
    all_cores = params["ttns"]["ttns"].cores
    n_comp = int(np.asarray(all_cores[0]).shape[0])
    gram = np.asarray(jax.vmap(type(model.bases).l2_integral)(model.bases))
    weighted, Zs = [], []
    for c in range(n_comp):
        ttns_c = TTNSOpt(tuple(np.asarray(core)[c] for core in all_cores))
        Z_c = float(quadratic_form_ttns(ttns_c, jnp.asarray(gram), parent))
        d_c = _eval_pair_marginal_sq_ttns_on_grid(
            ttns_c, parent, model.bases, gram, dim_i, dim_j,
            xi_centers, xj_centers, chunk_size=chunk_size,
        )
        weighted.append(d_c * Z_c)
        Zs.append(Z_c)
    return np.sum(weighted, axis=0) / max(float(np.sum(Zs)), 1e-300)


def ttde_finite_logp(model, params, X, batch_sz: int = 512):
    """平方模型 log_p：返回 (finite_mean_ll, nonpositive_rate)。

    平方密度 p=q²/Z 处处非负，但数值上 q 在稀疏区可下溢 → log_p=-inf。
    fair 口径：对 finite 点取平均，并报告 -inf(非正) 占比 —— 与线性模型一致。
    """
    lp = ttde_logp(model, params, X, batch_sz)
    finite = np.isfinite(lp)
    mean_ll = float(lp[finite].mean()) if finite.any() else float("nan")
    nonpos = float((~finite).mean())
    return mean_ll, nonpos


# ------------------------------------------------------------------ 主流程

def _pick_slice_pairs(mi_tree_parent, train_x, n_pairs: int = 3) -> list[tuple[int, int]]:
    """选 n_pairs 个切片对：优先 Chow–Liu 树边（最强依赖），不足补 |corr| 最大对。"""
    d = train_x.shape[1]
    parent = list(mi_tree_parent)
    edges = []
    for v in range(d):
        if v != parent[v]:
            edges.append((min(v, parent[v]), max(v, parent[v])))
    # 按互信息排序（用 |corr| 近似排序，避免重算 MI）
    C = np.corrcoef(train_x.T)
    edges.sort(key=lambda e: -abs(C[e[0], e[1]]))
    pairs = list(edges[:n_pairs])
    if len(pairs) < n_pairs:
        iu = np.triu_indices(d, 1)
        order = np.argsort(-np.abs(C[iu]))
        for k in order:
            e = (int(iu[0][k]), int(iu[1][k]))
            if e not in pairs:
                pairs.append(e)
            if len(pairs) >= n_pairs:
                break
    return pairs[:n_pairs]


def _grid_for_pair(train_x, test_x, dim_i, dim_j, n: int = 60):
    lo = min(float(train_x[:, dim_i].min()), float(test_x[:, dim_i].min()),
            float(train_x[:, dim_j].min()), float(test_x[:, dim_j].min()))
    hi = max(float(train_x[:, dim_i].max()), float(test_x[:, dim_i].max()),
            float(train_x[:, dim_j].max()), float(test_x[:, dim_j].max()))
    pad = 0.03 * (hi - lo)
    xi = np.linspace(lo - pad, hi + pad, n)
    xj = np.linspace(lo - pad, hi + pad, n)
    return xi, xj


def run(name: str, cfg: dict, data_dir: Path):
    print(f"\n{'#' * 70}\n# 数据集: {name}\n{'#' * 70}", flush=True)
    t_load = time.perf_counter()
    trn, val, tst = load_uci(name, data_dir)
    print(f"shapes: train={trn.shape} val={val.shape} test={tst.shape}  "
          f"(load {time.perf_counter()-t_load:.1f}s)", flush=True)

    # 子采样训练集控时（TTDE/线性 TTNS 在大集上慢）
    n_cap = cfg["train_cap"]
    if trn.shape[0] > n_cap:
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(trn.shape[0], n_cap, replace=False)
        trn = trn[idx]
    n_ttde = min(cfg["ttde_n_train"], trn.shape[0])
    tr_fit = trn[:n_ttde]
    tr_val = trn[n_ttde:n_ttde + cfg["monitor_val_sz"]] if trn.shape[0] > n_ttde else val
    if len(tr_val) == 0:
        tr_val = val[:cfg["monitor_val_sz"]]
    print(f"fit_train={tr_fit.shape} monitor_val={tr_val.shape}", flush=True)

    n_dims = trn.shape[1]
    q, m = cfg["q"], cfg["m"]

    # Chow–Liu 树（一次，线性/平方 TTNS 共用）
    mi_tree = [int(p) for p in estimate_chow_liu_tree(trn, n_bins=cfg["cl_bins"], root=0).parent]
    chain = [int(p) for p in chain_parent(n_dims)]
    print(f"n_dims={n_dims}  MI_tree_parent={mi_tree}  deg={tree_degrees(mi_tree)}", flush=True)

    r_ttns = cfg["r_ttns"]
    p_ttns = ttns_params(mi_tree, m, r_ttns)
    r_tt = tt_rank_for_params(p_ttns, n_dims, m) if cfg["match_params"] else (cfg.get("r_tt") or r_ttns)
    p_tt = tt_params(n_dims, m, r_tt)
    nc = cfg.get("n_comps", 1)
    print(f"[params/comp] TTNS(r={r_ttns})={p_ttns}  TT(r={r_tt})={p_tt}  "
          f"(match={cfg['match_params']})  n_comps={nc} → total≈TTNS {p_ttns*nc:,} / TT {p_tt*nc:,}", flush=True)

    bases = build_bases(jnp.asarray(trn), q, m)
    gram = vmap(type(bases).l2_integral)(bases)
    basis_integrals = vmap(type(bases).integral)(bases)

    results = {}  # name -> dict
    tr_j = jnp.asarray(tr_fit)
    val_j = jnp.asarray(tr_val)
    # n_comps>1(mixture) 时线性 TTNS(L2, 无混合)与切片边缘 helper(设 n_comps=1)不适用 → 跳过，聚焦两平方 MLE
    n_comps = cfg.get("n_comps", 1)
    run_linear = n_comps == 1
    run_slices = n_comps == 1

    # ---- 1) global_TTDE : 平方 TT(链) MLE ----
    print("\n=== [1/3] global_TTDE (squared TT chain, MLE) ===", flush=True)
    ttde_cfg = {**cfg, "q": q, "m": m, "ttde_rank": r_tt, "init_noise": cfg["ttde_init_noise"]}
    t0 = time.perf_counter()
    m_tt, p_tt_p, info_tt = fit_ttde_tt(tr_fit, tr_val, ttde_cfg, cfg["seed"])
    dt = time.perf_counter() - t0
    ll_test_tt, np_tt = ttde_finite_logp(m_tt, p_tt_p, tst)
    ll_train_tt, _ = ttde_finite_logp(m_tt, p_tt_p, tr_fit)
    results["global_TTDE"] = dict(ll_test=ll_test_tt, ll_train=ll_train_tt,
                                  params=info_tt["learned_params"], sec=dt,
                                  nonpos_rate=np_tt, rank=r_tt, topology="chain")
    print(f"  test_LL={ll_test_tt:.4f} train_LL={ll_train_tt:.4f} "
          f"params={info_tt['learned_params']} nonpos={np_tt:.3f} sec={dt:.1f}", flush=True)

    # ---- 2) global_TTNS : 线性 TTNS(MI 树) L2 ----（仅 n_comps=1 时）
    if run_linear:
        print("\n=== [2/3] global_TTNS (linear TTNS, MI tree, L2) ===", flush=True)
        t0 = time.perf_counter()
        k_ff = jax.random.PRNGKey(cfg["seed"] + 1)
        lin_ttns, _, r_lin, p_lin = fit_flat(
            mi_tree, "global_TTNS", tr_j, val_j, bases, gram, basis_integrals,
            cfg["budget"], cfg, k_ff)
        dt = time.perf_counter() - t0
        lin_ttns, _ = normalize_ttns_by_integral(lin_ttns, basis_integrals, list(mi_tree))
        # test_LL: log(clip(q))（q 已 ∫=1）+ 非正率
        bv = batch_basis_vectors_from_samples(bases, jnp.asarray(tst))
        q_test = np.asarray(batch_eval_q_ttns(lin_ttns, bv, list(mi_tree)))
        nonpos = float((q_test <= 0).mean())
        ll_test_lin = float(np.log(np.clip(q_test, 1e-12, None)).mean())
        bv_tr = batch_basis_vectors_from_samples(bases, jnp.asarray(tr_fit))
        q_tr = np.asarray(batch_eval_q_ttns(lin_ttns, bv_tr, list(mi_tree)))
        ll_train_lin = float(np.log(np.clip(q_tr, 1e-12, None)).mean())
        results["global_TTNS"] = dict(ll_test=ll_test_lin, ll_train=ll_train_lin,
                                      params=p_lin, sec=dt, nonpos_rate=nonpos,
                                      rank=r_lin, topology="MI_tree")
        print(f"  test_LL={ll_test_lin:.4f} train_LL={ll_train_lin:.4f} "
              f"params={p_lin} nonpos_rate={nonpos:.3f} sec={dt:.1f}", flush=True)
    else:
        print(f"\n=== [2/3] global_TTNS 跳过 (n_comps={n_comps}>1，线性 L2 无混合) ===", flush=True)

    # ---- 3) global_TTNSDE : 平方 TTNS(MI 树) MLE ----
    print("\n=== [3/3] global_TTNSDE (squared TTNS, MI tree, MLE) ===", flush=True)
    init_mode = cfg.get("ttns_init", "canonical")  # canonical(新,EM) | rank1(旧)
    ttnsde_cfg = {**cfg, "q": q, "m": m, "ttde_rank": r_ttns,
                  "init_noise": cfg["ttde_init_noise"], "ttns_init": init_mode}
    t0 = time.perf_counter()
    m_sq, p_sq_p, info_sq = fit_ttde_ttns(tr_fit, tr_val, ttnsde_cfg, cfg["seed"], mi_tree)
    dt = time.perf_counter() - t0
    ll_test_sq, np_sq = ttde_finite_logp(m_sq, p_sq_p, tst)
    ll_train_sq, _ = ttde_finite_logp(m_sq, p_sq_p, tr_fit)
    results["global_TTNSDE"] = dict(ll_test=ll_test_sq, ll_train=ll_train_sq,
                                    params=info_sq["learned_params"], sec=dt,
                                    nonpos_rate=np_sq, rank=r_ttns, topology="MI_tree",
                                    init=init_mode)
    print(f"  [{init_mode}] test_LL={ll_test_sq:.4f} train_LL={ll_train_sq:.4f} "
          f"params={info_sq['learned_params']} nonpos={np_sq:.3f} sec={dt:.1f}", flush=True)

    # ---- 保存参数快照：之后可离线补画图/诊断，不再必须重训 ----
    tag = f"_{cfg.get('out_tag')}" if cfg.get("out_tag") else ""
    param_path = REPORTS / f"uci_{name}_ttde_vs_ttns_params{tag}.pkl"
    with param_path.open("wb") as f:
        pickle.dump(
            dict(
                dataset=name, config=cfg, mi_tree=mi_tree, chain=chain,
                ttde_params=p_tt_p, ttde_bases=m_tt.bases,
                ttde_permutations=np.asarray(m_tt.permutations),
                ttnsde_params=p_sq_p, ttnsde_bases=m_sq.bases,
                ttnsde_tree_parent=np.asarray(m_sq.tree_parent),
            ),
            f,
        )
    print(f"saved params: {param_path}", flush=True)

    print("\n=== 计算切片密度 ===", flush=True)
    pairs = _pick_slice_pairs(mi_tree, trn, n_pairs=cfg["n_slice_pairs"])
    print(f"slice_pairs={pairs}", flush=True)

    grids = []
    dens = {"global_TTDE": [], "global_TTNSDE": []}
    if run_linear:
        dens["global_TTNS"] = []
        from simple_ttns_l2.experiments.per_layer_compare import build_ttde_envs
        cores_tt, gram_tt, env_tt, lenv_tt, Z_tt = build_ttde_envs(p_tt_p, m_tt.bases)
        sq_ttns = TTNSOpt(tuple(c[0] for c in p_sq_p["ttns"]["ttns"].cores))
        sq_parent = list(int(x) for x in np.asarray(m_sq.tree_parent))
        sq_gram = np.asarray(jax.vmap(type(m_sq.bases).l2_integral)(m_sq.bases))
    for (di, dj) in pairs:
        xi, xj = _grid_for_pair(trn, tst, di, dj, n=cfg["grid_n"])
        grids.append((xi, xj, di, dj))
        if run_linear:
            d_tt = _eval_pair_marginal_sq_tt_on_grid(cores_tt, gram_tt, lenv_tt, env_tt, Z_tt,
                                                     m_tt.bases, di, dj, xi, xj)
            d_lin = _eval_pair_marginal_linear_ttns_on_grid(lin_ttns, bases, list(mi_tree), di, dj, xi, xj)
            d_sq = _eval_pair_marginal_sq_ttns_on_grid(sq_ttns, sq_parent, m_sq.bases, sq_gram,
                                                        di, dj, xi, xj)
        else:
            d_tt = _eval_pair_marginal_sq_tt_mixture_on_grid(p_tt_p, m_tt, di, dj, xi, xj)
            d_sq = _eval_pair_marginal_sq_ttns_mixture_on_grid(
                p_sq_p, m_sq, list(mi_tree), di, dj, xi, xj)
        dens["global_TTDE"].append(d_tt)
        dens["global_TTNSDE"].append(d_sq)
        if run_linear:
            dens["global_TTNS"].append(d_lin)
        # 必要条件：2D 边缘在网格上的梯形积分应 ≈ 1（归一化密度的边缘）
        integ = []
        items = [("TTDE", d_tt)]
        if run_linear:
            items.append(("TTNS", d_lin))
        items.append(("TTNSDE", d_sq))
        for nm, Zg in items:
            trap = float(np.trapz(np.trapz(Zg, xj, axis=1), xi, axis=0))
            integ.append(f"{nm}∫={trap:.3f}")
        print(f"  pair({di},{dj}) done  {integ}", flush=True)

    # ---- 正确性 cross-check: 平方 TT 的 (i,i+1) 连续对边缘 vs ttde_block_logp ----
    if run_linear:
        from simple_ttns_l2.experiments.per_layer_compare import ttde_block_logp as _tblp
        a = 0
        b = min(1, n_dims - 1)
        xi_chk = np.linspace(float(trn[:, a].min()), float(trn[:, a].max()), 12)
        xj_chk = np.linspace(float(trn[:, b].min()), float(trn[:, b].max()), 12)
        d_chk = _eval_pair_marginal_sq_tt_on_grid(cores_tt, gram_tt, lenv_tt, env_tt, Z_tt,
                                                  m_tt.bases, a, b, xi_chk, xj_chk)
        # ttde_block_logp 接受 [n, len(block)]，block=[a,b] 连续
        gx, gy = np.meshgrid(xi_chk, xj_chk, indexing="ij")
        pts_chk = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
        ref = np.exp(_tblp(cores_tt, gram_tt, env_tt, lenv_tt, Z_tt, m_tt.bases, [a, b], pts_chk))
        max_err = float(np.max(np.abs(d_chk.reshape(-1) - ref)))
        print(f"[sanity] squared-TT pair({a},{b}) marginal vs ttde_block_logp  max|Δ|={max_err:.2e}", flush=True)

    return dict(dataset=name, n_dims=n_dims, config=cfg,
                mi_tree=mi_tree, chain=chain, pairs=pairs,
                results=results, grids=grids, dens=dens,
                test_x=tst, train_x=trn)


# ------------------------------------------------------------------ 出图

def plot_bars(res, out: Path):
    labels = list(res["results"].keys())
    ll_test = [res["results"][k]["ll_test"] for k in labels]
    ll_train = [res["results"][k]["ll_train"] for k in labels]
    params = [res["results"][k]["params"] for k in labels]
    cmap = {"global_TTDE": "tab:purple", "global_TTNS": "tab:green", "global_TTNSDE": "tab:brown"}
    colors = [cmap.get(k, "tab:gray") for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 0.38
    ax[0].bar(x - w / 2, ll_test, w, label="test_LL", color=colors)
    ax[0].bar(x + w / 2, ll_train, w, label="train_LL", color=colors, alpha=0.45, hatch="//")
    ax[0].set_title("mean log-density (test solid / train hatched)\nhigher test = better")
    ax[0].set_ylabel("log-density")
    for i, v in enumerate(ll_test):
        ax[0].text(x[i] - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[1].bar(x, params, color=colors)
    ax[1].set_yscale("log")
    ax[1].set_title("#parameters (log scale, match_params)")
    for i, v in enumerate(params):
        ax[1].text(x[i], v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, rotation=10, fontsize=8); a.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{res['dataset']} (d={res['n_dims']}): global TTDE vs linear TTNS vs squared TTNS",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_slices(res, out: Path):
    pairs = res["pairs"]
    grids = res["grids"]
    dens = res["dens"]
    test_x = res["test_x"]
    models = [m for m in ("global_TTDE", "global_TTNS", "global_TTNSDE") if m in dens]
    nP = len(pairs)
    fig, axes = plt.subplots(nP, 1 + len(models), figsize=(4.0 * (1 + len(models)), 3.4 * nP), squeeze=False)
    for r, (di, dj) in enumerate(pairs):
        xi, xj, _, _ = grids[r]
        # GT 2D hist
        ax = axes[r][0]
        h, ex, ey = np.histogram2d(test_x[:, di], test_x[:, dj], bins=40,
                                   range=[[xi.min(), xi.max()], [xj.min(), xj.max()]], density=True)
        ax.pcolormesh(ex, ey, h.T, cmap="Blues", shading="auto")
        ax.set_title(f"GT test hist (dims {di},{dj})")
        ax.set_xlabel(f"x[{di}]"); ax.set_ylabel(f"x[{dj}]")
        # models
        for c, name in enumerate(models):
            ax = axes[r][c + 1]
            Z = dens[name][r]
            vmax = max(np.nanmax(Z), 1e-12)
            ax.pcolormesh(xi, xj, Z.T, cmap="Blues", shading="auto",
                          vmin=0, vmax=vmax)
            ax.set_title(name)
            ax.set_xlabel(f"x[{di}]"); ax.set_ylabel(f"x[{dj}]")
    fig.suptitle(f"{res['dataset']}: 2D slice density (GT hist vs models)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_table(res):
    print(f"\n================= {res['dataset']} 三方对比 =================")
    print(f"n_dims={res['n_dims']}  MI树={res['mi_tree']}  deg={tree_degrees(res['mi_tree'])}")
    print(f"切片对={res['pairs']}")
    print(f"{'model':<16}{'test_LL(↑)':>12}{'train_LL':>12}{'params':>12}{'sec':>8}{'nonpos':>9}")
    for name, r in res["results"].items():
        np_str = f"{r.get('nonpos_rate', float('nan')):.3f}" if "nonpos_rate" in r else "—"
        print(f"{name:<16}{r['ll_test']:>12.4f}{r['ll_train']:>12.4f}{r['params']:>12,}{r['sec']:>8.1f}{np_str:>9}")
    # 相对差
    r = res["results"]
    if "global_TTNSDE" in r and "global_TTDE" in r:
        print(f"\nΔtest_LL (TTNSDE - TTDE)  = {r['global_TTNSDE']['ll_test'] - r['global_TTDE']['ll_test']:+.4f}")
    if "global_TTNS" in r:
        print(f"Δtest_LL (TTNSDE - TTNS)  = {r['global_TTNSDE']['ll_test'] - r['global_TTNS']['ll_test']:+.4f}")
        print(f"Δtest_LL (TTNS  - TTDE)  = {r['global_TTNS']['ll_test'] - r['global_TTDE']['ll_test']:+.4f}")
    print("=========================================================\n")


# ------------------------------------------------------------------ CLI

DEFAULT_CFG = dict(
    q=2, m=48, r_ttns=6, match_params=True, r_tt=None, budget=120000, rmax=40,
    lr=2e-3, steps=1200, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
    log_every=300, early_stop_patience=8, cl_bins=16,
    ttde_steps=1200, ttde_em_steps=12, ttde_patience=8, ttde_init_noise=0.03,
    ttde_grad_clip=10.0, monitor_val_sz=2000,
    ttde_n_train=60000, train_cap=80000, n_slice_pairs=3, grid_n=60, seed=0,
)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["power", "gas", "hepmass", "both", "all"], default="both")
    p.add_argument("--data-dir", default="data/data")
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--r-ttns", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--train-cap", type=int, default=None)
    p.add_argument("--n-comps", type=int, default=1, help="mixture 分量数(平方 TT/TTNS 共用);>1 时跳过线性+切片")
    p.add_argument("--no-match", action="store_true",
                   help="关闭 match_params：TT(链)与 TTNS(树)用同 rank(--r-ttns) 做 rank-matched 对比")
    p.add_argument("--out-tag", default=None, help="结果文件后缀，避免覆盖旧结果")
    p.add_argument("--ttns-init", choices=["canonical", "rank1"], default="canonical",
                   help="TTNSDE 非链初始化：canonical(新,EM) | rank1(旧)")
    args = p.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg["ttns_init"] = args.ttns_init
    cfg["n_comps"] = args.n_comps
    cfg["out_tag"] = args.out_tag
    if args.no_match:
        cfg["match_params"] = False
    if args.m is not None: cfg["m"] = args.m
    if args.r_ttns is not None: cfg["r_ttns"] = args.r_ttns
    if args.steps is not None:
        cfg["steps"] = args.steps; cfg["ttde_steps"] = args.steps
    if args.train_cap is not None:
        cfg["train_cap"] = args.train_cap; cfg["ttde_n_train"] = int(0.75 * args.train_cap)

    data_dir = REPO_ROOT / args.data_dir
    if args.dataset == "both":
        datasets = ["power", "gas"]
    elif args.dataset == "all":
        datasets = ["power", "gas", "hepmass"]
    else:
        datasets = [args.dataset]
    tag = f"_{args.out_tag}" if args.out_tag else ""
    REPORTS.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for ds in datasets:
        # 维度越高、MI 树越密 → 默认降 rank 控参数(TTNS 参数 = m·Σ r^deg，hub 度高会爆)
        if args.r_ttns is None:
            cfg["r_ttns"] = {"gas": 4, "hepmass": 3}.get(ds, 6)
        else:
            cfg["r_ttns"] = args.r_ttns
        res = run(ds, cfg, data_dir)
        print_table(res)
        plot_bars(res, REPORTS / f"uci_{ds}_ttde_vs_ttns_bars{tag}.png")
        if res["pairs"]:
            plot_slices(res, REPORTS / f"uci_{ds}_ttde_vs_ttns_slices{tag}.png")
        # 存不含大数组的部分
        dump = {k: v for k, v in res.items() if k not in ("grids", "dens", "test_x", "train_x")}
        all_res[ds] = dump
        print(f"saved: uci_{ds}_ttde_vs_ttns_bars{tag}.png", flush=True)

    out_json = REPORTS / f"uci_ttde_vs_ttns_metrics{tag}.json"
    out_json.write_text(json.dumps(all_res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
