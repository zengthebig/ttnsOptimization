"""原版 TTDE 的 TT（平方参数化 MLE）基线：拟合 + log 密度评估 + 条件 CDF 采样。

用于把 TTDE TT 纳入与分层/全局模型**一致的误差体系**（joint_LL / 边缘 W1 / 相关 Frobenius）。
- 模型：`ttde.score.experiment_setups.model_setups.PAsTTSqrOpt`（n_comps=1），密度 $p(x)=\tilde q(x)^2/Z$。
- 采样：平方 TT 的条件 CDF 逆采样——已采维用 $b(x)b(x)^\top$、未采维用 Gram $\int bb^\top$、
  当前维在网格上用 $b(y)b(y)^\top$ 得未归一化条件密度，归一化 + 逆 CDF。合法（平方密度恒非负）。

注意：本模块依赖运行方已把 `TTNSDE` 放入 `sys.path`（`import ttde` 解析到 TTNSDE 版）。
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import jax
import numpy as np
import optax
from jax import numpy as jnp, vmap

from ttde.dl_routine import batched_vmap
from ttde.score.experiment_setups import init_setups, model_setups


def _train_mle(model, params, train_x, val_x, cfg: dict, k_iter, label: str) -> Tuple[dict, float]:
    """平方 MLE 训练循环（TT / TTNS 共用）。返回 (best_params, best_val_nll)。"""
    clip = cfg.get("ttde_grad_clip", 0.0)
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(cfg["lr"])) if clip else optax.adam(cfg["lr"])
    opt_state = opt.init(params)
    bs = cfg["batch_sz"]

    def nll(p, xs):
        return -batched_vmap(lambda x: model.apply(p, x, method=model.log_p), bs)(xs).mean()

    @jax.jit
    def step(p, s, batch):
        loss, g = jax.value_and_grad(lambda pp: nll(pp, batch))(p)
        upd, s_new = opt.update(g, s, p)
        p_new = optax.apply_updates(p, upd)
        # 跳过非有限步(尖峰平方密度的近零点会使 -log(q^2) 梯度爆炸): 保留旧参数/状态
        ok = jnp.isfinite(loss) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf))
                                                     for leaf in jax.tree_util.tree_leaves(g)]))
        p_out = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, b, a), p, p_new)
        s_out = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, b, a), s, s_new)
        return p_out, s_out, loss

    eval_nll = jax.jit(lambda p, xs: nll(p, xs))
    val_mon = val_x[: cfg.get("monitor_val_sz", 4000)]
    best_val, best_params, bad = float("inf"), params, 0
    t0 = time.perf_counter()
    print(f"\n=== [{label}] ===\nstep,train_nll,val_nll,sec", flush=True)
    for s in range(1, cfg["ttde_steps"] + 1):
        k_iter, k_idx, k_noise = jax.random.split(k_iter, 3)
        idx = jax.random.randint(k_idx, (bs,), 0, train_x.shape[0])
        batch = train_x[idx] + jax.random.normal(k_noise, (bs, train_x.shape[1])) * cfg["train_noise"]
        params, opt_state, loss = step(params, opt_state, batch)
        if s % cfg["log_every"] == 0 or s == cfg["ttde_steps"]:
            vn = float(eval_nll(params, val_mon))
            print(f"{s},{float(loss):.4f},{vn:.4f},{time.perf_counter()-t0:.1f}", flush=True)
            if np.isfinite(vn) and vn + 1e-4 < best_val:
                best_val, best_params, bad = vn, params, 0
            else:
                bad += 1
                if bad >= cfg.get("ttde_patience", 8):
                    print(f"early_stop at {s}", flush=True)
                    break
    return best_params, best_val


def fit_ttde_tt(train_x, val_x, cfg: dict, seed: int) -> Tuple[object, dict, dict]:
    """拟合 TTDE TT（平方 MLE, 链式拓扑）。返回 (model, params, info)。"""
    train_x = jnp.asarray(train_x, dtype=jnp.float64)
    val_x = jnp.asarray(val_x, dtype=jnp.float64)
    setup = model_setups.PAsTTSqrOpt(q=cfg["q"], m=cfg["m"], rank=cfg["ttde_rank"], n_comps=1)
    init = init_setups.CanonicalRankK(em_steps=cfg.get("ttde_em_steps", 50), noise=cfg["init_noise"])
    key = jax.random.PRNGKey(seed + 1201)
    key, k_model, k_init, k_iter = jax.random.split(key, 4)
    model = setup.create(k_model, train_x)
    params = init(model, k_init, train_x)

    params, best_val = _train_mle(model, params, train_x, val_x, cfg, k_iter, "ttde_tt_mle")
    p = params["tt"]["tt"]
    n_params = int(p.first.size + p.inner.size + p.last.size)
    return model, params, {"learned_params": n_params, "best_val_nll": best_val}


def fit_ttde_ttns(train_x, val_x, cfg: dict, seed: int, tree_parent) -> Tuple[object, dict, dict]:
    """拟合 TTDE TTNS（平方 MLE, **给定树拓扑** `tree_parent`, 例如 chow-liu MI 树）。

    与 `fit_ttde_tt` 同为平方参数化 + MLE, 仅把链式 TT 换成单父 TTNS 树。非链拓扑下
    `init_canonical` 自动回退到 rank-1 初始化(库内已处理)。返回 (model, params, info)。
    """
    train_x = jnp.asarray(train_x, dtype=jnp.float64)
    val_x = jnp.asarray(val_x, dtype=jnp.float64)
    setup = model_setups.PAsTTNSSqrOpt(
        q=cfg["q"], m=cfg["m"], rank=cfg["ttde_rank"], n_comps=1,
        tree_parent=tuple(int(p) for p in tree_parent),
    )
    init = init_setups.CanonicalRankK(em_steps=cfg.get("ttde_em_steps", 50), noise=cfg["init_noise"])
    key = jax.random.PRNGKey(seed + 1201)
    key, k_model, k_init, k_iter = jax.random.split(key, 4)
    model = setup.create(k_model, train_x)
    params = init(model, k_init, train_x)

    params, best_val = _train_mle(model, params, train_x, val_x, cfg, k_iter, "ttde_ttns_mle")
    ttns = params["ttns"]["ttns"]
    n_params = int(sum(int(np.asarray(c).size) for c in ttns.cores))
    return model, params, {"learned_params": n_params, "best_val_nll": best_val}


def ttde_logp(model, params, X, batch_sz: int = 512) -> np.ndarray:
    lp = batched_vmap(lambda x: model.apply(params, x, method=model.log_p), batch_sz)(jnp.asarray(X))
    return np.asarray(lp)


def sample_ttde_tt(model, params, key, n: int, grid_size: int = 200) -> np.ndarray:
    """平方 TT 条件 CDF 采样（标准 TT 环境法），返回 [n, d]（列 = 维度序）。

    密度 $p(x)\propto \psi(x)^2$，$\psi(x)=\prod_k A^{(k)}(x_k)$，$A^{(k)}(x)=\sum_i b_i(x)G^{(k)}[:,i,:]$。
    对维度 $t$，固定 $x_{<t}$ 得左向量 $l$，未来维用 Gram 边缘化得右环境矩阵 $E_t$，
    则条件密度 $p(x_t\mid x_{<t})\propto \sum_{ij} c_{ij}\,b_i(x_t)b_j(x_t)$，$c=P E_t P^\top$，$P=l\,G^{(t)}$。
    CDF 用截断 Gram $\int_{lo}^y bb^\top$。对样本完全向量化。
    """
    first = np.asarray(params["tt"]["tt"].first[0])   # [1, m, r]
    inner = np.asarray(params["tt"]["tt"].inner[0])   # [d-2, r, m, r]
    last = np.asarray(params["tt"]["tt"].last[0])     # [r, m, 1]
    bases = model.bases
    d = int(bases.knots.shape[0])
    basis_call = type(bases).__call__
    gram = np.asarray(jax.vmap(type(bases).l2_integral)(bases))  # [d, m, m]
    knots = np.asarray(bases.knots)
    base_dims = [jax.tree_util.tree_map(lambda a, t=t: a[t], bases) for t in range(d)]

    cores = [first] + [inner[k] for k in range(inner.shape[0])] + [last]  # 每个 [rL, m, rR]

    # 右环境 env[t]：边缘化 dims t+1..d-1 后 dim t 右键的 [rR, rR] 矩阵
    env = [None] * d
    env[d - 1] = np.ones((1, 1))
    for t in range(d - 2, -1, -1):
        G = cores[t + 1]
        env[t] = np.einsum("ij,ais,sl,bjl->ab", gram[t + 1], G, env[t + 1], G)

    grids, PG = [], []
    for t in range(d):
        lo, hi = float(knots[t, 0]), float(knots[t, -1])
        g = jnp.linspace(lo, hi, grid_size)
        grids.append(np.asarray(g))
        PG.append(np.asarray(vmap(lambda y, t=t: base_dims[t].l2_up_to(y))(g)))  # [G, m, m]

    l = np.ones((n, 1))  # 左向量 [n, r_{-1}=1]
    xs = np.zeros((n, d))
    for t in range(d):
        G = cores[t]                                    # [rL, m, rR]
        E = env[t]                                      # [rR, rR]
        P = np.einsum("na,air->nir", l, G)             # [n, m, rR]
        c = np.einsum("nir,rs,njs->nij", P, E, P)      # [n, m, m]
        num = np.einsum("nij,gij->ng", c, PG[t])       # [n, G] 未归一化 CDF
        den = np.einsum("nij,ij->n", c, gram[t])       # [n]
        den = np.where(np.abs(den) < 1e-30, 1.0, den)
        cdf = np.clip(num / den[:, None], 0.0, None)
        cdf = cdf / np.maximum(cdf[:, -1:], 1e-30)
        key, kt = jax.random.split(key)
        u = np.asarray(jax.random.uniform(kt, (n,)))
        gt = grids[t]
        idx = np.clip(np.array([np.searchsorted(cdf[i], u[i]) for i in range(n)]), 1, grid_size - 1)
        c0 = cdf[np.arange(n), idx - 1]
        c1 = cdf[np.arange(n), idx]
        fr = np.where(c1 > c0, (u - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
        xt = gt[idx - 1] + fr * (gt[idx] - gt[idx - 1])
        xs[:, t] = xt
        bt = np.asarray(vmap(lambda y, t=t: basis_call(base_dims[t], y))(jnp.asarray(xt)))  # [n, m]
        A = np.einsum("ni,air->nar", bt, G)            # [n, rL, rR]
        l = np.einsum("na,nar->nr", l, A)              # [n, rR]
    return xs
