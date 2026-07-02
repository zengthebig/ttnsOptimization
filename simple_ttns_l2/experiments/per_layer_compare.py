"""逐层联合分布对比：分层 TTNS 森林  vs  全局 TTDE 边缘化。

思想(无 oracle, 完全公平)：两个模型都给出每层 6 维联合 p(L_i)。
- 分层森林：第 i 层的森林 = 该层联合模型(纯数据, 层内 Chow-Liu)。
- 全局 TTDE：24 维平方 TT 拟合后, 用 Gram 把非本层维度积掉 → 本层的解析边缘密度。
对每层比：联合对数似然(密度) + 层内相关 corr_fro + 边缘 W1(采样)。

TTDE 平方 TT 的连续块边缘密度：
  p(x_{a..b}) = <M_block, E_b> / Z,
  M_block 由左环境 lenv[a] 出发, 逐维做 A^T M A(A=Σ_i b_i(x)G[:,i,:]),
  E_b=env[b] 为右环境, Z 为全局 ∫q^2。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from scipy.stats import wasserstein_distance  # noqa: E402

from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, forest_log_density, sample_forest  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402
from simple_ttns_l2.experiments.ttde_tt_baseline import fit_ttde_tt, ttde_logp, sample_ttde_tt  # noqa: E402


def build_ttde_envs(params, bases):
    first = np.asarray(params["tt"]["tt"].first[0])
    inner = np.asarray(params["tt"]["tt"].inner[0])
    last = np.asarray(params["tt"]["tt"].last[0])
    d = int(bases.knots.shape[0])
    gram = np.asarray(jax.vmap(type(bases).l2_integral)(bases))
    cores = [first] + [inner[k] for k in range(inner.shape[0])] + [last]
    # 右环境 env[t]: 积掉 t+1..d-1
    env = [None] * d
    env[d - 1] = np.ones((1, 1))
    for t in range(d - 2, -1, -1):
        G = cores[t + 1]
        env[t] = np.einsum("ij,ais,sl,bjl->ab", gram[t + 1], G, env[t + 1], G)
    # 左环境 lenv[t]: 积掉 0..t-1  (shape [rL_t, rL_t])
    lenv = [None] * (d + 1)
    lenv[0] = np.ones((1, 1))
    for t in range(1, d + 1):
        G = cores[t - 1]
        lenv[t] = np.einsum("ac,aib,ij,cjd->bd", lenv[t - 1], G, gram[t - 1], G)
    Z = float(lenv[d][0, 0])
    return cores, gram, env, lenv, Z


def ttde_block_logp(cores, gram, env, lenv, Z, bases, block, X):
    """连续块 block=[a..b] 的边缘对数似然, X 为 [n, len(block)] 对应这些维度。"""
    a, b = block[0], block[-1]
    base_dims = [jax.tree_util.tree_map(lambda arr, t=t: arr[t], bases) for t in block]
    basis_call = type(bases).__call__
    n = X.shape[0]
    M = np.broadcast_to(lenv[a], (n,) + lenv[a].shape).copy()
    for pos, k in enumerate(block):
        bt = np.asarray(vmap(lambda y, bd=base_dims[pos]: basis_call(bd, y))(jnp.asarray(X[:, pos])))  # [n,m]
        A = np.einsum("ni,air->nar", bt, cores[k])           # [n, rL, rR]
        M = np.einsum("nac,nab,ncd->nbd", M, A, A)           # [n, rR, rR]
    p = np.einsum("nbd,bd->n", M, env[b])                     # [n]
    return np.log(np.clip(p, 1e-300, None)) - np.log(max(Z, 1e-300))


def layer_metrics(pred, true):
    d = pred.shape[1]
    w1 = float(np.mean([wasserstein_distance(pred[:, j], true[:, j]) for j in range(d)]))
    cf = float(np.linalg.norm(np.corrcoef(pred.T) - np.corrcoef(true.T)))
    return w1, cf


def main():
    cfg = dict(
        layer_sizes=[6, 6, 6, 6], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
        n_total=50000, q=2, m=24, rank=8, source_mode="bimodal", src_sigma=0.06,
        lr=0.002, steps=1000, batch_sz=512, init_noise=0.01, train_noise=0.001,
        log_every=250, early_stop_patience=8, mi_threshold=0.02,
        ttde_rank=18, ttde_steps=2000, ttde_em_steps=50, ttde_patience=8,
        ttde_grid=200, ttde_n_sample=8000, monitor_val_sz=4000, n_sample=8000, seed=0,
    )
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params_d = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params_d)
    sources = {**sources, **bimodal_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    val_x = train_x[int(0.85 * n_tr):]
    layers = [list(l) for l in spec.layers]

    # ---- 分层森林(每层一个) ----
    t0 = time.perf_counter()
    forests = []
    for li, Lg in enumerate(layers):
        k_l, key = jax.random.split(key)
        forests.append(fit_layer_forest(jnp.asarray(train_x[:, Lg]), Lg, cfg, k_l, label=f"L{li}", mi_threshold=cfg["mi_threshold"]))
    t_forest = time.perf_counter() - t0

    # ---- 全局 TTDE ----
    t0 = time.perf_counter()
    model, tparams, tinfo = fit_ttde_tt(train_x, val_x, cfg, cfg["seed"])
    t_ttde = time.perf_counter() - t0
    cores, gram, env, lenv, Z = build_ttde_envs(tparams, model.bases)

    # sanity: 全维度边缘 == ttde_logp
    chk = ttde_block_logp(cores, gram, env, lenv, Z, model.bases, list(range(xs.shape[1])), test_x[:512])
    ref = ttde_logp(model, tparams, test_x[:512])
    print(f"[sanity] full-marginal vs ttde_logp  max|Δ|={np.max(np.abs(chk-ref)):.2e}", flush=True)

    # 采样(各一次)
    k_s, key = jax.random.split(key)
    ttde_samp = sample_ttde_tt(model, tparams, k_s, cfg["n_sample"], grid_size=cfg["ttde_grid"])

    print("\n==== 逐层联合分布对比 (分层森林 vs 全局TTDE 边缘) ====")
    print("layer |   LL_forest   LL_ttde |  W1_forest  W1_ttde | corr_forest corr_ttde")
    rows = []
    for li, Lg in enumerate(layers):
        ll_f, _ = forest_log_density(forests[li], test_x[:, Lg])
        ll_f = float(np.asarray(ll_f).mean())
        ll_t = float(ttde_block_logp(cores, gram, env, lenv, Z, model.bases, Lg, test_x[:, Lg]).mean())
        k_m, key = jax.random.split(key)
        fs = np.asarray(sample_forest(forests[li], k_m, cfg["n_sample"], grid_size=400))
        true_l = test_x[:cfg["n_sample"], Lg]
        w1_f, cf_f = layer_metrics(fs, true_l)
        w1_t, cf_t = layer_metrics(ttde_samp[:, Lg], true_l)
        rows.append((li, ll_f, ll_t, w1_f, w1_t, cf_f, cf_t))
        print(f"  L{li}  | {ll_f:9.3f} {ll_t:9.3f} | {w1_f:8.4f} {w1_t:8.4f} | {cf_f:9.3f} {cf_t:9.3f}")

    arr = np.array([r[1:] for r in rows])
    print("-" * 74)
    print(f" mean | {arr[:,0].mean():9.3f} {arr[:,1].mean():9.3f} | {arr[:,2].mean():8.4f} {arr[:,3].mean():8.4f} | {arr[:,4].mean():9.3f} {arr[:,5].mean():9.3f}")
    print(f"\n时间: 分层森林 fit={t_forest:.1f}s   TTDE fit={t_ttde:.1f}s")

    np.save(REPO_ROOT / "simple_ttns_l2" / "reports" / "per_layer_compare.npy",
            np.array([r for r in rows], dtype=float))


if __name__ == "__main__":
    main()
