"""可行性评估：非负核(core=raw^2)+ MLE 的线性 TTNS。

动机：线性 TTNS 保证分层解析传播的闭包，但 L2 目标弱、会出负密度。
若把每个 core 元素重参数化为非负(raw^2)，配非负 B 样条基 → q>=0 恒成立，
于是可直接上 MLE，同时**保持线性**(边缘/条件仍是线性 TT，分层传播不受影响)。

本脚本在复杂数据(24 节点双峰 max-plus DAG)上，对全局链式模型比较：
- 非负核 + MLE（本方案③）
- 参照：TTDE(平方+MLE) 与 global_TT(线性+L2) 的既有结果。
统一误差体系：joint_LL / 边缘 W1 / 相关 Frobenius。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np
import optax
from jax import numpy as jnp, value_and_grad, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from scipy.stats import wasserstein_distance  # noqa: E402
from ttde.ttns.ttns_opt import TTNSOpt  # noqa: E402

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.objective import (  # noqa: E402
    batch_basis_vectors_from_samples, batch_eval_q_ttns, integral_q_ttns,
    normalize_ttns_by_integral,
)
from simple_ttns_l2.ttns_sampler import sample_ttns  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402
from simple_ttns_l2.experiments.fit_deep_dag_vs_tree import _count_params  # noqa: E402


def fit_nonneg_mle(train_x, val_x, bases, parent, rank, cfg, key):
    basis_integrals = vmap(type(bases).integral)(bases)
    t0 = init_ttns_from_rank1(key, bases, jnp.asarray(train_x), parent, rank, noise=0.0)
    raw = [jnp.sqrt(jnp.abs(c) + 1e-4) for c in t0.cores]  # core = raw^2 >= 0

    def build(raw):
        return TTNSOpt(tuple(r ** 2 for r in raw))

    def nll(raw, bv):
        ttns = build(raw)
        q = batch_eval_q_ttns(ttns, bv, parent)
        Z = integral_q_ttns(ttns, basis_integrals, parent)
        return -jnp.mean(jnp.log(jnp.clip(q, 1e-30, None))) + jnp.log(jnp.clip(Z, 1e-30, None))

    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(cfg["lr"]))
    state = opt.init(raw)

    @jax.jit
    def step(raw, state, bv):
        loss, g = value_and_grad(nll)(raw, bv)
        upd, state = opt.update(g, state, raw)
        return optax.apply_updates(raw, upd), state, loss

    val_bv = batch_basis_vectors_from_samples(bases, jnp.asarray(val_x[:4000]))
    eval_nll = jax.jit(lambda raw, bv: nll(raw, bv))
    n = train_x.shape[0]
    best, best_raw, bad = float("inf"), raw, 0
    t0t = time.perf_counter()
    print("\n=== [nonneg_mle global] ===\nstep,train_nll,val_nll,sec", flush=True)
    for s in range(1, cfg["steps"] + 1):
        key, ki, kn = jax.random.split(key, 3)
        idx = np.asarray(jax.random.randint(ki, (cfg["batch_sz"],), 0, n))
        batch = train_x[idx] + np.asarray(jax.random.normal(kn, (cfg["batch_sz"], train_x.shape[1]))) * cfg["train_noise"]
        bv = batch_basis_vectors_from_samples(bases, jnp.asarray(batch))
        raw, state, loss = step(raw, state, bv)
        if s % cfg["log_every"] == 0 or s == cfg["steps"]:
            vn = float(eval_nll(raw, val_bv))
            print(f"{s},{float(loss):.4f},{vn:.4f},{time.perf_counter()-t0t:.1f}", flush=True)
            if np.isfinite(vn) and vn + 1e-4 < best:
                best, best_raw, bad = vn, raw, 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    print(f"early_stop at {s}", flush=True)
                    break
    ttns = build(best_raw)
    ttns, _ = normalize_ttns_by_integral(ttns, basis_integrals, parent)
    n_params = _count_params(ttns.cores)
    return ttns, n_params


def joint_ll(ttns, bases, parent, X):
    basis_integrals = vmap(type(bases).integral)(bases)
    Z = float(integral_q_ttns(ttns, basis_integrals, parent))
    bv = batch_basis_vectors_from_samples(bases, jnp.asarray(X))
    q = np.asarray(batch_eval_q_ttns(ttns, bv, parent))
    finite = q > 0
    return float(np.mean(np.log(q[finite]) - np.log(max(Z, 1e-30)))), float(1 - finite.mean())


def main():
    cfg = dict(
        layer_sizes=[6, 6, 6, 6], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
        n_total=40000, q=2, m=24, rank=18, source_mode="bimodal", src_sigma=0.06,
        lr=2e-3, steps=2000, batch_sz=512, train_noise=1e-3, log_every=250, patience=8,
        n_sample=3000, seed=0,
    )
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **bimodal_sources(spec, cfg)}
    k_d, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_d, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    val_x = train_x[int(0.85 * n_tr):]
    n_dims = xs.shape[1]

    bases = build_bases(jnp.asarray(train_x), cfg["q"], cfg["m"])
    parent = [int(p) for p in estimate_chow_liu_tree(train_x, n_bins=16, root=0).parent]
    print("chow-liu parent:", parent)

    k_f, key = jax.random.split(key)
    ttns, n_params = fit_nonneg_mle(train_x, val_x, bases, parent, cfg["rank"], cfg, k_f)
    ll, nonpos = joint_ll(ttns, bases, parent, test_x)

    k_s, key = jax.random.split(key)
    samp = np.asarray(sample_ttns(ttns, bases, parent, k_s, cfg["n_sample"], grid_size=200))
    gt = test_x[:cfg["n_sample"]]
    w1 = float(np.mean([wasserstein_distance(samp[:, j], gt[:, j]) for j in range(n_dims)]))
    cf = float(np.linalg.norm(np.corrcoef(samp.T) - np.corrcoef(gt.T)))

    print("\n==== 非负核 + MLE（全局 Chow-Liu 树）评估 ====")
    print(f"params={n_params}  joint_LL={ll:.4f}  nonpos_rate={nonpos:.4f}  W1={w1:.4f}  corr_fro={cf:.4f}")
    print("\n参照(同复杂数据, chain 拓扑, 见 global_vs_layered_complex_metrics.json):")
    print("  global_TT (线性+L2)  params=191520  joint_LL=1.17   W1=0.0434  corr_fro=7.468")
    print("  ttde_TT_mle(平方+MLE) params=171936  joint_LL=15.82  W1=0.0051  corr_fro=2.974")


if __name__ == "__main__":
    main()
