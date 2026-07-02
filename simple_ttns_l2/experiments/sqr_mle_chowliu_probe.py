"""快测：平方 + MLE + Chow-Liu 树 的全局 TTNS（= TTNS 版 TTDE）。

密度 p(x) = q(x)^2 / Z, 其中 q(x)=<T, ⊗ b_k(x_k)> 为 Chow-Liu 树结构的线性 TTNS,
Z = ∫ q^2 = 二次型(Gram 收缩), 解析可算。核不受符号约束(满表达力)。
MLE 目标: max E_data[ 2 log|q| - log Z ]。

只报 joint_LL(无需采样, 平方天然非负 → nonpos=0), 以最快速度回答"到底有多强"。
参数量用小 rank 控制。数据/基复用 nonneg_mle_probe。
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

from ttde.ttns.ttns_opt import TTNSOpt  # noqa: E402

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.objective import batch_eval_q_ttns, integral_q2_ttns  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402
from simple_ttns_l2.experiments.fit_deep_dag_vs_tree import _count_params  # noqa: E402
from simple_ttns_l2.objective import batch_basis_vectors_from_samples  # noqa: E402


def fit_sqr_mle(train_x, val_x, bases, parent, rank, cfg, key):
    gram = vmap(type(bases).l2_integral)(bases)
    t0 = init_ttns_from_rank1(key, bases, jnp.asarray(train_x), parent, rank, noise=cfg["init_noise"])
    raw = list(t0.cores)  # 无符号约束

    def nll(raw, bv):
        ttns = TTNSOpt(tuple(raw))
        q = batch_eval_q_ttns(ttns, bv, parent)
        Z = integral_q2_ttns(ttns, gram, parent)
        return -jnp.mean(2.0 * jnp.log(jnp.abs(q) + 1e-30)) + jnp.log(jnp.clip(Z, 1e-30, None))

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
    print("\n=== [sqr_mle chow-liu] ===\nstep,train_nll,val_nll,sec", flush=True)
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
    ttns = TTNSOpt(tuple(best_raw))
    return ttns, _count_params(ttns.cores)


def joint_ll_sqr(ttns, bases, parent, X):
    gram = vmap(type(bases).l2_integral)(bases)
    Z = float(integral_q2_ttns(ttns, gram, parent))
    bv = batch_basis_vectors_from_samples(bases, jnp.asarray(X))
    q = np.asarray(batch_eval_q_ttns(ttns, bv, parent))
    return float(np.mean(2.0 * np.log(np.abs(q) + 1e-30) - np.log(max(Z, 1e-30))))


def main():
    cfg = dict(
        layer_sizes=[6, 6, 6, 6], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
        n_total=40000, q=2, m=24, rank=6, source_mode="bimodal", src_sigma=0.06,
        lr=2e-3, steps=2000, batch_sz=512, train_noise=1e-3, init_noise=1e-2,
        log_every=250, patience=8, seed=0,
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
    ttns, n_params = fit_sqr_mle(train_x, val_x, bases, parent, cfg["rank"], cfg, k_f)
    ll = joint_ll_sqr(ttns, bases, parent, test_x)

    print("\n==== 平方 + MLE（全局 Chow-Liu 树, rank=%d）评估 ====" % cfg["rank"])
    print(f"params={n_params}  joint_LL={ll:.4f}  nonpos_rate=0.0000")
    print("\n参照(同复杂数据):")
    print("  非负核+MLE   chain      params=171936   joint_LL=9.10")
    print("  非负核+MLE   chow-liu   params=5540400  joint_LL=13.03")
    print("  ttde_TT      chain(平方) params=171936   joint_LL=15.82")


if __name__ == "__main__":
    main()
