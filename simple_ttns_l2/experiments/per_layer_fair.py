"""受控逐层对比：控制目标函数与参数化, 只变拓扑。

前一版 per_layer_compare 把"L2+线性+树"的森林和"MLE+平方+链"的 TTDE 边缘对比,
混淆了 目标/参数化/拓扑 三个变量, 归因不可靠。

本脚本让每层模型也用 **平方 + MLE**(与 TTDE 一致), 层内用 Chow-Liu 树, 再比每层 LL:
- 若"每层平方MLE树" ≈ 或 > TTDE 边缘 → 说明之前森林落后主要因 L2/线性, 而非拓扑;
- 若仍明显落后 → 拓扑(单父树)确实是短板。
TTDE 边缘 LL 直接复用同种子数据的既有结果(见 per_layer_compare 输出)。
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
from simple_ttns_l2.objective import (  # noqa: E402
    batch_eval_q_ttns, integral_q2_ttns, batch_basis_vectors_from_samples,
)
from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, forest_log_density  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402
from simple_ttns_l2.experiments.fit_deep_dag_vs_tree import _count_params  # noqa: E402


def fit_layer_sqr_mle(train_x, val_x, bases, parent, rank, cfg, key):
    gram = vmap(type(bases).l2_integral)(bases)
    t0 = init_ttns_from_rank1(key, bases, jnp.asarray(train_x), parent, rank, noise=cfg["init_noise"])
    raw = list(t0.cores)

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

    val_bv = batch_basis_vectors_from_samples(bases, jnp.asarray(val_x))
    eval_nll = jax.jit(lambda raw, bv: nll(raw, bv))
    n = train_x.shape[0]
    best, best_raw, bad = float("inf"), raw, 0
    for s in range(1, cfg["sqr_steps"] + 1):
        key, ki, kn = jax.random.split(key, 3)
        idx = np.asarray(jax.random.randint(ki, (cfg["batch_sz"],), 0, n))
        batch = train_x[idx] + np.asarray(jax.random.normal(kn, (cfg["batch_sz"], train_x.shape[1]))) * cfg["train_noise"]
        bv = batch_basis_vectors_from_samples(bases, jnp.asarray(batch))
        raw, state, loss = step(raw, state, bv)
        if s % 250 == 0 or s == cfg["sqr_steps"]:
            vn = float(eval_nll(raw, val_bv))
            if np.isfinite(vn) and vn + 1e-4 < best:
                best, best_raw, bad = vn, raw, 0
            else:
                bad += 1
                if bad >= 8:
                    break
    ttns = TTNSOpt(tuple(best_raw))
    return ttns, _count_params(ttns.cores)


def layer_sqr_ll(ttns, bases, parent, X):
    gram = vmap(type(bases).l2_integral)(bases)
    Z = float(integral_q2_ttns(ttns, gram, parent))
    bv = batch_basis_vectors_from_samples(bases, jnp.asarray(X))
    q = np.asarray(batch_eval_q_ttns(ttns, bv, parent))
    return float(np.mean(2.0 * np.log(np.abs(q) + 1e-30) - np.log(max(Z, 1e-30))))


# TTDE 边缘 LL(来自 per_layer_compare, 同种子同配置)
TTDE_MARG_LL = {0: 4.170, 1: 1.670, 2: 1.710, 3: 1.935}
FOREST_L2_LL = {0: 4.179, 1: 1.381, 2: 0.859, 3: 0.839}


def main():
    cfg = dict(
        layer_sizes=[6, 6, 6, 6], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
        n_total=50000, q=2, m=24, source_mode="bimodal", src_sigma=0.06,
        lr=0.002, batch_sz=512, init_noise=0.01, train_noise=0.001,
        sqr_rank=8, sqr_steps=2000, mi_threshold=0.02, seed=0,
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

    print("\n==== 受控逐层 LL 对比 (越大越好) ====")
    print("layer | forest(L2线性树) | sqrMLE-tree | sqrMLE-chain | TTDE边缘(平方MLE链)")
    t0 = time.perf_counter()
    for li, Lg in enumerate(layers):
        sub_tr, sub_val, sub_te = train_x[:, Lg], val_x[:, Lg], test_x[:, Lg]
        bases = build_bases(jnp.asarray(sub_tr), cfg["q"], cfg["m"])
        # 树拓扑(层内 Chow-Liu)
        parent_tree = [int(p) for p in estimate_chow_liu_tree(sub_tr, n_bins=16, root=0).parent]
        parent_chain = [0] + list(range(0, len(Lg) - 1))
        k1, key = jax.random.split(key)
        m_tree, _ = fit_layer_sqr_mle(sub_tr, sub_val, bases, parent_tree, cfg["sqr_rank"], cfg, k1)
        ll_tree = layer_sqr_ll(m_tree, bases, parent_tree, sub_te)
        k2, key = jax.random.split(key)
        m_chain, _ = fit_layer_sqr_mle(sub_tr, sub_val, bases, parent_chain, cfg["sqr_rank"], cfg, k2)
        ll_chain = layer_sqr_ll(m_chain, bases, parent_chain, sub_te)
        print(f"  L{li}  |     {FOREST_L2_LL[li]:6.3f}       |   {ll_tree:6.3f}    |    {ll_chain:6.3f}    |     {TTDE_MARG_LL[li]:6.3f}")
    print(f"\n用时: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
