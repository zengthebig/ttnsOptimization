"""多层全解析链验证：解析链(clarify.md) vs A 链(采样传播+重拟合) vs 真值。

- 真值：L0 森林采样 → 逐层真 max-plus 传播(全阶真依赖)。
- A 链：每层从上一层 A 模型采样 → 真 max-plus → 重拟合树 TTNS(误差随层累积)。
- 解析链：每层解析算边缘+树边 → 解析 L2 拟合树 TTNS，用作下一层上层模型，**全程无采样**。

逐层比：joint_LL@独立真值(↑)、corr_fro vs 真值(↓)、最相关对高阶量。看误差是否随层累积、
解析链是否仍胜 A。输出表格 + 折线图。
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, sample_forest, forest_log_density  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import fit_analytic_chain  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402

cfg = dict(
    layer_sizes=[6, 6, 6, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
    n_total=40000, q=2, m=24, rank=8, source_mode="bimodal", src_sigma=0.06,
    lr=0.002, steps=1000, batch_sz=512, init_noise=0.01, train_noise=0.001,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
)
N = 8000
N_EVAL = 8000


def higher(x, i, j):
    xi = x[:, i] - x[:, i].mean(); xj = x[:, j] - x[:, j].mean()
    return (np.corrcoef(np.abs(xi), np.abs(xj))[0, 1],
            np.mean(xi ** 2 * xj) / (x[:, i].std() ** 2 * x[:, j].std() + 1e-12))


def main():
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    n_layers = len(spec.layers)
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **bimodal_sources(spec, cfg)}
    k_d, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_d, cfg["n_total"], clip=(-1e9, 1e9)))
    train_x = xs[: int(0.7 * xs.shape[0])]
    L0 = list(spec.layers[0])

    # L0 数据森林（两条链共享同一 L0）
    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f, label="L0", mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))

    # 真值链：L0 采样 → 逐层真 max-plus（训练 + 评测两套）
    def true_chain(n, seed):
        k = jax.random.PRNGKey(1000 + seed)
        s = np.asarray(sample_forest(forest0, k, n, grid_size=400))
        rng = np.random.default_rng(seed)
        out = {0: s}
        for li in range(1, n_layers):
            s = propagate_layer(spec, li, s, params, rng)
            out[li] = s
        return out
    true_tr = true_chain(N, 1)
    true_ev = true_chain(N_EVAL, 2)

    # A 链：每层从上一层 A 模型采样 → 真 max-plus → 重拟合
    A_forests = {0: forest0}
    k_as, key = jax.random.split(key)
    s_prev = np.asarray(sample_forest(forest0, k_as, N, grid_size=400))
    for li in range(1, n_layers):
        rng = np.random.default_rng(500 + li)
        s_li = propagate_layer(spec, li, s_prev, params, rng)
        k_a, key = jax.random.split(key)
        A_forests[li] = fit_layer_forest(jnp.asarray(s_li), list(spec.layers[li]), cfg, k_a,
                                         label=f"A_L{li}", mi_threshold=cfg["mi_threshold"])
        k_s, key = jax.random.split(key)
        s_prev = np.asarray(sample_forest(A_forests[li], k_s, N, grid_size=400))

    # 解析链
    k_an, key = jax.random.split(key)
    AN_forests = fit_analytic_chain(forest0, spec, params, k_an, s_max0,
                                    q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
                                    n_s=100, n_s_pair=80, lr=3e-3, steps=1500,
                                    init_noise=cfg["init_noise"], log_every=0)

    # 逐层评测
    rows = []
    for li in range(1, n_layers):
        tev = true_ev[li]
        Ct = np.corrcoef(true_tr[li].T)
        llA, _ = forest_log_density(A_forests[li], tev)
        llAN, npos = forest_log_density(AN_forests[li], tev)
        k1, key = jax.random.split(key)
        aS = np.asarray(sample_forest(A_forests[li], k1, N, grid_size=400))
        k2, key = jax.random.split(key)
        anS = np.asarray(sample_forest(AN_forests[li], k2, N, grid_size=400))
        Ca, Can = np.corrcoef(aS.T), np.corrcoef(anS.T)
        iu = np.triu_indices(tev.shape[1], 1)
        a_, b_ = iu[0][np.argmax(np.abs(Ct[iu]))], iu[1][np.argmax(np.abs(Ct[iu]))]
        _, ck_t = higher(true_tr[li], a_, b_)
        _, ck_a = higher(aS, a_, b_)
        _, ck_an = higher(anS, a_, b_)
        rows.append(dict(
            li=li, llA=llA.mean(), llAN=llAN.mean(), npos=npos,
            froA=np.linalg.norm(Ct - Ca), froAN=np.linalg.norm(Ct - Can),
            pair=(a_, b_), corr_t=Ct[a_, b_], corr_a=Ca[a_, b_], corr_an=Can[a_, b_],
            csk_t=ck_t, csk_a=ck_a, csk_an=ck_an,
        ))

    print("\n============= 多层全解析链 vs A 链 vs 真值 =============")
    print(f"{'层':>3} | {'joint_LL(A)':>11} {'joint_LL(解析)':>13} | {'corr_fro(A)':>11} {'corr_fro(解析)':>13} | 最相关对 corr 真/A/解析")
    for r in rows:
        print(f"L{r['li']:>2} | {r['llA']:>11.4f} {r['llAN']:>13.4f} | "
              f"{r['froA']:>11.4f} {r['froAN']:>13.4f} | "
              f"(y{r['pair'][0]},y{r['pair'][1]}) {r['corr_t']:.3f}/{r['corr_a']:.3f}/{r['corr_an']:.3f}")
    print("\n  coskew(最相关对) 真/A/解析:")
    for r in rows:
        print(f"    L{r['li']}: {r['csk_t']:.3f} / {r['csk_a']:.3f} / {r['csk_an']:.3f}  (解析非正率={r['npos']:.3f})")
    print("=======================================================\n")

    # 折线图
    lis = [r["li"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(lis, [r["llA"] for r in rows], "o-", label="A chain")
    ax[0].plot(lis, [r["llAN"] for r in rows], "s-", label="analytic chain")
    ax[0].set_title("joint_LL @truth (higher=better)"); ax[0].set_xlabel("layer"); ax[0].legend()
    ax[1].plot(lis, [r["froA"] for r in rows], "o-", label="A chain")
    ax[1].plot(lis, [r["froAN"] for r in rows], "s-", label="analytic chain")
    ax[1].set_title("corr_fro vs truth (lower=better)"); ax[1].set_xlabel("layer"); ax[1].legend()
    ax[2].plot(lis, [r["corr_t"] for r in rows], "k^-", label="truth")
    ax[2].plot(lis, [r["corr_a"] for r in rows], "o-", label="A chain")
    ax[2].plot(lis, [r["corr_an"] for r in rows], "s-", label="analytic chain")
    ax[2].set_title("top-pair corr (closer to truth=better)"); ax[2].set_xlabel("layer"); ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3); a.set_xticks(lis)
    fig.suptitle("clarify.md analytic chain vs A chain: per-layer quality (error accumulation?)", fontweight="bold")
    fig.tight_layout()
    out = REPO_ROOT / "simple_ttns_l2" / "reports" / "analytic_chain_compare.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print("saved:", out)


if __name__ == "__main__":
    main()
