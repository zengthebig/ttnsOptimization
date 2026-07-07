"""切片密度对比图：一眼看出分层解析链(R5) vs 采样链(R7)的拟合效果。

单预算配置(rank=16), 内存友好(n_total=12000, 采样 n=4000)。复用
sampled_vs_analytic_chain 的拟合流程(fit_layer_forest -> fit_analytic_chain(R5)
/ fit_sampled_chain(R7))。只出图, 不改现有实验脚本。

图: 上行 1D 边缘密度(浅层 L1 vs 深层 L4), 下行 L4 最强相关对的 2D 散点
(truth / R5 / R7)。配色: truth 灰, R5 红, R7 蓝。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from simple_ttns_l2.dag_pipeline import build_clustered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, sample_forest  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import fit_analytic_chain, fit_sampled_chain, fit_analytic_chain_joint  # noqa: E402
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"

C_TRUTH = "0.45"      # gray
C_R5 = "#d62728"      # red   (analytic chain)
C_R7 = "#1f77b4"      # blue  (sampled chain)

# 单配置, 内存友好版 CFG_BIG。
CFG = dict(
    n_layers=5, clusters=[2, 3, 4, 5, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
    n_total=12000, n_sample=4000, n_fit=12000, q=2, m=24, rank=16,
    src_sigma=0.03,
    lr=2e-3, steps=500, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
    n_s=100, n_s_pair=80, an_lr=3e-3, an_steps=500,
)


def fit_all(cfg):
    """照抄 sampled_vs_analytic_chain.run 的拟合流程, 返回真值与两条链的每层森林。"""
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    layers = [list(l) for l in spec.layers]
    print(f"[spec] nodes={spec.n} layers={len(layers)} per-layer={len(layers[0])} "
          f"clusters={cfg['clusters']} edges={len(spec.edges)}", flush=True)

    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    L0 = layers[0]

    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))

    t0 = time.perf_counter()
    k_an, key = jax.random.split(key)
    AN = fit_analytic_chain(forest0, spec, params, k_an, s_max0,
                            q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
                            n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                            lr=cfg["an_lr"], steps=cfg["an_steps"],
                            init_noise=cfg["init_noise"], log_every=0)
    print(f"[R5 analytic chain] fit {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    k_sp, key = jax.random.split(key)
    SP = fit_sampled_chain(forest0, spec, params, k_sp, cfg)
    print(f"[R7 sampled chain]  fit {time.perf_counter()-t0:.1f}s", flush=True)

    JN = None
    if cfg.get("with_r6"):
        t0 = time.perf_counter()
        k_jn, key = jax.random.split(key)
        JN = fit_analytic_chain_joint(forest0, spec, params, k_jn, s_max0,
                                      q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
                                      n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                                      n_s_joint=cfg.get("n_s_joint", 44),
                                      lr=cfg["an_lr"], steps=cfg["an_steps"],
                                      init_noise=cfg["init_noise"], log_every=0)
        print(f"[R6 joint chain]    fit {time.perf_counter()-t0:.1f}s", flush=True)

    return spec, layers, test_x, AN, SP, JN, key


def strongest_pair(mat):
    """返回 |corr| 最大的一对局部索引 (i, j)。"""
    C = np.corrcoef(mat.T)
    A = np.abs(C.copy())
    np.fill_diagonal(A, 0.0)
    i, j = np.unravel_index(np.argmax(A), A.shape)
    return int(i), int(j), float(C[i, j])


def main():
    cfg = CFG
    t_all = time.perf_counter()
    spec, layers, test_x, AN, SP, JN, key = fit_all(cfg)
    n = cfg["n_sample"]

    li_shallow, li_deep = 1, 4                      # L1 浅层, L4 最深层
    tev_s = test_x[:, layers[li_shallow]]
    tev_d = test_x[:, layers[li_deep]]

    # 各层各一次采样, 复用给 1D / 2D。
    k1, key = jax.random.split(key)
    k2, key = jax.random.split(key)
    k3, key = jax.random.split(key)
    k4, key = jax.random.split(key)
    r5_s = np.asarray(sample_forest(AN[li_shallow], k1, n, grid_size=400))
    r7_s = np.asarray(sample_forest(SP[li_shallow], k2, n, grid_size=400))
    r5_d = np.asarray(sample_forest(AN[li_deep], k3, n, grid_size=400))
    r7_d = np.asarray(sample_forest(SP[li_deep], k4, n, grid_size=400))

    # 浅层选一个方差最大的维(展开清楚); 深层用最强相关对之一。
    j_s = int(np.argmax(tev_s.var(axis=0)))
    i_d, j_d, corr_true = strongest_pair(tev_d)
    print(f"[pick] shallow L{li_shallow} local#{j_s} (gid={layers[li_shallow][j_s]})", flush=True)
    print(f"[pick] deep L{li_deep} strongest pair local#{i_d},#{j_d} "
          f"(gid={layers[li_deep][i_d]},{layers[li_deep][j_d]}) true|corr|={corr_true:+.3f}", flush=True)

    fig = plt.figure(figsize=(13.5, 8.4))
    gs = GridSpec(2, 6, figure=fig, hspace=0.32, wspace=0.42)

    # ---------- Row 1: 1D marginals ----------
    def plot_1d(ax, truth, r5, r7, gid, tag):
        lo = float(np.percentile(truth, 0.5)); hi = float(np.percentile(truth, 99.5))
        pad = 0.12 * (hi - lo + 1e-9)
        bins = np.linspace(lo - pad, hi + pad, 60)
        ax.hist(truth, bins=bins, density=True, color=C_TRUTH, alpha=0.55, label="truth")
        ax.hist(r5, bins=bins, density=True, histtype="step", color=C_R5, lw=1.9, label="R5 analytic")
        ax.hist(r7, bins=bins, density=True, histtype="step", color=C_R7, lw=1.9, label="R7 sampled")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"{tag}  node{gid}", fontsize=11)
        ax.set_xlabel("value"); ax.set_ylabel("density")

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    plot_1d(ax_a, tev_s[:, j_s], r5_s[:, j_s], r7_s[:, j_s],
            layers[li_shallow][j_s], f"1D marginal | shallow L{li_shallow}")
    plot_1d(ax_b, tev_d[:, i_d], r5_d[:, i_d], r7_d[:, i_d],
            layers[li_deep][i_d], f"1D marginal | deep L{li_deep}")
    ax_a.legend(fontsize=9, loc="upper right")

    # ---------- Row 2: 2D strongest-corr pair in L4 ----------
    gx, gy = layers[li_deep][i_d], layers[li_deep][j_d]

    def plot_2d(ax, xs_, ys_, color, tag):
        r = float(np.corrcoef(xs_, ys_)[0, 1])
        ax.scatter(xs_, ys_, s=5, alpha=0.16, color=color, edgecolors="none", rasterized=True)
        ax.set_title(f"{tag}  r={r:+.3f}", fontsize=11)
        ax.set_xlabel(f"node{gx}"); ax.set_ylabel(f"node{gy}")
        return r

    # 共享坐标范围(用真值 1/99 分位, 避免采样离群拉爆)。
    xr = (float(np.percentile(tev_d[:, i_d], 1)), float(np.percentile(tev_d[:, i_d], 99)))
    yr = (float(np.percentile(tev_d[:, j_d], 1)), float(np.percentile(tev_d[:, j_d], 99)))
    dx = 0.1 * (xr[1] - xr[0] + 1e-9); dy = 0.1 * (yr[1] - yr[0] + 1e-9)

    ax0 = fig.add_subplot(gs[1, 0:2])
    ax1 = fig.add_subplot(gs[1, 2:4])
    ax2 = fig.add_subplot(gs[1, 4:6])
    r_t = plot_2d(ax0, tev_d[:, i_d], tev_d[:, j_d], C_TRUTH, "truth")
    r_5 = plot_2d(ax1, r5_d[:, i_d], r5_d[:, j_d], C_R5, "R5 analytic")
    r_7 = plot_2d(ax2, r7_d[:, i_d], r7_d[:, j_d], C_R7, "R7 sampled")
    for ax in (ax0, ax1, ax2):
        ax.set_xlim(xr[0] - dx, xr[1] + dx); ax.set_ylim(yr[0] - dy, yr[1] + dy)

    fig.suptitle(
        f"Layered chain slice fit  (rank={cfg['rank']}, m={cfg['m']})   "
        f"top: 1D marginals (shallow vs deep)   "
        f"bottom: strongest-corr pair in L{li_deep} (node{gx}-node{gy})",
        fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "budget_sweep_slice_viz.png"
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}", flush=True)
    print(f"[2D corr] truth r={r_t:+.3f}  R5 r={r_5:+.3f}  R7 r={r_7:+.3f}", flush=True)
    print(f"total {time.perf_counter()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
