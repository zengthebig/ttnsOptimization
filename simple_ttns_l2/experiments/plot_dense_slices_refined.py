"""dense_dag_r567 的**精细三块切片图**(对标 budget_sweep_slice_refined.png)。

跨簇 DAG(100节点·5层·簇[4,4,4,4,4]·fanin=3·旋转跨簇桥)+ immediate 分块 + R5/R7。
三块:
  Block 1 · 逐层(L1..L4)top-3 方差节点 1D 边缘密度(truth 灰填 + R5/R7 KDE)+ 残差带。
  Block 2 · 显式误差 vs 深度:左 每维密度 gap (ceiling-LL)/K；右 每对相关误差 corr_fro/sqrt(K(K-1))。
  Block 3 · 最强相关对 浅(L1)vs 深(L4):truth/R5/R7 2D 散点,标题带样本 r。
自包含:oracle 上限用 kNN(Kozachenko-Leonenko)熵,块用 immediate 分块。
保存 simple_ttns_l2/reports/dense_dag_r567_slice_refined.png。
用法：env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.plot_dense_slices_refined [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

from simple_ttns_l2.dag_pipeline import build_crossed_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, forest_log_density, sample_forest  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import fit_analytic_chain, fit_sampled_chain, structural_blocks  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG, complex_sources  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
OUT = REPORTS / "dense_dag_r567_slice_refined.png"
N_PLOT = 8000
C_TRUTH, C_R5, C_R7 = "0.45", "#d62728", "#1f77b4"


def knn_entropy_ceiling(X, k=3):
    """Kozachenko-Leonenko 微分熵 → held-out 平均对数密度上限 -H(nats)。"""
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if N <= k + 1:
        return float("nan")
    dist, _ = cKDTree(X).query(X, k=k + 1)
    r = np.maximum(dist[:, -1], 1e-12)
    log_Vd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
    H = digamma(N) - digamma(k) + log_Vd + (d / N) * np.sum(np.log(r))
    return float(-H)


def strongest_pair(mat):
    """相关阵最大 |非对角| 对。返回 (i, j, r)。"""
    C = np.corrcoef(mat.T)
    K = C.shape[0]
    best, bi, bj = -1.0, 0, min(1, K - 1)
    for i in range(K):
        for j in range(i + 1, K):
            if abs(C[i, j]) > best:
                best, bi, bj = abs(C[i, j]), i, j
    return bi, bj, float(C[bi, bj])


def corr_fro(forest, tev, key, n, n_rep=3):
    Ct = np.corrcoef(tev.T) if tev.shape[1] > 1 else np.array([[1.0]])
    fros = []
    for _ in range(n_rep):
        key, kk = jax.random.split(key)
        s = np.asarray(sample_forest(forest, kk, n, grid_size=400))
        Cm = np.corrcoef(s.T) if s.shape[1] > 1 else np.array([[1.0]])
        fros.append(float(np.linalg.norm(Ct - Cm)))
    return float(np.mean(fros))


def fit_and_sample(seed):
    cfg = dict(CFG)
    key = jax.random.PRNGKey(seed)
    spec = build_crossed_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"],
                              cross_pairs=cfg.get("cross_pairs"), cross_fanin=cfg.get("cross_fanin", 1),
                              rotate_cross=cfg.get("rotate_cross", False), wrap=True)
    layers = [list(l) for l in spec.layers]
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_d, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_d, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    L0 = layers[0]

    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f, label="L0",
                               mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))
    k_an, key = jax.random.split(key)
    R5 = fit_analytic_chain(forest0, spec, params, k_an, s_max0, q=cfg["q"], m=cfg["m"],
                            rank=cfg["rank"], n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                            lr=cfg["an_lr"], steps=cfg["an_steps"], init_noise=cfg["init_noise"],
                            log_every=0, block_mode=cfg["block_mode"],
                            grad_clip=cfg.get("an_grad_clip", 1.0),
                            early_stop_patience=cfg.get("an_early_stop_patience", 80))
    k_sp, key = jax.random.split(key)
    R7 = fit_sampled_chain(forest0, spec, params, k_sp, cfg)

    # 大真值样本(平滑直方图/散点) + 每层模型采样
    k_big, key = jax.random.split(key)
    truth_big = np.asarray(sample_joint(spec, sources, kernels, k_big, N_PLOT, clip=(-1e9, 1e9)))
    r5s, r7s = {}, {}
    for li in range(1, len(layers)):
        k1, key = jax.random.split(key)
        r5s[li] = np.asarray(sample_forest(R5[li], k1, N_PLOT, grid_size=400))
        k2, key = jax.random.split(key)
        r7s[li] = np.asarray(sample_forest(R7[li], k2, N_PLOT, grid_size=400))
    return cfg, spec, layers, test_x, truth_big, R5, R7, r5s, r7s, key


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg, spec, layers, test_x, truth_big, R5, R7, r5s, r7s, key = fit_and_sample(args.seed)
    down = list(range(1, len(layers)))
    xl = [f"L{li}" for li in down]

    # oracle 上限(immediate 块) + 误差 vs 深度。corr 误差直接用画图已采好的样本(不额外重采样)。
    gap5, gap7, fro5, fro7 = [], [], [], []
    for li in down:
        tev = test_x[:, layers[li]]; K = len(layers[li])
        blocks = structural_blocks(spec, li, mode=cfg["block_mode"])
        ceil = sum(knn_entropy_ceiling(tev[:, blk]) for blk in blocks)
        ll5 = float(forest_log_density(R5[li], tev)[0].mean())
        ll7 = float(forest_log_density(R7[li], tev)[0].mean())
        gap5.append((ceil - ll5) / K); gap7.append((ceil - ll7) / K)
        denom = np.sqrt(K * (K - 1))
        Ct = np.corrcoef(tev.T)
        fro5.append(float(np.linalg.norm(Ct - np.corrcoef(r5s[li].T))) / denom)
        fro7.append(float(np.linalg.norm(Ct - np.corrcoef(r7s[li].T))) / denom)
        print(f"[L{li}] gap/K R5={gap5[-1]:.4f} R7={gap7[-1]:.4f}  "
              f"corr_err R5={fro5[-1]:.4f} R7={fro7[-1]:.4f}", flush=True)

    # ================= 绘图: 4行(Block1) + 1(Block2) + 2(Block3) =================
    nrow = len(down) + 3
    fig = plt.figure(figsize=(18.0, 4.6 * nrow))
    hr = [1] * len(down) + [1.15, 1.05, 1.05]
    gs = GridSpec(nrow, 6, figure=fig, hspace=0.66, wspace=0.5, height_ratios=hr,
                  top=0.955, bottom=0.03, left=0.055, right=0.985)
    N_NODE = 3
    block_top, res_axes, res_absmax = {}, [], 0.0

    def smooth(a):
        return np.convolve(a, np.array([0.25, 0.5, 0.25]), mode="same")

    for ri, li in enumerate(down):
        tev = test_x[:, layers[li]]
        order = np.argsort(tev.var(axis=0))[::-1][:N_NODE]
        for ci, j in enumerate(order):
            j = int(j)
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[ri, 2 * ci:2 * ci + 2],
                                          height_ratios=[3, 1], hspace=0.08)
            ax = fig.add_subplot(sub[0]); axr = fig.add_subplot(sub[1], sharex=ax)
            if ri == 0 and ci == 0:
                block_top[1] = ax
            gid = layers[li][j]
            truth, r5, r7 = truth_big[:, gid], r5s[li][:, j], r7s[li][:, j]
            lo = float(np.percentile(truth, 0.5)); hi = float(np.percentile(truth, 99.5))
            pad = 0.12 * (hi - lo + 1e-9)
            bins = np.linspace(lo - pad, hi + pad, 80)
            ax.hist(truth, bins=bins, density=True, color=C_TRUTH, alpha=0.55, label="truth")
            ax.hist(r5, bins=bins, density=True, histtype="step", color=C_R5, lw=1.2, alpha=0.5)
            ax.hist(r7, bins=bins, density=True, histtype="step", color=C_R7, lw=1.2, alpha=0.5)
            xs = np.linspace(lo - pad, hi + pad, 400)
            try:
                ax.plot(xs, gaussian_kde(r5)(xs), color=C_R5, lw=1.7, label="R5 analytic")
                ax.plot(xs, gaussian_kde(r7)(xs), color=C_R7, lw=1.7, label="R7 sampled")
            except np.linalg.LinAlgError:
                ax.plot([], [], color=C_R5, lw=1.7, label="R5 analytic")
                ax.plot([], [], color=C_R7, lw=1.7, label="R7 sampled")
            tag = "max-var" if ci == 0 else f"var#{ci + 1}"
            ax.set_title(f"L{li}  node{gid} ({tag})", fontsize=9.5)
            ax.set_ylabel("density", fontsize=9); ax.tick_params(labelbottom=False, labelsize=8)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8, loc="upper right")
            t_d, _ = np.histogram(truth, bins=bins, density=True)
            r5_d, _ = np.histogram(r5, bins=bins, density=True)
            r7_d, _ = np.histogram(r7, bins=bins, density=True)
            centers = 0.5 * (bins[:-1] + bins[1:])
            res5, res7 = smooth(r5_d - t_d), smooth(r7_d - t_d)
            axr.axhline(0, ls="--", color="gray", lw=0.9)
            axr.plot(centers, res5, color=C_R5, lw=1.2); axr.plot(centers, res7, color=C_R7, lw=1.2)
            axr.set_xlim(lo - pad, hi + pad)
            axr.set_xlabel("value", fontsize=8); axr.set_ylabel("err", fontsize=7.5)
            axr.tick_params(labelsize=7)
            res_axes.append(axr)
            res_absmax = max(res_absmax, float(np.max(np.abs(np.concatenate([res5, res7])))))
    for axr in res_axes:
        axr.set_ylim(-1.08 * (res_absmax + 1e-9), 1.08 * (res_absmax + 1e-9))

    # Block 2
    r_b2 = len(down)
    ax_g = fig.add_subplot(gs[r_b2, 0:3]); block_top[2] = ax_g
    ax_g.plot(xl, gap5, "o-", color=C_R5, lw=1.9, label="R5 analytic (tree)")
    ax_g.plot(xl, gap7, "s-", color=C_R7, lw=1.9, label="R7 sampled")
    ax_g.axhline(0, ls="--", color="gray", lw=1.2, label="oracle (gap=0)")
    ax_g.set_title("Per-dim density gap vs layer  (ceiling - joint_LL)/K,  lower=better", fontsize=11)
    ax_g.set_xlabel("layer"); ax_g.set_ylabel("gap per dim (nats)"); ax_g.legend(fontsize=9); ax_g.grid(alpha=0.3)
    ax_c = fig.add_subplot(gs[r_b2, 3:6])
    ax_c.plot(xl, fro5, "o-", color=C_R5, lw=1.9, label="R5 analytic (tree)")
    ax_c.plot(xl, fro7, "s-", color=C_R7, lw=1.9, label="R7 sampled")
    ax_c.axhline(0, ls="--", color="gray", lw=1.2, label="oracle (err=0)")
    ax_c.set_title("Per-pair corr error vs layer  corr_fro/sqrt(K(K-1)),  lower=better", fontsize=11)
    ax_c.set_xlabel("layer"); ax_c.set_ylabel("corr error per pair"); ax_c.legend(fontsize=9); ax_c.grid(alpha=0.3)

    # Block 3: 强相关对 浅(L1) vs 深(L_last)
    for row, li in enumerate((down[0], down[-1])):
        tev = test_x[:, layers[li]]
        i_p, j_p, _ = strongest_pair(tev)
        gx, gy = layers[li][i_p], layers[li][j_p]
        xr = (float(np.percentile(tev[:, i_p], 1)), float(np.percentile(tev[:, i_p], 99)))
        yr = (float(np.percentile(tev[:, j_p], 1)), float(np.percentile(tev[:, j_p], 99)))
        dx, dy = 0.1 * (xr[1] - xr[0] + 1e-9), 0.1 * (yr[1] - yr[0] + 1e-9)
        series = [("truth", truth_big[:, gx], truth_big[:, gy], C_TRUTH),
                  ("R5 analytic", r5s[li][:, i_p], r5s[li][:, j_p], C_R5),
                  ("R7 sampled", r7s[li][:, i_p], r7s[li][:, j_p], C_R7)]
        srow = GridSpecFromSubplotSpec(1, len(series), subplot_spec=gs[r_b2 + 1 + row, 0:6], wspace=0.32)
        for col, (tag, xx, yy, color) in enumerate(series):
            ax = fig.add_subplot(srow[0, col])
            if row == 0 and col == 0:
                block_top[3] = ax
            r = float(np.corrcoef(xx, yy)[0, 1])
            ax.scatter(xx, yy, s=4, alpha=0.09, color=color, edgecolors="none", rasterized=True)
            ax.set_title(f"L{li} ({'shallow' if row == 0 else 'deep'}) {tag}  r={r:+.3f}", fontsize=10.5)
            ax.set_xlabel(f"node{gx}"); ax.set_ylabel(f"node{gy}")
            ax.set_xlim(xr[0] - dx, xr[1] + dx); ax.set_ylim(yr[0] - dy, yr[1] + dy)

    fig.text(0.5, 0.988, "dense_dag_r567 refined slice fit  "
             f"(crossed DAG {spec.n} nodes, {len(layers)} layers, clusters{cfg['clusters']}, "
             f"immediate blocks, R5 vs R7)",
             ha="center", fontsize=16, weight="bold")
    headers = {1: "Block 1 - Per-layer 1D marginal density (L1..L%d x top-3 var nodes) with residual band" % down[-1],
               2: "Block 2 - Explicit error vs depth (density gap / corr error)",
               3: "Block 3 - Strongest-corr pair: shallow L1 vs deep L%d (truth / R5 / R7)" % down[-1]}
    for blk, ax in block_top.items():
        fig.text(0.5, ax.get_position().y1 + 0.008, headers[blk], ha="center", fontsize=12.5, weight="bold")

    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"saved: {OUT}  ({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
