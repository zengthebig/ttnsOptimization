"""为 R5/R7 nonneg 链生成 marginal / 精细切片图。

输出（marginal，对标 dense_dag_r567_slices.png）：
  dense_dag_r567_r5_nonneg_slices.png
  dense_dag_r567_r7_nonneg_slices.png

输出（精细切片，对标 dense_dag_r567_slice_refined.png）：
  dense_dag_r567_r5_nonneg_slice_refined.png
  dense_dag_r567_r7_nonneg_slice_refined.png
"""
from __future__ import annotations

import argparse
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

from simple_ttns_l2.layered_forest import sample_forest, forest_log_density  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_remedy_tests import (  # noqa: E402
    prepare_data,
    fit_analytic_chain_nonneg,
    fit_layer_forest_nonneg,
    fit_sampled_chain_nonneg,
)

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
OUT_R5 = REPORTS / "dense_dag_r567_r5_nonneg_slices.png"
OUT_R7 = REPORTS / "dense_dag_r567_r7_nonneg_slices.png"
OUT_R5_REF = REPORTS / "dense_dag_r567_r5_nonneg_slice_refined.png"
OUT_R7_REF = REPORTS / "dense_dag_r567_r7_nonneg_slice_refined.png"
N_PLOT = 8000
GRID = 400


def plot_marginal_slices(
    test_x: np.ndarray,
    layers: list,
    samples: dict,
    *,
    model_label: str,
    model_color: str,
    title: str,
    out_path: Path,
) -> None:
    """全节点一维边缘直方图：GT(黑) vs 单条 nonneg 链。"""
    n_plot = min(N_PLOT, test_x.shape[0])
    nL = len(layers)
    nN = max(len(l) for l in layers)
    fig, axes = plt.subplots(nL, nN, figsize=(1.55 * nN, 1.95 * nL), squeeze=False)
    for li in range(nL):
        nodes = layers[li]
        truth_layer = test_x[:n_plot, nodes]
        for j in range(nN):
            ax = axes[li][j]
            if j >= len(nodes):
                ax.axis("off")
                continue
            gtv = truth_layer[:, j]
            lo = float(np.percentile(gtv, 0.5))
            hi = float(np.percentile(gtv, 99.5))
            pad = 0.15 * (hi - lo + 1e-9)
            bins = np.linspace(lo - pad, hi + pad, 70)
            ax.hist(gtv, bins=bins, density=True, histtype="step", color="k", lw=1.9, label="GT")
            ax.hist(
                samples[li][:, j], bins=bins, density=True, histtype="step",
                color=model_color, lw=1.3, label=model_label,
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_yticks([])
            ax.set_title(f"L{li}·n{nodes[j]}", fontsize=7.5)
            if li == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"saved: {out_path}", flush=True)


def _collect_samples(chain: dict, n_plot: int, key):
    samples = {}
    for li in range(len(chain)):
        key, ks = jax.random.split(key)
        samples[li] = np.asarray(sample_forest(chain[li], ks, n_plot, grid_size=GRID))
    return samples, key


def plot_refined_slices(
    test_x: np.ndarray,
    layers: list,
    chain: dict,
    samples: dict,
    *,
    model_label: str,
    model_color: str,
    seed: int,
    out_path: Path,
) -> None:
    """精细切片：L1–L4 top-3 边缘+残差 / LL·corr vs 深度 / 最强相关对散点。"""
    n_plot = min(N_PLOT, test_x.shape[0])
    down = list(range(1, len(layers)))
    if not down:
        return
    nrow = len(down) + 3
    fig = plt.figure(figsize=(18.0, 4.4 * nrow))
    gs = GridSpec(nrow, 6, figure=fig, hspace=0.66, wspace=0.5,
                  height_ratios=[1] * len(down) + [1.1, 1.05, 1.05],
                  top=0.955, bottom=0.035, left=0.055, right=0.985)
    res_axes, res_absmax = [], 0.0

    def smooth(a):
        return np.convolve(a, np.array([0.25, 0.5, 0.25]), mode="same")

    for ri, li in enumerate(down):
        tev = test_x[:n_plot, layers[li]]
        order = np.argsort(tev.var(axis=0))[::-1][:3]
        for ci, j in enumerate(order):
            j = int(j)
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[ri, 2 * ci:2 * ci + 2],
                                          height_ratios=[3, 1], hspace=0.08)
            ax = fig.add_subplot(sub[0])
            axr = fig.add_subplot(sub[1], sharex=ax)
            truth = tev[:, j]
            lo = float(np.percentile(truth, 0.5))
            hi = float(np.percentile(truth, 99.5))
            pad = 0.12 * (hi - lo + 1e-9)
            bins = np.linspace(lo - pad, hi + pad, 80)
            ax.hist(truth, bins=bins, density=True, color="0.45", alpha=0.55, label="truth")
            t_d, _ = np.histogram(truth, bins=bins, density=True)
            centers = 0.5 * (bins[:-1] + bins[1:])
            vals = samples[li][:, j]
            ax.hist(vals, bins=bins, density=True, histtype="step",
                    color=model_color, lw=1.5, label=model_label)
            d_m, _ = np.histogram(vals, bins=bins, density=True)
            res = smooth(d_m - t_d)
            axr.plot(centers, res, color=model_color, lw=1.1)
            res_absmax = max(res_absmax, float(np.max(np.abs(res))))
            ax.set_title(f"L{li} node{layers[li][j]} (var#{ci + 1})", fontsize=9.5)
            ax.set_ylabel("density", fontsize=9)
            ax.tick_params(labelbottom=False, labelsize=8)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8, loc="upper right")
            axr.axhline(0, ls="--", color="gray", lw=0.9)
            axr.set_xlabel("value", fontsize=8)
            axr.set_ylabel("err", fontsize=7.5)
            axr.tick_params(labelsize=7)
            res_axes.append(axr)
    for axr in res_axes:
        axr.set_ylim(-1.08 * (res_absmax + 1e-9), 1.08 * (res_absmax + 1e-9))

    xl = [f"L{li}" for li in down]
    ax_ll = fig.add_subplot(gs[len(down), 0:3])
    ax_corr = fig.add_subplot(gs[len(down), 3:6])
    ll_vals, corr_vals = [], []
    for li in down:
        tev = test_x[:, layers[li]]
        ll_vals.append(float(forest_log_density(chain[li], tev)[0].mean()) / max(len(layers[li]), 1))
        Ct = np.corrcoef(tev.T)
        denom = np.sqrt(len(layers[li]) * max(len(layers[li]) - 1, 1))
        corr_vals.append(float(np.linalg.norm(Ct - np.corrcoef(samples[li].T))) / denom)
    ax_ll.plot(xl, ll_vals, "o-", color=model_color, lw=1.8, label=model_label)
    ax_corr.plot(xl, corr_vals, "o-", color=model_color, lw=1.8, label=model_label)
    ax_ll.set_title("Per-dim held-out joint LL vs layer (higher=better)", fontsize=11)
    ax_ll.set_ylabel("LL/K")
    ax_ll.grid(alpha=0.3)
    ax_ll.legend(fontsize=9)
    ax_corr.set_title("Per-pair corr error vs layer (lower=better)", fontsize=11)
    ax_corr.set_ylabel("corr_fro/sqrt(K(K-1))")
    ax_corr.grid(alpha=0.3)
    ax_corr.legend(fontsize=9)

    def strongest_pair(mat):
        C = np.corrcoef(mat.T)
        best, bi, bj = -1.0, 0, min(1, C.shape[0] - 1)
        for i in range(C.shape[0]):
            for j in range(i + 1, C.shape[0]):
                if abs(C[i, j]) > best:
                    best, bi, bj = abs(C[i, j]), i, j
        return bi, bj

    for row, li in enumerate((down[0], down[-1])):
        tev = test_x[:n_plot, layers[li]]
        i_p, j_p = strongest_pair(tev)
        gx, gy = layers[li][i_p], layers[li][j_p]
        xr = np.percentile(tev[:, i_p], [1, 99])
        yr = np.percentile(tev[:, j_p], [1, 99])
        dx, dy = 0.1 * (xr[1] - xr[0] + 1e-9), 0.1 * (yr[1] - yr[0] + 1e-9)
        series = [
            ("truth", tev[:, i_p], tev[:, j_p], "0.45"),
            (model_label, samples[li][:, i_p], samples[li][:, j_p], model_color),
        ]
        srow = GridSpecFromSubplotSpec(1, len(series), subplot_spec=gs[len(down) + 1 + row, 0:6],
                                       wspace=0.32)
        for col, (tag, xx, yy, color) in enumerate(series):
            ax = fig.add_subplot(srow[0, col])
            r = float(np.corrcoef(xx, yy)[0, 1])
            ax.scatter(xx, yy, s=4, alpha=0.09, color=color, edgecolors="none", rasterized=True)
            ax.set_title(f"L{li} {tag} node{gx}-{gy} r={r:+.3f}", fontsize=10)
            ax.set_xlim(xr[0] - dx, xr[1] + dx)
            ax.set_ylim(yr[0] - dy, yr[1] + dy)
            ax.set_xlabel(f"node{gx}")
            ax.set_ylabel(f"node{gy}")

    fig.suptitle(f"dense_dag_r567 {model_label} refined slice (seed={seed})", fontsize=15, weight="bold")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"saved: {out_path}", flush=True)


def _plot_chain(
    test_x, layers, chain, key, *,
    model_label, model_color, seed,
    out_marginal, out_refined, marginal_title,
    skip_marginal: bool = False,
):
    n_plot = min(N_PLOT, test_x.shape[0])
    samples, key = _collect_samples(chain, n_plot, key)
    if not skip_marginal:
        plot_marginal_slices(
            test_x, layers, samples,
            model_label=model_label, model_color=model_color,
            title=marginal_title, out_path=out_marginal,
        )
    plot_refined_slices(
        test_x, layers, chain, samples,
        model_label=model_label, model_color=model_color,
        seed=seed, out_path=out_refined,
    )
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r5-only", action="store_true")
    ap.add_argument("--r7-only", action="store_true")
    ap.add_argument("--refined-only", action="store_true",
                    help="只生成精细切片图（跳过 5×20 marginal 重绘）")
    args = ap.parse_args()
    run_r5 = not args.r7_only
    run_r7 = not args.r5_only

    cfg = dict(CFG)
    cfg.update(nn_lr=2e-3, nn_steps=400, nn_batch_sz=512, nn_patience=6)
    t0 = time.perf_counter()

    key, spec, params, layers, train_x, test_x, forest0, s_max0 = prepare_data(args.seed, cfg)
    L0 = layers[0]

    if run_r5:
        print("[plot] fitting R5 nonneg chain...", flush=True)
        t1 = time.perf_counter()
        k_r5, key = jax.random.split(key)
        r5_chain = fit_analytic_chain_nonneg(forest0, spec, params, k_r5, s_max0, cfg)
        print(f"[plot] R5 nonneg fit {time.perf_counter()-t1:.1f}s", flush=True)
        key = _plot_chain(
            test_x, layers, r5_chain, key,
            model_label="R5 nonneg", model_color="#9467bd", seed=args.seed,
            out_marginal=OUT_R5, out_refined=OUT_R5_REF,
            marginal_title="dense_dag_r567 R5 nonneg per-node marginal slices (core→raw² + analytic L2)",
            skip_marginal=args.refined_only,
        )

    if run_r7:
        print("[plot] fitting R7 nonneg chain...", flush=True)
        t1 = time.perf_counter()
        k_f0, key = jax.random.split(key)
        forest0_nn = fit_layer_forest_nonneg(
            jnp.asarray(train_x[:, L0]), L0, cfg, k_f0,
            label="L0_nn", mi_threshold=cfg["mi_threshold"],
        )
        k_r7, key = jax.random.split(key)
        r7_chain = fit_sampled_chain_nonneg(forest0_nn, spec, params, k_r7, cfg)
        print(f"[plot] R7 nonneg fit {time.perf_counter()-t1:.1f}s", flush=True)
        key = _plot_chain(
            test_x, layers, r7_chain, key,
            model_label="R7 nonneg", model_color="#2ca02c", seed=args.seed,
            out_marginal=OUT_R7, out_refined=OUT_R7_REF,
            marginal_title="dense_dag_r567 R7 nonneg per-node marginal slices (core→raw² + block MLE)",
            skip_marginal=args.refined_only,
        )

    print(f"total {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
