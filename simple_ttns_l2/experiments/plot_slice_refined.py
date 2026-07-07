"""精细切片密度图:分层解析链(R5) vs 采样链(R7)随深度的拟合 + 显式误差。

更深(7 层 × 6 维 = 42 节点,下游 L1..L6)、每层更小(clusters=[3,3]),内存友好。
复用 plot_slice_viz.fit_all 的拟合流程与 budget_sweep_layered 的 oracle 上限 / corr_fro,
只出图,不改现有实验脚本。

三块:
  Block 1 · 逐层 1D 边缘密度(L1..L6 每层方差最大维,2×3)。
  Block 2 · 显式误差:左 每维密度 gap vs 层((ceiling-LL)/K);右 每对相关误差 vs 层
            (corr_fro/sqrt(K(K-1)))。两图画 oracle(=0)参照线。
  Block 3 · 强相关对浅(L1)vs深(L6):truth/R5/R7 2D 散点,标题带样本 r。

保存 simple_ttns_l2/reports/budget_sweep_slice_refined.png。配色:truth 灰、R5 红、R7 蓝。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np

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

from simple_ttns_l2.layered_forest import forest_log_density, sample_forest  # noqa: E402
from simple_ttns_l2.experiments.plot_slice_viz import fit_all, strongest_pair  # noqa: E402
from simple_ttns_l2.experiments.budget_sweep_layered import (  # noqa: E402
    layer_ceilings, corr_fro_layer,
)

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"

C_TRUTH = "0.45"      # gray
C_R5 = "#d62728"      # red   (analytic chain)
C_R7 = "#1f77b4"      # blue  (sampled chain)

# 更深、每层更小,省内存。7 层 × 6 维 = 42 节点,下游 L1..L6。
CFG = dict(
    n_layers=7, clusters=[3, 3], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
    n_total=12000, n_sample=4000, n_fit=12000, q=2, m=24, rank=16, src_sigma=0.03,
    lr=2e-3, steps=500, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
    n_s=100, n_s_pair=80, an_lr=3e-3, an_steps=500,
)


def main():
    t_all = time.perf_counter()
    cfg = CFG
    spec, layers, test_x, AN, SP, key = fit_all(cfg)
    n = cfg["n_sample"]

    down = list(range(1, len(layers)))          # 下游层 L1..L6
    ceils = layer_ceilings(test_x, spec)
    print("[oracle 上限] " + "  ".join(f"L{i}={c:.3f}" for i, c in enumerate(ceils)), flush=True)

    # 每层各采一次(1D + 2D 复用)。
    r5_samp, r7_samp = {}, {}
    for li in down:
        k1, key2 = jax.random.split(key); key = key2
        k2, key2 = jax.random.split(key); key = key2
        r5_samp[li] = np.asarray(sample_forest(AN[li], k1, n, grid_size=400))
        r7_samp[li] = np.asarray(sample_forest(SP[li], k2, n, grid_size=400))

    # -------- Block 2 误差指标(每维密度 gap、每对相关误差)--------
    gap_r5, gap_r7, fro_r5, fro_r7 = [], [], [], []
    for li in down:
        tev = test_x[:, layers[li]]
        K = len(layers[li])
        ll5 = float(forest_log_density(AN[li], tev)[0].mean())
        ll7 = float(forest_log_density(SP[li], tev)[0].mean())
        gap_r5.append((ceils[li] - ll5) / K)
        gap_r7.append((ceils[li] - ll7) / K)
        k1, key = jax.random.split(key)
        f5, _ = corr_fro_layer(AN[li], tev, k1, n)
        k2, key = jax.random.split(key)
        f7, _ = corr_fro_layer(SP[li], tev, k2, n)
        denom = np.sqrt(K * (K - 1))
        fro_r5.append(f5 / denom)
        fro_r7.append(f7 / denom)
        print(f"[L{li}] gap/K R5={gap_r5[-1]:.4f} R7={gap_r7[-1]:.4f}  "
              f"corr_err R5={fro_r5[-1]:.4f} R7={fro_r7[-1]:.4f}", flush=True)

    # ================= 绘图 =================
    fig = plt.figure(figsize=(15.5, 19.0))
    gs = GridSpec(5, 6, figure=fig, hspace=0.62, wspace=0.5,
                  height_ratios=[1.0, 1.0, 1.15, 1.05, 1.05],
                  top=0.925, bottom=0.035, left=0.055, right=0.985)

    xl = [f"L{li}" for li in down]
    block_top_ax = {}   # 记录每块左上子图,用于放分块标题

    # ---------- Block 1: 逐层 1D 边缘 (2x3) ----------
    for idx, li in enumerate(down):
        r, c = idx // 3, idx % 3
        ax = fig.add_subplot(gs[r, c])
        if idx == 0:
            block_top_ax[1] = ax
        tev = test_x[:, layers[li]]
        j = int(np.argmax(tev.var(axis=0)))
        gid = layers[li][j]
        truth, r5, r7 = tev[:, j], r5_samp[li][:, j], r7_samp[li][:, j]
        lo = float(np.percentile(truth, 0.5)); hi = float(np.percentile(truth, 99.5))
        pad = 0.12 * (hi - lo + 1e-9)
        bins = np.linspace(lo - pad, hi + pad, 60)
        ax.hist(truth, bins=bins, density=True, color=C_TRUTH, alpha=0.55, label="truth")
        ax.hist(r5, bins=bins, density=True, histtype="step", color=C_R5, lw=1.8, label="R5 analytic")
        ax.hist(r7, bins=bins, density=True, histtype="step", color=C_R7, lw=1.8, label="R7 sampled")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"L{li}  node{gid} (max-var dim)", fontsize=10.5)
        ax.set_xlabel("value"); ax.set_ylabel("density")
        if idx == 0:
            ax.legend(fontsize=8.5, loc="upper right")

    # ---------- Block 2: 显式误差 vs 层 ----------
    ax_g = fig.add_subplot(gs[2, 0:3])
    block_top_ax[2] = ax_g
    ax_g.plot(xl, gap_r5, "o-", color=C_R5, lw=1.9, label="R5 analytic")
    ax_g.plot(xl, gap_r7, "s-", color=C_R7, lw=1.9, label="R7 sampled")
    ax_g.axhline(0, ls="--", color="gray", lw=1.2, label="oracle (gap=0)")
    ax_g.set_title("Per-dim density gap vs layer  (ceiling - joint_LL)/K,  lower=better",
                   fontsize=11)
    ax_g.set_xlabel("layer"); ax_g.set_ylabel("gap per dim (nats)")
    ax_g.legend(fontsize=9); ax_g.grid(alpha=0.3)

    ax_c = fig.add_subplot(gs[2, 3:6])
    ax_c.plot(xl, fro_r5, "o-", color=C_R5, lw=1.9, label="R5 analytic")
    ax_c.plot(xl, fro_r7, "s-", color=C_R7, lw=1.9, label="R7 sampled")
    ax_c.axhline(0, ls="--", color="gray", lw=1.2, label="oracle (err=0)")
    ax_c.set_title("Per-pair corr error vs layer  corr_fro/sqrt(K(K-1)),  lower=better",
                   fontsize=11)
    ax_c.set_xlabel("layer"); ax_c.set_ylabel("corr error per pair")
    ax_c.legend(fontsize=9); ax_c.grid(alpha=0.3)

    # ---------- Block 3: 强相关对 浅(L1) vs 深(L6) ----------
    scatter_stats = {}
    for row, li in enumerate((down[0], down[-1])):     # L1 浅, L6 深
        tev = test_x[:, layers[li]]
        i_p, j_p, corr_true = strongest_pair(tev)
        gx, gy = layers[li][i_p], layers[li][j_p]
        xr = (float(np.percentile(tev[:, i_p], 1)), float(np.percentile(tev[:, i_p], 99)))
        yr = (float(np.percentile(tev[:, j_p], 1)), float(np.percentile(tev[:, j_p], 99)))
        dx = 0.1 * (xr[1] - xr[0] + 1e-9); dy = 0.1 * (yr[1] - yr[0] + 1e-9)
        series = [("truth", tev[:, i_p], tev[:, j_p], C_TRUTH),
                  ("R5 analytic", r5_samp[li][:, i_p], r5_samp[li][:, j_p], C_R5),
                  ("R7 sampled", r7_samp[li][:, i_p], r7_samp[li][:, j_p], C_R7)]
        rs = {}
        for col, (tag, xx, yy, color) in enumerate(series):
            ax = fig.add_subplot(gs[3 + row, 2 * col:2 * col + 2])
            if row == 0 and col == 0:
                block_top_ax[3] = ax
            r = float(np.corrcoef(xx, yy)[0, 1])
            rs[tag] = r
            ax.scatter(xx, yy, s=5, alpha=0.16, color=color, edgecolors="none", rasterized=True)
            depth = "shallow" if row == 0 else "deep"
            ax.set_title(f"L{li} ({depth}) {tag}  r={r:+.3f}", fontsize=10.5)
            ax.set_xlabel(f"node{gx}"); ax.set_ylabel(f"node{gy}")
            ax.set_xlim(xr[0] - dx, xr[1] + dx); ax.set_ylim(yr[0] - dy, yr[1] + dy)
        scatter_stats[li] = dict(pair=(gx, gy), r_truth=rs["truth"],
                                 r_r5=rs["R5 analytic"], r_r7=rs["R7 sampled"])

    # 分块标题(锚到每块左上子图上方,避免与子图重叠)
    fig.text(0.5, 0.958, "Refined layered-chain slice fit  "
             f"(7 layers x 6 dims, rank={cfg['rank']}, m={cfg['m']})",
             ha="center", fontsize=15, weight="bold")
    headers = {
        1: "Block 1 - Per-layer 1D marginal density (L1..L6, max-var dim)",
        2: "Block 2 - Explicit error vs depth (density gap / corr error)",
        3: "Block 3 - Strongest-corr pair: shallow L1 vs deep L6 (truth / R5 / R7)",
    }
    for blk, ax in block_top_ax.items():
        y = ax.get_position().y1 + 0.011
        fig.text(0.5, y, headers[blk], ha="center", fontsize=12.5, weight="bold")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "budget_sweep_slice_refined.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}", flush=True)

    # -------- 误差数字小结(浅 L1 vs 深 L6)--------
    print("\n==== error vs depth summary ====", flush=True)
    print(f"per-dim density gap/K:  L1 R5={gap_r5[0]:.4f} R7={gap_r7[0]:.4f}   "
          f"L6 R5={gap_r5[-1]:.4f} R7={gap_r7[-1]:.4f}", flush=True)
    print(f"per-pair corr error:    L1 R5={fro_r5[0]:.4f} R7={fro_r7[0]:.4f}   "
          f"L6 R5={fro_r5[-1]:.4f} R7={fro_r7[-1]:.4f}", flush=True)
    for li in (down[0], down[-1]):
        s = scatter_stats[li]
        print(f"strongest pair L{li} nodes{s['pair']}: truth r={s['r_truth']:+.3f}  "
              f"R5 r={s['r_r5']:+.3f}  R7 r={s['r_r7']:+.3f}", flush=True)
    print(f"total {time.perf_counter()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
