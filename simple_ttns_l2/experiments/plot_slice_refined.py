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

from simple_ttns_l2.layered_forest import forest_log_density, sample_forest  # noqa: E402
from simple_ttns_l2.experiments.plot_slice_viz import fit_all, strongest_pair  # noqa: E402
from simple_ttns_l2.experiments.budget_sweep_layered import (  # noqa: E402
    layer_ceilings, corr_fro_layer,
)
from simple_ttns_l2.dag_pipeline import sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources  # noqa: E402

# 可视化专用大样本量(真值从生成过程直接抽,与拟合 n_total 解耦; 模型森林同量采样)。
# 直方图/残差/散点更密更平滑,不加重拟合内存。16k 兼顾密度与本机内存(22k+ 会 swap)。
N_PLOT = 16000

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"

C_TRUTH = "0.45"      # gray
C_R5 = "#d62728"      # red   (analytic chain, tree projection)
C_R6 = "#2ca02c"      # green (analytic chain, full joint)
C_R7 = "#1f77b4"      # blue  (sampled chain)

# 更深、每层更小,省内存。7 层 × 6 维 = 42 节点,下游 L1..L6。
# 层间延迟改 log-skew-normal(正支撑、右偏重尾),非均匀。R6(完整联合)也跑,三方对比。
CFG = dict(
    n_layers=7, clusters=[3, 3], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, kind="logskewnorm",
               e_xi=-2.12, e_omega=0.45, e_alpha=4.0,
               d_xi=-2.12, d_omega=0.45, d_alpha=4.0),
    n_total=12000, n_sample=6000, n_fit=12000, q=2, m=24, rank=16, src_sigma=0.03,
    lr=2e-3, steps=500, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
    n_s=100, n_s_pair=80, an_lr=3e-3, an_steps=500,
    with_r6=True, n_s_joint=44,
)


def main():
    t_all = time.perf_counter()
    cfg = CFG
    spec, layers, test_x, AN, SP, JN, key = fit_all(cfg)
    n = N_PLOT

    down = list(range(1, len(layers)))          # 下游层 L1..L6
    ceils = layer_ceilings(test_x, spec)
    print("[oracle 上限] " + "  ".join(f"L{i}={c:.3f}" for i, c in enumerate(ceils)), flush=True)

    # 可视化用大真值样本:从生成过程直接抽 N_PLOT(与拟合 n_total 解耦,truth 更密)。
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_big, key = jax.random.split(key)
    truth_big = np.asarray(sample_joint(spec, sources, kernels, k_big, N_PLOT, clip=(-1e9, 1e9)))
    print(f"[plot] 大真值样本 {truth_big.shape}  模型每层采样 n={n}", flush=True)

    # 每层各采一次(1D + 2D 复用),模型采 N_PLOT 与真值同量。
    r5_samp, r7_samp, r6_samp = {}, {}, {}
    for li in down:
        k1, key2 = jax.random.split(key); key = key2
        k2, key2 = jax.random.split(key); key = key2
        r5_samp[li] = np.asarray(sample_forest(AN[li], k1, n, grid_size=400))
        r7_samp[li] = np.asarray(sample_forest(SP[li], k2, n, grid_size=400))
        if JN is not None:
            k3, key2 = jax.random.split(key); key = key2
            r6_samp[li] = np.asarray(sample_forest(JN[li], k3, n, grid_size=400))

    # -------- Block 2 误差指标(每维密度 gap、每对相关误差)--------
    gap_r5, gap_r7, fro_r5, fro_r7 = [], [], [], []
    gap_r6, fro_r6 = [], []
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
        if JN is not None:
            ll6 = float(forest_log_density(JN[li], tev)[0].mean())
            gap_r6.append((ceils[li] - ll6) / K)
            k3, key = jax.random.split(key)
            f6, _ = corr_fro_layer(JN[li], tev, k3, n)
            fro_r6.append(f6 / denom)
        print(f"[L{li}] gap/K R5={gap_r5[-1]:.4f} R7={gap_r7[-1]:.4f}"
              + (f" R6={gap_r6[-1]:.4f}" if JN is not None else "")
              + f"  corr_err R5={fro_r5[-1]:.4f} R7={fro_r7[-1]:.4f}"
              + (f" R6={fro_r6[-1]:.4f}" if JN is not None else ""), flush=True)

    # ================= 绘图 =================
    N_NODE = 3   # 每层画方差前 N_NODE 个节点(更多切片)
    fig = plt.figure(figsize=(18.0, 32.0))
    gs = GridSpec(9, 6, figure=fig, hspace=0.66, wspace=0.5,
                  height_ratios=[1, 1, 1, 1, 1, 1, 1.15, 1.05, 1.05],
                  top=0.95, bottom=0.028, left=0.055, right=0.985)

    xl = [f"L{li}" for li in down]
    block_top_ax = {}   # 记录每块左上子图,用于放分块标题
    res_axes = []       # Block 1 残差带,循环后统一设共享对称 ylim
    res_absmax = 0.0

    def _smooth(a):
        k = np.array([0.25, 0.5, 0.25])
        return np.convolve(a, k, mode="same")

    # ---------- Block 1: 逐层 top-N 方差节点 1D 边缘 (6 层 × N_NODE),每格 = 主密度 + 残差带 ----------
    for ri, li in enumerate(down):                       # 每层一行
        tev = test_x[:, layers[li]]
        order = np.argsort(tev.var(axis=0))[::-1][:N_NODE]   # 方差前 N_NODE 维
        for ci, j in enumerate(order):
            j = int(j)
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[ri, 2 * ci:2 * ci + 2],
                                          height_ratios=[3, 1], hspace=0.08)
            ax = fig.add_subplot(sub[0])                 # 主图(密度)
            ax_res = fig.add_subplot(sub[1], sharex=ax)  # 残差带
            if ri == 0 and ci == 0:
                block_top_ax[1] = ax
            gid = layers[li][j]
            truth, r5, r7 = truth_big[:, gid], r5_samp[li][:, j], r7_samp[li][:, j]
            r6 = r6_samp[li][:, j] if JN is not None else None
            lo = float(np.percentile(truth, 0.5)); hi = float(np.percentile(truth, 99.5))
            pad = 0.12 * (hi - lo + 1e-9)
            bins = np.linspace(lo - pad, hi + pad, 80)   # 同一组 bin 边界
            ax.hist(truth, bins=bins, density=True, color=C_TRUTH, alpha=0.55, label="truth")
            ax.hist(r5, bins=bins, density=True, histtype="step", color=C_R5, lw=1.2, alpha=0.5)
            ax.hist(r7, bins=bins, density=True, histtype="step", color=C_R7, lw=1.2, alpha=0.5)
            xs = np.linspace(lo - pad, hi + pad, 400)
            try:
                ax.plot(xs, gaussian_kde(r5)(xs), color=C_R5, lw=1.7, label="R5 analytic")
                if r6 is not None:
                    ax.plot(xs, gaussian_kde(r6)(xs), color=C_R6, lw=1.5, label="R6 analytic")
                ax.plot(xs, gaussian_kde(r7)(xs), color=C_R7, lw=1.7, label="R7 sampled")
            except np.linalg.LinAlgError:                # 退化(方差近 0)时跳过 KDE
                ax.plot([], [], color=C_R5, lw=1.7, label="R5 analytic")
                ax.plot([], [], color=C_R7, lw=1.7, label="R7 sampled")
            tag = "max-var" if ci == 0 else f"var#{ci + 1}"
            ax.set_title(f"L{li}  node{gid} ({tag})", fontsize=9.5)
            ax.set_ylabel("density", fontsize=9)
            ax.tick_params(labelbottom=False, labelsize=8)  # x 轴交给残差带
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8, loc="upper right")
            # 残差带:density err = model - truth,逐 bin 用同一 bin 边界算
            t_d, _ = np.histogram(truth, bins=bins, density=True)
            r5_d, _ = np.histogram(r5, bins=bins, density=True)
            r7_d, _ = np.histogram(r7, bins=bins, density=True)
            centers = 0.5 * (bins[:-1] + bins[1:])
            res5 = _smooth(r5_d - t_d); res7 = _smooth(r7_d - t_d)
            ax_res.axhline(0, ls="--", color="gray", lw=0.9)
            ax_res.plot(centers, res5, color=C_R5, lw=1.2)
            res_stack = [res5, res7]
            if r6 is not None:
                r6_d, _ = np.histogram(r6, bins=bins, density=True)
                res6 = _smooth(r6_d - t_d)
                ax_res.plot(centers, res6, color=C_R6, lw=1.1)
                res_stack.append(res6)
            ax_res.plot(centers, res7, color=C_R7, lw=1.2)
            ax_res.set_xlim(lo - pad, hi + pad)
            ax_res.set_xlabel("value", fontsize=8); ax_res.set_ylabel("err", fontsize=7.5)
            ax_res.tick_params(labelsize=7)
            res_axes.append(ax_res)
            res_absmax = max(res_absmax,
                             float(np.max(np.abs(np.concatenate(res_stack)))))

    # 所有残差带统一对称 ylim,便于跨节点/跨深度直接比较误差幅度。
    res_ylim = 1.08 * (res_absmax + 1e-9)
    for axr in res_axes:
        axr.set_ylim(-res_ylim, res_ylim)

    # ---------- Block 2: 显式误差 vs 层 ----------
    ax_g = fig.add_subplot(gs[6, 0:3])
    block_top_ax[2] = ax_g
    ax_g.plot(xl, gap_r5, "o-", color=C_R5, lw=1.9, label="R5 analytic (tree)")
    if gap_r6:
        ax_g.plot(xl, gap_r6, "^-", color=C_R6, lw=1.9, label="R6 analytic (joint)")
    ax_g.plot(xl, gap_r7, "s-", color=C_R7, lw=1.9, label="R7 sampled (joint)")
    ax_g.axhline(0, ls="--", color="gray", lw=1.2, label="oracle (gap=0)")
    ax_g.set_title("Per-dim density gap vs layer  (ceiling - joint_LL)/K,  lower=better",
                   fontsize=11)
    ax_g.set_xlabel("layer"); ax_g.set_ylabel("gap per dim (nats)")
    ax_g.legend(fontsize=9); ax_g.grid(alpha=0.3)

    ax_c = fig.add_subplot(gs[6, 3:6])
    ax_c.plot(xl, fro_r5, "o-", color=C_R5, lw=1.9, label="R5 analytic (tree)")
    if fro_r6:
        ax_c.plot(xl, fro_r6, "^-", color=C_R6, lw=1.9, label="R6 analytic (joint)")
    ax_c.plot(xl, fro_r7, "s-", color=C_R7, lw=1.9, label="R7 sampled (joint)")
    ax_c.axhline(0, ls="--", color="gray", lw=1.2, label="oracle (err=0)")
    ax_c.set_title("Per-pair corr error vs layer  corr_fro/sqrt(K(K-1)),  lower=better",
                   fontsize=11)
    ax_c.set_xlabel("layer"); ax_c.set_ylabel("corr error per pair")
    ax_c.legend(fontsize=9); ax_c.grid(alpha=0.3)

    # ---------- Block 3: 强相关对 浅(L1) vs 深(L6), truth/R5/R6/R7 四列 ----------
    scatter_stats = {}
    for row, li in enumerate((down[0], down[-1])):     # L1 浅, L6 深
        tev = test_x[:, layers[li]]
        i_p, j_p, corr_true = strongest_pair(tev)
        gx, gy = layers[li][i_p], layers[li][j_p]
        xr = (float(np.percentile(tev[:, i_p], 1)), float(np.percentile(tev[:, i_p], 99)))
        yr = (float(np.percentile(tev[:, j_p], 1)), float(np.percentile(tev[:, j_p], 99)))
        dx = 0.1 * (xr[1] - xr[0] + 1e-9); dy = 0.1 * (yr[1] - yr[0] + 1e-9)
        series = [("truth", truth_big[:, gx], truth_big[:, gy], C_TRUTH),
                  ("R5 analytic", r5_samp[li][:, i_p], r5_samp[li][:, j_p], C_R5)]
        if JN is not None:
            series.append(("R6 analytic", r6_samp[li][:, i_p], r6_samp[li][:, j_p], C_R6))
        series.append(("R7 sampled", r7_samp[li][:, i_p], r7_samp[li][:, j_p], C_R7))
        ncol = len(series)
        subrow = GridSpecFromSubplotSpec(1, ncol, subplot_spec=gs[7 + row, 0:6], wspace=0.32)
        rs = {}
        for col, (tag, xx, yy, color) in enumerate(series):
            ax = fig.add_subplot(subrow[0, col])
            if row == 0 and col == 0:
                block_top_ax[3] = ax
            r = float(np.corrcoef(xx, yy)[0, 1])
            rs[tag] = r
            ax.scatter(xx, yy, s=4, alpha=0.09, color=color, edgecolors="none", rasterized=True)
            depth = "shallow" if row == 0 else "deep"
            ax.set_title(f"L{li} ({depth}) {tag}  r={r:+.3f}", fontsize=10.5)
            ax.set_xlabel(f"node{gx}"); ax.set_ylabel(f"node{gy}")
            ax.set_xlim(xr[0] - dx, xr[1] + dx); ax.set_ylim(yr[0] - dy, yr[1] + dy)
        scatter_stats[li] = dict(pair=(gx, gy), r_truth=rs["truth"],
                                 r_r5=rs["R5 analytic"], r_r7=rs["R7 sampled"],
                                 r_r6=rs.get("R6 analytic"))

    # 分块标题(锚到每块左上子图上方,避免与子图重叠)
    fig.text(0.5, 0.988, "Refined layered-chain slice fit  "
             f"(7 layers x 6 dims, rank={cfg['rank']}, m={cfg['m']}, log-skew-normal delays)",
             ha="center", fontsize=16, weight="bold")
    headers = {
        1: "Block 1 - Per-layer 1D marginal density (L1..L6 x top-3 var nodes) with residual band",
        2: "Block 2 - Explicit error vs depth (density gap / corr error)",
        3: "Block 3 - Strongest-corr pair: shallow L1 vs deep L6 (truth / R5 / R6 / R7)",
    }
    for blk, ax in block_top_ax.items():
        y = ax.get_position().y1 + 0.011
        fig.text(0.5, y, headers[blk], ha="center", fontsize=12.5, weight="bold")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "budget_sweep_slice_refined.png"
    fig.savefig(out, dpi=170)
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
