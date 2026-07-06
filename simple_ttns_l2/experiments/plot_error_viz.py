"""Error visualization for the TTNS layered budget sweep.

Reads the two metrics JSONs and produces a 2x2 figure summarising *how large*
the density / correlation errors are and *how they scale with depth and budget*.

Idempotent / re-runnable. Only uses json + numpy + matplotlib (no project imports,
no jax) so it stays lightweight and finishes in seconds.

Run with:
    env -u PYTHONPATH python3 -u -m simple_ttns_l2.experiments.plot_error_viz
"""

import json
import sys
from pathlib import Path

# Keep the sys.path convention of the experiments package even though this
# script does not import any project modules.
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "TTNSDE"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
SWEEP_JSON = REPORTS / "budget_sweep_layered_metrics.json"
R6_JSON = REPORTS / "budget_r6_block_ref_metrics.json"
OUT_PNG = REPORTS / "budget_sweep_error_viz.png"

K = 20
PAIR_NORM = np.sqrt(K * (K - 1))  # sqrt(380)
DOWNSTREAM = [1, 2, 3, 4]         # L0 is the source layer; only downstream shown

# Consistent color families: R5 -> red, R7 -> blue, R6 -> green.
RED = plt.cm.Reds
BLUE = plt.cm.Blues
GREEN = ["#2ca25f", "#99d8c9"]  # R6: rank8 / rank16


def density_acc_pct(ceiling, ll):
    """Per-dim density accuracy = exp(-(ceiling-ll)/K) * 100%."""
    return np.exp(-(ceiling - ll) / K) * 100.0


def per_pair_err(fro):
    return fro / PAIR_NORM


def load():
    with open(SWEEP_JSON) as f:
        sweep = json.load(f)
    with open(R6_JSON) as f:
        r6 = json.load(f)
    return sweep, r6


def rows_of(point):
    """Return the rows of the first seed of a point, keyed by layer index."""
    seed0 = point["seeds"][0]
    return {r["li"]: r for r in seed0["rows"]}


def get_series(point, field):
    """Return array over DOWNSTREAM layers for a given row field, nan if absent."""
    rows = rows_of(point)
    out = []
    for li in DOWNSTREAM:
        r = rows.get(li, {})
        out.append(r.get(field, np.nan))
    return np.array(out, dtype=float)


def main():
    sweep, r6 = load()
    ceilings = sweep["ceilings"]
    points = sweep["points"]
    labels = list(points.keys())

    # Precompute which points carry R5 / R7.
    r5_labels = [l for l in labels if not np.all(np.isnan(get_series(points[l], "ll_r5")))]
    r7_labels = [l for l in labels if not np.all(np.isnan(get_series(points[l], "ll_r7")))]

    # Best R7 by mean downstream density accuracy (highest = closest to oracle LL).
    def mean_dens_r7(l):
        ll = get_series(points[l], "ll_r7")
        acc = np.array([density_acc_pct(ceilings[li], ll[k])
                        for k, li in enumerate(DOWNSTREAM)])
        return np.nanmean(acc)
    best_r7 = max(r7_labels, key=mean_dens_r7) if r7_labels else None

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("TTNS Layered Density Estimation - Budget Sweep Error Diagnostics",
                 fontsize=15, fontweight="bold")
    x = np.array(DOWNSTREAM)
    xlabels = [f"L{li}" for li in DOWNSTREAM]

    # ---------------------------------------------------------------
    # (1) Top-left: per-layer per-dim density accuracy vs oracle
    # ---------------------------------------------------------------
    ax = axes[0, 0]
    n_r5 = max(len(r5_labels), 1)
    for i, l in enumerate(r5_labels):
        ll = get_series(points[l], "ll_r5")
        acc = np.array([density_acc_pct(ceilings[li], ll[k])
                        for k, li in enumerate(DOWNSTREAM)])
        ax.plot(x, acc, marker="o", lw=1.8,
                color=RED(0.45 + 0.5 * i / n_r5),
                label=f"R5  {l}")
    if best_r7 is not None:
        ll = get_series(points[best_r7], "ll_r7")
        acc = np.array([density_acc_pct(ceilings[li], ll[k])
                        for k, li in enumerate(DOWNSTREAM)])
        ax.plot(x, acc, marker="s", lw=2.2, ls="--",
                color=BLUE(0.75),
                label=f"R7 (best)  {best_r7}")
    ax.axhline(100.0, color="gray", ls=":", lw=1.2, label="oracle (100%)")
    ax.set_title("(1) Per-dim density accuracy vs oracle, by layer")
    ax.set_xlabel("Layer (downstream)")
    ax.set_ylabel("Per-dim density accuracy  exp(-gap/K)  [%]")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # ---------------------------------------------------------------
    # (2) Top-right: per-layer per-pair correlation error
    # ---------------------------------------------------------------
    ax = axes[0, 1]
    for i, l in enumerate(r5_labels):
        err = per_pair_err(get_series(points[l], "fro_r5"))
        ax.plot(x, err, marker="o", lw=1.6, ls="--",
                color=RED(0.45 + 0.5 * i / n_r5),
                label=f"R5  {l}")
    n_r7 = max(len(r7_labels), 1)
    for i, l in enumerate(r7_labels):
        err = per_pair_err(get_series(points[l], "fro_r7"))
        ax.plot(x, err, marker="s", lw=1.8,
                color=BLUE(0.4 + 0.55 * i / n_r7),
                label=f"R7  {l}")
    ax.axhline(0.0, color="gray", ls=":", lw=1.2, label="oracle (0)")
    ax.set_title("(2) Per-pair correlation error, by layer")
    ax.set_xlabel("Layer (downstream)")
    ax.set_ylabel("Per-pair error  fro / sqrt(K(K-1))  [corr units]")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # ---------------------------------------------------------------
    # (3) Bottom-left: budget-knob effect (dual y-axis)
    #     red  = downstream-mean per-dim density gap (R5)
    #     blue = downstream-mean per-pair correlation error (R7)
    # ---------------------------------------------------------------
    ax = axes[1, 0]
    xb = np.arange(len(labels))
    gap_r5 = []
    for l in labels:
        ll = get_series(points[l], "ll_r5")
        gaps = np.array([(ceilings[li] - ll[k]) / K
                         for k, li in enumerate(DOWNSTREAM)])
        gap_r5.append(np.nanmean(gaps) if not np.all(np.isnan(gaps)) else np.nan)
    gap_r5 = np.array(gap_r5)

    err_r7 = []
    for l in labels:
        err = per_pair_err(get_series(points[l], "fro_r7"))
        err_r7.append(np.nanmean(err) if not np.all(np.isnan(err)) else np.nan)
    err_r7 = np.array(err_r7)

    c_red, c_blue = "#c0392b", "#2166ac"
    ln1 = ax.plot(xb, gap_r5, marker="o", color=c_red, lw=2.0,
                  label="R5 mean per-dim density gap")[0]
    ax.set_ylabel("Mean per-dim density gap  (ceiling-LL)/K   [nats]", color=c_red)
    ax.tick_params(axis="y", labelcolor=c_red)
    ax.set_xticks(xb)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ln2 = ax2.plot(xb, err_r7, marker="s", color=c_blue, lw=2.0, ls="--",
                   label="R7 mean per-pair corr error")[0]
    ax2.set_ylabel("Mean per-pair correlation error  fro/sqrt(380)", color=c_blue)
    ax2.tick_params(axis="y", labelcolor=c_blue)

    ax.set_title("(3) Budget-knob effect (downstream L1-L4 mean)")
    ax.legend(handles=[ln1, ln2], fontsize=8, loc="upper right")

    # ---------------------------------------------------------------
    # (4) Bottom-right: R6 small-block reference (block 0, real non-tree corr)
    #     bars: tree(R5) / joint(R6) / samp(R7) fro, rank8 vs rank16 side-by-side
    # ---------------------------------------------------------------
    ax = axes[1, 1]
    metrics = [("fro_tree", "tree (R5)", "#e34a33"),
               ("fro_joint", "joint (R6)", "#2ca25f"),
               ("fro_samp", "samp (R7)", "#2166ac")]
    ranks = [k for k in ("rank8", "rank16") if k in r6]
    # Use block 0 (bi == 0): the block with real non-tree correlation.
    def block0(entries):
        for e in entries:
            if e.get("bi") == 0:
                return e
        return entries[0]

    grp = np.arange(len(metrics))
    width = 0.36
    for j, rk in enumerate(ranks):
        b0 = block0(r6[rk])
        vals = [b0[m] for m, _, _ in metrics]
        offset = (j - (len(ranks) - 1) / 2) * width
        bars = ax.bar(grp + offset, vals, width,
                      color=[c for _, _, c in metrics],
                      alpha=0.65 + 0.35 * j,
                      edgecolor="black", linewidth=0.6,
                      label=rk)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(grp)
    ax.set_xticklabels([nm for _, nm, _ in metrics])
    ax.set_ylabel("Fro error of pairwise correlation (block 0)")
    ax.set_title("(4) R6 small-block reference: target vs capacity")
    ax.grid(True, axis="y", alpha=0.3)
    # Legend: rank groups (bar alpha) + metric colors handled by x labels.
    ax.legend(fontsize=8, title="fit rank", loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot_error_viz] wrote {OUT_PNG}")
    print(f"[plot_error_viz] R5 points: {r5_labels}")
    print(f"[plot_error_viz] R7 points: {r7_labels}")
    print(f"[plot_error_viz] best R7 (by mean density acc): {best_r7}")


if __name__ == "__main__":
    main()
