"""汇总对比图：全局各方案 vs 分层(oracle)。

数据来自：
- global_vs_layered_complex_metrics.json (global_TT / global_TTNS / ttde_TT / layered_TTNS)
- sqr_mle_chowliu_probe (平方+MLE+Chow-Liu 树, 新)
- nonneg_mle_probe (非负核+MLE, chain / chow-liu)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# name, params, joint_LL, W1, corr_fro, category
# category: 0=global L2, 1=global MLE(no prior), 2=layered(oracle)
ROWS = [
    ("global_TT\n(linear+L2)",        191520, 1.17,  0.0434, 7.468, 0),
    ("global_TTNS\n(linear+L2)",      169920, 3.68,  0.0352, 7.330, 0),
    ("nonneg+MLE\n(chain)",           171936, 9.10,  0.0066, 7.054, 1),
    ("nonneg+MLE\n(Chow-Liu)",       5540400, 13.03, 0.0114, 4.072, 1),
    ("TTDE\n(square+MLE, chain)",     171936, 15.82, 0.0051, 2.974, 1),
    ("square+MLE\n(Chow-Liu tree)",    87696, 17.78, np.nan, np.nan, 1),
    ("layered TTNS\n(oracle DAG)",     19728, 20.69, 0.0049, 0.297, 2),
]

CMAP = {0: "#9aa5b1", 1: "#2f80ed", 2: "#eb5757"}
LABELS = {0: "global, linear+L2", 1: "global, MLE (no prior)", 2: "layered (oracle structure)"}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "figure.dpi": 130})

names = [r[0] for r in ROWS]
params = np.array([r[1] for r in ROWS], float)
ll = np.array([r[2] for r in ROWS], float)
w1 = np.array([r[3] for r in ROWS], float)
corr = np.array([r[4] for r in ROWS], float)
cats = [r[5] for r in ROWS]
colors = [CMAP[c] for c in cats]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) joint_LL horizontal bar, sorted ascending
ax = axes[0, 0]
order = np.argsort(ll)
y = np.arange(len(order))
ax.barh(y, ll[order], color=[colors[i] for i in order], edgecolor="black", linewidth=0.6)
ax.set_yticks(y)
ax.set_yticklabels([names[i] for i in order], fontsize=9)
for yi, i in enumerate(order):
    ax.text(ll[i] + 0.2, yi, f"{ll[i]:.1f}", va="center", fontsize=9)
ax.set_xlabel("joint log-likelihood  (higher = better)")
ax.set_title("(a) Density fit: joint log-likelihood", fontweight="bold")
ax.set_xlim(0, ll.max() * 1.12)

# (b) efficiency: params (log) vs joint_LL
ax = axes[0, 1]
for i in range(len(ROWS)):
    ax.scatter(params[i], ll[i], s=140, color=colors[i], edgecolor="black", zorder=3)
    ax.annotate(names[i].replace("\n", " "), (params[i], ll[i]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("learned parameters (log scale)")
ax.set_ylabel("joint log-likelihood")
ax.set_title("(b) Parameter efficiency (top-left = better)", fontweight="bold")

# (c) corr_fro bar (lower better), skip NaN
ax = axes[1, 0]
mask = ~np.isnan(corr)
idx = np.where(mask)[0]
x = np.arange(len(idx))
ax.bar(x, corr[idx], color=[colors[i] for i in idx], edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([names[i] for i in idx], fontsize=8, rotation=20, ha="right")
for xi, i in enumerate(idx):
    ax.text(xi, corr[i] + 0.1, f"{corr[i]:.2f}", ha="center", fontsize=9)
ax.set_ylabel("correlation error  ||Δcorr||_F  (lower = better)")
ax.set_title("(c) Correlation structure error", fontweight="bold")

# (d) W1 bar (lower better)
ax = axes[1, 1]
ax.bar(x, w1[idx] * 1e3, color=[colors[i] for i in idx], edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([names[i] for i in idx], fontsize=8, rotation=20, ha="right")
for xi, i in enumerate(idx):
    ax.text(xi, w1[i] * 1e3 + 0.5, f"{w1[i]*1e3:.1f}", ha="center", fontsize=9)
ax.set_ylabel("marginal W1  (×1e-3, lower = better)")
ax.set_title("(d) Marginal Wasserstein-1 error", fontweight="bold")

handles = [Patch(facecolor=CMAP[k], edgecolor="black", label=LABELS[k]) for k in (0, 1, 2)]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.005), fontsize=11)
fig.suptitle("Global estimators vs. layered TTNS  —  complex 24-node max-plus DAG",
             fontweight="bold", y=1.03, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = REPORTS / "model_showdown.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
