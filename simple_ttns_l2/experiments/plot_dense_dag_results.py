"""画 dense_dag_r567_three_way 的结果图:逐层 joint_LL@truth 与 corr_fro(R5 vs R7，误差棒=seed std）。

读 simple_ttns_l2/reports/dense_dag_r567_three_way_metrics.json（脚本跑完自动写的）。
输出 simple_ttns_l2/reports/dense_dag_r567_results.png。
用法：env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.plot_dense_dag_results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf"):
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
SRC = REPORTS / "dense_dag_r567_three_way_metrics.json"
OUT = REPORTS / "dense_dag_r567_results.png"

STYLE = {"R5_tree": ("#4C78A8", "o", "R5 解析树投影"),
         "R7_sampled": ("#E45756", "s", "R7 采样求L2")}


def main():
    d = json.loads(SRC.read_text())
    agg = d["aggregate"]
    seeds = d["seeds"]
    sp = d["per_seed"][0]["spec"]
    methods = [m for m in ("R5_tree", "R7_sampled") if m in agg["ll"]]
    n_layers = len(agg["ll"][methods[0]])
    xs = list(range(n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    panels = [("ll", "逐层 joint_LL@truth  (↑ 越高越好)", axes[0]),
              ("fro", "逐层 corr_fro vs truth  (↓ 越低越好)", axes[1])]
    for metric, title, ax in panels:
        for m in methods:
            col, mk, lab = STYLE[m]
            means = [agg[metric][m][li][0] for li in xs]
            stds = [agg[metric][m][li][1] for li in xs]
            ax.errorbar(xs, means, yerr=stds, marker=mk, color=col, lw=2.0, ms=8,
                        capsize=4, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L{li}" + ("\n(源)" if li == 0 else "") for li in xs])
        ax.set_xlabel("层")
        ax.set_title(title, fontsize=12.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10)

    fig.suptitle(
        f"dense_dag_r567 跨簇 DAG × 分层 TTNS(immediate 分块)  —  "
        f"{sp['n_nodes']}节点·{sp['n_layers']}层·簇{sp['clusters']}·fanin={sp['fanin']}·"
        f"{sp['n_edges']}边  |  {len(seeds)} seeds, 误差棒=std",
        fontsize=12.5, y=1.02)
    fig.tight_layout()
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
