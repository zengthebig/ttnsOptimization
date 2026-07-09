"""画 dense_dag_r567_three_way 用的那张复杂 DAG 的分层示意图。

严格用与实验同一配置重建 spec：build_clustered_spec(6, [3,3,4,4,4], fanin=3, wrap=True)。
结构：每层 18 维 = 5 个簇[3,3,4,4,4]；层间只在同簇内连边(每子节点连同簇上一层
fanin=3 个相邻父，wrap)；簇间独立 → 块可精确裂分，块内存在真实非树相关(环)。

输出：simple_ttns_l2/reports/dense_dag_r567_schematic.png
用法：env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.plot_dense_dag_schematic
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# 中文字体：macOS 自带 Arial Unicode 覆盖 CJK，否则标题/图例的中文会变方框
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

from simple_ttns_l2.dag_pipeline import build_crossed_spec  # noqa: E402

# ---- 与 dense_dag_r567_three_way.py 的 CFG 完全一致 ----
N_LAYERS = 5
CLUSTERS = [4, 4, 4, 4, 4]
FANIN = 3
CROSS_FANIN = 2
ROTATE_CROSS = True
WRAP = True

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
OUT = REPORTS / "dense_dag_r567_schematic.png"

CLUSTER_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
LAYER_GAP = 3.0          # 层间水平间距
NODE_GAP = 1.0           # 层内相邻节点垂直间距
CLUSTER_GAP = 0.9        # 簇之间额外留白


def cluster_of(within: int) -> int:
    off = 0
    for ci, csz in enumerate(CLUSTERS):
        if within < off + csz:
            return ci
        off += csz
    return len(CLUSTERS) - 1


def node_pos(node: int):
    sz = sum(CLUSTERS)
    li, within = node // sz, node % sz
    ci = cluster_of(within)
    # 层内垂直坐标：节点位置 + 前面各簇的累计留白（越上面簇号越小）
    y = within * NODE_GAP + ci * CLUSTER_GAP
    x = li * LAYER_GAP
    return x, y, li, within, ci


def main():
    spec = build_crossed_spec(N_LAYERS, CLUSTERS, fanin=FANIN, cross_pairs=None,
                              cross_fanin=CROSS_FANIN, rotate_cross=ROTATE_CROSS, wrap=WRAP)
    sz = sum(CLUSTERS)
    y_span = (sz - 1) * NODE_GAP + (len(CLUSTERS) - 1) * CLUSTER_GAP

    fig, ax = plt.subplots(figsize=(13, 8.5))

    # 边：簇内边按父簇淡色；跨簇边(u,v 不同簇)用深红粗线突出
    n_cross = 0
    for (u, v) in spec.edges:
        xu, yu, _, _, cu = node_pos(u)
        xv, yv, _, _, cv = node_pos(v)
        if cu != cv:
            n_cross += 1
            ax.plot([xu, xv], [y_span - yu, y_span - yv],
                    color="#C1272D", alpha=0.75, lw=1.6, zorder=2)
        else:
            ax.plot([xu, xv], [y_span - yu, y_span - yv],
                    color=CLUSTER_COLORS[cu], alpha=0.25, lw=0.9, zorder=1)

    # 节点
    for node in range(spec.n):
        x, y, li, within, ci = node_pos(node)
        ax.scatter([x], [y_span - y], s=190, color=CLUSTER_COLORS[ci],
                   edgecolors="white", linewidths=1.1, zorder=3)

    # immediate 分块框：每层(li>=1)按"只看上一层直接父"分块，框出每个 TTNS 块
    from simple_ttns_l2.analytic_tree_fit import structural_blocks
    for li in range(1, N_LAYERS):
        blocks = structural_blocks(spec, li, mode="immediate")
        layer_nodes = list(spec.layers[li])
        for blk in blocks:
            ys = [y_span - node_pos(layer_nodes[i])[1] for i in blk]
            x = li * LAYER_GAP
            y0, y1 = min(ys), max(ys)
            box = FancyBboxPatch(
                (x - 0.42, y0 - 0.42), 0.84, (y1 - y0) + 0.84,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                fill=False, edgecolor="#333333", ls=(0, (4, 2)), lw=1.3, zorder=2.5,
            )
            ax.add_patch(box)
            if len(blk) >= 6:  # 只给合并的大块标 size，避免刷屏
                ax.text(x + 0.5, (y0 + y1) / 2, f"K={len(blk)}", fontsize=8.5,
                        color="#333333", va="center", ha="left")

    # 层标签
    for li in range(N_LAYERS):
        ax.text(li * LAYER_GAP, y_span + 1.4,
                f"L{li}" + ("\n(源层)" if li == 0 else ""),
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    # 图例（簇 + 跨簇边）
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLUSTER_COLORS[ci],
                      markersize=11, label=f"簇{ci} (size {CLUSTERS[ci]})")
               for ci in range(len(CLUSTERS))]
    handles.append(Line2D([0], [0], color="#C1272D", lw=2.2, label="跨簇边(逐层旋转桥接)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=len(CLUSTERS) + 1, frameon=False, fontsize=10)

    n_edges = len(spec.edges)
    ax.set_title(
        f"dense_dag_r567 跨簇 DAG：{spec.n} 节点 · {N_LAYERS} 层 · {sz} 维/层 · "
        f"簇{CLUSTERS} · fanin={FANIN}(簇内密连) · {n_edges} 边(含 {n_cross} 跨簇)\n"
        f"跨簇边逐层旋转桥接 (L{{i}}: 簇(i-1)→i, cross_fanin={CROSS_FANIN})；immediate 分块每层块"
        f"恒 {{一对合并=8, 其余=4}} 有界，而追祖先(source)会随深度滚成整层→退回全局",
        fontsize=12.0)
    ax.axis("off")
    ax.set_xlim(-1.2, (N_LAYERS - 1) * LAYER_GAP + 1.2)
    ax.set_ylim(-2.0, y_span + 3.0)
    fig.tight_layout()
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved: {OUT}")
    print(f"spec: n={spec.n} layers={N_LAYERS} per_layer={sz} clusters={CLUSTERS} "
          f"fanin={FANIN} cross_fanin={CROSS_FANIN} edges={n_edges} cross_edges={n_cross}")


if __name__ == "__main__":
    main()
