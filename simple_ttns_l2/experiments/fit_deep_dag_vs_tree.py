"""复杂多层 DAG（4 层 / 20 节点）端到端：多父 DAG TTNS vs 树基线。

结构由 `build_layered_spec([5,5,5,5], fanin=2, wrap=False)` 生成：banded 多父阶梯——
每个下层节点连上层 2 个相邻父，相邻下层节点共享父 → 产生大量 4-环（树不可表达），
同时保持局部连接、treewidth ~2-3（收缩可行、训练秒级）。

20 节点 / 27 条层间边；树上限 19 条边 → 任何树至少缺 8 条必需边。验证在更深更宽、
更复杂的依赖结构下，多父 DAG 的优势是否随复杂度进一步放大。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from ttde.score.models.opt_for_tree_data import balanced_parent  # noqa: E402

from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.dag_pipeline import (  # noqa: E402
    build_graph_from_spec,
    build_layered_spec,
    sample_joint,
)
from simple_ttns_l2.dag_train_l2 import init_dag_from_rank1, train_dag_l2  # noqa: E402
from simple_ttns_l2.experiments.fit_diamond_dag_vs_tree import train_tree_l2  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"

SPEC = build_layered_spec([5, 5, 5, 5], fanin=2, wrap=False)


def _src(rng, n):
    return rng.uniform(0.0, 1.0, n)


def _kernel(parent_vals, rng, n):
    """child = 0.5 + 0.40·[2^k·Π_i(parent_i-0.5)] + 0.10·噪声 —— 父节点的**乘积交互**。

    乘积项使 child 与单个父几乎零相关（XOR 式），只有同时连上所有父、且能表达高阶交互
    的模型才能拟合。树仅靠成对路径无法捕捉，这正是多父 DAG 相对树的本质优势所在。
    """
    k = parent_vals.shape[1]
    prod = np.prod(parent_vals - 0.5, axis=1) * (2.0 ** k)
    return 0.5 + 0.40 * prod + 0.10 * rng.standard_normal(n)


SOURCES = {v: _src for v in SPEC.layers[0]}
KERNELS = {v: _kernel for layer in SPEC.layers[1:] for v in layer}


def main():
    cfg = dict(q=2, m=10, rank=6, lr=5e-4, steps=600, batch_sz=1024, train_noise=0.05,
               n_train=40000, n_val=12000, init_noise=1e-2, normalize_every=1, log_every=50, seed=0)
    key = jax.random.PRNGKey(cfg["seed"])
    k_tr, k_val, k_init = jax.random.split(key, 3)

    train_x = sample_joint(SPEC, SOURCES, KERNELS, k_tr, cfg["n_train"])
    val_x = sample_joint(SPEC, SOURCES, KERNELS, k_val, cfg["n_val"])

    bases = build_bases(train_x, cfg["q"], cfg["m"])
    gram = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    n = SPEC.n
    results: List[Dict] = []

    # --- 多父 DAG（正确多层 banded 结构）---
    graph = build_graph_from_spec(SPEC, cfg["m"])
    dag0 = init_dag_from_rank1(k_init, bases, train_x, graph, cfg["rank"], cfg["init_noise"])
    _, dag_sum = train_dag_l2(
        dag0, graph, bases, train_x, val_x, gram, basis_integrals,
        key=k_init, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
        normalize_every=cfg["normalize_every"], log_every=cfg["log_every"],
        train_noise=cfg["train_noise"], label="dag_deep",
    )
    dag_sum["model"] = "dag_deep"
    results.append(dag_sum)

    # --- 树基线：chain / balanced（取最优）---
    tree_parents = {
        "chain": [0] + list(range(0, n - 1)),
        "balanced": balanced_parent(n).tolist(),
    }
    for name, parent in tree_parents.items():
        t0 = init_ttns_from_rank1(k_init, bases, train_x, parent, cfg["rank"], cfg["init_noise"])
        _, summ = train_tree_l2(
            t0, parent, bases, train_x, val_x, gram, basis_integrals,
            key=k_init, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
            normalize_every=cfg["normalize_every"], log_every=cfg["log_every"],
            train_noise=cfg["train_noise"], label=f"tree_{name}",
        )
        summ["model"] = f"tree_{name}"
        results.append(summ)

    best_tree = min(r["final_val_l2"] for r in results if r["model"].startswith("tree_"))
    dag_l2 = next(r["final_val_l2"] for r in results if r["model"] == "dag_deep")
    improvement = (best_tree - dag_l2) / abs(best_tree) if best_tree != 0 else float("nan")

    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = {"config": cfg, "spec": {"layers": SPEC.layers, "edges": SPEC.edges, "n_edges": len(SPEC.edges)},
               "results": results, "dag_final_val_l2": dag_l2, "best_tree_final_val_l2": best_tree,
               "dag_vs_best_tree_rel_improvement": improvement}
    (REPORTS / "deep_dag_vs_tree_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    lines = ["# 复杂多层 DAG（4 层 / 20 节点）：多父 DAG TTNS vs 树基线", "",
             "结构：banded 多父阶梯 `[5,5,5,5]`，每下层节点连上层 2 个相邻父，相邻下层节点共享父",
             f"→ {len(SPEC.edges)} 条层间边、大量 4-环。树上限 {SPEC.n - 1} 条边 → 任何树至少缺 "
             f"{len(SPEC.edges) - (SPEC.n - 1)} 条必需边。",
             "传播按拓扑序逐层采样（源 → delay 核 → 下层），全联合用多父 DAGTTNS 拟合。",
             "指标：验证集 L2 目标 $\\int q^2-2\\,\\mathbb{E}[q]$（越低越好）。", "",
             f"配置：`{cfg}`", "",
             "| 模型 | final_val_l2 | best_val_l2 | 用时(s) |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['final_val_l2']:.4f} | {r['best_val_l2']:.4f} | {r['total_time_sec']:.1f} |")
    lines += ["",
              f"最优树 final_val_l2 = {best_tree:.4f}，DAG deep = {dag_l2:.4f}，",
              f"**相对提升 = {improvement*100:.1f}%**（L2 越低越好）。"]
    (REPORTS / "deep_dag_vs_tree_report_zh.md").write_text("\n".join(lines) + "\n")

    print("\n==== 汇总 ====")
    for r in results:
        print(f"{r['model']:>14s}  final_val_l2={r['final_val_l2']:.4f}  用时={r['total_time_sec']:.1f}s")
    print(f"DAG vs 最优树 相对提升 = {improvement*100:.1f}%")


if __name__ == "__main__":
    main()
