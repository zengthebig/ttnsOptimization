"""多层 DAG 传播 pipeline 端到端 demo：3 层多父 DAG vs 树基线。

结构（5 节点，6 条层间边）：
- L1={0,1}（源，U(0,1)）
- L2={2,3}：2←(0,1) 取"和"，3←(0,1) 取"差"
- L3={4}：4←(2,3) 取"和"
moral graph 含环且边数(6) > 树上限(4)，任何树至少缺 2 条必需边。验证多父 DAG TTNS
在端到端逐层传播 + 拟合下显著优于最优树。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from ttde.score.models.opt_for_tree_data import balanced_parent  # noqa: E402

from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.dag_pipeline import MultiLayerSpec, build_graph_from_spec, sample_joint  # noqa: E402
from simple_ttns_l2.dag_train_l2 import init_dag_from_rank1, train_dag_l2  # noqa: E402
from simple_ttns_l2.experiments.fit_diamond_dag_vs_tree import train_tree_l2  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"

SPEC = MultiLayerSpec(
    layers=((0, 1), (2, 3), (4,)),
    edges=((0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4)),
)


def _uniform_source(rng, n):
    return rng.uniform(0.0, 1.0, n)


def _kernel_sum(parent_vals, rng, n):
    return 0.5 + 0.45 * (parent_vals[:, 0] - 0.5) + 0.45 * (parent_vals[:, 1] - 0.5) + 0.05 * rng.standard_normal(n)


def _kernel_diff(parent_vals, rng, n):
    return 0.5 + 0.45 * (parent_vals[:, 0] - 0.5) - 0.45 * (parent_vals[:, 1] - 0.5) + 0.05 * rng.standard_normal(n)


SOURCES = {0: _uniform_source, 1: _uniform_source}
KERNELS = {2: _kernel_sum, 3: _kernel_diff, 4: _kernel_sum}


def main():
    cfg = dict(q=2, m=16, rank=6, lr=2e-3, steps=400, batch_sz=512,
               n_train=30000, n_val=10000, init_noise=1e-2, normalize_every=1, log_every=50, seed=0)
    key = jax.random.PRNGKey(cfg["seed"])
    k_tr, k_val, k_init = jax.random.split(key, 3)

    train_x = sample_joint(SPEC, SOURCES, KERNELS, k_tr, cfg["n_train"])
    val_x = sample_joint(SPEC, SOURCES, KERNELS, k_val, cfg["n_val"])

    bases = build_bases(train_x, cfg["q"], cfg["m"])
    gram = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    results: List[Dict] = []

    # --- 多父 DAG（正确多层结构）---
    graph = build_graph_from_spec(SPEC, cfg["m"])
    dag0 = init_dag_from_rank1(k_init, bases, train_x, graph, cfg["rank"], cfg["init_noise"])
    _, dag_sum = train_dag_l2(
        dag0, graph, bases, train_x, val_x, gram, basis_integrals,
        key=k_init, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
        normalize_every=cfg["normalize_every"], log_every=cfg["log_every"], label="dag_multilayer",
    )
    dag_sum["model"] = "dag_multilayer"
    results.append(dag_sum)

    # --- 树基线：chain / balanced / 若干手工 hub 树，取最优 ---
    tree_parents = {
        "chain": [0, 0, 1, 2, 3],
        "balanced": balanced_parent(5).tolist(),
        "hub@2": [2, 2, 2, 2, 2],          # 以 L2 的 x2 为根
        "spine": [2, 3, 4, 4, 4],          # 0-2,1-3,2-4,3-4：尽量贴近 DAG（仍缺 0-3,1-2）
    }
    for name, parent in tree_parents.items():
        t0 = init_ttns_from_rank1(k_init, bases, train_x, parent, cfg["rank"], cfg["init_noise"])
        _, summ = train_tree_l2(
            t0, parent, bases, train_x, val_x, gram, basis_integrals,
            key=k_init, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
            normalize_every=cfg["normalize_every"], log_every=cfg["log_every"], label=f"tree_{name}",
        )
        summ["model"] = f"tree_{name}"
        results.append(summ)

    best_tree = min(r["final_val_l2"] for r in results if r["model"].startswith("tree_"))
    dag_l2 = next(r["final_val_l2"] for r in results if r["model"] == "dag_multilayer")
    improvement = (best_tree - dag_l2) / abs(best_tree) if best_tree != 0 else float("nan")

    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = {"config": cfg, "spec": {"layers": SPEC.layers, "edges": SPEC.edges},
               "results": results, "dag_final_val_l2": dag_l2, "best_tree_final_val_l2": best_tree,
               "dag_vs_best_tree_rel_improvement": improvement}
    (REPORTS / "multilayer_dag_vs_tree_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    lines = ["# 多层 DAG 传播 pipeline：多父 DAG TTNS vs 树基线", "",
             "结构：L1={0,1} 源；L2={2,3} 各依赖 (0,1)（和/差）；L3={4} 依赖 (2,3)。",
             "传播按拓扑序逐层采样（源 → delay 核 → 下层），全联合用多父 DAGTTNS 拟合。",
             "moral graph 含环且边数(6)>树上限(4)，任何树至少缺 2 条必需边。",
             "指标：验证集 L2 目标 $\\int q^2-2\\,\\mathbb{E}[q]$（越低越好）。", "",
             f"配置：`{cfg}`", "",
             "| 模型 | final_val_l2 | best_val_l2 | 用时(s) |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['final_val_l2']:.6f} | {r['best_val_l2']:.6f} | {r['total_time_sec']:.1f} |")
    lines += ["",
              f"最优树 final_val_l2 = {best_tree:.6f}，DAG multilayer = {dag_l2:.6f}，",
              f"**相对提升 = {improvement*100:.1f}%**（L2 越低越好）。"]
    (REPORTS / "multilayer_dag_vs_tree_report_zh.md").write_text("\n".join(lines) + "\n")

    print("\n==== 汇总 ====")
    for r in results:
        print(f"{r['model']:>18s}  final_val_l2={r['final_val_l2']:.6f}")
    print(f"DAG vs 最优树 相对提升 = {improvement*100:.1f}%")


if __name__ == "__main__":
    main()
