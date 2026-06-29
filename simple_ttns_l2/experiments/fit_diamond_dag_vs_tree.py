"""Diamond（有环 moral graph）拟合：多父 DAG TTNS vs 最优树基线。

真分布为两层 DAG：源层 $\\{x_0,x_1\\}$，下层 $\\{x_2,x_3\\}$ 各自同时依赖 $x_0,x_1$。
其无向 moral graph 含 4-环（边 0-2,1-2,0-3,1-3），任何树（4 节点至多 3 条边）都至少
缺一条必需边，必然丢失某个子节点对某个父节点的依赖。本实验验证：在结构表达力受限的
场景下，真多父 DAG TTNS 的 L2 拟合显著优于任何树拓扑（chain/balanced/star）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jax
import numpy as np
import optax
from jax import numpy as jnp, value_and_grad

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from ttde.ttns.ttns_opt import TTNSOpt  # noqa: E402

from simple_ttns_l2.objective import (  # noqa: E402
    batch_basis_vectors_from_samples,
    integral_q2_ttns,
    mc_expectation_q_ttns,
    normalize_ttns_by_integral,
)
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1, l2_loss_on_batch  # noqa: E402
from simple_ttns_l2.dag_ttns import make_dag_graph  # noqa: E402
from simple_ttns_l2.dag_train_l2 import init_dag_from_rank1, train_dag_l2  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def sample_diamond(key: jnp.ndarray, n: int) -> jnp.ndarray:
    """x0,x1 ~ U(0,1)；x2≈sum(x0,x1)，x3≈diff(x0,x1)，各依赖两个父节点。"""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    x0 = rng.uniform(0.0, 1.0, n)
    x1 = rng.uniform(0.0, 1.0, n)
    eps = 0.05
    x2 = 0.5 + 0.45 * (x0 - 0.5) + 0.45 * (x1 - 0.5) + eps * rng.standard_normal(n)
    x3 = 0.5 + 0.45 * (x0 - 0.5) - 0.45 * (x1 - 0.5) + eps * rng.standard_normal(n)
    xs = np.stack([x0, x1, x2, x3], axis=1)
    return jnp.asarray(np.clip(xs, 1e-4, 1.0 - 1e-4))


def train_tree_l2(
    ttns: TTNSOpt,
    parent: Sequence[int],
    bases,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    gram: jnp.ndarray,
    basis_integrals: jnp.ndarray,
    *,
    key: jnp.ndarray,
    lr: float,
    train_steps: int,
    batch_sz: int,
    normalize_every: int,
    log_every: int,
    label: str,
) -> Tuple[TTNSOpt, Dict]:
    optimizer = optax.adam(lr)
    ttns, z0 = normalize_ttns_by_integral(ttns, basis_integrals, parent)
    opt_state = optimizer.init(ttns)
    val_basis = batch_basis_vectors_from_samples(bases, val_x)

    @jax.jit
    def step(curr, st, batch):
        loss, g = value_and_grad(lambda x: l2_loss_on_batch(x, bases, batch, parent, gram))(curr)
        upd, st = optimizer.update(g, st, curr)
        curr = optax.apply_updates(curr, upd)
        return curr, st, loss

    @jax.jit
    def eval_val(curr):
        return integral_q2_ttns(curr, gram, parent) - 2.0 * mc_expectation_q_ttns(curr, val_basis, parent)

    history: List[Dict] = []
    best_val = float("inf")
    best = ttns
    t0 = time.perf_counter()
    print(f"\n=== [{label}] init integral={float(z0):.6e} ===", flush=True)
    print("step,train_l2,val_l2,total_sec", flush=True)
    for s in range(1, train_steps + 1):
        key, k_idx = jax.random.split(key)
        idx = jax.random.randint(k_idx, (batch_sz,), 0, train_x.shape[0])
        ttns, opt_state, loss = step(ttns, opt_state, train_x[idx])
        if normalize_every > 0 and s % normalize_every == 0:
            ttns, _ = normalize_ttns_by_integral(ttns, basis_integrals, parent)
        if s % log_every == 0 or s == train_steps:
            vl = float(eval_val(ttns))
            history.append({"step": s, "train_l2": float(loss), "val_l2": vl,
                            "total_sec": time.perf_counter() - t0})
            print(f"{s},{float(loss):.6f},{vl:.6f},{time.perf_counter()-t0:.3f}", flush=True)
            if vl < best_val:
                best_val, best = vl, ttns
    summary = {"label": label, "final_val_l2": float(eval_val(best)), "best_val_l2": best_val,
               "total_time_sec": time.perf_counter() - t0, "history": history}
    return best, summary


def main():
    cfg = dict(q=2, m=16, rank=6, lr=2e-3, steps=400, batch_sz=512,
               n_train=20000, n_val=8000, init_noise=1e-2, normalize_every=1, log_every=50, seed=0)
    key = jax.random.PRNGKey(cfg["seed"])
    k_tr, k_val, k_init = jax.random.split(key, 3)

    train_x = sample_diamond(k_tr, cfg["n_train"])
    val_x = sample_diamond(k_val, cfg["n_val"])

    bases = build_bases(train_x, cfg["q"], cfg["m"])
    gram = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    results: List[Dict] = []

    # --- 多父 DAG diamond ---
    graph = make_dag_graph(4, [cfg["m"]] * 4, [(0, 2), (1, 2), (0, 3), (1, 3)])
    dag0 = init_dag_from_rank1(k_init, bases, train_x, graph, cfg["rank"], cfg["init_noise"])
    _, dag_sum = train_dag_l2(
        dag0, graph, bases, train_x, val_x, gram, basis_integrals,
        key=k_init, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
        normalize_every=cfg["normalize_every"], log_every=cfg["log_every"], label="dag_diamond",
    )
    dag_sum["model"] = "dag_diamond"
    results.append(dag_sum)

    # --- 树基线（chain / balanced / star@2）---
    tree_parents = {
        "chain": [0, 0, 1, 2],          # 0-1-2-3
        "balanced": [0, 0, 0, 1],       # root0; children 1,2; 3 under 1
        "star@2": [2, 2, 2, 2],         # root2; 0,1,3 都是 2 的孩子（最强树：x2 见 x0,x1）
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
    dag_l2 = next(r["final_val_l2"] for r in results if r["model"] == "dag_diamond")
    improvement = (best_tree - dag_l2) / abs(best_tree) if best_tree != 0 else float("nan")

    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = {"config": cfg, "results": results,
               "dag_final_val_l2": dag_l2, "best_tree_final_val_l2": best_tree,
               "dag_vs_best_tree_rel_improvement": improvement}
    (REPORTS / "diamond_dag_vs_tree_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    lines = ["# Diamond（有环）拟合：多父 DAG TTNS vs 树基线", "",
             "真分布：两层 DAG，下层 $x_2,x_3$ 各自同时依赖 $x_0,x_1$；moral graph 含 4-环，",
             "任何树都至少缺一条必需边。指标为验证集 L2 目标 $\\int q^2-2\\,\\mathbb{E}[q]$（越低越好）。", "",
             f"配置：`{cfg}`", "",
             "| 模型 | final_val_l2 | best_val_l2 | 用时(s) |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['final_val_l2']:.6f} | {r['best_val_l2']:.6f} | {r['total_time_sec']:.1f} |")
    lines += ["",
              f"最优树 final_val_l2 = {best_tree:.6f}，DAG diamond = {dag_l2:.6f}，",
              f"**相对提升 = {improvement*100:.1f}%**（L2 越低越好）。"]
    (REPORTS / "diamond_dag_vs_tree_report_zh.md").write_text("\n".join(lines) + "\n")

    print("\n==== 汇总 ====")
    for r in results:
        print(f"{r['model']:>16s}  final_val_l2={r['final_val_l2']:.6f}")
    print(f"DAG vs 最优树 相对提升 = {improvement*100:.1f}%")


if __name__ == "__main__":
    main()
