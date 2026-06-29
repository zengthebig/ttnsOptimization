"""平方（Born machine）模型：平方树能否凭交叉项追上平方 DAG？

模型：$q(x) = f(x)^2 / Z$，$f=\\langle T, b(x)\\rangle$，$Z=\\int f^2$。Born 模型是合法
（非负）密度，用 MLE/NLL 训练最自然，且只需 $f$ 与 $Z=\\int f^2$（均已实现并验证），
无需四次型 $\\int f^4$。

在 4 层 / 20 节点、乘积交互核、**参数量对齐** 下对比 平方树(chain/balanced) vs 平方 DAG。
若平方 DAG 仍远优于平方树 → 平方救不了树，结构（缺边/不可表达高阶交互）才是根本瓶颈。
指标：验证集 NLL $=-\\mathbb{E}[\\log q]=-\\mathbb{E}[\\log f^2]+\\log Z$（越低越好）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import jax
import numpy as np
from jax import numpy as jnp, value_and_grad
import optax

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from ttde.ttns.ttns_opt import batch_eval_rank1_ttns  # noqa: E402

from simple_ttns_l2.objective import batch_basis_vectors_from_samples, integral_q2_ttns  # noqa: E402
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_graph_from_spec, sample_joint  # noqa: E402
from simple_ttns_l2.dag_ttns import dag_batch_eval_rank1, dag_quadratic_form  # noqa: E402
from simple_ttns_l2.dag_train_l2 import init_dag_from_rank1  # noqa: E402
from simple_ttns_l2.experiments.fit_deep_dag_vs_tree import (  # noqa: E402
    SPEC, SOURCES, KERNELS, _count_params, _pick_tree_rank,
)

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
EPS = 1e-8


def train_square_mle(
    params,
    batch_f_fn: Callable,   # (params, basis_batch) -> f 值 [B]
    z_fn: Callable,         # (params) -> Z = ∫f^2
    bases,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    *,
    key,
    lr: float,
    steps: int,
    batch_sz: int,
    train_noise: float,
    grad_clip: float,
    log_every: int,
    early_stop_patience: int,
    label: str,
) -> Tuple[object, Dict]:
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    state = opt.init(params)
    val_basis = batch_basis_vectors_from_samples(bases, val_x)

    def nll(p, basis_batch):
        f = batch_f_fn(p, basis_batch)
        z = z_fn(p)
        return -jnp.mean(jnp.log(f * f + EPS)) + jnp.log(jnp.abs(z) + EPS)

    @jax.jit
    def step(p, st, basis_batch):
        loss, g = value_and_grad(lambda x: nll(x, basis_batch))(p)
        u, st = opt.update(g, st, p)
        p = optax.apply_updates(p, u)
        return p, st, loss

    @jax.jit
    def eval_val(p):
        return nll(p, val_basis)

    history: List[Dict] = []
    best = float("inf")
    best_p = params
    bad = 0
    t0 = time.perf_counter()
    print(f"\n=== [{label}] square/Born MLE ===\nstep,train_nll,val_nll,total_sec", flush=True)
    for s in range(1, steps + 1):
        key, k_idx, k_noise = jax.random.split(key, 3)
        idx = jax.random.randint(k_idx, (batch_sz,), 0, train_x.shape[0])
        bx = train_x[idx]
        if train_noise > 0:
            bx = bx + jax.random.normal(k_noise, bx.shape) * train_noise
        bb = batch_basis_vectors_from_samples(bases, bx)
        params, state, loss = step(params, state, bb)
        if s % log_every == 0 or s == steps:
            vl = float(eval_val(params))
            history.append({"step": s, "train_nll": float(loss), "val_nll": vl,
                            "total_sec": time.perf_counter() - t0})
            print(f"{s},{float(loss):.6f},{vl:.6f},{time.perf_counter()-t0:.3f}", flush=True)
            if vl < best:
                best, best_p, bad = vl, params, 0
            else:
                bad += 1
                if early_stop_patience > 0 and bad >= early_stop_patience:
                    print(f"early_stop at step={s} (best_val_nll={best:.4f})", flush=True)
                    break
    return best_p, {"label": label, "final_val_nll": float(eval_val(best_p)),
                    "best_val_nll": best, "total_time_sec": time.perf_counter() - t0, "history": history}


def main():
    cfg = dict(q=2, m=10, rank=6, lr=3e-4, steps=400, batch_sz=1024, train_noise=0.05,
               n_train=40000, n_val=12000, init_noise=1e-2, grad_clip=1.0, log_every=50, seed=0)
    key = jax.random.PRNGKey(cfg["seed"])
    k_tr, k_val, k_init = jax.random.split(key, 3)
    train_x = sample_joint(SPEC, SOURCES, KERNELS, k_tr, cfg["n_train"])
    val_x = sample_joint(SPEC, SOURCES, KERNELS, k_val, cfg["n_val"])
    bases = build_bases(train_x, cfg["q"], cfg["m"])
    gram = jax.vmap(type(bases).l2_integral)(bases)

    results: List[Dict] = []

    # --- 平方 DAG（Born）---
    graph = build_graph_from_spec(SPEC, cfg["m"])
    dag0 = init_dag_from_rank1(k_init, bases, train_x, graph, cfg["rank"], cfg["init_noise"])
    budget = _count_params(dag0.cores)
    _, sd = train_square_mle(
        dag0,
        lambda p, bb: dag_batch_eval_rank1(p, graph, bb),
        lambda p: dag_quadratic_form(p, graph, gram),
        bases, train_x, val_x, key=k_init, lr=cfg["lr"], steps=cfg["steps"],
        batch_sz=cfg["batch_sz"], train_noise=cfg["train_noise"], grad_clip=cfg["grad_clip"],
        log_every=cfg["log_every"], early_stop_patience=4, label="square_dag",
    )
    sd.update({"model": "square_dag", "rank": cfg["rank"], "n_params": budget})
    results.append(sd)

    # --- 平方树（Born），参数量对齐 ---
    n = SPEC.n
    tree_parents = {"chain": [0] + list(range(0, n - 1)),
                    "balanced": __import__("ttde.score.models.opt_for_tree_data",
                                           fromlist=["balanced_parent"]).balanced_parent(n).tolist()}
    for name, parent in tree_parents.items():
        r_tree = _pick_tree_rank(bases, train_x, parent, budget)
        t0 = init_ttns_from_rank1(k_init, bases, train_x, parent, r_tree, cfg["init_noise"])
        params = _count_params(t0.cores)
        _, st = train_square_mle(
            t0,
            lambda p, bb, par=parent: batch_eval_rank1_ttns(p, bb, par),
            lambda p, par=parent: integral_q2_ttns(p, gram, par),
            bases, train_x, val_x, key=k_init, lr=cfg["lr"], steps=cfg["steps"],
            batch_sz=cfg["batch_sz"], train_noise=cfg["train_noise"], grad_clip=cfg["grad_clip"],
            log_every=cfg["log_every"], early_stop_patience=4, label=f"square_tree_{name}",
        )
        st.update({"model": f"square_tree_{name}", "rank": r_tree, "n_params": params})
        results.append(st)

    best_tree = min(r["final_val_nll"] for r in results if "tree" in r["model"])
    dag_nll = next(r["final_val_nll"] for r in results if r["model"] == "square_dag")
    gap = best_tree - dag_nll  # NLL 差（正 = DAG 更好）

    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = {"config": cfg, "results": results, "square_dag_val_nll": dag_nll,
               "best_square_tree_val_nll": best_tree, "dag_minus_tree_nll": gap}
    (REPORTS / "square_tree_vs_dag_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    lines = ["# 平方（Born）模型：平方树 vs 平方 DAG（4 层 / 20 节点，参数量对齐）", "",
             "模型 $q=f^2/Z$，MLE 训练；乘积交互核；指标验证集 NLL（越低越好）。",
             "问题：平方引入的交叉项能否让树克服结构缺陷、追上 DAG？", "",
             f"配置：`{cfg}`", "",
             "| 模型 | rank | 参数量 | final_val_nll | best_val_nll | 用时(s) |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['rank']} | {r['n_params']} | {r['final_val_nll']:.4f} | "
                     f"{r['best_val_nll']:.4f} | {r['total_time_sec']:.1f} |")
    lines += ["",
              f"平方 DAG val_nll = {dag_nll:.4f}，最优平方树 = {best_tree:.4f}，",
              f"**NLL 差 = {gap:.4f} nats**（>0 表示 DAG 仍更优；平方未能让树追上则证明瓶颈是结构）。"]
    (REPORTS / "square_tree_vs_dag_report_zh.md").write_text("\n".join(lines) + "\n")

    print("\n==== 汇总（Born/MLE, 参数量对齐）====")
    for r in results:
        print(f"{r['model']:>18s}  rank={r['rank']:>2d}  params={r['n_params']:>7d}  val_nll={r['final_val_nll']:.4f}")
    print(f"平方 DAG vs 最优平方树 NLL 差 = {gap:.4f} nats（>0=DAG 更优）")


if __name__ == "__main__":
    main()
