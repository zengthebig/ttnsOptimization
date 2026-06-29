"""只改一处：把 DAG TTNS 的每个 core 参数 $x\\to x^2$（非负参数化）。

B-spline 基处处非负，若 core 元素也非负，则 $q=\\langle T,b\\rangle$ 处处 $\\ge0$ →
天然合法密度（负值占比必为 0），归一化只需一阶积分 $\\int q$（最便宜），且 **DAG 也能
直接 MLE**——绕过"DAG 没法平方($\\int f^4$)"的死结。

代价：非负参数 + 非负 basis = 非负张量网络（≈离散隐变量 PGM），失去"符号相消"的
表达力。本实验测在同一例子（4 层/20 节点乘积核）上这个权衡的净效果。

对比（同结构、同 rank，无需参数对齐）：
- dag_linear_l2  ：基线，线性 + L2（可负，有 nonpositive_rate）
- dag_nonneg_l2  ：**只改参数平方**，仍 L2 训练
- dag_nonneg_mle ：参数平方 + 顺势 MLE（现在合法了）
共同尺子：held-out 平均 LL（越高越好）+ 负值占比。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import jax
import optax
from jax import numpy as jnp, value_and_grad

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.objective import batch_basis_vectors_from_samples  # noqa: E402
from simple_ttns_l2.train_l2 import build_bases  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_graph_from_spec, sample_joint  # noqa: E402
from simple_ttns_l2.dag_ttns import (  # noqa: E402
    DAGTTNS,
    dag_batch_eval_rank1,
    dag_integral,
    dag_quadratic_form,
)
from simple_ttns_l2.dag_train_l2 import init_dag_from_rank1, train_dag_l2  # noqa: E402
from simple_ttns_l2.experiments.fit_deep_dag_vs_tree import SPEC, SOURCES, KERNELS, _count_params  # noqa: E402
from simple_ttns_l2.experiments.fit_ttde_vs_dag_propagation import held_out_ll  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
EPS = 1e-12


def _square(ttns: DAGTTNS) -> DAGTTNS:
    """非负参数化：每个 core 元素 $x\\to x^2$。"""
    return DAGTTNS(tuple(c * c for c in ttns.cores))


def _normalize_nonneg(ttns: DAGTTNS, graph, basis_integrals, root: int = 0, eps: float = 1e-12) -> DAGTTNS:
    r"""把 $\int q=\int\langle\text{square}(T),b\rangle$ 归一化到 1：缩放 root core（平方后 $\propto a^2$，故乘 $1/\sqrt z$）。

    与基线线性 DAG 的每步归一化口径一致，保证"只改参数平方"这一处差异。
    """
    z = dag_integral(_square(ttns), graph, basis_integrals)
    safe = jnp.where(jnp.abs(z) < eps, 1.0, z)
    a = 1.0 / jnp.sqrt(jnp.abs(safe))
    cores = list(ttns.cores)
    cores[root] = cores[root] * a
    return DAGTTNS(tuple(cores))


def train_dag_reparam(
    dag0: DAGTTNS, graph, bases, gram, basis_integrals, train_x, val_x,
    *, mode: str, key, lr: float, steps: int, batch_sz: int, train_noise: float,
    grad_clip: float, lr_decay: bool, log_every: int, early_stop_patience: int, label: str,
):
    """对非负参数 DAG（cores²）做 L2 或 MLE 训练。"""
    lr_arg = optax.cosine_decay_schedule(lr, steps) if lr_decay else lr
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_arg))
    state = opt.init(dag0)
    val_basis = batch_basis_vectors_from_samples(bases, val_x)

    def l2_loss(p, bb):
        s = _square(p)
        return dag_quadratic_form(s, graph, gram) - 2.0 * dag_batch_eval_rank1(s, graph, bb).mean()

    def mle_loss(p, bb):
        s = _square(p)
        q = dag_batch_eval_rank1(s, graph, bb)
        z = dag_integral(s, graph, basis_integrals)
        return -jnp.mean(jnp.log(q + EPS)) + jnp.log(jnp.abs(z) + EPS)

    loss_fn: Callable = l2_loss if mode == "l2" else mle_loss

    @jax.jit
    def step(p, st, bb):
        loss, g = value_and_grad(lambda x: loss_fn(x, bb))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, loss

    @jax.jit
    def eval_val(p):
        return loss_fn(p, val_basis)

    best, best_p, bad = float("inf"), dag0, 0
    params = dag0
    t0 = time.perf_counter()
    print(f"\n=== [{label}] nonneg-param DAG ({mode}) ===\nstep,train_loss,val_loss,total_sec", flush=True)
    for s_i in range(1, steps + 1):
        key, k_idx, k_noise = jax.random.split(key, 3)
        idx = jax.random.randint(k_idx, (batch_sz,), 0, train_x.shape[0])
        bx = train_x[idx]
        if train_noise > 0:
            bx = bx + jax.random.normal(k_noise, bx.shape) * train_noise
        bb = batch_basis_vectors_from_samples(bases, bx)
        params, state, loss = step(params, state, bb)
        if mode == "l2":  # 与基线线性 DAG 一致：每步归一化（MLE 目标自带尺度不变性，无需）
            params = _normalize_nonneg(params, graph, basis_integrals)
        if s_i % log_every == 0 or s_i == steps:
            vl = float(eval_val(params))
            print(f"{s_i},{float(loss):.6f},{vl:.6f},{time.perf_counter()-t0:.3f}", flush=True)
            if vl < best:
                best, best_p, bad = vl, params, 0
            else:
                bad += 1
                if early_stop_patience > 0 and bad >= early_stop_patience:
                    print(f"early_stop at step={s_i} (best_val={best:.4f})", flush=True)
                    break
    return best_p, {"final_val_loss": float(eval_val(best_p)), "best_val_loss": best,
                    "total_time_sec": time.perf_counter() - t0}


def main():
    cfg = dict(q=2, m=10, rank=6, n_train=40000, n_val=12000, init_noise=1e-2,
               batch_sz=1024, train_noise=0.05, log_every=50, seed=0,
               lr=5e-4, steps=600, grad_clip=1.0, lr_decay=True, early_stop=4)
    for k in ("n_train", "n_val", "steps", "log_every"):
        if (env := os.environ.get(f"NN_DAG_{k.upper()}")) is not None:
            cfg[k] = int(env)
    key = jax.random.PRNGKey(cfg["seed"])
    k_tr, k_val, k_init = jax.random.split(key, 3)

    train_x = sample_joint(SPEC, SOURCES, KERNELS, k_tr, cfg["n_train"])
    val_x = sample_joint(SPEC, SOURCES, KERNELS, k_val, cfg["n_val"])
    bases = build_bases(train_x, cfg["q"], cfg["m"])
    gram = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)
    val_basis = batch_basis_vectors_from_samples(bases, val_x)

    graph = build_graph_from_spec(SPEC, cfg["m"])
    dag0 = init_dag_from_rank1(k_init, bases, train_x, graph, cfg["rank"], cfg["init_noise"])
    n_params = _count_params(dag0.cores)
    results: List[Dict] = []

    # 1) 基线：线性 DAG + L2
    base_best, base_sum = train_dag_l2(
        dag0, graph, bases, train_x, val_x, gram, basis_integrals,
        key=k_init, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
        normalize_every=1, log_every=cfg["log_every"], train_noise=cfg["train_noise"],
        lr_decay_steps=cfg["steps"] if cfg["lr_decay"] else 0, label="dag_linear_l2",
    )
    bq = dag_batch_eval_rank1(base_best, graph, val_basis)
    bz = dag_integral(base_best, graph, basis_integrals)
    bll, bnp = held_out_ll(bq, bz)
    results.append({"model": "dag_linear_l2", "variant": "线性+L2",
                    "n_params": n_params, "val_ll": bll, "nonpositive_rate": bnp,
                    "total_time_sec": base_sum["total_time_sec"]})

    # 2) 非负参数 DAG + L2（只改参数平方）  3) 非负参数 DAG + MLE
    for mode, tag in (("l2", "参数平方+L2"), ("mle", "参数平方+MLE")):
        best_p, summ = train_dag_reparam(
            dag0, graph, bases, gram, basis_integrals, train_x, val_x,
            mode=mode, key=k_init, lr=cfg["lr"], steps=cfg["steps"], batch_sz=cfg["batch_sz"],
            train_noise=cfg["train_noise"], grad_clip=cfg["grad_clip"], lr_decay=cfg["lr_decay"],
            log_every=cfg["log_every"], early_stop_patience=cfg["early_stop"],
            label=f"dag_nonneg_{mode}",
        )
        s = _square(best_p)
        q = dag_batch_eval_rank1(s, graph, val_basis)
        z = dag_integral(s, graph, basis_integrals)
        ll, npr = held_out_ll(q, z)
        results.append({"model": f"dag_nonneg_{mode}", "variant": tag,
                        "n_params": n_params, "val_ll": ll, "nonpositive_rate": npr,
                        "total_time_sec": summ["total_time_sec"]})

    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = {"config": cfg, "common_metric": "held_out_mean_ll (higher=better)", "results": results}
    (REPORTS / "nonneg_param_dag_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    lines = ["# 非负参数化（每个 core 参数 $x\\to x^2$）对 DAG TTNS 的影响", "",
             "同一例子（4 层/20 节点 banded 多父、乘积交互核），同结构同 rank。", "",
             "B-spline 基处处非负 + core 参数非负 → $q\\ge0$ 处处成立 → 合法密度、负值占比为 0、",
             "归一化只需一阶 $\\int q$，DAG 可直接 MLE。代价：非负张量网络失去符号相消的表达力。", "",
             "共同尺子：held-out 平均 LL（越高越好）；负值占比越低越好。", "",
             f"配置：`{cfg}`", "",
             "| 模型 | 变体 | 参数量 | val_LL↑ | 负值占比 | 用时(s) |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['variant']} | {r['n_params']} | {r['val_ll']:.4f} | "
                     f"{r['nonpositive_rate']:.4f} | {r['total_time_sec']:.1f} |")
    (REPORTS / "nonneg_param_dag_report_zh.md").write_text("\n".join(lines) + "\n")

    print("\n==== 汇总：非负参数化 DAG ====")
    for r in results:
        print(f"{r['model']:>16s}  {r['variant']:<14s}  val_LL={r['val_ll']:.4f}  "
              f"nonpos={r['nonpositive_rate']:.4f}")


if __name__ == "__main__":
    main()
