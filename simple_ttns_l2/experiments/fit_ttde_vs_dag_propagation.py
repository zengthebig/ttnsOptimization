"""TTDE 做法 vs 我们 DAG 传播做法：同一个例子、同一把尺子。

- **TTDE 做法**：平方（Born）树 $q=f^2/Z$，MLE/NLL 训练（原版 TTDE 的密度方案）。
- **我们 DAG 传播做法**：线性多父 DAG $q=\\langle T,b\\rangle$，L2 目标训练。

两者原生指标不同（NLL vs L2），无法直接并排。这里统一用 **held-out 平均对数似然 LL**
（越高越好）做共同尺子：把每个模型都看成密度 $q_{\\text{norm}}=v/Z$ 后在验证集上算 $\\mathbb{E}[\\log q_{\\text{norm}}]$。

- 平方树：$v=f^2\\ge0$，$Z=\\int f^2$ → 恒为合法密度，`nonpositive_rate=0`。
- 线性 DAG：$v=\\langle T,b\\rangle$（可正可负），$Z=\\int q$ → 在 $v>0$ 的点上算 LL，
  并如实报告 `nonpositive_rate`（负值占比）。DAG 不平方，保留线性语义。

数据：4 层 / 20 节点、banded 多父、乘积交互核（复用 fit_deep_dag_vs_tree 的 SPEC/核）。
参数量对齐：以线性 DAG 参数量为预算，给平方树取 ≤ 预算的最大 rank。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

from ttde.score.models.opt_for_tree_data import balanced_parent  # noqa: E402
from ttde.ttns.ttns_opt import TTNSOpt, batch_eval_rank1_ttns  # noqa: E402

from simple_ttns_l2.objective import (  # noqa: E402
    batch_basis_vectors_from_samples,
    integral_q2_ttns,
)
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_graph_from_spec, sample_joint  # noqa: E402
from simple_ttns_l2.dag_ttns import (  # noqa: E402
    dag_batch_eval_rank1,
    dag_integral,
)
from simple_ttns_l2.dag_train_l2 import init_dag_from_rank1, train_dag_l2  # noqa: E402
from simple_ttns_l2.experiments.fit_deep_dag_vs_tree import (  # noqa: E402
    SPEC, SOURCES, KERNELS, _count_params, _pick_tree_rank,
)
from simple_ttns_l2.experiments.fit_square_tree_vs_dag import train_square_mle  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
EPS = 1e-12


def held_out_ll(v_vals: jnp.ndarray, z: jnp.ndarray) -> Tuple[float, float]:
    r"""把模型看成密度 $v/Z$，在 $v>0$ 的点上算平均 $\log(v/Z)$，并返回非正值占比。

    - 平方树：传入 $v=f^2$、$Z=\int f^2$ → 几乎处处 $v>0$，nonpositive_rate≈0。
    - 线性 DAG：传入 $v=q$、$Z=\int q$ → 负值点无法定义 $\log$，按 nonpositive_rate 统计。
    """
    v = np.asarray(v_vals, dtype=np.float64)
    z = float(z)
    pos = v > 0
    nonpos_rate = float(1.0 - pos.mean())
    if pos.sum() == 0 or z <= 0:
        return float("nan"), nonpos_rate
    ll = np.log(v[pos]) - np.log(z)
    return float(ll.mean()), nonpos_rate


# ---------------- 完整 TTDE：多排列混合平方树 ----------------
# q(x) ∝ Σ_k <T_k, b(x)[π_k]>^2，Z = Σ_k ∫<T_k, b[π_k]>^2；MLE/NLL 训练（原版 TTDE 方案）。

def init_mixture(key, bases, train_x, parent, rank: int, n_comp: int, noise: float):
    """K 个同拓扑平方 TT 分量 + K 个变量排列（首个为恒等，其余随机）。"""
    n = len(parent)
    k_trees, k_perm = jax.random.split(key)
    tree_keys = jax.random.split(k_trees, n_comp)
    trees = [init_ttns_from_rank1(tk, bases, train_x, parent, rank, noise) for tk in tree_keys]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)
    perm_keys = jax.random.split(k_perm, max(n_comp - 1, 1))
    perms = [jnp.arange(n)] + [jax.random.permutation(perm_keys[i], n) for i in range(n_comp - 1)]
    perms = jnp.stack(perms)
    return stacked, perms


def mixture_unnorm_q(stacked, perms, parent, basis_batch):
    r"""返回每个样本的 $\sum_k f_k(x)^2$（恒非负），形状 `[B]`。"""
    def one(core_tuple, perm):
        tk = TTNSOpt(tuple(core_tuple))
        return batch_eval_rank1_ttns(tk, basis_batch[:, perm, :], parent)
    fs = jax.vmap(one)(stacked.cores, perms)  # [K, B]
    return jnp.sum(fs * fs, axis=0)


def mixture_partition(stacked, perms, parent, gram):
    r"""$Z = \sum_k \int f_k^2$。"""
    def one_z(core_tuple, perm):
        tk = TTNSOpt(tuple(core_tuple))
        return integral_q2_ttns(tk, gram[perm], parent)
    return jnp.sum(jax.vmap(one_z)(stacked.cores, perms))


def train_square_mixture_mle(stacked, perms, parent, bases, gram, train_x, val_x, *,
                             key, lr, steps, batch_sz, train_noise, grad_clip,
                             log_every, early_stop_patience, label):
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    state = opt.init(stacked)
    val_basis = batch_basis_vectors_from_samples(bases, val_x)

    def nll(s, basis_batch):
        q = mixture_unnorm_q(s, perms, parent, basis_batch)
        z = mixture_partition(s, perms, parent, gram)
        return -jnp.mean(jnp.log(q + EPS)) + jnp.log(jnp.abs(z) + EPS)

    @jax.jit
    def step(s, st, basis_batch):
        loss, g = value_and_grad(lambda x: nll(x, basis_batch))(s)
        u, st = opt.update(g, st, s)
        return optax.apply_updates(s, u), st, loss

    @jax.jit
    def eval_val(s):
        return nll(s, val_basis)

    best, best_s, bad = float("inf"), stacked, 0
    t0 = time.perf_counter()
    print(f"\n=== [{label}] mixture(K={perms.shape[0]}) square/Born MLE ===\nstep,train_nll,val_nll,total_sec", flush=True)
    for s_i in range(1, steps + 1):
        key, k_idx, k_noise = jax.random.split(key, 3)
        idx = jax.random.randint(k_idx, (batch_sz,), 0, train_x.shape[0])
        bx = train_x[idx]
        if train_noise > 0:
            bx = bx + jax.random.normal(k_noise, bx.shape) * train_noise
        bb = batch_basis_vectors_from_samples(bases, bx)
        stacked, state, loss = step(stacked, state, bb)
        if s_i % log_every == 0 or s_i == steps:
            vl = float(eval_val(stacked))
            print(f"{s_i},{float(loss):.6f},{vl:.6f},{time.perf_counter()-t0:.3f}", flush=True)
            if vl < best:
                best, best_s, bad = vl, stacked, 0
            else:
                bad += 1
                if early_stop_patience > 0 and bad >= early_stop_patience:
                    print(f"early_stop at step={s_i} (best_val_nll={best:.4f})", flush=True)
                    break
    return best_s, {"final_val_nll": float(eval_val(best_s)), "best_val_nll": best,
                    "total_time_sec": time.perf_counter() - t0}


def main():
    cfg = dict(q=2, m=10, rank=6, n_train=40000, n_val=12000, init_noise=1e-2,
               batch_sz=1024, train_noise=0.05, log_every=50, seed=0,
               dag_lr=5e-4, dag_steps=600, dag_normalize_every=1, dag_lr_decay=True,
               tree_lr=3e-4, tree_steps=400, tree_grad_clip=1.0, tree_early_stop=4,
               mix_n_comp=8, mix_lr=3e-4, mix_steps=400)
    for k in ("n_train", "n_val", "dag_steps", "tree_steps", "mix_steps", "log_every"):
        if (env := os.environ.get(f"TTDE_DAG_{k.upper()}")) is not None:
            cfg[k] = int(env)
    key = jax.random.PRNGKey(cfg["seed"])
    k_tr, k_val, k_init = jax.random.split(key, 3)

    train_x = sample_joint(SPEC, SOURCES, KERNELS, k_tr, cfg["n_train"])
    val_x = sample_joint(SPEC, SOURCES, KERNELS, k_val, cfg["n_val"])

    bases = build_bases(train_x, cfg["q"], cfg["m"])
    gram = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)
    val_basis = batch_basis_vectors_from_samples(bases, val_x)

    n = SPEC.n
    results: List[Dict] = []

    # ---- 我们的做法：线性多父 DAG（L2 目标） ----
    graph = build_graph_from_spec(SPEC, cfg["m"])
    dag0 = init_dag_from_rank1(k_init, bases, train_x, graph, cfg["rank"], cfg["init_noise"])
    budget = _count_params(dag0.cores)
    dag_best, dag_sum = train_dag_l2(
        dag0, graph, bases, train_x, val_x, gram, basis_integrals,
        key=k_init, lr=cfg["dag_lr"], train_steps=cfg["dag_steps"], batch_sz=cfg["batch_sz"],
        normalize_every=cfg["dag_normalize_every"], log_every=cfg["log_every"],
        train_noise=cfg["train_noise"],
        lr_decay_steps=cfg["dag_steps"] if cfg["dag_lr_decay"] else 0,
        label="dag_linear_l2",
    )
    dag_q = dag_batch_eval_rank1(dag_best, graph, val_basis)
    dag_z = dag_integral(dag_best, graph, basis_integrals)
    dag_ll, dag_nonpos = held_out_ll(dag_q, dag_z)
    results.append({
        "model": "dag_linear_l2", "method": "我们DAG传播(线性+L2)",
        "rank": cfg["rank"], "n_params": budget,
        "native_metric": "val_l2", "native_value": dag_sum["final_val_l2"],
        "val_ll": dag_ll, "nonpositive_rate": dag_nonpos,
        "total_time_sec": dag_sum["total_time_sec"],
    })

    # ---- TTDE 做法：平方（Born）树（MLE），参数量对齐 ----
    tree_parents = {
        "chain": [0] + list(range(0, n - 1)),
        "balanced": balanced_parent(n).tolist(),
    }
    for name, parent in tree_parents.items():
        r_tree = _pick_tree_rank(bases, train_x, parent, budget)
        t0 = init_ttns_from_rank1(k_init, bases, train_x, parent, r_tree, cfg["init_noise"])
        params = _count_params(t0.cores)
        tree_best, tree_sum = train_square_mle(
            t0,
            lambda p, bb, par=parent: batch_eval_rank1_ttns(p, bb, par),
            lambda p, par=parent: integral_q2_ttns(p, gram, par),
            bases, train_x, val_x, key=k_init, lr=cfg["tree_lr"], steps=cfg["tree_steps"],
            batch_sz=cfg["batch_sz"], train_noise=cfg["train_noise"], grad_clip=cfg["tree_grad_clip"],
            log_every=cfg["log_every"], early_stop_patience=cfg["tree_early_stop"],
            label=f"square_tree_{name}",
        )
        f = batch_eval_rank1_ttns(tree_best, val_basis, parent)
        z = integral_q2_ttns(tree_best, gram, parent)
        tree_ll, tree_nonpos = held_out_ll(f * f, z)
        results.append({
            "model": f"square_tree_{name}", "method": "TTDE(平方+MLE)",
            "rank": r_tree, "n_params": params,
            "native_metric": "val_nll", "native_value": tree_sum["final_val_nll"],
            "val_ll": tree_ll, "nonpositive_rate": tree_nonpos,
            "total_time_sec": tree_sum["total_time_sec"],
        })

    # ---- 完整 TTDE：多排列混合平方树（K 分量），参数量对齐 ----
    K = cfg["mix_n_comp"]
    for name, parent in tree_parents.items():
        r_mix = _pick_tree_rank(bases, train_x, parent, max(budget // K, 1))
        stacked0, perms = init_mixture(k_init, bases, train_x, parent, r_mix, K, cfg["init_noise"])
        params = _count_params(stacked0.cores)
        mix_best, mix_sum = train_square_mixture_mle(
            stacked0, perms, parent, bases, gram, train_x, val_x,
            key=k_init, lr=cfg["mix_lr"], steps=cfg["mix_steps"], batch_sz=cfg["batch_sz"],
            train_noise=cfg["train_noise"], grad_clip=cfg["tree_grad_clip"],
            log_every=cfg["log_every"], early_stop_patience=cfg["tree_early_stop"],
            label=f"ttde_mix_{name}",
        )
        q_mix = mixture_unnorm_q(mix_best, perms, parent, val_basis)
        z_mix = mixture_partition(mix_best, perms, parent, gram)
        mix_ll, mix_nonpos = held_out_ll(q_mix, z_mix)
        results.append({
            "model": f"ttde_mix_{name}", "method": f"完整TTDE(平方+{K}排列混合)",
            "rank": r_mix, "n_params": params,
            "native_metric": "val_nll", "native_value": mix_sum["final_val_nll"],
            "val_ll": mix_ll, "nonpositive_rate": mix_nonpos,
            "total_time_sec": mix_sum["total_time_sec"],
        })

    dag_ll_v = next(r["val_ll"] for r in results if r["model"] == "dag_linear_l2")
    tree_lls = [r["val_ll"] for r in results if r["model"] != "dag_linear_l2"]
    best_tree_ll = max(tree_lls)
    ll_gap = dag_ll_v - best_tree_ll  # >0 表示 DAG 共同 LL 更高（更好）

    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = {"config": cfg,
               "spec": {"layers": SPEC.layers, "edges": SPEC.edges, "n_edges": len(SPEC.edges)},
               "common_metric": "held_out_mean_ll (higher=better)",
               "results": results, "dag_val_ll": dag_ll_v,
               "best_square_tree_val_ll": best_tree_ll, "dag_minus_best_tree_ll": ll_gap}
    (REPORTS / "ttde_vs_dag_propagation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))

    lines = ["# TTDE 做法（平方树+MLE） vs 我们 DAG 传播做法（线性+L2）", "",
             "同一个例子（4 层 / 20 节点 banded 多父、乘积交互核），同一把尺子。", "",
             "- **TTDE 做法**：$q=f^2/Z$，MLE 训练（原版 TTDE 的平方密度方案）。",
             "- **我们 DAG 传播**：线性多父 DAG $q=\\langle T,b\\rangle$，L2 目标训练。",
             "",
             "原生指标不同（NLL vs L2），故统一用 **held-out 平均对数似然 LL**（越高越好）：",
             "把模型看成密度 $v/Z$ 后算 $\\mathbb{E}[\\log(v/Z)]$。平方树 $v=f^2$ 恒正；",
             "线性 DAG $v=q$ 可能为负，在 $v>0$ 的点上算 LL 并报告 `nonpositive_rate`（负值占比）。",
             "**参数量对齐**：以 DAG 参数量为预算，给平方树取 ≤ 预算的最大 rank。", "",
             f"配置：`{cfg}`", "",
             "| 模型 | 做法 | rank | 参数量 | 共同指标 val_LL↑ | 负值占比 | 原生指标 | 原生值 | 用时(s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['method']} | {r['rank']} | {r['n_params']} | "
            f"{r['val_ll']:.4f} | {r['nonpositive_rate']:.4f} | {r['native_metric']} | "
            f"{r['native_value']:.4f} | {r['total_time_sec']:.1f} |")
    lines += ["",
              f"DAG 共同 LL = {dag_ll_v:.4f}，最优树（含完整 TTDE 混合）共同 LL = {best_tree_ll:.4f}，",
              f"**LL 差 = {ll_gap:.4f} nats**（>0 表示线性 DAG 传播在同一尺子上更优）。"]
    (REPORTS / "ttde_vs_dag_propagation_report_zh.md").write_text("\n".join(lines) + "\n")

    print("\n==== 汇总：TTDE 平方树 vs 我们 DAG 传播（共同 LL）====")
    for r in results:
        print(f"{r['model']:>20s}  {r['method']:<18s}  rank={r['rank']:>2d}  "
              f"params={r['n_params']:>7d}  val_LL={r['val_ll']:.4f}  "
              f"nonpos={r['nonpositive_rate']:.4f}")
    print(f"DAG vs 最优平方树  共同 LL 差 = {ll_gap:.4f} nats（>0=DAG 更优）")


if __name__ == "__main__":
    main()
