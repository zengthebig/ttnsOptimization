"""多父 DAG TTNS 的 L2 拟合（训练循环 + 初始化）。

复用 `simple_ttns_l2` 的 spline 基与 L2 目标 $L=\\int q^2 - 2\\,\\mathbb{E}_{data}[q]$，
但把张量收缩换成 `dag_ttns` 的多父算子，从而能拟合真 DAG（含有环 moral graph）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import jax
import optax
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from ttde.score.models.continuous_canonical_init import continuous_rank_1  # noqa: E402

from simple_ttns_l2.objective import batch_basis_vectors_from_samples  # noqa: E402
from simple_ttns_l2.dag_ttns import (  # noqa: E402
    DAGGraph,
    DAGTTNS,
    dag_batch_eval_rank1,
    dag_normalize_by_integral,
    dag_quadratic_form,
    dag_ttns_from_rank1,
)


def init_dag_from_rank1(
    key: jnp.ndarray,
    bases,
    samples: jnp.ndarray,
    graph: DAGGraph,
    rank: int,
    noise: float,
) -> DAGTTNS:
    """用逐维 rank-1 拟合初始化，再补零扩到给定 bond rank，并加噪打破对称。"""
    rank1 = continuous_rank_1(bases, samples, jnp.ones(len(samples)))  # [n_dims, basis_dim]
    vectors = [rank1[i] for i in range(graph.n)]
    ttns = dag_ttns_from_rank1(vectors, graph, edge_ranks=rank, default_rank=rank)
    if noise <= 0:
        return ttns
    keys = jax.random.split(key, graph.n)
    cores = [c + jax.random.normal(k, c.shape, dtype=c.dtype) * noise for c, k in zip(ttns.cores, keys)]
    return DAGTTNS(tuple(cores))


def dag_l2_loss_on_batch(
    ttns: DAGTTNS,
    graph: DAGGraph,
    bases,
    xs_batch: jnp.ndarray,
    gram_matrices: jnp.ndarray,
) -> jnp.ndarray:
    bvb = batch_basis_vectors_from_samples(bases, xs_batch)
    int_q2 = dag_quadratic_form(ttns, graph, gram_matrices)
    mc_q = dag_batch_eval_rank1(ttns, graph, bvb).mean()
    return int_q2 - 2.0 * mc_q


def train_dag_l2(
    ttns: DAGTTNS,
    graph: DAGGraph,
    bases,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    gram_matrices: jnp.ndarray,
    basis_integrals: jnp.ndarray,
    *,
    key: jnp.ndarray,
    lr: float,
    train_steps: int,
    batch_sz: int,
    train_noise: float = 0.0,
    normalize_every: int = 1,
    log_every: int = 25,
    label: str = "dag",
) -> Tuple[DAGTTNS, Dict]:
    optimizer = optax.adam(learning_rate=lr)
    ttns, z0 = dag_normalize_by_integral(ttns, graph, basis_integrals)
    opt_state = optimizer.init(ttns)

    val_basis = batch_basis_vectors_from_samples(bases, val_x)

    @jax.jit
    def train_step(curr, st, batch):
        loss, grads = jax.value_and_grad(
            lambda x: dag_l2_loss_on_batch(x, graph, bases, batch, gram_matrices)
        )(curr)
        updates, st = optimizer.update(grads, st, curr)
        curr = optax.apply_updates(curr, updates)
        return curr, st, loss

    @jax.jit
    def eval_val_l2(curr):
        int_q2 = dag_quadratic_form(curr, graph, gram_matrices)
        mc = dag_batch_eval_rank1(curr, graph, val_basis).mean()
        return int_q2 - 2.0 * mc

    history: List[Dict] = []
    best_val = float("inf")
    best_ttns = ttns
    t_start = time.perf_counter()
    print(f"\n=== [{label}] init integral={float(z0):.6e} ===", flush=True)
    print("step,train_l2,val_l2,total_sec", flush=True)

    for step in range(1, train_steps + 1):
        key, k_idx, k_noise = jax.random.split(key, 3)
        idx = jax.random.randint(k_idx, (batch_sz,), 0, train_x.shape[0])
        batch = train_x[idx]
        if train_noise > 0:
            batch = batch + jax.random.normal(k_noise, batch.shape) * train_noise
        ttns, opt_state, loss = train_step(ttns, opt_state, batch)
        if normalize_every > 0 and step % normalize_every == 0:
            ttns, _ = dag_normalize_by_integral(ttns, graph, basis_integrals)

        if step % log_every == 0 or step == train_steps:
            vl = float(eval_val_l2(ttns))
            total = time.perf_counter() - t_start
            history.append({"step": step, "train_l2": float(loss), "val_l2": vl, "total_sec": total})
            print(f"{step},{float(loss):.6f},{vl:.6f},{total:.3f}", flush=True)
            if vl < best_val:
                best_val = vl
                best_ttns = ttns

    summary = {
        "label": label,
        "final_val_l2": float(eval_val_l2(best_ttns)),
        "best_val_l2": best_val,
        "total_time_sec": time.perf_counter() - t_start,
        "history": history,
    }
    return best_ttns, summary
