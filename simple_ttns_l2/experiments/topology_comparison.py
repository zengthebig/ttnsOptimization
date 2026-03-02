from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import jax
import optax
from jax import numpy as jnp

# Ensure imports resolve to local simple_ttns_l2 and TTNSDE/ttde.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.objective import (
    batch_basis_vectors_from_samples,
    integral_q2_ttns,
    integral_q_ttns,
    mc_expectation_q_ttns,
    normalize_ttns_by_integral,
)
from simple_ttns_l2.train_l2 import (
    build_bases,
    init_ttns_from_rank1,
    l2_loss_on_batch,
    make_parent,
)

jax.config.update("jax_enable_x64", True)


@dataclass
class ExperimentConfig:
    n_dims: int = 6
    q: int = 2
    m: int = 16
    rank: int = 2
    batch_sz: int = 256
    lr: float = 1e-3
    train_noise: float = 1e-2
    init_noise: float = 1e-2
    train_steps: int = 100
    log_every: int = 5
    seed: int = 0
    n_train: int = 4000
    n_val: int = 2000
    n_test: int = 4000
    monitor_train_sz: int = 2000
    monitor_val_sz: int = 2000
    # Early stopping on val_l2 logs; set patience<=0 to disable.
    early_stop_patience_logs: int = 6
    early_stop_min_delta: float = 1e-4
    early_stop_warmup_logs: int = 4
    early_stop_restore_best: bool = True


def _root_from_parent(parent: Sequence[int]) -> int:
    return next(i for i, p in enumerate(parent) if p == i or p == -1)


def _depths(parent: Sequence[int]) -> List[int]:
    n = len(parent)
    root = _root_from_parent(parent)
    depths = [-1] * n
    depths[root] = 0
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if depths[i] != -1:
                continue
            p = parent[i]
            if p != -1 and p != i and depths[p] != -1:
                depths[i] = depths[p] + 1
                changed = True
    return depths


def make_tree_gaussian_params(parent: Sequence[int]) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = len(parent)
    depths = _depths(parent)
    weights = jnp.zeros((n,), dtype=jnp.float64)
    noises = jnp.ones((n,), dtype=jnp.float64)
    root = _root_from_parent(parent)
    for i in range(n):
        if i == root:
            continue
        d = depths[i]
        # Strong local dependency with depth-decay, for clearer topology contrast.
        w = max(0.45, 0.85 - 0.12 * max(0, d - 1))
        weights = weights.at[i].set(w)
        noises = noises.at[i].set(jnp.sqrt(1.0 - w * w))
    return weights, noises


def sample_tree_gaussian(
    key: jnp.ndarray,
    n_samples: int,
    parent: Sequence[int],
    weights: jnp.ndarray,
    noises: jnp.ndarray,
) -> jnp.ndarray:
    n = len(parent)
    root = _root_from_parent(parent)
    x = jnp.zeros((n_samples, n), dtype=jnp.float64)

    key, k_root = jax.random.split(key)
    x = x.at[:, root].set(jax.random.normal(k_root, (n_samples,), dtype=jnp.float64))

    # Parent index is always smaller for both balanced/chain from make_parent.
    for i in range(n):
        if i == root:
            continue
        key, k_eps = jax.random.split(key)
        p = parent[i]
        eps = jax.random.normal(k_eps, (n_samples,), dtype=jnp.float64) * noises[i]
        x = x.at[:, i].set(weights[i] * x[:, p] + eps)
    return x


def run_one_training(
    cfg: ExperimentConfig,
    target_name: str,
    model_topology: str,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    test_x: jnp.ndarray,
    bases,
    gram_matrices: jnp.ndarray,
    basis_integrals: jnp.ndarray,
    seed_offset: int,
) -> Dict:
    parent_model = make_parent(cfg.n_dims, model_topology)
    key = jax.random.PRNGKey(cfg.seed + seed_offset)
    key, k_init, k_iter = jax.random.split(key, 3)

    ttns = init_ttns_from_rank1(
        key=k_init,
        bases=bases,
        samples=train_x,
        parent=parent_model,
        rank=cfg.rank,
        noise=cfg.init_noise,
    )
    ttns, z0 = normalize_ttns_by_integral(ttns, basis_integrals, parent_model)

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(ttns)

    train_monitor = train_x[: cfg.monitor_train_sz]
    val_monitor = val_x[: cfg.monitor_val_sz]
    train_monitor_basis = batch_basis_vectors_from_samples(bases, train_monitor)
    val_monitor_basis = batch_basis_vectors_from_samples(bases, val_monitor)

    history = []
    best_val_l2 = float("inf")
    best_step = 0
    bad_logs = 0
    logs_seen = 0
    best_ttns = ttns
    stopped_early = False
    stop_step = cfg.train_steps
    start = time.perf_counter()
    last_log = start
    print(
        f"\n=== target={target_name}, model={model_topology}, init_integral={float(z0):.6e} ===",
        flush=True,
    )
    print("step,train_l2,val_l2,integral,interval_sec,total_sec", flush=True)

    @jax.jit
    def train_step(curr_ttns, curr_opt_state, batch, key_noise):
        noisy_batch = batch + jax.random.normal(key_noise, batch.shape) * cfg.train_noise
        loss, grads = jax.value_and_grad(
            lambda x: l2_loss_on_batch(x, bases, noisy_batch, parent_model, gram_matrices)
        )(curr_ttns)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_ttns)
        next_ttns = optax.apply_updates(curr_ttns, updates)
        next_ttns, z = normalize_ttns_by_integral(next_ttns, basis_integrals, parent_model)
        return next_ttns, next_opt_state, loss, z

    eval_l2 = jax.jit(lambda curr_ttns, xs: l2_loss_on_batch(curr_ttns, bases, xs, parent_model, gram_matrices))
    eval_int_q2 = jax.jit(lambda curr_ttns: integral_q2_ttns(curr_ttns, gram_matrices, parent_model))
    eval_mc = jax.jit(
        lambda curr_ttns, basis_batch: mc_expectation_q_ttns(curr_ttns, basis_batch, parent_model)
    )
    eval_integral = jax.jit(lambda curr_ttns: integral_q_ttns(curr_ttns, basis_integrals, parent_model))

    for step in range(1, cfg.train_steps + 1):
        k_iter, k_idx, k_noise = jax.random.split(k_iter, 3)
        idx = jax.random.randint(k_idx, (cfg.batch_sz,), 0, train_x.shape[0])
        batch = train_x[idx]
        ttns, opt_state, _, curr_integral = train_step(ttns, opt_state, batch, k_noise)

        if step % cfg.log_every == 0:
            now = time.perf_counter()
            int_q2 = eval_int_q2(ttns)
            train_l2 = int_q2 - 2.0 * eval_mc(ttns, train_monitor_basis)
            val_l2 = int_q2 - 2.0 * eval_mc(ttns, val_monitor_basis)
            z = curr_integral
            interval_sec = now - last_log
            total_sec = now - start
            last_log = now
            print(
                f"{step},{float(train_l2):.6f},{float(val_l2):.6f},{float(z):.6e},"
                f"{interval_sec:.3f},{total_sec:.3f}",
                flush=True,
            )
            history.append(
                {
                    "step": step,
                    "train_l2": float(train_l2),
                    "val_l2": float(val_l2),
                    "integral": float(z),
                    "interval_sec": interval_sec,
                    "total_sec": total_sec,
                }
            )

            logs_seen += 1
            val_l2_f = float(val_l2)
            improved = val_l2_f < (best_val_l2 - float(cfg.early_stop_min_delta))
            if improved:
                best_val_l2 = val_l2_f
                best_step = step
                best_ttns = ttns
                bad_logs = 0
            elif logs_seen > int(cfg.early_stop_warmup_logs):
                bad_logs += 1

            if int(cfg.early_stop_patience_logs) > 0 and bad_logs >= int(cfg.early_stop_patience_logs):
                stopped_early = True
                stop_step = step
                print(
                    f"early_stop at step={step}: best_val_l2={best_val_l2:.6f} (step={best_step}), "
                    f"patience_logs={cfg.early_stop_patience_logs}, min_delta={cfg.early_stop_min_delta}",
                    flush=True,
                )
                break

    if bool(cfg.early_stop_restore_best):
        ttns = best_ttns

    final_val_l2 = float(eval_l2(ttns, val_x))
    final_test_l2 = float(eval_l2(ttns, test_x))
    final_integral = float(eval_integral(ttns))
    total_time = time.perf_counter() - start

    return {
        "target_topology": target_name,
        "model_topology": model_topology,
        "final_val_l2": final_val_l2,
        "final_test_l2": final_test_l2,
        "final_integral": final_integral,
        "total_time_sec": total_time,
        "stopped_early": stopped_early,
        "stop_step": int(stop_step),
        "best_step": int(best_step),
        "best_val_l2": float(best_val_l2),
        "history": history,
    }


def build_report(cfg: ExperimentConfig, results: List[Dict], report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    by_target = {"balanced": {}, "chain": {}}
    for row in results:
        by_target[row["target_topology"]][row["model_topology"]] = row

    b_bal = by_target["balanced"]["balanced"]["final_test_l2"]
    b_chn = by_target["balanced"]["chain"]["final_test_l2"]
    c_bal = by_target["chain"]["balanced"]["final_test_l2"]
    c_chn = by_target["chain"]["chain"]["final_test_l2"]

    balanced_target_prefers_balanced = b_bal < b_chn
    chain_target_prefers_chain = c_chn < c_bal

    lines = []
    lines.append("# Topology Comparison Report\n")
    lines.append(f"- Date: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Config: `{json.dumps(asdict(cfg), ensure_ascii=True)}`\n")

    lines.append("## Final Metrics (lower L2 is better)\n")
    lines.append("| target_topology | model_topology | final_val_l2 | final_test_l2 | final_integral | total_time_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in results:
        lines.append(
            f"| {row['target_topology']} | {row['model_topology']} | "
            f"{row['final_val_l2']:.6f} | {row['final_test_l2']:.6f} | "
            f"{row['final_integral']:.6f} | {row['total_time_sec']:.3f} |"
        )

    lines.append("\n## Topology Preference Check\n")
    lines.append(
        f"- Target `balanced`: balanced-model test L2 = `{b_bal:.6f}`, chain-model test L2 = `{b_chn:.6f}`. "
        f"Conclusion: `{'balanced better' if balanced_target_prefers_balanced else 'chain better or tie'}`."
    )
    lines.append(
        f"- Target `chain`: chain-model test L2 = `{c_chn:.6f}`, balanced-model test L2 = `{c_bal:.6f}`. "
        f"Conclusion: `{'chain better' if chain_target_prefers_chain else 'balanced better or tie'}`."
    )

    lines.append("\n## Overall\n")
    if balanced_target_prefers_balanced and chain_target_prefers_chain:
        lines.append(
            "Matched topology wins in both directions under this setup."
        )
    else:
        lines.append(
            "Matched topology does not win in both directions under this setup; "
            "consider larger rank/steps or multi-seed averaging."
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cfg = ExperimentConfig()
    print("running topology comparison with config:", asdict(cfg), flush=True)

    results = []
    seed_cursor = 0

    for target_name in ("balanced", "chain"):
        parent_target = make_parent(cfg.n_dims, target_name)
        key = jax.random.PRNGKey(cfg.seed + 1000 + seed_cursor)
        seed_cursor += 100
        k_train, k_val, k_test = jax.random.split(key, 3)

        weights, noises = make_tree_gaussian_params(parent_target)
        train_x = sample_tree_gaussian(k_train, cfg.n_train, parent_target, weights, noises)
        val_x = sample_tree_gaussian(k_val, cfg.n_val, parent_target, weights, noises)
        test_x = sample_tree_gaussian(k_test, cfg.n_test, parent_target, weights, noises)

        # Use same bases/integral tensors for both model topologies under this target.
        bases = build_bases(train_x, q=cfg.q, m=cfg.m)
        gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
        basis_integrals = jax.vmap(type(bases).integral)(bases)

        for model_topology in ("balanced", "chain"):
            result = run_one_training(
                cfg=cfg,
                target_name=target_name,
                model_topology=model_topology,
                train_x=train_x,
                val_x=val_x,
                test_x=test_x,
                bases=bases,
                gram_matrices=gram_matrices,
                basis_integrals=basis_integrals,
                seed_offset=seed_cursor,
            )
            seed_cursor += 1
            results.append(result)

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    metrics_path = report_dir / "topology_comparison_metrics.json"
    report_path = report_dir / "topology_comparison_report.md"
    report_dir.mkdir(parents=True, exist_ok=True)

    serializable = {
        "config": asdict(cfg),
        "results": results,
    }
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    build_report(cfg, results, report_path)

    print("\n=== summary ===", flush=True)
    for row in results:
        print(
            f"target={row['target_topology']:8s} model={row['model_topology']:8s} "
            f"test_l2={row['final_test_l2']:.6f} val_l2={row['final_val_l2']:.6f} "
            f"integral={row['final_integral']:.6f} time={row['total_time_sec']:.3f}s",
            flush=True,
        )
    print("saved metrics:", metrics_path, flush=True)
    print("saved report :", report_path, flush=True)


if __name__ == "__main__":
    main()
