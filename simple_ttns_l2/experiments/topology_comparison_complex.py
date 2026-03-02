from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import jax
from jax import numpy as jnp

# Ensure imports resolve to local simple_ttns_l2 and TTNSDE/ttde.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.train_l2 import build_bases, make_parent
from simple_ttns_l2.experiments.topology_comparison import (
    ExperimentConfig,
    build_report,
    run_one_training,
)

jax.config.update("jax_enable_x64", True)


def sample_complex_tree_distribution(
    key: jnp.ndarray,
    n_samples: int,
    parent: list[int],
) -> jnp.ndarray:
    """
    6D nonlinear tree-structured target:
    - root: bimodal Gaussian mixture
    - each child: nonlinear function of parent + heteroscedastic Gaussian noise
    """
    n_dims = len(parent)
    x = jnp.zeros((n_samples, n_dims), dtype=jnp.float64)

    root = 0
    key, k_bern, k_g1, k_g2 = jax.random.split(key, 4)
    z = jax.random.bernoulli(k_bern, p=0.5, shape=(n_samples,))
    g1 = jax.random.normal(k_g1, (n_samples,), dtype=jnp.float64) * 0.7 - 1.4
    g2 = jax.random.normal(k_g2, (n_samples,), dtype=jnp.float64) * 0.6 + 1.2
    root_x = jnp.where(z, g1, g2)
    x = x.at[:, root].set(root_x)

    for i in range(1, n_dims):
        p = parent[i]
        key, k_eps = jax.random.split(key)
        u = x[:, p]
        mean = 0.62 * u + 0.24 * (u ** 2 - 1.0) + 0.14 * jnp.sin(2.3 * u)
        sigma = 0.28 + 0.08 * jnp.abs(u)
        eps = jax.random.normal(k_eps, (n_samples,), dtype=jnp.float64) * sigma
        x = x.at[:, i].set(mean + eps)

    return x


def main():
    cfg = ExperimentConfig(
        n_dims=6,
        q=2,
        m=24,
        rank=3,
        batch_sz=256,
        lr=1e-3,
        train_noise=1e-2,
        init_noise=1e-2,
        train_steps=100,
        log_every=5,
        seed=123,
        n_train=5000,
        n_val=2500,
        n_test=5000,
        monitor_train_sz=2000,
        monitor_val_sz=2000,
    )
    print("running complex topology comparison with config:", asdict(cfg), flush=True)

    results = []
    seed_cursor = 0

    for target_name in ("balanced", "chain"):
        parent_target = make_parent(cfg.n_dims, target_name)
        key = jax.random.PRNGKey(cfg.seed + 2000 + seed_cursor)
        seed_cursor += 100
        k_train, k_val, k_test = jax.random.split(key, 3)

        train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
        val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
        test_x = sample_complex_tree_distribution(k_test, cfg.n_test, parent_target)

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
    metrics_path = report_dir / "topology_comparison_complex_metrics.json"
    report_path = report_dir / "topology_comparison_complex_report.md"
    report_dir.mkdir(parents=True, exist_ok=True)

    serializable = {
        "config": asdict(cfg),
        "results": results,
    }
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    build_report(cfg, results, report_path)

    print("\n=== summary (complex target) ===", flush=True)
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

