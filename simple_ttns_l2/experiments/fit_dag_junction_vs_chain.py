from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.junction_tree import fork_junction_parent
from simple_ttns_l2.experiments.dag_target import fork_dag_edges, sample_fork_dag_distribution
from simple_ttns_l2.train_l2 import build_bases, make_parent
from simple_ttns_l2.experiments.topology_comparison import ExperimentConfig
from simple_ttns_l2.experiments.topology_slice_visualization import (
    _empirical_pair_density,
    _eval_pair_marginal_on_grid,
    _iae_2d,
    _train_one_model,
)

jax.config.update("jax_enable_x64", True)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        n_dims=6,
        q=2,
        m=64,
        rank=16,
        batch_sz=256,
        lr=1e-3,
        train_noise=1e-2,
        init_noise=0.0,
        train_steps=300,
        log_every=10,
        seed=4242,
        n_train=8000,
        n_val=4000,
        n_test=4000,
        monitor_train_sz=3000,
        monitor_val_sz=3000,
    )


def _lr_policy(cfg: ExperimentConfig) -> Dict:
    hold_steps = int(0.40 * cfg.train_steps)
    return {
        "mode": "delayed_cosine",
        "label": f"dag_delayed_cosine_hold{hold_steps}_0.1x",
        "init_lr": float(cfg.lr),
        "final_lr": float(cfg.lr * 0.1),
        "hold_steps": hold_steps,
        "patience_logs": 0,
        "factor": 1.0,
        "min_lr": float(cfg.lr * 0.1),
        "min_delta": 0.0,
        "cooldown_logs": 0,
        "best_val": float("inf"),
        "bad_logs": 0,
        "cooldown_left": 0,
        "curr_lr": float(cfg.lr),
    }


def _write_report(
    path: Path,
    cfg: ExperimentConfig,
    dag_edges: Tuple[Tuple[int, int], ...],
    junction_parent: List[int],
    train_rows: List[Dict],
    slice_rows: List[Dict],
):
    lines = [
        "# Fork DAG：联结树 TTNS vs Chain TTNS 实验报告",
        "",
        "## 目的",
        "",
        "在含**双父节点**依赖（$x_2 \\sim f(x_0, x_1)$）的合成 fork DAG 目标上，",
        "对比联结树风格 TTNS 与 chain TTNS 的 2D 切片 IAE。",
        "",
        "## 配置",
        "",
        f"- 训练配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`",
        f"- DAG 边: `{list(dag_edges)}`",
        f"- 联结树 parent: `{junction_parent}`",
        f"- chain parent: `{make_parent(cfg.n_dims, 'chain')}`",
        "",
        "## 训练结果",
        "",
        "| model | final_val_l2 | final_integral | total_time_sec |",
        "|---|---:|---:|---:|",
    ]
    for row in train_rows:
        lines.append(
            f"| {row['model_topology']} | {row['final_val_l2']:.6f} | "
            f"{row['final_integral']:.6f} | {row['total_time_sec']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 切片 IAE",
            "",
            "| pair | noise_floor | IAE_junction | IAE_chain | ratio_junction | ratio_chain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {pair} | {row['noise_floor']:.6f} | {row['iae_junction']:.6f} | "
            f"{row['iae_chain']:.6f} | {row['ratio_junction']:.3f} | {row['ratio_chain']:.3f} |"
        )

    mean_j = float(np.mean([r["iae_junction"] for r in slice_rows]))
    mean_c = float(np.mean([r["iae_chain"] for r in slice_rows]))
    key_rows = [r for r in slice_rows if (r["dim_i"], r["dim_j"]) in ((0, 2), (1, 2), (0, 1))]
    mean_key_j = float(np.mean([r["iae_junction"] for r in key_rows])) if key_rows else mean_j
    mean_key_c = float(np.mean([r["iae_chain"] for r in key_rows])) if key_rows else mean_c
    imp = (mean_key_c - mean_key_j) / max(mean_key_c, 1e-12)

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 全切片 mean IAE：junction={mean_j:.4f}，chain={mean_c:.4f}。",
            f"- 双父相关切片 {(0, 2), (1, 2), (0, 1)} mean IAE：junction={mean_key_j:.4f}，chain={mean_key_c:.4f}（相对提升 {imp * 100:.1f}%）。",
        ]
    )
    if imp >= 0.20:
        lines.append("- 联结树 TTNS 在关键切片上显著优于 chain，支持 Phase 2 M3 假设。")
    else:
        lines.append("- 优势未达 20%；可增大 rank/步数或调整联结树 parent。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    cfg = _config()
    lr_policy = _lr_policy(cfg)
    dag_edges = fork_dag_edges(cfg.n_dims)
    junction_parent = fork_junction_parent(cfg.n_dims)
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 5)]
    grid_bins = 48
    n_slice_samples = 200000

    print("fork DAG experiment config:", asdict(cfg), flush=True)
    print("junction parent:", junction_parent, flush=True)

    key = jax.random.PRNGKey(cfg.seed)
    k_train, k_val, k_sa, k_sb = jax.random.split(key, 4)
    train_x = sample_fork_dag_distribution(k_train, cfg.n_train)
    val_x = sample_fork_dag_distribution(k_val, cfg.n_val)
    slice_a = sample_fork_dag_distribution(k_sa, n_slice_samples)
    slice_b = sample_fork_dag_distribution(k_sb, n_slice_samples)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    trained = {}
    train_rows = []
    for name, parent in (
        ("junction", junction_parent),
        ("chain", make_parent(cfg.n_dims, "chain")),
    ):
        model, parent_used, summary = _train_one_model(
            cfg=cfg,
            target_topology="fork_dag",
            model_topology=name,
            train_x=train_x,
            val_x=val_x,
            bases=bases,
            gram_matrices=gram_matrices,
            basis_integrals=basis_integrals,
            seed_offset=100 if name == "chain" else 0,
            lr_policy_template=lr_policy,
            model_parent=parent,
        )
        trained[name] = (model, parent_used)
        train_rows.append(summary)

    model_j, parent_j = trained["junction"]
    model_c, parent_c = trained["chain"]
    a_np = np.asarray(slice_a)

    slice_rows = []
    for dim_i, dim_j in pairs:
        xi_lo = np.quantile(a_np[:, dim_i], 0.01)
        xi_hi = np.quantile(a_np[:, dim_i], 0.99)
        xj_lo = np.quantile(a_np[:, dim_j], 0.01)
        xj_hi = np.quantile(a_np[:, dim_j], 0.99)
        xi_pad = 0.05 * (xi_hi - xi_lo + 1e-12)
        xj_pad = 0.05 * (xj_hi - xj_lo + 1e-12)
        xi_edges = np.linspace(xi_lo - xi_pad, xi_hi + xi_pad, grid_bins + 1)
        xj_edges = np.linspace(xj_lo - xj_pad, xj_hi + xj_pad, grid_bins + 1)
        xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        xj_centers = 0.5 * (xj_edges[:-1] + xj_edges[1:])

        target_density = _empirical_pair_density(a_np, dim_i, dim_j, xi_edges, xj_edges)
        target_density_2 = _empirical_pair_density(np.asarray(slice_b), dim_i, dim_j, xi_edges, xj_edges)
        junction_density = _eval_pair_marginal_on_grid(
            model_j, parent_j, bases, basis_integrals, dim_i, dim_j, xi_centers, xj_centers
        )
        chain_density = _eval_pair_marginal_on_grid(
            model_c, parent_c, bases, basis_integrals, dim_i, dim_j, xi_centers, xj_centers
        )
        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])
        floor = _iae_2d(target_density, target_density_2, dx, dy)
        iae_j = _iae_2d(target_density, junction_density, dx, dy)
        iae_c = _iae_2d(target_density, chain_density, dx, dy)
        slice_rows.append(
            {
                "dim_i": dim_i,
                "dim_j": dim_j,
                "noise_floor": floor,
                "iae_junction": iae_j,
                "iae_chain": iae_c,
                "ratio_junction": float(iae_j / max(floor, 1e-12)),
                "ratio_chain": float(iae_c / max(floor, 1e-12)),
            }
        )
        print(f"pair=({dim_i},{dim_j}) IAE_junction={iae_j:.6f} IAE_chain={iae_c:.6f}", flush=True)

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = report_dir / "dag_junction_vs_chain_report_zh.md"
    out_json = report_dir / "dag_junction_vs_chain_metrics.json"

    _write_report(out_md, cfg, dag_edges, junction_parent, train_rows, slice_rows)
    out_json.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "dag_edges": [list(e) for e in dag_edges],
                "junction_parent": junction_parent,
                "chain_parent": make_parent(cfg.n_dims, "chain"),
                "training": train_rows,
                "slice_metrics": slice_rows,
                "generated_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved report:", out_md, flush=True)
    print("saved metrics:", out_json, flush=True)


if __name__ == "__main__":
    run()
