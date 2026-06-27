from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

KEY_PAIRS_6D: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
KEY_PAIRS_7D: Tuple[Tuple[int, int], ...] = ((0, 2), (1, 2), (2, 6), (5, 6))
KEY_PAIRS_8D: Tuple[Tuple[int, int], ...] = ((0, 2), (1, 2), (2, 7), (6, 7))
ALL_PAIRS_6D: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2), (2, 3), (3, 5))
ALL_PAIRS_7D: Tuple[Tuple[int, int], ...] = ((0, 2), (1, 2), (2, 6), (5, 6), (2, 5), (0, 6))
ALL_PAIRS_8D: Tuple[Tuple[int, int], ...] = ((0, 2), (1, 2), (2, 7), (6, 7), (2, 6), (0, 7))
MODEL_ORDER = ("junction", "balanced", "chain")


def _config() -> ExperimentConfig:
    """7D fork DAG + 与 fit_ceiling 对齐的超参。"""
    return ExperimentConfig(
        n_dims=7,
        q=2,
        m=56,
        rank=16,
        batch_sz=128,
        lr=1e-3,
        train_noise=1e-2,
        init_noise=1e-2,
        train_steps=400,
        log_every=10,
        seed=4242,
        n_train=8000,
        n_val=4000,
        n_test=4000,
        monitor_train_sz=3000,
        monitor_val_sz=3000,
    )


def _key_pairs(n_dims: int) -> Tuple[Tuple[int, int], ...]:
    if n_dims == 7:
        return KEY_PAIRS_7D
    if n_dims == 8:
        return KEY_PAIRS_8D
    return KEY_PAIRS_6D


def _all_pairs(n_dims: int) -> Tuple[Tuple[int, int], ...]:
    if n_dims == 7:
        return ALL_PAIRS_7D
    if n_dims == 8:
        return ALL_PAIRS_8D
    return ALL_PAIRS_6D


def _lr_policy(cfg: ExperimentConfig) -> Dict:
    hold_steps = int(0.40 * cfg.train_steps)
    return {
        "mode": "delayed_cosine",
        "label": f"dag_ceiling_delayed_cosine_hold{hold_steps}_0.1x",
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


def _model_parent(name: str, n_dims: int, junction_parent: List[int]) -> List[int]:
    if name == "junction":
        return list(junction_parent)
    return make_parent(n_dims, name)


def _aggregate_key_slice(slice_rows: List[Dict], model: str, key_pairs: Sequence[Tuple[int, int]]) -> float:
    key = f"iae_{model}"
    sub = [r for r in slice_rows if (r["dim_i"], r["dim_j"]) in key_pairs]
    return float(np.mean([r[key] for r in sub]))


def _write_report(
    path: Path,
    cfg: ExperimentConfig,
    dag_edges: Sequence[Tuple[int, int]],
    junction_parent: List[int],
    train_rows: List[Dict],
    slice_rows: List[Dict],
    key_pairs: Sequence[Tuple[int, int]],
    aggregate: Dict | None = None,
):
    lines = [
        "# Fork DAG：联结树 / Balanced / Chain TTNS 对比报告",
        "",
        "## 目的",
        "",
        "在含**双父节点**依赖（$x_2 \\sim f(x_0, x_1)$）的合成 fork DAG 目标上，",
        "对比联结树风格 TTNS、balanced TTNS 与 chain TTNS 的 2D 切片 IAE。",
        "",
        "## 配置",
        "",
        f"- 训练配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`",
        f"- DAG 边: `{list(dag_edges)}`",
        f"- junction parent: `{junction_parent}`",
        f"- balanced parent: `{make_parent(cfg.n_dims, 'balanced')}`",
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
            "| pair | noise_floor | IAE_junction | IAE_balanced | IAE_chain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {pair} | {row['noise_floor']:.6f} | {row['iae_junction']:.6f} | "
            f"{row['iae_balanced']:.6f} | {row['iae_chain']:.6f} |"
        )

    mean_j = _aggregate_key_slice(slice_rows, "junction", key_pairs)
    mean_b = _aggregate_key_slice(slice_rows, "balanced", key_pairs)
    mean_c = _aggregate_key_slice(slice_rows, "chain", key_pairs)
    imp_jc = (mean_c - mean_j) / max(mean_c, 1e-12)
    imp_bc = (mean_c - mean_b) / max(mean_c, 1e-12)
    winner = min(MODEL_ORDER, key=lambda m: _aggregate_key_slice(slice_rows, m, key_pairs))
    key_label = ",".join(f"({i},{j})" for i, j in key_pairs)

    lines.extend(
        [
            "",
            f"## 关键切片聚合 ({key_label})",
            "",
            f"- mean IAE junction: **{mean_j:.6f}**",
            f"- mean IAE balanced: **{mean_b:.6f}**",
            f"- mean IAE chain: **{mean_c:.6f}**",
            f"- junction vs chain 提升: **{imp_jc * 100:.1f}%**",
            f"- balanced vs chain 提升: **{imp_bc * 100:.1f}%**",
            f"- 最优模型: **{winner}**",
            "",
            "## 结论",
            "",
        ]
    )
    if imp_jc >= 0.20:
        lines.append(
            f"- 联结树 TTNS 在关键切片上优于 chain **{imp_jc * 100:.1f}%**，达到 Phase 2 M3 标准。"
        )
    else:
        lines.append(
            f"- 联结树相对 chain 提升 **{imp_jc * 100:.1f}%**，未达 20% 阈值。"
        )
    if aggregate and "per_seed" in aggregate:
        lines.extend(["", "## 多 seed 聚合", "", "| seed | junction | balanced | chain | j vs c |", "|-----:|---:|---:|---:|---:|"])
        for row in aggregate["per_seed"]:
            lines.append(
                f"| {row['seed']} | {row['mean_iae_junction']:.6f} | {row['mean_iae_balanced']:.6f} | "
                f"{row['mean_iae_chain']:.6f} | {row['improvement_junction_vs_chain'] * 100:.1f}% |"
            )
        lines.append(
            f"\n- 跨 seed 平均：junction vs chain **{aggregate['mean_improvement_junction_vs_chain'] * 100:.1f}%**"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dag_experiment(seed: int | None = None, write_outputs: bool = True) -> Dict:
    cfg = _config()
    if seed is not None:
        cfg = replace(cfg, seed=seed)

    lr_policy = _lr_policy(cfg)
    dag_edges = fork_dag_edges(cfg.n_dims)
    junction_parent = fork_junction_parent(cfg.n_dims)
    grid_bins = 48
    n_slice_samples = 200000

    key_pairs = _key_pairs(cfg.n_dims)
    all_pairs = _all_pairs(cfg.n_dims)

    print("fork DAG experiment seed=", cfg.seed, "n_dims=", cfg.n_dims, "rank=", cfg.rank, flush=True)

    key = jax.random.PRNGKey(cfg.seed)
    k_train, k_val, k_sa, k_sb = jax.random.split(key, 4)
    train_x = sample_fork_dag_distribution(k_train, cfg.n_train, n_dims=cfg.n_dims)
    val_x = sample_fork_dag_distribution(k_val, cfg.n_val, n_dims=cfg.n_dims)
    slice_a = sample_fork_dag_distribution(k_sa, n_slice_samples, n_dims=cfg.n_dims)
    slice_b = sample_fork_dag_distribution(k_sb, n_slice_samples, n_dims=cfg.n_dims)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    trained: Dict[str, Tuple] = {}
    train_rows: List[Dict] = []
    seed_offset = 0
    for name in MODEL_ORDER:
        parent = _model_parent(name, cfg.n_dims, junction_parent)
        model, parent_used, summary = _train_one_model(
            cfg=cfg,
            target_topology="fork_dag",
            model_topology=name,
            train_x=train_x,
            val_x=val_x,
            bases=bases,
            gram_matrices=gram_matrices,
            basis_integrals=basis_integrals,
            seed_offset=seed_offset,
            lr_policy_template=lr_policy,
            model_parent=parent,
        )
        seed_offset += 50
        trained[name] = (model, parent_used)
        train_rows.append(summary)

    a_np = np.asarray(slice_a)
    slice_rows: List[Dict] = []

    for dim_i, dim_j in all_pairs:
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
        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])
        floor = _iae_2d(target_density, target_density_2, dx, dy)

        densities = {}
        for name in MODEL_ORDER:
            model, parent = trained[name]
            densities[name] = _eval_pair_marginal_on_grid(
                model, parent, bases, basis_integrals, dim_i, dim_j, xi_centers, xj_centers
            )

        row = {
            "dim_i": dim_i,
            "dim_j": dim_j,
            "noise_floor": floor,
            "iae_junction": _iae_2d(target_density, densities["junction"], dx, dy),
            "iae_balanced": _iae_2d(target_density, densities["balanced"], dx, dy),
            "iae_chain": _iae_2d(target_density, densities["chain"], dx, dy),
        }
        slice_rows.append(row)
        print(
            f"pair=({dim_i},{dim_j}) j={row['iae_junction']:.4f} b={row['iae_balanced']:.4f} c={row['iae_chain']:.4f}",
            flush=True,
        )

    mean_j = _aggregate_key_slice(slice_rows, "junction", key_pairs)
    mean_b = _aggregate_key_slice(slice_rows, "balanced", key_pairs)
    mean_c = _aggregate_key_slice(slice_rows, "chain", key_pairs)
    result = {
        "config": asdict(cfg),
        "dag_edges": [list(e) for e in dag_edges],
        "junction_parent": junction_parent,
        "balanced_parent": make_parent(cfg.n_dims, "balanced"),
        "chain_parent": make_parent(cfg.n_dims, "chain"),
        "training": train_rows,
        "slice_metrics": slice_rows,
        "key_slice_mean_iae_junction": mean_j,
        "key_slice_mean_iae_balanced": mean_b,
        "key_slice_mean_iae_chain": mean_c,
        "key_slice_improvement_junction_vs_chain": (mean_c - mean_j) / max(mean_c, 1e-12),
        "generated_epoch_sec": time.time(),
    }

    if not write_outputs:
        return result

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = report_dir / "dag_junction_vs_chain_report_zh.md"
    out_json = report_dir / "dag_junction_vs_chain_metrics.json"

    _write_report(out_md, cfg, dag_edges, junction_parent, train_rows, slice_rows, key_pairs)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("key slice improvement junction vs chain:", f"{result['key_slice_improvement_junction_vs_chain'] * 100:.2f}%", flush=True)
    print("saved report:", out_md, flush=True)
    return result


def run():
    run_dag_experiment(write_outputs=True)


if __name__ == "__main__":
    run()
