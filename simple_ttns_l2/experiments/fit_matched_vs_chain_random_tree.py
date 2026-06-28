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

from simple_ttns_l2.experiments.random_tree_target import (
    build_random_tree_target_spec,
    sample_random_tree_distribution,
)
from simple_ttns_l2.junction_tree import validate_tree_parent
from simple_ttns_l2.train_l2 import build_bases, make_parent
from simple_ttns_l2.experiments.topology_comparison import ExperimentConfig
from simple_ttns_l2.experiments.topology_slice_visualization import (
    _empirical_pair_density,
    _eval_pair_marginal_on_grid,
    _iae_2d,
    _train_one_model,
)

jax.config.update("jax_enable_x64", True)

SLICE_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 3), (2, 5))
MODEL_ORDER = ("matched", "chain")


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        n_dims=6,
        q=2,
        m=64,
        rank=16,
        batch_sz=256,
        lr=1e-3,
        train_noise=1e-2,
        init_noise=1e-2,
        train_steps=400,
        log_every=10,
        seed=313,
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
        "label": f"random_tree_delayed_cosine_hold{hold_steps}_0.1x",
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


def _mean_iae(slice_rows: List[Dict], model: str) -> float:
    key = f"iae_{model}"
    return float(np.mean([r[key] for r in slice_rows]))


def _write_report(
    path: Path,
    cfg: ExperimentConfig,
    matched_parent: List[int],
    spec_parent: Tuple[int, ...],
    train_rows: List[Dict],
    slice_rows: List[Dict],
    aggregate: Dict | None = None,
):
    mean_m = _mean_iae(slice_rows, "matched")
    mean_c = _mean_iae(slice_rows, "chain")
    imp = (mean_c - mean_m) / max(mean_c, 1e-12)
    lines = [
        "# 随机树目标：匹配 TTNS vs Chain TTNS 对比报告",
        "",
        "## 目的",
        "",
        "在**随机递归树**合成目标上，对比与目标 parent 一致的 TTNS（matched）与 chain TTNS 的 2D 切片 IAE。",
        "",
        "## 配置",
        "",
        f"- 训练配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`",
        f"- 目标 parent: `{list(spec_parent)}`",
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
            "| pair | noise_floor | IAE_matched | IAE_chain |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {pair} | {row['noise_floor']:.6f} | {row['iae_matched']:.6f} | {row['iae_chain']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 切片聚合 ((0,1),(0,3),(2,5))",
            "",
            f"- mean IAE matched: **{mean_m:.6f}**",
            f"- mean IAE chain: **{mean_c:.6f}**",
            f"- matched vs chain 提升: **{imp * 100:.1f}%**",
            "",
            "## 结论",
            "",
        ]
    )
    if imp >= 0.20:
        lines.append(f"- 匹配拓扑 TTNS 优于 chain **{imp * 100:.1f}%**，达到 M1.2 标准。")
    else:
        lines.append(f"- 匹配拓扑相对 chain 提升 **{imp * 100:.1f}%**，未达 20% 阈值。")
    if aggregate and "per_seed" in aggregate:
        lines.extend(
            [
                "",
                "## 多 seed 聚合",
                "",
                "| seed | target parent | matched | chain | m vs c |",
                "|-----:|---|---:|---:|---:|",
            ]
        )
        for row in aggregate["per_seed"]:
            lines.append(
                f"| {row['seed']} | `{row['target_parent']}` | {row['mean_iae_matched']:.6f} | "
                f"{row['mean_iae_chain']:.6f} | {row['improvement_matched_vs_chain'] * 100:.1f}% |"
            )
        lines.append(
            f"\n- 跨 seed 平均：matched vs chain **{aggregate['mean_improvement_matched_vs_chain'] * 100:.1f}%**"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_random_tree_experiment(seed: int | None = None, write_outputs: bool = True) -> Dict:
    cfg = _config()
    if seed is not None:
        cfg = replace(cfg, seed=seed)

    lr_policy = _lr_policy(cfg)
    grid_bins = 48
    n_slice_samples = 200000

    key_spec = jax.random.PRNGKey(cfg.seed + 9000)
    spec = build_random_tree_target_spec(key_spec, cfg.n_dims)
    matched_parent = list(spec.parent)
    validate_tree_parent(matched_parent)

    print(
        "random tree experiment seed=", cfg.seed,
        "target_parent=", matched_parent,
        flush=True,
    )

    key = jax.random.PRNGKey(cfg.seed)
    k_train, k_val, k_sa, k_sb = jax.random.split(key, 4)
    train_x = sample_random_tree_distribution(k_train, cfg.n_train, spec)
    val_x = sample_random_tree_distribution(k_val, cfg.n_val, spec)
    slice_a = sample_random_tree_distribution(k_sa, n_slice_samples, spec)
    slice_b = sample_random_tree_distribution(k_sb, n_slice_samples, spec)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    trained: Dict[str, Tuple] = {}
    train_rows: List[Dict] = []
    seed_offset = 0
    for name in MODEL_ORDER:
        parent = matched_parent if name == "matched" else make_parent(cfg.n_dims, "chain")
        model, parent_used, summary = _train_one_model(
            cfg=cfg,
            target_topology="random_tree",
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
    for dim_i, dim_j in SLICE_PAIRS:
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
            "iae_matched": _iae_2d(target_density, densities["matched"], dx, dy),
            "iae_chain": _iae_2d(target_density, densities["chain"], dx, dy),
        }
        slice_rows.append(row)
        print(
            f"pair=({dim_i},{dim_j}) m={row['iae_matched']:.4f} c={row['iae_chain']:.4f}",
            flush=True,
        )

    mean_m = _mean_iae(slice_rows, "matched")
    mean_c = _mean_iae(slice_rows, "chain")
    result = {
        "config": asdict(cfg),
        "target_spec_parent": list(spec.parent),
        "matched_parent": matched_parent,
        "chain_parent": make_parent(cfg.n_dims, "chain"),
        "training": train_rows,
        "slice_metrics": slice_rows,
        "mean_iae_matched": mean_m,
        "mean_iae_chain": mean_c,
        "improvement_matched_vs_chain": (mean_c - mean_m) / max(mean_c, 1e-12),
        "generated_epoch_sec": time.time(),
    }

    if not write_outputs:
        return result

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = report_dir / "random_tree_matched_vs_chain_report_zh.md"
    out_json = report_dir / "random_tree_matched_vs_chain_metrics.json"

    _write_report(out_md, cfg, matched_parent, spec.parent, train_rows, slice_rows)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "improvement matched vs chain:",
        f"{result['improvement_matched_vs_chain'] * 100:.2f}%",
        flush=True,
    )
    print("saved report:", out_md, flush=True)
    return result


def run():
    run_random_tree_experiment(write_outputs=True)


if __name__ == "__main__":
    run()
