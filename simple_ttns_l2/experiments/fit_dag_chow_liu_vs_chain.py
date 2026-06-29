"""在 7D fork DAG 目标上对比数据驱动的 Chow–Liu TTNS vs balanced vs chain。

动机：联结树 junction 是手工拼的过渡方案（在单点强行挂多子节点），既慢又不一定优。
Chow–Liu 从训练数据自动估计互信息最大生成树，无需手工指定拓扑。本实验检验：
数据驱动的 Chow–Liu 树能否自动发现 fork DAG 的强依赖、并在关键切片上优于 chain。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree
from simple_ttns_l2.experiments.dag_target import fork_dag_edges, sample_fork_dag_distribution
from simple_ttns_l2.experiments.fit_dag_junction_vs_chain import (
    _aggregate_key_slice,
    _all_pairs,
    _config,
    _key_pairs,
    _lr_policy,
)
from simple_ttns_l2.experiments.topology_slice_visualization import (
    _empirical_pair_density,
    _eval_pair_marginal_on_grid,
    _iae_2d,
    _train_one_model,
)
from simple_ttns_l2.train_l2 import build_bases, make_parent

MODEL_ORDER = ("chow_liu", "balanced", "chain")
SEEDS = [313, 2602, 20260227]
N_BINS = 16
CHOW_LIU_ROOT = 0


def _model_parent(name: str, n_dims: int, chow_liu_parent: Sequence[int]) -> List[int]:
    if name == "chow_liu":
        return list(chow_liu_parent)
    return make_parent(n_dims, name)


def _matched_dag_edges(chow_liu_parent: Sequence[int], dag_edges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """统计 Chow–Liu 树边命中了哪些 DAG 真实边（无向）。"""
    cl_undirected = {tuple(sorted((node, p))) for node, p in enumerate(chow_liu_parent) if p != node}
    dag_undirected = {tuple(sorted(e)) for e in dag_edges}
    return sorted(cl_undirected & dag_undirected)


def run_chow_liu_experiment(seed: int | None = None, n_dims: int | None = None) -> Dict:
    cfg = _config()
    if seed is not None:
        cfg = replace(cfg, seed=seed)
    if n_dims is not None:
        cfg = replace(cfg, n_dims=n_dims)

    lr_policy = _lr_policy(cfg)
    dag_edges = fork_dag_edges(cfg.n_dims)
    grid_bins = 48
    n_slice_samples = 200000

    key_pairs = _key_pairs(cfg.n_dims)
    all_pairs = _all_pairs(cfg.n_dims)

    print("Chow-Liu DAG experiment seed=", cfg.seed, "n_dims=", cfg.n_dims, "rank=", cfg.rank, flush=True)

    key = jax.random.PRNGKey(cfg.seed)
    k_train, k_val, k_sa, k_sb = jax.random.split(key, 4)
    train_x = sample_fork_dag_distribution(k_train, cfg.n_train, n_dims=cfg.n_dims)
    val_x = sample_fork_dag_distribution(k_val, cfg.n_val, n_dims=cfg.n_dims)
    slice_a = sample_fork_dag_distribution(k_sa, n_slice_samples, n_dims=cfg.n_dims)
    slice_b = sample_fork_dag_distribution(k_sb, n_slice_samples, n_dims=cfg.n_dims)

    chow_liu = estimate_chow_liu_tree(np.asarray(train_x), n_bins=N_BINS, root=CHOW_LIU_ROOT)
    chow_liu_parent = list(chow_liu.parent)
    matched_edges = _matched_dag_edges(chow_liu_parent, dag_edges)
    print("chow_liu parent =", chow_liu_parent, flush=True)
    print("chow_liu edges  =", list(chow_liu.edges), flush=True)
    print("matched DAG edges =", matched_edges, f"({len(matched_edges)}/{len(dag_edges)})", flush=True)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    trained: Dict[str, Tuple] = {}
    train_rows: List[Dict] = []
    seed_offset = 0
    for name in MODEL_ORDER:
        parent = _model_parent(name, cfg.n_dims, chow_liu_parent)
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

        row = {"dim_i": dim_i, "dim_j": dim_j, "noise_floor": floor}
        for name in MODEL_ORDER:
            model, parent = trained[name]
            density = _eval_pair_marginal_on_grid(
                model, parent, bases, basis_integrals, dim_i, dim_j, xi_centers, xj_centers
            )
            row[f"iae_{name}"] = _iae_2d(target_density, density, dx, dy)
        slice_rows.append(row)
        print(
            f"pair=({dim_i},{dim_j}) "
            + " ".join(f"{name[:2]}={row[f'iae_{name}']:.4f}" for name in MODEL_ORDER),
            flush=True,
        )

    means = {name: _aggregate_key_slice(slice_rows, name, key_pairs) for name in MODEL_ORDER}
    mean_cl = means["chow_liu"]
    mean_c = means["chain"]
    result = {
        "config": asdict(cfg),
        "dag_edges": [list(e) for e in dag_edges],
        "chow_liu_parent": chow_liu_parent,
        "chow_liu_edges": [list(e) for e in chow_liu.edges],
        "matched_dag_edges": [list(e) for e in matched_edges],
        "n_matched_edges": len(matched_edges),
        "balanced_parent": make_parent(cfg.n_dims, "balanced"),
        "chain_parent": make_parent(cfg.n_dims, "chain"),
        "training": train_rows,
        "slice_metrics": slice_rows,
        "key_slice_mean_iae_chow_liu": mean_cl,
        "key_slice_mean_iae_balanced": means["balanced"],
        "key_slice_mean_iae_chain": mean_c,
        "key_slice_improvement_chow_liu_vs_chain": (mean_c - mean_cl) / max(mean_c, 1e-12),
        "generated_epoch_sec": time.time(),
    }
    return result


def _write_report(path: Path, aggregate: Dict, last: Dict) -> None:
    cfg = last["config"]
    key_pairs = _key_pairs(cfg["n_dims"])
    key_label = ",".join(f"({i},{j})" for i, j in key_pairs)
    means = {
        "chow_liu": last["key_slice_mean_iae_chow_liu"],
        "balanced": last["key_slice_mean_iae_balanced"],
        "chain": last["key_slice_mean_iae_chain"],
    }
    winner = min(means, key=means.get)
    imp = last["key_slice_improvement_chow_liu_vs_chain"]

    lines = [
        "# Fork DAG：数据驱动 Chow–Liu / Balanced / Chain TTNS 对比报告",
        "",
        "## 目的",
        "",
        "在含**双父节点**依赖（$x_2 \\sim f(x_0, x_1)$）的合成 fork DAG 目标上，",
        "用**数据驱动的 Chow–Liu 最大生成树**（从训练样本估互信息）替代手工联结树，",
        "对比 Chow–Liu TTNS、balanced TTNS、chain TTNS 的 2D 切片 IAE 与训练耗时。",
        "",
        "## 配置",
        "",
        f"- 训练配置: `{json.dumps(cfg, ensure_ascii=False)}`",
        f"- DAG 边: `{last['dag_edges']}`",
        f"- Chow–Liu 估树（root={CHOW_LIU_ROOT}, n_bins={N_BINS}）parent: `{last['chow_liu_parent']}`",
        f"- Chow–Liu 树边: `{last['chow_liu_edges']}`",
        f"- 命中 DAG 真边: `{last['matched_dag_edges']}`（{last['n_matched_edges']}/{len(last['dag_edges'])}）",
        f"- balanced parent: `{last['balanced_parent']}`",
        f"- chain parent: `{last['chain_parent']}`",
        "",
        "## 训练结果",
        "",
        "| model | final_val_l2 | final_integral | total_time_sec |",
        "|---|---:|---:|---:|",
    ]
    for row in last["training"]:
        lines.append(
            f"| {row['model_topology']} | {row['final_val_l2']:.6f} | "
            f"{row['final_integral']:.6f} | {row['total_time_sec']:.3f} |"
        )

    lines += ["", "## 切片 IAE", "", "| pair | noise_floor | IAE_chow_liu | IAE_balanced | IAE_chain |", "|---|---:|---:|---:|---:|"]
    for r in last["slice_metrics"]:
        lines.append(
            f"| ({r['dim_i']},{r['dim_j']}) | {r['noise_floor']:.6f} | "
            f"{r['iae_chow_liu']:.6f} | {r['iae_balanced']:.6f} | {r['iae_chain']:.6f} |"
        )

    lines += [
        "",
        f"## 关键切片聚合 ({key_label})",
        "",
        f"- mean IAE chow_liu: **{means['chow_liu']:.6f}**",
        f"- mean IAE balanced: **{means['balanced']:.6f}**",
        f"- mean IAE chain: **{means['chain']:.6f}**",
        f"- chow_liu vs chain 提升: **{imp * 100:.1f}%**",
        f"- 最优模型: **{winner}**",
        "",
        "## 结论",
        "",
    ]
    if imp >= 0.20:
        lines.append(f"- 数据驱动 Chow–Liu TTNS 在关键切片上优于 chain **{imp * 100:.1f}%**，达到 20% 标准。")
    else:
        lines.append(f"- Chow–Liu 相对 chain 提升 **{imp * 100:.1f}%**，未达 20% 阈值。")

    if "per_seed" in aggregate:
        lines += ["", "## 多 seed 聚合", "", "| seed | chow_liu | balanced | chain | cl vs c | 命中边 |", "|-----:|---:|---:|---:|---:|---:|"]
        for r in aggregate["per_seed"]:
            lines.append(
                f"| {r['seed']} | {r['mean_iae_chow_liu']:.6f} | {r['mean_iae_balanced']:.6f} | "
                f"{r['mean_iae_chain']:.6f} | {r['improvement_chow_liu_vs_chain'] * 100:.1f}% | "
                f"{r['n_matched_edges']}/{len(last['dag_edges'])} |"
            )
        lines.append(
            f"\n- 跨 seed 平均：chow_liu vs chain **{aggregate['mean_improvement_chow_liu_vs_chain'] * 100:.1f}%**"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(n_dims: int = 7):
    per_seed = []
    for seed in SEEDS:
        print(f"=== Chow-Liu multiseed seed {seed} ({n_dims}D) ===", flush=True)
        r = run_chow_liu_experiment(seed=seed, n_dims=n_dims)
        per_seed.append(
            {
                "seed": seed,
                "mean_iae_chow_liu": r["key_slice_mean_iae_chow_liu"],
                "mean_iae_balanced": r["key_slice_mean_iae_balanced"],
                "mean_iae_chain": r["key_slice_mean_iae_chain"],
                "improvement_chow_liu_vs_chain": r["key_slice_improvement_chow_liu_vs_chain"],
                "n_matched_edges": r["n_matched_edges"],
                "chow_liu_parent": r["chow_liu_parent"],
            }
        )

    mean_imp = sum(r["improvement_chow_liu_vs_chain"] for r in per_seed) / len(per_seed)
    aggregate = {"per_seed": per_seed, "mean_improvement_chow_liu_vs_chain": mean_imp}

    last = run_chow_liu_experiment(seed=SEEDS[-1], n_dims=n_dims)
    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if n_dims == 7 else f"_{n_dims}d"
    out_md = report_dir / f"dag_chow_liu_vs_chain{suffix}_multiseed_report_zh.md"
    out_json = report_dir / f"dag_chow_liu_vs_chain{suffix}_multiseed_metrics.json"
    _write_report(out_md, aggregate, last)
    out_json.write_text(
        json.dumps({"aggregate": aggregate, "last_seed_detail": last}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("mean improvement chow_liu vs chain across seeds:", f"{mean_imp * 100:.2f}%", flush=True)
    print("saved:", out_md, flush=True)


if __name__ == "__main__":
    n_dims = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    run(n_dims=n_dims)
