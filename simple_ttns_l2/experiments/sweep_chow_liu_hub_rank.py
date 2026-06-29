"""扫描：给 Chow–Liu 树的高扇出枢纽边降 rank，换取速度（单 seed）。

背景：fork DAG 的 Chow–Liu 最优树把中心变量 x2 选为高扇出 hub（度=4），
单节点收缩量级 ~ rank^(fanout+1) = rank^4，是训练瓶颈。本扫描只对 hub 关联边
降 rank（非 hub 边保持高 rank），观察 速度 / val_l2 / 关键切片 IAE 的权衡曲线。
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree
from simple_ttns_l2.experiments.dag_target import fork_dag_edges, sample_fork_dag_distribution
from simple_ttns_l2.experiments.fit_dag_junction_vs_chain import _config, _key_pairs, _lr_policy
from simple_ttns_l2.experiments.topology_slice_visualization import (
    _empirical_pair_density,
    _eval_pair_marginal_on_grid,
    _iae_2d,
    _train_one_model,
)
from simple_ttns_l2.train_l2 import build_bases, make_parent

SEED = 20260227
N_BINS = 16
CHOW_LIU_ROOT = 0
HUB_RANKS = [16, 12, 8, 4, 2]  # 16 == baseline（全 16）
BASE_RANK = 16


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _hub_edges(parent: List[int]) -> Tuple[int, List[Tuple[int, int]]]:
    """返回 (度最大的 hub 节点, 其关联无向边列表)。"""
    deg = Counter()
    edges = []
    for node, p in enumerate(parent):
        if p == node:
            continue
        edges.append(_edge_key(node, p))
        deg[node] += 1
        deg[p] += 1
    hub = max(deg, key=deg.get)
    hub_edges = sorted(e for e in edges if hub in e)
    return hub, hub_edges


def _perstep_ms(history: List[Dict]) -> float:
    intervals = [h["interval_sec"] for h in history[2:]]  # 跳过 JIT 预热
    return (st.median(intervals) / 10.0 * 1000.0) if intervals else float("nan")


def _key_slice_mean_iae(model, parent, bases, basis_integrals, target_densities, grids, key_pairs) -> float:
    vals = []
    for (i, j) in key_pairs:
        xi_centers, xj_centers, dx, dy = grids[(i, j)]
        density = _eval_pair_marginal_on_grid(model, parent, bases, basis_integrals, i, j, xi_centers, xj_centers)
        vals.append(_iae_2d(target_densities[(i, j)], density, dx, dy))
    return float(np.mean(vals))


def run():
    cfg = replace(_config(), seed=SEED)
    lr_policy = _lr_policy(cfg)
    dag_edges = fork_dag_edges(cfg.n_dims)
    key_pairs = _key_pairs(cfg.n_dims)
    grid_bins = 48
    n_slice = 100000

    key = jax.random.PRNGKey(cfg.seed)
    k_train, k_val, k_sa = jax.random.split(key, 3)
    train_x = sample_fork_dag_distribution(k_train, cfg.n_train, n_dims=cfg.n_dims)
    val_x = sample_fork_dag_distribution(k_val, cfg.n_val, n_dims=cfg.n_dims)
    slice_a = sample_fork_dag_distribution(k_sa, n_slice, n_dims=cfg.n_dims)

    chow_liu = estimate_chow_liu_tree(np.asarray(train_x), n_bins=N_BINS, root=CHOW_LIU_ROOT)
    cl_parent = list(chow_liu.parent)
    hub, hub_edges = _hub_edges(cl_parent)
    print(f"chow_liu parent = {cl_parent}", flush=True)
    print(f"hub node = {hub} (fanout-bearing), hub edges = {hub_edges}", flush=True)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    # 预计算关键切片的目标密度与网格（与模型无关）
    a_np = np.asarray(slice_a)
    target_densities = {}
    grids = {}
    for (i, j) in key_pairs:
        xi_lo, xi_hi = np.quantile(a_np[:, i], 0.01), np.quantile(a_np[:, i], 0.99)
        xj_lo, xj_hi = np.quantile(a_np[:, j], 0.01), np.quantile(a_np[:, j], 0.99)
        xi_pad = 0.05 * (xi_hi - xi_lo + 1e-12)
        xj_pad = 0.05 * (xj_hi - xj_lo + 1e-12)
        xi_edges = np.linspace(xi_lo - xi_pad, xi_hi + xi_pad, grid_bins + 1)
        xj_edges = np.linspace(xj_lo - xj_pad, xj_hi + xj_pad, grid_bins + 1)
        xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        xj_centers = 0.5 * (xj_edges[:-1] + xj_edges[1:])
        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])
        target_densities[(i, j)] = _empirical_pair_density(a_np, i, j, xi_edges, xj_edges)
        grids[(i, j)] = (xi_centers, xj_centers, dx, dy)

    rows = []

    # chain 下界参考
    chain_model, chain_parent, chain_sum = _train_one_model(
        cfg=cfg, target_topology="fork_dag", model_topology="chain",
        train_x=train_x, val_x=val_x, bases=bases, gram_matrices=gram_matrices,
        basis_integrals=basis_integrals, seed_offset=0, lr_policy_template=lr_policy,
        model_parent=make_parent(cfg.n_dims, "chain"),
    )
    chain_iae = _key_slice_mean_iae(chain_model, chain_parent, bases, basis_integrals, target_densities, grids, key_pairs)
    rows.append({
        "config": "chain (baseline)", "hub_rank": None,
        "ms_step": _perstep_ms(chain_sum["history"]), "total_s": chain_sum["total_time_sec"],
        "stop_step": chain_sum["stop_step"], "val_l2": chain_sum["final_val_l2"], "key_iae": chain_iae,
    })

    seed_offset = 50
    for hr in HUB_RANKS:
        edge_ranks = None if hr == BASE_RANK else {e: hr for e in hub_edges}
        model, parent_used, summary = _train_one_model(
            cfg=cfg, target_topology="fork_dag", model_topology="chow_liu",
            train_x=train_x, val_x=val_x, bases=bases, gram_matrices=gram_matrices,
            basis_integrals=basis_integrals, seed_offset=seed_offset, lr_policy_template=lr_policy,
            model_parent=cl_parent, model_edge_ranks=edge_ranks,
        )
        seed_offset += 50
        iae = _key_slice_mean_iae(model, parent_used, bases, basis_integrals, target_densities, grids, key_pairs)
        label = f"chow_liu hub_rank={hr}" + (" (baseline=16)" if hr == BASE_RANK else "")
        rows.append({
            "config": label, "hub_rank": hr,
            "ms_step": _perstep_ms(summary["history"]), "total_s": summary["total_time_sec"],
            "stop_step": summary["stop_step"], "val_l2": summary["final_val_l2"], "key_iae": iae,
        })

    base = next(r for r in rows if r["hub_rank"] == BASE_RANK)

    print("\n=== Hub-rank sweep (Chow-Liu, fork DAG 7D, seed %d) ===" % SEED)
    print(f"hub node {hub}, hub edges {hub_edges}, non-hub edges rank={BASE_RANK}")
    header = f"{'config':<28} {'ms/step':>8} {'speedup':>8} {'total_s':>8} {'steps':>6} {'val_l2':>9} {'key_IAE':>8}"
    print(header)
    for r in rows:
        sp = base["ms_step"] / r["ms_step"] if r["ms_step"] else float("nan")
        print(f"{r['config']:<28} {r['ms_step']:8.1f} {sp:7.2f}x {r['total_s']:8.1f} "
              f"{r['stop_step']:6d} {r['val_l2']:9.4f} {r['key_iae']:8.4f}")

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    out_json = report_dir / "chow_liu_hub_rank_sweep_metrics.json"
    out_json.write_text(json.dumps({
        "seed": SEED, "chow_liu_parent": cl_parent, "hub_node": hub,
        "hub_edges": [list(e) for e in hub_edges], "base_rank": BASE_RANK,
        "key_pairs": [list(p) for p in key_pairs], "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md = report_dir / "chow_liu_hub_rank_sweep_report_zh.md"
    lines = [
        "# Chow–Liu 枢纽边降 rank 速度扫描（单 seed）",
        "",
        "## 目的",
        "",
        "fork DAG 的 Chow–Liu 最优树把中心变量 $x_2$ 选为高扇出枢纽（hub），",
        "单节点收缩量级约 $\\text{rank}^{(\\text{fanout}+1)}$，是训练瓶颈。",
        "本扫描**只对 hub 关联边降 rank**（非 hub 边保持 rank=16），观察速度与精度权衡。",
        "",
        "## 配置",
        "",
        f"- seed: {SEED}，7D fork DAG，非 hub 边 rank={BASE_RANK}",
        f"- Chow–Liu parent: `{cl_parent}`",
        f"- hub 节点: **{hub}**（度最大），hub 边: `{[list(e) for e in hub_edges]}`",
        f"- 关键切片: `{[list(p) for p in key_pairs]}`",
        "- 速度口径: 稳态每步耗时（跳过前 2 个 log 的 JIT 预热，interval/10）",
        "",
        "## 结果",
        "",
        "| 配置 | ms/step | 提速 | 总耗时(s) | 步数 | val_l2 | 关键切片 IAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        sp = base["ms_step"] / r["ms_step"] if r["ms_step"] else float("nan")
        lines.append(
            f"| {r['config']} | {r['ms_step']:.1f} | {sp:.2f}x | {r['total_s']:.1f} | "
            f"{r['stop_step']} | {r['val_l2']:.4f} | {r['key_iae']:.4f} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- 提速基准为 `hub_rank=16`（即原始全 rank=16 的 Chow–Liu）。",
        "- `val_l2` 越低越好；关键切片 IAE 越低越好。",
        "- chain 为速度下界 / 精度上界参考。",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nsaved:", out_md, flush=True)


if __name__ == "__main__":
    run()
