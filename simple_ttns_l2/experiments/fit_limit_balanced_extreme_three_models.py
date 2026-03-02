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

# Ensure imports resolve to local simple_ttns_l2 and TTNSDE/ttde.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.train_l2 import build_bases, make_parent
from simple_ttns_l2.experiments.topology_comparison import ExperimentConfig
from simple_ttns_l2.experiments.topology_comparison_complex import sample_complex_tree_distribution
from simple_ttns_l2.experiments.topology_slice_visualization import (
    _empirical_pair_density,
    _eval_pair_marginal_on_grid,
    _iae_2d,
    _train_one_model,
)
from simple_ttns_l2.experiments.fit_ceiling_tt_vs_balanced import _eval_pair_marginal_tt, _train_tt_model

jax.config.update("jax_enable_x64", True)


def _config() -> ExperimentConfig:
    # Keep consistent with fit_limit_balanced_extreme.py for fair comparison.
    return ExperimentConfig(
        n_dims=6,
        q=2,
        m=96,
        rank=24,
        batch_sz=256,
        lr=1e-3,
        train_noise=1e-2,
        init_noise=1e-2,
        train_steps=600,
        log_every=20,
        seed=20260227,
        n_train=10000,
        n_val=5000,
        n_test=5000,
        monitor_train_sz=4000,
        monitor_val_sz=4000,
        early_stop_patience_logs=0,
        early_stop_min_delta=1e-6,
        early_stop_warmup_logs=0,
        early_stop_restore_best=True,
    )


def _lr_policy(cfg: ExperimentConfig) -> Dict:
    hold_steps = int(0.50 * cfg.train_steps)
    return {
        "mode": "delayed_cosine",
        "label": f"balanced_extreme_hold{hold_steps}_0.05x",
        "init_lr": float(cfg.lr),
        "final_lr": float(cfg.lr * 0.05),
        "hold_steps": hold_steps,
        "patience_logs": 0,
        "factor": 1.0,
        "min_lr": float(cfg.lr * 0.05),
        "min_delta": 0.0,
        "cooldown_logs": 0,
        "best_val": float("inf"),
        "bad_logs": 0,
        "cooldown_left": 0,
        "curr_lr": float(cfg.lr),
    }


def _draw_panel(lines: List[str], x0: float, y0: float, w: float, h: float, panel: np.ndarray, vmax: float):
    nx, ny = panel.shape
    cw = w / ny
    ch = h / nx
    denom = max(vmax, 1e-12)
    for i in range(nx):
        for j in range(ny):
            val = max(0.0, float(panel[i, j]))
            t = max(0.0, min(1.0, val / denom))
            r = int(round(255 * (1.0 - 0.95 * t)))
            g = int(round(255 * (1.0 - 0.70 * t)))
            b = int(round(255 * (1.0 - 0.20 * t)))
            x = x0 + j * cw
            y = y0 + (nx - 1 - i) * ch
            lines.append(
                f"<rect x='{x:.3f}' y='{y:.3f}' width='{cw + 0.2:.3f}' height='{ch + 0.2:.3f}' "
                f"fill='#{r:02x}{g:02x}{b:02x}' stroke='none'/>"
            )
    lines.append(
        f"<rect x='{x0:.3f}' y='{y0:.3f}' width='{w:.3f}' height='{h:.3f}' fill='none' stroke='#222' stroke-width='1'/>"
    )


def _write_svg(rows: List[Dict], out_path: Path):
    panel_w = 170
    panel_h = 170
    left = 290
    top = 92
    col_gap = 20
    row_gap = 92
    cols = ["target empirical", "balanced TTNS", "chain TTNS", "pure TT"]

    width = int(left + 4 * panel_w + 3 * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 82)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>"
        "Balanced Tree Fit Ceiling: 3 Models (High Capacity)"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Same balanced target, same training setup, compare balanced TTNS / chain TTNS / pure TT."
        "</text>"
    )

    for c, name in enumerate(cols):
        x = left + c * (panel_w + col_gap) + panel_w / 2.0
        lines.append(
            f"<text x='{x:.1f}' y='{top - 18}' text-anchor='middle' "
            f"font-family='monospace' font-size='13' fill='#111'>{name}</text>"
        )

    for r, row in enumerate(rows):
        y = top + r * (panel_h + row_gap)
        lines.append(
            f"<text x='24' y='{y + 22:.1f}' font-family='monospace' font-size='13' fill='#111'>"
            f"pair=(x{row['dim_i']},x{row['dim_j']})"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 42:.1f}' font-family='monospace' font-size='12' fill='#444'>"
            f"IAE: bal={row['iae_balanced_ttns']:.4f}, chain={row['iae_chain_ttns']:.4f}, tt={row['iae_tt']:.4f}"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"floor={row['noise_floor']:.4f}, ratios=({row['ratio_balanced_ttns']:.2f}x, "
            f"{row['ratio_chain_ttns']:.2f}x, {row['ratio_tt']:.2f}x)"
            "</text>"
        )

        v = max(
            float(np.max(row["target_density"])),
            float(np.max(np.maximum(row["balanced_ttns_density"], 0.0))),
            float(np.max(np.maximum(row["chain_ttns_density"], 0.0))),
            float(np.max(np.maximum(row["tt_density"], 0.0))),
            1e-12,
        )
        panels = [
            row["target_density"],
            row["balanced_ttns_density"],
            row["chain_ttns_density"],
            row["tt_density"],
        ]
        for c, panel in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            _draw_panel(lines, x, y, panel_w, panel_h, panel, vmax=v)
            lines.append(
                f"<text x='{x + panel_w - 6:.1f}' y='{y + panel_h + 16:.1f}' text-anchor='end' "
                f"font-family='monospace' font-size='11' fill='#555'>max={float(np.max(panel)):.3e}</text>"
            )
            min_val = float(np.min(panel))
            if min_val < -1e-12:
                lines.append(
                    f"<text x='{x + 6:.1f}' y='{y + panel_h + 16:.1f}' text-anchor='start' "
                    f"font-family='monospace' font-size='11' fill='#b00020'>min={min_val:.3e}</text>"
                )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _l2_2d(a: np.ndarray, b: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2) * dx * dy))


def _write_report(path: Path, cfg: ExperimentConfig, train_rows: List[Dict], slice_rows: List[Dict], svg_name: str):
    lines: List[str] = []
    lines.append("# Balanced Tree 高复杂度三模型对比报告")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("在同一个 `balanced target` 上，比较 `balanced TTNS / chain TTNS / pure TT` 的拟合上限。")
    lines.append("")
    lines.append(f"- 配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- 切片图: `{svg_name}`")
    lines.append("")
    lines.append("## 2. 训练结果")
    lines.append("")
    lines.append(
        "| model | lr_schedule | final_lr | final_val_l2 | best_val_l2 | stop_step | best_step | "
        "final_integral | total_time_sec |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in train_rows:
        lines.append(
            f"| {row['model_topology']} | {row['lr_schedule']} | {row['final_lr']:.3e} | "
            f"{row['final_val_l2']:.6f} | {row.get('best_val_l2', float('nan')):.6f} | "
            f"{int(row.get('stop_step', 0))} | {int(row.get('best_step', 0))} | "
            f"{row['final_integral']:.6f} | {row['total_time_sec']:.3f} |"
        )
    lines.append("")
    lines.append("## 3. 切片指标")
    lines.append("")
    lines.append(
        "| pair | noise_floor | IAE_balanced_ttns | IAE_chain_ttns | IAE_tt | "
        "ratio_balanced_ttns | ratio_chain_ttns | ratio_tt |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {pair} | {row['noise_floor']:.6f} | {row['iae_balanced_ttns']:.6f} | "
            f"{row['iae_chain_ttns']:.6f} | {row['iae_tt']:.6f} | "
            f"{row['ratio_balanced_ttns']:.3f} | {row['ratio_chain_ttns']:.3f} | {row['ratio_tt']:.3f} |"
        )
    lines.append("")

    mean_bal = float(np.mean([r["iae_balanced_ttns"] for r in slice_rows]))
    mean_chain = float(np.mean([r["iae_chain_ttns"] for r in slice_rows]))
    mean_tt = float(np.mean([r["iae_tt"] for r in slice_rows]))
    lines.append("## 4. 聚合结论")
    lines.append("")
    lines.append(
        f"- mean_IAE: balanced_ttns=`{mean_bal:.6f}`, chain_ttns=`{mean_chain:.6f}`, tt=`{mean_tt:.6f}`."
    )
    best = sorted(
        [("balanced_ttns", mean_bal), ("chain_ttns", mean_chain), ("tt", mean_tt)],
        key=lambda x: x[1],
    )
    lines.append(f"- 最优模型: `{best[0][0]}`。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    cfg = _config()
    lr_policy = _lr_policy(cfg)
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 56
    n_slice_samples = 220000

    print("running balanced extreme three-model comparison with config:", asdict(cfg), flush=True)
    print("lr_policy:", lr_policy["label"], flush=True)

    target_topology = "balanced"
    parent_target = make_parent(cfg.n_dims, target_topology)
    key = jax.random.PRNGKey(cfg.seed + 33333)
    k_train, k_val, k_slice_a, k_slice_b = jax.random.split(key, 4)

    train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
    val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
    slice_a = sample_complex_tree_distribution(k_slice_a, n_slice_samples, parent_target)
    slice_b = sample_complex_tree_distribution(k_slice_b, n_slice_samples, parent_target)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    train_rows: List[Dict] = []

    bal_model, bal_parent, bal_summary = _train_one_model(
        cfg=cfg,
        target_topology=target_topology,
        model_topology="balanced",
        train_x=train_x,
        val_x=val_x,
        bases=bases,
        gram_matrices=gram_matrices,
        basis_integrals=basis_integrals,
        seed_offset=101,
        lr_policy_template=lr_policy,
    )
    bal_summary["model_topology"] = "balanced_ttns"
    train_rows.append(bal_summary)

    chain_model, chain_parent, chain_summary = _train_one_model(
        cfg=cfg,
        target_topology=target_topology,
        model_topology="chain",
        train_x=train_x,
        val_x=val_x,
        bases=bases,
        gram_matrices=gram_matrices,
        basis_integrals=basis_integrals,
        seed_offset=202,
        lr_policy_template=lr_policy,
    )
    chain_summary["model_topology"] = "chain_ttns"
    train_rows.append(chain_summary)

    tt_model, tt_summary = _train_tt_model(
        cfg=cfg,
        target_topology=target_topology,
        train_x=train_x,
        val_x=val_x,
        bases=bases,
        gram_matrices=gram_matrices,
        basis_integrals=basis_integrals,
        seed_offset=303,
    )
    train_rows.append(tt_summary)

    a_np = np.asarray(slice_a)
    b_np = np.asarray(slice_b)
    slice_rows: List[Dict] = []
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
        target_density_2 = _empirical_pair_density(b_np, dim_i, dim_j, xi_edges, xj_edges)

        bal_density = _eval_pair_marginal_on_grid(
            ttns=bal_model,
            parent=bal_parent,
            bases=bases,
            basis_integrals=basis_integrals,
            dim_i=dim_i,
            dim_j=dim_j,
            xi_centers=xi_centers,
            xj_centers=xj_centers,
        )
        chain_density = _eval_pair_marginal_on_grid(
            ttns=chain_model,
            parent=chain_parent,
            bases=bases,
            basis_integrals=basis_integrals,
            dim_i=dim_i,
            dim_j=dim_j,
            xi_centers=xi_centers,
            xj_centers=xj_centers,
        )
        tt_density = _eval_pair_marginal_tt(
            tt=tt_model,
            bases=bases,
            basis_integrals=basis_integrals,
            dim_i=dim_i,
            dim_j=dim_j,
            xi_centers=xi_centers,
            xj_centers=xj_centers,
        )

        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])
        noise_floor = _iae_2d(target_density, target_density_2, dx, dy)
        iae_bal = _iae_2d(target_density, bal_density, dx, dy)
        iae_chain = _iae_2d(target_density, chain_density, dx, dy)
        iae_tt = _iae_2d(target_density, tt_density, dx, dy)
        l2_bal = _l2_2d(target_density, bal_density, dx, dy)
        l2_chain = _l2_2d(target_density, chain_density, dx, dy)
        l2_tt = _l2_2d(target_density, tt_density, dx, dy)

        row = {
            "target_topology": target_topology,
            "dim_i": dim_i,
            "dim_j": dim_j,
            "noise_floor": noise_floor,
            "iae_balanced_ttns": iae_bal,
            "iae_chain_ttns": iae_chain,
            "iae_tt": iae_tt,
            "ratio_balanced_ttns": float(iae_bal / max(noise_floor, 1e-12)),
            "ratio_chain_ttns": float(iae_chain / max(noise_floor, 1e-12)),
            "ratio_tt": float(iae_tt / max(noise_floor, 1e-12)),
            "l2_balanced_ttns": l2_bal,
            "l2_chain_ttns": l2_chain,
            "l2_tt": l2_tt,
            "target_density": target_density,
            "balanced_ttns_density": bal_density,
            "chain_ttns_density": chain_density,
            "tt_density": tt_density,
        }
        slice_rows.append(row)
        print(
            f"slice pair=({dim_i},{dim_j}), floor={noise_floor:.6f}, "
            f"IAE_bal={iae_bal:.6f}, IAE_chain={iae_chain:.6f}, IAE_tt={iae_tt:.6f}",
            flush=True,
        )

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_svg = report_dir / "fit_limit_balanced_extreme_three_models.svg"
    out_md = report_dir / "fit_limit_balanced_extreme_three_models_report_zh.md"
    out_json = report_dir / "fit_limit_balanced_extreme_three_models_metrics.json"

    _write_svg(slice_rows, out_svg)
    _write_report(out_md, cfg, train_rows, slice_rows, out_svg.name)

    compact_rows = []
    for row in slice_rows:
        compact_rows.append(
            {
                "dim_i": row["dim_i"],
                "dim_j": row["dim_j"],
                "noise_floor": row["noise_floor"],
                "iae_balanced_ttns": row["iae_balanced_ttns"],
                "iae_chain_ttns": row["iae_chain_ttns"],
                "iae_tt": row["iae_tt"],
                "ratio_balanced_ttns": row["ratio_balanced_ttns"],
                "ratio_chain_ttns": row["ratio_chain_ttns"],
                "ratio_tt": row["ratio_tt"],
                "l2_balanced_ttns": row["l2_balanced_ttns"],
                "l2_chain_ttns": row["l2_chain_ttns"],
                "l2_tt": row["l2_tt"],
            }
        )
    out_json.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "lr_policy": lr_policy,
                "training": train_rows,
                "slice_metrics": compact_rows,
                "mean_iae_balanced_ttns": float(np.mean([r["iae_balanced_ttns"] for r in slice_rows])),
                "mean_iae_chain_ttns": float(np.mean([r["iae_chain_ttns"] for r in slice_rows])),
                "mean_iae_tt": float(np.mean([r["iae_tt"] for r in slice_rows])),
                "generated_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== balanced extreme three-model comparison done ===", flush=True)
    print("saved svg   :", out_svg, flush=True)
    print("saved report:", out_md, flush=True)
    print("saved metric:", out_json, flush=True)


if __name__ == "__main__":
    run()

