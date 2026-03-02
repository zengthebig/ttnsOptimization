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

jax.config.update("jax_enable_x64", True)


def _config() -> ExperimentConfig:
    # Higher-capacity "limit" setup for balanced target only.
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


def _draw_heatmap(lines: List[str], x0: float, y0: float, w: float, h: float, data: np.ndarray, vmax: float):
    nx, ny = data.shape
    cw = w / ny
    ch = h / nx
    denom = max(vmax, 1e-12)
    for i in range(nx):
        for j in range(ny):
            val = max(0.0, float(data[i, j]))
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
        f"<rect x='{x0:.3f}' y='{y0:.3f}' width='{w:.3f}' height='{h:.3f}' "
        "fill='none' stroke='#222' stroke-width='1'/>"
    )


def _write_svg(rows: List[Dict], out_svg: Path):
    panel_w = 200
    panel_h = 200
    left = 260
    top = 86
    col_gap = 30
    row_gap = 96
    cols = ["target empirical", "balanced model", "abs diff"]

    width = int(left + 3 * panel_w + 2 * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 80)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>"
        "Balanced Tree Fit Limit (High Capacity)"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Target=balanced tree, model=balanced TTNS. abs diff panel = |target - model|."
        "</text>"
    )

    for c, name in enumerate(cols):
        x = left + c * (panel_w + col_gap) + panel_w / 2.0
        lines.append(
            f"<text x='{x:.1f}' y='{top - 18}' text-anchor='middle' "
            f"font-family='monospace' font-size='14' fill='#111'>{name}</text>"
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
            f"IAE={row['iae']:.4f}, floor={row['noise_iae']:.4f}, ratio={row['ratio_to_floor']:.2f}x, "
            f"L2={row['l2']:.4f}</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"x{row['dim_i']} in [{row['xi_min']:.2f},{row['xi_max']:.2f}], "
            f"x{row['dim_j']} in [{row['xj_min']:.2f},{row['xj_max']:.2f}]</text>"
        )

        panels = [row["target_density"], row["model_density"], row["abs_diff"]]
        v0 = max(float(np.max(panels[0])), float(np.max(panels[1])), 1e-12)
        vd = max(float(np.max(panels[2])), 1e-12)
        vmaxs = [v0, v0, vd]
        for c, panel in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            _draw_heatmap(lines, x, y, panel_w, panel_h, panel, vmax=vmaxs[c])
            lines.append(
                f"<text x='{x + panel_w - 6:.1f}' y='{y + panel_h + 16:.1f}' text-anchor='end' "
                f"font-family='monospace' font-size='11' fill='#555'>max={float(np.max(panel)):.3e}</text>"
            )

    lines.append("</svg>")
    out_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(report_path: Path, cfg: ExperimentConfig, train_summary: Dict, slice_rows: List[Dict], svg_name: str):
    lines: List[str] = []
    lines.append("# Balanced Tree 极限拟合报告（高复杂度）")
    lines.append("")
    lines.append("## 实验目标")
    lines.append("")
    lines.append("仅针对 `balanced target`，提升模型复杂度并评估 balanced TTNS 的拟合上限。")
    lines.append("")
    lines.append(f"- Config: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- Figure: `{svg_name}`")
    lines.append("")
    lines.append("## 训练摘要")
    lines.append("")
    lines.append("| model | lr_schedule | final_lr | final_val_l2 | best_val_l2 | stop_step | best_step | final_integral | total_time_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {train_summary['model_topology']} | {train_summary['lr_schedule']} | "
        f"{train_summary['final_lr']:.3e} | {train_summary['final_val_l2']:.6f} | "
        f"{train_summary['best_val_l2']:.6f} | {train_summary['stop_step']} | {train_summary['best_step']} | "
        f"{train_summary['final_integral']:.6f} | {train_summary['total_time_sec']:.3f} |"
    )
    lines.append("")
    lines.append("## 切片拟合摘要")
    lines.append("")
    lines.append("| pair | IAE(model,target) | IAE(target,target2) | ratio_to_floor | L2 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {pair} | {row['iae']:.6f} | {row['noise_iae']:.6f} | {row['ratio_to_floor']:.3f} | {row['l2']:.6f} |"
        )
    lines.append("")
    lines.append(
        f"- Mean IAE = `{float(np.mean([r['iae'] for r in slice_rows])):.6f}`, "
        f"Mean ratio_to_floor = `{float(np.mean([r['ratio_to_floor'] for r in slice_rows])):.3f}`."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _l2_2d(a: np.ndarray, b: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2) * dx * dy))


def run():
    cfg = _config()
    lr_policy_template = _lr_policy(cfg)
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 56
    n_slice_samples = 220000

    print("running balanced extreme fit with config:", asdict(cfg), flush=True)
    print("lr_policy:", lr_policy_template["label"], flush=True)

    target_topology = "balanced"
    model_topology = "balanced"
    parent_target = make_parent(cfg.n_dims, target_topology)
    key = jax.random.PRNGKey(cfg.seed + 12345)
    k_train, k_val, k_slice_a, k_slice_b = jax.random.split(key, 4)

    train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
    val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
    slice_a = sample_complex_tree_distribution(k_slice_a, n_slice_samples, parent_target)
    slice_b = sample_complex_tree_distribution(k_slice_b, n_slice_samples, parent_target)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    model, model_parent, train_summary = _train_one_model(
        cfg=cfg,
        target_topology=target_topology,
        model_topology=model_topology,
        train_x=train_x,
        val_x=val_x,
        bases=bases,
        gram_matrices=gram_matrices,
        basis_integrals=basis_integrals,
        seed_offset=54321,
        lr_policy_template=lr_policy_template,
    )

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
        model_density = _eval_pair_marginal_on_grid(
            ttns=model,
            parent=model_parent,
            bases=bases,
            basis_integrals=basis_integrals,
            dim_i=dim_i,
            dim_j=dim_j,
            xi_centers=xi_centers,
            xj_centers=xj_centers,
        )
        abs_diff = np.abs(target_density - model_density)

        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])
        iae = _iae_2d(target_density, model_density, dx, dy)
        noise_iae = _iae_2d(target_density, target_density_2, dx, dy)
        l2 = _l2_2d(target_density, model_density, dx, dy)
        ratio = float(iae / max(noise_iae, 1e-12))

        slice_rows.append(
            {
                "dim_i": dim_i,
                "dim_j": dim_j,
                "xi_min": float(xi_edges[0]),
                "xi_max": float(xi_edges[-1]),
                "xj_min": float(xj_edges[0]),
                "xj_max": float(xj_edges[-1]),
                "target_density": target_density,
                "model_density": model_density,
                "abs_diff": abs_diff,
                "iae": iae,
                "noise_iae": noise_iae,
                "ratio_to_floor": ratio,
                "l2": l2,
            }
        )
        print(
            f"slice pair=({dim_i},{dim_j}), IAE={iae:.6f}, floor={noise_iae:.6f}, "
            f"ratio={ratio:.3f}, L2={l2:.6f}",
            flush=True,
        )

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_svg = report_dir / "fit_limit_balanced_extreme.svg"
    out_md = report_dir / "fit_limit_balanced_extreme_report_zh.md"
    out_json = report_dir / "fit_limit_balanced_extreme_metrics.json"

    _write_svg(slice_rows, out_svg)
    _write_report(out_md, cfg, train_summary, slice_rows, out_svg.name)

    compact_rows = []
    for row in slice_rows:
        compact_rows.append(
            {
                "dim_i": row["dim_i"],
                "dim_j": row["dim_j"],
                "iae": row["iae"],
                "noise_iae": row["noise_iae"],
                "ratio_to_floor": row["ratio_to_floor"],
                "l2": row["l2"],
            }
        )
    out_json.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "lr_policy": lr_policy_template,
                "training": train_summary,
                "slice_metrics": compact_rows,
                "mean_iae": float(np.mean([r["iae"] for r in slice_rows])),
                "mean_ratio_to_floor": float(np.mean([r["ratio_to_floor"] for r in slice_rows])),
                "generated_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== balanced extreme fit done ===", flush=True)
    print("saved svg   :", out_svg, flush=True)
    print("saved report:", out_md, flush=True)
    print("saved metric:", out_json, flush=True)


if __name__ == "__main__":
    run()

