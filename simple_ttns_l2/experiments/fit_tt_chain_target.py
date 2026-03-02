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
from simple_ttns_l2.experiments.topology_slice_visualization import _empirical_pair_density, _iae_2d
from simple_ttns_l2.experiments.fit_ceiling_tt_vs_balanced import _eval_pair_marginal_tt, _train_tt_model

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
        init_noise=1e-2,
        train_steps=300,
        log_every=10,
        seed=910,
        n_train=8000,
        n_val=4000,
        n_test=4000,
        monitor_train_sz=3000,
        monitor_val_sz=3000,
    )


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
    panel_w = 220
    panel_h = 220
    left = 250
    top = 86
    col_gap = 36
    row_gap = 96
    cols = ["target empirical", "model TT", "|target - TT|"]

    width = int(left + 3 * panel_w + 2 * col_gap + 40)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 80)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append("<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>Pure TT Fit On Chain Target</text>")
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Direct TT training only (no chained TTNS), complex chain target."
        "</text>"
    )

    for c, name in enumerate(cols):
        x = left + c * (panel_w + col_gap) + panel_w / 2.0
        lines.append(
            f"<text x='{x:.1f}' y='{top - 18}' text-anchor='middle' font-family='monospace' "
            f"font-size='14' fill='#111'>{name}</text>"
        )

    for r, row in enumerate(rows):
        y = top + r * (panel_h + row_gap)
        lines.append(
            f"<text x='24' y='{y + 22:.1f}' font-family='monospace' font-size='13' fill='#111'>"
            f"pair=(x{row['dim_i']},x{row['dim_j']})</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 42:.1f}' font-family='monospace' font-size='12' fill='#444'>"
            f"IAE={row['iae']:.4f}, floor={row['noise_floor']:.4f}, ratio={row['ratio']:.2f}x</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"L2={row['l2']:.4f}</text>"
        )

        p0 = row["target_density"]
        p1 = row["tt_density"]
        p2 = row["abs_diff"]
        vmax_main = max(float(np.max(p0)), float(np.max(np.maximum(p1, 0.0))), 1e-12)
        vmax_diff = max(float(np.max(p2)), 1e-12)
        panels = [p0, p1, p2]
        vmaxs = [vmax_main, vmax_main, vmax_diff]
        for c, panel in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            _draw_panel(lines, x, y, panel_w, panel_h, panel, vmax=vmaxs[c])
            lines.append(
                f"<text x='{x + panel_w - 6:.1f}' y='{y + panel_h + 16:.1f}' text-anchor='end' "
                f"font-family='monospace' font-size='11' fill='#555'>max={float(np.max(panel)):.3e}</text>"
            )
            mn = float(np.min(panel))
            if mn < -1e-12:
                lines.append(
                    f"<text x='{x + 6:.1f}' y='{y + panel_h + 16:.1f}' text-anchor='start' "
                    f"font-family='monospace' font-size='11' fill='#b00020'>min={mn:.3e}</text>"
                )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, cfg: ExperimentConfig, train_summary: Dict, rows: List[Dict], svg_name: str):
    lines: List[str] = []
    lines.append("# Pure TT On Chain Target Report")
    lines.append("")
    lines.append(f"- Config: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- Figure: `{svg_name}`")
    lines.append("")
    lines.append("## Training Summary")
    lines.append("")
    lines.append("| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {train_summary['target_topology']} | {train_summary['model_topology']} | {train_summary['lr_schedule']} | "
        f"{train_summary['final_lr']:.3e} | {train_summary['final_val_l2']:.6f} | "
        f"{train_summary['final_integral']:.6f} | {train_summary['total_time_sec']:.3f} |"
    )
    lines.append("")
    lines.append("## Slice Summary")
    lines.append("")
    lines.append("| pair | noise_floor_IAE | IAE_TT | ratio_to_floor | L2 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {pair} | {row['noise_floor']:.6f} | {row['iae']:.6f} | {row['ratio']:.3f} | {row['l2']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    cfg = _config()
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 48
    n_slice_samples = 200000

    print("running pure TT on chain target with config:", asdict(cfg), flush=True)
    parent_target = make_parent(cfg.n_dims, "chain")
    key = jax.random.PRNGKey(cfg.seed + 20000)
    k_train, k_val, k_slice_a, k_slice_b = jax.random.split(key, 4)

    train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
    val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
    slice_a = sample_complex_tree_distribution(k_slice_a, n_slice_samples, parent_target)
    slice_b = sample_complex_tree_distribution(k_slice_b, n_slice_samples, parent_target)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    tt_model, summary = _train_tt_model(
        cfg=cfg,
        target_topology="chain",
        train_x=train_x,
        val_x=val_x,
        bases=bases,
        gram_matrices=gram_matrices,
        basis_integrals=basis_integrals,
        seed_offset=0,
    )

    a_np = np.asarray(slice_a)
    b_np = np.asarray(slice_b)
    rows: List[Dict] = []
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
        tt_density = _eval_pair_marginal_tt(
            tt=tt_model,
            bases=bases,
            basis_integrals=basis_integrals,
            dim_i=dim_i,
            dim_j=dim_j,
            xi_centers=xi_centers,
            xj_centers=xj_centers,
        )
        abs_diff = np.abs(target_density - tt_density)

        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])
        floor = _iae_2d(target_density, target_density_2, dx, dy)
        iae = _iae_2d(target_density, tt_density, dx, dy)
        l2 = float(np.sqrt(np.sum((target_density - tt_density) ** 2) * dx * dy))
        ratio = float(iae / max(floor, 1e-12))
        row = {
            "dim_i": dim_i,
            "dim_j": dim_j,
            "noise_floor": floor,
            "iae": iae,
            "ratio": ratio,
            "l2": l2,
            "target_density": target_density,
            "tt_density": tt_density,
            "abs_diff": abs_diff,
        }
        rows.append(row)
        print(
            f"slice pair=({dim_i},{dim_j}), floor={floor:.6f}, IAE_TT={iae:.6f}, ratio={ratio:.3f}",
            flush=True,
        )

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_svg = report_dir / "fit_tt_chain_target.svg"
    out_md = report_dir / "fit_tt_chain_target_report.md"
    out_json = report_dir / "fit_tt_chain_target_metrics.json"

    _write_svg(rows, out_svg)
    _write_report(out_md, cfg, summary, rows, out_svg.name)

    compact = []
    for row in rows:
        compact.append(
            {
                "dim_i": row["dim_i"],
                "dim_j": row["dim_j"],
                "noise_floor": row["noise_floor"],
                "iae": row["iae"],
                "ratio": row["ratio"],
                "l2": row["l2"],
            }
        )
    out_json.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "training": summary,
                "slice_metrics": compact,
                "generated_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== pure TT chain target done ===", flush=True)
    print("saved svg   :", out_svg, flush=True)
    print("saved report:", out_md, flush=True)
    print("saved metric:", out_json, flush=True)


if __name__ == "__main__":
    run()

