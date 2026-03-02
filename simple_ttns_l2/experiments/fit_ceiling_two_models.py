from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
        "label": f"ceiling_delayed_cosine_hold{hold_steps}_0.1x",
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


def _l2_2d(a: np.ndarray, b: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2) * dx * dy))


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
    panel_w = 185
    panel_h = 185
    left = 250
    top = 86
    col_gap = 30
    row_gap = 92
    cols = ["target empirical", "model balanced", "model chain"]

    width = int(left + 3 * panel_w + 2 * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 80)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>"
        "Fit Ceiling: Two Models On Same Example"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Rows share the same target samples; compare balanced vs chain model limits."
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
            f"target={row['target_topology']}, pair=(x{row['dim_i']},x{row['dim_j']})"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 42:.1f}' font-family='monospace' font-size='12' fill='#444'>"
            f"IAE_bal={row['iae_balanced']:.4f}, IAE_chain={row['iae_chain']:.4f}, "
            f"noise_floor={row['noise_floor']:.4f}</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"ratio_bal={row['ratio_balanced']:.2f}x, ratio_chain={row['ratio_chain']:.2f}x"
            "</text>"
        )

        v = max(
            float(np.max(row["target_density"])),
            float(np.max(row["balanced_density"])),
            float(np.max(row["chain_density"])),
            1e-12,
        )
        panels = [row["target_density"], row["balanced_density"], row["chain_density"]]
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


def _write_report(path: Path, cfg: ExperimentConfig, train_rows: List[Dict], slice_rows: List[Dict], svg_name: str):
    lines: List[str] = []
    lines.append("# Fit Ceiling Two-Model Report")
    lines.append("")
    lines.append(
        "This compares balanced and chain models on exactly the same target data to check how far each can fit."
    )
    lines.append("")
    lines.append(f"- Config: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- Figure: `{svg_name}`")
    lines.append("")
    lines.append("## Training Summary")
    lines.append("")
    lines.append("| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in train_rows:
        lines.append(
            f"| {row['target_topology']} | {row['model_topology']} | {row['lr_schedule']} | "
            f"{row['final_lr']:.3e} | {row['final_val_l2']:.6f} | "
            f"{row['final_integral']:.6f} | {row['total_time_sec']:.3f} |"
        )
    lines.append("")
    lines.append("## Slice Summary")
    lines.append("")
    lines.append("| target_topology | pair | noise_floor_IAE | IAE_balanced | IAE_chain | ratio_balanced | ratio_chain | L2_balanced | L2_chain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {row['target_topology']} | {pair} | {row['noise_floor']:.6f} | "
            f"{row['iae_balanced']:.6f} | {row['iae_chain']:.6f} | "
            f"{row['ratio_balanced']:.3f} | {row['ratio_chain']:.3f} | "
            f"{row['l2_balanced']:.6f} | {row['l2_chain']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fit_ceiling():
    cfg = _config()
    lr_policy = _lr_policy(cfg)
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 48
    n_slice_samples = 200000

    print("running two-model fit ceiling with config:", asdict(cfg), flush=True)
    print("lr_policy:", lr_policy["label"], flush=True)

    seed_cursor = 0
    train_rows: List[Dict] = []
    slice_rows: List[Dict] = []

    for target_topology in ("balanced", "chain"):
        parent_target = make_parent(cfg.n_dims, target_topology)
        key = jax.random.PRNGKey(cfg.seed + 12000 + seed_cursor)
        seed_cursor += 100
        k_train, k_val, k_slice_a, k_slice_b = jax.random.split(key, 4)

        train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
        val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
        slice_a = sample_complex_tree_distribution(k_slice_a, n_slice_samples, parent_target)
        slice_b = sample_complex_tree_distribution(k_slice_b, n_slice_samples, parent_target)

        bases = build_bases(train_x, q=cfg.q, m=cfg.m)
        gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
        basis_integrals = jax.vmap(type(bases).integral)(bases)

        trained = {}
        for model_topology in ("balanced", "chain"):
            model, model_parent, summary = _train_one_model(
                cfg=cfg,
                target_topology=target_topology,
                model_topology=model_topology,
                train_x=train_x,
                val_x=val_x,
                bases=bases,
                gram_matrices=gram_matrices,
                basis_integrals=basis_integrals,
                seed_offset=seed_cursor,
                lr_policy_template=lr_policy,
            )
            seed_cursor += 1
            trained[model_topology] = (model, model_parent)
            train_rows.append(summary)

        a_np = np.asarray(slice_a)
        b_np = np.asarray(slice_b)
        model_bal, parent_bal = trained["balanced"]
        model_chn, parent_chn = trained["chain"]

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
            balanced_density = _eval_pair_marginal_on_grid(
                ttns=model_bal,
                parent=parent_bal,
                bases=bases,
                basis_integrals=basis_integrals,
                dim_i=dim_i,
                dim_j=dim_j,
                xi_centers=xi_centers,
                xj_centers=xj_centers,
            )
            chain_density = _eval_pair_marginal_on_grid(
                ttns=model_chn,
                parent=parent_chn,
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
            iae_balanced = _iae_2d(target_density, balanced_density, dx, dy)
            iae_chain = _iae_2d(target_density, chain_density, dx, dy)
            l2_balanced = _l2_2d(target_density, balanced_density, dx, dy)
            l2_chain = _l2_2d(target_density, chain_density, dx, dy)

            row = {
                "target_topology": target_topology,
                "dim_i": dim_i,
                "dim_j": dim_j,
                "noise_floor": noise_floor,
                "iae_balanced": iae_balanced,
                "iae_chain": iae_chain,
                "ratio_balanced": float(iae_balanced / max(noise_floor, 1e-12)),
                "ratio_chain": float(iae_chain / max(noise_floor, 1e-12)),
                "l2_balanced": l2_balanced,
                "l2_chain": l2_chain,
                "target_density": target_density,
                "balanced_density": balanced_density,
                "chain_density": chain_density,
            }
            slice_rows.append(row)
            print(
                f"slice target={target_topology}, pair=({dim_i},{dim_j}), floor={noise_floor:.6f}, "
                f"IAE_bal={iae_balanced:.6f}, IAE_chain={iae_chain:.6f}",
                flush=True,
            )

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_svg = report_dir / "fit_ceiling_two_models_complex.svg"
    out_md = report_dir / "fit_ceiling_two_models_complex_report.md"
    out_json = report_dir / "fit_ceiling_two_models_complex_metrics.json"

    _write_svg(slice_rows, out_svg)
    _write_report(out_md, cfg, train_rows, slice_rows, out_svg.name)

    compact_rows = []
    for row in slice_rows:
        compact_rows.append(
            {
                "target_topology": row["target_topology"],
                "dim_i": row["dim_i"],
                "dim_j": row["dim_j"],
                "noise_floor": row["noise_floor"],
                "iae_balanced": row["iae_balanced"],
                "iae_chain": row["iae_chain"],
                "ratio_balanced": row["ratio_balanced"],
                "ratio_chain": row["ratio_chain"],
                "l2_balanced": row["l2_balanced"],
                "l2_chain": row["l2_chain"],
            }
        )
    out_json.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "lr_policy": lr_policy,
                "training": train_rows,
                "slice_metrics": compact_rows,
                "generated_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== fit ceiling two-model done ===", flush=True)
    print("saved svg   :", out_svg, flush=True)
    print("saved report:", out_md, flush=True)
    print("saved metric:", out_json, flush=True)


if __name__ == "__main__":
    run_fit_ceiling()

