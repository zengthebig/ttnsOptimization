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
        seed=2602,
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
        "label": f"delayed_cosine_hold{hold_steps}_0.1x",
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
    panel_w = 168
    panel_h = 168
    left = 302
    top = 92
    col_gap = 24
    row_gap = 92
    cols = ["target empirical", "balanced TTNS", "chain TTNS", "pure TT"]

    width = int(left + 4 * panel_w + 3 * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 82)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>"
        "InitNoise=0: Three-Model Slice Comparison"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Same target data, same hyper-parameters, three models side-by-side."
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
            f"target={row['target_topology']}, pair=(x{row['dim_i']},x{row['dim_j']})"
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


def _winner_by_min(vs: Dict[str, float]) -> str:
    items = sorted(vs.items(), key=lambda kv: kv[1])
    return items[0][0]


def _write_report(
    out_path: Path,
    cfg: ExperimentConfig,
    train_rows: List[Dict],
    slice_rows: List[Dict],
    agg_rows: List[Dict],
    svg_name: str,
):
    lines: List[str] = []
    lines.append("# InitNoise=0 三模型对比实验报告")
    lines.append("")
    lines.append("## 1. 实验目的")
    lines.append("")
    lines.append(
        "在完全固定 `init_noise=0` 的条件下，对比三类模型在两类目标分布（balanced / chain）上的拟合能力："
        "`balanced TTNS`、`chain TTNS`、`pure TT`。"
    )
    lines.append("")
    lines.append("## 2. 实验设置")
    lines.append("")
    lines.append(f"- 配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- 切片图: `{svg_name}`")
    lines.append("- 指标: `L2`、2D 切片 `IAE`、`ratio_to_floor = IAE / noise_floor`")
    lines.append("")
    lines.append("## 3. 训练结果对比")
    lines.append("")
    lines.append(
        "| target | model | lr_schedule | final_val_l2 | best_val_l2 | "
        "stopped_early | stop_step | best_step | final_integral | total_time_sec |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in train_rows:
        lines.append(
            f"| {row['target_topology']} | {row['model_topology']} | {row['lr_schedule']} | "
            f"{row['final_val_l2']:.6f} | {row.get('best_val_l2', float('nan')):.6f} | "
            f"{str(bool(row.get('stopped_early', False))).lower()} | {int(row.get('stop_step', 0))} | "
            f"{int(row.get('best_step', 0))} | {row['final_integral']:.6f} | {row['total_time_sec']:.3f} |"
        )
    lines.append("")
    lines.append("## 4. 切片误差明细（2D）")
    lines.append("")
    lines.append(
        "| target | pair | noise_floor | IAE_balanced_ttns | IAE_chain_ttns | IAE_tt | "
        "ratio_balanced_ttns | ratio_chain_ttns | ratio_tt | winner |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        winners = {
            "balanced_ttns": row["iae_balanced_ttns"],
            "chain_ttns": row["iae_chain_ttns"],
            "tt": row["iae_tt"],
        }
        winner = _winner_by_min(winners)
        lines.append(
            f"| {row['target_topology']} | {pair} | {row['noise_floor']:.6f} | "
            f"{row['iae_balanced_ttns']:.6f} | {row['iae_chain_ttns']:.6f} | {row['iae_tt']:.6f} | "
            f"{row['ratio_balanced_ttns']:.3f} | {row['ratio_chain_ttns']:.3f} | {row['ratio_tt']:.3f} | {winner} |"
        )
    lines.append("")
    lines.append("## 5. 聚合统计（按 target 汇总）")
    lines.append("")
    lines.append("| target | model | mean_IAE | mean_ratio_to_floor | mean_L2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in agg_rows:
        lines.append(
            f"| {row['target_topology']} | {row['model']} | {row['mean_iae']:.6f} | "
            f"{row['mean_ratio']:.3f} | {row['mean_l2']:.6f} |"
        )
    lines.append("")
    lines.append("## 6. 结论（中文总结）")
    lines.append("")
    for target in ("balanced", "chain"):
        curr = [r for r in agg_rows if r["target_topology"] == target]
        curr = sorted(curr, key=lambda x: x["mean_iae"])
        winner = curr[0]
        runner = curr[1]
        lines.append(
            f"- target=`{target}`: 最优是 `{winner['model']}`，其 mean_IAE={winner['mean_iae']:.6f}，"
            f"相对第二名 `{runner['model']}` 提升约 "
            f"{(runner['mean_iae'] - winner['mean_iae']) / max(runner['mean_iae'], 1e-12) * 100:.2f}% 。"
        )
    lines.append("")
    lines.append(
        "- 由于 `init_noise=0`，本实验重点反映模型结构与训练动力学差异，而不是初始化噪声导致的轨迹分叉。"
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    cfg = _config()
    lr_policy = _lr_policy(cfg)
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 48
    n_slice_samples = 200000

    print("running three-model comparison with init_noise=0", flush=True)
    print("config:", asdict(cfg), flush=True)
    print("lr_policy:", lr_policy["label"], flush=True)

    train_rows: List[Dict] = []
    slice_rows: List[Dict] = []

    for target_topology in ("balanced", "chain"):
        parent_target = make_parent(cfg.n_dims, target_topology)
        key = jax.random.PRNGKey(cfg.seed + (7000 if target_topology == "balanced" else 9000))
        k_train, k_val, k_slice_a, k_slice_b = jax.random.split(key, 4)

        train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
        val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
        slice_a = sample_complex_tree_distribution(k_slice_a, n_slice_samples, parent_target)
        slice_b = sample_complex_tree_distribution(k_slice_b, n_slice_samples, parent_target)

        bases = build_bases(train_x, q=cfg.q, m=cfg.m)
        gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
        basis_integrals = jax.vmap(type(bases).integral)(bases)

        # Keep seed_offset aligned across models for fairness.
        common_seed_offset = 0 if target_topology == "balanced" else 10000

        model_bal, parent_bal, summary_bal = _train_one_model(
            cfg=cfg,
            target_topology=target_topology,
            model_topology="balanced",
            train_x=train_x,
            val_x=val_x,
            bases=bases,
            gram_matrices=gram_matrices,
            basis_integrals=basis_integrals,
            seed_offset=common_seed_offset,
            lr_policy_template=lr_policy,
        )
        summary_bal["model_topology"] = "balanced_ttns"
        train_rows.append(summary_bal)

        model_chain, parent_chain, summary_chain = _train_one_model(
            cfg=cfg,
            target_topology=target_topology,
            model_topology="chain",
            train_x=train_x,
            val_x=val_x,
            bases=bases,
            gram_matrices=gram_matrices,
            basis_integrals=basis_integrals,
            seed_offset=common_seed_offset,
            lr_policy_template=lr_policy,
        )
        summary_chain["model_topology"] = "chain_ttns"
        train_rows.append(summary_chain)

        model_tt, summary_tt = _train_tt_model(
            cfg=cfg,
            target_topology=target_topology,
            train_x=train_x,
            val_x=val_x,
            bases=bases,
            gram_matrices=gram_matrices,
            basis_integrals=basis_integrals,
            seed_offset=common_seed_offset,
        )
        summary_tt["model_topology"] = "tt"
        train_rows.append(summary_tt)

        a_np = np.asarray(slice_a)
        b_np = np.asarray(slice_b)
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
                ttns=model_chain,
                parent=parent_chain,
                bases=bases,
                basis_integrals=basis_integrals,
                dim_i=dim_i,
                dim_j=dim_j,
                xi_centers=xi_centers,
                xj_centers=xj_centers,
            )
            tt_density = _eval_pair_marginal_tt(
                tt=model_tt,
                bases=bases,
                basis_integrals=basis_integrals,
                dim_i=dim_i,
                dim_j=dim_j,
                xi_centers=xi_centers,
                xj_centers=xj_centers,
            )

            dx = float(xi_edges[1] - xi_edges[0])
            dy = float(xj_edges[1] - xj_edges[0])
            floor = _iae_2d(target_density, target_density_2, dx, dy)
            iae_bal = _iae_2d(target_density, balanced_density, dx, dy)
            iae_chain = _iae_2d(target_density, chain_density, dx, dy)
            iae_tt = _iae_2d(target_density, tt_density, dx, dy)
            l2_bal = _l2_2d(target_density, balanced_density, dx, dy)
            l2_chain = _l2_2d(target_density, chain_density, dx, dy)
            l2_tt = _l2_2d(target_density, tt_density, dx, dy)

            row = {
                "target_topology": target_topology,
                "dim_i": dim_i,
                "dim_j": dim_j,
                "noise_floor": floor,
                "iae_balanced_ttns": iae_bal,
                "iae_chain_ttns": iae_chain,
                "iae_tt": iae_tt,
                "ratio_balanced_ttns": float(iae_bal / max(floor, 1e-12)),
                "ratio_chain_ttns": float(iae_chain / max(floor, 1e-12)),
                "ratio_tt": float(iae_tt / max(floor, 1e-12)),
                "l2_balanced_ttns": l2_bal,
                "l2_chain_ttns": l2_chain,
                "l2_tt": l2_tt,
                "target_density": target_density,
                "balanced_ttns_density": balanced_density,
                "chain_ttns_density": chain_density,
                "tt_density": tt_density,
            }
            slice_rows.append(row)
            print(
                f"slice target={target_topology}, pair=({dim_i},{dim_j}), floor={floor:.6f}, "
                f"IAE_bal={iae_bal:.6f}, IAE_chain={iae_chain:.6f}, IAE_tt={iae_tt:.6f}",
                flush=True,
            )

    agg_rows: List[Dict] = []
    for target_topology in ("balanced", "chain"):
        curr = [r for r in slice_rows if r["target_topology"] == target_topology]
        for model in ("balanced_ttns", "chain_ttns", "tt"):
            if model == "balanced_ttns":
                iae_key, ratio_key, l2_key = "iae_balanced_ttns", "ratio_balanced_ttns", "l2_balanced_ttns"
            elif model == "chain_ttns":
                iae_key, ratio_key, l2_key = "iae_chain_ttns", "ratio_chain_ttns", "l2_chain_ttns"
            else:
                iae_key, ratio_key, l2_key = "iae_tt", "ratio_tt", "l2_tt"
            agg_rows.append(
                {
                    "target_topology": target_topology,
                    "model": model,
                    "mean_iae": float(np.mean([r[iae_key] for r in curr])),
                    "mean_ratio": float(np.mean([r[ratio_key] for r in curr])),
                    "mean_l2": float(np.mean([r[l2_key] for r in curr])),
                }
            )

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_svg = report_dir / "fit_three_models_init0.svg"
    out_json = report_dir / "fit_three_models_init0_metrics.json"
    out_md = report_dir / "fit_three_models_init0_report_zh.md"

    _write_svg(slice_rows, out_svg)
    _write_report(out_md, cfg, train_rows, slice_rows, agg_rows, out_svg.name)

    compact_slices = []
    for row in slice_rows:
        compact_slices.append(
            {
                "target_topology": row["target_topology"],
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
                "slice_metrics": compact_slices,
                "aggregate_metrics": agg_rows,
                "generated_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== three-model init0 experiment done ===", flush=True)
    print("saved svg   :", out_svg, flush=True)
    print("saved report:", out_md, flush=True)
    print("saved metric:", out_json, flush=True)


if __name__ == "__main__":
    run()
