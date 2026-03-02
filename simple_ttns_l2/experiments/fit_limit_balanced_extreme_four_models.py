from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from simple_ttns_l2.train_l2 import build_bases, make_parent  # noqa: E402
from simple_ttns_l2.experiments.topology_comparison import ExperimentConfig  # noqa: E402
from simple_ttns_l2.experiments.topology_comparison_complex import (  # noqa: E402
    sample_complex_tree_distribution,
)
from simple_ttns_l2.experiments.topology_slice_visualization import (  # noqa: E402
    _empirical_pair_density,
    _eval_pair_marginal_on_grid,
    _iae_2d,
    _train_one_model,
)
from simple_ttns_l2.experiments.fit_ceiling_tt_vs_balanced import (  # noqa: E402
    _eval_pair_marginal_tt,
    _train_tt_model,
)
from simple_ttns_l2.experiments.compare_original_ttns_vs_l2_complex_balanced import (  # noqa: E402
    _eval_pair_marginal_sq_ttns_on_grid,
    _extract_single_component_ttns,
    _train_original_mle_ttns,
)

jax.config.update("jax_enable_x64", True)


def _config() -> ExperimentConfig:
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
    panel_w = 150
    panel_h = 150
    left = 330
    top = 92
    col_gap = 18
    row_gap = 92
    cols = ["target empirical", "original TTNS", "balanced TTNS", "chain TTNS", "pure TT"]

    width = int(left + 5 * panel_w + 4 * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 82)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>"
        "Balanced Target Fit Ceiling: 4 Models"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Same complex balanced target, compare original TTNS(MLE) / balanced TTNS(L2) / chain TTNS(L2) / pure TT(L2)."
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
            f"IAE: orig={row['iae_original_ttns']:.4f}, bal={row['iae_balanced_ttns']:.4f}, "
            f"chain={row['iae_chain_ttns']:.4f}, tt={row['iae_tt']:.4f}"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"floor={row['noise_floor']:.4f}, ratios=({row['ratio_original_ttns']:.2f}x, "
            f"{row['ratio_balanced_ttns']:.2f}x, {row['ratio_chain_ttns']:.2f}x, {row['ratio_tt']:.2f}x)"
            "</text>"
        )

        vmax = max(
            float(np.max(row["target_density"])),
            float(np.max(np.maximum(row["original_density"], 0.0))),
            float(np.max(np.maximum(row["balanced_ttns_density"], 0.0))),
            float(np.max(np.maximum(row["chain_ttns_density"], 0.0))),
            float(np.max(np.maximum(row["tt_density"], 0.0))),
            1e-12,
        )
        panels = [
            row["target_density"],
            row["original_density"],
            row["balanced_ttns_density"],
            row["chain_ttns_density"],
            row["tt_density"],
        ]
        for c, panel in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            _draw_panel(lines, x, y, panel_w, panel_h, panel, vmax=vmax)
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
    lines.append("# Complex Balanced Target 极限拟合对比报告")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("在同一个 `complex balanced target` 上，比较 4 个模型在高参数量配置下能拟合到多好。")
    lines.append("")
    lines.append(f"- 配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- 切片图: `{svg_name}`")
    lines.append("")
    lines.append("## 2. 训练结果")
    lines.append("")
    lines.append(
        "| model | objective | native_metric | val_native | aux_metric | total_time_sec |"
    )
    lines.append("|---|---|---|---:|---:|---:|")
    for row in train_rows:
        if row["model_name"] == "original_ttns_mle":
            lines.append(
                f"| original_ttns | MLE / NLL | val_nll | {row['final_val_nll']:.6f} | "
                f"finite_val_ll={row['finite_val_ll']:.6f} | {row['total_time_sec']:.3f} |"
            )
        else:
            lines.append(
                f"| {row['model_topology']} | L2 | val_l2 | {row['final_val_l2']:.6f} | "
                f"best_val_l2={row.get('best_val_l2', float('nan')):.6f} | {row['total_time_sec']:.3f} |"
            )
    lines.append("")
    lines.append("说明: `original_ttns` 的原生目标是 NLL，其余三个模型的原生目标是 L2，因此真正可比的是下面的统一切片指标。")
    lines.append("")
    lines.append("## 3. 切片指标")
    lines.append("")
    lines.append(
        "| pair | noise_floor | IAE_original | IAE_balanced_ttns | IAE_chain_ttns | IAE_tt | "
        "ratio_original | ratio_balanced_ttns | ratio_chain_ttns | ratio_tt |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {pair} | {row['noise_floor']:.6f} | {row['iae_original_ttns']:.6f} | "
            f"{row['iae_balanced_ttns']:.6f} | {row['iae_chain_ttns']:.6f} | {row['iae_tt']:.6f} | "
            f"{row['ratio_original_ttns']:.3f} | {row['ratio_balanced_ttns']:.3f} | "
            f"{row['ratio_chain_ttns']:.3f} | {row['ratio_tt']:.3f} |"
        )
    lines.append("")

    mean_original = float(np.mean([r["iae_original_ttns"] for r in slice_rows]))
    mean_bal = float(np.mean([r["iae_balanced_ttns"] for r in slice_rows]))
    mean_chain = float(np.mean([r["iae_chain_ttns"] for r in slice_rows]))
    mean_tt = float(np.mean([r["iae_tt"] for r in slice_rows]))
    lines.append("## 4. 聚合结论")
    lines.append("")
    lines.append(
        f"- mean_IAE: original_ttns=`{mean_original:.6f}`, balanced_ttns=`{mean_bal:.6f}`, "
        f"chain_ttns=`{mean_chain:.6f}`, tt=`{mean_tt:.6f}`。"
    )
    best = sorted(
        [
            ("original_ttns", mean_original),
            ("balanced_ttns", mean_bal),
            ("chain_ttns", mean_chain),
            ("tt", mean_tt),
        ],
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
    em_steps = 10

    print("running balanced extreme four-model comparison with config:", asdict(cfg), flush=True)
    print("lr_policy:", lr_policy["label"], flush=True)

    target_topology = "balanced"
    parent_target = make_parent(cfg.n_dims, target_topology)
    key = jax.random.PRNGKey(cfg.seed + 33333)
    k_train, k_val, k_test, k_slice_a, k_slice_b = jax.random.split(key, 5)

    train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
    val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
    test_x = sample_complex_tree_distribution(k_test, cfg.n_test, parent_target)
    slice_a = sample_complex_tree_distribution(k_slice_a, n_slice_samples, parent_target)
    slice_b = sample_complex_tree_distribution(k_slice_b, n_slice_samples, parent_target)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    train_rows: List[Dict] = []

    original_model, original_params, original_summary = _train_original_mle_ttns(
        cfg=cfg,
        model_topology="balanced",
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        em_steps=em_steps,
    )
    train_rows.append(original_summary)

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
    bal_summary["model_name"] = "balanced_ttns"
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
    chain_summary["model_name"] = "chain_ttns"
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
    tt_summary["model_name"] = "tt"
    train_rows.append(tt_summary)

    original_ttns = _extract_single_component_ttns(original_params["ttns"]["ttns"], component=0)
    original_parent = original_model.tree_parent.tolist()
    original_bases = original_model.bases
    original_gram_matrices = jax.vmap(type(original_bases).l2_integral)(original_bases)

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

        original_density = _eval_pair_marginal_sq_ttns_on_grid(
            original_ttns,
            original_parent,
            original_bases,
            original_gram_matrices,
            dim_i,
            dim_j,
            xi_centers,
            xj_centers,
        )
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
        iae_original = _iae_2d(target_density, original_density, dx, dy)
        iae_bal = _iae_2d(target_density, bal_density, dx, dy)
        iae_chain = _iae_2d(target_density, chain_density, dx, dy)
        iae_tt = _iae_2d(target_density, tt_density, dx, dy)

        row = {
            "target_topology": target_topology,
            "dim_i": dim_i,
            "dim_j": dim_j,
            "noise_floor": noise_floor,
            "iae_original_ttns": iae_original,
            "iae_balanced_ttns": iae_bal,
            "iae_chain_ttns": iae_chain,
            "iae_tt": iae_tt,
            "ratio_original_ttns": iae_original / max(noise_floor, 1e-12),
            "ratio_balanced_ttns": iae_bal / max(noise_floor, 1e-12),
            "ratio_chain_ttns": iae_chain / max(noise_floor, 1e-12),
            "ratio_tt": iae_tt / max(noise_floor, 1e-12),
            "target_density": target_density,
            "original_density": original_density,
            "balanced_ttns_density": bal_density,
            "chain_ttns_density": chain_density,
            "tt_density": tt_density,
        }
        slice_rows.append(row)

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "fit_limit_balanced_extreme_four_models_metrics.json"
    svg_path = report_dir / "fit_limit_balanced_extreme_four_models.svg"
    report_path = report_dir / "fit_limit_balanced_extreme_four_models_report_zh.md"

    serializable = {
        "config": asdict(cfg),
        "train_rows": train_rows,
        "slice_rows": [
            {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "target_density",
                    "original_density",
                    "balanced_ttns_density",
                    "chain_ttns_density",
                    "tt_density",
                }
            }
            for row in slice_rows
        ],
    }
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    _write_svg(slice_rows, svg_path)
    _write_report(report_path, cfg, train_rows, slice_rows, svg_path.name)

    print("\n=== summary ===", flush=True)
    for row in train_rows:
        if row["model_name"] == "original_ttns_mle":
            print(
                f"model=original_ttns val_nll={row['final_val_nll']:.6f} "
                f"finite_val_ll={row['finite_val_ll']:.6f} time={row['total_time_sec']:.3f}s",
                flush=True,
            )
        else:
            print(
                f"model={row['model_topology']:>14s} val_l2={row['final_val_l2']:.6f} "
                f"time={row['total_time_sec']:.3f}s",
                flush=True,
            )
    for row in slice_rows:
        print(
            f"pair=({row['dim_i']},{row['dim_j']}): "
            f"orig={row['iae_original_ttns']:.6f}, bal={row['iae_balanced_ttns']:.6f}, "
            f"chain={row['iae_chain_ttns']:.6f}, tt={row['iae_tt']:.6f}",
            flush=True,
        )
    print("saved metrics:", metrics_path, flush=True)
    print("saved svg    :", svg_path, flush=True)
    print("saved report :", report_path, flush=True)


if __name__ == "__main__":
    run()
