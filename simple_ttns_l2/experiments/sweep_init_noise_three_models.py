from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, replace
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


def _base_config() -> ExperimentConfig:
    # Keep setting close to previous experiments but reduce steps for sweep runtime.
    return ExperimentConfig(
        n_dims=6,
        q=2,
        m=64,
        rank=16,
        batch_sz=256,
        lr=1e-3,
        train_noise=1e-2,
        init_noise=0.0,
        train_steps=120,
        log_every=10,
        seed=2602,
        n_train=4000,
        n_val=2000,
        n_test=2000,
        monitor_train_sz=1500,
        monitor_val_sz=1500,
    )


def _init_noise_grid() -> List[float]:
    return [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


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


def _format_noise(v: float) -> str:
    return f"{v:.1e}"


def _noise_tag(v: float) -> str:
    # e.g. 1e-04 -> 1em04, 5e-03 -> 5em03
    return f"{v:.0e}".replace("+", "").replace("-", "m")


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


def _write_svg_for_noise(rows: List[Dict], out_path: Path, init_noise: float):
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
        f"InitNoise Sweep Slice Comparison (init_noise={_format_noise(init_noise)})"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Heatmap in 2D marginal density; same target data and same train setup."
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


def _write_summary_csv(path: Path, rows: List[Dict]):
    cols = [
        "init_noise",
        "target_topology",
        "model",
        "final_val_l2",
        "mean_iae",
        "mean_ratio_to_floor",
        "mean_l2",
        "total_time_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in cols})


def _write_report_zh(
    path: Path,
    base_cfg: ExperimentConfig,
    init_noises: List[float],
    summary_rows: List[Dict],
    detail_rows: List[Dict],
    slice_svg_files: Dict[str, str],
):
    lines: List[str] = []
    lines.append("# 非零初始化噪声扫描实验报告（三模型）")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append(
        "验证在非零 `init_noise` 下，`balanced TTNS / chain TTNS / pure TT` 是否仍然一致，"
        "并观察随初始化噪声变化时三模型的相对表现。"
    )
    lines.append("")
    lines.append("## 2. 设定")
    lines.append("")
    lines.append(f"- Base config: `{json.dumps(asdict(base_cfg), ensure_ascii=True)}`")
    lines.append(f"- 扫描 `init_noise`: `{[float(x) for x in init_noises]}`")
    lines.append("- 目标分布: `balanced` 与 `chain`")
    lines.append("- 指标: `final_val_l2`, 2D 切片 `mean_IAE`, `mean_ratio_to_floor`, `mean_L2`")
    lines.append("")
    lines.append("## 3. 汇总表（每个 init_noise × target × model）")
    lines.append("")
    lines.append(
        "| init_noise | target | model | final_val_l2 | mean_IAE | mean_ratio_to_floor | mean_L2 | total_time_sec |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {_format_noise(row['init_noise'])} | {row['target_topology']} | {row['model']} | "
            f"{row['final_val_l2']:.6f} | {row['mean_iae']:.6f} | {row['mean_ratio_to_floor']:.3f} | "
            f"{row['mean_l2']:.6f} | {row['total_time_sec']:.3f} |"
        )
    lines.append("")
    lines.append("## 3.1 切片图文件（每个 init_noise 一张）")
    lines.append("")
    for noise in init_noises:
        label = _format_noise(noise)
        svg_name = slice_svg_files.get(label)
        if svg_name is not None:
            lines.append(f"- init_noise=`{label}`: `{svg_name}`")
    lines.append("")
    lines.append("## 4. 逐切片对比（IAE）")
    lines.append("")
    lines.append(
        "| init_noise | target | pair | floor | IAE_balanced_ttns | IAE_chain_ttns | IAE_tt | "
        "ratio_balanced_ttns | ratio_chain_ttns | ratio_tt |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in detail_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {_format_noise(row['init_noise'])} | {row['target_topology']} | {pair} | {row['noise_floor']:.6f} | "
            f"{row['iae_balanced_ttns']:.6f} | {row['iae_chain_ttns']:.6f} | {row['iae_tt']:.6f} | "
            f"{row['ratio_balanced_ttns']:.3f} | {row['ratio_chain_ttns']:.3f} | {row['ratio_tt']:.3f} |"
        )
    lines.append("")
    lines.append("## 5. 结论（中文）")
    lines.append("")

    for target in ("balanced", "chain"):
        target_rows = [r for r in summary_rows if r["target_topology"] == target]
        for noise in init_noises:
            curr = [r for r in target_rows if abs(r["init_noise"] - noise) < 1e-18]
            curr = sorted(curr, key=lambda x: x["mean_iae"])
            best = curr[0]
            second = curr[1]
            gain = (second["mean_iae"] - best["mean_iae"]) / max(second["mean_iae"], 1e-12) * 100.0
            lines.append(
                f"- target=`{target}`, init_noise=`{_format_noise(noise)}`: 最优 `{best['model']}`，"
                f"mean_IAE={best['mean_iae']:.6f}，相对第二名提升约 {gain:.2f}%。"
            )
    lines.append("")
    lines.append(
        "- 如果不同模型之间差异明显增大，说明非主通道被激活后，树结构/参数化差异开始真实影响训练轨迹。"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    base_cfg = _base_config()
    init_noises = _init_noise_grid()
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 42
    n_slice_samples = 80000

    print("running init_noise sweep for three models", flush=True)
    print("base_cfg:", asdict(base_cfg), flush=True)
    print("init_noises:", init_noises, flush=True)

    # Cache target datasets and bases so noise sweep is comparable and faster.
    data_cache: Dict[str, Dict] = {}
    for target_topology in ("balanced", "chain"):
        parent_target = make_parent(base_cfg.n_dims, target_topology)
        key = jax.random.PRNGKey(base_cfg.seed + (7000 if target_topology == "balanced" else 9000))
        k_train, k_val, k_slice_a, k_slice_b = jax.random.split(key, 4)
        train_x = sample_complex_tree_distribution(k_train, base_cfg.n_train, parent_target)
        val_x = sample_complex_tree_distribution(k_val, base_cfg.n_val, parent_target)
        slice_a = sample_complex_tree_distribution(k_slice_a, n_slice_samples, parent_target)
        slice_b = sample_complex_tree_distribution(k_slice_b, n_slice_samples, parent_target)

        bases = build_bases(train_x, q=base_cfg.q, m=base_cfg.m)
        gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
        basis_integrals = jax.vmap(type(bases).integral)(bases)

        a_np = np.asarray(slice_a)
        b_np = np.asarray(slice_b)
        slice_specs = []
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
            dx = float(xi_edges[1] - xi_edges[0])
            dy = float(xj_edges[1] - xj_edges[0])
            floor = _iae_2d(target_density, target_density_2, dx, dy)
            slice_specs.append(
                {
                    "dim_i": dim_i,
                    "dim_j": dim_j,
                    "xi_centers": xi_centers,
                    "xj_centers": xj_centers,
                    "target_density": target_density,
                    "noise_floor": floor,
                    "dx": dx,
                    "dy": dy,
                }
            )

        data_cache[target_topology] = {
            "train_x": train_x,
            "val_x": val_x,
            "bases": bases,
            "gram_matrices": gram_matrices,
            "basis_integrals": basis_integrals,
            "slice_specs": slice_specs,
        }

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []
    detail_rows: List[Dict] = []
    slice_svg_files: Dict[str, str] = {}
    lr_policy_template = _lr_policy(base_cfg)

    for init_noise in init_noises:
        cfg = replace(base_cfg, init_noise=init_noise)
        print(f"\n=== init_noise={init_noise:.1e} ===", flush=True)
        slice_rows_for_svg: List[Dict] = []
        for target_topology in ("balanced", "chain"):
            print(f"\n--- target={target_topology} ---", flush=True)
            data = data_cache[target_topology]
            train_x = data["train_x"]
            val_x = data["val_x"]
            bases = data["bases"]
            gram_matrices = data["gram_matrices"]
            basis_integrals = data["basis_integrals"]
            common_seed_offset = 0 if target_topology == "balanced" else 10000

            bal_model, bal_parent, bal_summary = _train_one_model(
                cfg=cfg,
                target_topology=target_topology,
                model_topology="balanced",
                train_x=train_x,
                val_x=val_x,
                bases=bases,
                gram_matrices=gram_matrices,
                basis_integrals=basis_integrals,
                seed_offset=common_seed_offset,
                lr_policy_template=lr_policy_template,
            )

            chain_model, chain_parent, chain_summary = _train_one_model(
                cfg=cfg,
                target_topology=target_topology,
                model_topology="chain",
                train_x=train_x,
                val_x=val_x,
                bases=bases,
                gram_matrices=gram_matrices,
                basis_integrals=basis_integrals,
                seed_offset=common_seed_offset,
                lr_policy_template=lr_policy_template,
            )

            tt_model, tt_summary = _train_tt_model(
                cfg=cfg,
                target_topology=target_topology,
                train_x=train_x,
                val_x=val_x,
                bases=bases,
                gram_matrices=gram_matrices,
                basis_integrals=basis_integrals,
                seed_offset=common_seed_offset,
            )

            iae_bal_list = []
            iae_chain_list = []
            iae_tt_list = []
            ratio_bal_list = []
            ratio_chain_list = []
            ratio_tt_list = []
            l2_bal_list = []
            l2_chain_list = []
            l2_tt_list = []

            for spec in data["slice_specs"]:
                i = spec["dim_i"]
                j = spec["dim_j"]
                target_density = spec["target_density"]
                xi_centers = spec["xi_centers"]
                xj_centers = spec["xj_centers"]
                floor = spec["noise_floor"]
                dx = spec["dx"]
                dy = spec["dy"]

                bal_density = _eval_pair_marginal_on_grid(
                    ttns=bal_model,
                    parent=bal_parent,
                    bases=bases,
                    basis_integrals=basis_integrals,
                    dim_i=i,
                    dim_j=j,
                    xi_centers=xi_centers,
                    xj_centers=xj_centers,
                )
                chain_density = _eval_pair_marginal_on_grid(
                    ttns=chain_model,
                    parent=chain_parent,
                    bases=bases,
                    basis_integrals=basis_integrals,
                    dim_i=i,
                    dim_j=j,
                    xi_centers=xi_centers,
                    xj_centers=xj_centers,
                )
                tt_density = _eval_pair_marginal_tt(
                    tt=tt_model,
                    bases=bases,
                    basis_integrals=basis_integrals,
                    dim_i=i,
                    dim_j=j,
                    xi_centers=xi_centers,
                    xj_centers=xj_centers,
                )

                iae_bal = _iae_2d(target_density, bal_density, dx, dy)
                iae_chain = _iae_2d(target_density, chain_density, dx, dy)
                iae_tt = _iae_2d(target_density, tt_density, dx, dy)
                ratio_bal = float(iae_bal / max(floor, 1e-12))
                ratio_chain = float(iae_chain / max(floor, 1e-12))
                ratio_t = float(iae_tt / max(floor, 1e-12))
                l2_bal = _l2_2d(target_density, bal_density, dx, dy)
                l2_chain = _l2_2d(target_density, chain_density, dx, dy)
                l2_t = _l2_2d(target_density, tt_density, dx, dy)

                iae_bal_list.append(iae_bal)
                iae_chain_list.append(iae_chain)
                iae_tt_list.append(iae_tt)
                ratio_bal_list.append(ratio_bal)
                ratio_chain_list.append(ratio_chain)
                ratio_tt_list.append(ratio_t)
                l2_bal_list.append(l2_bal)
                l2_chain_list.append(l2_chain)
                l2_tt_list.append(l2_t)

                detail_rows.append(
                    {
                        "init_noise": float(init_noise),
                        "target_topology": target_topology,
                        "dim_i": i,
                        "dim_j": j,
                        "noise_floor": float(floor),
                        "iae_balanced_ttns": float(iae_bal),
                        "iae_chain_ttns": float(iae_chain),
                        "iae_tt": float(iae_tt),
                        "ratio_balanced_ttns": float(ratio_bal),
                        "ratio_chain_ttns": float(ratio_chain),
                        "ratio_tt": float(ratio_t),
                    }
                )
                slice_rows_for_svg.append(
                    {
                        "target_topology": target_topology,
                        "dim_i": i,
                        "dim_j": j,
                        "noise_floor": float(floor),
                        "iae_balanced_ttns": float(iae_bal),
                        "iae_chain_ttns": float(iae_chain),
                        "iae_tt": float(iae_tt),
                        "ratio_balanced_ttns": float(ratio_bal),
                        "ratio_chain_ttns": float(ratio_chain),
                        "ratio_tt": float(ratio_t),
                        "target_density": target_density,
                        "balanced_ttns_density": bal_density,
                        "chain_ttns_density": chain_density,
                        "tt_density": tt_density,
                    }
                )

            summary_rows.append(
                {
                    "init_noise": float(init_noise),
                    "target_topology": target_topology,
                    "model": "balanced_ttns",
                    "final_val_l2": float(bal_summary["final_val_l2"]),
                    "mean_iae": float(np.mean(iae_bal_list)),
                    "mean_ratio_to_floor": float(np.mean(ratio_bal_list)),
                    "mean_l2": float(np.mean(l2_bal_list)),
                    "total_time_sec": float(bal_summary["total_time_sec"]),
                }
            )
            summary_rows.append(
                {
                    "init_noise": float(init_noise),
                    "target_topology": target_topology,
                    "model": "chain_ttns",
                    "final_val_l2": float(chain_summary["final_val_l2"]),
                    "mean_iae": float(np.mean(iae_chain_list)),
                    "mean_ratio_to_floor": float(np.mean(ratio_chain_list)),
                    "mean_l2": float(np.mean(l2_chain_list)),
                    "total_time_sec": float(chain_summary["total_time_sec"]),
                }
            )
            summary_rows.append(
                {
                    "init_noise": float(init_noise),
                    "target_topology": target_topology,
                    "model": "tt",
                    "final_val_l2": float(tt_summary["final_val_l2"]),
                    "mean_iae": float(np.mean(iae_tt_list)),
                    "mean_ratio_to_floor": float(np.mean(ratio_tt_list)),
                    "mean_l2": float(np.mean(l2_tt_list)),
                    "total_time_sec": float(tt_summary["total_time_sec"]),
                }
            )

            print(
                f"target={target_topology}, init_noise={init_noise:.1e}, "
                f"mean_IAE(bal,chain,tt)=({np.mean(iae_bal_list):.6f}, "
                f"{np.mean(iae_chain_list):.6f}, {np.mean(iae_tt_list):.6f})",
                flush=True,
            )

        slice_svg_name = f"init_noise_sweep_three_models_slice_{_noise_tag(init_noise)}.svg"
        slice_svg_path = report_dir / slice_svg_name
        _write_svg_for_noise(slice_rows_for_svg, slice_svg_path, init_noise)
        slice_svg_files[_format_noise(init_noise)] = slice_svg_name
        print("saved slice :", slice_svg_path, flush=True)

    out_json = report_dir / "init_noise_sweep_three_models_metrics.json"
    out_csv = report_dir / "init_noise_sweep_three_models_summary.csv"
    out_md = report_dir / "init_noise_sweep_three_models_report_zh.md"

    out_json.write_text(
        json.dumps(
            {
                "base_config": asdict(base_cfg),
                "init_noises": init_noises,
                "summary_rows": summary_rows,
                "detail_rows": detail_rows,
                "slice_svg_files": slice_svg_files,
                "generated_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_summary_csv(out_csv, summary_rows)
    _write_report_zh(out_md, base_cfg, init_noises, summary_rows, detail_rows, slice_svg_files)

    print("\n=== init_noise sweep done ===", flush=True)
    print("saved metric:", out_json, flush=True)
    print("saved table :", out_csv, flush=True)
    print("saved report:", out_md, flush=True)


if __name__ == "__main__":
    run()
