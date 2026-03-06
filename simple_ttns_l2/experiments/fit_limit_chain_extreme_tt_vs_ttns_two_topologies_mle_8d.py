from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_ttns_l2.experiments.fit_limit_balanced_extreme_four_models import _config  # noqa: E402
from simple_ttns_l2.experiments.topology_comparison_complex import sample_complex_tree_distribution  # noqa: E402
from simple_ttns_l2.experiments.topology_slice_visualization import _empirical_pair_density, _iae_2d  # noqa: E402
from simple_ttns_l2.train_l2 import make_parent  # noqa: E402


PYTHON = "/home/sbzeng/.conda/envs/ttns/bin/python"


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return int(val)


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return float(val)


def _get_target_topology_from_env() -> str:
    val = os.getenv("TARGET_TOPOLOGY", "chain")
    topology = val.strip().lower()
    if topology not in ("chain", "balanced"):
        raise ValueError(f"Unsupported TARGET_TOPOLOGY={val!r}. Use 'chain' or 'balanced'.")
    return topology


def _target_title(topology: str) -> str:
    return "Chain" if topology == "chain" else "Balanced"


def _fmt_metric(v, digits: int = 6) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(x):
        return "nan"
    if np.isposinf(x):
        return "inf"
    if np.isneginf(x):
        return "-inf"
    return f"{x:.{digits}f}"


def _draw_panel(
    lines: List[str],
    x0: float,
    y0: float,
    w: float,
    h: float,
    panel: np.ndarray,
    vmax: float,
    border: str = "#222",
    border_width: float = 1.0,
):
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
        f"<rect x='{x0:.3f}' y='{y0:.3f}' width='{w:.3f}' height='{h:.3f}' "
        f"fill='none' stroke='{border}' stroke-width='{border_width:.1f}'/>"
    )


def _write_svg(rows: List[Dict], out_path: Path, target_topology: str):
    panel_w = 176
    panel_h = 176
    left = 220
    top = 92
    col_gap = 20
    row_gap = 90
    cols = [
        "target empirical",
        "TTDE: TT + MLE",
        "TTNSDE: TTNS(balanced) + MLE",
        "TTNSDE: TTNS(chain) + MLE",
    ]

    width = int(left + 4 * panel_w + 3 * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 82)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='23' fill='#111'>"
        f"Complex {_target_title(target_topology)} Target (8D): TT+MLE vs TTNS(balanced/chain)+MLE"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        f"Same {target_topology} target and same extreme config, compare TTDE(TT+MLE), TTNSDE(balanced-TTNS+MLE), TTNSDE(chain-TTNS+MLE)."
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
        iaes = {
            "tt": row["iae_ttde_tt"],
            "balanced": row["iae_ttnsde_balanced"],
            "chain": row["iae_ttnsde_chain"],
        }
        best = min(iaes, key=iaes.get)
        lines.append(
            f"<text x='24' y='{y + 22:.1f}' font-family='monospace' font-size='13' fill='#111'>"
            f"pair=(x{row['dim_i']},x{row['dim_j']})"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 42:.1f}' font-family='monospace' font-size='12' fill='#444'>"
            f"IAE_tt={row['iae_ttde_tt']:.4f}, IAE_bal={row['iae_ttnsde_balanced']:.4f}, "
            f"IAE_chain={row['iae_ttnsde_chain']:.4f}, floor={row['noise_floor']:.4f}"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"ratio_tt={row['ratio_ttde_tt']:.2f}x, ratio_bal={row['ratio_ttnsde_balanced']:.2f}x, "
            f"ratio_chain={row['ratio_ttnsde_chain']:.2f}x, best={best}"
            "</text>"
        )

        vmax = max(
            float(np.max(row["target_density"])),
            float(np.max(np.maximum(row["ttde_tt_density"], 0.0))),
            float(np.max(np.maximum(row["ttnsde_balanced_density"], 0.0))),
            float(np.max(np.maximum(row["ttnsde_chain_density"], 0.0))),
            1e-12,
        )
        panels = [
            ("target_density", None, None),
            ("ttde_tt_density", row["iae_ttde_tt"], "tt"),
            ("ttnsde_balanced_density", row["iae_ttnsde_balanced"], "balanced"),
            ("ttnsde_chain_density", row["iae_ttnsde_chain"], "chain"),
        ]
        for c, (key, iae, tag) in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            border = "#222"
            width_stroke = 1.0
            if tag == best:
                border = "#0a7f39"
                width_stroke = 3.0
            _draw_panel(lines, x, y, panel_w, panel_h, row[key], vmax=vmax, border=border, border_width=width_stroke)
            if iae is not None:
                lines.append(
                    f"<text x='{x + 6:.1f}' y='{y + panel_h + 14:.1f}' text-anchor='start' "
                    f"font-family='monospace' font-size='10' fill='#555'>IAE={iae:.3e}</text>"
                )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    cfg,
    tt_summary: Dict,
    ttns_bal_summary: Dict,
    ttns_chain_summary: Dict,
    rows: List[Dict],
    svg_name: str,
    target_topology: str,
):
    lines: List[str] = []
    lines.append(f"# Complex {_target_title(target_topology)} Target (8D): TT+MLE vs TTNS(balanced/chain)+MLE")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append(f"在同一个 `complex {target_topology} target` 上，严格比较三条 MLE 路径：")
    lines.append("")
    lines.append("- `TTDE/ttde`: `TT + square-density + MLE`")
    lines.append("- `TTNSDE/ttde`: `balanced TTNS + square-density + MLE`")
    lines.append("- `TTNSDE/ttde`: `chain TTNS + square-density + MLE`")
    lines.append("")
    lines.append(f"- 配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- 切片图: `{svg_name}`")
    lines.append("")
    lines.append("## 2. 训练结果")
    lines.append("")
    lines.append("| model | objective | topology | val_nll | finite_val_ll | total_time_sec |")
    lines.append("|---|---|---|---:|---:|---:|")
    lines.append(
        f"| ttde_tt_mle | MLE | tt | {_fmt_metric(tt_summary.get('final_val_nll'))} | {_fmt_metric(tt_summary.get('finite_val_ll'))} | {_fmt_metric(tt_summary.get('total_time_sec'), digits=3)} |"
    )
    lines.append(
        f"| ttnsde_ttns_mle_balanced | MLE | balanced tree | {_fmt_metric(ttns_bal_summary.get('final_val_nll'))} | {_fmt_metric(ttns_bal_summary.get('finite_val_ll'))} | {_fmt_metric(ttns_bal_summary.get('total_time_sec'), digits=3)} |"
    )
    lines.append(
        f"| ttnsde_ttns_mle_chain | MLE | chain tree | {_fmt_metric(ttns_chain_summary.get('final_val_nll'))} | {_fmt_metric(ttns_chain_summary.get('finite_val_ll'))} | {_fmt_metric(ttns_chain_summary.get('total_time_sec'), digits=3)} |"
    )
    lines.append("")
    lines.append("## 3. 切片指标")
    lines.append("")
    lines.append("| pair | noise_floor | IAE_ttde_tt | IAE_ttns_balanced | IAE_ttns_chain | ratio_ttde_tt | ratio_ttns_balanced | ratio_ttns_chain | better |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        iae_map = {
            "ttde_tt": row["iae_ttde_tt"],
            "ttnsde_balanced": row["iae_ttnsde_balanced"],
            "ttnsde_chain": row["iae_ttnsde_chain"],
        }
        better = min(iae_map, key=iae_map.get)
        lines.append(
            f"| {pair} | {row['noise_floor']:.6f} | {row['iae_ttde_tt']:.6f} | {row['iae_ttnsde_balanced']:.6f} | "
            f"{row['iae_ttnsde_chain']:.6f} | {row['ratio_ttde_tt']:.3f} | {row['ratio_ttnsde_balanced']:.3f} | "
            f"{row['ratio_ttnsde_chain']:.3f} | {better} |"
        )
    lines.append("")
    mean_tt = float(np.mean([r["iae_ttde_tt"] for r in rows]))
    mean_bal = float(np.mean([r["iae_ttnsde_balanced"] for r in rows]))
    mean_chain = float(np.mean([r["iae_ttnsde_chain"] for r in rows]))
    lines.append("## 4. 聚合结论")
    lines.append("")
    lines.append(f"- mean_IAE: ttde_tt=`{mean_tt:.6f}`, ttnsde_balanced=`{mean_bal:.6f}`, ttnsde_chain=`{mean_chain:.6f}`。")
    ranking = sorted([("ttde_tt", mean_tt), ("ttnsde_balanced", mean_bal), ("ttnsde_chain", mean_chain)], key=lambda x: x[1])
    lines.append(f"- 排名: `{ranking[0][0]} < {ranking[1][0]} < {ranking[2][0]}`。")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    cfg0 = _config()
    cfg = replace(
        cfg0,
        n_dims=_env_int("N_DIMS", 8),
        m=_env_int("M", cfg0.m),
        rank=_env_int("RANK", cfg0.rank),
        batch_sz=_env_int("BATCH_SZ", cfg0.batch_sz),
        lr=_env_float("LR", cfg0.lr),
        train_steps=_env_int("TRAIN_STEPS", cfg0.train_steps),
        seed=_env_int("SEED", cfg0.seed),
        n_train=_env_int("N_TRAIN", cfg0.n_train),
        n_val=_env_int("N_VAL", cfg0.n_val),
        n_test=_env_int("N_TEST", cfg0.n_test),
        monitor_train_sz=_env_int("MONITOR_TRAIN_SZ", cfg0.monitor_train_sz),
        monitor_val_sz=_env_int("MONITOR_VAL_SZ", cfg0.monitor_val_sz),
    )
    target_topology = _get_target_topology_from_env()
    pairs = [(0, 1), (0, 3), (2, 5)]
    grid_bins = _env_int("GRID_BINS", 56)
    n_slice_samples = _env_int("N_SLICE_SAMPLES", 220000)

    parent_target = make_parent(cfg.n_dims, target_topology)
    train_x = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed), cfg.n_train, parent_target)
    val_x = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed + 1), cfg.n_val, parent_target)
    test_x = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed + 2), cfg.n_test, parent_target)
    slice_a = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed + 3), n_slice_samples, parent_target)
    slice_b = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed + 4), n_slice_samples, parent_target)

    a_np = np.asarray(slice_a)
    pair_ranges = []
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
        pair_ranges.append(np.stack([xi_centers, xj_centers], axis=0))

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    run_tag = os.getenv("RUN_TAG", "").strip()
    suffix = f"_{run_tag}" if run_tag else ""
    tmp_dir = report_dir / f"_tt_vs_ttns_two_topologies_mle_tmp_{target_topology}_target_8d{suffix}"
    tmp_dir.mkdir(exist_ok=True)
    data_npz = tmp_dir / "data.npz"
    np.savez_compressed(
        data_npz,
        train_x=np.asarray(train_x),
        val_x=np.asarray(val_x),
        test_x=np.asarray(test_x),
        slice_a=np.asarray(slice_a),
        slice_b=np.asarray(slice_b),
        slice_pairs=np.asarray(pairs, dtype=np.int32),
        pair_ranges=np.asarray(pair_ranges, dtype=np.float64),
    )

    tt_summary_json = tmp_dir / "ttde_tt_summary.json"
    tt_density_npz = tmp_dir / "ttde_tt_density.npz"
    ttns_bal_summary_json = tmp_dir / "ttnsde_ttns_balanced_summary.json"
    ttns_bal_density_npz = tmp_dir / "ttnsde_ttns_balanced_density.npz"
    ttns_chain_summary_json = tmp_dir / "ttnsde_ttns_chain_summary.json"
    ttns_chain_density_npz = tmp_dir / "ttnsde_ttns_chain_density.npz"
    ttns_use_fast_nll = _env_int("TTNS_USE_FAST_NLL", 0) == 1

    common_args = [
        "--data-npz", str(data_npz),
        "--q", str(cfg.q),
        "--m", str(cfg.m),
        "--rank", str(cfg.rank),
        "--batch-sz", str(cfg.batch_sz),
        "--lr", str(cfg.lr),
        "--train-steps", str(cfg.train_steps),
        "--log-every", str(cfg.log_every),
        "--init-noise", str(cfg.init_noise),
        "--train-noise", str(cfg.train_noise),
        "--seed", str(cfg.seed),
        "--monitor-train-sz", str(cfg.monitor_train_sz),
        "--monitor-val-sz", str(cfg.monitor_val_sz),
    ]

    subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "simple_ttns_l2" / "experiments" / "ttde_tt_mle_helper.py"),
            "--summary-json", str(tt_summary_json),
            "--density-npz", str(tt_density_npz),
            *common_args,
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "simple_ttns_l2" / "experiments" / "ttnsde_ttns_mle_helper.py"),
            "--summary-json", str(ttns_bal_summary_json),
            "--density-npz", str(ttns_bal_density_npz),
            "--tree-topology", "balanced",
            "--em-steps", "50",
            *(["--use-fast-nll"] if ttns_use_fast_nll else []),
            *common_args,
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "simple_ttns_l2" / "experiments" / "ttnsde_ttns_mle_helper.py"),
            "--summary-json", str(ttns_chain_summary_json),
            "--density-npz", str(ttns_chain_density_npz),
            "--tree-topology", "chain",
            "--em-steps", "50",
            *(["--use-fast-nll"] if ttns_use_fast_nll else []),
            *common_args,
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    tt_summary = json.loads(tt_summary_json.read_text(encoding="utf-8"))
    ttns_bal_summary = json.loads(ttns_bal_summary_json.read_text(encoding="utf-8"))
    ttns_chain_summary = json.loads(ttns_chain_summary_json.read_text(encoding="utf-8"))
    tt_density_data = np.load(tt_density_npz)
    ttns_bal_density_data = np.load(ttns_bal_density_npz)
    ttns_chain_density_data = np.load(ttns_chain_density_npz)

    b_np = np.asarray(slice_b)
    rows: List[Dict] = []
    for idx_pair, (dim_i, dim_j) in enumerate(pairs):
        xi_centers = pair_ranges[idx_pair][0]
        xj_centers = pair_ranges[idx_pair][1]
        xi_edges = np.linspace(xi_centers[0] - 0.5 * (xi_centers[1] - xi_centers[0]), xi_centers[-1] + 0.5 * (xi_centers[1] - xi_centers[0]), len(xi_centers) + 1)
        xj_edges = np.linspace(xj_centers[0] - 0.5 * (xj_centers[1] - xj_centers[0]), xj_centers[-1] + 0.5 * (xj_centers[1] - xj_centers[0]), len(xj_centers) + 1)

        target_density = _empirical_pair_density(a_np, dim_i, dim_j, xi_edges, xj_edges)
        target_density_2 = _empirical_pair_density(b_np, dim_i, dim_j, xi_edges, xj_edges)
        tt_density = tt_density_data[f"density_{idx_pair}"]
        ttns_bal_density = ttns_bal_density_data[f"density_{idx_pair}"]
        ttns_chain_density = ttns_chain_density_data[f"density_{idx_pair}"]
        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])
        noise_floor = _iae_2d(target_density, target_density_2, dx, dy)
        iae_tt = _iae_2d(target_density, tt_density, dx, dy)
        iae_ttns_bal = _iae_2d(target_density, ttns_bal_density, dx, dy)
        iae_ttns_chain = _iae_2d(target_density, ttns_chain_density, dx, dy)
        rows.append(
            {
                "dim_i": dim_i,
                "dim_j": dim_j,
                "noise_floor": float(noise_floor),
                "iae_ttde_tt": float(iae_tt),
                "iae_ttnsde_balanced": float(iae_ttns_bal),
                "iae_ttnsde_chain": float(iae_ttns_chain),
                "ratio_ttde_tt": float(iae_tt / max(noise_floor, 1e-12)),
                "ratio_ttnsde_balanced": float(iae_ttns_bal / max(noise_floor, 1e-12)),
                "ratio_ttnsde_chain": float(iae_ttns_chain / max(noise_floor, 1e-12)),
                "target_density": target_density,
                "ttde_tt_density": tt_density,
                "ttnsde_balanced_density": ttns_bal_density,
                "ttnsde_chain_density": ttns_chain_density,
            }
        )

    metrics = {
        "config": asdict(cfg),
        "ttde_tt_summary": tt_summary,
        "ttnsde_ttns_balanced_summary": ttns_bal_summary,
        "ttnsde_ttns_chain_summary": ttns_chain_summary,
        "slice_metrics": [
            {
                k: v
                for k, v in row.items()
                if k not in ("target_density", "ttde_tt_density", "ttnsde_balanced_density", "ttnsde_chain_density")
            }
            for row in rows
        ],
    }

    metrics_path = report_dir / f"fit_limit_{target_topology}_extreme_tt_vs_ttns_two_topologies_mle_8d{suffix}_metrics.json"
    svg_path = report_dir / f"fit_limit_{target_topology}_extreme_tt_vs_ttns_two_topologies_mle_8d{suffix}.svg"
    report_path = report_dir / f"fit_limit_{target_topology}_extreme_tt_vs_ttns_two_topologies_mle_8d{suffix}_report_zh.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_svg(rows, svg_path, target_topology=target_topology)
    _write_report(
        report_path,
        cfg,
        tt_summary,
        ttns_bal_summary,
        ttns_chain_summary,
        rows,
        svg_path.name,
        target_topology=target_topology,
    )
    print(f"wrote: {metrics_path}", flush=True)
    print(f"wrote: {svg_path}", flush=True)
    print(f"wrote: {report_path}", flush=True)


if __name__ == "__main__":
    run()
