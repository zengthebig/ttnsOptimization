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

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.experiments.fit_limit_balanced_extreme_four_models import _config  # noqa: E402
from simple_ttns_l2.experiments.random_tree_target import (  # noqa: E402
    build_random_tree_target_spec,
    sample_random_tree_distribution,
)
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
    if topology not in ("chain", "balanced", "random"):
        raise ValueError(f"Unsupported TARGET_TOPOLOGY={val!r}. Use 'chain', 'balanced', or 'random'.")
    return topology


def _target_title(topology: str) -> str:
    if topology == "chain":
        return "Chain"
    if topology == "balanced":
        return "Balanced"
    return "Random Tree"


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


def _model_specs() -> List[Dict]:
    return [
        {
            "key": "ttde_tt",
            "title": "TTDE: TT + MLE",
            "metric_key": "iae_ttde_tt",
            "ratio_key": "ratio_ttde_tt",
            "density_key": "ttde_tt_density",
            "summary_key": "ttde_tt_summary",
        },
        {
            "key": "ttnsde_balanced",
            "title": "TTNSDE: TTNS(balanced)",
            "metric_key": "iae_ttnsde_balanced",
            "ratio_key": "ratio_ttnsde_balanced",
            "density_key": "ttnsde_balanced_density",
            "summary_key": "ttnsde_ttns_balanced_summary",
        },
        {
            "key": "ttnsde_chain",
            "title": "TTNSDE: TTNS(chain)",
            "metric_key": "iae_ttnsde_chain",
            "ratio_key": "ratio_ttnsde_chain",
            "density_key": "ttnsde_chain_density",
            "summary_key": "ttnsde_ttns_chain_summary",
        },
        {
            "key": "ttnsde_chow_liu",
            "title": "TTNSDE: TTNS(Chow-Liu)",
            "metric_key": "iae_ttnsde_chow_liu",
            "ratio_key": "ratio_ttnsde_chow_liu",
            "density_key": "ttnsde_chow_liu_density",
            "summary_key": "ttnsde_ttns_chow_liu_summary",
        },
    ]


def _write_svg(rows: List[Dict], out_path: Path, target_topology: str):
    specs = _model_specs()
    panel_w = 154
    panel_h = 154
    left = 244
    top = 92
    col_gap = 18
    row_gap = 92
    cols = ["target empirical", *(spec["title"] for spec in specs)]

    width = int(left + len(cols) * panel_w + (len(cols) - 1) * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 82)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='23' fill='#111'>"
        f"Complex {_target_title(target_topology)} Target (8D): TT + TTNS(3 topologies) + MLE"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        f"Same {target_topology} target, compare TTDE(TT+MLE), TTNS(balanced), TTNS(chain), TTNS(Chow-Liu)."
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
        iae_map = {spec["key"]: row[spec["metric_key"]] for spec in specs}
        best = min(iae_map, key=iae_map.get)
        lines.append(
            f"<text x='24' y='{y + 22:.1f}' font-family='monospace' font-size='13' fill='#111'>"
            f"pair=(x{row['dim_i']},x{row['dim_j']})"
            "</text>"
        )
        metric_txt = ", ".join(
            f"{spec['key']}={row[spec['metric_key']]:.4f}" for spec in specs
        )
        ratio_txt = ", ".join(
            f"{spec['key']}={row[spec['ratio_key']]:.2f}x" for spec in specs
        )
        lines.append(
            f"<text x='24' y='{y + 42:.1f}' font-family='monospace' font-size='12' fill='#444'>"
            f"IAE: {metric_txt}, floor={row['noise_floor']:.4f}"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"ratios: {ratio_txt}, best={best}"
            "</text>"
        )

        vmax = max(
            float(np.max(row["target_density"])),
            *(float(np.max(np.maximum(row[spec["density_key"]], 0.0))) for spec in specs),
            1e-12,
        )
        panels = [("target_density", None, None)] + [
            (spec["density_key"], row[spec["metric_key"]], spec["key"]) for spec in specs
        ]
        for c, (density_key, iae, tag) in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            border = "#222"
            border_width = 1.0
            if tag == best:
                border = "#0a7f39"
                border_width = 3.0
            _draw_panel(
                lines,
                x,
                y,
                panel_w,
                panel_h,
                row[density_key],
                vmax=vmax,
                border=border,
                border_width=border_width,
            )
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
    summaries: Dict[str, Dict],
    rows: List[Dict],
    svg_name: str,
    target_topology: str,
    target_parent: List[int],
    chow_liu_parent: List[int],
    chow_liu_edges: List[List[int]],
    target_spec: Dict | None,
):
    specs = _model_specs()
    lines: List[str] = []
    lines.append(f"# Complex {_target_title(target_topology)} Target (8D): TT + TTNS(3 topologies) + MLE")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("在同一个 target 样本上，比较 4 条 MLE 路径：")
    lines.append("")
    lines.append("- `TTDE/ttde`: `TT + square-density + MLE`")
    lines.append("- `TTNSDE/ttde`: `balanced TTNS + MLE`")
    lines.append("- `TTNSDE/ttde`: `chain TTNS + MLE`")
    lines.append("- `TTNSDE/ttde`: `Chow-Liu TTNS + MLE`")
    lines.append("")
    lines.append(f"- 配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- target_topology: `{target_topology}`")
    lines.append(f"- target_parent: `{target_parent}`")
    lines.append(f"- chow_liu_parent: `{chow_liu_parent}`")
    lines.append(f"- chow_liu_edges: `{chow_liu_edges}`")
    if target_spec is not None:
        lines.append(f"- random_target_spec: `{json.dumps(target_spec, ensure_ascii=True)}`")
    lines.append(f"- 切片图: `{svg_name}`")
    lines.append("")
    lines.append("## 2. 训练结果")
    lines.append("")
    lines.append("| model | objective | topology | val_nll | finite_val_ll | total_time_sec |")
    lines.append("|---|---|---|---:|---:|---:|")
    for spec in specs:
        summary = summaries[spec["summary_key"]]
        topology = {
            "ttde_tt_summary": "tt",
            "ttnsde_ttns_balanced_summary": "balanced tree",
            "ttnsde_ttns_chain_summary": "chain tree",
            "ttnsde_ttns_chow_liu_summary": "chow-liu tree",
        }[spec["summary_key"]]
        lines.append(
            f"| {spec['key']} | MLE | {topology} | {_fmt_metric(summary.get('final_val_nll'))} | "
            f"{_fmt_metric(summary.get('finite_val_ll'))} | {_fmt_metric(summary.get('total_time_sec'), digits=3)} |"
        )
    lines.append("")
    lines.append("## 3. 切片指标")
    lines.append("")
    header = "| pair | noise_floor | " + " | ".join(
        f"IAE_{spec['key']}" for spec in specs
    ) + " | " + " | ".join(f"ratio_{spec['key']}" for spec in specs) + " | better |"
    lines.append(header)
    lines.append("|---:|---:|" + "---:|" * (2 * len(specs)) + "---|")
    for row in rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        better = min(specs, key=lambda spec: row[spec["metric_key"]])["key"]
        metric_cells = " | ".join(f"{row[spec['metric_key']]:.6f}" for spec in specs)
        ratio_cells = " | ".join(f"{row[spec['ratio_key']]:.3f}" for spec in specs)
        lines.append(
            f"| {pair} | {row['noise_floor']:.6f} | {metric_cells} | {ratio_cells} | {better} |"
        )
    lines.append("")
    mean_rows = [(spec["key"], float(np.mean([row[spec["metric_key"]] for row in rows]))) for spec in specs]
    ranking = sorted(mean_rows, key=lambda x: x[1])
    mean_text = ", ".join(f"{name}=`{score:.6f}`" for name, score in mean_rows)
    lines.append("## 4. 聚合结论")
    lines.append("")
    lines.append(f"- mean_IAE: {mean_text}。")
    lines.append("- 排名: `" + " < ".join(name for name, _ in ranking) + "`。")
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
    chow_liu_bins = _env_int("CHOW_LIU_BINS", 16)
    target_spec = None
    if target_topology == "random":
        target_spec_obj = build_random_tree_target_spec(jax.random.PRNGKey(cfg.seed + 777), cfg.n_dims)
        target_parent = list(target_spec_obj.parent)
        target_spec = asdict(target_spec_obj)
        train_x = sample_random_tree_distribution(jax.random.PRNGKey(cfg.seed), cfg.n_train, target_spec_obj)
        val_x = sample_random_tree_distribution(jax.random.PRNGKey(cfg.seed + 1), cfg.n_val, target_spec_obj)
        test_x = sample_random_tree_distribution(jax.random.PRNGKey(cfg.seed + 2), cfg.n_test, target_spec_obj)
        slice_a = sample_random_tree_distribution(jax.random.PRNGKey(cfg.seed + 3), n_slice_samples, target_spec_obj)
        slice_b = sample_random_tree_distribution(jax.random.PRNGKey(cfg.seed + 4), n_slice_samples, target_spec_obj)
    else:
        target_parent = make_parent(cfg.n_dims, target_topology)
        train_x = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed), cfg.n_train, target_parent)
        val_x = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed + 1), cfg.n_val, target_parent)
        test_x = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed + 2), cfg.n_test, target_parent)
        slice_a = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed + 3), n_slice_samples, target_parent)
        slice_b = sample_complex_tree_distribution(jax.random.PRNGKey(cfg.seed + 4), n_slice_samples, target_parent)

    chow_liu_tree = estimate_chow_liu_tree(np.asarray(train_x), n_bins=chow_liu_bins, root=0)
    chow_liu_parent = list(chow_liu_tree.parent)
    chow_liu_edges = [list(edge) for edge in chow_liu_tree.edges]

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
    tmp_dir = report_dir / f"_tt_vs_ttns_three_topologies_mle_tmp_{target_topology}_target_8d{suffix}"
    tmp_dir.mkdir(exist_ok=True)
    data_npz = tmp_dir / "data.npz"
    chow_liu_json = tmp_dir / "chow_liu_parent.json"
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
    chow_liu_json.write_text(json.dumps(chow_liu_parent), encoding="utf-8")

    summaries: Dict[str, Dict] = {}
    density_files: Dict[str, Path] = {}
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

    tt_summary_json = tmp_dir / "ttde_tt_summary.json"
    tt_density_npz = tmp_dir / "ttde_tt_density.npz"
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
    summaries["ttde_tt_summary"] = json.loads(tt_summary_json.read_text(encoding="utf-8"))
    density_files["ttde_tt_density"] = tt_density_npz

    ttns_specs = [
        (
            "ttnsde_ttns_balanced_summary",
            "ttnsde_ttns_balanced_density",
            ["--tree-topology", "balanced", "--em-steps", "50"],
        ),
        (
            "ttnsde_ttns_chain_summary",
            "ttnsde_ttns_chain_density",
            ["--tree-topology", "chain", "--em-steps", "50"],
        ),
        (
            "ttnsde_ttns_chow_liu_summary",
            "ttnsde_ttns_chow_liu_density",
            [
                "--tree-topology", "balanced",
                "--tree-parent-json", str(chow_liu_json),
                "--em-steps", "50",
            ],
        ),
    ]
    for summary_stem, density_stem, extra_args in ttns_specs:
        summary_json = tmp_dir / f"{summary_stem}.json"
        density_npz = tmp_dir / f"{density_stem}.npz"
        subprocess.run(
            [
                PYTHON,
                str(REPO_ROOT / "simple_ttns_l2" / "experiments" / "ttnsde_ttns_mle_helper.py"),
                "--summary-json", str(summary_json),
                "--density-npz", str(density_npz),
                *(["--use-fast-nll"] if ttns_use_fast_nll else []),
                *extra_args,
                *common_args,
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        summaries[summary_stem] = json.loads(summary_json.read_text(encoding="utf-8"))
        density_files[density_stem] = density_npz

    density_data = {
        "ttde_tt_density": np.load(density_files["ttde_tt_density"]),
        "ttnsde_balanced_density": np.load(density_files["ttnsde_ttns_balanced_density"]),
        "ttnsde_chain_density": np.load(density_files["ttnsde_ttns_chain_density"]),
        "ttnsde_chow_liu_density": np.load(density_files["ttnsde_ttns_chow_liu_density"]),
    }

    b_np = np.asarray(slice_b)
    rows: List[Dict] = []
    for idx_pair, (dim_i, dim_j) in enumerate(pairs):
        xi_centers = pair_ranges[idx_pair][0]
        xj_centers = pair_ranges[idx_pair][1]
        dx = float(xi_centers[1] - xi_centers[0])
        dy = float(xj_centers[1] - xj_centers[0])
        xi_edges = np.linspace(xi_centers[0] - 0.5 * dx, xi_centers[-1] + 0.5 * dx, len(xi_centers) + 1)
        xj_edges = np.linspace(xj_centers[0] - 0.5 * dy, xj_centers[-1] + 0.5 * dy, len(xj_centers) + 1)
        target_density = _empirical_pair_density(a_np, dim_i, dim_j, xi_edges, xj_edges)
        target_density_2 = _empirical_pair_density(b_np, dim_i, dim_j, xi_edges, xj_edges)
        noise_floor = _iae_2d(target_density, target_density_2, dx, dy)
        row = {
            "dim_i": dim_i,
            "dim_j": dim_j,
            "noise_floor": float(noise_floor),
            "target_density": target_density,
        }
        for spec in _model_specs():
            density = density_data[spec["density_key"]][f"density_{idx_pair}"]
            iae = _iae_2d(target_density, density, dx, dy)
            row[spec["density_key"]] = density
            row[spec["metric_key"]] = float(iae)
            row[spec["ratio_key"]] = float(iae / max(noise_floor, 1e-12))
        rows.append(row)

    metrics = {
        "config": asdict(cfg),
        "target_topology": target_topology,
        "target_parent": target_parent,
        "target_spec": target_spec,
        "chow_liu_parent": chow_liu_parent,
        "chow_liu_edges": chow_liu_edges,
        **summaries,
        "slice_metrics": [
            {
                k: v
                for k, v in row.items()
                if "density" not in k
            }
            for row in rows
        ],
    }

    metrics_path = report_dir / f"fit_limit_{target_topology}_extreme_tt_vs_ttns_three_topologies_mle_8d{suffix}_metrics.json"
    svg_path = report_dir / f"fit_limit_{target_topology}_extreme_tt_vs_ttns_three_topologies_mle_8d{suffix}.svg"
    report_path = report_dir / f"fit_limit_{target_topology}_extreme_tt_vs_ttns_three_topologies_mle_8d{suffix}_report_zh.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_svg(rows, svg_path, target_topology=target_topology)
    _write_report(
        report_path,
        cfg,
        summaries,
        rows,
        svg_path.name,
        target_topology=target_topology,
        target_parent=target_parent,
        chow_liu_parent=chow_liu_parent,
        chow_liu_edges=chow_liu_edges,
        target_spec=target_spec,
    )
    print(f"wrote: {metrics_path}", flush=True)
    print(f"wrote: {svg_path}", flush=True)
    print(f"wrote: {report_path}", flush=True)


if __name__ == "__main__":
    run()
