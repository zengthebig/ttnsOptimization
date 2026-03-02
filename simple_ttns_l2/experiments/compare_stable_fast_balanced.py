from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jax
import numpy as np
import optax
from jax import numpy as jnp, value_and_grad

# Ensure imports resolve to local simple_ttns_l2 and TTNSDE/ttde.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.experiments.topology_comparison import ExperimentConfig
from simple_ttns_l2.experiments.topology_comparison_complex import sample_complex_tree_distribution
from simple_ttns_l2.objective import (
    batch_basis_vectors_from_samples,
    eval_q_ttns,
    integral_q2_ttns,
    integral_q_ttns,
    l2_objective_ttns,
    mc_expectation_q_ttns,
    normalize_ttns_by_integral,
)
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1, make_parent

jax.config.update("jax_enable_x64", True)


REPORT_DIR = REPO_ROOT / "simple_ttns_l2" / "reports"


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
        log_every=5,
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


def _lr_for_step(step: int, train_steps: int, init_lr: float, final_lr: float, hold_steps: int) -> float:
    if step <= hold_steps:
        return init_lr
    denom = max(1, train_steps - hold_steps)
    progress = min(1.0, max(0.0, (step - hold_steps) / denom))
    return final_lr + 0.5 * (init_lr - final_lr) * (1.0 + np.cos(np.pi * progress))


def _color(value: float, vmax: float) -> str:
    if vmax <= 0:
        return "#f4f4f4"
    t = max(0.0, min(1.0, value / vmax))
    r = int(round(255 * (1.0 - 0.95 * t)))
    g = int(round(255 * (1.0 - 0.70 * t)))
    b = int(round(255 * (1.0 - 0.20 * t)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_heatmap(lines: List[str], x0: float, y0: float, w: float, h: float, data: np.ndarray, vmax: float):
    nx, ny = data.shape
    cell_w = w / ny
    cell_h = h / nx
    for i in range(nx):
        for j in range(ny):
            val = max(0.0, float(data[i, j]))
            x = x0 + j * cell_w
            y = y0 + (nx - 1 - i) * cell_h
            lines.append(
                f"<rect x='{x:.3f}' y='{y:.3f}' width='{cell_w + 0.2:.3f}' height='{cell_h + 0.2:.3f}' "
                f"fill='{_color(val, vmax)}' stroke='none'/>"
            )
    lines.append(
        f"<rect x='{x0:.3f}' y='{y0:.3f}' width='{w:.3f}' height='{h:.3f}' "
        f"fill='none' stroke='#222' stroke-width='1'/>"
    )


def _write_slice_svg(rows: List[Dict], out_path: Path):
    cols = ["target empirical", "stable TTNS", "fast TTNS"]
    panel_w = 180
    panel_h = 180
    left = 260
    top = 84
    col_gap = 24
    row_gap = 86

    width = int(left + 3 * panel_w + 2 * col_gap + 32)
    height = int(top + len(rows) * panel_h + max(0, len(rows) - 1) * row_gap + 72)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>"
        "Stable vs Fast TTNS: 2D Slice Comparison"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Same seed, same balanced target, same balanced TTNS model."
        "</text>"
    )

    for c, name in enumerate(cols):
        x = left + c * (panel_w + col_gap) + panel_w / 2
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
            f"IAE stable={row['iae_stable']:.4f}, IAE fast={row['iae_fast']:.4f}, "
            f"winner={row['winner']}</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"x{row['dim_i']} in [{row['xi_min']:.2f}, {row['xi_max']:.2f}], "
            f"x{row['dim_j']} in [{row['xj_min']:.2f}, {row['xj_max']:.2f}]</text>"
        )

        vmax = max(
            float(np.max(row["target_density"])),
            float(np.max(np.maximum(row["stable_density"], 0.0))),
            float(np.max(np.maximum(row["fast_density"], 0.0))),
            1e-12,
        )

        panels = [row["target_density"], row["stable_density"], row["fast_density"]]
        for c, panel in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            _draw_heatmap(lines, x, y, panel_w, panel_h, panel, vmax=vmax)
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


def _write_curve_svg(history_rows: List[Dict], out_path: Path):
    width = 1080
    height = 680
    margin_l = 80
    margin_r = 36
    margin_t = 56
    margin_b = 64
    gap = 58
    panel_h = (height - margin_t - margin_b - gap) / 2.0
    panel_w = width - margin_l - margin_r

    def scale_x(step: float) -> float:
        max_step = max(float(r["step"]) for r in history_rows)
        return margin_l + (step / max_step) * panel_w

    def panel_bounds(panel_idx: int) -> Tuple[float, float]:
        y0 = margin_t + panel_idx * (panel_h + gap)
        return y0, y0 + panel_h

    def scale_y(vmin: float, vmax: float, y_top: float, y_bottom: float, value: float) -> float:
        if abs(vmax - vmin) < 1e-12:
            return 0.5 * (y_top + y_bottom)
        return y_bottom - (value - vmin) / (vmax - vmin) * (y_bottom - y_top)

    val_stable = [float(r["stable_val_l2"]) for r in history_rows]
    val_fast = [float(r["fast_val_l2"]) for r in history_rows]
    time_stable = [float(r["stable_total_sec"]) for r in history_rows]
    time_fast = [float(r["fast_total_sec"]) for r in history_rows]

    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
    lines.append(
        "<text x='28' y='32' font-family='monospace' font-size='24' fill='#111'>"
        "Stable vs Fast TTNS: Loss and Time"
        "</text>"
    )
    lines.append(
        "<text x='28' y='52' font-family='monospace' font-size='12' fill='#555'>"
        "Blue = stable, Orange = fast."
        "</text>"
    )

    panels = [
        ("val_l2", val_stable + val_fast),
        ("total_sec", time_stable + time_fast),
    ]
    for idx, (label, values) in enumerate(panels):
        y_top, y_bottom = panel_bounds(idx)
        lines.append(
            f"<rect x='{margin_l:.1f}' y='{y_top:.1f}' width='{panel_w:.1f}' height='{panel_h:.1f}' "
            "fill='none' stroke='#222' stroke-width='1'/>"
        )
        lines.append(
            f"<text x='{margin_l:.1f}' y='{y_top - 10:.1f}' font-family='monospace' font-size='13' fill='#111'>"
            f"{label}</text>"
        )
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y_bottom - frac * panel_h
            lines.append(
                f"<line x1='{margin_l:.1f}' y1='{yy:.1f}' x2='{margin_l + panel_w:.1f}' y2='{yy:.1f}' "
                "stroke='#e6e6e6' stroke-width='1'/>"
            )
            val = min(values) + frac * (max(values) - min(values))
            lines.append(
                f"<text x='{margin_l - 8:.1f}' y='{yy + 4:.1f}' text-anchor='end' "
                f"font-family='monospace' font-size='11' fill='#666'>{val:.3f}</text>"
            )

    for step in (0, 150, 300, 450, 600):
        x = scale_x(step)
        y_top_0, y_bottom_0 = panel_bounds(0)
        y_top_1, y_bottom_1 = panel_bounds(1)
        lines.append(
            f"<line x1='{x:.1f}' y1='{y_top_0:.1f}' x2='{x:.1f}' y2='{y_bottom_1:.1f}' "
            "stroke='#ececec' stroke-width='1'/>"
        )
        lines.append(
            f"<text x='{x:.1f}' y='{height - 20:.1f}' text-anchor='middle' "
            f"font-family='monospace' font-size='11' fill='#666'>{step}</text>"
        )

    def draw_series(values: List[float], panel_idx: int, stroke: str):
        y_top, y_bottom = panel_bounds(panel_idx)
        if panel_idx == 0:
            vmin = min(val_stable + val_fast)
            vmax = max(val_stable + val_fast)
        else:
            vmin = min(time_stable + time_fast)
            vmax = max(time_stable + time_fast)
        points = []
        for row, value in zip(history_rows, values):
            points.append(
                f"{scale_x(float(row['step'])):.2f},{scale_y(vmin, vmax, y_top, y_bottom, float(value)):.2f}"
            )
        lines.append(
            f"<polyline fill='none' stroke='{stroke}' stroke-width='2.2' points='{' '.join(points)}'/>"
        )

    draw_series(val_stable, 0, "#1355ff")
    draw_series(val_fast, 0, "#d96a00")
    draw_series(time_stable, 1, "#1355ff")
    draw_series(time_fast, 1, "#d96a00")

    legend_y = margin_t + 8
    lines.append(
        f"<line x1='{width - 210:.1f}' y1='{legend_y:.1f}' x2='{width - 182:.1f}' y2='{legend_y:.1f}' "
        "stroke='#1355ff' stroke-width='3'/>"
    )
    lines.append(
        f"<text x='{width - 174:.1f}' y='{legend_y + 4:.1f}' font-family='monospace' font-size='12' fill='#111'>stable</text>"
    )
    lines.append(
        f"<line x1='{width - 106:.1f}' y1='{legend_y:.1f}' x2='{width - 78:.1f}' y2='{legend_y:.1f}' "
        "stroke='#d96a00' stroke-width='3'/>"
    )
    lines.append(
        f"<text x='{width - 70:.1f}' y='{legend_y + 4:.1f}' font-family='monospace' font-size='12' fill='#111'>fast</text>"
    )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _empirical_pair_density(
    samples: np.ndarray,
    dim_i: int,
    dim_j: int,
    xi_edges: np.ndarray,
    xj_edges: np.ndarray,
) -> np.ndarray:
    hist, _, _ = np.histogram2d(samples[:, dim_i], samples[:, dim_j], bins=[xi_edges, xj_edges], density=True)
    return hist


def _iae_2d(d_ref: np.ndarray, d_model: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(np.abs(d_ref - d_model)) * dx * dy)


def _eval_pair_marginal_on_grid(
    ttns,
    parent: Sequence[int],
    bases,
    basis_integrals: jnp.ndarray,
    dim_i: int,
    dim_j: int,
    xi_centers: np.ndarray,
    xj_centers: np.ndarray,
    stable: bool = True,
) -> np.ndarray:
    xi = jnp.asarray(xi_centers, dtype=jnp.float64)
    xj = jnp.asarray(xj_centers, dtype=jnp.float64)
    gx, gy = jnp.meshgrid(xi, xj, indexing="ij")
    pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

    base_vectors = jnp.asarray(basis_integrals, dtype=jnp.float64)
    basis_call = type(bases).__call__
    basis_i = jax.tree_util.tree_map(lambda arr: arr[dim_i], bases)
    basis_j = jax.tree_util.tree_map(lambda arr: arr[dim_j], bases)

    def one_eval(point: jnp.ndarray) -> jnp.ndarray:
        vi = basis_call(basis_i, point[0])
        vj = basis_call(basis_j, point[1])
        vectors = base_vectors.at[dim_i].set(vi).at[dim_j].set(vj)
        return eval_q_ttns(ttns, vectors, parent, stable=stable)

    vals = jax.vmap(one_eval)(pts)
    return np.asarray(vals).reshape((len(xi_centers), len(xj_centers)))


def _build_slice_rows(
    target_samples: np.ndarray,
    stable_model,
    fast_model,
    parent_model: Sequence[int],
    bases,
    basis_integrals: jnp.ndarray,
) -> List[Dict]:
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 56
    slice_rows: List[Dict] = []

    for dim_i, dim_j in pairs:
        s_i = target_samples[:, dim_i]
        s_j = target_samples[:, dim_j]
        xi_edges = np.linspace(float(np.percentile(s_i, 0.5)), float(np.percentile(s_i, 99.5)), grid_bins + 1)
        xj_edges = np.linspace(float(np.percentile(s_j, 0.5)), float(np.percentile(s_j, 99.5)), grid_bins + 1)
        xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        xj_centers = 0.5 * (xj_edges[:-1] + xj_edges[1:])
        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])

        target_density = _empirical_pair_density(target_samples, dim_i, dim_j, xi_edges, xj_edges)
        stable_density = _eval_pair_marginal_on_grid(
            stable_model, parent_model, bases, basis_integrals, dim_i, dim_j, xi_centers, xj_centers, stable=True
        )
        fast_density = _eval_pair_marginal_on_grid(
            fast_model, parent_model, bases, basis_integrals, dim_i, dim_j, xi_centers, xj_centers, stable=True
        )
        iae_stable = _iae_2d(target_density, stable_density, dx, dy)
        iae_fast = _iae_2d(target_density, fast_density, dx, dy)

        slice_rows.append(
            {
                "dim_i": dim_i,
                "dim_j": dim_j,
                "xi_min": float(xi_edges[0]),
                "xi_max": float(xi_edges[-1]),
                "xj_min": float(xj_edges[0]),
                "xj_max": float(xj_edges[-1]),
                "target_density": target_density,
                "stable_density": stable_density,
                "fast_density": fast_density,
                "iae_stable": iae_stable,
                "iae_fast": iae_fast,
                "winner": "stable" if iae_stable < iae_fast else "fast",
            }
        )
    return slice_rows


def _write_history_csv(rows: List[Dict], out_path: Path):
    fieldnames = [
        "step",
        "stable_train_l2",
        "fast_train_l2",
        "stable_val_l2",
        "fast_val_l2",
        "stable_total_sec",
        "fast_total_sec",
        "stable_lr",
        "fast_lr",
        "abs_diff_train_l2",
        "abs_diff_val_l2",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_report(path: Path, cfg: ExperimentConfig, stable_summary: Dict, fast_summary: Dict, history_rows: List[Dict], slice_rows: List[Dict], curve_svg: str, slice_svg: str, history_csv: str):
    speedup = stable_summary["total_time_sec"] / max(fast_summary["total_time_sec"], 1e-12)
    final_val_gap = abs(stable_summary["final_val_l2_ref"] - fast_summary["final_val_l2_ref"])
    mean_train_gap = float(np.mean([row["abs_diff_train_l2"] for row in history_rows]))
    mean_val_gap = float(np.mean([row["abs_diff_val_l2"] for row in history_rows]))

    lines: List[str] = []
    lines.append("# Stable / Fast TTNS 对齐实验报告")
    lines.append("")
    lines.append("## 1. 实验目的")
    lines.append("")
    lines.append("验证 TTNS 的快速收缩实现是否改变训练结果，并量化训练时间收益。")
    lines.append("")
    lines.append("- 目标分布：`balanced target (complex)`")
    lines.append("- 模型结构：`balanced TTNS`")
    lines.append("- 初始化：同一个 seed，同一份训练/验证/切片样本")
    lines.append(f"- 配置：`{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append("")
    lines.append("## 2. 总结论")
    lines.append("")
    lines.append(
        f"- 总训练时间：stable=`{stable_summary['total_time_sec']:.3f}s`，fast=`{fast_summary['total_time_sec']:.3f}s`，加速比=`{speedup:.3f}x`。"
    )
    lines.append(
        f"- 最终对齐评估（统一用 stable evaluator）：stable val_l2=`{stable_summary['final_val_l2_ref']:.6f}`，"
        f"fast val_l2=`{fast_summary['final_val_l2_ref']:.6f}`，绝对差=`{final_val_gap:.6e}`。"
    )
    lines.append(
        f"- 训练轨迹平均差异：mean |train_l2 diff|=`{mean_train_gap:.6e}`，mean |val_l2 diff|=`{mean_val_gap:.6e}`。"
    )
    lines.append("")
    lines.append("## 3. 终点指标")
    lines.append("")
    lines.append("| variant | total_time_sec | best_step | best_val_l2(native) | final_val_l2_ref | final_integral_ref |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| stable | {stable_summary['total_time_sec']:.3f} | {stable_summary['best_step']} | "
        f"{stable_summary['best_val_l2']:.6f} | {stable_summary['final_val_l2_ref']:.6f} | {stable_summary['final_integral_ref']:.6f} |"
    )
    lines.append(
        f"| fast | {fast_summary['total_time_sec']:.3f} | {fast_summary['best_step']} | "
        f"{fast_summary['best_val_l2']:.6f} | {fast_summary['final_val_l2_ref']:.6f} | {fast_summary['final_integral_ref']:.6f} |"
    )
    lines.append("")
    lines.append("## 4. 切片对比")
    lines.append("")
    lines.append(f"- 曲线图：`{curve_svg}`")
    lines.append(f"- 切片图：`{slice_svg}`")
    lines.append(f"- 每 5 步详细历史：`{history_csv}`")
    lines.append("")
    lines.append("| pair | IAE stable | IAE fast | winner |")
    lines.append("|---:|---:|---:|---:|")
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(f"| {pair} | {row['iae_stable']:.6f} | {row['iae_fast']:.6f} | {row['winner']} |")
    lines.append("")
    lines.append("## 5. 解释")
    lines.append("")
    lines.append("- 这次对比中，`stable` 和 `fast` 优化的数学目标完全相同，差异只来自浮点计算顺序变化。")
    lines.append("- 如果最终 `val_l2_ref` 和切片 IAE 只出现很小差异，就说明加速版本没有实质改变训练结果。")
    lines.append("- 如果差异明显，则说明数值路径变化已经足够大，影响了非凸优化轨迹。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _train_variant(
    cfg: ExperimentConfig,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    bases,
    gram_matrices: jnp.ndarray,
    basis_integrals: jnp.ndarray,
    stable: bool,
) -> Tuple[object, Sequence[int], Dict]:
    parent_model = make_parent(cfg.n_dims, "balanced")
    key = jax.random.PRNGKey(cfg.seed + 101)
    key, k_init, k_iter = jax.random.split(key, 3)

    ttns = init_ttns_from_rank1(
        key=k_init,
        bases=bases,
        samples=train_x,
        parent=parent_model,
        rank=cfg.rank,
        noise=cfg.init_noise,
    )
    ttns, z0 = normalize_ttns_by_integral(ttns, basis_integrals, parent_model, stable=stable)

    optimizer = optax.adam(learning_rate=1.0)
    opt_state = optimizer.init(ttns)

    hold_steps = int(0.50 * cfg.train_steps)
    final_lr = cfg.lr * 0.05
    train_monitor = train_x[: cfg.monitor_train_sz]
    val_monitor = val_x[: cfg.monitor_val_sz]
    train_monitor_basis = batch_basis_vectors_from_samples(bases, train_monitor)
    val_monitor_basis = batch_basis_vectors_from_samples(bases, val_monitor)

    history: List[Dict] = []
    best_val_l2 = float("inf")
    best_step = 0
    best_ttns = ttns
    stopped_early = False
    stop_step = cfg.train_steps

    variant_name = "stable" if stable else "fast"
    t_start = time.perf_counter()
    t_prev = t_start
    print(f"\n=== [{variant_name}] init integral={float(z0):.6e} ===", flush=True)
    print("step,train_l2,val_l2,integral,lr,interval_sec,total_sec", flush=True)

    @jax.jit
    def train_step(curr_ttns, curr_opt_state, noisy_batch, lr_t):
        loss, grads = value_and_grad(
            lambda x: l2_objective_ttns(
                x,
                batch_basis_vectors_from_samples(bases, noisy_batch),
                gram_matrices,
                parent_model,
                stable=stable,
            )
        )(curr_ttns)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_ttns)
        updates = jax.tree_util.tree_map(lambda u: u * lr_t, updates)
        next_ttns = optax.apply_updates(curr_ttns, updates)
        next_ttns, z = normalize_ttns_by_integral(next_ttns, basis_integrals, parent_model, stable=stable)
        return next_ttns, next_opt_state, loss, z

    eval_int_q2_native = jax.jit(
        lambda curr_ttns: integral_q2_ttns(curr_ttns, gram_matrices, parent_model, stable=stable)
    )
    eval_mc_native = jax.jit(
        lambda curr_ttns, basis_batch: mc_expectation_q_ttns(curr_ttns, basis_batch, parent_model, stable=stable)
    )
    eval_l2_ref = jax.jit(
        lambda curr_ttns, xs: l2_objective_ttns(
            curr_ttns,
            batch_basis_vectors_from_samples(bases, xs),
            gram_matrices,
            parent_model,
            stable=True,
        )
    )
    eval_integral_ref = jax.jit(lambda curr_ttns: integral_q_ttns(curr_ttns, basis_integrals, parent_model, stable=True))

    for step in range(1, cfg.train_steps + 1):
        k_iter, k_idx, k_noise = jax.random.split(k_iter, 3)
        idx = jax.random.randint(k_idx, (cfg.batch_sz,), 0, train_x.shape[0])
        batch = train_x[idx]
        noisy_batch = batch + jax.random.normal(k_noise, batch.shape) * cfg.train_noise
        lr_t = _lr_for_step(step, cfg.train_steps, cfg.lr, final_lr, hold_steps)
        ttns, opt_state, _, curr_integral = train_step(ttns, opt_state, noisy_batch, lr_t)

        if step % cfg.log_every == 0:
            now = time.perf_counter()
            int_q2 = eval_int_q2_native(ttns)
            train_l2 = int_q2 - 2.0 * eval_mc_native(ttns, train_monitor_basis)
            val_l2 = int_q2 - 2.0 * eval_mc_native(ttns, val_monitor_basis)
            interval = now - t_prev
            total = now - t_start
            t_prev = now
            history.append(
                {
                    "step": step,
                    "train_l2": float(train_l2),
                    "val_l2": float(val_l2),
                    "integral": float(curr_integral),
                    "lr": float(lr_t),
                    "interval_sec": float(interval),
                    "total_sec": float(total),
                }
            )
            print(
                f"{step},{float(train_l2):.6f},{float(val_l2):.6f},{float(curr_integral):.6e},"
                f"{float(lr_t):.3e},{interval:.3f},{total:.3f}",
                flush=True,
            )
            if float(val_l2) < best_val_l2:
                best_val_l2 = float(val_l2)
                best_step = step
                best_ttns = ttns

    if bool(cfg.early_stop_restore_best):
        ttns = best_ttns

    final_val_l2_ref = float(eval_l2_ref(ttns, val_x))
    final_integral_ref = float(eval_integral_ref(ttns))
    total_time_sec = float(time.perf_counter() - t_start)
    summary = {
        "variant": variant_name,
        "stable": bool(stable),
        "best_step": int(best_step),
        "best_val_l2": float(best_val_l2),
        "stopped_early": bool(stopped_early),
        "stop_step": int(stop_step),
        "final_val_l2_ref": final_val_l2_ref,
        "final_integral_ref": final_integral_ref,
        "total_time_sec": total_time_sec,
        "history": history,
    }
    return ttns, parent_model, summary


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _config()
    target_topology = "balanced"
    parent_target = make_parent(cfg.n_dims, target_topology)

    key = jax.random.PRNGKey(cfg.seed + 33333)
    k_train, k_val, k_slice = jax.random.split(key, 3)
    train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
    val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
    slice_x = sample_complex_tree_distribution(k_slice, 220000, parent_target)

    print("building bases...", flush=True)
    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    stable_model, parent_model, stable_summary = _train_variant(
        cfg, train_x, val_x, bases, gram_matrices, basis_integrals, stable=True
    )
    fast_model, _, fast_summary = _train_variant(
        cfg, train_x, val_x, bases, gram_matrices, basis_integrals, stable=False
    )

    history_rows: List[Dict] = []
    for stable_row, fast_row in zip(stable_summary["history"], fast_summary["history"]):
        history_rows.append(
            {
                "step": int(stable_row["step"]),
                "stable_train_l2": float(stable_row["train_l2"]),
                "fast_train_l2": float(fast_row["train_l2"]),
                "stable_val_l2": float(stable_row["val_l2"]),
                "fast_val_l2": float(fast_row["val_l2"]),
                "stable_total_sec": float(stable_row["total_sec"]),
                "fast_total_sec": float(fast_row["total_sec"]),
                "stable_lr": float(stable_row["lr"]),
                "fast_lr": float(fast_row["lr"]),
                "abs_diff_train_l2": abs(float(stable_row["train_l2"]) - float(fast_row["train_l2"])),
                "abs_diff_val_l2": abs(float(stable_row["val_l2"]) - float(fast_row["val_l2"])),
            }
        )

    slice_rows = _build_slice_rows(
        np.asarray(slice_x), stable_model, fast_model, parent_model, bases, basis_integrals
    )

    metrics = {
        "config": asdict(cfg),
        "stable_summary": stable_summary,
        "fast_summary": fast_summary,
        "history_rows": history_rows,
        "slice_rows": [
            {
                **{k: v for k, v in row.items() if k not in ("target_density", "stable_density", "fast_density")},
                "target_density_shape": list(row["target_density"].shape),
                "stable_density_shape": list(row["stable_density"].shape),
                "fast_density_shape": list(row["fast_density"].shape),
            }
            for row in slice_rows
        ],
    }

    metrics_path = REPORT_DIR / "stable_fast_balanced_metrics.json"
    history_csv_path = REPORT_DIR / "stable_fast_balanced_history.csv"
    curve_svg_path = REPORT_DIR / "stable_fast_balanced_curve.svg"
    slice_svg_path = REPORT_DIR / "stable_fast_balanced_slices.svg"
    report_path = REPORT_DIR / "stable_fast_balanced_report_zh.md"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_history_csv(history_rows, history_csv_path)
    _write_curve_svg(history_rows, curve_svg_path)
    _write_slice_svg(slice_rows, slice_svg_path)
    _write_report(
        report_path,
        cfg,
        stable_summary,
        fast_summary,
        history_rows,
        slice_rows,
        curve_svg_path.name,
        slice_svg_path.name,
        history_csv_path.name,
    )

    print("saved:", metrics_path, flush=True)
    print("saved:", history_csv_path, flush=True)
    print("saved:", curve_svg_path, flush=True)
    print("saved:", slice_svg_path, flush=True)
    print("saved:", report_path, flush=True)


if __name__ == "__main__":
    run()
