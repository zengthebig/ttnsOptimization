from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

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

from simple_ttns_l2.train_l2 import build_bases, make_parent
from simple_ttns_l2.experiments.topology_comparison import ExperimentConfig
from simple_ttns_l2.experiments.topology_comparison_complex import sample_complex_tree_distribution
from simple_ttns_l2.experiments.topology_slice_visualization import (
    _empirical_pair_density,
    _eval_pair_marginal_on_grid,
    _iae_2d,
    _train_one_model,
)
from ttde.score.models.continuous_canonical_init import continuous_rank_1
from ttde.tt.tt_opt import TTOpt, TTOperatorOpt, normalized_dot_operator, normalized_inner_product

jax.config.update("jax_enable_x64", True)


def _recover_scalar(normalized) -> jnp.ndarray:
    val = normalized.value * jnp.exp(normalized.log_norm)
    return jnp.asarray(val).reshape(()).astype(jnp.float64)


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
        seed=514,
        n_train=8000,
        n_val=4000,
        n_test=4000,
        monitor_train_sz=3000,
        monitor_val_sz=3000,
    )


def _lr_for_step(step: int, train_steps: int, init_lr: float, final_lr: float, hold_steps: int) -> float:
    if step <= hold_steps:
        return init_lr
    denom = max(1, train_steps - hold_steps)
    progress = min(1.0, max(0.0, (step - hold_steps) / denom))
    return final_lr + 0.5 * (init_lr - final_lr) * (1.0 + np.cos(np.pi * progress))


def _init_tt_from_rank1(
    key: jnp.ndarray,
    bases,
    samples: jnp.ndarray,
    rank: int,
    noise: float,
) -> TTOpt:
    rank1 = continuous_rank_1(bases, samples, jnp.ones(len(samples)))
    canonical = jnp.pad(rank1[None], ((0, rank - 1), (0, 0), (0, 0)))
    tt = TTOpt.from_canonical(canonical)
    if noise <= 0:
        return tt
    k1, k2, k3 = jax.random.split(key, 3)
    return TTOpt(
        first=tt.first + jax.random.normal(k1, tt.first.shape) * noise,
        inner=tt.inner + jax.random.normal(k2, tt.inner.shape) * noise,
        last=tt.last + jax.random.normal(k3, tt.last.shape) * noise,
    )


def _eval_q_tt(tt: TTOpt, basis_vectors: jnp.ndarray) -> jnp.ndarray:
    b_tensor = TTOpt.rank_1_from_vectors(basis_vectors)
    return _recover_scalar(normalized_inner_product(tt, b_tensor))


def _basis_vectors_at_point(bases, x: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(type(bases).__call__)(bases, x)


def _eval_q_tt_batch(tt: TTOpt, basis_batch: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(_eval_q_tt, in_axes=(None, 0))(tt, basis_batch)


def _integral_q_tt(tt: TTOpt, basis_integrals: jnp.ndarray) -> jnp.ndarray:
    return _eval_q_tt(tt, basis_integrals)


def _integral_q2_tt(tt: TTOpt, gram_matrices: jnp.ndarray) -> jnp.ndarray:
    d_op = TTOperatorOpt.rank_1_from_matrices(gram_matrices)
    return _recover_scalar(normalized_inner_product(tt, normalized_dot_operator(tt, d_op)))


def _batch_basis_vectors_from_samples(bases, xs: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(_basis_vectors_at_point, in_axes=(None, 0))(bases, xs)


def _l2_loss_tt(tt: TTOpt, bases, xs_batch: jnp.ndarray, gram_matrices: jnp.ndarray) -> jnp.ndarray:
    basis_batch = _batch_basis_vectors_from_samples(bases, xs_batch)
    int_q2 = _integral_q2_tt(tt, gram_matrices)
    mc_q = _eval_q_tt_batch(tt, basis_batch).mean()
    return int_q2 - 2.0 * mc_q


def _normalize_tt(tt: TTOpt, basis_integrals: jnp.ndarray, eps: float = 1e-12) -> Tuple[TTOpt, jnp.ndarray]:
    z = _integral_q_tt(tt, basis_integrals)
    safe_z = jnp.where(jnp.abs(z) < eps, 1.0, z)
    scale = 1.0 / safe_z
    return TTOpt(first=tt.first * scale, inner=tt.inner, last=tt.last), z


def _train_tt_model(
    cfg: ExperimentConfig,
    target_topology: str,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    bases,
    gram_matrices: jnp.ndarray,
    basis_integrals: jnp.ndarray,
    seed_offset: int,
) -> Tuple[TTOpt, Dict]:
    key = jax.random.PRNGKey(cfg.seed + seed_offset)
    key, k_init, k_iter = jax.random.split(key, 3)
    tt = _init_tt_from_rank1(k_init, bases, train_x, rank=cfg.rank, noise=cfg.init_noise)
    tt, z0 = _normalize_tt(tt, basis_integrals)

    optimizer = optax.adam(learning_rate=1.0)
    opt_state = optimizer.init(tt)

    hold_steps = int(0.40 * cfg.train_steps)
    final_lr = cfg.lr * 0.1
    history = []
    train_monitor = train_x[: cfg.monitor_train_sz]
    val_monitor = val_x[: cfg.monitor_val_sz]
    best_val_l2 = float("inf")
    best_step = 0
    bad_logs = 0
    logs_seen = 0
    best_tt = tt
    stopped_early = False
    stop_step = cfg.train_steps

    t_start = time.perf_counter()
    t_prev = t_start
    print(
        f"\n=== [{target_topology} target] [tt model] init integral={float(z0):.6e} ===",
        flush=True,
    )
    print("step,train_l2,val_l2,integral,lr,interval_sec,total_sec", flush=True)

    @jax.jit
    def train_step(curr_tt, curr_opt_state, noisy_batch, lr_t):
        loss, grads = value_and_grad(_l2_loss_tt, argnums=0)(curr_tt, bases, noisy_batch, gram_matrices)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_tt)
        updates = jax.tree_util.tree_map(lambda u: u * lr_t, updates)
        next_tt = optax.apply_updates(curr_tt, updates)
        next_tt, z = _normalize_tt(next_tt, basis_integrals)
        return next_tt, next_opt_state, loss, z

    eval_l2 = jax.jit(lambda curr_tt, xs: _l2_loss_tt(curr_tt, bases, xs, gram_matrices))
    eval_integral = jax.jit(lambda curr_tt: _integral_q_tt(curr_tt, basis_integrals))

    for step in range(1, cfg.train_steps + 1):
        k_iter, k_idx, k_noise = jax.random.split(k_iter, 3)
        idx = jax.random.randint(k_idx, (cfg.batch_sz,), 0, train_x.shape[0])
        batch = train_x[idx]
        noisy_batch = batch + jax.random.normal(k_noise, batch.shape) * cfg.train_noise
        lr_t = _lr_for_step(step, cfg.train_steps, cfg.lr, final_lr, hold_steps)
        tt, opt_state, _, _ = train_step(tt, opt_state, noisy_batch, lr_t)

        if step % cfg.log_every == 0:
            now = time.perf_counter()
            train_l2 = eval_l2(tt, train_monitor)
            val_l2 = eval_l2(tt, val_monitor)
            z = eval_integral(tt)
            interval = now - t_prev
            total = now - t_start
            t_prev = now
            history.append(
                {
                    "step": step,
                    "train_l2": float(train_l2),
                    "val_l2": float(val_l2),
                    "integral": float(z),
                    "lr": float(lr_t),
                    "interval_sec": interval,
                    "total_sec": total,
                }
            )
            print(
                f"{step},{float(train_l2):.6f},{float(val_l2):.6f},{float(z):.6e},"
                f"{lr_t:.3e},{interval:.3f},{total:.3f}",
                flush=True,
            )

            logs_seen += 1
            val_l2_f = float(val_l2)
            improved = val_l2_f < (best_val_l2 - float(cfg.early_stop_min_delta))
            if improved:
                best_val_l2 = val_l2_f
                best_step = step
                best_tt = tt
                bad_logs = 0
            elif logs_seen > int(cfg.early_stop_warmup_logs):
                bad_logs += 1

            if int(cfg.early_stop_patience_logs) > 0 and bad_logs >= int(cfg.early_stop_patience_logs):
                stopped_early = True
                stop_step = step
                print(
                    f"early_stop at step={step}: best_val_l2={best_val_l2:.6f} (step={best_step}), "
                    f"patience_logs={cfg.early_stop_patience_logs}, min_delta={cfg.early_stop_min_delta}",
                    flush=True,
                )
                break

    if bool(cfg.early_stop_restore_best):
        tt = best_tt

    final_val_l2 = float(eval_l2(tt, val_x))
    final_integral = float(eval_integral(tt))
    total_time = time.perf_counter() - t_start
    summary = {
        "target_topology": target_topology,
        "model_topology": "tt",
        "lr_schedule": f"tt_delayed_cosine_hold{hold_steps}_0.1x",
        "final_lr": float(_lr_for_step(stop_step, cfg.train_steps, cfg.lr, final_lr, hold_steps)),
        "final_val_l2": final_val_l2,
        "final_integral": final_integral,
        "total_time_sec": total_time,
        "stopped_early": stopped_early,
        "stop_step": int(stop_step),
        "best_step": int(best_step),
        "best_val_l2": float(best_val_l2),
        "history": history,
    }
    return tt, summary


def _eval_pair_marginal_tt(
    tt: TTOpt,
    bases,
    basis_integrals: jnp.ndarray,
    dim_i: int,
    dim_j: int,
    xi_centers: np.ndarray,
    xj_centers: np.ndarray,
) -> np.ndarray:
    xi = jnp.asarray(xi_centers, dtype=jnp.float64)
    xj = jnp.asarray(xj_centers, dtype=jnp.float64)
    gx, gy = jnp.meshgrid(xi, xj, indexing="ij")
    pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

    base_vectors = jnp.asarray(basis_integrals, dtype=jnp.float64)
    basis_call = type(bases).__call__
    basis_i = jax.tree_util.tree_map(lambda arr: arr[dim_i], bases)
    basis_j = jax.tree_util.tree_map(lambda arr: arr[dim_j], bases)

    def one_eval(point):
        vi = basis_call(basis_i, point[0])
        vj = basis_call(basis_j, point[1])
        vectors = base_vectors.at[dim_i].set(vi).at[dim_j].set(vj)
        return _eval_q_tt(tt, vectors)

    vals = jax.vmap(one_eval)(pts)
    return np.asarray(vals).reshape((len(xi_centers), len(xj_centers)))


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
    cols = ["target empirical", "model balanced_ttns", "model tt"]

    width = int(left + 3 * panel_w + 2 * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 80)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>"
        "Fit Ceiling: Balanced-TTNS vs Pure-TT (Same Target)"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Same target data, high-capacity setting, compare balanced TTNS and TT."
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
            f"IAE_balTTNS={row['iae_bal_ttns']:.4f}, IAE_TT={row['iae_tt']:.4f}, floor={row['noise_floor']:.4f}"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"ratio_balTTNS={row['ratio_bal_ttns']:.2f}x, ratio_TT={row['ratio_tt']:.2f}x"
            "</text>"
        )

        v = max(
            float(np.max(row["target_density"])),
            float(np.max(np.maximum(row["balanced_density"], 0.0))),
            float(np.max(np.maximum(row["tt_density"], 0.0))),
            1e-12,
        )
        panels = [row["target_density"], row["balanced_density"], row["tt_density"]]
        for c, panel in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            _draw_panel(lines, x, y, panel_w, panel_h, panel, vmax=v)
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


def _write_report(path: Path, cfg: ExperimentConfig, train_rows: List[Dict], slice_rows: List[Dict], svg_name: str):
    lines: List[str] = []
    lines.append("# Fit Ceiling Report: Balanced-TTNS vs Pure-TT")
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
            f"| {row['target_topology']} | {row['model_topology']} | {row['lr_schedule']} | {row['final_lr']:.3e} | "
            f"{row['final_val_l2']:.6f} | {row['final_integral']:.6f} | {row['total_time_sec']:.3f} |"
        )
    lines.append("")
    lines.append("## Slice Summary")
    lines.append("")
    lines.append("| target_topology | pair | noise_floor_IAE | IAE_balTTNS | IAE_TT | ratio_balTTNS | ratio_TT |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {row['target_topology']} | {pair} | {row['noise_floor']:.6f} | {row['iae_bal_ttns']:.6f} | "
            f"{row['iae_tt']:.6f} | {row['ratio_bal_ttns']:.3f} | {row['ratio_tt']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    cfg = _config()
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 48
    n_slice_samples = 200000

    print("running balanced-ttns vs tt with config:", asdict(cfg), flush=True)
    seed_cursor = 0
    train_rows: List[Dict] = []
    slice_rows: List[Dict] = []

    for target_topology in ("balanced", "chain"):
        parent_target = make_parent(cfg.n_dims, target_topology)
        key = jax.random.PRNGKey(cfg.seed + 15000 + seed_cursor)
        seed_cursor += 100
        k_train, k_val, k_slice_a, k_slice_b = jax.random.split(key, 4)

        train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
        val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
        slice_a = sample_complex_tree_distribution(k_slice_a, n_slice_samples, parent_target)
        slice_b = sample_complex_tree_distribution(k_slice_b, n_slice_samples, parent_target)

        bases = build_bases(train_x, q=cfg.q, m=cfg.m)
        gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
        basis_integrals = jax.vmap(type(bases).integral)(bases)

        # Model A: balanced TTNS
        bal_ttns, bal_parent, summary_bal = _train_one_model(
            cfg=cfg,
            target_topology=target_topology,
            model_topology="balanced",
            train_x=train_x,
            val_x=val_x,
            bases=bases,
            gram_matrices=gram_matrices,
            basis_integrals=basis_integrals,
            seed_offset=seed_cursor,
            lr_policy_template={
                "mode": "delayed_cosine",
                "label": f"ceiling_delayed_cosine_hold{int(0.40*cfg.train_steps)}_0.1x",
                "init_lr": float(cfg.lr),
                "final_lr": float(cfg.lr * 0.1),
                "hold_steps": int(0.40 * cfg.train_steps),
                "patience_logs": 0,
                "factor": 1.0,
                "min_lr": float(cfg.lr * 0.1),
                "min_delta": 0.0,
                "cooldown_logs": 0,
                "best_val": float("inf"),
                "bad_logs": 0,
                "cooldown_left": 0,
                "curr_lr": float(cfg.lr),
            },
        )
        seed_cursor += 1
        train_rows.append(summary_bal)

        # Model B: pure TT
        tt_model, summary_tt = _train_tt_model(
            cfg=cfg,
            target_topology=target_topology,
            train_x=train_x,
            val_x=val_x,
            bases=bases,
            gram_matrices=gram_matrices,
            basis_integrals=basis_integrals,
            seed_offset=seed_cursor,
        )
        seed_cursor += 1
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
                ttns=bal_ttns,
                parent=bal_parent,
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
            floor = _iae_2d(target_density, target_density_2, dx, dy)
            iae_bal = _iae_2d(target_density, balanced_density, dx, dy)
            iae_tt = _iae_2d(target_density, tt_density, dx, dy)
            row = {
                "target_topology": target_topology,
                "dim_i": dim_i,
                "dim_j": dim_j,
                "noise_floor": floor,
                "iae_bal_ttns": iae_bal,
                "iae_tt": iae_tt,
                "ratio_bal_ttns": float(iae_bal / max(floor, 1e-12)),
                "ratio_tt": float(iae_tt / max(floor, 1e-12)),
                "target_density": target_density,
                "balanced_density": balanced_density,
                "tt_density": tt_density,
            }
            slice_rows.append(row)
            print(
                f"slice target={target_topology}, pair=({dim_i},{dim_j}), floor={floor:.6f}, "
                f"IAE_balTTNS={iae_bal:.6f}, IAE_TT={iae_tt:.6f}",
                flush=True,
            )

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_svg = report_dir / "fit_ceiling_balanced_ttns_vs_tt.svg"
    out_md = report_dir / "fit_ceiling_balanced_ttns_vs_tt_report.md"
    out_json = report_dir / "fit_ceiling_balanced_ttns_vs_tt_metrics.json"

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
                "iae_bal_ttns": row["iae_bal_ttns"],
                "iae_tt": row["iae_tt"],
                "ratio_bal_ttns": row["ratio_bal_ttns"],
                "ratio_tt": row["ratio_tt"],
            }
        )
    out_json.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "training": train_rows,
                "slice_metrics": compact_rows,
                "generated_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== balanced-ttns vs tt done ===", flush=True)
    print("saved svg   :", out_svg, flush=True)
    print("saved report:", out_md, flush=True)
    print("saved metric:", out_json, flush=True)


if __name__ == "__main__":
    run()
