from __future__ import annotations

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

from simple_ttns_l2.objective import (
    batch_basis_vectors_from_samples,
    eval_q_ttns,
    integral_q2_ttns,
    integral_q_ttns,
    mc_expectation_q_ttns,
    normalize_ttns_by_integral,
)
from simple_ttns_l2.train_l2 import (
    build_bases,
    init_ttns_from_rank1,
    l2_loss_on_batch,
    make_parent,
)
from simple_ttns_l2.experiments.topology_comparison import (
    ExperimentConfig,
    make_tree_gaussian_params,
    sample_tree_gaussian,
)
from simple_ttns_l2.experiments.topology_comparison_complex import sample_complex_tree_distribution

jax.config.update("jax_enable_x64", True)


def _format_seconds(sec: float) -> str:
    return f"{sec:.3f}s"


def _sample_target(
    target_kind: str,
    key: jnp.ndarray,
    n_samples: int,
    parent_target: Sequence[int],
) -> jnp.ndarray:
    if target_kind == "linear":
        weights, noises = make_tree_gaussian_params(parent_target)
        return sample_tree_gaussian(key, n_samples, parent_target, weights, noises)
    if target_kind in (
        "complex",
        "complex_large",
        "complex_xlarge",
        "complex_xlarge_decay",
        "complex_xlarge_delaydecay",
        "complex_xlarge_adaptive",
    ):
        return sample_complex_tree_distribution(key, n_samples, list(parent_target))
    raise ValueError(f"unsupported target_kind={target_kind}")


def _config_for_target_kind(target_kind: str) -> ExperimentConfig:
    if target_kind == "linear":
        return ExperimentConfig()
    if target_kind == "complex":
        return ExperimentConfig(
            n_dims=6,
            q=2,
            m=24,
            rank=3,
            batch_sz=256,
            lr=1e-3,
            train_noise=1e-2,
            init_noise=1e-2,
            train_steps=100,
            log_every=5,
            seed=123,
            n_train=5000,
            n_val=2500,
            n_test=5000,
            monitor_train_sz=2000,
            monitor_val_sz=2000,
        )
    if target_kind == "complex_large":
        return ExperimentConfig(
            n_dims=6,
            q=2,
            m=36,
            rank=6,
            batch_sz=256,
            lr=1e-3,
            train_noise=1e-2,
            init_noise=1e-2,
            train_steps=100,
            log_every=5,
            seed=123,
            n_train=5000,
            n_val=2500,
            n_test=5000,
            monitor_train_sz=2000,
            monitor_val_sz=2000,
        )
    if target_kind == "complex_xlarge":
        return ExperimentConfig(
            n_dims=6,
            q=2,
            m=48,
            rank=10,
            batch_sz=256,
            lr=1e-3,
            train_noise=1e-2,
            init_noise=1e-2,
            train_steps=100,
            log_every=5,
            seed=123,
            n_train=5000,
            n_val=2500,
            n_test=5000,
            monitor_train_sz=2000,
            monitor_val_sz=2000,
        )
    if target_kind == "complex_xlarge_decay":
        return ExperimentConfig(
            n_dims=6,
            q=2,
            m=48,
            rank=10,
            batch_sz=256,
            lr=5e-4,
            train_noise=1e-2,
            init_noise=1e-2,
            train_steps=100,
            log_every=5,
            seed=123,
            n_train=5000,
            n_val=2500,
            n_test=5000,
            monitor_train_sz=2000,
            monitor_val_sz=2000,
        )
    if target_kind == "complex_xlarge_delaydecay":
        return ExperimentConfig(
            n_dims=6,
            q=2,
            m=48,
            rank=10,
            batch_sz=256,
            lr=1e-3,
            train_noise=1e-2,
            init_noise=1e-2,
            train_steps=100,
            log_every=5,
            seed=123,
            n_train=5000,
            n_val=2500,
            n_test=5000,
            monitor_train_sz=2000,
            monitor_val_sz=2000,
        )
    if target_kind == "complex_xlarge_adaptive":
        return ExperimentConfig(
            n_dims=6,
            q=2,
            m=48,
            rank=10,
            batch_sz=256,
            lr=1e-3,
            train_noise=1e-2,
            init_noise=1e-2,
            train_steps=100,
            log_every=5,
            seed=123,
            n_train=5000,
            n_val=2500,
            n_test=5000,
            monitor_train_sz=2000,
            monitor_val_sz=2000,
        )
    raise ValueError(f"unsupported target_kind={target_kind}")


def _build_lr_policy(target_kind: str, cfg: ExperimentConfig) -> Dict:
    hold_steps = max(0, int(0.40 * cfg.train_steps))
    if target_kind == "complex_xlarge_decay":
        return {
            "mode": "cosine",
            "label": "cosine_decay_0.1x",
            "init_lr": float(cfg.lr),
            "final_lr": float(cfg.lr * 0.1),
            "hold_steps": 0,
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
    if target_kind == "complex_xlarge_delaydecay":
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
    if target_kind == "complex_xlarge_adaptive":
        return {
            "mode": "delayed_adaptive",
            "label": f"delayed_adaptive_hold{hold_steps}_factor0.5_pat2",
            "init_lr": float(cfg.lr),
            "final_lr": float(cfg.lr * 0.1),
            "hold_steps": hold_steps,
            "patience_logs": 2,
            "factor": 0.5,
            "min_lr": 1e-5,
            "min_delta": 1e-4,
            "cooldown_logs": 1,
            "best_val": float("inf"),
            "bad_logs": 0,
            "cooldown_left": 0,
            "curr_lr": float(cfg.lr),
        }
    return {
        "mode": "constant",
        "label": "constant",
        "init_lr": float(cfg.lr),
        "final_lr": float(cfg.lr),
        "hold_steps": 0,
        "patience_logs": 0,
        "factor": 1.0,
        "min_lr": float(cfg.lr),
        "min_delta": 0.0,
        "cooldown_logs": 0,
        "best_val": float("inf"),
        "bad_logs": 0,
        "cooldown_left": 0,
        "curr_lr": float(cfg.lr),
    }


def _lr_for_step(policy: Dict, step: int, train_steps: int) -> float:
    mode = policy["mode"]
    if mode == "constant":
        return float(policy["init_lr"])
    if mode in ("cosine", "delayed_cosine"):
        hold = int(policy["hold_steps"])
        init_lr = float(policy["init_lr"])
        final_lr = float(policy["final_lr"])
        if step <= hold:
            return init_lr
        denom = max(1, train_steps - hold)
        progress = min(1.0, max(0.0, (step - hold) / denom))
        return final_lr + 0.5 * (init_lr - final_lr) * (1.0 + np.cos(np.pi * progress))
    if mode == "delayed_adaptive":
        return float(policy["curr_lr"])
    raise ValueError(f"unsupported lr mode={mode}")


def _update_policy_after_log(policy: Dict, step: int, val_l2: float) -> Tuple[Dict, str]:
    if policy["mode"] != "delayed_adaptive":
        return policy, ""
    if step <= int(policy["hold_steps"]):
        return policy, ""

    if policy["cooldown_left"] > 0:
        policy["cooldown_left"] -= 1

    improved = val_l2 < (policy["best_val"] - policy["min_delta"])
    if improved:
        policy["best_val"] = float(val_l2)
        policy["bad_logs"] = 0
        return policy, ""

    policy["bad_logs"] += 1
    if policy["bad_logs"] < policy["patience_logs"] or policy["cooldown_left"] > 0:
        return policy, ""

    old_lr = float(policy["curr_lr"])
    new_lr = max(float(policy["min_lr"]), old_lr * float(policy["factor"]))
    policy["bad_logs"] = 0
    policy["cooldown_left"] = int(policy["cooldown_logs"])
    if new_lr < old_lr - 1e-15:
        policy["curr_lr"] = new_lr
        msg = (
            f"adaptive_lr_decay at step={step}: "
            f"{old_lr:.3e} -> {new_lr:.3e} (val_l2={val_l2:.6f})"
        )
        return policy, msg
    return policy, ""


def _train_one_model(
    cfg: ExperimentConfig,
    target_topology: str,
    model_topology: str,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    bases,
    gram_matrices: jnp.ndarray,
    basis_integrals: jnp.ndarray,
    seed_offset: int,
    lr_policy_template: Dict,
    model_parent: Sequence[int] | None = None,
    model_edge_ranks: Dict[Tuple[int, int], int] | None = None,
):
    parent_model = list(model_parent) if model_parent is not None else make_parent(cfg.n_dims, model_topology)
    key = jax.random.PRNGKey(cfg.seed + seed_offset)
    key, k_init, k_iter = jax.random.split(key, 3)

    ttns = init_ttns_from_rank1(
        key=k_init,
        bases=bases,
        samples=train_x,
        parent=parent_model,
        rank=cfg.rank,
        noise=cfg.init_noise,
        edge_ranks=model_edge_ranks,
    )
    ttns, z0 = normalize_ttns_by_integral(ttns, basis_integrals, parent_model)
    lr_policy = dict(lr_policy_template)
    optimizer = optax.adam(learning_rate=1.0)
    opt_state = optimizer.init(ttns)

    train_monitor = train_x[: cfg.monitor_train_sz]
    val_monitor = val_x[: cfg.monitor_val_sz]
    train_monitor_basis = batch_basis_vectors_from_samples(bases, train_monitor)
    val_monitor_basis = batch_basis_vectors_from_samples(bases, val_monitor)
    history = []
    best_val_l2 = float("inf")
    best_step = 0
    bad_logs = 0
    logs_seen = 0
    best_ttns = ttns
    stopped_early = False
    stop_step = cfg.train_steps

    t_start = time.perf_counter()
    t_prev = t_start
    print(
        f"\n=== [{target_topology} target] [{model_topology} model] init integral={float(z0):.6e} ===",
        flush=True,
    )
    print(
        f"step,train_l2,val_l2,integral,lr,interval_sec,total_sec "
        f"(lr_policy={lr_policy['label']})",
        flush=True,
    )

    @jax.jit
    def train_step(curr_ttns, curr_opt_state, noisy_batch, lr_t):
        loss, grads = value_and_grad(
            lambda x: l2_loss_on_batch(x, bases, noisy_batch, parent_model, gram_matrices)
        )(curr_ttns)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_ttns)
        updates = jax.tree_util.tree_map(lambda u: u * lr_t, updates)
        next_ttns = optax.apply_updates(curr_ttns, updates)
        next_ttns, z = normalize_ttns_by_integral(next_ttns, basis_integrals, parent_model)
        return next_ttns, next_opt_state, loss, z

    eval_l2 = jax.jit(lambda curr_ttns, xs: l2_loss_on_batch(curr_ttns, bases, xs, parent_model, gram_matrices))
    eval_int_q2 = jax.jit(lambda curr_ttns: integral_q2_ttns(curr_ttns, gram_matrices, parent_model))
    eval_mc = jax.jit(
        lambda curr_ttns, basis_batch: mc_expectation_q_ttns(curr_ttns, basis_batch, parent_model)
    )
    eval_integral = jax.jit(lambda curr_ttns: integral_q_ttns(curr_ttns, basis_integrals, parent_model))

    for step in range(1, cfg.train_steps + 1):
        k_iter, k_idx, k_noise = jax.random.split(k_iter, 3)
        idx = jax.random.randint(k_idx, (cfg.batch_sz,), 0, train_x.shape[0])
        batch = train_x[idx]
        noisy_batch = batch + jax.random.normal(k_noise, batch.shape) * cfg.train_noise
        lr_t = _lr_for_step(lr_policy, step, cfg.train_steps)
        ttns, opt_state, _, curr_integral = train_step(ttns, opt_state, noisy_batch, lr_t)

        if step % cfg.log_every == 0:
            now = time.perf_counter()
            int_q2 = eval_int_q2(ttns)
            train_l2 = int_q2 - 2.0 * eval_mc(ttns, train_monitor_basis)
            val_l2 = int_q2 - 2.0 * eval_mc(ttns, val_monitor_basis)
            integral = curr_integral
            interval = now - t_prev
            total = now - t_start
            t_prev = now

            lr_policy, lr_msg = _update_policy_after_log(lr_policy, step, float(val_l2))
            curr_lr = _lr_for_step(lr_policy, step, cfg.train_steps)
            history.append(
                {
                    "step": step,
                    "train_l2": float(train_l2),
                    "val_l2": float(val_l2),
                    "integral": float(integral),
                    "lr": float(curr_lr),
                    "interval_sec": interval,
                    "total_sec": total,
                }
            )
            print(
                f"{step},{float(train_l2):.6f},{float(val_l2):.6f},{float(integral):.6e},"
                f"{float(curr_lr):.3e},"
                f"{interval:.3f},{total:.3f}",
                flush=True,
            )
            if lr_msg:
                print(lr_msg, flush=True)

            logs_seen += 1
            val_l2_f = float(val_l2)
            improved = val_l2_f < (best_val_l2 - float(cfg.early_stop_min_delta))
            if improved:
                best_val_l2 = val_l2_f
                best_step = step
                best_ttns = ttns
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
        ttns = best_ttns

    final_val_l2 = float(eval_l2(ttns, val_x))
    final_integral = float(eval_integral(ttns))
    total_time = time.perf_counter() - t_start
    summary = {
        "target_topology": target_topology,
        "model_topology": model_topology,
        "lr_schedule": lr_policy["label"],
        "final_lr": float(_lr_for_step(lr_policy, stop_step, cfg.train_steps)),
        "final_val_l2": final_val_l2,
        "final_integral": final_integral,
        "total_time_sec": total_time,
        "stopped_early": stopped_early,
        "stop_step": int(stop_step),
        "best_step": int(best_step),
        "best_val_l2": float(best_val_l2),
        "history": history,
    }
    return ttns, parent_model, summary


def _eval_pair_marginal_on_grid(
    ttns,
    parent: Sequence[int],
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

    def one_eval(point: jnp.ndarray) -> jnp.ndarray:
        vi = basis_call(basis_i, point[0])
        vj = basis_call(basis_j, point[1])
        vectors = base_vectors.at[dim_i].set(vi).at[dim_j].set(vj)
        return eval_q_ttns(ttns, vectors, parent)

    vals = jax.vmap(one_eval)(pts)
    return np.asarray(vals).reshape((len(xi_centers), len(xj_centers)))


def _empirical_pair_density(
    samples: np.ndarray,
    dim_i: int,
    dim_j: int,
    xi_edges: np.ndarray,
    xj_edges: np.ndarray,
) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        samples[:, dim_i],
        samples[:, dim_j],
        bins=[xi_edges, xj_edges],
        density=True,
    )
    return hist


def _iae_2d(d_ref: np.ndarray, d_model: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(np.abs(d_ref - d_model)) * dx * dy)


def _color(value: float, vmax: float) -> str:
    if vmax <= 0:
        return "#f4f4f4"
    t = max(0.0, min(1.0, value / vmax))
    # white -> blue
    r = int(round(255 * (1.0 - 0.95 * t)))
    g = int(round(255 * (1.0 - 0.70 * t)))
    b = int(round(255 * (1.0 - 0.20 * t)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_heatmap(
    lines: List[str],
    x0: float,
    y0: float,
    w: float,
    h: float,
    data: np.ndarray,
    vmax: float,
):
    nx, ny = data.shape
    cell_w = w / ny
    cell_h = h / nx
    for i in range(nx):
        for j in range(ny):
            val = float(data[i, j])
            if val < 0.0:
                val = 0.0
            x = x0 + j * cell_w
            y = y0 + (nx - 1 - i) * cell_h
            lines.append(
                f"<rect x='{x:.3f}' y='{y:.3f}' width='{cell_w + 0.25:.3f}' height='{cell_h + 0.25:.3f}' "
                f"fill='{_color(val, vmax)}' stroke='none'/>"
            )
    lines.append(
        f"<rect x='{x0:.3f}' y='{y0:.3f}' width='{w:.3f}' height='{h:.3f}' "
        f"fill='none' stroke='#222' stroke-width='1'/>"
    )


def _write_slice_svg(rows: List[Dict], out_svg_path: Path, title: str):
    cols = ["target empirical", "model balanced", "model chain"]
    panel_w = 190
    panel_h = 190
    left = 210
    top = 80
    col_gap = 36
    row_gap = 86

    n_rows = len(rows)
    width = int(left + len(cols) * panel_w + (len(cols) - 1) * col_gap + 40)
    height = int(top + n_rows * panel_h + (n_rows - 1) * row_gap + 80)

    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append(f"<text x='24' y='34' font-family='monospace' font-size='24' fill='#111'>{title}</text>")
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='13' fill='#444'>"
        "Heatmap is 2D marginal density; darker color means higher density."
        "</text>"
    )

    for c, col_name in enumerate(cols):
        x = left + c * (panel_w + col_gap) + panel_w / 2
        lines.append(
            f"<text x='{x:.1f}' y='{top - 20}' text-anchor='middle' "
            f"font-family='monospace' font-size='14' fill='#111'>{col_name}</text>"
        )

    for r, row in enumerate(rows):
        y = top + r * (panel_h + row_gap)
        label = row["label"]
        lines.append(
            f"<text x='24' y='{y + 24:.1f}' font-family='monospace' font-size='13' fill='#111'>{label}</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 44:.1f}' font-family='monospace' font-size='12' fill='#444'>"
            f"IAE balanced={row['iae_balanced']:.4f}, IAE chain={row['iae_chain']:.4f}, "
            f"winner={row['winner']}</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 62:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"pair range: x{row['dim_i']} in [{row['xi_min']:.2f}, {row['xi_max']:.2f}], "
            f"x{row['dim_j']} in [{row['xj_min']:.2f}, {row['xj_max']:.2f}]</text>"
        )

        row_vmax = max(
            float(np.max(row["target_density"])),
            float(np.max(np.maximum(row["balanced_density"], 0.0))),
            float(np.max(np.maximum(row["chain_density"], 0.0))),
            1e-12,
        )

        panels = [
            row["target_density"],
            row["balanced_density"],
            row["chain_density"],
        ]
        for c, panel in enumerate(panels):
            x = left + c * (panel_w + col_gap)
            _draw_heatmap(lines, x, y, panel_w, panel_h, panel, row_vmax)
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
    out_svg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_md(
    out_md_path: Path,
    target_kind: str,
    cfg: ExperimentConfig,
    model_summaries: List[Dict],
    rows: List[Dict],
    svg_name: str,
):
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# 2D Slice Report ({target_kind})")
    lines.append("")
    lines.append(f"- Config: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- Figure: `{svg_name}`")
    lines.append("")
    lines.append("## Model Training Summary")
    lines.append("")
    lines.append("| target_topology | model_topology | lr_schedule | final_lr | final_val_l2 | final_integral | total_time_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in model_summaries:
        lines.append(
            f"| {row['target_topology']} | {row['model_topology']} | {row['lr_schedule']} | "
            f"{row['final_lr']:.3e} | {row['final_val_l2']:.6f} | "
            f"{row['final_integral']:.6f} | {row['total_time_sec']:.3f} |"
        )
    lines.append("")
    lines.append("## Slice Error Summary")
    lines.append("")
    lines.append("| target_topology | pair | IAE balanced | IAE chain | winner |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        lines.append(
            f"| {row['target_topology']} | {pair} | {row['iae_balanced']:.6f} | {row['iae_chain']:.6f} | "
            f"{row['winner']} |"
        )
    out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_slice_visualization(
    target_kind: str = "complex",
    grid_bins: int = 42,
    n_slice_samples: int = 120000,
):
    cfg = _config_for_target_kind(target_kind)
    lr_policy_template = _build_lr_policy(target_kind, cfg)
    print(f"running 2D slice visualization for target_kind={target_kind}", flush=True)
    print("config:", asdict(cfg), flush=True)
    print("lr_schedule:", lr_policy_template["label"], flush=True)

    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    seed_cursor = 0
    rows: List[Dict] = []
    model_summaries: List[Dict] = []

    for target_topology in ("balanced", "chain"):
        parent_target = make_parent(cfg.n_dims, target_topology)
        key = jax.random.PRNGKey(cfg.seed + 3000 + seed_cursor)
        seed_cursor += 100
        k_train, k_val, k_slice = jax.random.split(key, 3)

        train_x = _sample_target(target_kind, k_train, cfg.n_train, parent_target)
        val_x = _sample_target(target_kind, k_val, cfg.n_val, parent_target)
        slice_x = _sample_target(target_kind, k_slice, n_slice_samples, parent_target)

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
                lr_policy_template=lr_policy_template,
            )
            seed_cursor += 1
            trained[model_topology] = (model, model_parent)
            model_summaries.append(summary)

        slice_np = np.asarray(slice_x)
        for dim_i, dim_j in pairs:
            xi_lo = np.quantile(slice_np[:, dim_i], 0.01)
            xi_hi = np.quantile(slice_np[:, dim_i], 0.99)
            xj_lo = np.quantile(slice_np[:, dim_j], 0.01)
            xj_hi = np.quantile(slice_np[:, dim_j], 0.99)
            xi_pad = 0.05 * (xi_hi - xi_lo + 1e-12)
            xj_pad = 0.05 * (xj_hi - xj_lo + 1e-12)
            xi_edges = np.linspace(xi_lo - xi_pad, xi_hi + xi_pad, grid_bins + 1)
            xj_edges = np.linspace(xj_lo - xj_pad, xj_hi + xj_pad, grid_bins + 1)
            xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
            xj_centers = 0.5 * (xj_edges[:-1] + xj_edges[1:])

            target_density = _empirical_pair_density(slice_np, dim_i, dim_j, xi_edges, xj_edges)
            model_bal, parent_bal = trained["balanced"]
            model_chn, parent_chn = trained["chain"]

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
            iae_balanced = _iae_2d(target_density, balanced_density, dx, dy)
            iae_chain = _iae_2d(target_density, chain_density, dx, dy)
            winner = "balanced" if iae_balanced < iae_chain else "chain"

            rows.append(
                {
                    "target_topology": target_topology,
                    "dim_i": dim_i,
                    "dim_j": dim_j,
                    "xi_min": float(xi_edges[0]),
                    "xi_max": float(xi_edges[-1]),
                    "xj_min": float(xj_edges[0]),
                    "xj_max": float(xj_edges[-1]),
                    "target_density": target_density,
                    "balanced_density": balanced_density,
                    "chain_density": chain_density,
                    "iae_balanced": iae_balanced,
                    "iae_chain": iae_chain,
                    "winner": winner,
                    "label": f"target={target_topology}, pair=(x{dim_i},x{dim_j})",
                }
            )
            print(
                f"slice target={target_topology}, pair=({dim_i},{dim_j}), "
                f"IAE balanced={iae_balanced:.6f}, IAE chain={iae_chain:.6f}, winner={winner}",
                flush=True,
            )

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_svg_path = report_dir / f"topology_slice_{target_kind}.svg"
    out_md_path = report_dir / f"topology_slice_{target_kind}_report.md"
    out_json_path = report_dir / f"topology_slice_{target_kind}_metrics.json"

    _write_slice_svg(
        rows=rows,
        out_svg_path=out_svg_path,
        title=f"2D Slice Fit Comparison ({target_kind})",
    )
    _write_report_md(
        out_md_path=out_md_path,
        target_kind=target_kind,
        cfg=cfg,
        model_summaries=model_summaries,
        rows=rows,
        svg_name=out_svg_path.name,
    )

    json_rows = []
    for row in rows:
        json_rows.append(
            {
                "target_topology": row["target_topology"],
                "dim_i": row["dim_i"],
                "dim_j": row["dim_j"],
                "xi_min": row["xi_min"],
                "xi_max": row["xi_max"],
                "xj_min": row["xj_min"],
                "xj_max": row["xj_max"],
                "iae_balanced": row["iae_balanced"],
                "iae_chain": row["iae_chain"],
                "winner": row["winner"],
            }
        )
    out_json_path.write_text(
        json.dumps(
            {
                "target_kind": target_kind,
                "config": asdict(cfg),
                "model_summaries": model_summaries,
                "slice_metrics": json_rows,
                "generated_utc_epoch_sec": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_model_time = sum(item["total_time_sec"] for item in model_summaries)
    print("\n=== done ===", flush=True)
    print("saved svg   :", out_svg_path, flush=True)
    print("saved report:", out_md_path, flush=True)
    print("saved metric:", out_json_path, flush=True)
    print("sum model train time:", _format_seconds(total_model_time), flush=True)


if __name__ == "__main__":
    target_kind = "complex"
    if len(sys.argv) > 1:
        target_kind = sys.argv[1].strip().lower()
    run_slice_visualization(target_kind=target_kind)
