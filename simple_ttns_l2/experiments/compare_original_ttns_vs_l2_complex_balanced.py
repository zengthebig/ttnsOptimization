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
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from simple_ttns_l2.experiments.topology_comparison import (  # noqa: E402
    ExperimentConfig,
    make_tree_gaussian_params,
    sample_tree_gaussian,
)
from simple_ttns_l2.experiments.topology_comparison_complex import (  # noqa: E402
    sample_complex_tree_distribution,
)
from simple_ttns_l2.experiments.topology_slice_visualization import (  # noqa: E402
    _empirical_pair_density,
    _eval_pair_marginal_on_grid,
    _iae_2d,
    _train_one_model,
)
from simple_ttns_l2.train_l2 import build_bases, make_parent  # noqa: E402
from ttde.dl_routine import batched_vmap  # noqa: E402
from ttde.score.experiment_setups import init_setups, model_setups  # noqa: E402
from ttde.ttns.ttns_opt import TTNSOpt, quadratic_form_ttns  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _config() -> ExperimentConfig:
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
        seed=20260302,
        n_train=5000,
        n_val=2500,
        n_test=5000,
        monitor_train_sz=2000,
        monitor_val_sz=2000,
        early_stop_patience_logs=0,
        early_stop_min_delta=1e-4,
        early_stop_warmup_logs=0,
        early_stop_restore_best=False,
    )


def _constant_lr_policy(cfg: ExperimentConfig) -> Dict:
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


def _safe_float(x) -> float | None:
    val = float(x)
    if np.isfinite(val):
        return val
    return None


def _extract_single_component_ttns(ttns_batched: TTNSOpt, component: int = 0) -> TTNSOpt:
    return TTNSOpt(tuple(core[component] for core in ttns_batched.cores))


def _nll_loss(model, params, xs: jnp.ndarray, batch_sz: int) -> jnp.ndarray:
    def log_p(x):
        return model.apply(params, x, method=model.log_p)

    return -batched_vmap(log_p, batch_sz)(xs).mean()


def _mean_log_p(model, params, xs: jnp.ndarray, batch_sz: int) -> jnp.ndarray:
    def log_p(x):
        return model.apply(params, x, method=model.log_p)

    return batched_vmap(log_p, batch_sz)(xs).mean()


def _log_prob_stats(model, params, xs: jnp.ndarray, batch_sz: int) -> Dict[str, jnp.ndarray]:
    def log_p(x):
        return model.apply(params, x, method=model.log_p)

    log_ps = batched_vmap(log_p, batch_sz)(xs)
    finite_mask = log_ps != -jnp.inf
    finite_sum = jnp.sum(jnp.where(finite_mask, log_ps, 0.0))
    finite_count = finite_mask.sum()
    finite_mean_ll = finite_sum / jnp.maximum(finite_count, 1)
    nonpositive_rate = jnp.clip(1.0 - finite_count / len(log_ps), 0.0, 1.0)
    return {
        "finite_mean_ll": finite_mean_ll,
        "nonpositive_rate": nonpositive_rate,
    }


def _train_original_mle_ttns(
    cfg: ExperimentConfig,
    model_topology: str,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    test_x: jnp.ndarray,
    em_steps: int,
) -> Tuple[object, Dict, Dict]:
    model_setup = model_setups.PAsTTNSSqrOpt(
        q=cfg.q,
        m=cfg.m,
        rank=cfg.rank,
        n_comps=1,
        tree_topology=model_topology,
    )
    init_setup = init_setups.CanonicalRankK(em_steps=em_steps, noise=cfg.init_noise)

    key = jax.random.PRNGKey(cfg.seed + 900)
    key, k_model, k_init, k_iter = jax.random.split(key, 4)
    model = model_setup.create(k_model, train_x)
    params = init_setup(model, k_init, train_x)

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(params)

    train_monitor = train_x[: cfg.monitor_train_sz]
    val_monitor = val_x[: cfg.monitor_val_sz]

    @jax.jit
    def train_step(curr_params, curr_opt_state, noisy_batch):
        loss, grads = jax.value_and_grad(lambda p: _nll_loss(model, p, noisy_batch, cfg.batch_sz))(curr_params)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
        next_params = optax.apply_updates(curr_params, updates)
        return next_params, next_opt_state, loss

    eval_nll = jax.jit(lambda p, xs: _nll_loss(model, p, xs, cfg.batch_sz))
    eval_mean_ll = jax.jit(lambda p, xs: _mean_log_p(model, p, xs, cfg.batch_sz))
    eval_log_int_p = jax.jit(lambda p: model.apply(p, method=model.log_int_p))
    eval_stats = jax.jit(lambda p, xs: _log_prob_stats(model, p, xs, cfg.batch_sz))

    history = []
    t_start = time.perf_counter()
    t_prev = t_start
    print(
        f"\n=== [complex balanced target] [original_ttns_mle:{model_topology}] ===",
        flush=True,
    )
    print("step,train_nll,val_nll,val_ll,log_int_p,interval_sec,total_sec", flush=True)

    for step in range(1, cfg.train_steps + 1):
        k_iter, k_idx, k_noise = jax.random.split(k_iter, 3)
        idx = jax.random.randint(k_idx, (cfg.batch_sz,), 0, train_x.shape[0])
        batch = train_x[idx]
        noisy_batch = batch + jax.random.normal(k_noise, batch.shape) * cfg.train_noise
        params, opt_state, _ = train_step(params, opt_state, noisy_batch)

        if step % cfg.log_every == 0:
            now = time.perf_counter()
            train_nll = float(eval_nll(params, train_monitor))
            val_nll = float(eval_nll(params, val_monitor))
            val_ll = float(eval_mean_ll(params, val_monitor))
            log_int_p = float(eval_log_int_p(params))
            interval = now - t_prev
            total = now - t_start
            t_prev = now
            history.append(
                {
                    "step": step,
                    "train_nll": train_nll,
                    "val_nll": val_nll,
                    "val_ll": val_ll,
                    "log_int_p": log_int_p,
                    "interval_sec": interval,
                    "total_sec": total,
                }
            )
            print(
                f"{step},{train_nll:.6f},{val_nll:.6f},{val_ll:.6f},{log_int_p:.6f},{interval:.3f},{total:.3f}",
                flush=True,
            )

    total_time = time.perf_counter() - t_start
    train_stats = eval_stats(params, train_x)
    val_stats = eval_stats(params, val_x)
    test_stats = eval_stats(params, test_x)
    summary = {
        "model_name": "original_ttns_mle",
        "model_topology": model_topology,
        "native_metric_name": "val_nll",
        "final_train_nll": _safe_float(eval_nll(params, train_x)),
        "final_val_nll": _safe_float(eval_nll(params, val_x)),
        "final_test_nll": _safe_float(eval_nll(params, test_x)),
        "final_train_ll": _safe_float(eval_mean_ll(params, train_x)),
        "final_val_ll": _safe_float(eval_mean_ll(params, val_x)),
        "final_test_ll": _safe_float(eval_mean_ll(params, test_x)),
        "finite_train_ll": float(train_stats["finite_mean_ll"]),
        "finite_val_ll": float(val_stats["finite_mean_ll"]),
        "finite_test_ll": float(test_stats["finite_mean_ll"]),
        "train_nonpositive_rate": float(train_stats["nonpositive_rate"]),
        "val_nonpositive_rate": float(val_stats["nonpositive_rate"]),
        "test_nonpositive_rate": float(test_stats["nonpositive_rate"]),
        "final_log_int_p": float(eval_log_int_p(params)),
        "total_time_sec": total_time,
        "history": history,
    }
    return model, params, summary


def _eval_pair_marginal_sq_ttns_on_grid(
    ttns: TTNSOpt,
    parent: Sequence[int],
    bases,
    gram_matrices: jnp.ndarray,
    dim_i: int,
    dim_j: int,
    xi_centers: np.ndarray,
    xj_centers: np.ndarray,
) -> np.ndarray:
    xi = jnp.asarray(xi_centers, dtype=jnp.float64)
    xj = jnp.asarray(xj_centers, dtype=jnp.float64)
    gx, gy = jnp.meshgrid(xi, xj, indexing="ij")
    pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

    base_mats = jnp.asarray(gram_matrices, dtype=jnp.float64)
    basis_call = type(bases).__call__
    basis_i = jax.tree_util.tree_map(lambda arr: arr[dim_i], bases)
    basis_j = jax.tree_util.tree_map(lambda arr: arr[dim_j], bases)
    z = quadratic_form_ttns(ttns, base_mats, parent)

    def one_eval(point: jnp.ndarray) -> jnp.ndarray:
        vi = basis_call(basis_i, point[0])
        vj = basis_call(basis_j, point[1])
        mats = (
            base_mats
            .at[dim_i].set(jnp.outer(vi, vi))
            .at[dim_j].set(jnp.outer(vj, vj))
        )
        return quadratic_form_ttns(ttns, mats, parent) / z

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
            g = int(round(255 * (1.0 - 0.72 * t)))
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
    panel_w = 176
    panel_h = 176
    left = 260
    top = 92
    col_gap = 20
    row_gap = 90
    cols = ["target empirical", "original TTNS (MLE)", "simple TTNS (L2)"]

    width = int(left + 3 * panel_w + 2 * col_gap + 36)
    height = int(top + len(rows) * panel_h + (len(rows) - 1) * row_gap + 82)
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
    lines.append(
        "<text x='24' y='34' font-family='monospace' font-size='23' fill='#111'>"
        "Complex Balanced Target: Original TTNS vs L2 TTNS"
        "</text>"
    )
    lines.append(
        "<text x='24' y='56' font-family='monospace' font-size='12' fill='#555'>"
        "Same complex balanced-tree target, same topology, different training objective."
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
            f"IAE original={row['iae_original_mle']:.4f}, IAE l2={row['iae_simple_l2']:.4f}"
            "</text>"
        )
        lines.append(
            f"<text x='24' y='{y + 60:.1f}' font-family='monospace' font-size='12' fill='#666'>"
            f"range: x{row['dim_i']} in [{row['xi_min']:.2f}, {row['xi_max']:.2f}], "
            f"x{row['dim_j']} in [{row['xj_min']:.2f}, {row['xj_max']:.2f}]"
            "</text>"
        )

        vmax = max(
            float(np.max(row["target_density"])),
            float(np.max(row["original_density"])),
            float(np.max(np.maximum(row["l2_density"], 0.0))),
            1e-12,
        )
        panels = [row["target_density"], row["original_density"], row["l2_density"]]
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


def _write_report(
    path: Path,
    cfg: ExperimentConfig,
    original_summary: Dict,
    l2_summary: Dict,
    slice_rows: List[Dict],
    svg_name: str,
):
    mean_iae_original = float(np.mean([row["iae_original_mle"] for row in slice_rows]))
    mean_iae_l2 = float(np.mean([row["iae_simple_l2"] for row in slice_rows]))

    lines: List[str] = []
    lines.append("# 原始 TTNSDE 训练方式 vs simple_ttns_l2")
    lines.append("")
    lines.append("## 1. 实验设置")
    lines.append("")
    lines.append("- 目标分布: `6 维 complex balanced target`。")
    lines.append("- 模型拓扑: 两边都用 `balanced TTNS`。")
    lines.append("- 原始模型: `TTNSDE/ttde` 的 `PAsTTNSSqrOpt + MLE(NLL)`。")
    lines.append("- 当前模型: `simple_ttns_l2` 的 `TTNS + L2 objective`。")
    lines.append(f"- 配置: `{json.dumps(asdict(cfg), ensure_ascii=True)}`")
    lines.append(f"- 切片图: `{svg_name}`")
    lines.append("")
    lines.append("## 2. 训练指标")
    lines.append("")
    lines.append(
        "| model | objective | native_metric | val_native | finite_val_ll | finite_test_ll | "
        "val_nonpositive | test_nonpositive | total_time_sec |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| original_ttns | MLE / NLL | val_nll | {original_summary['final_val_nll']:.6f} | "
        f"{original_summary['finite_val_ll']:.6f} | {original_summary['finite_test_ll']:.6f} | "
        f"{original_summary['val_nonpositive_rate']:.4f} | {original_summary['test_nonpositive_rate']:.4f} | "
        f"{original_summary['total_time_sec']:.3f} |"
    )
    lines.append(
        f"| simple_ttns_l2 | L2 surrogate | val_l2 | {l2_summary['final_val_l2']:.6f} | "
        f"nan | nan | nan | nan | {l2_summary['total_time_sec']:.3f} |"
    )
    lines.append("")
    lines.append("说明: 两个 native metric 不同，不能直接横向比较。真正可比的是下面的共同外部指标: 2D marginal IAE。")
    lines.append("")
    lines.append("## 3. 共同外部指标: 2D 切片 IAE")
    lines.append("")
    lines.append("| pair | IAE_original_mle | IAE_simple_l2 | better |")
    lines.append("|---:|---:|---:|---|")
    for row in slice_rows:
        pair = f"({row['dim_i']},{row['dim_j']})"
        better = "original_ttns" if row["iae_original_mle"] < row["iae_simple_l2"] else "simple_ttns_l2"
        lines.append(
            f"| {pair} | {row['iae_original_mle']:.6f} | {row['iae_simple_l2']:.6f} | {better} |"
        )
    lines.append("")
    lines.append("## 4. 聚合结论")
    lines.append("")
    lines.append(f"- mean_IAE(original_ttns) = `{mean_iae_original:.6f}`")
    lines.append(f"- mean_IAE(simple_ttns_l2) = `{mean_iae_l2:.6f}`")
    if mean_iae_original < mean_iae_l2:
        lines.append("- 这次在共同切片指标上，`original TTNS(MLE)` 更好。")
    else:
        lines.append("- 这次在共同切片指标上，`simple TTNS(L2)` 更好。")
    lines.append(
        f"- 时间对比: original=`{original_summary['total_time_sec']:.3f}s`, "
        f"simple_l2=`{l2_summary['total_time_sec']:.3f}s`。"
    )
    lines.append("")
    lines.append("## 5. 备注")
    lines.append("")
    lines.append("- 原始模型是平方密度参数化，因此切片是通过二次型 marginal 计算出来的。")
    lines.append("- 当前 simple 模型是直接线性密度参数化，因此切片是线性 marginal。")
    lines.append("- 两者不是同一参数化族，所以这次实验比较的是“原始训练方式 vs 当前模型设定”的整体效果。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    cfg = _config()
    target_topology = "balanced"
    model_topology = "balanced"
    pairs: List[Tuple[int, int]] = [(0, 1), (0, 3), (2, 5)]
    grid_bins = 56
    n_slice_samples = 220000
    em_steps = 10

    print("running complex balanced comparison: original TTNS(MLE) vs simple TTNS(L2)", flush=True)
    print("config:", asdict(cfg), flush=True)

    parent_target = make_parent(cfg.n_dims, target_topology)
    key = jax.random.PRNGKey(cfg.seed)
    k_train, k_val, k_test, k_slice = jax.random.split(key, 4)

    train_x = sample_complex_tree_distribution(k_train, cfg.n_train, parent_target)
    val_x = sample_complex_tree_distribution(k_val, cfg.n_val, parent_target)
    test_x = sample_complex_tree_distribution(k_test, cfg.n_test, parent_target)
    slice_x = sample_complex_tree_distribution(k_slice, n_slice_samples, parent_target)

    bases = build_bases(train_x, q=cfg.q, m=cfg.m)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    basis_integrals = jax.vmap(type(bases).integral)(bases)

    original_model, original_params, original_summary = _train_original_mle_ttns(
        cfg=cfg,
        model_topology=model_topology,
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        em_steps=em_steps,
    )

    l2_ttns, l2_parent, l2_summary = _train_one_model(
        cfg=cfg,
        target_topology=target_topology,
        model_topology=model_topology,
        train_x=train_x,
        val_x=val_x,
        bases=bases,
        gram_matrices=gram_matrices,
        basis_integrals=basis_integrals,
        seed_offset=1,
        lr_policy_template=_constant_lr_policy(cfg),
    )
    l2_summary["model_name"] = "simple_ttns_l2"

    original_ttns = _extract_single_component_ttns(original_params["ttns"]["ttns"], component=0)
    original_parent = original_model.tree_parent.tolist()
    original_bases = original_model.bases
    original_gram_matrices = jax.vmap(type(original_bases).l2_integral)(original_bases)

    rows: List[Dict] = []
    for dim_i, dim_j in pairs:
        xi_min, xi_max = np.quantile(np.asarray(slice_x[:, dim_i]), [0.01, 0.99])
        xj_min, xj_max = np.quantile(np.asarray(slice_x[:, dim_j]), [0.01, 0.99])
        pad_i = 0.08 * max(1e-6, xi_max - xi_min)
        pad_j = 0.08 * max(1e-6, xj_max - xj_min)
        xi_edges = np.linspace(float(xi_min - pad_i), float(xi_max + pad_i), grid_bins + 1)
        xj_edges = np.linspace(float(xj_min - pad_j), float(xj_max + pad_j), grid_bins + 1)
        xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        xj_centers = 0.5 * (xj_edges[:-1] + xj_edges[1:])
        dx = float(xi_edges[1] - xi_edges[0])
        dy = float(xj_edges[1] - xj_edges[0])

        target_density = _empirical_pair_density(np.asarray(slice_x), dim_i, dim_j, xi_edges, xj_edges)
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
        l2_density = _eval_pair_marginal_on_grid(
            l2_ttns,
            l2_parent,
            bases,
            basis_integrals,
            dim_i,
            dim_j,
            xi_centers,
            xj_centers,
        )

        rows.append(
            {
                "dim_i": dim_i,
                "dim_j": dim_j,
                "xi_min": float(xi_edges[0]),
                "xi_max": float(xi_edges[-1]),
                "xj_min": float(xj_edges[0]),
                "xj_max": float(xj_edges[-1]),
                "target_density": target_density,
                "original_density": original_density,
                "l2_density": l2_density,
                "iae_original_mle": _iae_2d(target_density, original_density, dx, dy),
                "iae_simple_l2": _iae_2d(target_density, l2_density, dx, dy),
            }
        )

    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "compare_original_ttns_vs_l2_complex_balanced_metrics.json"
    svg_path = report_dir / "compare_original_ttns_vs_l2_complex_balanced.svg"
    report_path = report_dir / "compare_original_ttns_vs_l2_complex_balanced_report_zh.md"

    _write_svg(rows, svg_path)
    _write_report(report_path, cfg, original_summary, l2_summary, rows, svg_path.name)

    metrics = {
        "config": asdict(cfg),
        "target_topology": target_topology,
        "model_topology": model_topology,
        "em_steps": em_steps,
        "original_ttns_mle": original_summary,
        "simple_ttns_l2": l2_summary,
        "slice_metrics": [
            {
                k: v
                for k, v in row.items()
                if k not in {"target_density", "original_density", "l2_density"}
            }
            for row in rows
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== summary ===", flush=True)
    print(
        f"original_ttns_mle: val_nll={original_summary['final_val_nll']:.6f}, "
        f"finite_test_ll={original_summary['finite_test_ll']:.6f}, "
        f"test_nonpositive={original_summary['test_nonpositive_rate']:.4f}, "
        f"time={original_summary['total_time_sec']:.3f}s",
        flush=True,
    )
    print(
        f"simple_ttns_l2   : val_l2={l2_summary['final_val_l2']:.6f}, "
        f"time={l2_summary['total_time_sec']:.3f}s",
        flush=True,
    )
    for row in rows:
        print(
            f"pair=({row['dim_i']},{row['dim_j']}): "
            f"IAE original={row['iae_original_mle']:.6f}, IAE l2={row['iae_simple_l2']:.6f}",
            flush=True,
        )
    print("saved metrics:", metrics_path, flush=True)
    print("saved svg    :", svg_path, flush=True)
    print("saved report :", report_path, flush=True)


if __name__ == "__main__":
    run()
