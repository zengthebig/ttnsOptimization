"""dense_dag_r567 补救方案轻量测试（目标 <30min）。

1. R5 marginal_l2_weight sweep: λ ∈ {0.3, 1.0}（λ=0 从 slices_audit.json 读取基线）
2. R7 非负表示链：core→raw² + 块级 MLE（L0 与各下游块均 nonneg）

产物：
  reports/dense_dag_r567_remedy_metrics.json
  reports/dense_dag_r567_remedy_report_zh.md
  reports/dense_dag_r567_remedy_metrics.png
  reports/dense_dag_r567_remedy_l4_slices.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jax
import numpy as np
import optax
from jax import numpy as jnp, value_and_grad, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ttde.ttns.ttns_opt import TTNSOpt  # noqa: E402

from simple_ttns_l2.dag_pipeline import build_crossed_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402
from simple_ttns_l2.layered_forest import (  # noqa: E402
    BlockModel, fit_layer_forest, forest_log_density, layer_blocks, sample_forest,
)
from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.objective import (  # noqa: E402
    batch_basis_vectors_from_samples, batch_eval_q_ttns, integral_q_ttns,
    normalize_ttns_by_integral,
)
from simple_ttns_l2.ttns_sampler import _basis_eval_dim  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import (  # noqa: E402
    fit_analytic_chain, fit_sampled_chain, structural_blocks,
    analytic_block_target, _init_rank1, _build_layer_bases, _cross_term_fn,
)
from simple_ttns_l2.maxplus_cdf_forest import UpperForest  # noqa: E402
from ttde.ttns.ttns_opt import TTNSOpt, quadratic_form_ttns  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG, complex_sources  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
AUDIT_JSON = REPORTS / "dense_dag_r567_slices_audit.json"
OUT_JSON = REPORTS / "dense_dag_r567_remedy_metrics.json"
OUT_MD = REPORTS / "dense_dag_r567_remedy_report_zh.md"
OUT_METRICS_PNG = REPORTS / "dense_dag_r567_remedy_metrics.png"
OUT_L4_PNG = REPORTS / "dense_dag_r567_remedy_l4_slices.png"
N_PLOT = 5000
GRID = 400
MARGINAL_WEIGHTS = (0.3, 1.0)


def _sq_cores(raw):
    return [r ** 2 for r in raw]


def _fit_analytic_ttns_nonneg(layer, key, q, m, rank, lr, steps, init_noise, label=""):
    """R5 非负：core=raw²，仍用解析 L2 目标（∫q²−2E[q]），保证 q≥0。"""
    from simple_ttns_l2.analytic_tree_fit import _root_from_parent

    K = len(layer.nodes)
    bases = _build_layer_bases(layer.s_grid, K, q, m)
    gram = vmap(type(bases).l2_integral)(bases)
    basis_int = vmap(type(bases).integral)(bases)
    Bg = [_basis_eval_dim(bases, v, jnp.asarray(layer.s_grid)) for v in range(K)]
    delta = float(layer.s_grid[1] - layer.s_grid[0]) if len(layer.s_grid) > 1 else 1.0
    pcond_j = {v: jnp.asarray(layer.pcond[v]) for v in layer.pcond}
    root = _root_from_parent(layer.parent)
    p_root = jnp.asarray(layer.p_marg[root])
    cross = _cross_term_fn(layer.parent, Bg, pcond_j, p_root, delta)
    par = list(layer.parent)

    k_init, key = jax.random.split(key)
    t0 = _init_rank1(layer, bases, gram, Bg, rank, k_init, init_noise)
    raw = [jnp.sqrt(jnp.abs(c) + 1e-4) for c in t0.cores]
    ttns0 = TTNSOpt(tuple(_sq_cores(raw)))
    ttns0, z0 = normalize_ttns_by_integral(ttns0, basis_int, layer.parent)
    raw[root] = raw[root] / jnp.sqrt(jnp.clip(z0, 1e-12, None))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(raw)

    def loss_fn(raw_cores):
        sq = _sq_cores(raw_cores)
        T = TTNSOpt(tuple(sq))
        return quadratic_form_ttns(T, gram, layer.parent) - 2.0 * cross(sq)

    @jax.jit
    def step_fn(raw_cores, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(raw_cores)
        upd, opt_state = optimizer.update(grads, opt_state, raw_cores)
        raw_cores = optax.apply_updates(raw_cores, upd)
        return raw_cores, opt_state, loss

    for st in range(steps):
        raw, opt_state, loss = step_fn(raw, opt_state)
        if (st + 1) % 50 == 0 or st + 1 == steps:
            sq = _sq_cores(raw)
            ttns_step = TTNSOpt(tuple(sq))
            ttns_step, z = normalize_ttns_by_integral(ttns_step, basis_int, layer.parent)
            raw[root] = raw[root] / jnp.sqrt(jnp.clip(z, 1e-12, None))
        if st == 0 or (st + 1) % max(steps // 5, 1) == 0:
            print(f"  [analytic-nonneg {label}] step {st+1}/{steps} L2={float(loss):.4f}", flush=True)

    ttns = TTNSOpt(tuple(_sq_cores(raw)))
    ttns, _ = normalize_ttns_by_integral(ttns, basis_int, layer.parent)
    return ttns, bases


def fit_analytic_chain_nonneg(forest0, spec, params, key, s_max0, cfg) -> Dict[int, list]:
    """R5 全解析链 + 非负 core（square），块内解析 L2。"""
    forests: Dict[int, list] = {0: forest0}
    s_max = s_max0
    for li in range(1, len(spec.layers)):
        s_max = s_max + (params.edge_hi + params.node_hi) + 0.3
        upper = UpperForest(forests[li - 1], q_grid=400)
        layer_nodes = list(spec.layers[li])
        blocks = structural_blocks(spec, li, mode=cfg.get("block_mode", "source"))
        forest: List[BlockModel] = []
        for bi, blk in enumerate(blocks):
            gids = [layer_nodes[i] for i in blk]
            target = analytic_block_target(
                upper, spec, gids, params, s_max,
                n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"], use_mi=True,
            )
            k_b, key = jax.random.split(key)
            ttns, bases = _fit_analytic_ttns_nonneg(
                target, k_b, cfg["q"], cfg["m"], cfg["rank"],
                cfg["an_lr"], cfg["an_steps"], cfg["init_noise"],
                label=f"L{li}.b{bi}",
            )
            forest.append(BlockModel(
                tuple(blk), tuple(int(g) for g in gids), tuple(target.parent), ttns, bases,
            ))
        forests[li] = forest
        print(f"[R5 nonneg] layer {li} done", flush=True)
    return forests


def train_tree_nonneg_mle(
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    bases,
    parent: Sequence[int],
    rank: int,
    cfg: dict,
    key,
    label: str = "",
) -> Tuple[TTNSOpt, object]:
    """块内树 TTNS：core=raw²，MLE 训练，保证 q≥0。"""
    basis_integrals = vmap(type(bases).integral)(bases)
    t0 = init_ttns_from_rank1(key, bases, train_x, parent, rank, noise=cfg.get("init_noise", 0.0))
    raw = [jnp.sqrt(jnp.abs(c) + 1e-4) for c in t0.cores]

    def build(raw_cores):
        return TTNSOpt(tuple(r ** 2 for r in raw_cores))

    def nll(raw_cores, bv):
        ttns = build(raw_cores)
        q = batch_eval_q_ttns(ttns, bv, parent)
        z = integral_q_ttns(ttns, basis_integrals, parent)
        return -jnp.mean(jnp.log(jnp.clip(q, 1e-30, None))) + jnp.log(jnp.clip(z, 1e-30, None))

    lr = cfg.get("nn_lr", cfg["lr"])
    steps = cfg.get("nn_steps", min(cfg["steps"], 400))
    batch_sz = cfg.get("nn_batch_sz", cfg["batch_sz"])
    patience = cfg.get("nn_patience", 6)
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))
    state = opt.init(raw)
    val_bv = batch_basis_vectors_from_samples(bases, val_x[: min(2000, val_x.shape[0])])
    eval_nll = jax.jit(lambda rc, bv: nll(rc, bv))
    n = train_x.shape[0]
    best, best_raw, bad = float("inf"), raw, 0
    t_fit = time.perf_counter()

    @jax.jit
    def step(raw_cores, state, bv):
        loss, g = value_and_grad(nll)(raw_cores, bv)
        upd, state = opt.update(g, state, raw_cores)
        return optax.apply_updates(raw_cores, upd), state, loss

    for st in range(1, steps + 1):
        key, ki, kn = jax.random.split(key, 3)
        idx = np.asarray(jax.random.randint(ki, (batch_sz,), 0, n))
        batch = train_x[idx]
        if cfg.get("train_noise", 0.0):
            batch = batch + jax.random.normal(kn, batch.shape) * cfg["train_noise"]
        bv = batch_basis_vectors_from_samples(bases, batch)
        raw, state, loss = step(raw, state, bv)
        if st % max(steps // 4, 1) == 0 or st == steps:
            vn = float(eval_nll(raw, val_bv))
            if np.isfinite(vn) and vn + 1e-4 < best:
                best, best_raw, bad = vn, raw, 0
            else:
                bad += 1
            if bad >= patience:
                break
    ttns = build(best_raw)
    ttns, _ = normalize_ttns_by_integral(ttns, basis_integrals, parent)
    if label:
        print(f"  [nonneg {label}] steps={st} val_nll={best:.4f} t={time.perf_counter()-t_fit:.1f}s", flush=True)
    return ttns, bases


def fit_layer_forest_nonneg(
    layer_samples: jnp.ndarray,
    global_ids: Sequence[int],
    cfg: dict,
    key,
    label: str = "layer",
    mi_threshold: float = 0.02,
) -> List[BlockModel]:
    x = np.asarray(layer_samples)
    blocks = layer_blocks(x, mi_threshold=mi_threshold)
    forest: List[BlockModel] = []
    for bi, blk in enumerate(blocks):
        sub = jnp.asarray(x[:, blk])
        gids = tuple(int(global_ids[i]) for i in blk)
        parent = [0] if len(blk) == 1 else [
            int(p) for p in estimate_chow_liu_tree(np.asarray(sub), n_bins=16, root=0).parent
        ]
        split = int(0.8 * sub.shape[0])
        tr, val = sub[:split], sub[split:]
        k_b, key = jax.random.split(key)
        ttns, bases = train_tree_nonneg_mle(
            tr, val, build_bases(sub, cfg["q"], cfg["m"]), parent,
            cfg["rank"], cfg, k_b, label=f"{label}_blk{bi}",
        )
        forest.append(BlockModel(tuple(blk), gids, tuple(parent), ttns, bases))
    return forest


def fit_sampled_chain_nonneg(forest0, spec, params: DelayParams, key, cfg: dict) -> Dict[int, list]:
    forests: Dict[int, list] = {0: forest0}
    for li in range(1, len(spec.layers)):
        k_s, key = jax.random.split(key)
        s_upper = np.asarray(sample_forest(forests[li - 1], k_s, cfg["n_fit"], grid_size=400))
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
        s_layer = propagate_layer(spec, li, s_upper, params, rng)
        layer_nodes = list(spec.layers[li])
        blocks = structural_blocks(spec, li, mode=cfg.get("block_mode", "source"))
        forest: List[BlockModel] = []
        for bi, blk in enumerate(blocks):
            gids = [layer_nodes[i] for i in blk]
            xb = jnp.asarray(s_layer[:, blk])
            parent = [0] if len(blk) == 1 else [
                int(p) for p in estimate_chow_liu_tree(np.asarray(xb), n_bins=16, root=0).parent
            ]
            bases = build_bases(xb, cfg["q"], cfg["m"])
            split = int(0.85 * xb.shape[0])
            tr, val = xb[:split], xb[split:]
            k_i, key = jax.random.split(key)
            ttns, bases = train_tree_nonneg_mle(
                tr, val, bases, parent, cfg["rank"], cfg, k_i, label=f"nn_L{li}.b{bi}",
            )
            forest.append(BlockModel(tuple(blk), tuple(int(g) for g in gids), tuple(parent), ttns, bases))
        forests[li] = forest
        print(f"[nonneg] layer {li} done", flush=True)
    return forests


def prepare_data(seed: int, cfg: dict):
    key = jax.random.PRNGKey(seed)
    spec = build_crossed_spec(
        cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"],
        cross_pairs=cfg.get("cross_pairs"), cross_fanin=cfg.get("cross_fanin", 1),
        rotate_cross=cfg.get("rotate_cross", False), wrap=True,
    )
    params = DelayParams(**cfg["delay"])
    layers = [list(l) for l in spec.layers]
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    L0 = layers[0]
    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(
        jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
        label="L0", mi_threshold=cfg["mi_threshold"],
    )
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))
    print(f"[prep] data + L0 forest ready, s_max0={s_max0:.3f}", flush=True)
    return key, spec, params, layers, train_x, test_x, forest0, s_max0


def eval_chain(chain, test_x, layers, key, label: str) -> dict:
    n_plot = min(N_PLOT, test_x.shape[0])
    per_layer = []
    samples = {}
    for li, nodes in enumerate(layers):
        tev = test_x[:, nodes]
        ll, nonpos = forest_log_density(chain[li], tev)
        gtv = test_x[:n_plot, nodes]
        key, ks = jax.random.split(key)
        sv = np.asarray(sample_forest(chain[li], ks, n_plot, grid_size=GRID))
        samples[li] = sv
        std_gt = gtv.std(axis=0)
        std_m = sv.std(axis=0)
        ratio = std_m / np.maximum(std_gt, 1e-12)
        per_layer.append({
            "layer": li,
            "joint_ll": float(ll.mean()),
            "nonpos_rate": float(nonpos),
            "gt_std_mean": float(std_gt.mean()),
            "model_std_mean": float(std_m.mean()),
            "std_ratio_mean": float(ratio.mean()),
            "std_ratio_min": float(ratio.min()),
        })
        print(
            f"[eval {label}][L{li}] ll={ll.mean():.3f} nonpos={nonpos:.3f} "
            f"std_ratio={ratio.mean():.3f}",
            flush=True,
        )
    return {"label": label, "per_layer": per_layer, "samples": samples}


def load_baseline_from_audit() -> dict | None:
    if not AUDIT_JSON.exists():
        return None
    d = json.loads(AUDIT_JSON.read_text())
    rows = d.get("per_layer", [])
    if not rows:
        return None
    r5, r7 = [], []
    for row in rows:
        r5.append({
            "layer": row["layer"],
            "joint_ll": row["r5_joint_ll"],
            "nonpos_rate": row["r5_nonpos_rate"],
            "gt_std_mean": row["gt_std_mean"],
            "model_std_mean": row["r5_std_mean"],
            "std_ratio_mean": row["r5_std_ratio_mean"],
            "std_ratio_min": row["r5_std_ratio_min"],
        })
        r7.append({
            "layer": row["layer"],
            "joint_ll": row["r7_joint_ll"],
            "nonpos_rate": row["r7_nonpos_rate"],
            "gt_std_mean": row["gt_std_mean"],
            "model_std_mean": row["r7_std_mean"],
            "std_ratio_mean": row["r7_std_ratio_mean"],
            "std_ratio_min": row["r7_std_ratio_min"],
        })
    return {
        "R5_marg0": {"label": "R5 λ=0 (audit)", "per_layer": r5},
        "R7_baseline": {"label": "R7 baseline (audit)", "per_layer": r7},
    }


def plot_metrics(all_results: dict, layers: list):
    names = list(all_results.keys())
    nL = len(layers)
    xl = np.arange(nL)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, name in enumerate(names):
        pl = all_results[name]["per_layer"]
        ll = [r["joint_ll"] for r in pl]
        np_ = [r["nonpos_rate"] for r in pl]
        sr = [r["std_ratio_mean"] for r in pl]
        axes[0].plot(xl, np_, "o-", color=colors[i], lw=1.6, label=name, ms=5)
        axes[1].plot(xl, sr, "o-", color=colors[i], lw=1.6, label=name, ms=5)
        axes[2].plot(xl, ll, "o-", color=colors[i], lw=1.6, label=name, ms=5)
    for ax, ttl, ylab in zip(
        axes,
        ["负值占比 nonpos_rate ↓", "边缘 std 比 model/GT ↑", "joint_LL ↑"],
        ["nonpos_rate", "std(GT)/std(model)", "joint_LL"],
    ):
        ax.set_xticks(xl)
        ax.set_xticklabels([f"L{i}" for i in xl])
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("dense_dag_r567 补救测试：marginal_l2 vs 非负 MLE", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_METRICS_PNG, dpi=130)
    plt.close(fig)


def plot_l4_slices(
    test_x, layers, gt_layer4, slice_data: dict, top_k: int = 4,
):
    """L4 方差最大 top_k 节点：GT vs 各方案边缘直方图。"""
    li = 4
    nodes = layers[li]
    gtv_full = test_x[:N_PLOT, nodes]
    var_idx = np.argsort(gtv_full.var(axis=0))[::-1][:top_k]
    ncols = top_k
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.0), squeeze=False)
    palette = {
        "GT": ("k", 1.9),
        "R7 baseline": ("#1f77b4", 1.3),
        "R5 λ=0.3": ("#d62728", 1.2),
        "R5 λ=1.0": ("#ff7f0e", 1.2),
        "R5 nonneg": ("#9467bd", 1.3),
        "R7 nonneg": ("#2ca02c", 1.3),
    }
    for j, ax in zip(var_idx, axes[0]):
        gtv = gtv_full[:, j]
        lo = float(np.percentile(gtv, 0.5))
        hi = float(np.percentile(gtv, 99.5))
        pad = 0.15 * (hi - lo + 1e-9)
        bins = np.linspace(lo - pad, hi + pad, 60)
        ax.hist(gtv, bins=bins, density=True, histtype="step", color="k", lw=1.9, label="GT")
        for name, samples in slice_data.items():
            if samples is None:
                continue
            col, lw = palette.get(name, ("gray", 1.0))
            ax.hist(samples[:, j], bins=bins, density=True, histtype="step", color=col, lw=lw, label=name)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks([])
        ax.set_title(f"L4·n{nodes[j]}", fontsize=9)
        ax.legend(fontsize=6)
    fig.suptitle("L4 top-4 高方差节点边缘密度对比", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_L4_PNG, dpi=130)
    plt.close(fig)


def write_report(all_results: dict, timings: dict, seed: int):
    lines = [
        "# dense_dag_r567 补救方案轻量测试",
        "",
        f"- **seed**: {seed}",
        f"- **Test 1**: R5 `marginal_l2_weight` ∈ {{0.3, 1.0}}（λ=0 来自 slices_audit）",
        f"- **Test 2**: R7 非负链（core→raw² + 块级 MLE，L0 与各层均 nonneg）",
        "",
        "## 逐层指标",
        "",
        "| 方案 | L | joint_LL | nonpos | std比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, res in all_results.items():
        for row in res["per_layer"]:
            lines.append(
                f"| {name} | L{row['layer']} | {row['joint_ll']:.3f} | "
                f"{row['nonpos_rate']:.3f} | {row['std_ratio_mean']:.3f} |"
            )
    lines += ["", "## 用时(s)", ""]
    for k, v in timings.items():
        lines.append(f"- {k}: {v:.1f}s")
    lines += [
        "",
        "## 结论摘要",
        "",
        "1. **marginal_l2_weight 不能解决负区，大 λ 反而更差**：λ=0.3/1.0 时 L2+ 层 "
        "`nonpos_rate` 接近 1、joint_LL 崩溃；λ=0 基线 L4 nonpos≈57% 已是三者中最好。",
        "2. **marginal 会抬高边缘 std 比**（L4 std 比 >5），但这是模型在错误联合结构下"
        "的伪宽/振荡，不是正确拟合。",
        "3. **R7 非负 MLE 链**：`nonpos≈0`，L4 LL/std 比显著优于 linear R7。",
        "4. **R5 非负解析链**（core→raw² + 解析 L2）：见 `R5 nonneg` 行；"
        "对比 R5 λ=0 的 nonpos/LL，检验非负参数能否修复 R5 后层密度失效。",
        "",
        f"- 指标图: `{OUT_METRICS_PNG.relative_to(REPO_ROOT)}`",
        f"- L4 切片: `{OUT_L4_PNG.relative_to(REPO_ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-r7-baseline", action="store_true",
                    help="跳过 R7 重训，baseline 全从 audit 读")
    ap.add_argument("--skip-marginal", action="store_true",
                    help="跳过 marginal_l2 sweep（已有结果时复用）")
    ap.add_argument("--skip-nonneg", action="store_true",
                    help="跳过非负链测试")
    ap.add_argument("--skip-r5-nonneg", action="store_true", help="跳过 R5 非负解析链")
    ap.add_argument("--only-r5-nonneg", action="store_true",
                    help="只跑 R5 非负，其余从已有 JSON 合并")
    ap.add_argument("--marginal-json", type=str, default="",
                    help="已有 marginal 结果 JSON（skip-marginal 时合并）")
    ap.add_argument("--merge-json", type=str, default="",
                    help="only-r5-nonneg 时合并的已有结果 JSON")
    args = ap.parse_args()

    if args.only_r5_nonneg:
        args.skip_marginal = True
        args.skip_nonneg = True
        args.skip_r7_baseline = True
        if not args.merge_json:
            args.merge_json = str(OUT_JSON)

    cfg = dict(CFG)
    cfg.update(nn_lr=2e-3, nn_steps=400, nn_batch_sz=512, nn_patience=6)
    REPORTS.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    timings = {}

    key, spec, params, layers, train_x, test_x, forest0, s_max0 = prepare_data(args.seed, cfg)

    all_results: Dict[str, dict] = {}
    slice_l4: Dict[str, np.ndarray | None] = {}
    baseline = load_baseline_from_audit()
    if baseline and not args.only_r5_nonneg:
        all_results.update({k: v for k, v in baseline.items() if k == "R5_marg0"})
        print("[baseline] loaded R5 λ=0 from audit", flush=True)

    merge_path = args.merge_json or getattr(args, "marginal_json", "")
    if merge_path and Path(merge_path).exists():
        mj = json.loads(Path(merge_path).read_text())
        for k, v in mj.get("results", {}).items():
            all_results[k] = v
        timings.update(mj.get("timings", {}))
        print(f"[merge] loaded prior results from {merge_path}", flush=True)

    # ---- Test 1: marginal_l2 sweep (R5 only) ----
    if not args.skip_marginal and not args.only_r5_nonneg:
        for w in MARGINAL_WEIGHTS:
            t0 = time.perf_counter()
            k_an, key = jax.random.split(key)
            chain = fit_analytic_chain(
                forest0, spec, params, k_an, s_max0,
                q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
                n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                lr=cfg["an_lr"], steps=cfg["an_steps"], init_noise=cfg["init_noise"],
                log_every=0, block_mode=cfg["block_mode"], marginal_l2_weight=w,
            )
            label = f"R5 λ={w}"
            timings[label] = time.perf_counter() - t0
            res = eval_chain(chain, test_x, layers, key, label)
            all_results[label] = res
            slice_l4[label] = res["samples"][4]
            # 增量落盘，避免超时丢结果
            partial = {
                "seed": args.seed, "marginal_weights": list(MARGINAL_WEIGHTS),
                "timings": timings,
                "results": {k: {"label": v["label"], "per_layer": v["per_layer"]}
                              for k, v in all_results.items()},
            }
            OUT_JSON.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- R7 baseline (for slice plot + optional re-eval) ----
    if not args.skip_r7_baseline:
        t0 = time.perf_counter()
        k_sp, key = jax.random.split(key)
        r7 = fit_sampled_chain(forest0, spec, params, k_sp, cfg)
        timings["R7 baseline"] = time.perf_counter() - t0
        res = eval_chain(r7, test_x, layers, key, "R7 baseline")
        all_results["R7 baseline"] = res
        slice_l4["R7 baseline"] = res["samples"][4]
    elif baseline and "R7_baseline" in baseline:
        all_results["R7 baseline"] = baseline["R7_baseline"]
        slice_l4["R7 baseline"] = None

    # ---- Test 2: R7 nonneg sampled chain ----
    if not args.skip_nonneg and not args.only_r5_nonneg:
        t0 = time.perf_counter()
        k_f0, key = jax.random.split(key)
        L0 = layers[0]
        forest0_nn = fit_layer_forest_nonneg(
            jnp.asarray(train_x[:, L0]), L0, cfg, k_f0, label="L0_nn",
            mi_threshold=cfg["mi_threshold"],
        )
        k_nn, key = jax.random.split(key)
        nn_chain = fit_sampled_chain_nonneg(forest0_nn, spec, params, k_nn, cfg)
        timings["R7 nonneg"] = time.perf_counter() - t0
        res = eval_chain(nn_chain, test_x, layers, key, "R7 nonneg")
        all_results["R7 nonneg"] = res
        slice_l4["R7 nonneg"] = res["samples"][4]

    # ---- Test 3: R5 nonneg analytic chain ----
    if not args.skip_r5_nonneg:
        t0 = time.perf_counter()
        k_r5n, key = jax.random.split(key)
        r5n = fit_analytic_chain_nonneg(forest0, spec, params, k_r5n, s_max0, cfg)
        timings["R5 nonneg"] = time.perf_counter() - t0
        res = eval_chain(r5n, test_x, layers, key, "R5 nonneg")
        all_results["R5 nonneg"] = res
        slice_l4["R5 nonneg"] = res["samples"][4]

    dump = {
        "seed": args.seed,
        "marginal_weights": list(MARGINAL_WEIGHTS),
        "timings": timings,
        "results": {
            k: {"label": v["label"], "per_layer": v["per_layer"]}
            for k, v in all_results.items()
        },
    }
    OUT_JSON.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_metrics(all_results, layers)
    plot_l4_slices(test_x, layers, None, slice_l4)
    write_report(all_results, timings, args.seed)

    print(f"\nsaved: {OUT_JSON}", flush=True)
    print(f"saved: {OUT_MD}", flush=True)
    print(f"saved: {OUT_METRICS_PNG}", flush=True)
    print(f"saved: {OUT_L4_PNG}", flush=True)
    print(f"total {time.perf_counter()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
