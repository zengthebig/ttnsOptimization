"""三方对比 + 图像：全局 TT(chain) vs 全局 TTNS(chow-liu) vs 分层 TTNS(源层森林 + 已知 max-plus 核)。

三者都拟合同一份多层 DAG 全联合数据。指标：
- 留出集平均**联合对数密度** joint_LL（越高越好）；
- **采样质量**：从各模型采样 → 逐节点边缘 Wasserstein-1（平均，越低越好）、相关矩阵 Frobenius 误差（越低越好）；
- 学习参数量。

图像：
1. `global_vs_layered_overview.png`：joint_LL / 平均 W1 / corr_fro 柱状图 + 各模型相关矩阵热图。
2. `global_vs_layered_marginals.png`：逐节点边缘密度切片（GT vs 三模型）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

from ttde.score.models.opt_for_tree_data import chain_parent  # noqa: E402

from simple_ttns_l2.train_l2 import build_bases  # noqa: E402
from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402
from simple_ttns_l2.layered_forest import (  # noqa: E402
    fit_layer_forest, forest_log_density, maxplus_cond_logdensity, sample_forest,
)
from simple_ttns_l2.ttns_sampler import sample_ttns  # noqa: E402
from simple_ttns_l2.experiments.fit_deep_dag_vs_tree import _count_params  # noqa: E402
from simple_ttns_l2.experiments.fit_layered_vs_flat_tt import fit_flat, flat_joint_loglik  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def bimodal_sources(spec, cfg: dict) -> Dict:
    """双峰(bimodal)源分布：每个源节点 ~ 0.5 N(0.25,σ) + 0.5 N(0.85,σ)，比均匀复杂得多。"""
    sig = cfg.get("src_sigma", 0.06)

    def make():
        def src(rng, n):
            comp = rng.integers(0, 2, size=n)
            mu = np.where(comp == 0, 0.25, 0.85)
            return mu + rng.normal(0.0, sig, size=n)
        return src

    return {node: make() for layer in [spec.layers[0]] for node in layer if not spec.parents(node)}


def sample_layered(forest0, spec, params, key, n: int, seed: int) -> np.ndarray:
    """分层模型联合采样：源层森林采样 + 逐层已知 max-plus 核推进。返回 [n, n_dims]。"""
    n_dims = sum(len(l) for l in spec.layers)
    full = np.zeros((n, n_dims))
    s0 = sample_forest(forest0, key, n, grid_size=400)
    full[:, list(spec.layers[0])] = s0
    rng = np.random.default_rng(seed)
    prev = s0
    for li in range(1, len(spec.layers)):
        cur = propagate_layer(spec, li, prev, params, rng)
        full[:, list(spec.layers[li])] = cur
        prev = cur
    return full


def sample_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict:
    """逐节点边缘 W1（平均）+ 相关矩阵 Frobenius 误差。"""
    d = pred.shape[1]
    w1 = float(np.mean([wasserstein_distance(pred[:, j], gt[:, j]) for j in range(d)]))
    cf = float(np.linalg.norm(np.corrcoef(pred.T) - np.corrcoef(gt.T)))
    return {"w1_marg": w1, "corr_fro": cf}


def run(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)
    if cfg.get("source_mode") == "bimodal":
        sources = {**sources, **bimodal_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n = xs.shape[0]
    n_tr = int(0.7 * n)
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    n_dims = xs.shape[1]

    # ---- 分层模型：源层森林 + 已知 max-plus 核 ----
    k_src, key = jax.random.split(key)
    L0 = list(spec.layers[0])
    forest = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_src, label="L0",
                              mi_threshold=cfg["mi_threshold"])
    blocks_info = [list(bm.global_vars) for bm in forest]
    src_params = sum(_count_params(bm.ttns.cores) for bm in forest)
    ll_src, nonpos_src = forest_log_density(forest, test_x[:, L0])
    ll = ll_src.copy()
    for li in range(1, len(spec.layers)):
        for v in spec.layers[li]:
            ll += maxplus_cond_logdensity(test_x[:, v], test_x[:, list(spec.parents(v))], params)
    layered = {"model": "layered_TTNS", "learned_params": src_params, "blocks_L0": blocks_info,
               "joint_loglik": float(ll.mean()), "nonpos_rate": nonpos_src}

    # ---- 全局扁平：TT(chain) / TTNS(chow-liu) ----
    bases = build_bases(jnp.asarray(train_x), cfg["q"], cfg["m"])
    gram = vmap(type(bases).l2_integral)(bases)
    basis_integrals = vmap(type(bases).integral)(bases)
    tr_j, val_j = jnp.asarray(train_x[:int(0.85 * n_tr)]), jnp.asarray(train_x[int(0.85 * n_tr):])
    flat_specs = {
        "global_TT": [int(p) for p in chain_parent(n_dims)],
        "global_TTNS": [int(p) for p in estimate_chow_liu_tree(train_x, n_bins=16, root=0).parent],
    }
    flat_models = {}
    results = [layered]
    for name, parent in flat_specs.items():
        k_f, key = jax.random.split(key)
        ttns, parent, r, np_ = fit_flat(parent, name, tr_j, val_j, bases, gram, basis_integrals,
                                        cfg["budget"], cfg, k_f)
        fll, fnp = flat_joint_loglik(ttns, parent, bases, test_x)
        flat_models[name] = (ttns, parent)
        results.append({"model": name, "rank": r, "learned_params": np_,
                        "joint_loglik": fll, "nonpos_rate": fnp})

    # ---- 采样质量 ----
    n_s = cfg["n_sample"]
    gt_s = test_x[:n_s]
    k_l, key = jax.random.split(key)
    samp = {"layered_TTNS": sample_layered(forest, spec, params, k_l, n_s, cfg["seed"] + 1)}
    for name, (ttns, parent) in flat_models.items():
        k_m, key = jax.random.split(key)
        samp[name] = np.asarray(sample_ttns(ttns, bases, list(parent), k_m, n_s, grid_size=400))
    for r in results:
        m = sample_metrics(samp[r["model"]], gt_s)
        r.update(m)

    return spec, results, samp, gt_s


def plot_overview(results, samp, gt_s, out: Path):
    order = ["global_TT", "global_TTNS", "layered_TTNS"]
    rmap = {r["model"]: r for r in results}
    colors = {"global_TT": "tab:orange", "global_TTNS": "tab:green", "layered_TTNS": "tab:red"}

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])

    def bar(ax, key, title, better):
        vals = [rmap[m][key] for m in order]
        ax.bar(range(3), vals, color=[colors[m] for m in order])
        ax.set_xticks(range(3)); ax.set_xticklabels(order, rotation=15, fontsize=8)
        ax.set_title(f"{title}\n({better})", fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)

    bar(fig.add_subplot(gs[0, 0]), "joint_loglik", "joint log-likelihood", "higher=better")
    bar(fig.add_subplot(gs[0, 1]), "w1_marg", "marginal Wasserstein-1 (mean)", "lower=better")
    bar(fig.add_subplot(gs[0, 2]), "corr_fro", "correlation Frobenius error", "lower=better")
    axp = fig.add_subplot(gs[0, 3])
    pv = [rmap[m]["learned_params"] for m in order]
    axp.bar(range(3), pv, color=[colors[m] for m in order])
    axp.set_yscale("log"); axp.set_xticks(range(3)); axp.set_xticklabels(order, rotation=15, fontsize=8)
    axp.set_title("learned params (log)\nlower=cheaper", fontsize=10)
    for i, v in enumerate(pv):
        axp.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)

    # 相关热图：GT + 三模型
    mats = [("GT", np.corrcoef(gt_s.T))] + [(m, np.corrcoef(samp[m].T)) for m in order]
    for j, (name, C) in enumerate(mats):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(C, vmin=-0.2, vmax=1.0, cmap="viridis")
        ax.set_title(f"corr: {name}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("global TT vs global TTNS vs layered TTNS", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_marginals(spec, samp, gt_s, out: Path):
    order = ["global_TT", "global_TTNS", "layered_TTNS"]
    colors = {"global_TT": "tab:orange", "global_TTNS": "tab:green", "layered_TTNS": "tab:red"}
    nL = len(spec.layers)
    nN = max(len(l) for l in spec.layers)
    fig, axes = plt.subplots(nL, nN, figsize=(3.0 * nN, 2.4 * nL), squeeze=False)
    for li in range(nL):
        nodes = list(spec.layers[li])
        for j in range(nN):
            ax = axes[li][j]
            if j >= len(nodes):
                ax.axis("off"); continue
            node = nodes[j]
            gv = gt_s[:, node]
            lo, hi = float(np.percentile(gv, 0.5)), float(np.percentile(gv, 99.5))
            allv = [gv] + [samp[m][:, node] for m in order]
            blo = min(float(v.min()) for v in allv); bhi = max(float(v.max()) for v in allv)
            bins = np.linspace(blo, bhi, 70)
            ax.hist(gv, bins=bins, density=True, histtype="step", color="k", lw=2.0, label="GT")
            for m in order:
                ax.hist(samp[m][:, node], bins=bins, density=True, histtype="step",
                        color=colors[m], lw=1.2, label=m)
            ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
            ax.set_title(f"L{li} node{node}", fontsize=9)
            if li == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.suptitle("per-node marginal density: GT vs global TT / global TTNS / layered TTNS", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    cfg = dict(
        layer_sizes=[4, 4, 4], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
        n_total=40000, n_sample=8000, q=2, m=16, rank=8, budget=120000, rmax=40,
        lr=2e-3, steps=1000, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
        log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
    )
    spec, results, samp, gt_s = run(cfg)

    REPORTS.mkdir(parents=True, exist_ok=True)
    plot_overview(results, samp, gt_s, REPORTS / "global_vs_layered_overview.png")
    plot_marginals(spec, samp, gt_s, REPORTS / "global_vs_layered_marginals.png")
    (REPORTS / "global_vs_layered_metrics.json").write_text(
        json.dumps({"config": cfg, "results": results}, indent=2, ensure_ascii=False))

    print("\n==== 全局 TT vs 全局 TTNS vs 分层 TTNS ====")
    print(f"{'model':<16}{'params':>10}{'joint_LL':>12}{'W1_marg':>12}{'corr_fro':>12}")
    for r in results:
        print(f"{r['model']:<16}{r['learned_params']:>10}{r['joint_loglik']:>12.4f}"
              f"{r['w1_marg']:>12.4f}{r['corr_fro']:>12.4f}")
    print("\n图已保存：reports/global_vs_layered_overview.png, reports/global_vs_layered_marginals.png")


if __name__ == "__main__":
    main()
