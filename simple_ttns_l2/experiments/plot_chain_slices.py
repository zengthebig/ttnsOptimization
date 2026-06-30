"""画 A 链 / B 链每层每节点的边缘密度切片（GT vs A vs B），直观看深层 B 在哪崩。

复用 fit_layered_forest_chain_schemes 的链逻辑，但把每层目标样本收集出来作图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, sample_forest  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import UpperForest, sample_layer_copula  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def collect(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)

    k_gt, key = jax.random.split(key)
    gt = np.asarray(sample_joint(spec, sources, kernels, k_gt, cfg["n"], clip=(-1e9, 1e9)))
    gt_layers = [gt[:, list(spec.layers[li])] for li in range(len(spec.layers))]
    n_eval = cfg["n_eval"]

    k_src, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(gt_layers[0]), list(spec.layers[0]), cfg, k_src,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    k_s0, key = jax.random.split(key)
    samp0 = sample_forest(forest0, k_s0, n_eval, grid_size=cfg["grid_size"])

    rng = np.random.default_rng(cfg["seed"] + 7)
    A_samp = [samp0]
    forest_a = forest0
    for li in range(1, len(spec.layers)):
        k_s, key = jax.random.split(key)
        samp_up = sample_forest(forest_a, k_s, n_eval, grid_size=cfg["grid_size"])
        predA = propagate_layer(spec, li, samp_up, params, rng)
        A_samp.append(predA)
        k_f, key = jax.random.split(key)
        forest_a = fit_layer_forest(jnp.asarray(predA), list(spec.layers[li]), cfg, k_f,
                                    label=f"A_L{li}", mi_threshold=cfg["mi_threshold"])

    B_samp = [samp0]
    forest_b = forest0
    for li in range(1, len(spec.layers)):
        upper = UpperForest(forest_b, q_grid=cfg["q_grid"])
        s_max = float(gt_layers[li].max() * 1.1)
        k_s, key = jax.random.split(key)
        predB = sample_layer_copula(upper, spec, li, params, k_s, n_eval, s_max, n_s=cfg["n_s"])
        B_samp.append(predB)
        k_f, key = jax.random.split(key)
        forest_b = fit_layer_forest(jnp.asarray(predB), list(spec.layers[li]), cfg, k_f,
                                    label=f"B_L{li}", mi_threshold=cfg["mi_threshold"])

    return spec, gt_layers, A_samp, B_samp


def main():
    cfg = dict(
        layer_sizes=[4, 4, 4, 4], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
        n=20000, n_eval=5000, q=2, m=16, rank=8, lr=2e-3, steps=800, batch_sz=512,
        init_noise=1e-2, train_noise=1e-3, log_every=10000, early_stop_patience=6,
        mi_threshold=0.02, grid_size=400, q_grid=400, n_s=140, seed=0,
    )
    spec, gt_layers, A_samp, B_samp = collect(cfg)
    nL = len(spec.layers)
    nN = max(len(l) for l in spec.layers)

    fig, axes = plt.subplots(nL, nN, figsize=(3.0 * nN, 2.4 * nL), squeeze=False)
    for li in range(nL):
        nodes = list(spec.layers[li])
        for j in range(nN):
            ax = axes[li][j]
            if j >= len(nodes):
                ax.axis("off")
                continue
            gtv = gt_layers[li][:, j]
            av = A_samp[li][:, j]
            bv = B_samp[li][:, j]
            lo = float(min(gtv.min(), av.min(), bv.min()))
            hi = float(max(gtv.max(), av.max(), bv.max()))
            # 限制 x 轴，避免 B 发散把图拉爆
            lo2, hi2 = float(np.percentile(gtv, 0.5)), float(np.percentile(gtv, 99.5))
            pad = 0.15 * (hi2 - lo2 + 1e-9)
            xlo, xhi = lo2 - pad, hi2 + pad
            bins = np.linspace(lo, hi, 80)
            ax.hist(gtv, bins=bins, density=True, histtype="step", color="k", lw=1.8, label="GT")
            ax.hist(av, bins=bins, density=True, histtype="step", color="tab:blue", lw=1.3, label="A")
            ax.hist(bv, bins=bins, density=True, histtype="step", color="tab:red", lw=1.3, label="B")
            ax.set_xlim(xlo, xhi)
            ax.set_title(f"L{li} node{nodes[j]}", fontsize=9)
            if li == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Per-layer marginal slices: GT vs A-chain vs B-chain (copula)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "chain_slices.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
