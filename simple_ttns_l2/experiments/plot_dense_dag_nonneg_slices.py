"""为 R5/R7 nonneg 链生成完整 marginal 切片图（5层×20节点，对标 dense_dag_r567_slices.png）。

输出：
  simple_ttns_l2/reports/dense_dag_r567_r5_nonneg_slices.png
  simple_ttns_l2/reports/dense_dag_r567_r7_nonneg_slices.png

用法：
  python3 -m simple_ttns_l2.experiments.plot_dense_dag_nonneg_slices --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
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

from simple_ttns_l2.layered_forest import sample_forest  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_remedy_tests import (  # noqa: E402
    prepare_data,
    fit_analytic_chain_nonneg,
    fit_layer_forest_nonneg,
    fit_sampled_chain_nonneg,
)

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
OUT_R5 = REPORTS / "dense_dag_r567_r5_nonneg_slices.png"
OUT_R7 = REPORTS / "dense_dag_r567_r7_nonneg_slices.png"
N_PLOT = 8000
GRID = 400


def plot_marginal_slices(
    test_x: np.ndarray,
    layers: list,
    chain: dict,
    *,
    model_label: str,
    model_color: str,
    title: str,
    out_path: Path,
) -> None:
    """全节点一维边缘直方图：GT(黑) vs 单条 nonneg 链。"""
    n_plot = min(N_PLOT, test_x.shape[0])
    key = jax.random.PRNGKey(0)
    samples = {}
    for li in range(len(layers)):
        key, ks = jax.random.split(key)
        samples[li] = np.asarray(sample_forest(chain[li], ks, n_plot, grid_size=GRID))

    nL = len(layers)
    nN = max(len(l) for l in layers)
    fig, axes = plt.subplots(nL, nN, figsize=(1.55 * nN, 1.95 * nL), squeeze=False)
    for li in range(nL):
        nodes = layers[li]
        truth_layer = test_x[:n_plot, nodes]
        for j in range(nN):
            ax = axes[li][j]
            if j >= len(nodes):
                ax.axis("off")
                continue
            gtv = truth_layer[:, j]
            lo = float(np.percentile(gtv, 0.5))
            hi = float(np.percentile(gtv, 99.5))
            pad = 0.15 * (hi - lo + 1e-9)
            bins = np.linspace(lo - pad, hi + pad, 70)
            ax.hist(gtv, bins=bins, density=True, histtype="step", color="k", lw=1.9, label="GT")
            ax.hist(
                samples[li][:, j], bins=bins, density=True, histtype="step",
                color=model_color, lw=1.3, label=model_label,
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_yticks([])
            ax.set_title(f"L{li}·n{nodes[j]}", fontsize=7.5)
            if li == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"saved: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r5-only", action="store_true")
    ap.add_argument("--r7-only", action="store_true")
    args = ap.parse_args()
    run_r5 = not args.r7_only
    run_r7 = not args.r5_only

    cfg = dict(CFG)
    cfg.update(nn_lr=2e-3, nn_steps=400, nn_batch_sz=512, nn_patience=6)
    t0 = time.perf_counter()

    key, spec, params, layers, train_x, test_x, forest0, s_max0 = prepare_data(args.seed, cfg)
    L0 = layers[0]

    if run_r5:
        print("[plot] fitting R5 nonneg chain...", flush=True)
        t1 = time.perf_counter()
        k_r5, key = jax.random.split(key)
        r5_chain = fit_analytic_chain_nonneg(forest0, spec, params, k_r5, s_max0, cfg)
        print(f"[plot] R5 nonneg fit {time.perf_counter()-t1:.1f}s", flush=True)
        plot_marginal_slices(
            test_x, layers, r5_chain,
            model_label="R5 nonneg",
            model_color="#9467bd",
            title="dense_dag_r567 R5 nonneg per-node marginal slices (core→raw² + analytic L2)",
            out_path=OUT_R5,
        )

    if run_r7:
        print("[plot] fitting R7 nonneg chain...", flush=True)
        t1 = time.perf_counter()
        k_f0, key = jax.random.split(key)
        forest0_nn = fit_layer_forest_nonneg(
            jnp.asarray(train_x[:, L0]), L0, cfg, k_f0,
            label="L0_nn", mi_threshold=cfg["mi_threshold"],
        )
        k_r7, key = jax.random.split(key)
        r7_chain = fit_sampled_chain_nonneg(forest0_nn, spec, params, k_r7, cfg)
        print(f"[plot] R7 nonneg fit {time.perf_counter()-t1:.1f}s", flush=True)
        plot_marginal_slices(
            test_x, layers, r7_chain,
            model_label="R7 nonneg",
            model_color="#2ca02c",
            title="dense_dag_r567 R7 nonneg per-node marginal slices (core→raw² + block MLE)",
            out_path=OUT_R7,
        )

    print(f"total {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
