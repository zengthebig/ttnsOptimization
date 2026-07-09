"""dense_dag_r567 的**密度切片图**：每个 (层, 节点) 一个子图，画真值 GT vs R5 vs R7 的边缘密度
（直方图），直观看模型在哪层/哪个节点开始偏离真值——对标 plot_chain_slices.py 的 GT/A/B 切片。

复用 dense_dag_r567_three_way 的 CFG 与拟合链（L0 森林 → R5 解析链 / R7 采样链），拟合 1 个 seed，
逐层从各自森林采样后作边缘密度切片。
输出 simple_ttns_l2/reports/dense_dag_r567_slices.png。
用法：env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.plot_dense_dag_slices [--seed 0]
"""
from __future__ import annotations

import argparse
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
from matplotlib import font_manager as fm  # noqa: E402

for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf"):
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

from simple_ttns_l2.dag_pipeline import build_crossed_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, sample_forest  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import fit_analytic_chain, fit_sampled_chain  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG, complex_sources  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
OUT = REPORTS / "dense_dag_r567_slices.png"
N_EVAL = 5000
GRID = 400


def collect(seed: int):
    cfg = dict(CFG)
    key = jax.random.PRNGKey(seed)
    spec = build_crossed_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"],
                              cross_pairs=cfg.get("cross_pairs"),
                              cross_fanin=cfg.get("cross_fanin", 1),
                              rotate_cross=cfg.get("rotate_cross", False), wrap=True)
    params = DelayParams(**cfg["delay"])
    layers = [list(l) for l in spec.layers]

    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    gt = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * gt.shape[0])
    train_x = gt[:n_tr]
    gt_layers = [gt[n_tr:][:, Lg] for Lg in layers]  # 用留出集画真值
    L0 = layers[0]

    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))

    k_an, key = jax.random.split(key)
    R5 = fit_analytic_chain(forest0, spec, params, k_an, s_max0,
                            q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
                            n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                            lr=cfg["an_lr"], steps=cfg["an_steps"],
                            init_noise=cfg["init_noise"], log_every=0,
                            block_mode=cfg["block_mode"])
    k_sp, key = jax.random.split(key)
    R7 = fit_sampled_chain(forest0, spec, params, k_sp, cfg)

    # 逐层从各自森林采样
    R5_s, R7_s = [], []
    for li in range(len(layers)):
        k1, key = jax.random.split(key)
        R5_s.append(np.asarray(sample_forest(R5[li], k1, N_EVAL, grid_size=GRID)))
        k2, key = jax.random.split(key)
        R7_s.append(np.asarray(sample_forest(R7[li], k2, N_EVAL, grid_size=GRID)))
    return spec, layers, gt_layers, R5_s, R7_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec, layers, gt_layers, R5_s, R7_s = collect(args.seed)
    nL = len(layers)
    nN = max(len(l) for l in layers)

    fig, axes = plt.subplots(nL, nN, figsize=(1.55 * nN, 1.95 * nL), squeeze=False)
    for li in range(nL):
        nodes = layers[li]
        for j in range(nN):
            ax = axes[li][j]
            if j >= len(nodes):
                ax.axis("off")
                continue
            gtv, r5v, r7v = gt_layers[li][:, j], R5_s[li][:, j], R7_s[li][:, j]
            lo2 = float(np.percentile(gtv, 0.5))
            hi2 = float(np.percentile(gtv, 99.5))
            pad = 0.15 * (hi2 - lo2 + 1e-9)
            bins = np.linspace(lo2 - pad, hi2 + pad, 70)
            ax.hist(gtv, bins=bins, density=True, histtype="step", color="k", lw=1.9, label="GT")
            ax.hist(r5v, bins=bins, density=True, histtype="step", color="#4C78A8", lw=1.3, label="R5")
            ax.hist(r7v, bins=bins, density=True, histtype="step", color="#E45756", lw=1.3, label="R7")
            ax.set_xlim(lo2 - pad, hi2 + pad)
            ax.set_yticks([])
            ax.set_title(f"L{li}·n{nodes[j]}", fontsize=7.5)
            if li == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("dense_dag_r567 逐层逐节点边缘密度切片：GT(黑) vs R5(蓝) vs R7(红) "
                 "— 跨簇DAG·immediate分块·100节点", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
