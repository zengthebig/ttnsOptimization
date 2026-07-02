"""诊断：L1 那条"丢掉"的相关，到底丢在哪一步？

打印：① 真值相关矩阵 ② 解析(Scheme B Hoeffding)相关矩阵 ③ 选中的 Chow-Liu 树边。
若解析相关矩阵里该对相关很大、只是没被选进树边 → 是"单树"这个模型选择丢的，与 copula 无关；
Scheme B 其实全对（含非树边）。
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

from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, sample_forest  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import UpperForest  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import analytic_layer_target  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402

cfg = dict(
    layer_sizes=[6, 6, 6, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
    n_total=40000, q=2, m=24, rank=8, source_mode="bimodal", src_sigma=0.06,
    lr=0.002, steps=1000, batch_sz=512, init_noise=0.01, train_noise=0.001,
    log_every=100000, early_stop_patience=8, mi_threshold=0.02, seed=0,
)
LI = 1


def edges_of(parent):
    return {tuple(sorted((v, p))) for v, p in enumerate(parent) if p not in (-1, v)}


def main():
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **bimodal_sources(spec, cfg)}
    k_d, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_d, cfg["n_total"], clip=(-1e9, 1e9)))
    train_x = xs[: int(0.7 * xs.shape[0])]
    L0 = list(spec.layers[0])
    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f, label="L0", mi_threshold=cfg["mi_threshold"])

    # 真值 L1
    k_s, key = jax.random.split(key)
    s0 = np.asarray(sample_forest(forest0, k_s, 8000, grid_size=400))
    true_l1 = propagate_layer(spec, LI, s0, params, np.random.default_rng(1))
    s_max = float(true_l1.max()) * 1.05
    Ct = np.corrcoef(true_l1.T)

    # 解析层目标（含全相关矩阵 + Chow-Liu 树）
    upper = UpperForest(forest0, q_grid=400)
    layer = analytic_layer_target(upper, spec, LI, params, s_max, n_s=100, n_s_pair=80)
    Can = layer.corr
    tree_edges = edges_of(layer.parent)

    np.set_printoptions(precision=3, suppress=True)
    print("\n真值相关矩阵 Ct:\n", Ct)
    print("\n解析(Scheme B)相关矩阵 Can:\n", Can)
    print("\nChow-Liu 树边(局部索引):", sorted(tree_edges))

    K = true_l1.shape[1]
    print("\n每对: 真值corr | 解析corr | 是否树边  (按真值|corr|降序)")
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    pairs.sort(key=lambda ab: -abs(Ct[ab]))
    for a, b in pairs:
        is_tree = tuple(sorted((a, b))) in tree_edges
        flag = "★树边" if is_tree else "  非树边"
        print(f"  (y{a},y{b}): 真={Ct[a,b]:+.3f}  解析={Can[a,b]:+.3f}  {flag}")

    # 结论量化：解析相关矩阵 vs 真值（这是 Scheme B 本身的精度，与树/copula 无关）
    iu = np.triu_indices(K, 1)
    print(f"\n解析相关矩阵 vs 真值 的逐元素平均绝对误差(仅上三角) = {np.mean(np.abs(Ct[iu]-Can[iu])):.4f}")
    print("→ 若这个误差很小，说明 Scheme B 把所有两两相关(含非树边)都算对了；")
    print("  相关丢失只发生在'把它塞进单棵树'这一步，与 copula 无关。\n")


if __name__ == "__main__":
    main()
