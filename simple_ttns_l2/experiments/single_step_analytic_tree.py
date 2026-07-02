"""单步验证 L0->L1：解析树 TTNS (clarify.md 全解析链) vs A(采样重拟合) vs 真值。

- 真值 L1 = 祖先采样(L0 森林采样 → 真 max-plus)，携带全阶真依赖。
- A     = 把真值 L1 样本重拟合成树 TTNS(Scheme A：propagate+refit，样本噪声估边缘/边)。
- 解析   = fit_next_layer_tree：解析算节点边缘 + 树边 → 解析 L2 拟合树 TTNS，**不采样**。

都是"单父树 TTNS"，比：① 边缘 mean/std ② 相关矩阵 corr_fro ③ 在真值 L1 上的 joint_LL
④ 采样重构的高阶量(corr(|dev|)/coskew)。假设：解析 ≥ A(精确 vs 噪声估计的同一树目标)。
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
from simple_ttns_l2.layered_forest import fit_layer_forest, sample_forest, forest_log_density  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import UpperForest  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import fit_next_layer_tree  # noqa: E402
from simple_ttns_l2.ttns_sampler import sample_ttns  # noqa: E402
from simple_ttns_l2.objective import batch_basis_vectors_from_samples, batch_eval_q_ttns  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402

cfg = dict(
    layer_sizes=[6, 6, 6, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
    n_total=40000, q=2, m=24, rank=8, source_mode="bimodal", src_sigma=0.06,
    lr=0.002, steps=1000, batch_sz=512, init_noise=0.01, train_noise=0.001,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
)
N = 8000          # 真值/采样量
N_EVAL = 8000     # joint_LL 评测的独立真值样本
LI = 1


def tree_logdensity(ttns, bases, parent, x, eps=1e-12):
    bv = batch_basis_vectors_from_samples(bases, jnp.asarray(x))
    q = np.asarray(batch_eval_q_ttns(ttns, bv, list(parent)))
    return np.log(np.clip(q, eps, None)), float((q <= 0).mean())


def higher(x, i, j):
    xi = x[:, i] - x[:, i].mean(); xj = x[:, j] - x[:, j].mean()
    return dict(
        corr_abs=np.corrcoef(np.abs(xi), np.abs(xj))[0, 1],
        coskew=np.mean(xi ** 2 * xj) / (x[:, i].std() ** 2 * x[:, j].std() + 1e-12),
    )


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

    # 真值 L1: 祖先采样(训练用 + 评测用)
    k_s, key = jax.random.split(key)
    s0 = np.asarray(sample_forest(forest0, k_s, N, grid_size=400))
    true_l1 = propagate_layer(spec, LI, s0, params, np.random.default_rng(1))
    k_se, key = jax.random.split(key)
    s0e = np.asarray(sample_forest(forest0, k_se, N_EVAL, grid_size=400))
    true_l1_eval = propagate_layer(spec, LI, s0e, params, np.random.default_rng(2))

    s_max = float(true_l1.max()) * 1.05
    L1 = list(spec.layers[LI])

    # A：把真值 L1 重拟合成树 TTNS(单块 → 一棵 chow-liu 树)
    k_a, key = jax.random.split(key)
    forestA = fit_layer_forest(jnp.asarray(true_l1), L1, cfg, k_a, label="A_L1", mi_threshold=cfg["mi_threshold"])
    k_as, key = jax.random.split(key)
    a_samp = np.asarray(sample_forest(forestA, k_as, N, grid_size=400))
    llA, _ = forest_log_density(forestA, true_l1_eval)

    # 解析：clarify.md 全解析链
    upper = UpperForest(forest0, q_grid=400)
    k_an, key = jax.random.split(key)
    ttnsY, parentY, basesY, layer = fit_next_layer_tree(
        upper, spec, LI, params, k_an, s_max=s_max,
        q=cfg["q"], m=cfg["m"], rank=cfg["rank"], n_s=100, n_s_pair=80,
        lr=3e-3, steps=1500, init_noise=cfg["init_noise"], log_every=500,
    )
    k_ans, key = jax.random.split(key)
    an_samp = np.asarray(sample_ttns(ttnsY, basesY, parentY, k_ans, N, grid_size=400))
    llAN, nonpos = tree_logdensity(ttnsY, basesY, parentY, true_l1_eval)

    # 相关矩阵
    Ct = np.corrcoef(true_l1.T)
    Ca = np.corrcoef(a_samp.T)
    Can = np.corrcoef(an_samp.T)

    print("\n================ 单步 L0->L1：解析树 vs A vs 真值 ================")
    print(f"树拓扑(解析 chow-liu parent，局部索引): {parentY}")
    print(f"\n【joint_LL@真值L1(越高越好)】  A={llA.mean():.4f}   解析={llAN.mean():.4f}"
          f"   (解析非正率={nonpos:.3f})")
    print(f"\n【corr_fro vs 真值(越低越好)】 A={np.linalg.norm(Ct-Ca):.4f}   解析={np.linalg.norm(Ct-Can):.4f}")

    print("\n【① 边缘 mean/std  (真值 | A | 解析)】")
    for j in range(true_l1.shape[1]):
        print(f"  y{j}: mean {true_l1[:,j].mean():.3f}|{a_samp[:,j].mean():.3f}|{an_samp[:,j].mean():.3f}"
              f"   std {true_l1[:,j].std():.3f}|{a_samp[:,j].std():.3f}|{an_samp[:,j].std():.3f}")

    # 高阶：最相关一对
    iu = np.triu_indices(true_l1.shape[1], 1)
    a_, b_ = iu[0][np.argmax(np.abs(Ct[iu]))], iu[1][np.argmax(np.abs(Ct[iu]))]
    ht, ha, hn = higher(true_l1, a_, b_), higher(a_samp, a_, b_), higher(an_samp, a_, b_)
    print(f"\n【② 最相关对 (y{a_},y{b_})  corr 真={Ct[a_,b_]:.3f} A={Ca[a_,b_]:.3f} 解析={Can[a_,b_]:.3f}】")
    print(f"  corr(|dev|): 真={ht['corr_abs']:.3f}  A={ha['corr_abs']:.3f}  解析={hn['corr_abs']:.3f}")
    print(f"  coskew     : 真={ht['coskew']:.3f}  A={ha['coskew']:.3f}  解析={hn['coskew']:.3f}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
