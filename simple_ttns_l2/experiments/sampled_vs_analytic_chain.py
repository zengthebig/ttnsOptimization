"""链级对比:解析链(树投影目标) vs 采样链(采样求 L2, 完整联合目标)。

两条链结构完全一致(DAG 结构分块、逐层单父树 TTNS),唯一区别是**每块拟合目标的来源**:
- analytic : 解析算树投影 p_tree(现有 fit_analytic_chain, 不改动);
- sampled  : 采上层 + max-plus merge 得样本, 标准 L2(train_tree_l2)拟合(完整联合, 允许 MC 噪声)。

逐层用留出真值 test_x[:,layer] 比:joint_LL(↑) 与采样 corr_fro(↓)。
"""
from __future__ import annotations

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

from simple_ttns_l2.dag_pipeline import build_clustered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, forest_log_density, sample_forest  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import fit_analytic_chain, fit_sampled_chain  # noqa: E402
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources  # noqa: E402


def corr_fro_layer(forest, tev, key, n, n_rep=3):
    Ct = np.corrcoef(tev.T) if tev.shape[1] > 1 else np.array([[1.0]])
    fros = []
    for _ in range(n_rep):
        key, k = jax.random.split(key)
        s = np.asarray(sample_forest(forest, k, n, grid_size=400))
        Cm = np.corrcoef(s.T) if s.shape[1] > 1 else np.array([[1.0]])
        fros.append(float(np.linalg.norm(Ct - Cm)))
    return float(np.mean(fros)), float(np.std(fros))


def run(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    layers = [list(l) for l in spec.layers]
    print(f"[spec] 节点总数={spec.n}  层数={len(layers)}  每层={len(layers[0])}维  "
          f"簇/块大小={cfg['clusters']}  边数={len(spec.edges)}", flush=True)

    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    L0 = layers[0]

    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))

    t0 = time.perf_counter()
    k_an, key = jax.random.split(key)
    AN = fit_analytic_chain(forest0, spec, params, k_an, s_max0,
                            q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
                            n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                            lr=cfg["an_lr"], steps=cfg["an_steps"],
                            init_noise=cfg["init_noise"], log_every=0)
    t_an = time.perf_counter() - t0

    t0 = time.perf_counter()
    k_sp, key = jax.random.split(key)
    SP = fit_sampled_chain(forest0, spec, params, k_sp, cfg)
    t_sp = time.perf_counter() - t0

    rows = []
    for li in range(len(layers)):
        tev = test_x[:, layers[li]]
        ll_an = float(forest_log_density(AN[li], tev)[0].mean())
        ll_sp = float(forest_log_density(SP[li], tev)[0].mean())
        k1, key = jax.random.split(key)
        k2, key = jax.random.split(key)
        fro_an, sd_an = corr_fro_layer(AN[li], tev, k1, cfg["n_sample"])
        fro_sp, sd_sp = corr_fro_layer(SP[li], tev, k2, cfg["n_sample"])
        rows.append(dict(li=li, K=len(layers[li]),
                         ll_an=ll_an, ll_sp=ll_sp, fro_an=fro_an, fro_sp=fro_sp,
                         sd_an=sd_an, sd_sp=sd_sp))
    return rows, dict(analytic=t_an, sampled=t_sp)


def print_table(rows, timings):
    print("\n" + "=" * 88)
    print("链级对比: analytic(树投影) vs sampled(采样求L2, 完整联合)  —— 同结构/同参数")
    print("=" * 88)
    print(f"{'层':<4}{'K':<4}{'LL_an':>10}{'LL_samp':>10}{'ΔLL':>8}"
          f"{'fro_an':>10}{'fro_samp':>10}{'Δfro':>8}")
    for r in rows:
        dll = r["ll_sp"] - r["ll_an"]
        dfro = r["fro_sp"] - r["fro_an"]
        print(f"L{r['li']:<3}{r['K']:<4}{r['ll_an']:>10.4f}{r['ll_sp']:>10.4f}{dll:>+8.4f}"
              f"{r['fro_an']:>10.4f}{r['fro_sp']:>10.4f}{dfro:>+8.4f}")
        print(f"      (fro sd: an={r['sd_an']:.4f}  samp={r['sd_sp']:.4f})")
    print("-" * 88)
    print("ΔLL>0 / Δfro<0 表示采样链更好。")
    print(f"总用时: analytic={timings['analytic']:.1f}s  sampled={timings['sampled']:.1f}s")
    print("=" * 88)


CFG_SMALL = dict(
    n_layers=3, clusters=[3, 3], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
    n_total=20000, n_sample=16000, n_fit=20000, q=2, m=24, rank=8,
    src_sigma=0.03,
    lr=2e-3, steps=800, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
    n_s=100, n_s_pair=80, an_lr=3e-3, an_steps=800,
)

# 大例子: 异构簇 [2,3,4,5,6] → 每层 20 维、5 个不同大小的块; 5 层 = 100 节点, 4 个下游层
CFG_BIG = dict(
    n_layers=5, clusters=[2, 3, 4, 5, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
    n_total=24000, n_sample=16000, n_fit=20000, q=2, m=24, rank=8,
    src_sigma=0.03,
    lr=2e-3, steps=700, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
    log_every=350, early_stop_patience=8, mi_threshold=0.02, seed=0,
    n_s=100, n_s_pair=80, an_lr=3e-3, an_steps=700,
)


def main():
    cfg = CFG_BIG if "--big" in sys.argv else CFG_SMALL
    t0 = time.perf_counter()
    rows, timings = run(cfg)
    print_table(rows, timings)
    print(f"总用时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
