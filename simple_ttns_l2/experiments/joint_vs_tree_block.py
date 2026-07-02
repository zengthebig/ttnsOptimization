"""单块隔离测试：层间"完整传播"(完整联合目标) vs 原"树投影目标"。

在 clustered DAG 的第 1 层上，对每个 K>=2 的结构块，分别用两种目标拟合**同一种**单父树 TTNS：
- tree :  analytic_block_target + _fit_analytic_ttns          (目标 = Chow-Liu 树投影 p_tree)
- joint:  analytic_block_target_joint + _fit_analytic_ttns_joint (目标 = 完整块联合 p_Y)

两者 q 的树结构相同(解析 MI)、参数量相同、优化超参相同，唯一区别是**拟合目标口径**。
用留出真值 test_x[:,gids] 比较：块 test-LL(↑) 与 3x3 corr_fro(↓) 与 top-pair corr。
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

from jax import vmap  # noqa: E402

from simple_ttns_l2.dag_pipeline import build_clustered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, sample_forest  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import UpperForest  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import (  # noqa: E402
    structural_blocks, analytic_block_target, analytic_block_target_joint,
    _fit_analytic_ttns, _fit_analytic_ttns_joint,
)
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.objective import (  # noqa: E402
    batch_basis_vectors_from_samples, batch_eval_q_ttns, normalize_ttns_by_integral,
)
from simple_ttns_l2.experiments.fit_diamond_dag_vs_tree import train_tree_l2  # noqa: E402
from simple_ttns_l2.ttns_sampler import sample_ttns  # noqa: E402
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources  # noqa: E402


def block_logp(ttns, bases, parent, Xblock, eps=1e-12):
    bv = batch_basis_vectors_from_samples(bases, jnp.asarray(Xblock))
    q = np.asarray(batch_eval_q_ttns(ttns, bv, list(parent)))
    return np.log(np.clip(q, eps, None))


def fit_block_sampled(forest0, spec, li, blk, parent, params, cfg, key):
    """采样交叉项变体：上层采一次 + max-plus merge → 块样本 y~p_Y，标准 L2(train_tree_l2)拟合。

    交叉项 = batch 均值 q(y_n)(蒙特卡洛无偏)；∫q² 仍解析(gram)。树结构 = 传入 parent。"""
    k_s, key = jax.random.split(key)
    s_upper = np.asarray(sample_forest(forest0, k_s, cfg["n_fit"], grid_size=400))
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    s_layer = propagate_layer(spec, li, s_upper, params, rng)  # [n, layer_size]
    Xb = jnp.asarray(s_layer[:, blk])
    bases = build_bases(Xb, cfg["q"], cfg["m"])
    gram = vmap(type(bases).l2_integral)(bases)
    basis_int = vmap(type(bases).integral)(bases)
    split = int(0.85 * Xb.shape[0])
    tr, val = Xb[:split], Xb[split:]
    k_i, key = jax.random.split(key)
    t0 = init_ttns_from_rank1(k_i, bases, tr, list(parent), cfg["rank"], cfg["init_noise"])
    best, _ = train_tree_l2(
        t0, list(parent), bases, tr, val, gram, basis_int,
        key=k_i, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
        normalize_every=1, log_every=cfg["log_every"], label=f"samp.b",
        train_noise=cfg["train_noise"], early_stop_patience=cfg["early_stop_patience"],
    )
    best, _ = normalize_ttns_by_integral(best, basis_int, list(parent))
    return best, bases


def corr_fro(ttns, bases, parent, key, n, Ct, n_rep=3):
    """采样估 corr_fro，重复 n_rep 次取均值以压低采样方差。返回 (mean_fro, std_fro)。"""
    fros = []
    for _ in range(n_rep):
        key, k = jax.random.split(key)
        xs = np.asarray(sample_ttns(ttns, bases, list(parent), k, n, grid_size=400))
        Cm = np.corrcoef(xs.T) if xs.shape[1] > 1 else np.array([[1.0]])
        fros.append(float(np.linalg.norm(Ct - Cm)))
    return float(np.mean(fros)), float(np.std(fros))


def run(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    layers = [list(l) for l in spec.layers]

    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    L0 = layers[0]

    # L0 数据森林 → UpperForest
    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))
    upper = UpperForest(forest0, q_grid=400)

    li = cfg["test_layer"]
    s_max = s_max0 + li * ((params.edge_hi + params.node_hi) + 0.3)
    layer_nodes = layers[li]
    blocks = structural_blocks(spec, li)

    print(f"\n第 {li} 层 结构块: {[[layer_nodes[i] for i in b] for b in blocks]}")
    print(f"s_max={s_max:.3f}\n")

    rows = []
    for bi, blk in enumerate(blocks):
        gids = [layer_nodes[i] for i in blk]
        K = len(gids)
        if K < 2:
            continue
        tev = test_x[:, gids]
        Ct = np.corrcoef(tev.T)

        # tree 目标
        t0 = time.perf_counter()
        tgt_tree = analytic_block_target(upper, spec, gids, params, s_max,
                                         n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"], use_mi=True)
        k1, key = jax.random.split(key)
        ttns_t, bases_t = _fit_analytic_ttns(tgt_tree, k1, cfg["q"], cfg["m"], cfg["rank"],
                                             cfg["an_lr"], cfg["an_steps"], cfg["init_noise"],
                                             log_every=0, label=f"tree.b{bi}")
        t_tree = time.perf_counter() - t0

        # joint 目标
        t0 = time.perf_counter()
        tgt_joint = analytic_block_target_joint(upper, spec, gids, params, s_max,
                                                n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                                                n_s_joint=cfg["n_s_joint"], use_mi=True)
        k2, key = jax.random.split(key)
        ttns_j, bases_j = _fit_analytic_ttns_joint(tgt_joint, k2, cfg["q"], cfg["m"], cfg["rank"],
                                                   cfg["an_lr"], cfg["an_steps"], cfg["init_noise"],
                                                   log_every=0, label=f"joint.b{bi}")
        t_joint = time.perf_counter() - t0

        # 采样交叉项变体(树结构沿用解析 MI 树, 与 tree/joint 公平)
        t0 = time.perf_counter()
        k3, key = jax.random.split(key)
        ttns_s, bases_s = fit_block_sampled(forest0, spec, li, blk, tgt_tree.parent,
                                            params, cfg, k3)
        t_samp = time.perf_counter() - t0

        ll_t = float(block_logp(ttns_t, bases_t, tgt_tree.parent, tev).mean())
        ll_j = float(block_logp(ttns_j, bases_j, tgt_joint.parent, tev).mean())
        ll_s = float(block_logp(ttns_s, bases_s, tgt_tree.parent, tev).mean())
        ks1, key = jax.random.split(key)
        ks2, key = jax.random.split(key)
        ks3, key = jax.random.split(key)
        fro_t, fro_t_sd = corr_fro(ttns_t, bases_t, tgt_tree.parent, ks1, cfg["n_sample"], Ct)
        fro_j, fro_j_sd = corr_fro(ttns_j, bases_j, tgt_joint.parent, ks2, cfg["n_sample"], Ct)
        fro_s, fro_s_sd = corr_fro(ttns_s, bases_s, tgt_tree.parent, ks3, cfg["n_sample"], Ct)

        iu = np.triu_indices(K, 1)
        k_top = int(np.argmax(np.abs(Ct[iu])))
        a_, b_ = int(iu[0][k_top]), int(iu[1][k_top])
        rows.append(dict(bi=bi, gids=gids, K=K, parent_tree=list(tgt_tree.parent),
                         parent_joint=list(tgt_joint.parent),
                         ll_tree=ll_t, ll_joint=ll_j, ll_samp=ll_s,
                         fro_tree=fro_t, fro_joint=fro_j, fro_samp=fro_s,
                         fro_tree_sd=fro_t_sd, fro_joint_sd=fro_j_sd, fro_samp_sd=fro_s_sd,
                         corr_t=float(Ct[a_, b_]), pair=(a_, b_),
                         t_tree=t_tree, t_joint=t_joint, t_samp=t_samp))
    return rows


def print_table(rows):
    print("=" * 104)
    print("单块对比: tree(树投影解析) vs joint(完整联合网格) vs samp(采样交叉项) —— 同树结构/同参数/同超参")
    print("=" * 104)
    print(f"{'块':<4}{'节点':<14}{'LL_tree':>9}{'LL_joint':>9}{'LL_samp':>9}"
          f"{'fro_tree':>9}{'fro_joint':>9}{'fro_samp':>9}")
    for r in rows:
        print(f"{r['bi']:<4}{str(r['gids']):<14}"
              f"{r['ll_tree']:>9.4f}{r['ll_joint']:>9.4f}{r['ll_samp']:>9.4f}"
              f"{r['fro_tree']:>9.4f}{r['fro_joint']:>9.4f}{r['fro_samp']:>9.4f}")
        print(f"      (fro sd: tree={r['fro_tree_sd']:.4f} joint={r['fro_joint_sd']:.4f} "
              f"samp={r['fro_samp_sd']:.4f})")
    print("-" * 104)
    print("LL↑ 越大越好; fro↓ 越小越好。joint 与 samp 都以'完整联合'为目标(前者解析网格, 后者蒙特卡洛)。")
    print(f"用时(每块): tree={np.mean([r['t_tree'] for r in rows]):.1f}s  "
          f"joint={np.mean([r['t_joint'] for r in rows]):.1f}s  "
          f"samp={np.mean([r['t_samp'] for r in rows]):.1f}s")
    print("=" * 104)


def main():
    cfg = dict(
        n_layers=3, clusters=[3, 3], fanin=2, test_layer=1,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
        n_total=20000, n_sample=16000, n_fit=20000, q=2, m=24, rank=8,
        src_sigma=0.03,
        lr=2e-3, steps=800, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
        log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
        n_s=100, n_s_pair=80, n_s_joint=60, an_lr=3e-3, an_steps=800,
    )
    t0 = time.perf_counter()
    rows = run(cfg)
    print_table(rows)
    print(f"总用时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
