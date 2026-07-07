"""正确性校验:推广后的 R5 解析边缘(任意延迟分布) vs 蒙特卡洛。

(1) 向后兼容:uniform 模式下 edge_cdf/node_quadrature 与旧实现一致(隐含,通过 MC 对齐验证)。
(2) log-skew-normal:R5 解析 marginal_cdf 求导得密度,须与"采样上层+merge+直方图"吻合。

构造:L0 单块 2 节点 TTNS(拟合自数据),L1 某节点两父在该块 → 用 UpperModel.marginal_cdf
解析算其边缘,对照 MC。两种延迟分布各测一次。
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

from simple_ttns_l2.dag_pipeline import build_clustered_spec, sample_joint
from simple_ttns_l2.maxplus_pipeline import (
    DelayParams, ground_truth_samplers, sample_edge, sample_node,
)
from simple_ttns_l2.layered_forest import fit_layer_forest
from simple_ttns_l2.ttns_sampler import sample_ttns
from simple_ttns_l2.maxplus_cdf import UpperModel, marginal_cdf


def run_one(kind: str) -> float:
    if kind == "uniform":
        params = DelayParams(edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3, kind="uniform")
    else:
        params = DelayParams(kind="logskewnorm",
                             e_xi=-2.12, e_omega=0.45, e_alpha=4.0,
                             d_xi=-2.12, d_omega=0.45, d_alpha=4.0)
    cfg = dict(q=2, m=24, rank=8, lr=2e-3, steps=500, batch_sz=512,
               init_noise=1e-2, train_noise=1e-3, log_every=600, early_stop_patience=8)
    # 2 层, L0 单块 2 节点, L1 每节点 2 父
    spec = build_clustered_spec(2, [2], fanin=2, wrap=True)
    src, ker = ground_truth_samplers(spec, params)
    key = jax.random.PRNGKey(0)
    kd, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, src, ker, kd, 20000, clip=(-1e9, 1e9)))
    L0 = list(spec.layers[0])
    kf, key = jax.random.split(key)
    forest = fit_layer_forest(jnp.asarray(xs[:, L0]), L0, cfg, kf, label="L0", mi_threshold=0.0)
    bm = forest[0]                      # 单块
    upper = UpperModel(bm.ttns, bm.bases, list(bm.parent), q_grid=400)

    # 目标 L1 节点:取其两父都在 L0 块内的第一个
    L1 = list(spec.layers[1])
    node = L1[0]
    ps = list(spec.parents(node))
    gid2local = {g: i for i, g in enumerate(bm.global_vars)}
    parents_local = [gid2local[p] for p in ps]

    s_max = float(np.asarray(bm.bases.knots).max()) + 2.0
    s_grid = np.linspace(0.0, s_max, 400)
    F = marginal_cdf(upper, parents_local, s_grid, params, n_d=96)
    dens_an = np.gradient(np.clip(F, 0, 1), s_grid)

    # MC: 采上层 TTNS + merge
    ks, key = jax.random.split(key)
    xup = np.asarray(sample_ttns(bm.ttns, bm.bases, list(bm.parent), ks, 200000, grid_size=400))
    rng = np.random.default_rng(0)
    pv = np.stack([xup[:, gid2local[p]] for p in ps], axis=1)
    e = sample_edge(params, rng, pv.shape)
    d = sample_node(params, rng, pv.shape[0])
    y = (pv + e).max(axis=1) + d
    hist, edges = np.histogram(y, bins=120, range=(0, s_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens_mc = np.interp(centers, s_grid, dens_an)

    # 比较(在 MC 有支撑处)
    mask = hist > 0.02 * hist.max()
    l1 = float(np.mean(np.abs(dens_mc[mask] - hist[mask])))
    rel = l1 / float(hist[mask].mean())
    print(f"[{kind}] 解析 vs MC 边缘密度: 平均|Δ|={l1:.4f}  相对={rel*100:.1f}%  "
          f"(MC 均值峰={hist.max():.2f})", flush=True)
    return rel


def main():
    print("=" * 60)
    r_u = run_one("uniform")
    r_l = run_one("logskewnorm")
    print("-" * 60)
    ok = (r_u < 0.10) and (r_l < 0.12)
    print(f"{'PASS' if ok else 'FAIL'}: uniform 相对误差={r_u*100:.1f}%  "
          f"logskewnorm 相对误差={r_l*100:.1f}%  (阈值 ~10-12%, 采样噪声内)")


if __name__ == "__main__":
    main()
