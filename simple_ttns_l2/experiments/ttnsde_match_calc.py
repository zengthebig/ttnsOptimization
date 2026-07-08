"""零成本测算: 把 TTNS(MI树) 参数量压到 TTDE(TT链 rank=8) 同量级, 对应 rank 是多少。

仅生成数据 + 建 MI 树, 不做任何 TTDE 拟合。
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_clustered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources  # noqa: E402
from simple_ttns_l2.experiments.ttde_ttns_vs_tt import ttns_params, tt_params, tree_degrees  # noqa: E402


def main():
    cfg = dict(n_layers=3, clusters=[3, 3], fanin=2,
               delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
               n_total=20000, m=24, src_sigma=0.03, seed=0)
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_dims = xs.shape[1]
    train_x = xs[:int(0.7 * xs.shape[0])]
    mi_tree = [int(p) for p in estimate_chow_liu_tree(train_x, n_bins=16, root=0).parent]
    m = cfg["m"]

    p_tt8 = tt_params(n_dims, m, 8)
    print(f"n_dims={n_dims}  m={m}")
    print(f"MI树 度分布 = {tree_degrees(mi_tree)}  (max deg={max(tree_degrees(mi_tree))})")
    print(f"目标: TTDE 即 TT(链) rank=8 参数量 = {p_tt8:,}\n")
    print(f"{'rank':>5}{'TTNS(MI)参数量':>18}{'相对TT8':>12}")
    for r in range(1, 9):
        p = ttns_params(mi_tree, m, r)
        print(f"{r:>5}{p:>18,}{p / p_tt8:>11.2f}x")


if __name__ == "__main__":
    main()
