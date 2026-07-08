"""TTNSDE(TTNS/MI 树)参数量扫描:降 rank 看 test_LL 如何变化。

同一份 clustered 18 维数据(与 ttde_ttns_vs_tt 一致)。TTNS 平方参数量 = m·Σ_v r^deg(v),
hub 节点的 r^deg 主导 → 降 rank 急剧降参数量。逐 rank 记录 params 与 test/train LL,
并以 TT(链) rank=8 作参照,判断"更少参数能否保持结构优势"。
"""
from __future__ import annotations

import sys
import time
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
from simple_ttns_l2.experiments.ttde_tt_baseline import fit_ttde_tt, fit_ttde_ttns, ttde_logp  # noqa: E402
from simple_ttns_l2.experiments.ttde_ttns_vs_tt import ttns_params, tt_params, tree_degrees  # noqa: E402
from ttde.score.models.opt_for_tree_data import chain_parent  # noqa: E402


def main():
    cfg = dict(
        n_layers=3, clusters=[3, 3], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
        n_total=20000, q=2, m=24, src_sigma=0.03,
        lr=2e-3, batch_sz=512, train_noise=1e-3, log_every=200, seed=0,
        ttde_steps=800, ttde_em_steps=12, ttde_patience=8,
        monitor_val_sz=2000, ttde_n_train=10000, ttde_init_noise=0.03, ttde_grad_clip=10.0,
    )
    ranks = [3, 4]  # rank3≈14k(<TT8的25k), rank4≈54k; 对齐/低于 TTDE(TT链r=8) 参数量

    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_dims = xs.shape[1]
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    n_ttde = min(cfg["ttde_n_train"], n_tr)
    tr, val = train_x[:n_ttde], train_x[n_ttde:n_ttde + cfg["monitor_val_sz"]]
    ttde_cfg = {**cfg, "init_noise": cfg["ttde_init_noise"]}

    mi_tree = [int(p) for p in estimate_chow_liu_tree(train_x, n_bins=16, root=0).parent]
    chain = [int(p) for p in chain_parent(n_dims)]
    print(f"n_dims={n_dims}  MI树度分布={tree_degrees(mi_tree)}", flush=True)
    for r in ranks:
        print(f"  预计 TTNS(r={r}) 参数量 = {ttns_params(mi_tree, cfg['m'], r):,}", flush=True)

    rows = []
    # 参照: TT(链) rank=8
    t0 = time.perf_counter()
    m_tt, p_tt, info_tt = fit_ttde_tt(tr, val, {**ttde_cfg, "ttde_rank": 8}, cfg["seed"])
    rows.append(dict(name="TT(chain) r=8", params=info_tt["learned_params"],
                     ll_test=float(ttde_logp(m_tt, p_tt, test_x).mean()),
                     ll_train=float(ttde_logp(m_tt, p_tt, tr).mean()), sec=time.perf_counter() - t0))
    print(f"[TT r=8] done params={info_tt['learned_params']}", flush=True)

    for r in ranks:
        t0 = time.perf_counter()
        model, mp, info = fit_ttde_ttns(tr, val, {**ttde_cfg, "ttde_rank": r}, cfg["seed"], mi_tree)
        rows.append(dict(name=f"TTNS(MI) r={r}", params=info["learned_params"],
                         ll_test=float(ttde_logp(model, mp, test_x).mean()),
                         ll_train=float(ttde_logp(model, mp, tr).mean()),
                         sec=time.perf_counter() - t0))
        print(f"[TTNS r={r}] done test_LL={rows[-1]['ll_test']:.4f} "
              f"params={info['learned_params']}", flush=True)

    print("\n" + "=" * 68)
    print("TTNSDE 参数量扫描: 降 rank 看 test_LL (clustered 18-dim)")
    print("=" * 68)
    print(f"{'model':<18}{'params':>12}{'test_LL(↑)':>12}{'train_LL':>12}{'sec':>8}")
    for r in rows:
        print(f"{r['name']:<18}{r['params']:>12,}{r['ll_test']:>12.4f}{r['ll_train']:>12.4f}{r['sec']:>8.1f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
