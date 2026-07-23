"""诊断解析 target 与解析 L2 fit 的相关传播。

本脚本不生成正式 attempt；它复用 seed=0 / dense_dag_r567 配置，
逐层比较同一个上层 forest 下下一层的三种相关：

1. truth: 真实 test_x 的 L4 相关；
2. sampled_prop: 从 L3 forest 采样再用 max-plus 传播得到的经验相关；
3. analytic: `UpperForest.pair_cdf` + Hoeffding covariance 得到的解析相关。

每层 fit 完之后还会从拟合好的 forest 采样，检查解析 L2 fit 是否保住了该层相关。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.analytic_tree_fit import analytic_block_target, structural_blocks  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_remedy_tests import (  # noqa: E402
    _fit_analytic_ttns_nonneg,
    prepare_data,
)
from simple_ttns_l2.layered_forest import BlockModel, sample_forest  # noqa: E402
from simple_ttns_l2.maxplus_cdf import cov_hoeffding, moments_from_marginal  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import UpperForest  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import propagate_layer  # noqa: E402

AR_ROOT = Path(__file__).resolve().parent


def _fit_nonneg_until_l3(forest0, spec, params, key, s_max0, cfg):
    forests = {0: forest0}
    s_max = s_max0
    for li in range(1, 4):
        s_max = s_max + (params.edge_hi + params.node_hi) + 0.3
        upper = UpperForest(forests[li - 1], q_grid=400)
        layer_nodes = list(spec.layers[li])
        blocks = structural_blocks(spec, li, mode=cfg.get("block_mode", "immediate"))
        forest = []
        for bi, blk in enumerate(blocks):
            gids = [layer_nodes[i] for i in blk]
            target = analytic_block_target(
                upper, spec, gids, params, s_max,
                n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"], use_mi=True,
            )
            k_b, key = jax.random.split(key)
            ttns, bases = _fit_analytic_ttns_nonneg(
                target, k_b, cfg["q"], cfg["m"], cfg["rank"],
                cfg["an_lr"], cfg["an_steps"], cfg["init_noise"],
                label=f"audit_L{li}.b{bi}",
            )
            forest.append(BlockModel(
                tuple(blk), tuple(int(g) for g in gids), tuple(target.parent), ttns, bases,
            ))
        forests[li] = forest
        print(f"[audit] fitted nonneg layer {li}", flush=True)
    return forests, s_max, key


def _strongest_pair(corr: np.ndarray):
    best = (-1.0, 0, min(1, corr.shape[0] - 1))
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            val = abs(float(corr[i, j]))
            if val > best[0]:
                best = (val, i, j)
    return best[1], best[2], float(corr[best[1], best[2]])


def _analytic_corr_block(upper: UpperForest, spec, gids, params, s_max, n_s_pair):
    K = len(gids)
    sp = np.linspace(0.0, s_max, n_s_pair)
    parents = {v: list(spec.parents(v)) for v in gids}
    Fp = {v: upper.marginal_cdf(parents[v], sp, params) for v in gids}
    var = {v: max(moments_from_marginal(Fp[v], sp)[1], 1e-12) for v in gids}
    corr = np.eye(K)
    for a in range(K):
        for b in range(a + 1, K):
            Fvw = upper.pair_cdf(parents[gids[a]], parents[gids[b]], sp, sp, params)
            cov = cov_hoeffding(Fvw, Fp[gids[a]], Fp[gids[b]], sp, sp)
            corr[a, b] = corr[b, a] = float(cov / np.sqrt(var[gids[a]] * var[gids[b]]))
    return corr


def _audit_layer(forest, spec, params, layers, test_x, key, li, s_max, cfg, n_sample):
    upper = UpperForest(forest, q_grid=400)
    layer_nodes = list(spec.layers[li])
    blocks = structural_blocks(spec, li, mode="immediate")

    key, k_s, k_rng = jax.random.split(key, 3)
    upper_samples = np.asarray(sample_forest(forest, k_s, n_sample, grid_size=400))
    rng = np.random.default_rng(int(jax.random.randint(k_rng, (), 0, 2**31 - 1)))
    sampled_layer = propagate_layer(spec, li, upper_samples, params, rng)
    truth_layer = test_x[:n_sample, layers[li]]

    rows = []
    for bi, blk in enumerate(blocks):
        gids = [layer_nodes[i] for i in blk]
        analytic_corr = _analytic_corr_block(upper, spec, gids, params, s_max, cfg["n_s_pair"])
        sampled_corr = np.corrcoef(sampled_layer[:, blk].T)
        truth_corr = np.corrcoef(truth_layer[:, blk].T)
        target = analytic_block_target(
            upper, spec, gids, params, s_max,
            n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"], use_mi=True,
        )

        i, j, r_truth = _strongest_pair(truth_corr)
        rows.append({
            "layer": li,
            "block": bi,
            "local_vars": [int(x) for x in blk],
            "global_vars": [int(x) for x in gids],
            "tree_parent": [int(x) for x in target.parent],
            "truth_strongest_pair": [int(gids[i]), int(gids[j])],
            "truth_r": r_truth,
            "sampled_prop_r_at_truth_pair": float(sampled_corr[i, j]),
            "analytic_r_at_truth_pair": float(analytic_corr[i, j]),
            "sampled_minus_analytic": float(sampled_corr[i, j] - analytic_corr[i, j]),
            "truth_corr": truth_corr.tolist(),
            "sampled_prop_corr": sampled_corr.tolist(),
            "analytic_corr": analytic_corr.tolist(),
        })
    return rows, key


def _audit_fitted_forest(forest, layers, test_x, key, li, n_sample):
    key, k_s = jax.random.split(key)
    samples = np.asarray(sample_forest(forest, k_s, n_sample, grid_size=400))
    truth_layer = test_x[:n_sample, layers[li]]
    rows = []
    for bi, bm in enumerate(forest):
        blk = list(bm.local_vars)
        gids = [int(g) for g in bm.global_vars]
        sampled_corr = np.corrcoef(samples[:, blk].T)
        truth_corr = np.corrcoef(truth_layer[:, blk].T)
        i, j, r_truth = _strongest_pair(truth_corr)
        rows.append({
            "layer": li,
            "block": bi,
            "global_vars": gids,
            "tree_parent": [int(x) for x in bm.parent],
            "truth_strongest_pair": [int(gids[i]), int(gids[j])],
            "truth_r": r_truth,
            "fitted_sample_r_at_truth_pair": float(sampled_corr[i, j]),
            "truth_corr": truth_corr.tolist(),
            "fitted_sample_corr": sampled_corr.tolist(),
        })
    return rows, key


def _fit_next_layer(forest_prev, spec, params, key, s_max, cfg, li):
    upper = UpperForest(forest_prev, q_grid=400)
    layer_nodes = list(spec.layers[li])
    blocks = structural_blocks(spec, li, mode=cfg.get("block_mode", "immediate"))
    forest = []
    for bi, blk in enumerate(blocks):
        gids = [layer_nodes[i] for i in blk]
        target = analytic_block_target(
            upper, spec, gids, params, s_max,
            n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"], use_mi=True,
        )
        k_b, key = jax.random.split(key)
        ttns, bases = _fit_analytic_ttns_nonneg(
            target, k_b, cfg["q"], cfg["m"], cfg["rank"],
            cfg["an_lr"], cfg["an_steps"], cfg["init_noise"],
            label=f"audit_L{li}.b{bi}",
        )
        forest.append(BlockModel(
            tuple(blk), tuple(int(g) for g in gids), tuple(target.parent), ttns, bases,
        ))
    return forest, key


def run_audit(
    seed: int,
    out_path: Path,
    n_sample: int,
    an_steps: int,
    rank: int,
    init_noise: float | None,
    max_layer: int,
):
    cfg = dict(CFG)
    cfg.update(an_steps=an_steps, rank=rank, block_mode="immediate")
    if init_noise is not None:
        cfg["init_noise"] = float(init_noise)
    t0 = time.perf_counter()
    key, spec, params, layers, train_x, test_x, forest0, s_max0 = prepare_data(seed, cfg)
    del train_x

    forests = {0: forest0}
    s_max = s_max0
    all_rows = []
    fit_rows = []
    for li in range(1, max_layer + 1):
        s_max = s_max + (params.edge_hi + params.node_hi) + 0.3
        layer_rows, key = _audit_layer(
            forests[li - 1], spec, params, layers, test_x, key, li, s_max, cfg, n_sample,
        )
        all_rows.extend(layer_rows)
        print(f"[audit] compared layer {li}", flush=True)
        if li < max_layer:
            forest, key = _fit_next_layer(forests[li - 1], spec, params, key, s_max, cfg, li)
            forests[li] = forest
            print(f"[audit] fitted nonneg layer {li}", flush=True)
            rows_fit, key = _audit_fitted_forest(forest, layers, test_x, key, li, n_sample)
            fit_rows.extend(rows_fit)
            print(f"[audit] sampled fitted layer {li}", flush=True)

    if max_layer not in forests and max_layer > 0:
        # `max_layer` 本身也需要 fit-quality 诊断时，补拟合该层但不再继续传播。
        forest, key = _fit_next_layer(forests[max_layer - 1], spec, params, key, s_max, cfg, max_layer)
        forests[max_layer] = forest
        print(f"[audit] fitted nonneg layer {max_layer}", flush=True)
        rows_fit, key = _audit_fitted_forest(forest, layers, test_x, key, max_layer, n_sample)
        fit_rows.extend(rows_fit)
        print(f"[audit] sampled fitted layer {max_layer}", flush=True)

    rows = [row for row in all_rows if row["layer"] == max_layer]
    dump = {
        "seed": seed,
        "cfg": {
            "rank": rank,
            "an_steps": an_steps,
            "init_noise": cfg["init_noise"],
            "n_sample": n_sample,
            "max_layer": max_layer,
            "block_mode": "immediate",
            "n_s": cfg["n_s"],
            "n_s_pair": cfg["n_s_pair"],
        },
        "elapsed_seconds": time.perf_counter() - t0,
        "layer_rows": all_rows,
        "fit_rows": fit_rows,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}", flush=True)
    for row in rows:
        print(
            f"[block {row['block']}] pair={row['truth_strongest_pair']} "
            f"truth={row['truth_r']:+.3f} sampled={row['sampled_prop_r_at_truth_pair']:+.3f} "
            f"analytic={row['analytic_r_at_truth_pair']:+.3f}",
            flush=True,
        )
    for row in fit_rows:
        print(
            f"[fit L{row['layer']}.b{row['block']}] pair={row['truth_strongest_pair']} "
            f"truth={row['truth_r']:+.3f} fitted={row['fitted_sample_r_at_truth_pair']:+.3f}",
            flush=True,
        )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sample", type=int, default=5000)
    ap.add_argument("--an-steps", type=int, default=1200)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--init-noise", type=float, default=None)
    ap.add_argument("--max-layer", type=int, default=4)
    ap.add_argument("--out", default=str(AR_ROOT / "artifacts" / "target_audit_l4.json"))
    args = ap.parse_args()
    run_audit(
        args.seed, Path(args.out), args.n_sample, args.an_steps,
        args.rank, args.init_noise, args.max_layer,
    )


if __name__ == "__main__":
    main()
