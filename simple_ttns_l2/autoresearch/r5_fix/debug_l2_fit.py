"""定位解析块内 L2 拟合问题。

只诊断 seed=0 的 L1 block0：

1. 构造同一个 `analytic_block_target`；
2. 分别得到 rank-1 独立初始化、解析非负 L2 拟合模型、解析 target 采样 + MLE 模型；
3. 对三者比较解析 L2 loss、target 样本 MC L2 loss、模型采样相关。

若 MLE 模型在 MC loss 上好、但解析 loss 上不好，说明 `_cross_term_fn` 或积分轴约定有 bug。
若 MLE 模型在两种 loss 上都好，而解析 L2 拟合模型不好，说明主要是优化/参数化问题。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from ttde.ttns.ttns_opt import TTNSOpt, quadratic_form_ttns  # noqa: E402

from simple_ttns_l2.analytic_tree_fit import (  # noqa: E402
    _build_layer_bases,
    _cross_term_fn,
    _fit_analytic_ttns,
    _init_rank1,
    analytic_block_target,
    structural_blocks,
)
from simple_ttns_l2.experiments.dense_dag_r567_remedy_tests import (  # noqa: E402
    _fit_analytic_ttns_nonneg,
    _sample_analytic_tree_target,
    prepare_data,
    train_tree_nonneg_mle,
)
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG  # noqa: E402
from simple_ttns_l2.layered_forest import sample_forest  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import UpperForest  # noqa: E402
from simple_ttns_l2.objective import (  # noqa: E402
    batch_basis_vectors_from_samples,
    batch_eval_q_ttns,
    integral_q_ttns,
    normalize_ttns_by_integral,
)
from simple_ttns_l2.train_l2 import build_bases  # noqa: E402
from simple_ttns_l2.ttns_sampler import _basis_eval_dim, sample_ttns  # noqa: E402

AR_ROOT = Path(__file__).resolve().parent


def _corr_at_truth_pair(samples: np.ndarray, truth_pair: tuple[int, int]) -> float:
    return float(np.corrcoef(samples[:, truth_pair[0]], samples[:, truth_pair[1]])[0, 1])


def _strongest_pair(x: np.ndarray) -> tuple[int, int, float]:
    corr = np.corrcoef(x.T)
    best = (-1.0, 0, 1)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            val = abs(float(corr[i, j]))
            if val > best[0]:
                best = (val, i, j)
    return best[1], best[2], float(corr[best[1], best[2]])


def _analytic_loss_parts(ttns, bases, parent, target):
    Bg = [_basis_eval_dim(bases, v, jnp.asarray(target.s_grid)) for v in range(len(parent))]
    gram = vmap(type(bases).l2_integral)(bases)
    basis_int = vmap(type(bases).integral)(bases)
    delta = float(target.s_grid[1] - target.s_grid[0]) if len(target.s_grid) > 1 else 1.0
    pcond_j = {v: jnp.asarray(target.pcond[v]) for v in target.pcond}
    root = next(i for i, p in enumerate(parent) if p == -1 or p == i)
    cross_fn = _cross_term_fn(parent, Bg, pcond_j, jnp.asarray(target.p_marg[root]), delta)
    int_q2 = quadratic_form_ttns(ttns, gram, parent)
    cross = cross_fn(list(ttns.cores))
    z = integral_q_ttns(ttns, basis_int, parent)
    return {
        "z": float(z),
        "int_q2": float(int_q2),
        "cross_analytic": float(cross),
        "l2_analytic": float(int_q2 - 2.0 * cross),
    }


def _mc_loss_parts(ttns, bases, parent, target_samples: np.ndarray):
    gram = vmap(type(bases).l2_integral)(bases)
    int_q2 = quadratic_form_ttns(ttns, gram, parent)
    bv = batch_basis_vectors_from_samples(bases, jnp.asarray(target_samples))
    q = batch_eval_q_ttns(ttns, bv, parent)
    return {
        "cross_mc": float(jnp.mean(q)),
        "q_mean": float(jnp.mean(q)),
        "q_min": float(jnp.min(q)),
        "q_p01": float(jnp.percentile(q, 1)),
        "l2_mc": float(int_q2 - 2.0 * jnp.mean(q)),
    }


def run_debug(
    out_path: Path,
    seed: int,
    steps: int,
    n_target: int,
    n_eval: int,
    lrs: list[float],
):
    cfg = dict(CFG)
    cfg.update(an_steps=steps, block_mode="immediate")
    t0 = time.perf_counter()
    key, spec, params, layers, train_x, test_x, forest0, s_max0 = prepare_data(seed, cfg)
    del train_x

    s_max_l1 = s_max0 + (params.edge_hi + params.node_hi) + 0.3
    upper = UpperForest(forest0, q_grid=400)
    layer_nodes = list(spec.layers[1])
    blk = structural_blocks(spec, 1, mode="immediate")[0]
    gids = [layer_nodes[i] for i in blk]
    target = analytic_block_target(
        upper, spec, gids, params, s_max_l1,
        n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"], use_mi=True,
    )
    truth = np.asarray(test_x[:n_eval, layers[1]])[:, blk]
    pi, pj, truth_r = _strongest_pair(truth)

    key, k_eval, k_l2, k_mle, k_s1, k_s2, k_s3 = jax.random.split(key, 7)
    target_eval = _sample_analytic_tree_target(target, k_eval, n_eval)

    # rank-1 独立初始化，使用解析 L2 的同一 bases。
    bases_rank1 = _build_layer_bases(target.s_grid, len(target.parent), cfg["q"], cfg["m"])
    gram_rank1 = vmap(type(bases_rank1).l2_integral)(bases_rank1)
    basis_int_rank1 = vmap(type(bases_rank1).integral)(bases_rank1)
    Bg_rank1 = [_basis_eval_dim(bases_rank1, v, jnp.asarray(target.s_grid)) for v in range(len(target.parent))]
    t_rank1 = _init_rank1(target, bases_rank1, gram_rank1, Bg_rank1, cfg["rank"], k_s1, cfg["init_noise"])
    t_rank1, _ = normalize_ttns_by_integral(t_rank1, basis_int_rank1, target.parent)

    # 线性解析 L2：检查解析 L2 消息本身能否驱动相关（允许负密度）。
    t_linear, bases_linear = _fit_analytic_ttns(
        target, k_l2, cfg["q"], cfg["m"], cfg["rank"],
        cfg["lr"], steps, cfg["init_noise"], log_every=max(steps // 5, 1),
        label="debug_linear_L1.b0", normalize_every=1,
    )

    # 008 路径：同一解析 target 采样后 MLE。
    target_train = _sample_analytic_tree_target(target, k_mle, n_target)
    bases_mle = build_bases(jnp.asarray(target_train), cfg["q"], cfg["m"])
    split = int(0.85 * target_train.shape[0])
    cfg_mle = dict(cfg)
    cfg_mle.update(nn_steps=400, nn_batch_sz=512, nn_patience=6, nn_lr=0.002)
    t_mle, bases_mle = train_tree_nonneg_mle(
        jnp.asarray(target_train[:split]),
        jnp.asarray(target_train[split:]),
        bases_mle,
        target.parent,
        cfg["rank"],
        cfg_mle,
        k_s2,
        label="debug_mle_L1.b0",
    )

    models = {
        "rank1_init": (t_rank1, bases_rank1),
        "analytic_l2_linear": (t_linear, bases_linear),
        "target_sample_mle": (t_mle, bases_mle),
    }
    for idx, lr in enumerate(lrs):
        key, k_lr = jax.random.split(key)
        t_l2, bases_l2 = _fit_analytic_ttns_nonneg(
            target, k_lr, cfg["q"], cfg["m"], cfg["rank"],
            lr, steps, cfg["init_noise"], label=f"debug_nonneg_lr{lr:g}",
        )
        models[f"analytic_l2_nonneg_lr{lr:g}"] = (t_l2, bases_l2)

    rows = {}
    for name, (ttns, bases) in models.items():
        key, key_s = jax.random.split(key)
        samples = np.asarray(sample_ttns(ttns, bases, target.parent, key_s, n_eval, grid_size=400))
        rows[name] = {
            **_analytic_loss_parts(ttns, bases, target.parent, target),
            **_mc_loss_parts(ttns, bases, target.parent, target_eval),
            "sample_r_at_truth_pair": _corr_at_truth_pair(samples, (pi, pj)),
            "sample_std_mean": float(samples.std(axis=0).mean()),
        }

    dump = {
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - t0,
        "block": 0,
        "global_vars": [int(g) for g in gids],
        "parent": [int(p) for p in target.parent],
        "truth_pair_local": [int(pi), int(pj)],
        "truth_pair_global": [int(gids[pi]), int(gids[pj])],
        "truth_r": truth_r,
        "target_sample_r": _corr_at_truth_pair(target_eval, (pi, pj)),
        "lrs": [float(x) for x in lrs],
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dump, indent=2, ensure_ascii=False), flush=True)
    print(f"saved: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--n-target", type=int, default=20000)
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--lrs", default="3e-5,3e-4,1e-3,2e-3")
    ap.add_argument(
        "--out",
        default=str(AR_ROOT / "artifacts" / "l2_fit_debug_l1_block0.json"),
    )
    args = ap.parse_args()
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    run_debug(Path(args.out), args.seed, args.steps, args.n_target, args.n_eval, lrs)


if __name__ == "__main__":
    main()
