"""大例子(100 节点)全局基线补测:仅拟合全局 TT / TTNS / TTDE,不重跑三条链。

数据与 `sampled_vs_analytic_chain --big` 完全一致(同 spec/源/seed/n_total,test_x 逐点相同),
故本表的每层 joint_LL@truth、corr_fro 可直接与已跑出的链结果(R5/R7)拼表对比。

每层指标口径与链脚本一致:
- joint_LL@truth: 该层联合密度在 test_x[:,layer] 上平均对数密度(全局模型对非本层维用基积分积掉);
- corr_fro: 模型采样层内相关矩阵 vs 真值 Frobenius 误差,3 次采样平均(n_sample)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from ttde.score.models.opt_for_tree_data import chain_parent  # noqa: E402

from simple_ttns_l2.train_l2 import build_bases  # noqa: E402
from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_clustered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.ttns_sampler import sample_ttns  # noqa: E402
from simple_ttns_l2.experiments.per_layer_all_methods import (  # noqa: E402
    complex_sources, linear_block_logp, CFG_BIG,
)
from simple_ttns_l2.experiments.fit_layered_vs_flat_tt import fit_flat  # noqa: E402
from simple_ttns_l2.experiments.ttde_tt_baseline import fit_ttde_tt, ttde_logp, sample_ttde_tt  # noqa: E402
from simple_ttns_l2.experiments.per_layer_compare import build_ttde_envs, ttde_block_logp  # noqa: E402

BASE_MODELS = ["global_TT", "global_TTNS", "global_TTDE"]


def corr_fro(sample_fn, tev, key, n, n_rep=3):
    Ct = np.corrcoef(tev.T) if tev.shape[1] > 1 else np.array([[1.0]])
    fros = []
    for _ in range(n_rep):
        key, k = jax.random.split(key)
        s = sample_fn(k, n)
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
    n = xs.shape[0]
    n_dims = xs.shape[1]
    n_tr = int(0.7 * n)
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    n_sample = cfg["n_sample"]
    timings = {}

    bases = build_bases(jnp.asarray(train_x), cfg["q"], cfg["m"])
    gram = vmap(type(bases).l2_integral)(bases)
    basis_integrals = vmap(type(bases).integral)(bases)
    tr_j = jnp.asarray(train_x[:int(0.85 * n_tr)])
    val_j = jnp.asarray(train_x[int(0.85 * n_tr):])
    flat_specs = {
        "global_TT": [int(p) for p in chain_parent(n_dims)],
        "global_TTNS": [int(p) for p in estimate_chow_liu_tree(train_x, n_bins=16, root=0).parent],
    }
    flat_models = {}
    for name, parent in flat_specs.items():
        t0 = time.perf_counter()
        k_ff, key = jax.random.split(key)
        ttns, parent, r, np_ = fit_flat(parent, name, tr_j, val_j, bases, gram, basis_integrals,
                                        cfg["budget"], cfg, k_ff)
        flat_models[name] = (ttns, list(parent), r, np_)
        timings[name] = time.perf_counter() - t0
        print(f"[{name}] rank={r} params={np_} 用时 {timings[name]:.1f}s", flush=True)

    # ---- 全局 TTDE ----
    t0 = time.perf_counter()
    n_ttde = min(cfg["ttde_n_train"], n_tr)
    ttde_tr = train_x[:n_ttde]
    ttde_val = train_x[n_ttde:n_ttde + cfg["monitor_val_sz"]]
    ttde_cfg = {**cfg, "init_noise": cfg["ttde_init_noise"]}
    ttde_model, ttde_params, ttde_info = fit_ttde_tt(ttde_tr, ttde_val, ttde_cfg, cfg["seed"])
    cores, gram_t, env, lenv, Z = build_ttde_envs(ttde_params, ttde_model.bases)
    timings["global_TTDE"] = time.perf_counter() - t0
    print(f"[global_TTDE] params={ttde_info.get('learned_params', -1)} "
          f"用时 {timings['global_TTDE']:.1f}s", flush=True)

    # ---- 采样函数 ----
    samplers = {}
    for name, (ttns, parent, _, _) in flat_models.items():
        samplers[name] = (lambda t, p: (lambda k, m: np.asarray(
            sample_ttns(t, bases, p, k, m, grid_size=400))))(ttns, parent)
    samplers["global_TTDE"] = lambda k, m: np.asarray(
        sample_ttde_tt(ttde_model, ttde_params, k, m, grid_size=cfg["ttde_grid"]))

    # ---- 逐层指标 ----
    rows = []
    for li in range(len(layers)):
        Lg = layers[li]
        tev = test_x[:, Lg]
        row = {"li": li, "K": len(Lg), "ll": {}, "fro": {}, "sd": {}}
        for name, (ttns, parent, _, _) in flat_models.items():
            row["ll"][name] = float(np.asarray(linear_block_logp(ttns, bases, parent, Lg, tev)).mean())
        row["ll"]["global_TTDE"] = float(np.asarray(
            ttde_block_logp(cores, gram_t, env, lenv, Z, ttde_model.bases, Lg, tev)).mean())
        for name in BASE_MODELS:
            k_m, key = jax.random.split(key)
            fn = (lambda nm: (lambda k, m: samplers[nm](k, m)[:, Lg]))(name)
            row["fro"][name], row["sd"][name] = corr_fro(fn, tev, k_m, n_sample)
        rows.append(row)
        print(f"[layer {li}] done", flush=True)

    # 全联合 LL
    fj_ll = {}
    from simple_ttns_l2.experiments.fit_layered_vs_flat_tt import flat_joint_loglik
    for name, (ttns, parent, _, _) in flat_models.items():
        fj_ll[name], _ = flat_joint_loglik(ttns, parent, bases, test_x)
    fj_ll["global_TTDE"] = float(ttde_logp(ttde_model, ttde_params, test_x).mean())

    return rows, timings, fj_ll, {**{n: flat_models[n][3] for n in flat_models},
                                  "global_TTDE": ttde_info.get("learned_params", -1)}


def print_table(rows, timings, fj_ll, nparams):
    print("\n" + "=" * 78)
    print("全局基线(100 节点)—— 与已跑出的 R5/R7 链同数据同口径")
    print("=" * 78)
    print("\njoint_LL @truth (↑):")
    hdr = "层  " + "".join(f"{m:>15}" for m in BASE_MODELS)
    print(hdr)
    for r in rows:
        print(f"L{r['li']:<2}" + "".join(f"{r['ll'][m]:>15.4f}" for m in BASE_MODELS))
    print("\ncorr_fro vs truth (↓, 3次采样均值):")
    print(hdr)
    for r in rows:
        print(f"L{r['li']:<2}" + "".join(f"{r['fro'][m]:>15.4f}" for m in BASE_MODELS))
    print("\n全联合 joint_LL(↑):")
    for m in BASE_MODELS:
        print(f"  {m:<14}{fj_ll[m]:>12.4f}   (params={nparams[m]})")
    print("\n用时(s):")
    for k, v in timings.items():
        print(f"  {k:<14}{v:>8.1f}")
    print("=" * 78)


def main():
    # 只降"模型拟合"超参(不动 spec/源/seed/n_total → test_x 与链脚本逐点相同, 可拼表)
    cfg = {**CFG_BIG, "n_sample": 16000,
           "steps": 300, "budget": 150000, "rmax": 24,
           "ttde_rank": 12, "ttde_steps": 600, "ttde_em_steps": 8, "ttde_n_train": 8000}
    t0 = time.perf_counter()
    rows, timings, fj_ll, nparams = run(cfg)
    print_table(rows, timings, fj_ll, nparams)
    print(f"总用时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
