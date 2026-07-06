"""精简等预算基线:只算 joint_LL(逐层 + 全联合),**不做任何采样**(采样是慢点/内存点)。

用于答"等预算下分层链是否胜全局 TT/TTNS/TTDE"(判据#2)。budget = 分层链最大点参数量。
全局模型 corr_fro 已知很差(文档 §6.4, ~2-9),此处不再采样,只比主指标 joint_LL。
复用 baselines_big / per_layer_all_methods 的拟合与 LL 函数,不改原文件。

运行: env -u PYTHONPATH python3 -u -m simple_ttns_l2.experiments.budget_baselines_ll
"""
from __future__ import annotations

import json
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
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources, linear_block_logp, CFG_BIG  # noqa: E402
from simple_ttns_l2.experiments.fit_layered_vs_flat_tt import fit_flat, flat_joint_loglik  # noqa: E402
from simple_ttns_l2.experiments.ttde_tt_baseline import fit_ttde_tt, ttde_logp  # noqa: E402
from simple_ttns_l2.experiments.per_layer_compare import build_ttde_envs, ttde_block_logp  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def main():
    # 等预算目标:读分层链 JSON 里的最大参数量点
    sweep = json.loads((REPORTS / "budget_sweep_layered_metrics.json").read_text())
    best_np, best_lab = 0, None
    for lab, pt in sweep["points"].items():
        s0 = pt["seeds"][0]
        np_ = s0.get("params_r7") or s0.get("params_r5") or 0
        if np_ > best_np:
            best_np, best_lab = np_, lab
    budget = int(best_np)

    cfg = {**CFG_BIG, "seed": 0, "budget": budget, "rmax": 60}
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    layers = [list(l) for l in spec.layers]
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0]); n_dims = xs.shape[1]
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    print(f"[budget] {budget}(=最大分层链点 '{best_lab}');节点={n_dims} 层={len(layers)}", flush=True)

    bases = build_bases(jnp.asarray(train_x), cfg["q"], cfg["m"])
    gram = vmap(type(bases).l2_integral)(bases)
    basis_integrals = vmap(type(bases).integral)(bases)
    tr_j = jnp.asarray(train_x[:int(0.85 * n_tr)]); val_j = jnp.asarray(train_x[int(0.85 * n_tr):])

    flat_specs = {
        "global_TT": [int(p) for p in chain_parent(n_dims)],
        "global_TTNS": [int(p) for p in estimate_chow_liu_tree(train_x, n_bins=16, root=0).parent],
    }
    flat_models, nparams = {}, {}
    for name, parent in flat_specs.items():
        t0 = time.perf_counter()
        k_ff, key = jax.random.split(key)
        ttns, parent, r, np_ = fit_flat(parent, name, tr_j, val_j, bases, gram, basis_integrals,
                                        budget, cfg, k_ff)
        flat_models[name] = (ttns, list(parent)); nparams[name] = np_
        print(f"[{name}] rank={r} params={np_} 用时 {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    n_ttde = min(cfg["ttde_n_train"], n_tr)
    ttde_tr = train_x[:n_ttde]; ttde_val = train_x[n_ttde:n_ttde + cfg["monitor_val_sz"]]
    ttde_cfg = {**cfg, "init_noise": cfg["ttde_init_noise"]}
    ttde_model, ttde_params, ttde_info = fit_ttde_tt(ttde_tr, ttde_val, ttde_cfg, cfg["seed"])
    cores, gram_t, env, lenv, Z = build_ttde_envs(ttde_params, ttde_model.bases)
    nparams["global_TTDE"] = ttde_info.get("learned_params", -1)
    print(f"[global_TTDE] params={nparams['global_TTDE']} 用时 {time.perf_counter()-t0:.1f}s", flush=True)

    # 逐层 joint_LL@truth(无采样)
    per_layer = []
    for li, Lg in enumerate(layers):
        tev = test_x[:, Lg]
        row = {"li": li, "K": len(Lg), "ll": {}}
        for name, (ttns, parent) in flat_models.items():
            row["ll"][name] = float(np.asarray(linear_block_logp(ttns, bases, parent, Lg, tev)).mean())
        row["ll"]["global_TTDE"] = float(np.asarray(
            ttde_block_logp(cores, gram_t, env, lenv, Z, ttde_model.bases, Lg, tev)).mean())
        per_layer.append(row)
        print(f"[layer {li}] " + "  ".join(f"{k}={v:.3f}" for k, v in row["ll"].items()), flush=True)

    # 全联合 joint_LL
    fj = {}
    for name, (ttns, parent) in flat_models.items():
        fj[name], _ = flat_joint_loglik(ttns, parent, bases, test_x)
    fj["global_TTDE"] = float(ttde_logp(ttde_model, ttde_params, test_x).mean())
    print("\n全联合 joint_LL:", "  ".join(f"{k}={v:.3f}(p={nparams[k]})" for k, v in fj.items()), flush=True)

    out = dict(budget=budget, budget_from=best_lab, nparams=nparams,
               per_layer=per_layer, full_joint=fj)
    p = REPORTS / "budget_baselines_ll_metrics.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"[写出] {p}", flush=True)


if __name__ == "__main__":
    main()
