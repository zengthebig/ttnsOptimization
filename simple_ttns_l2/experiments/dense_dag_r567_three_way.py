"""更密连接 / 多父 + 更多节点 / 更深层 的复杂 DAG：三方(全局TT / 全局TTNS / 分层TTNS) × R5/R6/R7。

相对旧的 `sampled_vs_analytic_chain --big`(5 层、fanin=2、簇[2,3,4,5,6]=100 节点)本脚本把图加密加深:
  - **更密 / 多父**：fanin=3(簇大小 ≥3 时 = 簇内全连接，制造真实非树相关/环)；
  - **更多节点 / 更深层**：n_layers=6、clusters=[3,3,4,4,4]=18 维/层 → 108 节点、5 个下游层。

分层 TTNS 用**同一"每层单父 TTNS 森林"模型的三种拟合落地**做对比(结构相同，只差目标口径/传播)：
  - **R5** 全解析链-树投影   (`fit_analytic_chain`，模块自带)         —— 确定性、无采样、树投影目标；
  - **R6** 全解析链-完整联合 (本脚本 `fit_analytic_chain_joint`，块级复用模块 joint 目标) —— 完整 K 维联合、无 MC，
           但 O(G^K) 维度灾难 → 对 K>`joint_kmax` 的块**自动回退 R5 树目标并打印**(不静默截断)；
  - **R7** 采样求 L2 链       (`fit_sampled_chain`，模块自带)         —— 完整联合(样本)、可扩大块、MC 噪声累积。

全局基线只放用户要求的两个：**global_TT**(chain) 与 **global_TTNS**(chow-liu 树)，per-layer marginal 口径
与链一致(非本层维用基积分积掉，见 `linear_block_logp`)，故可与 R5/R6/R7 同表对比。

评测(逐层、逐 seed)：
  - joint_LL@truth (↑)：该层联合密度在留出 test_x[:,layer] 上平均对数密度；
  - corr_fro vs truth (↓)：模型采样层内相关矩阵 vs 真值 Frobenius 误差(3 次采样均值)。
≥3 seed → 每方法每层报 mean±std。**init_noise=0**(所有拟合的 rank-1 初始化均不加噪)。

用法见文件末 __main__；默认 --seeds 0,1,2。先出 spec 概览，再逐 seed 逐方法拟合评测。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

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
from simple_ttns_l2.maxplus_cdf_forest import UpperForest  # noqa: E402
from simple_ttns_l2.layered_forest import (  # noqa: E402
    BlockModel, fit_layer_forest, forest_log_density, sample_forest,
)
from simple_ttns_l2.ttns_sampler import sample_ttns  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import (  # noqa: E402
    fit_analytic_chain, fit_sampled_chain, structural_blocks,
    analytic_block_target, analytic_block_target_joint,
    _fit_analytic_ttns, _fit_analytic_ttns_joint,
)
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources, linear_block_logp  # noqa: E402
from simple_ttns_l2.experiments.fit_layered_vs_flat_tt import fit_flat, flat_joint_loglik  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"

R_CHAINS = ["R5_tree", "R6_joint", "R7_sampled"]
GLOBALS = ["global_TT", "global_TTNS"]
ALL_METHODS = GLOBALS + R_CHAINS


# --------------------------------------------------------------- R6：全解析链-完整联合(块级复用模块 joint 目标)
def fit_analytic_chain_joint(
    forest0, spec, params: DelayParams, key, s_max0: float,
    q: int = 2, m: int = 24, rank: int = 8,
    n_s: int = 100, n_s_pair: int = 80, n_s_joint: int = 22,
    lr: float = 3e-3, steps: int = 700, init_noise: float = 0.0,
    joint_kmax: int = 4, log_every: int = 0, use_mi: bool = True,
) -> Dict[int, list]:
    """R6 链：与 `fit_analytic_chain`(R5) 结构完全一致(同 DAG 结构分块、同 s_max 递推、同上层森林传播)，
    **唯一区别**是块目标从"树投影 p_tree"升级为块的完整 K 维联合 p_Y。

    维度灾难护栏：joint 交叉项是 O(G^K)。对 K>`joint_kmax` 的块回退到 R5 树目标(`analytic_block_target`
    + `_fit_analytic_ttns`)并打印一行说明——绝不静默截断。返回 {li: forest}。
    """
    forests: Dict[int, list] = {0: forest0}
    s_max = s_max0
    n_layers = len(spec.layers)
    for li in range(1, n_layers):
        s_max = s_max + (params.edge_hi + params.node_hi) + 0.3
        upper = UpperForest(forests[li - 1], q_grid=400)
        layer_nodes = list(spec.layers[li])
        blocks = structural_blocks(spec, li)
        forest: List[BlockModel] = []
        for bi, blk in enumerate(blocks):
            gids = [layer_nodes[i] for i in blk]
            K = len(gids)
            k_b, key = jax.random.split(key)
            if K <= joint_kmax:
                target = analytic_block_target_joint(
                    upper, spec, gids, params, s_max,
                    n_s=n_s, n_s_pair=n_s_pair, n_s_joint=n_s_joint, use_mi=use_mi,
                )
                ttns, bases = _fit_analytic_ttns_joint(
                    target, k_b, q, m, rank, lr, steps, init_noise, log_every,
                    label=f"R6.L{li}.b{bi}",
                )
                parent = target.parent
            else:
                print(f"  [R6 护栏] L{li} 块{bi} K={K}>joint_kmax={joint_kmax} → 回退 R5 树目标",
                      flush=True)
                target = analytic_block_target(
                    upper, spec, gids, params, s_max, n_s=n_s, n_s_pair=n_s_pair, use_mi=use_mi,
                )
                ttns, bases = _fit_analytic_ttns(
                    target, k_b, q, m, rank, lr, steps, init_noise, log_every,
                    label=f"R6fallback.L{li}.b{bi}",
                )
                parent = target.parent
            forest.append(BlockModel(tuple(blk), tuple(int(g) for g in gids),
                                     tuple(parent), ttns, bases))
        forests[li] = forest
    return forests


# --------------------------------------------------------------- 评测辅助
def _count_ttns_params(cores) -> int:
    return int(sum(int(np.prod(np.asarray(c).shape)) for c in cores))


def forest_params(forest: List[BlockModel]) -> int:
    return int(sum(_count_ttns_params(bm.ttns.cores) for bm in forest))


def corr_fro_sampler(sample_fn, tev, key, n, n_rep=3):
    """3 次采样平均层内相关 Frobenius 误差。sample_fn(key, n) -> [n, K]。"""
    Ct = np.corrcoef(tev.T) if tev.shape[1] > 1 else np.array([[1.0]])
    fros = []
    for _ in range(n_rep):
        key, k = jax.random.split(key)
        s = np.asarray(sample_fn(k, n))
        Cm = np.corrcoef(s.T) if s.shape[1] > 1 else np.array([[1.0]])
        fros.append(float(np.linalg.norm(Ct - Cm)))
    return float(np.mean(fros)), float(np.std(fros))


# --------------------------------------------------------------- 单 seed 全流程
def run_one_seed(cfg: dict, seed: int):
    key = jax.random.PRNGKey(seed)
    spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    layers = [list(l) for l in spec.layers]

    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    n_dims = xs.shape[1]
    L0 = layers[0]

    # ---- 源层数据森林(三条链共享同一 L0) ----
    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))

    timings, chains, params_by = {}, {}, {}

    # ---- R5：全解析链-树投影 ----
    t0 = time.perf_counter()
    k_an, key = jax.random.split(key)
    chains["R5_tree"] = fit_analytic_chain(
        forest0, spec, params, k_an, s_max0,
        q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
        n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
        lr=cfg["an_lr"], steps=cfg["an_steps"], init_noise=cfg["init_noise"], log_every=0)
    timings["R5_tree"] = time.perf_counter() - t0

    # ---- R6：全解析链-完整联合(大块回退 R5) ----
    t0 = time.perf_counter()
    k_j, key = jax.random.split(key)
    chains["R6_joint"] = fit_analytic_chain_joint(
        forest0, spec, params, k_j, s_max0,
        q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
        n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"], n_s_joint=cfg["n_s_joint"],
        lr=cfg["an_lr"], steps=cfg["an_steps"], init_noise=cfg["init_noise"],
        joint_kmax=cfg["joint_kmax"], log_every=0)
    timings["R6_joint"] = time.perf_counter() - t0

    # ---- R7：采样求 L2 链 ----
    t0 = time.perf_counter()
    k_sp, key = jax.random.split(key)
    chains["R7_sampled"] = fit_sampled_chain(forest0, spec, params, k_sp, cfg)
    timings["R7_sampled"] = time.perf_counter() - t0

    for name in R_CHAINS:
        params_by[name] = int(sum(forest_params(chains[name][li]) for li in range(len(layers))))

    # ---- 全局扁平：TT(chain) / TTNS(chow-liu)。默认关闭：108 维平方TTNS 的 L2
    #      归一化(∫q)与 ∫q² 都是 108 因子收缩，高维失去数值条件数 → loss 爆到 -1e20，
    #      且评测阶段拿爆掉的模型 grid 采样易 OOM 把进程拖死。用 --with-global 才跑。 ----
    flat_models = {}
    if cfg.get("with_global", False):
        bases = build_bases(jnp.asarray(train_x), cfg["q"], cfg["m"])
        gram = vmap(type(bases).l2_integral)(bases)
        basis_integrals = vmap(type(bases).integral)(bases)
        tr_j = jnp.asarray(train_x[:int(0.85 * n_tr)])
        val_j = jnp.asarray(train_x[int(0.85 * n_tr):])
        flat_specs = {
            "global_TT": [int(p) for p in chain_parent(n_dims)],
            "global_TTNS": [int(p) for p in estimate_chow_liu_tree(train_x, n_bins=16, root=0).parent],
        }
        for name, parent in flat_specs.items():
            t0 = time.perf_counter()
            k_ff, key = jax.random.split(key)
            ttns, parent, r, np_ = fit_flat(parent, name, tr_j, val_j, bases, gram, basis_integrals,
                                            cfg["budget"], cfg, k_ff)
            flat_models[name] = (ttns, list(parent))
            timings[name] = time.perf_counter() - t0
            params_by[name] = int(np_)
            print(f"[seed {seed}][{name}] rank={r} params={np_} 用时 {timings[name]:.1f}s", flush=True)

    # ---- 逐层评测：joint_LL@truth + corr_fro ----
    rows = []
    fll_all = {}
    for li in range(len(layers)):
        Lg = layers[li]
        tev = test_x[:, Lg]
        ll, fro, sd = {}, {}, {}
        # 分层链
        for name in R_CHAINS:
            ll[name] = float(forest_log_density(chains[name][li], tev)[0].mean())
            k_c, key = jax.random.split(key)
            fn = (lambda f: (lambda k, n: sample_forest(f, k, n, grid_size=400)))(chains[name][li])
            fro[name], sd[name] = corr_fro_sampler(fn, tev, k_c, cfg["n_sample"])
        # 全局(per-layer marginal，与链同口径)
        for name, (ttns, parent) in flat_models.items():
            ll[name] = float(np.asarray(linear_block_logp(ttns, bases, parent, Lg, tev)).mean())
            k_m, key = jax.random.split(key)
            fn = (lambda t, p: (lambda k, n: np.asarray(
                sample_ttns(t, bases, p, k, n, grid_size=400))[:, Lg]))(ttns, parent)
            fro[name], sd[name] = corr_fro_sampler(fn, tev, k_m, cfg["n_sample"])
        rows.append({"li": li, "K": len(Lg), "ll": ll, "fro": fro, "sd": sd})
        print(f"[seed {seed}][layer {li}] done", flush=True)

    # ---- 全局模型全联合 LL(仅 global 有意义) ----
    for name, (ttns, parent) in flat_models.items():
        fll_all[name], _ = flat_joint_loglik(ttns, parent, bases, test_x)

    spec_info = {"n_nodes": int(spec.n), "n_layers": len(layers),
                 "layer_dim": len(layers[0]), "n_edges": len(spec.edges),
                 "clusters": list(cfg["clusters"]), "fanin": cfg["fanin"]}
    return {"seed": seed, "spec": spec_info, "rows": rows, "timings": timings,
            "params": params_by, "full_joint_ll": fll_all}


# --------------------------------------------------------------- 多 seed 聚合
def aggregate(results: List[dict]):
    n_layers = results[0]["spec"]["n_layers"]
    agg = {"ll": {}, "fro": {}}  # [metric][method] -> list over layers of (mean, std)
    for metric in ("ll", "fro"):
        for method in ALL_METHODS:
            per_layer = []
            for li in range(n_layers):
                vals = [r["rows"][li][metric][method] for r in results]
                per_layer.append((float(np.mean(vals)), float(np.std(vals))))
            agg[metric][method] = per_layer
    params = {m: int(np.mean([r["params"][m] for r in results])) for m in ALL_METHODS}
    fj = {m: (float(np.mean([r["full_joint_ll"][m] for r in results])),
              float(np.std([r["full_joint_ll"][m] for r in results]))) for m in GLOBALS}
    timings = {m: float(np.mean([r["timings"][m] for r in results])) for m in ALL_METHODS}
    return agg, params, fj, timings


def print_report(results: List[dict], agg, params, fj, timings, seeds):
    spec = results[0]["spec"]
    n_layers = spec["n_layers"]
    print("\n" + "=" * 100)
    print(f"更密/多父·更深 DAG 三方×R5/R6/R7  ——  {len(seeds)} seeds={seeds}, init_noise=0")
    print(f"[spec] 节点={spec['n_nodes']}  层数={spec['n_layers']}  每层={spec['layer_dim']}维  "
          f"簇={spec['clusters']}  fanin={spec['fanin']}(多父/更密)  边数={spec['n_edges']}")
    print("=" * 100)

    def block(metric, title, arrow):
        print(f"\n{title} {arrow}  (mean±std over seeds):")
        hdr = "层 K |" + "".join(f"{m:>18}" for m in ALL_METHODS)
        print(hdr)
        for li in range(n_layers):
            K = results[0]["rows"][li]["K"]
            cells = "".join(f"{agg[metric][m][li][0]:>10.3f}±{agg[metric][m][li][1]:<7.3f}"
                            for m in ALL_METHODS)
            print(f"L{li:<2}{K:<2}|{cells}")

    block("ll", "joint_LL@truth", "(↑ 越高越好)")
    block("fro", "corr_fro vs truth", "(↓ 越低越好)")

    print("\n学习参数量：")
    for m in ALL_METHODS:
        print(f"  {m:<14}{params[m]:>12,}")
    if GLOBALS:
        print("\n全局模型全联合 joint_LL (↑，仅 global 可比)：")
        for m in GLOBALS:
            print(f"  {m:<14}{fj[m][0]:>10.3f}±{fj[m][1]:.3f}")
    print("\n平均单 seed 用时(s)：")
    for m in ALL_METHODS:
        print(f"  {m:<14}{timings[m]:>8.1f}")
    print("=" * 100)


# --------------------------------------------------------------- 配置
CFG = dict(
    # 更密 / 多父 + 更多节点 / 更深层：6 层、簇[3,3,4,4,4]=18维/层=108 节点、fanin=3(簇内近全连接)
    n_layers=6, clusters=[3, 3, 4, 4, 4], fanin=3,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
    n_total=24000, n_sample=8000, n_fit=20000, q=2, m=24, rank=8,
    src_sigma=0.03, budget=400000, rmax=48,
    lr=2e-3, steps=700, batch_sz=512, init_noise=0.0, train_noise=1e-3,
    log_every=350, early_stop_patience=8, mi_threshold=0.02,
    n_s=100, n_s_pair=80, n_s_joint=22, joint_kmax=4, an_lr=3e-3, an_steps=700,
    monitor_val_sz=2000,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2", help="逗号分隔，默认 0,1,2 (≥3)")
    ap.add_argument("--quick", action="store_true", help="小图快跑冒烟(3 层/簇[2,3]/短步)")
    ap.add_argument("--with-global", action="store_true",
                    help="也跑全局基线 global_TT/global_TTNS(108维L2会数值爆炸且评测易OOM，默认关闭)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    global GLOBALS, ALL_METHODS
    if not args.with_global:
        GLOBALS = []
        ALL_METHODS = list(R_CHAINS)

    cfg = dict(CFG)
    cfg["with_global"] = args.with_global
    if args.quick:
        cfg.update(n_layers=3, clusters=[2, 3], fanin=2, n_total=8000, n_sample=2000,
                   n_fit=6000, steps=120, an_steps=120, budget=80000, n_s_joint=16, joint_kmax=3)

    print(f"运行配置: seeds={seeds}, with_global={args.with_global}, init_noise={cfg['init_noise']}, "
          f"n_layers={cfg['n_layers']}, clusters={cfg['clusters']}, fanin={cfg['fanin']}", flush=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "dense_dag_r567_three_way_metrics.json"

    def emit(results, done_seeds):
        """每个 seed 跑完即打表 + 落盘(kill-safe 增量输出):即使被中途 kill，
        已完成 seed 的汇总表与 JSON 都已写出，不会只留在内存。"""
        agg, params, fj, timings = aggregate(results)
        print_report(results, agg, params, fj, timings, done_seeds)
        dump = {"config": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                            for k, v in cfg.items()},
                "seeds": done_seeds,
                "aggregate": {"ll": agg["ll"], "fro": agg["fro"],
                              "params": params, "full_joint_ll": fj, "timings": timings},
                "per_seed": results}
        out.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
        print(f"\nsaved({len(done_seeds)} seed): {out}", flush=True)

    t_all = time.perf_counter()
    results = []
    for seed in seeds:
        print(f"\n########## seed {seed} ##########", flush=True)
        results.append(run_one_seed(cfg, seed))
        emit(results, seeds[:len(results)])  # 增量:本 seed 完成即出表+落盘

    print(f"总用时 {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
