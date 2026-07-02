"""逐层五模型对比：clarify.md 全解析链 vs A 采样链 vs 全局 TT vs 全局 TTNS vs 全局 TTDE。

同一份多层 DAG 全联合数据(异质多峰源)。留出真值 = test_x(真实祖先采样)。
逐层(spec.layers[li], 在 build_layered_spec 下为连续全局 id)比:
- joint_LL@truth: 该层联合密度在 test_x[:,layer] 上的平均对数密度(↑);
- corr_fro: 模型采样层内相关矩阵 vs 真值的 Frobenius 误差(↓);
- top-pair(真值最相关对)的 corr / coskew。

各层密度:
- 解析链/A 链: forest_log_density(森林即该层联合);
- 全局 TTDE: ttde_block_logp(连续块解析边缘);
- 全局 TT/TTNS: linear_block_logp(本文件新增, 非本层维用基积分积掉)。

输出: 表格 + JSON + 折线图 + 逐节点边缘密度图。
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ttde.score.models.opt_for_tree_data import chain_parent  # noqa: E402

from simple_ttns_l2.train_l2 import build_bases  # noqa: E402
from simple_ttns_l2.objective import batch_eval_q_ttns  # noqa: E402
from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.dag_pipeline import build_layered_spec, build_clustered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402
from simple_ttns_l2.layered_forest import (  # noqa: E402
    fit_layer_forest, forest_log_density, sample_forest, maxplus_cond_logdensity,
)
from simple_ttns_l2.ttns_sampler import sample_ttns, _basis_eval_dim  # noqa: E402
from simple_ttns_l2.analytic_tree_fit import fit_analytic_chain, fit_sampled_chain  # noqa: E402
from simple_ttns_l2.experiments.fit_layered_vs_flat_tt import fit_flat, flat_joint_loglik  # noqa: E402
from simple_ttns_l2.experiments.ttde_tt_baseline import fit_ttde_tt, ttde_logp, sample_ttde_tt  # noqa: E402
from simple_ttns_l2.experiments.per_layer_compare import build_ttde_envs, ttde_block_logp  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import sample_layered  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"

MODELS = ["analytic_chain", "sampled_chain", "A_chain", "global_TT", "global_TTNS", "global_TTDE"]
# 全联合口径(28 维完整联合): 链式模型无法构成跨层联合, 故用"分层联合(L0森林 × max-plus条件核)"代表你的模型
FJ_MODELS = ["layered_joint", "global_TT", "global_TTNS", "global_TTDE"]
COLORS = {
    "analytic_chain": "tab:red", "sampled_chain": "tab:cyan", "A_chain": "tab:blue",
    "global_TT": "tab:orange", "global_TTNS": "tab:green", "global_TTDE": "tab:purple",
    "layered_joint": "tab:red",
}


# ------------------------------------------------------------------ 新增: 全局线性 TTNS 层边缘

def linear_block_logp(ttns, bases, parent, block, Xblock, eps=1e-12):
    """全局树 TTNS q 归一化后, block 维的边缘对数密度。

    对某维积分 = 该 leg 用基积分向量替换; 非 block 维全部用 basis_int, block 维用 b(x)。
    树结构下对任意子集成立。Xblock:[n, len(block)] 列序与 block 一致。
    """
    n_dims = len(parent)
    basis_int = np.asarray(vmap(type(bases).integral)(bases))  # [n_dims, m]
    m = basis_int.shape[1]
    n = Xblock.shape[0]
    V = np.broadcast_to(basis_int[None], (n, n_dims, m)).copy()
    for pos, d in enumerate(block):
        V[:, d, :] = np.asarray(_basis_eval_dim(bases, d, jnp.asarray(Xblock[:, pos])))
    q = np.asarray(batch_eval_q_ttns(ttns, jnp.asarray(V), list(parent)))
    return np.log(np.clip(q, eps, None))


# ------------------------------------------------------------------ 新增: 异质源分布库

def complex_sources(spec, cfg: dict):
    """按源节点循环分配 4 类异质源: 三峰混合 / Beta 偏态 / 非对称双峰 / 尖峰+均匀噪声。"""
    sig = cfg.get("src_sigma", 0.03)

    def trimodal():
        mus = np.array([0.2, 0.5, 0.8])

        def s(rng, n):
            c = rng.integers(0, 3, size=n)
            return mus[c] + rng.normal(0.0, sig, size=n)
        return s

    def skewed():
        def s(rng, n):
            return 0.05 + 0.85 * rng.beta(2.0, 6.0, size=n)
        return s

    def asym_bimodal():
        def s(rng, n):
            hot = rng.random(n) < 0.7
            return np.where(hot, rng.normal(0.3, 0.05, size=n), rng.normal(0.9, 0.03, size=n))
        return s

    def peak_noise():
        def s(rng, n):
            hot = rng.random(n) < 0.9
            return np.where(hot, rng.normal(0.5, 0.02, size=n), rng.uniform(0.0, 1.0, size=n))
        return s

    lib = [trimodal(), skewed(), asym_bimodal(), peak_noise()]
    srcs = {}
    i = 0
    for node in spec.layers[0]:
        if not spec.parents(node):
            srcs[node] = lib[i % len(lib)]
            i += 1
    return srcs


# ------------------------------------------------------------------ 指标

def coskew(x, i, j):
    """归一化 coskew: E[(xi-mi)^2 (xj-mj)] / (var_i * std_j)。"""
    xi = x[:, i] - x[:, i].mean()
    xj = x[:, j] - x[:, j].mean()
    return float(np.mean(xi ** 2 * xj) / (x[:, i].std() ** 2 * x[:, j].std() + 1e-12))


# ------------------------------------------------------------------ 主流程

def run(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    if cfg.get("clusters"):
        spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    else:
        spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    n_layers = len(spec.layers)
    layers = [list(l) for l in spec.layers]

    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n = xs.shape[0]
    n_dims = xs.shape[1]
    n_tr = int(0.7 * n)
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    n_sample = cfg["n_sample"]
    L0 = layers[0]

    timings = {}

    # ---- L0 数据森林(解析链与 A 链共享) ----
    t0 = time.perf_counter()
    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))
    timings["forest0"] = time.perf_counter() - t0

    # ---- A 链: 从 forest0 采样 → 逐层 propagate + 重拟合 ----
    t0 = time.perf_counter()
    A_forests = {0: forest0}
    k_as, key = jax.random.split(key)
    s_prev = np.asarray(sample_forest(forest0, k_as, n_sample, grid_size=400))
    for li in range(1, n_layers):
        rng = np.random.default_rng(500 + li)
        s_li = propagate_layer(spec, li, s_prev, params, rng)
        k_a, key = jax.random.split(key)
        A_forests[li] = fit_layer_forest(jnp.asarray(s_li), layers[li], cfg, k_a,
                                         label=f"A_L{li}", mi_threshold=cfg["mi_threshold"])
        k_s, key = jax.random.split(key)
        s_prev = np.asarray(sample_forest(A_forests[li], k_s, n_sample, grid_size=400))
    timings["A_chain"] = time.perf_counter() - t0

    # ---- 解析链 ----
    t0 = time.perf_counter()
    k_an, key = jax.random.split(key)
    AN_forests = fit_analytic_chain(forest0, spec, params, k_an, s_max0,
                                    q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
                                    n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                                    lr=cfg["an_lr"], steps=cfg["an_steps"],
                                    init_noise=cfg["init_noise"], log_every=0)
    timings["analytic_chain"] = time.perf_counter() - t0

    # ---- 采样求 L2 链(R7): 同结构, 每块目标=完整联合样本 ----
    t0 = time.perf_counter()
    k_sp, key = jax.random.split(key)
    SP_forests = fit_sampled_chain(forest0, spec, params, k_sp, cfg)
    timings["sampled_chain"] = time.perf_counter() - t0

    # ---- 全局 TT / TTNS ----
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

    # ---- 全局 TTDE ----
    # 尖峰源 + 大训练集会使 TTDE 的 EM 平方初始化在某些数据点密度为 0 → log(0) NaN。
    # 子采样训练集(TTDE 不需全量)+ 略大 init_noise + 梯度裁剪, 保证初始密度处处为正且训练稳定。
    t0 = time.perf_counter()
    n_ttde = min(cfg["ttde_n_train"], n_tr)
    ttde_tr = train_x[:n_ttde]
    ttde_val = train_x[n_ttde:n_ttde + cfg["monitor_val_sz"]]
    ttde_cfg = {**cfg, "init_noise": cfg["ttde_init_noise"]}
    ttde_model, ttde_params, ttde_info = fit_ttde_tt(ttde_tr, ttde_val, ttde_cfg, cfg["seed"])
    cores, gram_t, env, lenv, Z = build_ttde_envs(ttde_params, ttde_model.bases)
    timings["global_TTDE"] = time.perf_counter() - t0

    # TTDE sanity: 全维边缘 == ttde_logp
    chk = ttde_block_logp(cores, gram_t, env, lenv, Z, ttde_model.bases, list(range(n_dims)), test_x[:512])
    ref = ttde_logp(ttde_model, ttde_params, test_x[:512])
    sanity = float(np.max(np.abs(chk - ref)))
    print(f"[sanity] TTDE full-marginal vs ttde_logp  max|Δ|={sanity:.2e}", flush=True)

    # ---- 各模型采样(全局采一次复用; 链逐层森林采样) ----
    gt_s = test_x[:n_sample]
    full_samp = {}  # model -> [n_sample, n_dims] (仅边缘有效; 链的联合不精确)
    layer_samp = {m: {} for m in MODELS}  # model -> {li: [n_sample, layer_size]}

    # 链: 逐层森林采样
    for name, forests in (("analytic_chain", AN_forests), ("sampled_chain", SP_forests),
                          ("A_chain", A_forests)):
        full = np.zeros((n_sample, n_dims))
        for li in range(n_layers):
            k_m, key = jax.random.split(key)
            fs = np.asarray(sample_forest(forests[li], k_m, n_sample, grid_size=400))
            layer_samp[name][li] = fs
            full[:, layers[li]] = fs
        full_samp[name] = full

    # 全局 TT / TTNS: 采全维一次
    for name, (ttns, parent, _, _) in flat_models.items():
        k_m, key = jax.random.split(key)
        fs = np.asarray(sample_ttns(ttns, bases, parent, k_m, n_sample, grid_size=400))
        full_samp[name] = fs
        for li in range(n_layers):
            layer_samp[name][li] = fs[:, layers[li]]

    # 全局 TTDE: 采全维一次
    k_t, key = jax.random.split(key)
    ttde_s = sample_ttde_tt(ttde_model, ttde_params, k_t, n_sample, grid_size=cfg["ttde_grid"])
    full_samp["global_TTDE"] = ttde_s
    for li in range(n_layers):
        layer_samp["global_TTDE"][li] = ttde_s[:, layers[li]]

    # ---- 逐层指标 ----
    rows = []
    for li in range(n_layers):
        Lg = layers[li]
        tev = test_x[:, Lg]
        Ct = np.corrcoef(tev.T)
        K = len(Lg)
        # top-pair (真值最相关对)
        if K > 1:
            iu = np.triu_indices(K, 1)
            k_top = int(np.argmax(np.abs(Ct[iu])))
            a_, b_ = int(iu[0][k_top]), int(iu[1][k_top])
            corr_t = float(Ct[a_, b_])
            csk_t = coskew(tev, a_, b_)
        else:
            a_ = b_ = 0
            corr_t = 1.0
            csk_t = coskew(tev, 0, 0)

        row = {"li": li, "K": K, "pair": (a_, b_), "corr_t": corr_t, "csk_t": csk_t}

        # joint_LL@truth
        ll = {}
        ll["analytic_chain"], npos_an = forest_log_density(AN_forests[li], tev)
        ll["sampled_chain"], _ = forest_log_density(SP_forests[li], tev)
        ll["A_chain"], _ = forest_log_density(A_forests[li], tev)
        for name, (ttns, parent, _, _) in flat_models.items():
            ll[name] = linear_block_logp(ttns, bases, parent, Lg, tev)
        ll["global_TTDE"] = ttde_block_logp(cores, gram_t, env, lenv, Z, ttde_model.bases, Lg, tev)
        row["ll"] = {m: float(np.asarray(ll[m]).mean()) for m in MODELS}
        row["npos_analytic"] = float(npos_an)

        # corr_fro + top-pair corr/coskew (采样)
        row["fro"] = {}
        row["corr_m"] = {}
        row["csk_m"] = {}
        for m in MODELS:
            sm = layer_samp[m][li]
            Cm = np.corrcoef(sm.T) if K > 1 else np.array([[1.0]])
            row["fro"][m] = float(np.linalg.norm(Ct - Cm))
            row["corr_m"][m] = float(Cm[a_, b_])
            row["csk_m"][m] = coskew(sm, a_, b_)
        rows.append(row)

    # ---- 全联合(完整 28 维)口径 ----
    # 你的模型 = 分层联合: L0 森林密度 × 各层 max-plus 条件核(已知延迟, 闭式条件密度)。
    # 链式(analytic/A)只给每层边缘、不构成跨层联合, 故全联合表用 layered_joint 代表分层范式。
    from scipy.stats import wasserstein_distance
    k_lj, key = jax.random.split(key)
    lj_samp = np.asarray(sample_layered(forest0, spec, params, k_lj, n_sample, cfg["seed"] + 7))
    full_samp["layered_joint"] = lj_samp

    ll_lj = np.asarray(forest_log_density(forest0, test_x[:, L0])[0])
    for li in range(1, n_layers):
        for v in spec.layers[li]:
            ll_lj = ll_lj + maxplus_cond_logdensity(test_x[:, v], test_x[:, list(spec.parents(v))], params)

    Cgt = np.corrcoef(gt_s.T)
    fj = {}
    fj_ll = {"layered_joint": float(ll_lj.mean())}
    for name, (ttns, parent, _, _) in flat_models.items():
        fj_ll[name], _ = flat_joint_loglik(ttns, parent, bases, test_x)
    fj_ll["global_TTDE"] = float(ttde_logp(ttde_model, ttde_params, test_x).mean())
    for m in FJ_MODELS:
        sm = full_samp[m]
        fj[m] = dict(
            ll=fj_ll[m],
            corr_fro=float(np.linalg.norm(np.corrcoef(sm.T) - Cgt)),
            w1_marg=float(np.mean([wasserstein_distance(sm[:, j], gt_s[:, j]) for j in range(n_dims)])),
        )

    return dict(spec=spec, rows=rows, full_samp=full_samp, gt_s=gt_s,
                timings=timings, sanity=sanity, fj=fj,
                params={**{n: flat_models[n][3] for n in flat_models},
                        "global_TTDE": ttde_info.get("learned_params", -1)})


# ------------------------------------------------------------------ 出图 + 打印

def plot_metrics(res, out: Path):
    rows = res["rows"]
    lis = [r["li"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for m in MODELS:
        ax[0].plot(lis, [r["ll"][m] for r in rows], "o-", color=COLORS[m], label=m)
        ax[1].plot(lis, [r["fro"][m] for r in rows], "o-", color=COLORS[m], label=m)
    ax[0].set_title("joint_LL @truth per layer (higher=better)")
    ax[1].set_title("corr_fro vs truth per layer (lower=better)")
    ax[2].plot(lis, [r["corr_t"] for r in rows], "k^--", label="truth")
    for m in MODELS:
        ax[2].plot(lis, [r["corr_m"][m] for r in rows], "o-", color=COLORS[m], label=m)
    ax[2].set_title("top-pair corr (closer to truth=better)")
    for a in ax:
        a.set_xlabel("layer"); a.set_xticks(lis); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("per-layer 5-model comparison: analytic chain vs A chain vs global TT/TTNS/TTDE",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_marginals(res, out: Path):
    spec = res["spec"]
    samp = res["full_samp"]
    gt_s = res["gt_s"]
    nL = len(spec.layers)
    nN = max(len(l) for l in spec.layers)
    fig, axes = plt.subplots(nL, nN, figsize=(3.0 * nN, 2.4 * nL), squeeze=False)
    for li in range(nL):
        nodes = list(spec.layers[li])
        for j in range(nN):
            ax = axes[li][j]
            if j >= len(nodes):
                ax.axis("off"); continue
            node = nodes[j]
            gv = gt_s[:, node]
            lo, hi = float(np.percentile(gv, 0.5)), float(np.percentile(gv, 99.5))
            allv = [gv] + [samp[m][:, node] for m in MODELS]
            blo = min(float(v.min()) for v in allv)
            bhi = max(float(v.max()) for v in allv)
            bins = np.linspace(blo, bhi, 70)
            ax.hist(gv, bins=bins, density=True, histtype="step", color="k", lw=2.0, label="GT")
            for m in MODELS:
                ax.hist(samp[m][:, node], bins=bins, density=True, histtype="step",
                        color=COLORS[m], lw=1.1, label=m)
            ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
            ax.set_title(f"L{li} node{node}", fontsize=9)
            if li == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.suptitle("per-node marginal density: GT vs 5 models", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def print_tables(res):
    rows = res["rows"]
    print("\n================= 逐层五模型对比 =================")
    print(f"[sanity] TTDE max|Δ| = {res['sanity']:.2e}")
    print("\njoint_LL @truth (↑):")
    hdr = "层  " + "".join(f"{m:>16}" for m in MODELS)
    print(hdr)
    for r in rows:
        print(f"L{r['li']:<2}" + "".join(f"{r['ll'][m]:>16.4f}" for m in MODELS))
    print("\ncorr_fro vs truth (↓):")
    print(hdr)
    for r in rows:
        print(f"L{r['li']:<2}" + "".join(f"{r['fro'][m]:>16.4f}" for m in MODELS))
    print("\ntop-pair corr  (truth | 各模型):")
    for r in rows:
        pm = "".join(f"{r['corr_m'][m]:>10.3f}" for m in MODELS)
        print(f"L{r['li']} pair(y{r['pair'][0]},y{r['pair'][1]}) truth={r['corr_t']:.3f} |{pm}")
    print("\ntop-pair coskew (truth | 各模型):")
    for r in rows:
        pm = "".join(f"{r['csk_m'][m]:>10.3f}" for m in MODELS)
        print(f"L{r['li']} truth={r['csk_t']:.3f} |{pm}")
    print("\n用时(s):")
    for k, v in res["timings"].items():
        print(f"  {k:<16}{v:>8.1f}")
    print("=================================================\n")

    fj = res["fj"]
    print("========= 全联合(完整 28 维)对比 —— 这才是分层模型的主战场 =========")
    print(f"{'model':<16}{'joint_LL(↑)':>14}{'corr_fro(↓)':>14}{'W1_marg(↓)':>14}")
    for m in FJ_MODELS:
        print(f"{m:<16}{fj[m]['ll']:>14.4f}{fj[m]['corr_fro']:>14.4f}{fj[m]['w1_marg']:>14.4f}")
    print("==================================================================\n")


def plot_fulljoint(res, out: Path):
    fj = res["fj"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    def bar(a, key, title, better):
        vals = [fj[m][key] for m in FJ_MODELS]
        a.bar(range(len(FJ_MODELS)), vals, color=[COLORS[m] for m in FJ_MODELS])
        a.set_xticks(range(len(FJ_MODELS))); a.set_xticklabels(FJ_MODELS, rotation=15, fontsize=8)
        a.set_title(f"{title}\n({better})", fontsize=10)
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    bar(ax[0], "ll", "full-joint log-likelihood", "higher=better")
    bar(ax[1], "corr_fro", "full-joint corr Frobenius err", "lower=better")
    bar(ax[2], "w1_marg", "mean marginal W1", "lower=better")
    fig.suptitle("FULL-JOINT (28-dim) comparison: layered_joint (yours) vs global TT/TTNS/TTDE",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# 先小: clustered DAG(簇内相连、簇间独立) → structural_blocks 精确裂块, 块内有真实非树相关。
CFG_SMALL = dict(
    n_layers=3, clusters=[3, 3], fanin=2,  # 每层 6 维, 每层裂成 2 个大小 3 的块
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
    n_total=20000, n_sample=4000, n_fit=20000, q=2, m=24, rank=8, budget=120000, rmax=40,
    src_sigma=0.03,
    lr=2e-3, steps=800, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
    n_s=100, n_s_pair=80, an_lr=3e-3, an_steps=800,
    ttde_rank=16, ttde_steps=1200, ttde_em_steps=15, ttde_patience=8,
    ttde_grid=200, monitor_val_sz=2000,
    ttde_n_train=10000, ttde_init_noise=0.03, ttde_grad_clip=10.0,
)

# 大例子: 异构簇 [2,3,4,5,6] → 每层 20 维、5 个不同大小的块; 5 层 = 100 节点, 4 个下游层
CFG_BIG = {**CFG_SMALL,
    "n_layers": 5, "clusters": [2, 3, 4, 5, 6], "fanin": 2,
    "n_total": 24000, "n_sample": 8000, "n_fit": 20000,
    "steps": 700, "an_steps": 700, "log_every": 350,
    "budget": 400000, "rmax": 48,
    "ttde_rank": 20, "ttde_steps": 1500, "ttde_n_train": 12000,
}


def main():
    cfg = CFG_BIG if "--big" in sys.argv else CFG_SMALL
    t_all = time.perf_counter()
    res = run(cfg)
    print_tables(res)

    REPORTS.mkdir(parents=True, exist_ok=True)
    plot_metrics(res, REPORTS / "per_layer_all_methods_metrics.png")
    plot_marginals(res, REPORTS / "per_layer_all_methods_marginals.png")
    plot_fulljoint(res, REPORTS / "per_layer_all_methods_fulljoint.png")

    dump = dict(config=cfg, sanity=res["sanity"], timings=res["timings"],
                params=res["params"], full_joint=res["fj"],
                rows=[{k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in r.items()} for r in res["rows"]])
    (REPORTS / "per_layer_all_methods_metrics.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False))
    print("saved:", REPORTS / "per_layer_all_methods_metrics.png")
    print("saved:", REPORTS / "per_layer_all_methods_marginals.png")
    print("saved:", REPORTS / "per_layer_all_methods_fulljoint.png")
    print(f"总用时 {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
