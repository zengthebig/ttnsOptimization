"""全局 TTDE：TT(链) vs TTNS(MI 树) —— 平方参数化 + MLE，仅换拓扑。

问题：把 TTDE 的链式 TT 拓扑换成 **chow-liu MI 树** 的 TTNS 拓扑，全联合测试对数似然会更好吗？
（两者同为 $p(x)=\tilde q(x)^2/Z$ 的平方 MLE，同 basis/rank/预算，只差单父树拓扑。）

同一份多层 DAG 全联合数据（clustered，异质多峰源）。评测：全联合测试平均 log 密度（↑）。
无采样（平方 TTNS 树采样器未实现）；本脚本只比 log-likelihood（平方模型原生强项、直接可比）。
"""
from __future__ import annotations

import json
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
from simple_ttns_l2.dag_pipeline import build_clustered_spec, build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources  # noqa: E402
from simple_ttns_l2.experiments.ttde_tt_baseline import fit_ttde_tt, fit_ttde_ttns, ttde_logp  # noqa: E402
from ttde.score.models.opt_for_tree_data import chain_parent  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def tt_params(d: int, m: int, r: int) -> int:
    """平方 TT(链)参数量：first[1,m,r]+inner[(d-2),r,m,r]+last[r,m,1]。"""
    return m * r + (d - 2) * r * m * r + r * m


def tree_degrees(parent) -> list:
    """单父树各节点的度(孩子数 + 是否有父)。root: parent[root]==root。"""
    d = len(parent)
    roots = [v for v in range(d) if parent[v] == v or parent[v] == -1]
    root = roots[0]
    deg = [0] * d
    for v in range(d):
        if v == root:
            continue
        deg[v] += 1          # 到父的边
        deg[parent[v]] += 1  # 父到该孩子的边
    return deg


def ttns_params(parent, m: int, r: int) -> int:
    """平方 TTNS(树)参数量：m * Σ_v r^deg(v)。"""
    return int(m * sum(r ** dg for dg in tree_degrees(parent)))


def tt_rank_for_params(target: int, d: int, m: int) -> int:
    """反解使 TT(链)参数量最接近 target 的 rank。"""
    best_r, best_err = 1, float("inf")
    for r in range(1, 4096):
        err = abs(tt_params(d, m, r) - target)
        if err < best_err:
            best_r, best_err = r, err
        elif tt_params(d, m, r) > target * 4:
            break
    return best_r


def run(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    if cfg.get("clusters"):
        spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    else:
        spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])

    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n = xs.shape[0]
    n_dims = xs.shape[1]
    n_tr = int(0.7 * n)
    train_x, test_x = xs[:n_tr], xs[n_tr:]
    n_ttde = min(cfg["ttde_n_train"], n_tr)
    tr = train_x[:n_ttde]
    val = train_x[n_ttde:n_ttde + cfg["monitor_val_sz"]]
    ttde_cfg = {**cfg, "init_noise": cfg["ttde_init_noise"]}

    chain = [int(p) for p in chain_parent(n_dims)]
    mi_tree = [int(p) for p in estimate_chow_liu_tree(train_x, n_bins=16, root=0).parent]

    r_ttns = cfg.get("ttde_rank_ttns", cfg["ttde_rank"])
    # rank_tt: 显式给定则用之; 若 match_params=True 则按 TTNS 参数量自动反解使两者对齐。
    p_ttns = ttns_params(mi_tree, cfg["m"], r_ttns)
    if cfg.get("match_params"):
        r_tt = tt_rank_for_params(p_ttns, n_dims, cfg["m"])
    else:
        r_tt = cfg.get("ttde_rank_tt", cfg["ttde_rank"])
    p_tt = tt_params(n_dims, cfg["m"], r_tt)
    print(f"n_dims={n_dims}  chain_parent={chain}\nMI_tree_parent={mi_tree}", flush=True)
    print(f"MI 树度分布={tree_degrees(mi_tree)}", flush=True)
    print(f"[params] TTNS(r={r_ttns})={p_ttns}  vs  TT(r={r_tt})={p_tt}  "
          f"(match_params={cfg.get('match_params', False)})", flush=True)

    cfg_tt = {**ttde_cfg, "ttde_rank": r_tt}
    cfg_ttns = {**ttde_cfg, "ttde_rank": r_ttns}
    out = {}
    for name, fit_fn, parent in (
        ("global_TTDE_TT(chain)", lambda: fit_ttde_tt(tr, val, cfg_tt, cfg["seed"]), chain),
        ("global_TTDE_TTNS(MI)", lambda: fit_ttde_ttns(tr, val, cfg_ttns, cfg["seed"], mi_tree), mi_tree),
    ):
        t0 = time.perf_counter()
        model, mp, info = fit_fn()
        dt = time.perf_counter() - t0
        ll_test = float(ttde_logp(model, mp, test_x).mean())
        ll_train = float(ttde_logp(model, mp, tr).mean())
        out[name] = dict(ll_test=ll_test, ll_train=ll_train,
                         params=info["learned_params"], val_nll=info["best_val_nll"], sec=dt)
        print(f"[{name}] test_LL={ll_test:.4f} train_LL={ll_train:.4f} "
              f"params={info['learned_params']} sec={dt:.1f}", flush=True)

    print("\n================= 全局 TTDE: TT(链) vs TTNS(MI 树) =================")
    print(f"{'model':<26}{'test_LL(↑)':>12}{'train_LL':>12}{'params':>10}{'sec':>8}")
    for name, r in out.items():
        print(f"{name:<26}{r['ll_test']:>12.4f}{r['ll_train']:>12.4f}{r['params']:>10}{r['sec']:>8.1f}")
    d = out["global_TTDE_TTNS(MI)"]["ll_test"] - out["global_TTDE_TT(chain)"]["ll_test"]
    print(f"\nΔtest_LL (TTNS - TT) = {d:+.4f}  → {'TTNS(MI树) 更好' if d > 0 else 'TT(链) 更好或持平'}")
    print("==================================================================\n")
    return out


def plot_results(records: dict, out_path):
    """records: {label: {ll_test, ll_train, params}} → 双面板柱状图(test/train LL + 参数量对数轴)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(records.keys())
    x = np.arange(len(labels))
    ll_test = [records[k]["ll_test"] for k in labels]
    ll_train = [records[k]["ll_train"] for k in labels]
    params = [records[k]["params"] for k in labels]
    colors = ["tab:orange", "tab:brown", "tab:red"][: len(labels)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    w = 0.38
    ax[0].bar(x - w / 2, ll_test, w, label="test_LL", color=colors)
    ax[0].bar(x + w / 2, ll_train, w, label="train_LL", color=colors, alpha=0.45, hatch="//")
    ax[0].set_title("log-likelihood (test solid / train hatched)\nhigher test = better; train≫test ⇒ overfit")
    ax[0].set_ylabel("mean log-density")
    for i, v in enumerate(ll_test):
        ax[0].text(x[i] - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax[0].legend(fontsize=8)

    ax[1].bar(x, params, color=colors)
    ax[1].set_yscale("log")
    ax[1].set_title("#parameters (log scale)")
    for i, v in enumerate(params):
        ax[1].text(x[i], v, f"{v:,}", ha="center", va="bottom", fontsize=8)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=12, fontsize=8)
        a.grid(alpha=0.3, axis="y")
    fig.suptitle("global TTDE (squared MLE): TT(chain) vs TTNS(MI tree) — clustered 18-dim",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = dict(
        n_layers=3, clusters=[3, 3], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
        n_total=20000, q=2, m=24, src_sigma=0.03,
        lr=2e-3, batch_sz=512, train_noise=1e-3, log_every=100, seed=0,
        # 等参数量对齐: TTNS(MI树) rank=8(枢纽核 ~ rank^deg, 勿超8); match_params=True 自动把 TT rank 提到同参数量(~64)。
        ttde_rank=8, ttde_rank_ttns=8, match_params=True,
        ttde_steps=1000, ttde_em_steps=15, ttde_patience=10,
        monitor_val_sz=2000, ttde_n_train=10000, ttde_init_noise=0.03, ttde_grad_clip=10.0,
    )
    run(cfg)


if __name__ == "__main__":
    main()
