"""大预算扫描:分层 DAG × TTNS 链(R5 解析树投影 / R7 采样求 L2)随预算的绝对质量。

背景:分层链此前只在固定小预算(rank=8, m=24)验证过"参数少仍碾压全局"。本脚本
第一次对**分层链本身**做预算阶梯扫描,检验四条判据:
  (1) 逐层 joint_LL 是否随预算逼近 oracle 上限;
  (2) 等预算下是否仍胜全局 TT/TTNS/TTDE;
  (3) corr_fro 残差(块内非树边)能否随预算被压向 0;
  (4) 深层多块是否稳定不发散。

预算旋钮(本 clustered [2,3,4,5,6]×5=100 节点图下块 chow-liu 树最大度=2,故 rank^deg=rank²,
rank 抬升安全):
  - rank(bond 维): 8→16→24
  - m(样条基数): 24→48
  - n_fit(仅 R7 每层采样池): 20k→40k→80k, 看 MC 深层累积是否被样本预算压住
  - 全大点: rank=16 + m=48 + n_fit=80k

oracle 上限:逐层按**结构块**(≤6 维、块间独立)用 Kozachenko-Leonenko k-NN 微分熵估计
-H = held-out 平均对数密度的确切上限;corr_fro 的 oracle = 0(真值相关)。

等预算全局基线:复用 baselines_big.run,把全局树预算设为分层链参数量(--baselines)。

运行:
  env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.budget_sweep_layered            # 单 seed 链扫描
  env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.budget_sweep_layered --seeds 3  # 3 seed 结论点
  env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.budget_sweep_layered --baselines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.dag_pipeline import build_clustered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import (  # noqa: E402
    fit_layer_forest, forest_log_density, sample_forest,
)
from simple_ttns_l2.analytic_tree_fit import (  # noqa: E402
    fit_analytic_chain, fit_sampled_chain, structural_blocks,
)
from simple_ttns_l2.experiments.per_layer_all_methods import complex_sources, CFG_BIG  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


# ------------------------------------------------------------------ oracle 上限(k-NN 熵)

def knn_entropy_ceiling(X: np.ndarray, k: int = 3) -> float:
    """Kozachenko-Leonenko 微分熵估计,返回 held-out 平均对数密度上限 = -H(nats)。

    Ĥ = ψ(N) - ψ(k) + log(V_d) + (d/N) Σ log r_i, r_i = 第 k 近邻欧氏距离,
    V_d = π^{d/2}/Γ(d/2+1)(单位 d 维球体积)。返回 -Ĥ。
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if N <= k + 1:
        return float("nan")
    tree = cKDTree(X)
    dist, _ = tree.query(X, k=k + 1)  # 第 0 列是自身(0)
    r = dist[:, -1]
    r = np.maximum(r, 1e-12)
    log_Vd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
    H = digamma(N) - digamma(k) + log_Vd + (d / N) * np.sum(np.log(r))
    return float(-H)


def layer_ceilings(test_x: np.ndarray, spec) -> list:
    """逐层 oracle joint_LL 上限 = Σ_块 (-H_块),块 = structural_blocks(块间独立)。"""
    layers = [list(l) for l in spec.layers]
    outs = []
    for li in range(len(layers)):
        nodes = layers[li]
        blocks = structural_blocks(spec, li)
        c = 0.0
        for blk in blocks:
            gids = [nodes[i] for i in blk]
            c += knn_entropy_ceiling(test_x[:, gids])
        outs.append(c)
    return outs


# ------------------------------------------------------------------ 参数量

def forest_params(forest) -> int:
    return int(sum(int(np.prod(core.shape)) for bm in forest for core in bm.ttns.cores))


def chain_params(forests: dict) -> int:
    return int(sum(forest_params(f) for f in forests.values()))


# ------------------------------------------------------------------ corr_fro

def corr_fro_layer(forest, tev, key, n, n_rep=3):
    Ct = np.corrcoef(tev.T) if tev.shape[1] > 1 else np.array([[1.0]])
    fros = []
    for _ in range(n_rep):
        key, kk = jax.random.split(key)
        s = np.asarray(sample_forest(forest, kk, n, grid_size=400))
        Cm = np.corrcoef(s.T) if s.shape[1] > 1 else np.array([[1.0]])
        fros.append(float(np.linalg.norm(Ct - Cm)))
    return float(np.mean(fros)), float(np.std(fros))


# ------------------------------------------------------------------ 数据 + 单点

def build_data(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_clustered_spec(cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * xs.shape[0])
    return spec, params, xs[:n_tr], xs[n_tr:], key


def run_point(cfg: dict, data, run_r5=True, run_r7=True):
    spec, params, train_x, test_x, key = data
    layers = [list(l) for l in spec.layers]
    L0 = layers[0]

    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))

    out = {"layers": [len(l) for l in layers]}

    if run_r5:
        t0 = time.perf_counter()
        k_an, key = jax.random.split(key)
        AN = fit_analytic_chain(forest0, spec, params, k_an, s_max0,
                                q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
                                n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
                                lr=cfg["an_lr"], steps=cfg["an_steps"],
                                init_noise=cfg["init_noise"], log_every=0)
        out["t_r5"] = time.perf_counter() - t0
        out["params_r5"] = chain_params(AN)
    if run_r7:
        t0 = time.perf_counter()
        k_sp, key = jax.random.split(key)
        SP = fit_sampled_chain(forest0, spec, params, k_sp, cfg)
        out["t_r7"] = time.perf_counter() - t0
        out["params_r7"] = chain_params(SP)

    rows = []
    for li in range(len(layers)):
        tev = test_x[:, layers[li]]
        row = {"li": li, "K": len(layers[li])}
        if run_r5:
            row["ll_r5"] = float(forest_log_density(AN[li], tev)[0].mean())
            k1, key = jax.random.split(key)
            row["fro_r5"], row["fro_r5_sd"] = corr_fro_layer(AN[li], tev, k1, cfg["n_sample"])
        if run_r7:
            row["ll_r7"] = float(forest_log_density(SP[li], tev)[0].mean())
            k2, key = jax.random.split(key)
            row["fro_r7"], row["fro_r7_sd"] = corr_fro_layer(SP[li], tev, k2, cfg["n_sample"])
        rows.append(row)
    out["rows"] = rows
    return out


# ------------------------------------------------------------------ 预算阶梯

def budget_points():
    """(label, overrides, run_r5, run_r7)。R5 不用 n_fit,故 n_fit 点只跑 R7。"""
    return [
        ("base r8 m24 nf20k",   dict(rank=8,  m=24, n_fit=20000), True,  True),
        ("rank16",              dict(rank=16, m=24, n_fit=20000), True,  True),
        ("rank24",              dict(rank=24, m=24, n_fit=20000), True,  True),
        ("m48",                 dict(rank=8,  m=48, n_fit=20000), True,  True),
        ("nfit40k (R7)",        dict(rank=8,  m=24, n_fit=40000), False, True),
        ("nfit80k (R7)",        dict(rank=8,  m=24, n_fit=80000), False, True),
    ]


# ------------------------------------------------------------------ 聚合 + 绘图 + 报告

def _agg_point(point: dict):
    """把一个预算点的多 seed rows 聚成 per-layer 均值。返回 {li: {metric: mean}}。"""
    per_layer = {}
    seeds = point["seeds"]
    n_layers = len(seeds[0]["rows"])
    for li in range(n_layers):
        acc = {}
        for m in ("ll_r5", "ll_r7", "fro_r5", "fro_r7"):
            vals = [s["rows"][li].get(m) for s in seeds if s["rows"][li].get(m) is not None]
            if vals:
                acc[m] = float(np.mean(vals))
                acc[m + "_sd"] = float(np.std(vals))
        per_layer[li] = acc
    return per_layer


def plot_and_report(payload: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ceils = payload["ceilings"]
    points = payload["points"]
    labels = list(points.keys())
    aggs = {lab: _agg_point(points[lab]) for lab in labels}
    n_layers = len(ceils)
    down = list(range(1, n_layers))  # 下游层(相关结构非平凡)

    # 每点:下游层平均 gap-to-ceiling 与平均 corr_fro
    def mean_metric(lab, key):
        vals = [aggs[lab][li][key] for li in down if key in aggs[lab][li]]
        return float(np.mean(vals)) if vals else np.nan

    gap_r5 = [np.mean([ceils[li] - aggs[lab][li]["ll_r5"] for li in down if "ll_r5" in aggs[lab][li]])
              if any("ll_r5" in aggs[lab][li] for li in down) else np.nan for lab in labels]
    gap_r7 = [np.mean([ceils[li] - aggs[lab][li]["ll_r7"] for li in down if "ll_r7" in aggs[lab][li]])
              if any("ll_r7" in aggs[lab][li] for li in down) else np.nan for lab in labels]
    fro_r5 = [mean_metric(lab, "fro_r5") for lab in labels]
    fro_r7 = [mean_metric(lab, "fro_r7") for lab in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].plot(x, gap_r5, "o-", label="R5 analytic", color="#d62728")
    ax[0].plot(x, gap_r7, "s-", label="R7 sampled", color="#1f77b4")
    ax[0].axhline(0, ls="--", color="gray", lw=1, label="oracle ceiling (gap=0)")
    ax[0].set_title("Downstream-layer mean joint_LL gap to oracle (lower=better)")
    ax[0].set_ylabel("ceiling - joint_LL")
    ax[1].plot(x, fro_r5, "o-", label="R5 analytic", color="#d62728")
    ax[1].plot(x, fro_r7, "s-", label="R7 sampled", color="#1f77b4")
    ax[1].axhline(0, ls="--", color="gray", lw=1, label="oracle (corr_fro=0)")
    ax[1].set_title("Downstream-layer mean corr_fro (lower=better)")
    ax[1].set_ylabel("corr_fro vs truth")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        a.legend(fontsize=9)
        a.grid(alpha=0.3)
    fig.tight_layout()
    png = REPORTS / "budget_sweep_layered.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[写出] {png}", flush=True)

    # markdown 报告
    lines = ["# 大预算扫描:分层 DAG × TTNS 链(R5/R7)结果\n",
             f"图:100 节点 clustered `{payload['config'].get('clusters')}`×{payload['config'].get('n_layers')} 层;"
             f"seeds={len(points[labels[0]]['seeds'])}。\n",
             "\noracle 上限(k-NN 熵,逐层 joint_LL 上界): "
             + "  ".join(f"L{i}={c:.3f}" for i, c in enumerate(ceils)) + "\n",
             "\n## 逐点下游层平均指标\n",
             "| 预算点 | params_r5 | params_r7 | gap_r5↓ | gap_r7↓ | fro_r5↓ | fro_r7↓ |",
             "|---|---|---|---|---|---|---|"]
    for i, lab in enumerate(labels):
        p0 = points[lab]["seeds"][0]
        lines.append(f"| {lab} | {p0.get('params_r5','-')} | {p0.get('params_r7','-')} | "
                     f"{gap_r5[i]:.3f} | {gap_r7[i]:.3f} | {fro_r5[i]:.3f} | {fro_r7[i]:.3f} |")
    lines.append("\n> gap = oracle 上限 − joint_LL(越小越接近上限);fro = 采样相关矩阵 vs 真值 Frobenius 误差。\n")
    rep = REPORTS / "budget_sweep_layered_report_zh.md"
    rep.write_text("\n".join(lines))
    print(f"[写出] {rep}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--baselines", action="store_true", help="额外跑等预算全局基线")
    ap.add_argument("--quick", action="store_true", help="减步数 smoke")
    ap.add_argument("--only", type=str, default="", help="逗号分隔的点 label 子串过滤")
    ap.add_argument("--fresh", action="store_true", help="忽略已有 JSON,从头重跑")
    args = ap.parse_args()

    base = dict(CFG_BIG)
    if args.quick:
        # 缩小图 + 步数,仅端到端 smoke(验证 JSON/绘图/口径),非结论用
        base.update(n_layers=3, clusters=[3, 3], steps=60, an_steps=60,
                    n_total=4000, n_sample=1500, n_fit=4000, log_every=30)

    points = budget_points()
    if args.only:
        subs = [s.strip() for s in args.only.split(",") if s.strip()]
        points = [p for p in points if any(s in p[0] for s in subs)]

    # oracle 上限(用第一个 seed 的数据,不随预算变)
    d0 = build_data({**base, "seed": 0})
    ceils = layer_ceilings(d0[3], d0[0])
    print("\n[oracle 上限] 逐层 joint_LL 上限(k-NN 熵, -H):",
          "  ".join(f"L{i}={c:.3f}" for i, c in enumerate(ceils)), flush=True)

    all_res = {}
    out_json = REPORTS / "budget_sweep_layered_metrics.json"
    # resume:加载已完成点,跳过(除非 --fresh)
    if out_json.exists() and not args.fresh:
        try:
            prev = json.loads(out_json.read_text())
            all_res = prev.get("points", {})
            if all_res:
                print(f"[resume] 已加载 {len(all_res)} 个完成点: {list(all_res)}", flush=True)
        except Exception as e:
            print(f"[resume] 读取旧 JSON 失败,重新开始: {e}", flush=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    for label, ov, r5, r7 in points:
        prev_seeds = all_res.get(label, {}).get("seeds", []) if not args.fresh else []
        seed_rows = list(prev_seeds)
        if len(seed_rows) >= args.seeds:
            print(f"[skip] {label} 已有 {len(seed_rows)}≥{args.seeds} seed", flush=True)
            continue
        for sd in range(len(seed_rows), args.seeds):
            cfg = {**base, **ov, "seed": sd}
            data = d0 if (sd == 0 and cfg == {**base, "seed": 0}) else build_data(cfg)
            t0 = time.perf_counter()
            res = run_point(cfg, data, run_r5=r5, run_r7=r7)
            print(f"[{label}] seed={sd} r5={'y' if r5 else '-'} r7={'y' if r7 else '-'} "
                  f"用时 {time.perf_counter()-t0:.1f}s "
                  f"params_r5={res.get('params_r5','-')} params_r7={res.get('params_r7','-')}",
                  flush=True)
            for row in res["rows"]:
                print(f"    L{row['li']}(K={row['K']}): "
                      + (f"LL_r5={row.get('ll_r5', float('nan')):.3f} fro_r5={row.get('fro_r5', float('nan')):.3f}  " if r5 else "")
                      + (f"LL_r7={row.get('ll_r7', float('nan')):.3f} fro_r7={row.get('fro_r7', float('nan')):.3f}" if r7 else ""),
                      flush=True)
            seed_rows.append(res)
            # checkpoint:每 (点,seed) 完成即落盘(抗后台被杀)
            all_res[label] = dict(overrides=ov, run_r5=r5, run_r7=r7, seeds=seed_rows)
            payload = dict(config=base, ceilings=ceils, points=all_res)
            out_json.write_text(json.dumps(payload, indent=2, default=float))
            print(f"[ckpt] 写出 {label} seed={sd} → {out_json}", flush=True)

    payload = dict(config=base, ceilings=ceils, points=all_res)
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[写出] {out_json}", flush=True)

    try:
        plot_and_report(payload)
    except Exception as e:  # 绘图失败不影响数据落地
        print(f"[warn] 绘图/报告失败: {e}", flush=True)

    if args.baselines:
        run_baselines(base, all_res)


def run_baselines(base: dict, all_res: dict):
    """等预算全局基线:budget = 已扫描分层链的**最大**参数量(点)。"""
    from simple_ttns_l2.experiments import baselines_big
    # 取所有点里参数量最大的作等预算目标(优先 params_r7,退 params_r5)
    best_np, best_lab = 0, None
    for lab, pt in all_res.items():
        s0 = pt["seeds"][0]
        np_ = s0.get("params_r7") or s0.get("params_r5") or 0
        if np_ > best_np:
            best_np, best_lab = np_, lab
    if best_np <= 0:
        print("[baselines] 无可用参数量,跳过", flush=True)
        return
    cfg = {**base, "seed": 0, "budget": int(best_np), "rmax": 60, "n_sample": 1200}
    print(f"\n[baselines] 等预算 budget={best_np}(=最大分层链点 '{best_lab}' 参数量); "
          f"corr 采样量降到 1200 避免内存 swap", flush=True)
    rows, timings, fj_ll, nparams = baselines_big.run(cfg)
    baselines_big.print_table(rows, timings, fj_ll, nparams)
    payload = dict(budget=int(best_np), budget_from=best_lab, rows=rows, fj_ll=fj_ll, nparams=nparams)
    (REPORTS / "budget_sweep_baselines_metrics.json").write_text(
        json.dumps(payload, indent=2, default=float))
    print(f"[写出] {REPORTS/'budget_sweep_baselines_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
