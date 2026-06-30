"""方案 A：max-plus 多层 DAG 的"采样 + 逐层重拟合"传播，并对比真值。

流程：
1. 真值：祖先 max-plus 采样得到全联合样本（各层节点的"真分布"）。
2. 传播（方案 A，模型侧仅单父 TTNS）：
   - 第 0 层（源）：用真值源样本拟合 TTNS_0。
   - 第 li 层：从已拟合的 TTNS_{li-1} 采样 → 施加 max-plus（采新 delay）→
     得到第 li 层预测样本 → 用 chow-liu 单父 TTNS 重拟合 TTNS_li。
3. 验证：逐层比较"预测样本/拟合密度"与"真值样本"。
   - 关键对照：从**真值上层样本**传播（而非从 TTNS 采样）= 传播本身可达的上界，
     用于隔离 "TTNS 拟合+采样" 引入的误差。

指标（越小越好）：
- W1_marg：各变量一维 Wasserstein-1 距离（预测 vs 真值）的均值。
- corr_fro：相关矩阵 Frobenius 距离（预测 vs 真值）。
"""

from __future__ import annotations

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

from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.experiments.fit_diamond_dag_vs_tree import train_tree_l2  # noqa: E402
from simple_ttns_l2.ttns_sampler import sample_ttns  # noqa: E402
from simple_ttns_l2.dag_pipeline import MultiLayerSpec, build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def w1_1d(a: np.ndarray, b: np.ndarray) -> float:
    """一维经验 Wasserstein-1：对等量样本即 mean|sorted(a)-sorted(b)|。"""
    n = min(len(a), len(b))
    sa = np.sort(np.asarray(a))[:n] if len(a) == n else np.quantile(np.asarray(a), np.linspace(0, 1, n))
    sb = np.sort(np.asarray(b))[:n] if len(b) == n else np.quantile(np.asarray(b), np.linspace(0, 1, n))
    return float(np.mean(np.abs(sa - sb)))


def marg_w1(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean([w1_1d(pred[:, j], gt[:, j]) for j in range(pred.shape[1])]))


def corr_fro(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape[1] < 2:
        return 0.0
    cp = np.corrcoef(pred.T)
    cg = np.corrcoef(gt.T)
    return float(np.linalg.norm(cp - cg))


def fit_layer_ttns(layer_x: jnp.ndarray, cfg: dict, key, label: str):
    """对单层样本拟合 chow-liu 单父 TTNS（L2）。返回 (ttns, bases, parent, summary)。"""
    n_dims = layer_x.shape[1]
    bases = build_bases(layer_x, cfg["q"], cfg["m"])
    gram = vmap(type(bases).l2_integral)(bases)
    basis_integrals = vmap(type(bases).integral)(bases)
    if n_dims >= 2:
        parent = list(estimate_chow_liu_tree(np.asarray(layer_x), n_bins=16, root=0).parent)
    else:
        parent = [0]
    k_init, k_tr = jax.random.split(key)
    split = int(0.8 * layer_x.shape[0])
    tr, val = layer_x[:split], layer_x[split:]
    t0 = init_ttns_from_rank1(k_init, bases, tr, parent, cfg["rank"], cfg["init_noise"])
    ttns, summ = train_tree_l2(
        t0, parent, bases, tr, val, gram, basis_integrals,
        key=k_tr, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
        normalize_every=1, log_every=cfg["log_every"], label=label,
        train_noise=cfg["train_noise"], early_stop_patience=cfg["early_stop_patience"],
    )
    return ttns, bases, parent, summ


def run(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    n_layers = len(spec.layers)

    # ---- 真值全联合样本（祖先 max-plus 采样）----
    sources, kernels = ground_truth_samplers(spec, params)
    k_gt, key = jax.random.split(key)
    wide = (-1e9, 1e9)  # max-plus 值会增长，不裁剪到 [0,1]
    gt = np.asarray(sample_joint(spec, sources, kernels, k_gt, cfg["n"], clip=wide))
    gt_layers = [gt[:, list(spec.layers[li])] for li in range(n_layers)]

    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))

    results: List[Dict] = []
    ttns_list, bases_list, parent_list = [None] * n_layers, [None] * n_layers, [None] * n_layers

    # ---- 第 0 层：拟合源 ----
    k_l0, k_s0, key = jax.random.split(key, 3)
    t0 = time.perf_counter()
    ttns_list[0], bases_list[0], parent_list[0], s0 = fit_layer_ttns(
        jnp.asarray(gt_layers[0]), cfg, k_l0, "layer0_src")
    pred0 = np.asarray(sample_ttns(ttns_list[0], bases_list[0], parent_list[0], k_s0,
                                   n=cfg["n"], grid_size=cfg["grid_size"]))
    results.append({"layer": 0, "n_vars": len(spec.layers[0]), "fit_val_l2": s0["best_val_l2"],
                    "W1_marg_fit": marg_w1(pred0, gt_layers[0]),
                    "corr_fro_fit": corr_fro(pred0, gt_layers[0]),
                    "fit_sec": time.perf_counter() - t0})

    # ---- 逐层传播 + 重拟合 ----
    for li in range(1, n_layers):
        k_samp, k_fit, key = jax.random.split(key, 3)
        # 方案 A：从上一层拟合好的 TTNS 采样
        prev_samp = np.asarray(sample_ttns(
            ttns_list[li - 1], bases_list[li - 1], parent_list[li - 1], k_samp,
            n=cfg["n"], grid_size=cfg["grid_size"]))
        pred = propagate_layer(spec, li, prev_samp, params, rng)

        # 对照：从真值上层样本传播（隔离 TTNS 拟合/采样误差）
        ref = propagate_layer(spec, li, gt_layers[li - 1], params, rng)

        t1 = time.perf_counter()
        ttns_list[li], bases_list[li], parent_list[li], sl = fit_layer_ttns(
            jnp.asarray(pred), cfg, k_fit, f"layer{li}")

        rec = {
            "layer": li,
            "n_vars": len(spec.layers[li]),
            "fit_val_l2": sl["best_val_l2"],
            "W1_marg_pred": marg_w1(pred, gt_layers[li]),       # 方案A传播样本 vs 真值
            "W1_marg_ref": marg_w1(ref, gt_layers[li]),         # 真值上层传播 vs 真值（上界参考）
            "corr_fro_pred": corr_fro(pred, gt_layers[li]),
            "corr_fro_ref": corr_fro(ref, gt_layers[li]),
            "fit_sec": time.perf_counter() - t1,
        }
        results.append(rec)
        print(f"\n[layer {li}] W1_marg pred={rec['W1_marg_pred']:.4f} ref={rec['W1_marg_ref']:.4f} | "
              f"corr_fro pred={rec['corr_fro_pred']:.4f} ref={rec['corr_fro_ref']:.4f}", flush=True)

    return spec, results


def main():
    cfg = dict(
        layer_sizes=[4, 4, 4],
        fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.4, node_lo=0.0, node_hi=0.4),
        n=15000,
        q=2, m=16, rank=8, lr=2e-3, steps=800, batch_sz=512,
        init_noise=1e-2, train_noise=1e-3, log_every=200, early_stop_patience=6,
        grid_size=400, seed=0,
    )
    spec, results = run(cfg)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "maxplus_propagation_metrics.json").write_text(
        json.dumps({"config": cfg, "results": results}, indent=2, ensure_ascii=False))

    lines = [
        "# Max-plus 多层 DAG 传播（方案 A：采样 + 逐层重拟合）", "",
        "数据依赖为多层 DAG，模型侧仅用**单父 TTNS**；多来源聚合为 max-plus：",
        "$$x_v=\\max_{u\\in\\mathrm{pa}(v)}(x_u+e_{uv})+d_v.$$", "",
        "传播逐层进行：从上一层拟合好的 TTNS 采样 → 施加 max-plus（采新 delay）→",
        "重拟合下一层单父 TTNS。`pred` = 方案 A 传播样本，`ref` = 从真值上层样本传播",
        "（传播可达上界，隔离 TTNS 拟合+采样误差）。指标越小越好。", "",
        f"配置：`layer_sizes={cfg['layer_sizes']}, fanin={cfg['fanin']}, delay={cfg['delay']}`", "",
        f"`q={cfg['q']}, m={cfg['m']}, rank={cfg['rank']}, n={cfg['n']}, steps={cfg['steps']}`", "",
        "| 层 | 变量数 | 拟合val_l2 | W1_marg(pred) | W1_marg(ref) | corr_fro(pred) | corr_fro(ref) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["layer"] == 0:
            lines.append(f"| 0(源) | {r['n_vars']} | {fmt(r['fit_val_l2'])} | "
                         f"{r['W1_marg_fit']:.4f} | - | - | - |")
        else:
            lines.append(f"| {r['layer']} | {r['n_vars']} | {fmt(r['fit_val_l2'])} | "
                         f"{r['W1_marg_pred']:.4f} | {r['W1_marg_ref']:.4f} | "
                         f"{r['corr_fro_pred']:.4f} | {r['corr_fro_ref']:.4f} |")
    lines += ["",
              "解读：`pred` 接近 `ref` 说明 TTNS 表示+采样在传播链上保真；`pred` 与 `ref` 的差距即",
              "单父 TTNS 对各层联合密度的压缩损失 + 采样截断误差的累积。"]
    (REPORTS / "maxplus_propagation_report_zh.md").write_text("\n".join(lines) + "\n")

    print("\n==== 汇总 ====")
    for r in results:
        print(r)


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else str(v)


if __name__ == "__main__":
    main()
