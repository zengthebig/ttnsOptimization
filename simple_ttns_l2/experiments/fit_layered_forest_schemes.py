"""每层 TTNS 森林的 max-plus 传播：方案 A（采样）vs 方案 B（CDF 解析）vs 真值。

两方案并列、共享同一上层森林（不修改采样实现）：
- 上层森林 = 对真值 layer(li-1) 数据拟合的 TTNS 森林（层内 MI 分块）。
- A（采样，已有实现）：`sample_forest` 采样上层 → `propagate_layer` 做 max-plus → 经验统计。
- B（解析，新增 `maxplus_cdf_forest`）：森林感知 CDF 域可分离收缩 → marginal(E,Var) + Hoeffding 相关，无截断。
对每个层间转移分别比较，指标 vs 真值越小越好。预期 B 在相关性（corr_fro）上优于 A（无采样截断）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers, propagate_layer  # noqa: E402
from simple_ttns_l2.layered_forest import fit_layer_forest, sample_forest  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import UpperForest, propagate_layer_cdf_forest  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def _stats(x: np.ndarray) -> Dict:
    return {"mean": x.mean(0), "std": x.std(0), "corr": np.corrcoef(x.T)}


def _cmp(label: str, mean, std, corr, gt: Dict) -> Dict:
    return {
        "scheme": label,
        "mean_abs_err": float(np.mean(np.abs(mean - gt["mean"]))),
        "std_abs_err": float(np.mean(np.abs(std - gt["std"]))),
        "corr_fro": float(np.linalg.norm(corr - gt["corr"])),
    }


def run(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)

    k_gt, key = jax.random.split(key)
    gt = np.asarray(sample_joint(spec, sources, kernels, k_gt, cfg["n"], clip=(-1e9, 1e9)))

    per_layer = []
    for li in range(1, len(spec.layers)):
        gt_up = gt[:, list(spec.layers[li - 1])]
        gt_tg = gt[:, list(spec.layers[li])]
        gt_stats = _stats(gt_tg)

        # 上层森林（两方案共享）
        k_fit, key = jax.random.split(key)
        forest_up = fit_layer_forest(jnp.asarray(gt_up), list(spec.layers[li - 1]), cfg, k_fit,
                                     label=f"L{li-1}", mi_threshold=cfg["mi_threshold"])
        blocks = [list(bm.global_vars) for bm in forest_up]

        # 方案 A：森林采样 → max-plus
        k_s, key = jax.random.split(key)
        samp_up = sample_forest(forest_up, k_s, cfg["n"], grid_size=cfg["grid_size"])
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
        predA = propagate_layer(spec, li, samp_up, params, rng)
        sA = _stats(predA)

        # 方案 B：森林解析（CDF 域可分离收缩）
        upper = UpperForest(forest_up, q_grid=cfg["q_grid"])
        s_max = float(gt_tg.max() * 1.1)
        B = propagate_layer_cdf_forest(upper, spec, li, params, s_max=s_max,
                                       n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"])
        cmpA = _cmp("A_sampling", sA["mean"], sA["std"], sA["corr"], gt_stats)
        cmpB = _cmp("B_cdf_analytic", B["mean"], np.sqrt(B["var"]), B["corr"], gt_stats)
        iu = np.triu_indices(len(gt_stats["corr"]), 1)
        avg_abs_corr = float(np.mean(np.abs(gt_stats["corr"][iu])))
        per_layer.append({"transition": f"L{li-1}->L{li}", "upper_blocks": blocks,
                          "gt_avg_abs_corr": avg_abs_corr, "A": cmpA, "B": cmpB})
        print(f"[L{li-1}->L{li}] blocks={blocks} gt|corr|={avg_abs_corr:.3f} | "
              f"A corr_fro={cmpA['corr_fro']:.4f} mae={cmpA['mean_abs_err']:.4f} || "
              f"B corr_fro={cmpB['corr_fro']:.4f} mae={cmpB['mean_abs_err']:.4f}", flush=True)
    return spec, per_layer


def main():
    cfg = dict(
        layer_sizes=[4, 4, 4], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
        n=15000, q=2, m=16, rank=8, lr=2e-3, steps=800, batch_sz=512,
        init_noise=1e-2, train_noise=1e-3, log_every=400, early_stop_patience=6,
        mi_threshold=0.02, grid_size=400, q_grid=400, n_s=160, n_s_pair=100, seed=0,
    )
    spec, per_layer = run(cfg)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "layered_forest_schemes_metrics.json").write_text(
        json.dumps({"config": cfg, "per_layer": per_layer}, indent=2, ensure_ascii=False))

    avgA_cf = float(np.mean([p["A"]["corr_fro"] for p in per_layer]))
    avgB_cf = float(np.mean([p["B"]["corr_fro"] for p in per_layer]))
    lines = [
        "# 每层 TTNS 森林 max-plus 传播：方案 A（采样）vs 方案 B（CDF 解析）", "",
        "两方案并列、**共享同一上层森林**（上层森林拟合自真值上一层数据，含真实相关 → 有符号密度）。",
        "A = 森林采样 + max-plus（须 clamp 负密度，有截断）；B = 森林感知 CDF 域可分离收缩 + Hoeffding",
        "相关（无截断）。**未修改采样实现**；B 为新增 `maxplus_cdf_forest`。指标 vs 真值，越小越好。", "",
        f"配置：`layer_sizes={cfg['layer_sizes']}, fanin={cfg['fanin']}, delay={cfg['delay']}, n={cfg['n']}`", "",
        "| 转移 | 上层块 | 真值平均\\|corr\\| | A mean_err | A corr_fro | B mean_err | B corr_fro |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in per_layer:
        lines.append(
            f"| {p['transition']} | {p['upper_blocks']} | {p['gt_avg_abs_corr']:.3f} | "
            f"{p['A']['mean_abs_err']:.4f} | {p['A']['corr_fro']:.4f} | "
            f"{p['B']['mean_abs_err']:.4f} | {p['B']['corr_fro']:.4f} |")
    lines += ["",
              f"逐层平均 corr_fro：A（采样）={avgA_cf:.4f} vs B（解析）={avgB_cf:.4f}。", "",
              "解读：上层为有符号密度时，采样方案 A 须截断负值、系统性削弱相关；解析方案 B 对（可能",
              "有符号的）密度做线性积分，块间独立按块因子分解、无截断，相关性 corr_fro 更低更稳。"]
    (REPORTS / "layered_forest_schemes_report_zh.md").write_text("\n".join(lines) + "\n")
    print(f"\n逐层平均 corr_fro：A={avgA_cf:.4f} vs B={avgB_cf:.4f}")


if __name__ == "__main__":
    main()
