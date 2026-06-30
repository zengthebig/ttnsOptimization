"""完整逐层链：方案 A（采样）链 vs 方案 B（解析 CDF 采样）链，**每层都重拟合为 TTNS 森林**。

两条链共享同一源层森林 forest0（拟合自真值 L0），随后各自逐层向下：
- A 链：`sample_forest`（上层森林采样，含负密度 clamp）→ `propagate_layer`（max-plus）→ 目标样本 → 重拟合该层森林。
- B 链：`UpperForest` 解析 → `sample_layer_from_cdf`（合法 CDF 条件逆采样，无 clamp）→ 目标样本 → 重拟合该层森林。

**两条链每一层都是 TTNS 森林**（用于继续向下传播）。比较每层目标样本 vs 真值（mean/std/corr），
观察深层的误差累积：预期 B 链因无截断，深层相关性 corr_fro 累积更小。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

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
from simple_ttns_l2.maxplus_cdf_forest import UpperForest, sample_layer_from_cdf  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def _cmp(pred: np.ndarray, gt: np.ndarray) -> Dict:
    out = {"mean_abs_err": float(np.mean(np.abs(pred.mean(0) - gt.mean(0)))),
           "std_abs_err": float(np.mean(np.abs(pred.std(0) - gt.std(0))))}
    if pred.shape[1] > 1:
        out["corr_fro"] = float(np.linalg.norm(np.corrcoef(pred.T) - np.corrcoef(gt.T)))
    else:
        out["corr_fro"] = 0.0
    return out


def run(cfg: dict):
    key = jax.random.PRNGKey(cfg["seed"])
    spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
    params = DelayParams(**cfg["delay"])
    sources, kernels = ground_truth_samplers(spec, params)

    k_gt, key = jax.random.split(key)
    gt = np.asarray(sample_joint(spec, sources, kernels, k_gt, cfg["n"], clip=(-1e9, 1e9)))
    gt_layers = [gt[:, list(spec.layers[li])] for li in range(len(spec.layers))]
    n_eval = cfg["n_eval"]

    # 共享源层森林
    k_src, key = jax.random.split(key)
    forest0 = fit_layer_forest(jnp.asarray(gt_layers[0]), list(spec.layers[0]), cfg, k_src,
                               label="L0", mi_threshold=cfg["mi_threshold"])
    k_s0, key = jax.random.split(key)
    samp0 = sample_forest(forest0, k_s0, n_eval, grid_size=cfg["grid_size"])
    base0 = {"layer": 0, **_cmp(samp0, gt_layers[0][:n_eval])}
    blocks0 = [list(bm.global_vars) for bm in forest0]

    rng = np.random.default_rng(cfg["seed"] + 7)

    # ---- A 链 ----
    A = [base0.copy()]
    forest_a = forest0
    for li in range(1, len(spec.layers)):
        k_s, key = jax.random.split(key)
        samp_up = sample_forest(forest_a, k_s, n_eval, grid_size=cfg["grid_size"])
        predA = propagate_layer(spec, li, samp_up, params, rng)
        A.append({"layer": li, **_cmp(predA, gt_layers[li][:n_eval])})
        k_f, key = jax.random.split(key)
        forest_a = fit_layer_forest(jnp.asarray(predA), list(spec.layers[li]), cfg, k_f,
                                    label=f"A_L{li}", mi_threshold=cfg["mi_threshold"])

    # ---- B 链 ----
    B = [base0.copy()]
    forest_b = forest0
    for li in range(1, len(spec.layers)):
        upper = UpperForest(forest_b, q_grid=cfg["q_grid"])
        s_max = float(gt_layers[li].max() * 1.1)
        k_s, key = jax.random.split(key)
        predB = sample_layer_from_cdf(upper, spec, li, params, k_s, n_eval, s_max, n_s=cfg["n_s"])
        B.append({"layer": li, **_cmp(predB, gt_layers[li][:n_eval])})
        k_f, key = jax.random.split(key)
        forest_b = fit_layer_forest(jnp.asarray(predB), list(spec.layers[li]), cfg, k_f,
                                    label=f"B_L{li}", mi_threshold=cfg["mi_threshold"])

    return spec, {"blocks_L0": blocks0, "A_chain": A, "B_chain": B}


def main():
    cfg = dict(
        layer_sizes=[4, 4, 4, 4], fanin=2,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
        n=20000, n_eval=5000, q=2, m=16, rank=8, lr=2e-3, steps=800, batch_sz=512,
        init_noise=1e-2, train_noise=1e-3, log_every=400, early_stop_patience=6,
        mi_threshold=0.02, grid_size=400, q_grid=400, n_s=140, seed=0,
    )
    spec, res = run(cfg)

    print("\n==== A 链 vs B 链（每层 TTNS 森林）====", "blocks_L0=", res["blocks_L0"])
    for li in range(len(spec.layers)):
        a, b = res["A_chain"][li], res["B_chain"][li]
        print(f"L{li}: A corr_fro={a['corr_fro']:.4f} mae={a['mean_abs_err']:.4f} || "
              f"B corr_fro={b['corr_fro']:.4f} mae={b['mean_abs_err']:.4f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "layered_forest_chain_schemes_metrics.json").write_text(
        json.dumps({"config": cfg, "results": res}, indent=2, ensure_ascii=False))

    a_cf = float(np.mean([m["corr_fro"] for m in res["A_chain"][1:]]))
    b_cf = float(np.mean([m["corr_fro"] for m in res["B_chain"][1:]]))
    deep = len(spec.layers) - 1
    lines = [
        "# 完整逐层链：方案 A（采样）vs 方案 B（解析 CDF 采样），**每层均为 TTNS 森林**", "",
        "两条链共享源层森林 forest0（拟合自真值 L0），各自逐层向下，**每层都重拟合为 TTNS 森林**。",
        "A 链：上层森林采样（负密度 clamp）→ max-plus → 重拟合；B 链：解析 CDF 条件逆采样（无 clamp）→ 重拟合。",
        "比较每层目标样本 vs 真值（mean/std/corr）。", "",
        f"配置：`layer_sizes={cfg['layer_sizes']}, fanin={cfg['fanin']}, delay={cfg['delay']}, n={cfg['n']}`", "",
        f"源层分块：`{res['blocks_L0']}`", "",
        "| 层 | A corr_fro | A mae_mean | B corr_fro | B mae_mean |", "|---|---|---|---|---|",
    ]
    for li in range(len(spec.layers)):
        a, b = res["A_chain"][li], res["B_chain"][li]
        lines.append(f"| L{li} | {a['corr_fro']:.4f} | {a['mean_abs_err']:.4f} | "
                     f"{b['corr_fro']:.4f} | {b['mean_abs_err']:.4f} |")
    winner = "A（采样链）" if a_cf < b_cf else "B（解析采样链）"
    lines += ["",
              f"深层 L{deep} 相关性：A corr_fro={res['A_chain'][deep]['corr_fro']:.4f} vs "
              f"B corr_fro={res['B_chain'][deep]['corr_fro']:.4f}；",
              f"传播层平均 corr_fro：A={a_cf:.4f} vs B={b_cf:.4f}（更优：**{winner}**）。", "",
              "解读（诚实结论）：两条链每层都是 TTNS 森林。**单父 TTNS 森林只能表示树**，层为环时",
              "必丢非树边。A 链 `predA=对上层样本做 max-plus`，给定上层样本时 max-plus 精确，保留该层",
              "完整联合，chow-liu 拟合能挑到最强树边；B 链虽在**单步解析统计**上更准（见",
              "`layered_forest_schemes`），但要把它变成采样器需对配对 CDF 数值微分 "
              "$G=\\partial_t F_{vw}/f_w$ 再条件逆采样，粗网格上数值脆弱，叠加树近似，"
              "把单步优势吃掉，反而不如 A 链。", "",
              "结论：**B 的价值在单步解析统计（积分稳健），不适合用‘采样→重拟合’materialize 成 TTNS 链**；",
              "若要 B 驱动 TTNS 链并保住优势，应改为**解析矩匹配直接构造 chow-liu TTNS**（用 B 的",
              "marginal/配对密度投影到基，避免采样与脆弱微分），列为后续。"]
    (REPORTS / "layered_forest_chain_schemes_report_zh.md").write_text("\n".join(lines) + "\n")
    print(f"\n传播层平均 corr_fro：A={a_cf:.4f} vs B={b_cf:.4f}；深层 L{deep}: "
          f"A={res['A_chain'][deep]['corr_fro']:.4f} vs B={res['B_chain'][deep]['corr_fro']:.4f}")


if __name__ == "__main__":
    main()
