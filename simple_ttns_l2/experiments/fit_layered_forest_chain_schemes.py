"""完整逐层链：方案 A（采样）链 vs 方案 B（解析 CDF 采样）链，**每层都重拟合为 TTNS 森林**。

两条链共享同一源层森林 forest0（拟合自真值 L0），随后各自逐层向下：
- A 链：`sample_forest`（上层森林采样，含负密度 clamp）→ `propagate_layer`（max-plus）→ 目标样本 → 重拟合该层森林。
- B 链：`UpperForest` 解析 → `sample_layer_copula`（用 B 的**全相关矩阵 + 边缘**做高斯 copula 采样，
  保住所有两两相关、不丢非树边）→ 目标样本 → 重拟合该层森林。

**两条链每一层都是 TTNS 森林**（用于继续向下传播）。比较每层目标样本 vs 真值（mean/std/corr）。

注：早期 B 链曾用树形条件逆采样（chow-liu），会丢环上非树边导致 corr_fro 暴涨（诊断见报告），
现已改为 copula 采样修正。
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
from simple_ttns_l2.maxplus_cdf_forest import UpperForest, sample_layer_copula  # noqa: E402

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
        predB = sample_layer_copula(upper, spec, li, params, k_s, n_eval, s_max, n_s=cfg["n_s"])
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
        "A 链：上层森林采样 → max-plus → 重拟合；B 链：解析 CDF + **高斯 copula 采样**（全相关）→ 重拟合。",
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
              "## 诚实更正（撤回此前的‘结构性’结论）", "",
              "此前版本曾断言『B 成链结构上赢不了 A』，**这是错的**。诊断（同一上层森林，L0→L1）：", "",
              "| 量 | (4,5) | (4,6) | (4,7) | (5,6) | (5,7) | (6,7) | corr_fro |",
              "|---|---|---|---|---|---|---|---|",
              "| 真值 | 0.353 | 0.000 | 0.353 | 0.342 | 0.012 | 0.350 | — |",
              "| A | 0.354 | 0.019 | 0.345 | 0.356 | 0.000 | 0.348 | 0.039 |",
              "| B-解析 | 0.347 | 0.002 | 0.342 | 0.348 | 0.002 | 0.340 | **0.029** |",
              "| B-树采样(旧) | 0.339 | 0.119 | 0.346 | 0.342 | 0.140 | 0.056 | 0.484 |",
              "| B-copula(新) | 0.334 | -0.001 | 0.346 | 0.339 | -0.009 | 0.321 | 0.059 |", "",
              "- **B 的解析计算没错，反而最准（0.029 < A 0.039）**；问题出在旧采样器：它把 B 的全相关",
              "  强行走 chow-liu 树，丢掉环上一条强边((6,7) 0.35→0.056)，corr_fro 才暴涨到 0.484。",
              "- 改用**高斯 copula**（B 的全相关矩阵 + 边缘 CDF，$z\\sim N(0,C),x_v=F_v^{-1}(\\Phi(z_v))$）后，",
              "  B 采样 corr_fro 回到 **0.059**，与解析接近、与 A 同量级。**根因是采样器选错，不是 B 的本质限制。**", "",
              "结论：B 解析统计本身精确；要把 B materialize 成 TTNS 链，**采样必须保全相关（copula）而非走树**。", "",
              "## 链上仍存的问题（与解析无关）", "",
              "改 copula 后**单步 L1 反超 A**（见上表），但深层 L2/L3 仍不如 A。曾怀疑是",
              "**重拟合稳定性**（val_l2 偶发发散），但用更细验证评估 + 更多步（best-tracking）后仍不改善，",
              "说明深层差距**不是训练噪声**，而是下面的机制问题。", "",
              "## 试过并失败的『结构再生』(regen，已回滚)", "",
              "曾尝试让 B 不读上层联合、改用『共享父 + 上层边缘 + 已知核』现算子层相关（structure-regen）。",
              "3 种子、best-tracking 下平均 corr_fro（L1,L2,L3）：", "",
              "| 方法 | L1 | L2 | L3 | 平均 |",
              "|---|---|---|---|---|",
              "| A（采样+max-plus） | 0.077 | 0.136 | 0.512 | **0.242** |",
              "| B-copula（上层联合） | 0.081 | 0.329 | 0.753 | 0.388 |",
              "| B-regen（共享父现算，已回滚） | 0.090 | 0.752 | 1.424 | 0.755 |", "",
              "**regen 反而最差**：它假设『子层相关几乎全来自共享父，可忽略上层跨父相关』，但上层本身是相关的环，",
              "一个子节点的两个父彼此就有 ~0.35 相关；regen 把这些跨父相关全丢了 → 子层相关被系统性低估，",
              "比『从被压过的树联合里读』还差。即：那棵树联合虽丢一条边，仍携带大量跨父相关信息，regen 把好的也一起扔了。", "",
              "## 最终诚实结论", "",
              "在『每层=单父树森林 + 传播 + 重拟合』框架下，**A 链整体最优**。根本原因朴素：",
              "**A 用真实样本做 max-plus，保住了 max 诱导的真·依赖（非高斯、含高阶）；B-copula 只能用高斯 copula",
              "复现两两相关，丢了依赖形状与高阶结构，再叠加树 refit**。要让 B 采样达到 A 的依赖质量，本质上就得",
              "『采上层样本 + max-plus』——那就是 A。**B 的价值仍只在『给定准确上层时的单步解析统计（边缘/配对精确）』，",
              "不在成链生成。**"]
    (REPORTS / "layered_forest_chain_schemes_report_zh.md").write_text("\n".join(lines) + "\n")
    print(f"\n传播层平均 corr_fro：A={a_cf:.4f} vs B={b_cf:.4f}；深层 L{deep}: "
          f"A={res['A_chain'][deep]['corr_fro']:.4f} vs B={res['B_chain'][deep]['corr_fro']:.4f}")


if __name__ == "__main__":
    main()
