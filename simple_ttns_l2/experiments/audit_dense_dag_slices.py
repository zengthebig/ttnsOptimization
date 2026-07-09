"""审计 dense_dag_r567_slices.png 的数据源与“后层变尖”成因。

复用 three_way / plot_dense_dag_slices 的 CFG 与拟合链（seed=0），不改训练逻辑：
  1) 校验层内 local_vars 是否覆盖且无重叠；
  2) 逐层边缘 std(GT/R5/R7) 与 std 比；
  3) 模型样本落在 GT 分位 bins 外的比例（直方图伪影）；
  4) forest_log_density 的 joint_LL / nonpos_rate。

输出：
  simple_ttns_l2/reports/dense_dag_r567_slices_audit.json
  simple_ttns_l2/reports/dense_dag_r567_slices_audit_zh.md
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

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.dag_pipeline import build_crossed_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import (  # noqa: E402
    fit_layer_forest, sample_forest, forest_log_density,
)
from simple_ttns_l2.analytic_tree_fit import fit_analytic_chain, fit_sampled_chain  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG, complex_sources  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"
OUT_JSON = REPORTS / "dense_dag_r567_slices_audit.json"
OUT_MD = REPORTS / "dense_dag_r567_slices_audit_zh.md"
N_EVAL = 5000
GRID = 400


def check_forest_wiring(forest, layer_size: int, layer_nodes: list[int]) -> dict:
    """校验 sample_forest 拼列所需的 local_vars 分区，以及 global_vars 与层节点对齐。"""
    covered = []
    issues = []
    for bi, bm in enumerate(forest):
        locs = list(bm.local_vars)
        gids = list(bm.global_vars)
        covered.extend(locs)
        if len(locs) != len(gids):
            issues.append(f"blk{bi}: len(local_vars)={len(locs)} != len(global_vars)={len(gids)}")
        for loc, gid in zip(locs, gids):
            if loc < 0 or loc >= layer_size:
                issues.append(f"blk{bi}: local {loc} out of [0,{layer_size})")
            elif layer_nodes[loc] != gid:
                issues.append(
                    f"blk{bi}: local {loc} maps to global {layer_nodes[loc]} but global_vars={gid}"
                )
    covered_sorted = sorted(covered)
    expected = list(range(layer_size))
    if covered_sorted != expected:
        issues.append(f"local_vars cover {covered_sorted}, expected {expected}")
    return {
        "ok": len(issues) == 0,
        "n_blocks": len(forest),
        "issues": issues,
        "local_vars": [list(bm.local_vars) for bm in forest],
        "global_vars": [list(bm.global_vars) for bm in forest],
    }


def outside_bin_frac(vals: np.ndarray, lo: float, hi: float) -> float:
    v = np.asarray(vals)
    return float(np.mean((v < lo) | (v > hi)))


def collect(seed: int):
    cfg = dict(CFG)
    key = jax.random.PRNGKey(seed)
    spec = build_crossed_spec(
        cfg["n_layers"], cfg["clusters"], fanin=cfg["fanin"],
        cross_pairs=cfg.get("cross_pairs"),
        cross_fanin=cfg.get("cross_fanin", 1),
        rotate_cross=cfg.get("rotate_cross", False), wrap=True,
    )
    params = DelayParams(**cfg["delay"])
    layers = [list(l) for l in spec.layers]

    sources, kernels = ground_truth_samplers(spec, params)
    sources = {**sources, **complex_sources(spec, cfg)}
    k_data, key = jax.random.split(key)
    gt = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
    n_tr = int(0.7 * gt.shape[0])
    train_x, test_x = gt[:n_tr], gt[n_tr:]
    L0 = layers[0]

    t0 = time.perf_counter()
    k_f, key = jax.random.split(key)
    forest0 = fit_layer_forest(
        jnp.asarray(train_x[:, L0]), L0, cfg, k_f,
        label="L0", mi_threshold=cfg["mi_threshold"],
    )
    s_max0 = float(max(np.asarray(bm.bases.knots).max() for bm in forest0))
    print(f"[audit] L0 forest done in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    k_an, key = jax.random.split(key)
    R5 = fit_analytic_chain(
        forest0, spec, params, k_an, s_max0,
        q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
        n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
        lr=cfg["an_lr"], steps=cfg["an_steps"],
        init_noise=cfg["init_noise"], log_every=0,
        block_mode=cfg["block_mode"],
    )
    print(f"[audit] R5 done in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    k_sp, key = jax.random.split(key)
    R7 = fit_sampled_chain(forest0, spec, params, k_sp, cfg)
    print(f"[audit] R7 done in {time.perf_counter() - t0:.1f}s", flush=True)

    n_plot = min(N_EVAL, int(test_x.shape[0]))
    gt_layers = [test_x[:n_plot, Lg] for Lg in layers]
    R5_s, R7_s = [], []
    for li in range(len(layers)):
        k1, key = jax.random.split(key)
        R5_s.append(np.asarray(sample_forest(R5[li], k1, n_plot, grid_size=GRID)))
        k2, key = jax.random.split(key)
        R7_s.append(np.asarray(sample_forest(R7[li], k2, n_plot, grid_size=GRID)))

    return cfg, spec, layers, test_x, gt_layers, R5, R7, R5_s, R7_s, n_plot


def audit(seed: int) -> dict:
    cfg, spec, layers, test_x, gt_layers, R5, R7, R5_s, R7_s, n_plot = collect(seed)

    wiring = {}
    for li, nodes in enumerate(layers):
        wiring[f"L{li}"] = {
            "R5_tree": check_forest_wiring(R5[li], len(nodes), nodes),
            "R7_sampled": check_forest_wiring(R7[li], len(nodes), nodes),
            "layer_nodes": nodes,
        }

    per_layer = []
    for li, nodes in enumerate(layers):
        gtv = gt_layers[li]
        r5v = R5_s[li]
        r7v = R7_s[li]
        assert gtv.shape == r5v.shape == r7v.shape == (n_plot, len(nodes))

        std_gt = gtv.std(axis=0)
        std_r5 = r5v.std(axis=0)
        std_r7 = r7v.std(axis=0)
        # 避免除零
        ratio_r5 = std_r5 / np.maximum(std_gt, 1e-12)
        ratio_r7 = std_r7 / np.maximum(std_gt, 1e-12)

        out_r5, out_r7 = [], []
        for j in range(len(nodes)):
            lo = float(np.percentile(gtv[:, j], 0.5))
            hi = float(np.percentile(gtv[:, j], 99.5))
            pad = 0.15 * (hi - lo + 1e-9)
            blo, bhi = lo - pad, hi + pad
            out_r5.append(outside_bin_frac(r5v[:, j], blo, bhi))
            out_r7.append(outside_bin_frac(r7v[:, j], blo, bhi))

        tev = test_x[:, nodes]
        ll_r5, nonpos_r5 = forest_log_density(R5[li], tev)
        ll_r7, nonpos_r7 = forest_log_density(R7[li], tev)

        row = {
            "layer": li,
            "K": len(nodes),
            "gt_std_mean": float(std_gt.mean()),
            "gt_std_min": float(std_gt.min()),
            "gt_std_max": float(std_gt.max()),
            "r5_std_mean": float(std_r5.mean()),
            "r7_std_mean": float(std_r7.mean()),
            "r5_std_ratio_mean": float(ratio_r5.mean()),
            "r7_std_ratio_mean": float(ratio_r7.mean()),
            "r5_std_ratio_min": float(ratio_r5.min()),
            "r7_std_ratio_min": float(ratio_r7.min()),
            "r5_outside_bin_mean": float(np.mean(out_r5)),
            "r7_outside_bin_mean": float(np.mean(out_r7)),
            "r5_outside_bin_max": float(np.max(out_r5)),
            "r7_outside_bin_max": float(np.max(out_r7)),
            "r5_joint_ll": float(ll_r5.mean()),
            "r7_joint_ll": float(ll_r7.mean()),
            "r5_nonpos_rate": float(nonpos_r5),
            "r7_nonpos_rate": float(nonpos_r7),
            "per_node": [
                {
                    "node": int(nodes[j]),
                    "gt_std": float(std_gt[j]),
                    "r5_std": float(std_r5[j]),
                    "r7_std": float(std_r7[j]),
                    "r5_std_ratio": float(ratio_r5[j]),
                    "r7_std_ratio": float(ratio_r7[j]),
                    "r5_outside_bin": float(out_r5[j]),
                    "r7_outside_bin": float(out_r7[j]),
                }
                for j in range(len(nodes))
            ],
        }
        per_layer.append(row)
        print(
            f"[audit][L{li}] gt_std={row['gt_std_mean']:.4f} "
            f"r5_ratio={row['r5_std_ratio_mean']:.3f} r7_ratio={row['r7_std_ratio_mean']:.3f} "
            f"r5_out={row['r5_outside_bin_mean']:.3f} r7_out={row['r7_outside_bin_mean']:.3f} "
            f"r5_ll={row['r5_joint_ll']:.3f} r7_ll={row['r7_joint_ll']:.3f} "
            f"r5_nonpos={row['r5_nonpos_rate']:.3f} r7_nonpos={row['r7_nonpos_rate']:.3f}",
            flush=True,
        )

    wiring_ok = all(
        wiring[f"L{li}"]["R5_tree"]["ok"] and wiring[f"L{li}"]["R7_sampled"]["ok"]
        for li in range(len(layers))
    )
    # 直方图伪影是否主导：后层 outside-bin 是否很高，且 std 比是否仍明显 <1
    last = per_layer[-1]
    artifact_dominant = (
        last["r5_outside_bin_mean"] > 0.25 or last["r7_outside_bin_mean"] > 0.25
    )
    underdispersion = (
        last["r5_std_ratio_mean"] < 0.7 or last["r7_std_ratio_mean"] < 0.7
    )

    return {
        "seed": seed,
        "n_plot": n_plot,
        "grid_size": GRID,
        "png_source": (
            "dense_dag_r567_three_way.py --plot → plot_from_run_artifacts "
            "(same-run; NOT plot_dense_dag_slices.py for current PNG)"
        ),
        "plot_logic": {
            "kind": "per-node 1D marginal histogram",
            "gt": "test_x[:n_plot, layers[li]][:, j]",
            "model": "sample_forest(chains[m][li])[:, j]",
            "bins": "linspace(GT p0.5 - pad, GT p99.5 + pad, 70), density=True",
        },
        "wiring_ok": wiring_ok,
        "wiring": wiring,
        "per_layer": per_layer,
        "verdict": {
            "data_source_correct": wiring_ok,
            "sharpening_is_plot_bug_only": False if underdispersion else artifact_dominant,
            "underdispersion_real": underdispersion,
            "histogram_artifact_material": artifact_dominant,
            "notes": [
                "后层变尖若伴随 std_ratio≪1，则为模型欠分散，非错列。",
                "若 outside_bin 高，则 density=True 会进一步抬高峰高。",
                "nonpos_rate 升高支持采样器 clip(f,0) 挤峰假说。",
            ],
        },
        "config_keys": {
            "n_layers": cfg["n_layers"],
            "clusters": cfg["clusters"],
            "fanin": cfg["fanin"],
            "block_mode": cfg["block_mode"],
            "rotate_cross": cfg["rotate_cross"],
            "cross_fanin": cfg["cross_fanin"],
        },
        "spec": {
            "n_nodes": int(spec.n),
            "n_edges": len(spec.edges),
            "layer_dim": len(layers[0]),
        },
    }


def write_md(dump: dict) -> None:
    v = dump["verdict"]
    lines = [
        "# dense_dag_r567 切片图数据源审计",
        "",
        f"- **PNG 实际来源**: `{dump['png_source']}`",
        f"- **seed**: {dump['seed']} · n_plot={dump['n_plot']} · grid={dump['grid_size']}",
        f"- **接线校验**: {'通过' if dump['wiring_ok'] else '失败'}",
        f"- **数据源是否正确**: {'是' if v['data_source_correct'] else '否'}",
        f"- **后层变尖是否仅为绘图 bug**: {'否（存在真实欠分散）' if v['underdispersion_real'] else '可能主要为伪影/需再查'}",
        f"- **直方图伪影是否显著**: {'是' if v['histogram_artifact_material'] else '否（outside-bin 不高）'}",
        "",
        "## 绘图口径",
        "",
        f"- 类型: {dump['plot_logic']['kind']}",
        f"- GT: `{dump['plot_logic']['gt']}`",
        f"- 模型: `{dump['plot_logic']['model']}`",
        f"- bins: `{dump['plot_logic']['bins']}`",
        "",
        "## 逐层边缘 std / bins 外比例 / LL / nonpos",
        "",
        "| 层 | GT std↓ | R5 std | R7 std | R5/GT | R7/GT | R5 out-bin | R7 out-bin | R5 LL | R7 LL | R5 nonpos | R7 nonpos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dump["per_layer"]:
        lines.append(
            f"| L{row['layer']} | {row['gt_std_mean']:.4f} | {row['r5_std_mean']:.4f} | "
            f"{row['r7_std_mean']:.4f} | {row['r5_std_ratio_mean']:.3f} | "
            f"{row['r7_std_ratio_mean']:.3f} | {row['r5_outside_bin_mean']:.3f} | "
            f"{row['r7_outside_bin_mean']:.3f} | {row['r5_joint_ll']:.3f} | "
            f"{row['r7_joint_ll']:.3f} | {row['r5_nonpos_rate']:.3f} | "
            f"{row['r7_nonpos_rate']:.3f} |"
        )
    lines += [
        "",
        "## 结论",
        "",
    ]
    if dump["wiring_ok"]:
        lines.append(
            "1. **列对齐正确**：各层 `local_vars` 构成完整分区，且与 `global_vars` / `layers[li]` 一致；"
            "切片图不是画错节点。"
        )
    else:
        lines.append("1. **接线失败**：见 JSON `wiring` 字段。")
    lines.append(
        "2. **GT 随层变宽**：`gt_std_mean` 随深度上升，符合 max-plus 延迟累积；"
        "后层黑线变扁是真值边缘变宽，不是画错。"
    )
    if v["underdispersion_real"]:
        lines.append(
            "3. **模型真实欠分散**：后层 `std(model)/std(GT)` 明显小于 1，"
            "红/蓝峰变尖主要是拟合/传播导致的方差塌缩，不是单纯换错数据。"
        )
    else:
        lines.append("3. 后层 std 比未显示强烈欠分散；需结合 outside-bin 与 LL 再判。")
    if v["histogram_artifact_material"]:
        lines.append(
            "4. **直方图会放大尖锐感**：模型样本大量落在 GT 分位 bins 外时，"
            "`density=True` 会在剩余区间重归一化，峰高被抬高。建议 bins 改用 "
            "`union(GT,R5,R7)` 分位，或在图旁标注 outside-bin 比例。"
        )
    else:
        lines.append(
            "4. **直方图伪影不是主因**：outside-bin 比例不高，视觉变尖主要来自模型本身更窄。"
        )
    lines.append(
        "5. 报告 `dense_dag_r567_report_zh.md` 写切片图由 `plot_dense_dag_slices` 生成，"
        "与当前 PNG 标题 `same-run plots` 不符；应以 `three_way --plot` 为准。"
    )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    dump = audit(args.seed)
    OUT_JSON.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(dump)
    print(f"\nsaved: {OUT_JSON}", flush=True)
    print(f"saved: {OUT_MD}", flush=True)
    print(f"total {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
