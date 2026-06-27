from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from simple_ttns_l2.experiments.fit_ceiling_two_models import run_fit_ceiling

DEFAULT_SEEDS = (313, 2602, 20260227)


def _aggregate_balanced_iae(runs: List[Dict]) -> Dict:
    rows = []
    for run in runs:
        seed = run["config"]["seed"]
        for row in run["slice_metrics"]:
            if row["target_topology"] != "balanced":
                continue
            rows.append({**row, "seed": seed})

    if not rows:
        raise ValueError("no balanced-target slice rows in multiseed runs")

    mean_bal = float(np.mean([r["iae_balanced"] for r in rows]))
    mean_chain = float(np.mean([r["iae_chain"] for r in rows]))
    improvement = (mean_chain - mean_bal) / max(mean_chain, 1e-12)
    per_seed = []
    for seed in sorted({r["seed"] for r in rows}):
        sub = [r for r in rows if r["seed"] == seed]
        sb = float(np.mean([r["iae_balanced"] for r in sub]))
        sc = float(np.mean([r["iae_chain"] for r in sub]))
        per_seed.append(
            {
                "seed": seed,
                "mean_iae_balanced": sb,
                "mean_iae_chain": sc,
                "improvement_frac": (sc - sb) / max(sc, 1e-12),
            }
        )

    return {
        "mean_iae_balanced": mean_bal,
        "mean_iae_chain": mean_chain,
        "improvement_frac": improvement,
        "passes_20pct": improvement >= 0.20,
        "per_seed": per_seed,
        "n_slice_rows": len(rows),
    }


def _write_report(path: Path, seeds: List[int], runs: List[Dict], agg: Dict):
    lines = [
        "# Fit Ceiling 多 Seed 对比报告",
        "",
        "## 目的",
        "",
        "在 complex balanced 目标上，用多个随机种子验证 **balanced TTNS** 相对 **chain TTNS** 的切片 IAE 优势是否稳健。",
        "",
        f"- Seeds: `{seeds}`",
        f"- 单 seed 配置见各 run 的 `config` 字段",
        "",
        "## 聚合结果（balanced 目标，3 切片均值）",
        "",
        "| 指标 | 值 |",
        "|------|-----:|",
        f"| mean IAE (balanced TTNS) | {agg['mean_iae_balanced']:.6f} |",
        f"| mean IAE (chain TTNS) | {agg['mean_iae_chain']:.6f} |",
        f"| 相对提升 (chain→balanced) | {agg['improvement_frac'] * 100:.2f}% |",
        f"| 通过 ≥20% 标准 | {'是' if agg['passes_20pct'] else '否'} |",
        "",
        "## 分 seed 明细",
        "",
        "| seed | mean IAE balanced | mean IAE chain | 提升 |",
        "|-----:|---:|---:|---:|",
    ]
    for row in agg["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['mean_iae_balanced']:.6f} | "
            f"{row['mean_iae_chain']:.6f} | {row['improvement_frac'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
        ]
    )
    if agg["passes_20pct"]:
        lines.append(
            f"- {len(seeds)} 个 seed 聚合后，匹配拓扑 TTNS 的 mean IAE 比 chain 低 **{agg['improvement_frac'] * 100:.1f}%**，达到 Phase 2 M1.1 标准。"
        )
    else:
        lines.append(
            f"- 聚合提升为 **{agg['improvement_frac'] * 100:.1f}%**，未达到 20% 阈值；需检查 seed 或超参。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(seeds: List[int] | None = None):
    seeds = list(seeds or DEFAULT_SEEDS)
    print("multiseed fit_ceiling seeds:", seeds, flush=True)

    runs = []
    for i, seed in enumerate(seeds):
        print(f"\n--- run {i + 1}/{len(seeds)} seed={seed} ---", flush=True)
        write_last = i == len(seeds) - 1
        run = run_fit_ceiling(seed=seed, write_outputs=write_last)
        runs.append(run)

    agg = _aggregate_balanced_iae(runs)
    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = report_dir / "fit_ceiling_two_models_complex_multiseed_report_zh.md"
    out_json = report_dir / "fit_ceiling_two_models_complex_multiseed_metrics.json"

    _write_report(out_md, seeds, runs, agg)
    out_json.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "runs": runs,
                "aggregate_balanced": agg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== multiseed fit_ceiling done ===", flush=True)
    print("aggregate improvement:", f"{agg['improvement_frac'] * 100:.2f}%", flush=True)
    print("passes 20%:", agg["passes_20pct"], flush=True)
    print("saved report:", out_md, flush=True)
    print("saved metrics:", out_json, flush=True)


if __name__ == "__main__":
    main()
