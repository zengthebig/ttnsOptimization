from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_ttns_l2.experiments.fit_dag_junction_vs_chain import (
    _key_pairs,
    _write_report,
    run_dag_experiment,
)

SEEDS = [313, 2602, 20260227]


def run():
    per_seed = []
    for seed in SEEDS:
        print("=== DAG multiseed seed", seed, "===", flush=True)
        result = run_dag_experiment(seed=seed, write_outputs=False)
        per_seed.append(
            {
                "seed": seed,
                "mean_iae_junction": result["key_slice_mean_iae_junction"],
                "mean_iae_balanced": result["key_slice_mean_iae_balanced"],
                "mean_iae_chain": result["key_slice_mean_iae_chain"],
                "improvement_junction_vs_chain": result["key_slice_improvement_junction_vs_chain"],
            }
        )

    mean_imp = sum(r["improvement_junction_vs_chain"] for r in per_seed) / len(per_seed)
    aggregate = {"per_seed": per_seed, "mean_improvement_junction_vs_chain": mean_imp}

    # 用最后一个 seed 的详细结果写报告（附多 seed 表）
    last = run_dag_experiment(seed=SEEDS[-1], write_outputs=False)
    report_dir = REPO_ROOT / "simple_ttns_l2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = report_dir / "dag_junction_vs_chain_multiseed_report_zh.md"
    out_json = report_dir / "dag_junction_vs_chain_multiseed_metrics.json"

    from simple_ttns_l2.experiments.dag_target import fork_dag_edges
    from simple_ttns_l2.junction_tree import fork_junction_parent
    from simple_ttns_l2.experiments.topology_comparison import ExperimentConfig

    cfg = ExperimentConfig(**last["config"])
    _write_report(
        out_md,
        cfg,
        fork_dag_edges(cfg.n_dims),
        fork_junction_parent(cfg.n_dims),
        last["training"],
        last["slice_metrics"],
        _key_pairs(cfg.n_dims),
        aggregate=aggregate,
    )
    out_json.write_text(
        json.dumps({"aggregate": aggregate, "last_seed_detail": last}, indent=2),
        encoding="utf-8",
    )
    print("mean improvement junction vs chain across seeds:", f"{mean_imp * 100:.2f}%", flush=True)
    print("saved:", out_md, flush=True)


if __name__ == "__main__":
    run()
