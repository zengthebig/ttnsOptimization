"""R6 小块 corr 可达上限参照:在同一棵树结构下,把目标从"树投影(R5)"升级为
"完整联合(R6 网格)"能把 corr_fro 压到多低,以及加大 rank 是否进一步帮助。

复用 joint_vs_tree_block.run(不改原文件)。R6 的交叉项是 G^K 网格,仅 K≤3 可行,
故固定 clustered [3,3](块大小=3)。对 rank∈{8,16} 各跑一次,佐证:
"corr_fro 残差主要来自目标口径而非模型能力"——rank 抬升在树投影下补不动非树边,
但换完整联合目标即可大幅压低。

运行: env -u PYTHONPATH python3 -m simple_ttns_l2.experiments.budget_r6_block_ref
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.experiments.joint_vs_tree_block import run  # noqa: E402

REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def base_cfg(rank: int) -> dict:
    return dict(
        n_layers=3, clusters=[3, 3], fanin=2, test_layer=1,
        delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.3, node_lo=0.0, node_hi=0.3),
        n_total=20000, n_sample=16000, n_fit=20000, q=2, m=24, rank=rank,
        src_sigma=0.03,
        lr=2e-3, steps=800, batch_sz=512, init_noise=1e-2, train_noise=1e-3,
        log_every=400, early_stop_patience=8, mi_threshold=0.02, seed=0,
        n_s=100, n_s_pair=80, n_s_joint=60, an_lr=3e-3, an_steps=800,
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / "budget_r6_block_ref_metrics.json"
    out = {}
    if p.exists():
        try:
            out = json.loads(p.read_text())
            print(f"[resume] 已完成: {list(out)}", flush=True)
        except Exception:
            out = {}
    for rank in (8, 16):
        key = f"rank{rank}"
        if key in out:
            print(f"[skip] {key} 已完成", flush=True)
            continue
        print(f"\n########## rank={rank} ##########", flush=True)
        rows = run(base_cfg(rank))
        out[key] = rows
        for r in rows:
            print(f"  block{r['bi']} gids={r['gids']} K={r['K']}  "
                  f"LL tree/joint/samp={r['ll_tree']:.3f}/{r['ll_joint']:.3f}/{r['ll_samp']:.3f}  "
                  f"fro tree/joint/samp={r['fro_tree']:.3f}/{r['fro_joint']:.3f}/{r['fro_samp']:.3f}",
                  flush=True)
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f"[ckpt] 写出 {key} → {p}", flush=True)
    print(f"\n[写出] {p}", flush=True)


if __name__ == "__main__":
    main()
