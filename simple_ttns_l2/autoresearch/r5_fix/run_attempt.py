"""R5 autoresearch 统一 runner：拟合 → 指标 → marginal + refined 双图 → 更新 PROGRESS。

每次 attempt 必须同时产出：
  artifacts/<id>/marginal_slices.png
  artifacts/<id>/slice_refined.png

用法：
  python -m simple_ttns_l2.autoresearch.r5_fix.run_attempt \\
    --attempt-id 000 --name "baseline R5" --variant baseline

  python -m simple_ttns_l2.autoresearch.r5_fix.run_attempt \\
    --attempt-id 001 --config attempts/001.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.analytic_tree_fit import fit_analytic_chain  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_three_way import CFG  # noqa: E402
from simple_ttns_l2.experiments.dense_dag_r567_remedy_tests import (  # noqa: E402
    prepare_data,
    eval_chain,
    fit_analytic_chain_nonneg,
    fit_analytic_chain_nonneg_joint_blocks,
)
from simple_ttns_l2.experiments.plot_dense_dag_nonneg_slices import (  # noqa: E402
    plot_marginal_slices,
    plot_refined_slices,
    N_PLOT,
)

AR_ROOT = Path(__file__).resolve().parent
ARTIFACTS = AR_ROOT / "artifacts"
PROGRESS_MD = AR_ROOT / "PROGRESS.md"

FitFn = Callable[..., Dict[int, list]]


@dataclass(frozen=True)
class FitSpec:
    fit_fn: FitFn
    color: str
    description: str
    needs_s_max0: bool = True


def _fit_baseline(forest0, spec, params, key, s_max0, cfg, **kw):
    return fit_analytic_chain(
        forest0, spec, params, key, s_max0,
        q=cfg["q"], m=cfg["m"], rank=cfg["rank"],
        n_s=cfg["n_s"], n_s_pair=cfg["n_s_pair"],
        lr=cfg.get("an_lr", cfg["lr"]), steps=cfg.get("an_steps", cfg["steps"]),
        init_noise=cfg["init_noise"], log_every=cfg.get("log_every", 0),
        use_mi=True, block_mode=cfg["block_mode"],
        marginal_l2_weight=kw.get("marginal_l2_weight", 0.0),
        normalize_every=1,
    )


def _fit_nonneg(forest0, spec, params, key, s_max0, cfg, **kw):
    return fit_analytic_chain_nonneg(forest0, spec, params, key, s_max0, cfg)


def _fit_nonneg_corr(forest0, spec, params, key, s_max0, cfg, **kw):
    return fit_analytic_chain_nonneg(
        forest0, spec, params, key, s_max0, cfg,
        corr_weight=float(kw.get("corr_weight", 5.0)),
    )


def _fit_joint_block(forest0, spec, params, key, s_max0, cfg, **kw):
    return fit_analytic_chain_nonneg_joint_blocks(
        forest0, spec, params, key, s_max0, cfg,
        joint_kmax=int(kw.get("joint_kmax", cfg.get("joint_kmax", 4))),
        joint_layers=kw.get("joint_layers"),
    )


def _fit_marginal_l2(forest0, spec, params, key, s_max0, cfg, **kw):
    w = float(kw.get("marginal_l2_weight", 0.3))
    return _fit_baseline(forest0, spec, params, key, s_max0, cfg, marginal_l2_weight=w)


def _fit_block_source(forest0, spec, params, key, s_max0, cfg, **kw):
    cfg2 = dict(cfg)
    cfg2["block_mode"] = "source"
    return _fit_baseline(forest0, spec, params, key, s_max0, cfg2, **kw)


def _fit_finer_grid(forest0, spec, params, key, s_max0, cfg, **kw):
    cfg2 = dict(cfg)
    cfg2["n_s"] = int(kw.get("n_s", 200))
    cfg2["n_s_pair"] = int(kw.get("n_s_pair", 160))
    cfg2["an_steps"] = int(kw.get("an_steps", 8000))
    return _fit_baseline(forest0, spec, params, key, s_max0, cfg2, **kw)


def _fit_rank16(forest0, spec, params, key, s_max0, cfg, **kw):
    cfg2 = dict(cfg)
    cfg2["rank"] = int(kw.get("rank", 16))
    return _fit_baseline(forest0, spec, params, key, s_max0, cfg2, **kw)


VARIANTS: Dict[str, FitSpec] = {
    "baseline": FitSpec(_fit_baseline, "#d62728", "R5 全解析链 λ=0（复现基线）"),
    "nonneg": FitSpec(_fit_nonneg, "#9467bd", "R5 非负 core + 解析 L2"),
    "nonneg_corr": FitSpec(_fit_nonneg_corr, "#2ca02c", "R5 非负 core + 解析 L2 + 相关矩惩罚"),
    "joint_block": FitSpec(_fit_joint_block, "#1f77b4", "K≤4 小块完整联合解析目标 + 非负 core"),
    "marginal_l2": FitSpec(_fit_marginal_l2, "#ff7f0e", "R5 + marginal_l2_weight"),
    "block_source": FitSpec(_fit_block_source, "#8c564b", "block_mode=source"),
    "finer_grid": FitSpec(_fit_finer_grid, "#17becf", "增大解析网格/步数"),
    "rank16": FitSpec(_fit_rank16, "#bcbd22", "rank=16"),
}


def _corr_fro_norm(samples: np.ndarray, tev: np.ndarray) -> float:
    Ct = np.corrcoef(tev.T)
    K = tev.shape[1]
    denom = np.sqrt(max(K * max(K - 1, 1), 1))
    return float(np.linalg.norm(Ct - np.corrcoef(samples.T))) / denom


def _assert_outputs(out_dir: Path) -> None:
    marginal = out_dir / "marginal_slices.png"
    refined = out_dir / "slice_refined.png"
    missing = [p.name for p in (marginal, refined) if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"attempt 无效：缺少必产物图像 {missing}。"
            " autoresearch 要求每次同时输出 marginal_slices 与 slice_refined。"
        )


def _update_progress(
    attempt_id: str,
    name: str,
    variant: str,
    l4: dict,
    commit_hint: str,
    conclusion: str = "待填",
) -> None:
    row = (
        f"| {attempt_id} | {datetime.now(timezone.utc).strftime('%Y-%m-%d')} "
        f"| {name} | {variant} "
        f"| {l4['joint_ll']:.2f} | {l4['nonpos_rate']:.2f} "
        f"| {l4['std_ratio_mean']:.2f} | {commit_hint} | {conclusion} |"
    )
    text = PROGRESS_MD.read_text(encoding="utf-8")
    marker = "<!-- 新 attempt 行插入此处上方 -->"
    if marker not in text:
        raise RuntimeError(f"PROGRESS.md 缺少插入标记 {marker!r}")
    text = text.replace(marker, f"{row}\n\n{marker}")
    PROGRESS_MD.write_text(text, encoding="utf-8")
    print(f"[progress] updated {PROGRESS_MD}", flush=True)


def run_attempt(
    attempt_id: str,
    name: str,
    variant: str,
    seed: int = 0,
    cfg_overrides: dict | None = None,
    fit_kwargs: dict | None = None,
    conclusion: str = "待填",
    dry_run: bool = False,
) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"未知 variant {variant!r}；可选: {sorted(VARIANTS)}")

    spec_entry = VARIANTS[variant]
    out_dir = ARTIFACTS / attempt_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(CFG)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    manifest = {
        "attempt_id": attempt_id,
        "name": name,
        "variant": variant,
        "description": spec_entry.description,
        "seed": seed,
        "cfg_overrides": cfg_overrides or {},
        "fit_kwargs": fit_kwargs or {},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if dry_run:
        print(f"[dry-run] would write to {out_dir}", flush=True)
        return out_dir

    t0 = time.perf_counter()
    key, spec, params, layers, train_x, test_x, forest0, s_max0 = prepare_data(seed, cfg)

    t_fit = time.perf_counter()
    chain = spec_entry.fit_fn(
        forest0, spec, params, key, s_max0, cfg, **(fit_kwargs or {})
    )
    fit_seconds = time.perf_counter() - t_fit

    label = f"R5-fix/{attempt_id}:{name}"
    eval_key = jax.random.PRNGKey(seed + 999)
    result = eval_chain(chain, test_x, layers, eval_key, label)
    samples = result.pop("samples")

    # ---- 必产物 1: marginal ----
    plot_marginal_slices(
        test_x, layers, samples,
        model_label=label,
        model_color=spec_entry.color,
        title=f"dense_dag_r567 [{attempt_id}] {name} — per-node marginal slices",
        out_path=out_dir / "marginal_slices.png",
    )

    # ---- 必产物 2: refined slice ----
    plot_refined_slices(
        test_x, layers, chain, samples,
        model_label=label,
        model_color=spec_entry.color,
        seed=seed,
        out_path=out_dir / "slice_refined.png",
    )

    _assert_outputs(out_dir)

    corr_by_layer = {}
    n_plot = min(N_PLOT, test_x.shape[0])
    for li, nodes in enumerate(layers):
        if li == 0:
            continue
        tev = test_x[:n_plot, nodes]
        corr_by_layer[str(li)] = _corr_fro_norm(samples[li], tev)

    metrics = {
        "attempt_id": attempt_id,
        "name": name,
        "variant": variant,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - t0,
        "per_layer": result["per_layer"],
        "corr_fro_norm_by_layer": corr_by_layer,
        "artifacts": {
            "marginal_slices": str(out_dir / "marginal_slices.png"),
            "slice_refined": str(out_dir / "slice_refined.png"),
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    l4 = result["per_layer"][-1]
    commit_hint = f"`autoresearch(r5): {attempt_id} {name}`"
    _update_progress(attempt_id, name, variant, l4, commit_hint, conclusion)

    print("\n" + "=" * 60, flush=True)
    print(f"attempt {attempt_id} 完成", flush=True)
    print(f"  L4 joint_ll={l4['joint_ll']:.3f}  nonpos={l4['nonpos_rate']:.3f}  "
          f"std_ratio={l4['std_ratio_mean']:.3f}", flush=True)
    print(f"  marginal: {out_dir / 'marginal_slices.png'}", flush=True)
    print(f"  refined:  {out_dir / 'slice_refined.png'}", flush=True)
    print(f"\n下一步: ./simple_ttns_l2/autoresearch/r5_fix/finish_attempt.sh {attempt_id}", flush=True)
    print("=" * 60, flush=True)
    return out_dir


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="R5 autoresearch attempt runner")
    ap.add_argument("--attempt-id", required=True, help="如 000, 001")
    ap.add_argument("--name", default="", help="attempt 简短描述")
    ap.add_argument("--variant", default="baseline", choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=str, default="", help="JSON 配置文件（覆盖 CLI）")
    ap.add_argument("--conclusion", default="待填")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    attempt_id = args.attempt_id.strip()
    if not re.fullmatch(r"\d{3}", attempt_id):
        print(f"警告: attempt-id 建议三位数字，当前为 {attempt_id!r}", flush=True)

    name = args.name
    variant = args.variant
    seed = args.seed
    cfg_overrides = {}
    fit_kwargs = {}
    conclusion = args.conclusion

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = AR_ROOT / "attempts" / cfg_path.name if not cfg_path.exists() else cfg_path
        data = _load_config(cfg_path)
        attempt_id = data.get("attempt_id", attempt_id)
        name = data.get("name", name)
        variant = data.get("variant", variant)
        seed = int(data.get("seed", seed))
        cfg_overrides = data.get("cfg_overrides", {})
        fit_kwargs = data.get("fit_kwargs", {})
        conclusion = data.get("conclusion", conclusion)

    if not name:
        ap.error("--name 或 config.name 必填")

    run_attempt(
        attempt_id=attempt_id,
        name=name,
        variant=variant,
        seed=seed,
        cfg_overrides=cfg_overrides,
        fit_kwargs=fit_kwargs,
        conclusion=conclusion,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
