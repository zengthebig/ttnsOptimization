"""从已保存的 UCI mixture 参数快照重画 2D 切片图（不重训）。

用途：`uci_ttde_vs_ttns.py --n-comps>1` 会保存
`uci_<dataset>_ttde_vs_ttns_params_<tag>.pkl`。本脚本加载快照，重新计算 TTDE/TTNSDE
mixture 的 2D 边缘，并画更适合展示的切片图：

- 用每个维度自己的 quantile 窗口，避免极端点拉大网格；
- 对有限窗口内的 GT/模型密度做 display-normalization，只比较形状；
- 标题保留 raw grid integral，暴露数值积分稳定性，不把显示归一化当作模型归一化。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simple_ttns_l2.experiments.uci_ttde_vs_ttns import (  # noqa: E402
    _eval_pair_marginal_sq_tt_mixture_on_grid,
    _eval_pair_marginal_sq_ttns_mixture_on_grid,
    _pick_slice_pairs,
    load_uci,
)


REPORTS = REPO_ROOT / "simple_ttns_l2" / "reports"


def _display_normalize(Z: np.ndarray, xi: np.ndarray, xj: np.ndarray) -> tuple[np.ndarray, float]:
    raw = float(np.trapz(np.trapz(Z, xj, axis=1), xi, axis=0))
    if np.isfinite(raw) and raw > 1e-300:
        return Z / raw, raw
    return Z, raw


def _hist_on_centers(test_x, dim_i, dim_j, xi, xj):
    ex = np.empty(len(xi) + 1)
    ey = np.empty(len(xj) + 1)
    ex[1:-1] = 0.5 * (xi[:-1] + xi[1:])
    ey[1:-1] = 0.5 * (xj[:-1] + xj[1:])
    ex[0] = xi[0] - 0.5 * (xi[1] - xi[0])
    ex[-1] = xi[-1] + 0.5 * (xi[-1] - xi[-2])
    ey[0] = xj[0] - 0.5 * (xj[1] - xj[0])
    ey[-1] = xj[-1] + 0.5 * (xj[-1] - xj[-2])
    H, _, _ = np.histogram2d(test_x[:, dim_i], test_x[:, dim_j], bins=[ex, ey], density=True)
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    return H, ex, ey


def _quantile_grid(train_x, test_x, dim, n, qlo, qhi):
    xs = np.concatenate([train_x[:, dim], test_x[:, dim]])
    lo, hi = np.quantile(xs, [qlo, qhi])
    pad = 0.03 * max(float(hi - lo), 1e-9)
    return np.linspace(float(lo - pad), float(hi + pad), n)


def redraw(dataset: str, params_path: Path, out_path: Path, data_dir: Path,
           grid_n: int, qlo: float, qhi: float, n_pairs: int,
           tt_chunk: int, ttns_chunk: int):
    with params_path.open("rb") as f:
        snap = pickle.load(f)

    trn, _, tst = load_uci(dataset, data_dir)
    cfg = snap.get("config", {})
    train_cap = int(cfg.get("train_cap", 40000))
    if trn.shape[0] > train_cap:
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        trn = trn[rng.choice(trn.shape[0], train_cap, replace=False)]

    parent = [int(x) for x in snap["ttnsde_tree_parent"]]
    pairs = _pick_slice_pairs(parent, trn, n_pairs=n_pairs)
    print(f"{dataset}: pairs={pairs}", flush=True)

    tt_model = type("TTMixtureSnapshot", (), {
        "bases": snap["ttde_bases"],
        "permutations": snap["ttde_permutations"],
    })()
    sq_model = type("TTNSMixtureSnapshot", (), {"bases": snap["ttnsde_bases"]})()

    models = ["GT hist", "global_TTDE", "global_TTNSDE"]
    fig, axes = plt.subplots(len(pairs), len(models), figsize=(4.1 * len(models), 3.5 * len(pairs)),
                             squeeze=False)
    for r, (di, dj) in enumerate(pairs):
        xi = _quantile_grid(trn, tst, di, grid_n, qlo, qhi)
        xj = _quantile_grid(trn, tst, dj, grid_n, qlo, qhi)
        H, ex, ey = _hist_on_centers(tst, di, dj, xi, xj)
        Hn, Hint = _display_normalize(H, xi, xj)

        d_tt = _eval_pair_marginal_sq_tt_mixture_on_grid(
            snap["ttde_params"], tt_model, di, dj, xi, xj, chunk_size=tt_chunk)
        d_sq = _eval_pair_marginal_sq_ttns_mixture_on_grid(
            snap["ttnsde_params"], sq_model, parent, di, dj, xi, xj, chunk_size=ttns_chunk)
        d_ttn, Itt = _display_normalize(d_tt, xi, xj)
        d_sqn, Isq = _display_normalize(d_sq, xi, xj)

        vmax = max(float(np.nanmax(Hn)), float(np.nanmax(d_ttn)), float(np.nanmax(d_sqn)), 1e-12)
        ax = axes[r][0]
        ax.pcolormesh(ex, ey, Hn.T, cmap="Blues", shading="auto", vmin=0, vmax=vmax)
        ax.set_title(f"GT hist\nwin∫={Hint:.2f}")
        ax.set_xlabel(f"x[{di}]"); ax.set_ylabel(f"x[{dj}]")
        for c, (name, Z, raw) in enumerate((("global_TTDE", d_ttn, Itt), ("global_TTNSDE", d_sqn, Isq)), start=1):
            ax = axes[r][c]
            ax.pcolormesh(xi, xj, Z.T, cmap="Blues", shading="auto", vmin=0, vmax=vmax)
            ax.set_title(f"{name}\nraw win∫={raw:.2f}")
            ax.set_xlabel(f"x[{di}]"); ax.set_ylabel(f"x[{dj}]")
        print(f"  pair({di},{dj}) raw integrals: TTDE={Itt:.3f} TTNSDE={Isq:.3f}", flush=True)

    fig.suptitle(f"{dataset}: display-normalized 2D slices (same Blues scale per row)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["power", "gas"], required=True)
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/data")
    p.add_argument("--grid-n", type=int, default=120)
    p.add_argument("--qlo", type=float, default=0.005)
    p.add_argument("--qhi", type=float, default=0.995)
    p.add_argument("--n-pairs", type=int, default=3)
    p.add_argument("--tt-chunk", type=int, default=512)
    p.add_argument("--ttns-chunk", type=int, default=64)
    args = p.parse_args()
    redraw(args.dataset, args.params, args.out, args.data_dir, args.grid_n, args.qlo, args.qhi,
           args.n_pairs, args.tt_chunk, args.ttns_chunk)


if __name__ == "__main__":
    main()
