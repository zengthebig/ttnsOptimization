"""Create publication figures from the finalized theorem-validation JSON only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    REPO_ROOT
    / "simple_ttns_l2"
    / "reports"
    / "maxplus_cdf_theorem_validation_metrics.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "simple_ttns_l2" / "reports"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_two_series(
    ax,
    x,
    marginal,
    pair,
    xlabel: str,
    title: str,
) -> None:
    ax.loglog(x, marginal, marker="o", linewidth=1.7, label="Marginal CDF")
    ax.loglog(x, pair, marker="s", linewidth=1.7, label="Pair CDF")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maximum absolute error")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False)


def convergence_figure(metrics: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)

    uniform_q = metrics["runs"]
    _plot_two_series(
        axes[0, 0],
        [row["q_grid"] for row in uniform_q],
        [row["marginal_vs_q2001_max_abs"] for row in uniform_q],
        [row["pair_vs_q2001_max_abs"] for row in uniform_q],
        r"Local grid size $q_{\mathrm{grid}}$",
        "Uniform delay: local quadrature",
    )

    uniform_n = metrics["node_delay_convergence"]["runs"]
    _plot_two_series(
        axes[0, 1],
        [row["n_d"] for row in uniform_n],
        [row["marginal_vs_n_d2048_max_abs"] for row in uniform_n],
        [row["pair_vs_n_d2048_max_abs"] for row in uniform_n],
        r"Node quadrature size $n_d$",
        "Uniform delay: node quadrature",
    )

    logskew_q = metrics["logskew_delay"]["q_grid_runs"]
    _plot_two_series(
        axes[1, 0],
        [row["q_grid"] for row in logskew_q],
        [row["marginal_max_abs"] for row in logskew_q],
        [row["pair_max_abs"] for row in logskew_q],
        r"Local grid size $q_{\mathrm{grid}}$",
        "Log-skew-normal delay: local quadrature",
    )

    logskew_n = metrics["logskew_delay"]["n_d_runs"]
    _plot_two_series(
        axes[1, 1],
        [row["n_d"] for row in logskew_n],
        [row["marginal_max_abs"] for row in logskew_n],
        [row["pair_max_abs"] for row in logskew_n],
        r"Node quadrature size $n_d$",
        "Log-skew-normal delay: node quadrature",
    )

    fig.suptitle(
        "Empirical discretization error relative to high-resolution numerical references",
        fontsize=11,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, width, height, text, facecolor, edgecolor="#333333") -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def _arrow(ax, start, end, color="#4c4c4c", linewidth=1.25) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def shared_parent_figure(metrics: dict, output: Path) -> None:
    config = metrics["configuration"]
    shared = config["shared_parents"]
    if shared != [1]:
        raise ValueError(f"unexpected shared-parent configuration: {shared}")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    left, right = axes

    left.set_title("Shared-parent max-plus mapping")
    x_positions = [0.18, 0.50, 0.82]
    for idx, x in enumerate(x_positions):
        color = "#f4c26b" if idx == 1 else "#cfe2f3"
        _box(left, (x, 0.78), 0.18, 0.13, rf"$X_{idx + 1}$", color)
    _box(left, (0.34, 0.25), 0.20, 0.13, r"$Y_1$", "#d9ead3")
    _box(left, (0.68, 0.25), 0.20, 0.13, r"$Y_2$", "#d9ead3")
    _arrow(left, (0.18, 0.70), (0.31, 0.33))
    _arrow(left, (0.50, 0.70), (0.37, 0.33), color="#b45f06", linewidth=1.8)
    _arrow(left, (0.50, 0.70), (0.65, 0.33), color="#b45f06", linewidth=1.8)
    _arrow(left, (0.82, 0.70), (0.71, 0.33))
    left.text(
        0.50,
        0.55,
        r"shared parent $X_2$",
        ha="center",
        color="#7f3f00",
        fontsize=9,
    )
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.axis("off")

    right.set_title("Local projections followed by one TTNS contraction")
    _box(right, (0.18, 0.74), 0.29, 0.13, r"$M_1(s)$", "#cfe2f3")
    _box(right, (0.18, 0.50), 0.29, 0.13, r"$M_2(s,t)$", "#f4c26b")
    _box(right, (0.18, 0.26), 0.29, 0.13, r"$M_3(t)$", "#cfe2f3")
    _box(right, (0.55, 0.50), 0.25, 0.18, r"$\mathcal{C}_T$", "#ead1dc")
    _box(right, (0.84, 0.50), 0.24, 0.18, r"$F_{Y_1,Y_2}(s,t)$", "#d9ead3")
    for y in (0.74, 0.50, 0.26):
        _arrow(right, (0.33, y), (0.43, 0.50))
    _arrow(right, (0.68, 0.50), (0.72, 0.50))
    right.text(
        0.50,
        0.08,
        r"$M_{2,i}(s,t)=\int b_{2,i}(x)"
        r"F_{21}(s-x)F_{22}(t-x)\,dx$",
        ha="center",
        va="center",
        fontsize=9,
    )
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.axis("off")

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def heatmap_figure(metrics: dict, output: Path) -> None:
    plot_data = metrics["plot_data"]
    grid = np.asarray(plot_data["s_grid"], dtype=float)
    reference = np.asarray(plot_data["pair_reference_q2001"], dtype=float)
    estimate = np.asarray(plot_data["pair_q81"], dtype=float)
    error = np.abs(estimate - reference)
    log_error = np.log10(np.maximum(error, 1e-16))
    extent = [grid[0], grid[-1], grid[0], grid[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.25), constrained_layout=True)
    common = {"origin": "lower", "extent": extent, "aspect": "auto", "vmin": 0, "vmax": 1}
    image0 = axes[0].imshow(reference, cmap="viridis", **common)
    axes[0].set_title(r"Reference, $q_{\mathrm{grid}}=2001$")
    axes[1].imshow(estimate, cmap="viridis", **common)
    axes[1].set_title(r"TTNS contraction, $q_{\mathrm{grid}}=81$")
    image2 = axes[2].imshow(
        log_error,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
        vmin=-16,
        vmax=np.ceil(np.log10(max(float(error.max()), 1e-16))),
    )
    axes[2].set_title(r"$\log_{10}$ absolute error")
    for ax in axes:
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$s$")
    fig.colorbar(image0, ax=axes[:2], shrink=0.82, label=r"$F_{Y_1,Y_2}(s,t)$")
    fig.colorbar(image2, ax=axes[2], shrink=0.82, label=r"$\log_{10}|\Delta F|$")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics.get("status") != "pass" or not all(metrics.get("checks", {}).values()):
        raise RuntimeError("refusing to plot: validation JSON is not fully passing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    convergence_figure(
        metrics,
        args.output_dir / "maxplus_cdf_theorem_validation_convergence.png",
    )
    shared_parent_figure(
        metrics,
        args.output_dir / "maxplus_cdf_shared_parent_contraction.png",
    )
    heatmap_figure(
        metrics,
        args.output_dir / "maxplus_cdf_pair_validation_heatmaps.png",
    )


if __name__ == "__main__":
    main()
