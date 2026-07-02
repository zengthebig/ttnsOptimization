"""单步诊断：解析传播(Scheme B)丢了什么。

L0 森林(数据拟合) → L1:
- 真值 L1 = 祖先采样(L0 森林采样 → 真 max-plus), 携带全阶真依赖;
- B-copula L1 = Scheme B 解析(精确边缘 + 精确两两相关) + 高斯 copula 采样。
比: ① 边缘(mean/std) ② 相关矩阵 ③ 最相关一对的二维联合形状 + 超越相关的高阶量。
预期: ①② 几乎一致(B 精确到二阶), ③ 形状不同(max-plus 高阶/非高斯被高斯 copula 抹平)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
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
from simple_ttns_l2.maxplus_cdf_forest import UpperForest, sample_layer_copula, propagate_layer_cdf_forest  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402

cfg = dict(
    layer_sizes=[6, 6, 6, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
    n_total=40000, q=2, m=24, rank=8, source_mode="bimodal", src_sigma=0.06,
    lr=0.002, steps=1000, batch_sz=512, init_noise=0.01, train_noise=0.001,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
)
N = 8000
S_MAX = 3.0

key = jax.random.PRNGKey(cfg["seed"])
spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
params = DelayParams(**cfg["delay"])
sources, kernels = ground_truth_samplers(spec, params)
sources = {**sources, **bimodal_sources(spec, cfg)}
k_d, key = jax.random.split(key)
xs = np.asarray(sample_joint(spec, sources, kernels, k_d, cfg["n_total"], clip=(-1e9, 1e9)))
train_x = xs[: int(0.7 * xs.shape[0])]
L0 = list(spec.layers[0])

k_f, key = jax.random.split(key)
forest0 = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_f, label="L0", mi_threshold=cfg["mi_threshold"])

# 真值 L1: 祖先采样
k_s, key = jax.random.split(key)
s0 = np.asarray(sample_forest(forest0, k_s, N, grid_size=400))
rng = np.random.default_rng(1)
true_l1 = propagate_layer(spec, 1, s0, params, rng)

# B-copula L1
upper = UpperForest(forest0, q_grid=400)
k_b, key = jax.random.split(key)
b_l1 = sample_layer_copula(upper, spec, 1, params, k_b, N, S_MAX, n_s=200, n_s_pair=100)
Binfo = propagate_layer_cdf_forest(upper, spec, 1, params, S_MAX, n_s=120, n_s_pair=60)

# ① 边缘
print("\n=== ① 边缘 mean/std (真值 vs B) ===")
for j in range(true_l1.shape[1]):
    print(f"  y{j}: mean {true_l1[:,j].mean():.3f}/{b_l1[:,j].mean():.3f}  std {true_l1[:,j].std():.3f}/{b_l1[:,j].std():.3f}")

# ② 相关矩阵
Ct = np.corrcoef(true_l1.T)
Cb = np.corrcoef(b_l1.T)
print(f"\n=== ② 相关矩阵 Frobenius 差 ||Ct-Cb|| = {np.linalg.norm(Ct-Cb):.4f} ===")

# ③ 最相关一对
iu = np.triu_indices(true_l1.shape[1], 1)
pair = (iu[0][np.argmax(np.abs(Ct[iu]))], iu[1][np.argmax(np.abs(Ct[iu]))])
a, b = pair
print(f"\n=== ③ 最相关对 (y{a},y{b}) corr 真={Ct[a,b]:.3f} B={Cb[a,b]:.3f} ===")
# 超越相关的高阶量：|dev| 的相关(尾部共动) 与 三阶混合矩
def higher(x, i, j):
    xi = (x[:, i]-x[:, i].mean()); xj = (x[:, j]-x[:, j].mean())
    return dict(
        corr_abs=np.corrcoef(np.abs(xi), np.abs(xj))[0, 1],
        coskew=np.mean(xi**2 * xj) / (x[:, i].std()**2 * x[:, j].std() + 1e-12),
    )
ht, hb = higher(true_l1, a, b), higher(b_l1, a, b)
print(f"  corr(|dev|): 真={ht['corr_abs']:.3f}  B={hb['corr_abs']:.3f}")
print(f"  coskew    : 真={ht['coskew']:.3f}  B={hb['coskew']:.3f}")

# 画图
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
rng2 = [[min(true_l1[:, a].min(), b_l1[:, a].min()), max(true_l1[:, a].max(), b_l1[:, a].max())],
        [min(true_l1[:, b].min(), b_l1[:, b].min()), max(true_l1[:, b].max(), b_l1[:, b].max())]]
ax[0].hist2d(true_l1[:, a], true_l1[:, b], bins=80, range=rng2, cmap="viridis")
ax[0].set_title(f"(a) TRUE ancestral  corr={Ct[a,b]:.2f}")
ax[1].hist2d(b_l1[:, a], b_l1[:, b], bins=80, range=rng2, cmap="viridis")
ax[1].set_title(f"(b) Scheme B + Gaussian copula  corr={Cb[a,b]:.2f}")
for k in (0, 1):
    ax[k].set_xlabel(f"y{a}"); ax[k].set_ylabel(f"y{b}")
im = ax[2].imshow(np.abs(Ct - Cb), cmap="Reds", vmin=0)
ax[2].set_title("(c) |corr_true - corr_B|")
fig.colorbar(im, ax=ax[2], fraction=0.046)
fig.suptitle("Scheme B single-step (L0->L1): marginals+corr match, joint SHAPE differs", fontweight="bold")
fig.tight_layout()
out = REPO_ROOT / "simple_ttns_l2" / "reports" / "diag_scheme_b.png"
fig.savefig(out, bbox_inches="tight", dpi=130)
print("\nsaved:", out)
