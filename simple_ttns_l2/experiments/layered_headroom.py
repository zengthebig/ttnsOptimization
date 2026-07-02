"""量化 layered_TTNS 的 joint_LL 构成与改进上限。

joint_LL = [源层 6 维学习密度] + [深层 18 维 oracle max-plus 核(精确, 不可学)]
EM/更好初始化只能改善"源层学习密度"这一块。本脚本给出:
- 源层学习贡献 vs 深层 oracle 贡献;
- 源层的理论最优(真双峰密度)对数似然 = 任何初始化/训练能达到的天花板;
- 二者之差 = EM 至多能带来的 joint_LL 提升。
"""
from __future__ import annotations

import sys
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

from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import (  # noqa: E402
    fit_layer_forest, forest_log_density, maxplus_cond_logdensity,
)
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources  # noqa: E402

cfg = dict(
    layer_sizes=[6, 6, 6, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
    n_total=50000, q=2, m=24, rank=8, source_mode="bimodal", src_sigma=0.06,
    lr=0.002, steps=1000, batch_sz=512, init_noise=0.01, train_noise=0.001,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, seed=0,
)

key = jax.random.PRNGKey(cfg["seed"])
spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
params = DelayParams(**cfg["delay"])
sources, kernels = ground_truth_samplers(spec, params)
sources = {**sources, **bimodal_sources(spec, cfg)}
k_data, key = jax.random.split(key)
xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
n_tr = int(0.7 * xs.shape[0])
train_x, test_x = xs[:n_tr], xs[n_tr:]
L0 = list(spec.layers[0])

k_l, key = jax.random.split(key)
forest = fit_layer_forest(jnp.asarray(train_x[:, L0]), L0, cfg, k_l, label="L0", mi_threshold=cfg["mi_threshold"])

ll_src, _ = forest_log_density(forest, test_x[:, L0])
ll_src = np.asarray(ll_src)
ll_oracle = np.zeros(test_x.shape[0])
for li in range(1, len(spec.layers)):
    for v in spec.layers[li]:
        ll_oracle += np.asarray(maxplus_cond_logdensity(test_x[:, v], test_x[:, list(spec.parents(v))], params))


def true_bimodal_logpdf(x, mu0=0.25, mu1=0.85, sig=0.06):
    def npdf(z, mu):
        return np.exp(-0.5 * ((z - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    return np.log(0.5 * npdf(x, mu0) + 0.5 * npdf(x, mu1))


ll_src_true = np.zeros(test_x.shape[0])
for v in L0:
    ll_src_true += true_bimodal_logpdf(test_x[:, v])

src_learned = ll_src.mean()
src_ceiling = ll_src_true.mean()
oracle = ll_oracle.mean()

print("\n==== layered_TTNS  joint_LL 构成与改进上限 ====")
print(f"源层 6 维 (可学)     : {src_learned:8.3f}")
print(f"源层 6 维 理论最优   : {src_ceiling:8.3f}   <- 任何初始化/训练的天花板")
print(f"深层 18 维 oracle 核 : {oracle:8.3f}   <- 精确, 不可学")
print("-" * 44)
print(f"当前 joint_LL        : {src_learned + oracle:8.3f}")
print(f"理论上限 joint_LL    : {src_ceiling + oracle:8.3f}")
print(f"EM 至多能提升        : {src_ceiling - src_learned:8.3f}")
