"""只跑 layered_TTNS 一支, 分段计时(拟合/评估/采样), 复杂配置。"""
from __future__ import annotations

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

from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.layered_forest import (  # noqa: E402
    fit_layer_forest, forest_log_density, maxplus_cond_logdensity,
)
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources, sample_layered, sample_metrics  # noqa: E402
from simple_ttns_l2.experiments.fit_deep_dag_vs_tree import _count_params  # noqa: E402

cfg = dict(
    layer_sizes=[6, 6, 6, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
    n_total=50000, q=2, m=24, rank=8, source_mode="bimodal", src_sigma=0.06,
    lr=0.002, steps=1000, batch_sz=512, init_noise=0.01, train_noise=0.001,
    log_every=250, early_stop_patience=8, mi_threshold=0.02, n_sample=8000, seed=0,
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

# ---- 拟合(每层一个 Chow-Liu TTNS 森林) ----
t0 = time.perf_counter()
layer_forests, full_params, per_layer_time = [], 0, []
for li, layer in enumerate(spec.layers):
    tl = time.perf_counter()
    Lg = list(layer)
    k_l, key = jax.random.split(key)
    f = fit_layer_forest(jnp.asarray(train_x[:, Lg]), Lg, cfg, k_l, label=f"L{li}", mi_threshold=cfg["mi_threshold"])
    layer_forests.append(f)
    full_params += sum(_count_params(bm.ttns.cores) for bm in f)
    per_layer_time.append(time.perf_counter() - tl)
t_fit = time.perf_counter() - t0
forest = layer_forests[0]

# ---- 评估 joint_LL ----
t0 = time.perf_counter()
ll, nonpos = forest_log_density(forest, test_x[:, L0])
ll = ll.copy()
for li in range(1, len(spec.layers)):
    for v in spec.layers[li]:
        ll += maxplus_cond_logdensity(test_x[:, v], test_x[:, list(spec.parents(v))], params)
joint_ll = float(np.asarray(ll).mean())
t_eval = time.perf_counter() - t0

# ---- 采样 + 指标 ----
t0 = time.perf_counter()
k_s, key = jax.random.split(key)
samp = sample_layered(forest, spec, params, k_s, cfg["n_sample"], cfg["seed"] + 1)
t_sample = time.perf_counter() - t0
m = sample_metrics(samp, test_x[:cfg["n_sample"]])

print("\n==== layered_TTNS 计时(复杂配置, CPU x64) ====")
print(f"params={full_params}  joint_LL={joint_ll:.3f}  W1={m['w1_marg']:.4f}  corr_fro={m['corr_fro']:.3f}")
print(f"fit    : {t_fit:7.1f} s   per-layer={['%.1f'%x for x in per_layer_time]}")
print(f"eval   : {t_eval:7.1f} s")
print(f"sample : {t_sample:7.1f} s  (n={cfg['n_sample']})")
print(f"TOTAL  : {t_fit + t_eval + t_sample:7.1f} s")
