"""只跑 TTDE(平方+TT, 链式)一支, 分段计时(拟合/评估/采样), 复杂配置。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
jax.config.update("jax_enable_x64", True)

from simple_ttns_l2.dag_pipeline import build_layered_spec, sample_joint  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, ground_truth_samplers  # noqa: E402
from simple_ttns_l2.experiments.compare_global_vs_layered_plot import bimodal_sources, sample_metrics  # noqa: E402
from simple_ttns_l2.experiments.ttde_tt_baseline import fit_ttde_tt, ttde_logp, sample_ttde_tt  # noqa: E402

cfg = dict(
    layer_sizes=[6, 6, 6, 6], fanin=2,
    delay=dict(src_lo=0.0, src_hi=1.0, edge_lo=0.0, edge_hi=0.25, node_lo=0.0, node_hi=0.25),
    n_total=50000, q=2, m=24, source_mode="bimodal", src_sigma=0.06,
    lr=0.002, batch_sz=512, init_noise=0.01, train_noise=0.001, log_every=250,
    ttde_rank=18, ttde_steps=2000, ttde_em_steps=50, ttde_patience=8,
    ttde_grid=200, ttde_n_sample=8000, monitor_val_sz=4000, n_sample=8000, seed=0,
)

key = jax.random.PRNGKey(cfg["seed"])
spec = build_layered_spec(cfg["layer_sizes"], fanin=cfg["fanin"], wrap=True)
params_d = DelayParams(**cfg["delay"])
sources, kernels = ground_truth_samplers(spec, params_d)
sources = {**sources, **bimodal_sources(spec, cfg)}
k_data, key = jax.random.split(key)
xs = np.asarray(sample_joint(spec, sources, kernels, k_data, cfg["n_total"], clip=(-1e9, 1e9)))
n_tr = int(0.7 * xs.shape[0])
train_x, test_x = xs[:n_tr], xs[n_tr:]
val_x = train_x[int(0.85 * n_tr):]

t0 = time.perf_counter()
model, params, info = fit_ttde_tt(train_x, val_x, cfg, cfg["seed"])
t_fit = time.perf_counter() - t0

t0 = time.perf_counter()
lp = ttde_logp(model, params, test_x, batch_sz=cfg["batch_sz"])
finite = np.isfinite(lp)
joint_ll = float(lp[finite].mean())
t_eval = time.perf_counter() - t0

t0 = time.perf_counter()
k_t, key = jax.random.split(key)
samp = sample_ttde_tt(model, params, k_t, cfg["ttde_n_sample"], grid_size=cfg["ttde_grid"])
t_sample = time.perf_counter() - t0
m = sample_metrics(samp, test_x[:samp.shape[0]])

print("\n==== TTDE(平方+TT, 链式) 计时(复杂配置, CPU x64) ====")
print(f"params={info['learned_params']}  joint_LL={joint_ll:.3f}  W1={m['w1_marg']:.4f}  corr_fro={m['corr_fro']:.3f}")
print(f"fit    : {t_fit:7.1f} s")
print(f"eval   : {t_eval:7.1f} s")
print(f"sample : {t_sample:7.1f} s  (n={cfg['ttde_n_sample']})")
print(f"TOTAL  : {t_fit + t_eval + t_sample:7.1f} s")
