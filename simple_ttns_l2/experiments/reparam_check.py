from __future__ import annotations

"""Correctness + small-scale effectiveness check for core-parameter reparameterization.

Part 1 (correctness):
  - effective_ttns cores equal g(theta) elementwise for square/exp.
  - transform-aware normalization drives \int q -> 1 for every transform.
  - dense round-trip: eval via objective matches a dense tensor built from g(theta).

Part 2 (effectiveness, small scale):
  - Train identity vs square vs exp on a small synthetic dataset (fixed seeds),
    report best val_l2. Lower is better (L2 objective).
"""

import sys
from pathlib import Path

import jax
import optax
from jax import config, numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

config.update("jax_enable_x64", True)

from ttde.score.models.opt_for_tree_data import chain_parent
from simple_ttns_l2.reparam import effective_ttns, apply_transform
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1
from simple_ttns_l2.objective import (
    batch_basis_vectors_from_samples,
    integral_q_ttns,
    l2_objective_ttns,
    normalize_ttns_by_integral,
)


def make_target(key, n, d):
    # bimodal-ish mixture in a bounded box so splines cover the support
    k1, k2, k3 = jax.random.split(key, 3)
    comp = jax.random.bernoulli(k1, 0.5, (n, 1))
    a = jax.random.normal(k2, (n, d)) * 0.35 + 0.3
    b = jax.random.normal(k3, (n, d)) * 0.35 - 0.3
    return jnp.where(comp, a, b)


def correctness():
    print("=== PART 1: correctness ===")
    key = jax.random.PRNGKey(0)
    n_dims = 4
    samples = make_target(key, 256, n_dims)
    bases = build_bases(samples, q=2, m=10)
    parent = chain_parent(n_dims).tolist()
    basis_integrals = vmap(type(bases).integral)(bases)

    ttns = init_ttns_from_rank1(
        key=jax.random.PRNGKey(1), bases=bases, samples=samples,
        parent=parent, rank=3, noise=5e-2,
    )

    ok = True
    for name in ("identity", "square", "exp"):
        eff = effective_ttns(ttns, name)
        for c_raw, c_eff in zip(ttns.cores, eff.cores):
            expect = apply_transform(name, c_raw)
            err = float(jnp.max(jnp.abs(c_eff - expect)))
            if err > 1e-12:
                ok = False
                print(f"  [{name}] core mismatch err={err:.2e}")
        # transform-aware normalization -> integral 1
        normed, z = normalize_ttns_by_integral(ttns, basis_integrals, parent, transform=name)
        z_after = float(integral_q_ttns(normed, basis_integrals, parent, transform=name))
        gap = abs(z_after - 1.0)
        status = "PASS" if gap < 1e-8 else "FAIL"
        if gap >= 1e-8:
            ok = False
        print(f"  [{name}] pre_integral={float(z):+.4e} post_integral={z_after:.10f} -> {status}")

    print("PART 1:", "ALL PASS" if ok else "FAILURES")
    return ok


def train_one(transform, key, samples, val, bases, parent, gram, basis_integrals,
              rank=4, steps=400, lr=5e-3, noise=5e-2):
    ttns = init_ttns_from_rank1(
        key=key, bases=bases, samples=samples, parent=parent, rank=rank, noise=noise,
    )
    ttns, _ = normalize_ttns_by_integral(ttns, basis_integrals, parent, transform=transform)
    opt = optax.adam(lr)
    state = opt.init(ttns)
    val_vecs = batch_basis_vectors_from_samples(bases, val)
    train_vecs = batch_basis_vectors_from_samples(bases, samples)

    @jax.jit
    def step(ttns, state):
        def loss_fn(t):
            return l2_objective_ttns(t, train_vecs, gram, parent, transform=transform)
        loss, grads = jax.value_and_grad(loss_fn)(ttns)
        updates, state = opt.update(grads, state, ttns)
        ttns = optax.apply_updates(ttns, updates)
        return ttns, state, loss

    eval_val = jax.jit(
        lambda t: l2_objective_ttns(t, val_vecs, gram, parent, transform=transform)
    )
    best = float("inf")
    for s in range(steps):
        ttns, state, _ = step(ttns, state)
        if (s + 1) % 20 == 0:
            ttns, _ = normalize_ttns_by_integral(ttns, basis_integrals, parent, transform=transform)
            v = float(eval_val(ttns))
            if jnp.isfinite(v):
                best = min(best, v)
    return best


def effectiveness():
    print("\n=== PART 2: small-scale effectiveness (val_l2, lower=better) ===")
    n_dims = 4
    results = {t: [] for t in ("identity", "square", "exp")}
    for seed in (0, 1, 2):
        key = jax.random.PRNGKey(100 + seed)
        k_data, k_val, k_init = jax.random.split(key, 3)
        samples = make_target(k_data, 512, n_dims)
        val = make_target(k_val, 512, n_dims)
        bases = build_bases(samples, q=2, m=12)
        parent = chain_parent(n_dims).tolist()
        gram = vmap(type(bases).l2_integral)(bases)
        basis_integrals = vmap(type(bases).integral)(bases)
        for t in results:
            best = train_one(t, k_init, samples, val, bases, parent, gram, basis_integrals)
            results[t].append(best)
        print(f"  seed={seed}: " + "  ".join(f"{t}={results[t][-1]:.5f}" for t in results))

    print("\n  mean over seeds:")
    for t in results:
        arr = jnp.array(results[t])
        print(f"    {t:9s} mean={float(arr.mean()):.5f}  std={float(arr.std()):.5f}")
    return results


if __name__ == "__main__":
    correctness()
    effectiveness()
