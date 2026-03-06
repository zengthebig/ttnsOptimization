from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import lax

REPO_ROOT = Path(__file__).resolve().parents[2]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from ttde.dl_routine import batched_vmap  # noqa: E402
from ttde.score.experiment_setups import init_setups, model_setups  # noqa: E402
from ttde.ttns.ttns_opt import (  # noqa: E402
    TTNSOpt,
    batch_log_abs_eval_rank1_ttns,
    batch_quadratic_form_ttns,
    normalized_quadratic_form_ttns,
    quadratic_form_ttns,
)

jax.config.update("jax_enable_x64", True)


def _safe_float(x) -> float | None:
    val = float(x)
    if np.isfinite(val):
        return val
    return None


def _basis_vectors_from_samples(bases, xs: jnp.ndarray) -> jnp.ndarray:
    basis_call = type(bases).__call__
    return jax.vmap(lambda x: jax.vmap(basis_call)(bases, x))(xs)


def _batch_log_p_fast(
    model,
    params,
    xs: jnp.ndarray,
    batch_sz: int,
    parent: Sequence[int],
    perm: jnp.ndarray,
    gram_matrices_perm: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute log p(x) for a batch using shared normalization:
      log p(x) = unnormalized_log_p(x) - log_int_p
    This avoids recomputing log_int_p for each sample.
    """
    ttns = _extract_single_component_ttns(params["ttns"]["ttns"], component=0)
    log_int_p = normalized_quadratic_form_ttns(ttns, gram_matrices_perm, parent).log_norm

    def one_chunk(chunk_xs: jnp.ndarray) -> jnp.ndarray:
        basis_vectors = _basis_vectors_from_samples(model.bases, chunk_xs)
        basis_vectors = basis_vectors[:, perm, :]
        log_abs_q = batch_log_abs_eval_rank1_ttns(ttns, basis_vectors, parent)
        return 2.0 * log_abs_q - log_int_p

    n = xs.shape[0]
    n_batches = n // batch_sz
    xs_main = xs[: n_batches * batch_sz].reshape(n_batches, batch_sz, *xs.shape[1:])

    def body(_, chunk):
        return None, one_chunk(chunk)

    main_vals = lax.scan(body, None, xs_main)[1].reshape(-1)
    rem_vals = one_chunk(xs[n_batches * batch_sz :])
    return jnp.concatenate([main_vals, rem_vals], axis=0)


def _batch_log_p_legacy(model, params, xs: jnp.ndarray, batch_sz: int) -> jnp.ndarray:
    def log_p(x):
        return model.apply(params, x, method=model.log_p)

    return batched_vmap(log_p, batch_sz)(xs)


def _batch_log_p(
    model,
    params,
    xs: jnp.ndarray,
    batch_sz: int,
    use_fast_nll: bool,
    parent: Sequence[int],
    perm: jnp.ndarray,
    gram_matrices_perm: jnp.ndarray,
) -> jnp.ndarray:
    if use_fast_nll:
        return _batch_log_p_fast(model, params, xs, batch_sz, parent, perm, gram_matrices_perm)
    return _batch_log_p_legacy(model, params, xs, batch_sz)


def _nll_loss(
    model,
    params,
    xs: jnp.ndarray,
    batch_sz: int,
    use_fast_nll: bool,
    parent: Sequence[int],
    perm: jnp.ndarray,
    gram_matrices_perm: jnp.ndarray,
) -> jnp.ndarray:
    return -_batch_log_p(
        model,
        params,
        xs,
        batch_sz,
        use_fast_nll,
        parent,
        perm,
        gram_matrices_perm,
    ).mean()


def _mean_log_p(
    model,
    params,
    xs: jnp.ndarray,
    batch_sz: int,
    use_fast_nll: bool,
    parent: Sequence[int],
    perm: jnp.ndarray,
    gram_matrices_perm: jnp.ndarray,
) -> jnp.ndarray:
    return _batch_log_p(
        model,
        params,
        xs,
        batch_sz,
        use_fast_nll,
        parent,
        perm,
        gram_matrices_perm,
    ).mean()


def _log_prob_stats(
    model,
    params,
    xs: jnp.ndarray,
    batch_sz: int,
    use_fast_nll: bool,
    parent: Sequence[int],
    perm: jnp.ndarray,
    gram_matrices_perm: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    log_ps = _batch_log_p(
        model,
        params,
        xs,
        batch_sz,
        use_fast_nll,
        parent,
        perm,
        gram_matrices_perm,
    )
    finite_mask = log_ps != -jnp.inf
    finite_sum = jnp.sum(jnp.where(finite_mask, log_ps, 0.0))
    finite_count = finite_mask.sum()
    finite_mean_ll = finite_sum / jnp.maximum(finite_count, 1)
    nonpositive_rate = jnp.clip(1.0 - finite_count / len(log_ps), 0.0, 1.0)
    return {
        "finite_mean_ll": finite_mean_ll,
        "nonpositive_rate": nonpositive_rate,
    }


def _extract_single_component_ttns(ttns_batched: TTNSOpt, component: int = 0) -> TTNSOpt:
    return TTNSOpt(tuple(core[component] for core in ttns_batched.cores))


def _eval_pair_marginal_sq_ttns_on_grid(
    ttns: TTNSOpt,
    parent: Sequence[int],
    bases,
    gram_matrices: jnp.ndarray,
    dim_i: int,
    dim_j: int,
    xi_centers: np.ndarray,
    xj_centers: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    xi = jnp.asarray(xi_centers, dtype=jnp.float64)
    xj = jnp.asarray(xj_centers, dtype=jnp.float64)
    gx, gy = jnp.meshgrid(xi, xj, indexing="ij")
    pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

    base_mats = jnp.asarray(gram_matrices, dtype=jnp.float64)
    basis_call = type(bases).__call__
    basis_i = jax.tree_util.tree_map(lambda arr: arr[dim_i], bases)
    basis_j = jax.tree_util.tree_map(lambda arr: arr[dim_j], bases)
    z = quadratic_form_ttns(ttns, base_mats, parent)
    batch_quad = jax.jit(lambda mats_batch: batch_quadratic_form_ttns(ttns, mats_batch, parent))

    vals_parts = []
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start : start + chunk_size]
        vi = jax.vmap(lambda point: basis_call(basis_i, point[0]))(chunk)
        vj = jax.vmap(lambda point: basis_call(basis_j, point[1]))(chunk)
        mats_batch = jnp.broadcast_to(base_mats, (chunk.shape[0],) + base_mats.shape)
        mats_batch = mats_batch.at[:, dim_i].set(jax.vmap(jnp.outer)(vi, vi))
        mats_batch = mats_batch.at[:, dim_j].set(jax.vmap(jnp.outer)(vj, vj))
        vals_parts.append(batch_quad(mats_batch) / z)

    vals = jnp.concatenate(vals_parts, axis=0)
    return np.asarray(vals).reshape((len(xi_centers), len(xj_centers)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--density-npz", type=Path, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--batch-sz", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--log-every", type=int, required=True)
    parser.add_argument("--init-noise", type=float, required=True)
    parser.add_argument("--train-noise", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--monitor-train-sz", type=int, required=True)
    parser.add_argument("--monitor-val-sz", type=int, required=True)
    parser.add_argument("--tree-topology", type=str, default="balanced")
    parser.add_argument(
        "--tree-parent-json",
        type=Path,
        default=None,
        help="Optional JSON file containing an explicit parent array for a custom TTNS tree.",
    )
    parser.add_argument("--em-steps", type=int, default=50)
    parser.add_argument(
        "--use-fast-nll",
        action="store_true",
        help="Use batch-shared normalization optimization for NLL path (off by default).",
    )
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="Disable early stopping (enabled by default).",
    )
    parser.add_argument(
        "--early-stop-patience-logs",
        type=int,
        default=12,
        help="Patience in number of log events.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val_nll improvement to reset patience.",
    )
    parser.add_argument(
        "--early-stop-warmup-logs",
        type=int,
        default=5,
        help="Warmup log events before patience counting starts.",
    )
    parser.add_argument(
        "--disable-early-stop-restore-best",
        action="store_true",
        help="Disable restoring the best validation checkpoint after early stop.",
    )
    args = parser.parse_args()

    data = np.load(args.data_npz)
    train_x = jnp.asarray(data["train_x"], dtype=jnp.float64)
    val_x = jnp.asarray(data["val_x"], dtype=jnp.float64)
    test_x = jnp.asarray(data["test_x"], dtype=jnp.float64)
    slice_pairs = np.asarray(data["slice_pairs"], dtype=np.int32)
    pair_ranges = np.asarray(data["pair_ranges"], dtype=np.float64)

    requested_tree_parent = None
    if args.tree_parent_json is not None:
        requested_tree_parent = tuple(int(x) for x in json.loads(args.tree_parent_json.read_text(encoding="utf-8")))

    model_label = args.tree_topology if requested_tree_parent is None else "custom"

    model_setup = model_setups.PAsTTNSSqrOpt(
        q=args.q,
        m=args.m,
        rank=args.rank,
        n_comps=1,
        tree_topology=args.tree_topology,
        tree_parent=requested_tree_parent,
    )
    init_setup = init_setups.CanonicalRankK(em_steps=args.em_steps, noise=args.init_noise)

    key = jax.random.PRNGKey(args.seed + 900)
    key, k_model, k_init, k_iter = jax.random.split(key, 4)
    model = model_setup.create(k_model, train_x)
    params = init_setup(model, k_init, train_x)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    use_fast_nll = bool(args.use_fast_nll)
    early_stop_enabled = not args.disable_early_stop
    early_stop_restore_best = not args.disable_early_stop_restore_best
    parent = tuple(int(x) for x in np.asarray(model.tree_parent))
    perm = jnp.asarray(np.asarray(model.permutations[0]), dtype=jnp.int32)
    gram_matrices = jax.vmap(type(model.bases).l2_integral)(model.bases)
    gram_matrices_perm = gram_matrices[perm]

    train_monitor = train_x[: args.monitor_train_sz]
    val_monitor = val_x[: args.monitor_val_sz]

    @jax.jit
    def train_step(curr_params, curr_opt_state, noisy_batch):
        loss, grads = jax.value_and_grad(
            lambda p: _nll_loss(
                model,
                p,
                noisy_batch,
                args.batch_sz,
                use_fast_nll,
                parent,
                perm,
                gram_matrices_perm,
            )
        )(curr_params)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
        next_params = optax.apply_updates(curr_params, updates)
        return next_params, next_opt_state, loss

    eval_nll = jax.jit(
        lambda p, xs: _nll_loss(
            model, p, xs, args.batch_sz, use_fast_nll, parent, perm, gram_matrices_perm
        )
    )
    eval_mean_ll = jax.jit(
        lambda p, xs: _mean_log_p(
            model, p, xs, args.batch_sz, use_fast_nll, parent, perm, gram_matrices_perm
        )
    )
    eval_log_int_p = jax.jit(lambda p: model.apply(p, method=model.log_int_p))
    eval_stats = jax.jit(
        lambda p, xs: _log_prob_stats(
            model, p, xs, args.batch_sz, use_fast_nll, parent, perm, gram_matrices_perm
        )
    )

    history: List[Dict] = []
    best_val_nll = float("inf")
    best_step = 0
    best_params = params
    bad_logs = 0
    stopped_early = False
    stop_step = args.train_steps
    t_start = time.perf_counter()
    t_prev = t_start
    print(f"\n=== [complex target] [ttnsde_ttns_mle:{model_label}] ===", flush=True)
    print(f"resolved_tree_parent={parent}", flush=True)
    print(f"use_fast_nll={use_fast_nll}", flush=True)
    print(
        f"early_stop_enabled={early_stop_enabled}, patience_logs={args.early_stop_patience_logs}, "
        f"min_delta={args.early_stop_min_delta}, warmup_logs={args.early_stop_warmup_logs}, "
        f"restore_best={early_stop_restore_best}",
        flush=True,
    )
    print("step,train_nll,val_nll,val_ll,log_int_p,interval_sec,total_sec", flush=True)

    for step in range(1, args.train_steps + 1):
        k_iter, k_idx, k_noise = jax.random.split(k_iter, 3)
        idx = jax.random.randint(k_idx, (args.batch_sz,), 0, train_x.shape[0])
        batch = train_x[idx]
        noisy_batch = batch + jax.random.normal(k_noise, batch.shape) * args.train_noise
        params, opt_state, _ = train_step(params, opt_state, noisy_batch)

        if step % args.log_every == 0:
            now = time.perf_counter()
            train_nll = float(eval_nll(params, train_monitor))
            val_nll = float(eval_nll(params, val_monitor))
            val_ll = float(eval_mean_ll(params, val_monitor))
            log_int_p = float(eval_log_int_p(params))
            interval = now - t_prev
            total = now - t_start
            t_prev = now
            history.append(
                {
                    "step": step,
                    "train_nll": train_nll,
                    "val_nll": val_nll,
                    "val_ll": val_ll,
                    "log_int_p": log_int_p,
                    "interval_sec": interval,
                    "total_sec": total,
                }
            )
            print(
                f"{step},{train_nll:.6f},{val_nll:.6f},{val_ll:.6f},{log_int_p:.6f},{interval:.3f},{total:.3f}",
                flush=True,
            )

            if np.isfinite(val_nll) and (best_step == 0 or val_nll + args.early_stop_min_delta < best_val_nll):
                best_val_nll = val_nll
                best_step = step
                best_params = params
                bad_logs = 0
            elif early_stop_enabled and len(history) > args.early_stop_warmup_logs:
                bad_logs += 1
                if bad_logs >= args.early_stop_patience_logs:
                    stopped_early = True
                    stop_step = step
                    print(
                        f"early_stop_triggered at step={step}, best_step={best_step}, best_val_nll={best_val_nll:.6f}",
                        flush=True,
                    )
                    break

    if early_stop_enabled and early_stop_restore_best and best_step > 0:
        params = best_params

    total_time = time.perf_counter() - t_start
    train_stats = eval_stats(params, train_x)
    val_stats = eval_stats(params, val_x)
    test_stats = eval_stats(params, test_x)
    summary = {
        "model_name": "ttnsde_ttns_mle",
        "model_topology": model_label,
        "tree_parent": list(parent),
        "use_fast_nll": bool(use_fast_nll),
        "native_metric_name": "val_nll",
        "early_stop_enabled": bool(early_stop_enabled),
        "early_stop_restore_best": bool(early_stop_restore_best),
        "early_stop_patience_logs": int(args.early_stop_patience_logs),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "early_stop_warmup_logs": int(args.early_stop_warmup_logs),
        "stopped_early": bool(stopped_early),
        "stop_step": int(stop_step),
        "best_step": int(best_step),
        "best_val_nll": _safe_float(best_val_nll),
        "final_train_nll": _safe_float(eval_nll(params, train_x)),
        "final_val_nll": _safe_float(eval_nll(params, val_x)),
        "final_test_nll": _safe_float(eval_nll(params, test_x)),
        "final_train_ll": _safe_float(eval_mean_ll(params, train_x)),
        "final_val_ll": _safe_float(eval_mean_ll(params, val_x)),
        "final_test_ll": _safe_float(eval_mean_ll(params, test_x)),
        "finite_train_ll": float(train_stats["finite_mean_ll"]),
        "finite_val_ll": float(val_stats["finite_mean_ll"]),
        "finite_test_ll": float(test_stats["finite_mean_ll"]),
        "train_nonpositive_rate": float(train_stats["nonpositive_rate"]),
        "val_nonpositive_rate": float(val_stats["nonpositive_rate"]),
        "test_nonpositive_rate": float(test_stats["nonpositive_rate"]),
        "final_log_int_p": float(eval_log_int_p(params)),
        "total_time_sec": total_time,
        "history": history,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ttns = _extract_single_component_ttns(params["ttns"]["ttns"], component=0)
    parent = list(parent)
    densities = {}
    for idx_pair, ((dim_i, dim_j), pair_range) in enumerate(zip(slice_pairs, pair_ranges)):
        xi_centers = pair_range[0]
        xj_centers = pair_range[1]
        densities[f"density_{idx_pair}"] = _eval_pair_marginal_sq_ttns_on_grid(
            ttns,
            parent,
            model.bases,
            gram_matrices,
            int(dim_i),
            int(dim_j),
            xi_centers,
            xj_centers,
        )
    np.savez_compressed(args.density_npz, **densities)
    print(f"wrote summary: {args.summary_json}", flush=True)
    print(f"wrote densities: {args.density_npz}", flush=True)


if __name__ == "__main__":
    main()
