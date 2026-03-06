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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttde.dl_routine import batched_vmap  # noqa: E402
from ttde.score.experiment_setups import init_setups, model_setups  # noqa: E402
from ttde.tt.tt_opt import TTOpt, TTOperatorOpt, normalized_dot_operator, normalized_inner_product  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _safe_float(x) -> float | None:
    val = float(x)
    if np.isfinite(val):
        return val
    return None


def _recover_scalar(normalized) -> jnp.ndarray:
    return jnp.asarray(normalized.value * jnp.exp(normalized.log_norm)).reshape(())


def _nll_loss(model, params, xs: jnp.ndarray, batch_sz: int) -> jnp.ndarray:
    def log_p(x):
        return model.apply(params, x, method=model.log_p)

    return -batched_vmap(log_p, batch_sz)(xs).mean()


def _mean_log_p(model, params, xs: jnp.ndarray, batch_sz: int) -> jnp.ndarray:
    def log_p(x):
        return model.apply(params, x, method=model.log_p)

    return batched_vmap(log_p, batch_sz)(xs).mean()


def _log_prob_stats(model, params, xs: jnp.ndarray, batch_sz: int) -> Dict[str, jnp.ndarray]:
    def log_p(x):
        return model.apply(params, x, method=model.log_p)

    log_ps = batched_vmap(log_p, batch_sz)(xs)
    finite_mask = log_ps != -jnp.inf
    finite_sum = jnp.sum(jnp.where(finite_mask, log_ps, 0.0))
    finite_count = finite_mask.sum()
    finite_mean_ll = finite_sum / jnp.maximum(finite_count, 1)
    nonpositive_rate = jnp.clip(1.0 - finite_count / len(log_ps), 0.0, 1.0)
    return {
        "finite_mean_ll": finite_mean_ll,
        "nonpositive_rate": nonpositive_rate,
    }


def _extract_single_component_tt(tt_batched: TTOpt, component: int = 0) -> TTOpt:
    return TTOpt(
        first=tt_batched.first[component],
        inner=tt_batched.inner[component],
        last=tt_batched.last[component],
    )


def _eval_pair_marginal_sq_tt_on_grid(
    tt: TTOpt,
    bases,
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

    basis_call = type(bases).__call__
    basis_i = jax.tree_util.tree_map(lambda arr: arr[dim_i], bases)
    basis_j = jax.tree_util.tree_map(lambda arr: arr[dim_j], bases)
    gram_matrices = jax.vmap(type(bases).l2_integral)(bases)
    z = _recover_scalar(
        normalized_inner_product(
            tt,
            normalized_dot_operator(tt, TTOperatorOpt.rank_1_from_matrices(gram_matrices)),
        )
    )

    def one_eval(point: jnp.ndarray) -> jnp.ndarray:
        vi = basis_call(basis_i, point[0])
        vj = basis_call(basis_j, point[1])
        mats = gram_matrices.at[dim_i].set(jnp.outer(vi, vi)).at[dim_j].set(jnp.outer(vj, vj))
        op = TTOperatorOpt.rank_1_from_matrices(mats)
        numer = _recover_scalar(normalized_inner_product(tt, normalized_dot_operator(tt, op)))
        return numer / z

    eval_chunk = jax.jit(lambda chunk: jax.vmap(one_eval)(chunk))
    vals_parts = []
    for start in range(0, pts.shape[0], chunk_size):
        vals_parts.append(eval_chunk(pts[start : start + chunk_size]))
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

    cfg = {
        "q": args.q,
        "m": args.m,
        "rank": args.rank,
        "batch_sz": args.batch_sz,
        "lr": args.lr,
        "train_steps": args.train_steps,
        "log_every": args.log_every,
        "init_noise": args.init_noise,
        "train_noise": args.train_noise,
        "seed": args.seed,
        "monitor_train_sz": args.monitor_train_sz,
        "monitor_val_sz": args.monitor_val_sz,
    }

    model_setup = model_setups.PAsTTSqrOpt(q=args.q, m=args.m, rank=args.rank, n_comps=1)
    init_setup = init_setups.CanonicalRankK(em_steps=50, noise=args.init_noise)

    key = jax.random.PRNGKey(args.seed + 1201)
    key, k_model, k_init, k_iter = jax.random.split(key, 4)
    model = model_setup.create(k_model, train_x)
    params = init_setup(model, k_init, train_x)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    early_stop_enabled = not args.disable_early_stop
    early_stop_restore_best = not args.disable_early_stop_restore_best

    train_monitor = train_x[: args.monitor_train_sz]
    val_monitor = val_x[: args.monitor_val_sz]

    @jax.jit
    def train_step(curr_params, curr_opt_state, noisy_batch):
        loss, grads = jax.value_and_grad(lambda p: _nll_loss(model, p, noisy_batch, args.batch_sz))(curr_params)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
        next_params = optax.apply_updates(curr_params, updates)
        return next_params, next_opt_state, loss

    eval_nll = jax.jit(lambda p, xs: _nll_loss(model, p, xs, args.batch_sz))
    eval_mean_ll = jax.jit(lambda p, xs: _mean_log_p(model, p, xs, args.batch_sz))
    eval_log_int_p = jax.jit(lambda p: model.apply(p, method=model.log_int_p))
    eval_stats = jax.jit(lambda p, xs: _log_prob_stats(model, p, xs, args.batch_sz))

    history: List[Dict] = []
    best_val_nll = float("inf")
    best_step = 0
    best_params = params
    bad_logs = 0
    stopped_early = False
    stop_step = args.train_steps
    t_start = time.perf_counter()
    t_prev = t_start
    print("\n=== [complex target] [ttde_tt_mle] ===", flush=True)
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
        "model_name": "ttde_tt_mle",
        "model_topology": "tt",
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

    tt = _extract_single_component_tt(params["tt"]["tt"], component=0)
    densities = {}
    for idx_pair, ((dim_i, dim_j), pair_range) in enumerate(zip(slice_pairs, pair_ranges)):
        xi_centers = pair_range[0]
        xj_centers = pair_range[1]
        densities[f"density_{idx_pair}"] = _eval_pair_marginal_sq_tt_on_grid(
            tt,
            model.bases,
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
