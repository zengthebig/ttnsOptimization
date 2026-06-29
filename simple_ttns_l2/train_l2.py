from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence, Tuple

import click
import flax
import jax
import optax
from jax import config, numpy as jnp, value_and_grad, vmap

# Ensure this standalone package imports TTNSDE/ttde, not repository-root ttde.
REPO_ROOT = Path(__file__).resolve().parents[1]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from ttde import utils
from ttde.dl_routine import KEY
from ttde.score.experiment_setups.data_setups import NAME_TO_DATASET
from ttde.score.experiment_setups import data_setups
from ttde.score.models.continuous_canonical_init import continuous_rank_1
from ttde.score.models.opt_for_tree_data import balanced_parent, chain_parent
from ttde.tt.basis import SplineOnKnots, create_space_uniform_knots
from ttde.ttns.ttns_opt import TTNSOpt
from simple_ttns_l2.objective import (
    batch_basis_vectors_from_samples,
    integral_q_ttns,
    l2_objective_ttns,
    normalize_ttns_by_integral,
)

config.update("jax_enable_x64", True)


def one_basis(m: int, q: int, xs: jnp.ndarray):
    return SplineOnKnots.from_knots(q, create_space_uniform_knots(xs, m, q))


def build_bases(samples: jnp.ndarray, q: int, m: int):
    return vmap(one_basis, in_axes=(None, None, 1))(m, q, samples)


def make_parent(n_dims: int, topology: str) -> list[int]:
    if topology == "balanced":
        return balanced_parent(n_dims).tolist()
    if topology == "chain":
        return chain_parent(n_dims).tolist()
    raise ValueError(f"unsupported ttns_topology={topology}")


def init_ttns_from_rank1(
    key: jnp.ndarray,
    bases,
    samples: jnp.ndarray,
    parent: Sequence[int],
    rank: int,
    noise: float,
    edge_ranks=None,
) -> TTNSOpt:
    rank1 = continuous_rank_1(bases, samples, jnp.ones(len(samples)))
    ttns = TTNSOpt.from_rank1_vectors(rank1, parent, rank, edge_ranks=edge_ranks)
    if noise <= 0:
        return ttns

    keys = jax.random.split(key, len(ttns.cores))
    cores = []
    for core, curr_key in zip(ttns.cores, keys):
        cores.append(core + jax.random.normal(curr_key, core.shape) * noise)
    return TTNSOpt(tuple(cores))


def l2_loss_on_batch(
    ttns: TTNSOpt,
    bases,
    xs_batch: jnp.ndarray,
    parent: Sequence[int],
    gram_matrices: jnp.ndarray,
    stable: bool = False,
) -> jnp.ndarray:
    basis_vectors_batch = batch_basis_vectors_from_samples(bases, xs_batch)
    return l2_objective_ttns(ttns, basis_vectors_batch, gram_matrices, parent, stable=stable)


def l2_train_step(
    ttns: TTNSOpt,
    opt_state,
    optimizer,
    bases,
    xs_batch: jnp.ndarray,
    parent: Sequence[int],
    gram_matrices: jnp.ndarray,
    key_noise: jnp.ndarray,
    train_noise: float,
) -> Tuple[TTNSOpt, Any, jnp.ndarray]:
    noisy_batch = xs_batch + jax.random.normal(key_noise, xs_batch.shape) * train_noise

    def loss_fn(curr_ttns):
        return l2_loss_on_batch(curr_ttns, bases, noisy_batch, parent, gram_matrices)

    loss, grads = value_and_grad(loss_fn)(ttns)
    updates, opt_state = optimizer.update(grads, opt_state, ttns)
    ttns = optax.apply_updates(ttns, updates)
    return ttns, opt_state, loss


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(NAME_TO_DATASET.keys(), case_sensitive=False),
    required=True,
    help=f'Name of the dataset. Choose one of {", ".join(NAME_TO_DATASET.keys())}',
)
@click.option("--q", type=int, required=True, help="degree of splines")
@click.option("--m", type=int, required=True, help="number of basis functions")
@click.option("--rank", type=int, required=True, help="TTNS virtual rank")
@click.option(
    "--ttns-topology",
    type=click.Choice(["balanced", "chain"], case_sensitive=False),
    default="balanced",
    show_default=True,
    help="tree topology for TTNS model",
)
@click.option("--init-noise", type=float, default=1e-2, show_default=True, help="noise added to init TTNS cores")
@click.option("--batch-sz", type=int, required=True, help="batch size")
@click.option("--train-noise", type=float, default=0.0, show_default=True, help="Gaussian noise added to train batch")
@click.option("--lr", type=float, required=True, help="learning rate for Adam optimizer")
@click.option("--train-steps", type=int, required=True, help="number of train steps")
@click.option(
    "--normalize-every",
    type=int,
    default=1,
    show_default=True,
    help="normalize q by integral every N steps; <=0 disables",
)
@click.option("--seed", type=int, default=0, show_default=True, help="PRNG seed")
@click.option("--val-sz", type=int, default=4096, show_default=True, help="number of validation samples for periodic monitor")
@click.option(
    "--early-stop-patience-logs",
    type=int,
    default=6,
    show_default=True,
    help="early stop patience in number of validation logs; <=0 disables",
)
@click.option(
    "--early-stop-min-delta",
    type=float,
    default=1e-4,
    show_default=True,
    help="minimum val_l2 improvement to reset early stop patience",
)
@click.option(
    "--early-stop-warmup-logs",
    type=int,
    default=4,
    show_default=True,
    help="number of initial validation logs ignored by early stop bad-log counting",
)
@click.option(
    "--early-stop-restore-best/--no-early-stop-restore-best",
    default=True,
    show_default=True,
    help="restore best-val_l2 model parameters after training loop",
)
@click.option("--data-dir", type=Path, required=True, help="directory with MAF datasets")
@click.option("--work-dir", type=Path, required=True, help="directory where to store outputs")
def main(
    dataset: str,
    q: int,
    m: int,
    rank: int,
    ttns_topology: str,
    init_noise: float,
    batch_sz: int,
    train_noise: float,
    lr: float,
    train_steps: int,
    normalize_every: int,
    seed: int,
    val_sz: int,
    early_stop_patience_logs: int,
    early_stop_min_delta: float,
    early_stop_warmup_logs: int,
    early_stop_restore_best: bool,
    data_dir: Path,
    work_dir: Path,
):
    dataset_cfg = NAME_TO_DATASET[dataset](data_dir)
    data_train, data_val = data_setups.load_dataset(dataset_cfg)
    print("train/val shape:", data_train.X.shape, data_val.X.shape)

    print("building bases...")
    bases = build_bases(data_train.X, q=q, m=m)
    n_dims = data_train.X.shape[1]
    parent = make_parent(n_dims, ttns_topology.lower())

    print("precomputing integral tensors...")
    gram_matrices = vmap(type(bases).l2_integral)(bases)
    basis_integrals = vmap(type(bases).integral)(bases)

    key = KEY(seed)
    key, key_init, key_data = jax.random.split(key, 3)
    print("initializing TTNS...")
    ttns = init_ttns_from_rank1(
        key=key_init,
        bases=bases,
        samples=data_train.X,
        parent=parent,
        rank=rank,
        noise=init_noise,
    )

    if normalize_every > 0:
        ttns, z0 = normalize_ttns_by_integral(ttns, basis_integrals, parent)
        print(f"init integral z={float(z0):.6e}")

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(ttns)

    run_dir = utils.suffix_with_date(
        work_dir / dataset / f"L2TTNS_q={q}_m={m}_rank={rank}_topology={ttns_topology.lower()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print("run_dir:", run_dir)

    train_iter = data_train.train_iterator(key_data, batch_sz)
    print_every = max(1, train_steps // 20)
    val_sz = min(val_sz, len(data_val.X))
    best_val_l2 = float("inf")
    best_step = 0
    bad_logs = 0
    logs_seen = 0
    best_ttns = ttns
    stopped_early = False
    stop_step = train_steps

    @jax.jit
    def train_step(curr_ttns, curr_opt_state, xs_batch, key_noise):
        noisy_batch = xs_batch + jax.random.normal(key_noise, xs_batch.shape) * train_noise
        loss, grads = value_and_grad(
            lambda x: l2_loss_on_batch(x, bases, noisy_batch, parent, gram_matrices)
        )(curr_ttns)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_ttns)
        next_ttns = optax.apply_updates(curr_ttns, updates)
        return next_ttns, next_opt_state, loss

    eval_l2 = jax.jit(lambda curr_ttns, xs: l2_loss_on_batch(curr_ttns, bases, xs, parent, gram_matrices))
    eval_integral = jax.jit(lambda curr_ttns: integral_q_ttns(curr_ttns, basis_integrals, parent))

    for step in range(train_steps):
        key, key_noise = jax.random.split(key, 2)
        batch = next(train_iter)
        ttns, opt_state, train_loss = train_step(ttns, opt_state, batch, key_noise)

        curr_z = None
        if normalize_every > 0 and ((step + 1) % normalize_every == 0):
            ttns, curr_z = normalize_ttns_by_integral(ttns, basis_integrals, parent)

        if (step + 1) % print_every == 0 or (step + 1) == train_steps:
            if curr_z is None:
                curr_z = eval_integral(ttns)

            val_loss = jnp.nan
            if val_sz > 0:
                val_loss = eval_l2(ttns, data_val.X[:val_sz])

            percent = 100.0 * (step + 1) / train_steps
            print(
                f"[{percent:6.2f}%] step={step + 1}/{train_steps} "
                f"train_l2={float(train_loss):.6f} val_l2={float(val_loss):.6f} "
                f"integral={float(curr_z):.6e}",
                flush=True,
            )

            if val_sz > 0 and bool(jnp.isfinite(val_loss)):
                logs_seen += 1
                val_loss_f = float(val_loss)
                improved = val_loss_f < (best_val_l2 - float(early_stop_min_delta))
                if improved:
                    best_val_l2 = val_loss_f
                    best_step = step + 1
                    best_ttns = ttns
                    bad_logs = 0
                elif logs_seen > int(early_stop_warmup_logs):
                    bad_logs += 1

                if int(early_stop_patience_logs) > 0 and bad_logs >= int(early_stop_patience_logs):
                    stopped_early = True
                    stop_step = step + 1
                    print(
                        f"early_stop at step={step + 1}: best_val_l2={best_val_l2:.6f} "
                        f"(step={best_step}), patience_logs={early_stop_patience_logs}, "
                        f"min_delta={early_stop_min_delta}",
                        flush=True,
                    )
                    break

    if early_stop_restore_best and val_sz > 0 and best_step > 0:
        ttns = best_ttns

    with open(run_dir / "ttns.msgpack", "wb") as f:
        f.write(flax.serialization.to_bytes(ttns))

    best_val_l2_for_summary = None
    if best_step > 0 and best_val_l2 < float("inf"):
        best_val_l2_for_summary = float(best_val_l2)

    summary = {
        "dataset": dataset,
        "q": q,
        "m": m,
        "rank": rank,
        "ttns_topology": ttns_topology.lower(),
        "batch_sz": batch_sz,
        "train_noise": train_noise,
        "lr": lr,
        "train_steps": train_steps,
        "normalize_every": normalize_every,
        "seed": seed,
        "stopped_early": bool(stopped_early),
        "stop_step": int(stop_step),
        "best_step": int(best_step),
        "best_val_l2": best_val_l2_for_summary,
        "early_stop_patience_logs": int(early_stop_patience_logs),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_warmup_logs": int(early_stop_warmup_logs),
        "early_stop_restore_best": bool(early_stop_restore_best),
        "final_integral": float(integral_q_ttns(ttns, basis_integrals, parent)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved:", run_dir / "ttns.msgpack")
    print("saved:", run_dir / "summary.json")


if __name__ == "__main__":
    main()
