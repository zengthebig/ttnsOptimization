import jax
from jax import vmap, numpy as jnp
from jax.scipy.special import logsumexp

from ttde.dl_routine import MutableModule, repeat
from ttde.score.models.continuous_canonical_init import continuous_rank_1, int_of_p, em
from ttde.score.models.discrete_canonical_init import fuse_canonical_probs_and_alphas
from ttde.tt.basis import SplineOnKnots
from ttde.tt.tt_opt import TTOpt
from ttde.ttns.ttns_opt import (
    TTNSOpt,
    normalized_inner_product_ttns,
    normalized_eval_rank1_ttns,
    normalized_quadratic_form_ttns,
)
from ttde.utils import index


def balanced_parent(n_dims: int) -> jnp.ndarray:
    parent = [-1] * n_dims
    parent[0] = 0
    for node in range(1, n_dims):
        parent[node] = (node - 1) // 2
    return jnp.array(parent)


def chain_parent(n_dims: int) -> jnp.ndarray:
    parent = [-1] * n_dims
    parent[0] = 0
    for node in range(1, n_dims):
        parent[node] = node - 1
    return jnp.array(parent)


def normalize_tree_parent(parent, n_dims: int) -> jnp.ndarray:
    parent_list = [int(p) for p in parent]
    if len(parent_list) != n_dims:
        raise ValueError(f"tree_parent length mismatch: expected {n_dims}, got {len(parent_list)}")

    roots = [node for node, p in enumerate(parent_list) if p == node or p == -1]
    if len(roots) != 1:
        raise ValueError(f"tree_parent must contain exactly one root, got {roots}")

    root = roots[0]
    parent_list[root] = root
    children = [[] for _ in range(n_dims)]

    for node, p in enumerate(parent_list):
        if node == root:
            continue
        if p < 0 or p >= n_dims:
            raise ValueError(f"invalid parent index parent[{node}]={p} for n_dims={n_dims}")
        if p == node:
            raise ValueError(f"only the root may satisfy parent[node] == node; got node={node}")
        children[p].append(node)

    visited = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(children[node])

    if len(visited) != n_dims:
        raise ValueError("tree_parent must define a connected acyclic tree")

    return jnp.asarray(parent_list, dtype=jnp.int32)


def canonical_to_chain_ttns(tt_opt: TTOpt) -> TTNSOpt:
    # Chain-TTNS convention: leaf core has no child axis.
    last = jnp.squeeze(tt_opt.last, axis=-1)
    return TTNSOpt((tt_opt.first, *tt_opt.inner, last))


class PAsTTNSOptBase(MutableModule):
    bases: SplineOnKnots = None
    permutations: jnp.ndarray = None
    tree_parent: jnp.ndarray = None
    rank: int = None
    l2_matrices: jnp.ndarray = None
    l2_matrices_permuted: jnp.ndarray = None

    @classmethod
    def create(
        cls,
        key: jnp.ndarray,
        bases: SplineOnKnots,
        n_components: int,
        rank: int,
        tree_topology: str = "chain",
        tree_parent=None,
    ):
        assert n_components >= 1
        n_dims = len(bases.knots)

        permutations = [jnp.arange(n_dims)]
        perm_keys = jax.random.split(key, n_components - 1)
        for curr_key in perm_keys:
            permutations.append(jax.random.permutation(curr_key, n_dims))

        if tree_parent is not None:
            tree_parent = normalize_tree_parent(tree_parent, n_dims)
        elif tree_topology == "balanced":
            tree_parent = balanced_parent(n_dims)
        elif tree_topology == "chain":
            tree_parent = chain_parent(n_dims)
        else:
            raise ValueError(f"unsupported tree_topology={tree_topology}")

        permutations = jnp.array(permutations)

        # Cache basis L2 matrices once. They are constant during training.
        l2_matrices = vmap(type(bases).l2_integral)(bases)
        l2_matrices_permuted = l2_matrices[permutations]

        return cls(
            bases=bases,
            permutations=permutations,
            tree_parent=tree_parent,
            rank=rank,
            l2_matrices=l2_matrices,
            l2_matrices_permuted=l2_matrices_permuted,
        )

    @property
    def n_components(self):
        return self.permutations.shape[0]

    @property
    def n_dims(self):
        return self.bases.knots.shape[0]

    def p(self, x):
        return jnp.exp(self.log_p(x))

    def log_p(self, x, eps=-jnp.inf):
        return self.unnormalized_log_p(x, eps) - self.log_int_p()

    def setup(self):
        dims = [int(index(self.bases)[i].dim) for i in range(self.n_dims)]
        self.ttns = self.variable(
            'ttns',
            'ttns',
            repeat(TTNSOpt.zeros, self.n_components),
            self.tree_parent.tolist(),
            dims,
            self.rank,
        )

    def change_ttns(self, ttns: TTNSOpt):
        self.ttns.value = ttns

    def __call__(self):
        pass

    def ttns_log_sqr_norm(self):
        parent = self.tree_parent.tolist()

        def one_norm(ttns):
            return normalized_inner_product_ttns(ttns, ttns, parent).log_norm

        return logsumexp(vmap(one_norm)(self.ttns.value))

    def tt_log_sqr_norm(self):
        # Keep the old metric name used by the trainer.
        return self.ttns_log_sqr_norm()

    def is_chain_topology(self) -> bool:
        expected = chain_parent(self.n_dims).tolist()
        actual = self.tree_parent.tolist()
        return actual == expected

    def init_components_from_rank1(self, rank1_vectors: jnp.ndarray):
        perms = self.permutations
        parent = self.tree_parent.tolist()
        rank = self.rank

        def build_one(perm):
            vectors = rank1_vectors[perm]
            return TTNSOpt.from_rank1_vectors(vectors, parent, rank)

        self.change_ttns(vmap(build_one)(perms))

    def init_rank_1(self, key: jnp.ndarray, samples: jnp.ndarray, noise: float = 1e-2):
        rank1 = continuous_rank_1(self.bases, samples, jnp.ones(len(samples)))
        if self.is_chain_topology():
            canonical = jnp.pad(rank1[None], [(0, self.rank - 1), (0, 0), (0, 0)])
            self.init_components_from_one_canonical(canonical)
        else:
            self.init_components_from_rank1(rank1)
        self.add_noise(key, noise)

    def init_components_from_one_canonical(self, canonical: jnp.ndarray):
        if not self.is_chain_topology():
            raise ValueError(
                "init_components_from_one_canonical is only valid for chain topology; "
                "use rank-1 initialization for non-chain topologies."
            )
        # canonical: [rank, n_dims, basis_dim]
        canonicals = canonical[:, self.permutations, :]
        canonicals = jnp.moveaxis(canonicals, 1, 0)  # [n_components, rank, n_dims, basis_dim]
        tt_opts = vmap(TTOpt.from_canonical)(canonicals)
        ttns = vmap(canonical_to_chain_ttns)(tt_opts)
        self.change_ttns(ttns)

    def init_canonical(self, key: jnp.ndarray, samples: jnp.ndarray, n_steps: int):
        if not self.is_chain_topology():
            # Canonical/EM path is derived from TT-chain parametrization.
            # For general trees, use rank-1 initialization to avoid invalid mappings.
            rank1 = continuous_rank_1(self.bases, samples, jnp.ones(len(samples)))
            self.init_components_from_rank1(rank1)
            return

        rank1_probs = continuous_rank_1(self.bases, samples, jnp.ones(len(samples)), 10)

        noise_level = 0.1
        repeated_probs = jnp.repeat(rank1_probs[None], self.rank, 0)
        noise_tensor = jax.random.uniform(key, repeated_probs.shape)
        noised_probs = repeated_probs * (1 - noise_level + noise_tensor * noise_level * 2)
        noised_probs /= vmap(vmap(int_of_p), in_axes=(0, None))(noised_probs, self.bases)[..., None]

        init_probs = noised_probs
        init_alphas = jnp.ones(self.rank) / self.rank
        probs, alphas = em(self.bases, init_probs, init_alphas, samples, n_steps)
        fused_probs = fuse_canonical_probs_and_alphas(probs, alphas)
        self.init_components_from_one_canonical(fused_probs)

    def add_noise(self, key: jnp.ndarray, noise: float):
        ttns = self.ttns.value
        keys = jax.random.split(key, len(ttns.cores))
        updated_cores = []
        for core, core_key in zip(ttns.cores, keys):
            updated_cores.append(core + jax.random.normal(core_key, core.shape) * noise)
        self.change_ttns(TTNSOpt(tuple(updated_cores)))


class PAsTTNSSqrOpt(PAsTTNSOptBase):
    def _single_component_ttns(self, component: int = 0) -> TTNSOpt:
        ttns = self.ttns.value
        return TTNSOpt(tuple(core[component] for core in ttns.cores))

    def fix_nonsqrt_init(self):
        ttns = self.ttns.value
        self.change_ttns(TTNSOpt(tuple(jnp.sqrt(core) for core in ttns.cores)))

    def init_rank_1(self, key: jnp.ndarray, samples: jnp.ndarray, noise: float = 1e-2):
        super().init_rank_1(key, samples, noise)
        self.fix_nonsqrt_init()

    def init_canonical(self, key: jnp.ndarray, samples: jnp.ndarray, n_steps: int):
        super().init_canonical(key, samples, n_steps)
        self.fix_nonsqrt_init()

    def unnormalized_log_p(self, x, eps=-jnp.inf):
        parent = self.tree_parent.tolist()
        bs = vmap(type(self.bases).__call__)(self.bases, x)

        if self.n_components == 1:
            vectors = bs[self.permutations[0]]
            normalized = normalized_eval_rank1_ttns(self._single_component_ttns(0), vectors, parent)
            return jnp.where(normalized.log_norm == -jnp.inf, eps, 2 * normalized.log_norm)

        def one_log_p(curr_bs, perm, ttns):
            vectors = curr_bs[perm]
            normalized = normalized_eval_rank1_ttns(ttns, vectors, parent)
            return jnp.where(normalized.log_norm == -jnp.inf, eps, 2 * normalized.log_norm)

        log_ps = vmap(one_log_p, in_axes=(None, 0, 0))(bs, self.permutations, self.ttns.value)
        return logsumexp(log_ps)

    def log_int_p(self):
        parent = self.tree_parent.tolist()

        if self.n_components == 1:
            normalized = normalized_quadratic_form_ttns(
                self._single_component_ttns(0),
                self.l2_matrices_permuted[0],
                parent,
            )
            return normalized.log_norm

        def one_log_int_p(matrices, ttns):
            normalized = normalized_quadratic_form_ttns(ttns, matrices, parent)
            return normalized.log_norm

        log_int_ps = vmap(one_log_int_p, in_axes=(0, 0))(self.l2_matrices_permuted, self.ttns.value)
        return logsumexp(log_int_ps)
