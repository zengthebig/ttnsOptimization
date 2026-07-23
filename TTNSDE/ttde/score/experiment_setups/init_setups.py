from dataclasses import dataclass

import jax
from jax import numpy as jnp

from ttde.dl_routine import MutableModule
from ttde.score.experiment_setups import base


@dataclass
class CanonicalRankK(base.Base):
    em_steps: int
    noise: float

    def __call__(self, model: MutableModule, key: jnp.ndarray, samples: jnp.ndarray):
        init_key, canonical_key, noise_key = jax.random.split(key, 3)

        print('creating first params...')
        params = model.init(init_key)
        print('initializing canonical...')
        params = model.mutate(params, canonical_key, samples, n_steps=self.em_steps, method=model.init_canonical)
        print('adding noise...')
        params = model.mutate(params, noise_key, self.noise, method=model.add_noise)

        return params


@dataclass
class Rank1Only(base.Base):
    """仅 rank-1 初始化（不走 EM canonical），用于对照实验。

    链式拓扑下等价于 CanonicalRankK 的 rank=1 退化；非链拓扑下显式走
    ``init_rank_1``（rank-1 向量 → TTNS），跳过 EM。
    """
    noise: float

    def __call__(self, model: MutableModule, key: jnp.ndarray, samples: jnp.ndarray):
        init_key, rank1_key, noise_key = jax.random.split(key, 3)

        print('creating first params...')
        params = model.init(init_key)
        print('initializing rank-1 (no EM)...')
        params = model.mutate(params, rank1_key, samples, method=model.init_rank_1)
        print('adding noise...')
        params = model.mutate(params, noise_key, self.noise, method=model.add_noise)

        return params
