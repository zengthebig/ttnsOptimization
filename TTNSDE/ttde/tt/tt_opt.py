import jax
from flax import struct
from jax import numpy as jnp, vmap

from ttde.tt.tensors import TT, TTOperator
from ttde.utils import cached_einsum


@struct.dataclass
class TTOpt:
    """
    TT（Tensor-Train）张量的“优化态”存储形式。

    约定：
      - first : 第一个 TT core，形状 [1, dim, rank]
      - inner : 中间 TT cores 堆叠在一起，形状 [n_dims-2, rank, dim, rank]
      - last  : 最后一个 TT core，形状 [rank, dim, 1]

    这样做的目的：
      - 便于 JAX/Flax 在训练时把参数打包成一个 pytree（dataclass）。
      - 把中间 cores 用 stack 堆叠起来，方便用 vmap/scan 批处理。
    """
    first: jnp.ndarray
    inner: jnp.ndarray
    last: jnp.ndarray

    @classmethod
    def zeros(cls, n_dims: int, dim: int, rank: int):
        """
        构造全 0 的 TTOpt。

        参数：
          n_dims : 张量维数（TT 的 core 数）
          dim    : 每一维的离散维度大小
          rank   : TT-rank（这里假设所有中间 rank 相同）

        返回：
          TTOpt，其中：
            first shape = [1, dim, rank]
            inner shape = [n_dims-2, rank, dim, rank]
            last  shape = [rank, dim, 1]
        """
        return TTOpt(
            jnp.zeros([1, dim, rank]),
            jnp.zeros([n_dims - 2, rank, dim, rank]),
            jnp.zeros([rank, dim, 1]),
        )

    @classmethod
    def from_tt(cls, tt: TT):
        """
        从普通 TT（一般是 cores 列表形式）转换为 TTOpt（first/inner/last 分块形式）。

        tt.cores 通常是一个 Python list：
          [G0, G1, ..., G_{d-1}]
        其中 G0 shape [1, dim, r1]，Gi shape [r_i, dim, r_{i+1}]，最后 G_{d-1} shape [r_{d-1}, dim, 1]

        这里将中间的 cores (1:-1) 堆叠成一个 4D 数组 inner：
          inner shape = [d-2, rank, dim, rank]（假设 rank 统一）
        """
        return cls(
            tt.cores[0],                       # 第一个 core
            jnp.stack(tt.cores[1:-1], axis=0), # 中间 cores stack 起来
            tt.cores[-1],                      # 最后一个 core
        )

    @classmethod
    def rank_1_from_vectors(cls, vectors: jnp.ndarray):
        """
        把“逐维向量外积”的 rank-1 张量写成 TT 形式（TT-rank = 1）。

        输入：
          vectors shape = [N_DIMS, DIM]
            vectors[k] 是第 k 个维度上的向量 v^(k) (长度 DIM)

        该 rank-1 张量对应：
          T[i1,...,id] = Π_k vectors[k, i_k]

        输出 TTOpt：
          - first = vectors[0] 变形成 [1, DIM, 1]
          - inner = vectors[1:-1] 变形并堆叠成 [N_DIMS-2, 1, DIM, 1]
          - last = vectors[-1] 变形成 [1, DIM, 1]
        """
        return cls(
            vectors[0, None, :, None],        # [DIM] -> [1, DIM, 1]
            vectors[1:-1, None, :, None],     # [N-2, DIM] -> [N-2, 1, DIM, 1]
            vectors[-1, None, :, None],       # [DIM] -> [1, DIM, 1]
        )

    @classmethod
    def from_canonical(cls, vectors: jnp.ndarray):
        """
        把 CP / canonical（“外积和”）形式转换成 TT 形式。

        输入：
          vectors shape = [RANK, N_DIMS, DIM]
            vectors[r, k, :] 是第 r 个 CP 分量在第 k 个维度的向量 v^(r,k)

        该 canonical 张量对应：
          T[i1,...,id] = Σ_{r=1..R} Π_{k=1..d} vectors[r, k, i_k]

        构造思路：
          用 TT 的“链路索引”来表示 CP 分量 r（即 TT-rank = R）。
          中间 cores 做成“对角传递 r”的结构：
            inner[k, r, :, r] = vectors[r, k+1, :]
            inner[k, r, :, s] = 0 (s != r)
          这样 contraction 时强迫所有维度共享同一个 r，最终实现 Σ_r Π_k。

        输出：
          first shape = [1, DIM, R]
          inner shape = [N_DIMS-2, R, DIM, R]（每个中间 core 在 rank 维度上是对角的）
          last shape  = [R, DIM, 1]
        """
        # vectors[:, 0, :] 形状 [R, DIM]
        # 加 None -> [R, DIM, 1]
        # .T 对 3D 相当于把轴反转 (2,1,0)，得到 [1, DIM, R]
        first = vectors[:, 0, :, None].T  # [R, DIM, 1] -> [1, DIM, R]

        # 先创建全零的 inner： [N_DIMS-2, R, DIM, R]
        inner = jnp.zeros([vectors.shape[1] - 2, vectors.shape[0], vectors.shape[2], vectors.shape[0]])

        # 在每个中间 core 上把 (r,r) 的对角位置写入对应向量：
        # inner[k, r, :, r] = vectors[r, k+1, :]
        # 注意 vectors[:, 1:-1, :] 形状是 [R, N_DIMS-2, DIM]
        inner = inner.at[:, jnp.arange(vectors.shape[0]), :, jnp.arange(vectors.shape[0])].set(
            vectors[:, 1:-1, :]
        )

        # 最后一维：vectors[:, -1, :] 形状 [R, DIM]，加 None -> [R, DIM, 1]
        last = vectors[:, -1, :, None]

        return cls(first, inner, last)

    @property
    def n_dims(self) -> int:
        """
        TT 的维数（core 数量）。
        inner 的第 0 维长度是 (n_dims-2)，所以总数是 2 + inner.shape[0]。
        """
        return 2 + self.inner.shape[0]

    def to_nonopt_tt(self):
        """
        把 TTOpt 转回普通 TT（cores 列表形式）。

        关键点：self.inner 是一个 4D 数组 [n_dims-2, rank, dim, rank]，
        `*self.inner` 会沿第 0 维迭代，依次取出每个 3D core：
          self.inner[0], self.inner[1], ..., self.inner[-1]
        从而组成 TT 需要的 cores 列表：
          [first, inner0, inner1, ..., last]
        """
        return TT([self.first, *self.inner, self.last])

    def abs(self) -> 'TTOpt':
        """
        对 TT 的每个 core 做逐元素绝对值。
        常用于需要强制非负（例如密度模型）但又不想改训练流程时的一个操作。
        """
        return TTOpt(jnp.abs(self.first), jnp.abs(self.inner), jnp.abs(self.last))


@struct.dataclass
class TTOperatorOpt:
    """
    TT 形式的线性算子（TTOperator）的“优化态”存储。
    和 TTOpt 类似，但每个 core 多了一个“输入/输出”维度（矩阵而不是向量）。

    典型形状（rank 统一时）：
      - first : [1, dim_out, dim_in, rank]
      - inner : [n_dims-2, rank, dim_out, dim_in, rank]
      - last  : [rank, dim_out, dim_in, 1]

    这里用的是你代码里的约定（从 tt.cores 直接 stack）。
    """
    first: jnp.ndarray
    inner: jnp.ndarray
    last: jnp.ndarray

    @classmethod
    def from_tt_operator(cls, tt: TTOperator):
        """
        从普通 TTOperator（cores 列表）转换成 TTOperatorOpt（first/inner/last）。
        """
        return cls(tt.cores[0], jnp.stack(tt.cores[1:-1], axis=0), tt.cores[-1])

    @classmethod
    def rank_1_from_matrices(cls, matrices: jnp.ndarray):
        """
        构造 rank-1 的 TT-operator（所有 TT-rank=1）。

        输入：
          matrices shape = [N_DIMS, DIM, DIM]
            matrices[k] 是第 k 维上的一个线性变换矩阵 A^(k)

        对应整体算子是 Kronecker 积：
          A = A^(1) ⊗ A^(2) ⊗ ... ⊗ A^(d)

        输出：
          first shape = [1, DIM, DIM, 1]
          inner shape = [N_DIMS-2, 1, DIM, DIM, 1]
          last  shape = [1, DIM, DIM, 1]
        """
        return cls(
            matrices[0, None, :, :, None],
            matrices[1:-1, None, :, :, None],
            matrices[-1, None, :, :, None],
        )


@struct.dataclass
class NormalizedValue:
    """
    用“归一化 + log_norm”来表示一个向量/矩阵的数值，
    主要目的是：在连续 contraction 时避免数值爆炸/下溢。

    约定：
      value    : 归一化后的值（范数为 1，或在零向量时做特殊处理）
      log_norm : 原始范数的对数（这里存的是 log(||x||)）
    """
    value: jnp.ndarray
    log_norm: float

    @classmethod
    def from_value(cls, value):
        """
        给定一个 value，计算它的 2-范数并归一化，同时记录 log_norm。

        特殊处理：
          如果 value 全 0，则范数为 0，会导致除零。
          此时设置：
            log_norm = -inf
            value 归一化时用 updated_sqr_norm=1 做占位（避免 NaN）
        """
        sqr_norm = (value ** 2).sum()            # ||value||^2
        norm_is_zero = sqr_norm == 0
        updated_sqr_norm = jnp.where(norm_is_zero, 1., sqr_norm)

        return cls(
            # log(||value||) = 0.5 * log(||value||^2)
            log_norm=jnp.where(norm_is_zero, -jnp.inf, .5 * jnp.log(updated_sqr_norm)),
            value=value / jnp.sqrt(updated_sqr_norm),
        )


def normalized_inner_product(tt1: TTOpt, tt2: TTOpt):
    """
    计算两个 TT 的内积 <tt1, tt2>，但以“逐步归一化”的方式做 contraction，
    返回 NormalizedValue(value, log_norm)：
      - value    : 最终 contraction 得到的 1x1 矩阵（或标量）经过归一化的值
      - log_norm : 累积的对数范数（用于恢复真实尺度）

    关键点：
      - contraction 用 cached_einsum，减少重复编译/提升性能
      - 用 jax.lax.scan 在中间 cores 上循环（JIT 友好）
    """

    def body(state: NormalizedValue, cores):
        """
        扫描一步：把当前的“环境矩阵”state.value 与下一对 cores (G1, G2) 收缩。

        这里的 einsum:
          cached_einsum('ij,ikl,jkn->ln', state, G1, G2)
        可以理解为：
          - state.value: [r_left1, r_left2]（两条 TT 的 rank 环境）
          - G1:         [r_left1, dim, r_right1]
          - G2:         [r_left2, dim, r_right2]
          对 dim 做求和，更新到新的右侧 rank 环境 [r_right1, r_right2]
        """
        G1, G2 = cores
        contracted = NormalizedValue.from_value(
            cached_einsum('ij,ikl,jkn->ln', state.value, G1, G2)
        )

        # 把本步 contraction 的 log_norm 累加到总 log_norm 上
        return (
            NormalizedValue(
                value=contracted.value,
                log_norm=jnp.where(
                    state.log_norm == -jnp.inf,      # 如果之前已经是零范数，则一直保持 -inf
                    -jnp.inf,
                    state.log_norm + contracted.log_norm
                ),
            ),
            None,
        )

    # 先收缩 first cores：
    # cached_einsum('ikl,jkn->ln', G1_first, G2_first)
    # 结果是一个 rank 环境矩阵 [r1, r2]
    state = NormalizedValue.from_value(
        cached_einsum('ikl,jkn->ln', tt1.first, tt2.first)
    )

    # 对中间 cores 扫描（pairwise contraction）
    state, _ = jax.lax.scan(body, state, (tt1.inner, tt2.inner))

    # 最后收缩 last cores（单步）
    state, _ = body(state, (tt1.last, tt2.last))

    return state


def normalized_dot_operator(tt: TTOpt, tt_op: TTOperatorOpt):
    """
    计算 TT 张量 tt 被 TT-operator tt_op 作用后的结果，返回仍为 TTOpt。

    数学上对应：
      y = (tt_op) * tt
    其中 tt_op 通常表示 Kronecker 结构的线性算子。

    实现方式：
      对每个维度的 core 做局部 contraction：
        新 core = old core 与 operator core 在物理维度上相乘/收缩
      并把 (rank, operator_rank) 两套 rank 合并（reshape）成新的 TT-rank。
    """

    def body(x, A):
        """
        单个维度上的 core 乘法（把 operator 作用到该维的 TT core 上）。

        这里假设：
          x: TT core，形状 [r, m, s]
          A: operator core，形状 [t, m, n, u]
             （通常 m 是输入物理维度，n 是输出物理维度）

        einsum 'rms,tmnu->rtnsu'：
          - 对 m 做求和（输入物理维度被 operator 消去）
          - 输出得到形状 [r, t, n, s, u]
        然后 reshape，把 (r,t) 合并成新左 rank，把 (s,u) 合并成新右 rank：
          [r*t, n, s*u]
        这仍是一个合法的 TT core。
        """
        c = jnp.einsum('rms,tmnu->rtnsu', x, A)
        return c.reshape(c.shape[0] * c.shape[1], c.shape[2], c.shape[3] * c.shape[4])

    return TTOpt(
        body(tt.first, tt_op.first),      # 第一维
        vmap(body)(tt.inner, tt_op.inner),# 中间维度批量处理（逐 core）
        body(tt.last, tt_op.last),        # 最后一维
    )
