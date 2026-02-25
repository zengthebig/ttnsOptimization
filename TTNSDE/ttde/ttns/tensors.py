"""
这个文件定义了两个“张量列车/张量链”相关的数据结构：

1) TT: Tensor Train（张量列车）——用一串三维 core 来表示一个高维张量
   - 每个 core 的形状是 (r_left, dim, r_right)
   - r_left, r_right 是“TT rank”（内部连接的秩）
   - dim 是这个维度的物理维度（例如每个变量离散取值个数）

2) TTOperator: TT 格式的线性算子（矩阵/算子）——用一串四维 core 来表示一个高维线性变换
   - 每个 core 的形状是 (r_left, dim_from, dim_to, r_right)
   - dim_from 是输入维度，dim_to 是输出维度

此外提供一些常用操作：
- 生成全 0 的 TT
- 随机生成 TT / TTOperator
- 把 TT / Operator “还原成完整张量/完整算子”（full_tensor / full_operator）
- TT 的 reverse / astype / 减法
- TT core 的转置（用于 reverse）
- 两个 TT 的减法（返回一个新的 TT，秩会增大）

注意：代码用 flax.struct.dataclass 让这个类是“不可变 pytree”，方便 JAX 的 jit/vmap/grad。
"""

from __future__ import annotations

from typing import Sequence, List

import jax
from jax import numpy as jnp
from flax import struct


@struct.dataclass
class TT:
    """
    Tensor Train (TT) 表示法。

    一个 n 维张量 A[i1, i2, ..., in] 被表示成 n 个 core 的连乘：
      core_k 的形状是 (r_{k}, dim_k, r_{k+1})
    其中：
      - dim_k 是第 k 维的大小（物理维度）
      - r_k 是 TT rank（内部连接维度）
      - 约定 r_0 = r_{n} = 1，这样整条链最终收缩成标量/张量元素

    这里 cores 存放的是一个 list，每个元素是 jnp.ndarray（三维）
    """

    @classmethod
    def zeros(cls, dims: Sequence[int], rs: Sequence[int]) -> TT:
        """
        构造一个全 0 的 TT。

        参数：
          dims: 每个维度的大小 [dim1, dim2, ..., dim_n]
          rs:   TT ranks（不包含两端的 1）[r1, r2, ..., r_{n-1}]
                注意长度必须是 n-1

        返回：
          TT 对象，其中每个 core 都是全 0 数组。
        """
        # TT 的标准约束：n 个 dims 对应 n-1 个内部 rank
        assert len(dims) == len(rs) + 1

        # 两端 rank 固定为 1：r0=1, rn=1
        rs = [1] + list(rs) + [1]

        # 逐个维度创建 core：形状 (r_left, dim, r_right)
        cores = [jnp.zeros((rs[i], dim, rs[i + 1])) for i, dim in enumerate(dims)]

        return cls(cores)

    @classmethod
    def generate_random(cls, key: jnp.ndarray, dims: Sequence[int], rs: Sequence[int]) -> TT:
        """
        随机生成一个 TT（每个 core 元素 ~ N(0,1)）。

        参数：
          key:  JAX 随机数 key
          dims: 每个维度大小
          rs:   内部 ranks（长度 n-1）

        返回：
          TT 对象，cores 为随机正态。
        """
        assert len(dims) == len(rs) + 1

        rs = [1] + list(rs) + [1]

        # 为每个 core 分配一个子 key，避免随机数重复
        keys = jax.random.split(key, len(dims))

        # 对每个维度 dim 生成 (r_left, dim, r_right) 的随机 core
        cores = [
            jax.random.normal(key, (rs[i], dim, rs[i + 1]))
            for i, (dim, key) in enumerate(zip(dims, keys))
        ]

        return cls(cores)

    # TT 的核心数据：n 个 core，每个 core 是 (r_left, dim, r_right)
    cores: List[jnp.ndarray]

    @property
    def n_dims(self):
        """返回张量的维数 n（也就是 core 的个数）。"""
        return len(self.cores)

    @property
    def full_tensor(self) -> jnp.ndarray:
        """
        把 TT 还原成“完整的高维张量”。

        实现方式：
          从第一个 core 开始，依次与后续 core 做 einsum 收缩 TT rank 维度。
          每次收缩掉上一个结果的右 rank，与下一个 core 的左 rank 对齐。

        结果形状：
          (dim1, dim2, ..., dim_n)
        """
        res = self.cores[0]  # (1, dim1, r2)
        for core in self.cores[1:]:
            # res:  (..., r)   core: (r, i, R)
            # -> (..., i, R)  把 r 收缩掉，拼接出新的物理维度 i
            res = jnp.einsum('...r,riR->...iR', res, core)

        # TT 两端 rank 都是 1，所以第一维和最后一维可以 squeeze 掉
        return jnp.squeeze(res, (0, -1))

    def reverse(self) -> TT:
        """
        把 TT 的维度顺序反过来（核心顺序翻转）。

        注意：
          翻转 core 的顺序后，每个 core 的左右 rank 方向也反了，
          所以需要对 core 做 transpose_core 来交换左右 rank 轴。
        """
        return TT([transpose_core(core) for core in self.cores[::-1]])

    def astype(self, dtype: jnp.dtype) -> TT:
        """把 TT 的所有 core 转成指定 dtype（例如 float32 / float64）。"""
        return TT([core.astype(dtype) for core in self.cores])

    def __sub__(self, other: TT):
        """定义 TT 的减法：self - other。"""
        return subtract(self, other)


@struct.dataclass
class TTOperator:
    """
    TT 格式的线性算子（你可以理解成高维矩阵/算子）。

    如果普通矩阵是 2D：A[out, in]
    那高维算子可以看成：A[i1..in, j1..jn]（输入 n 维 -> 输出 n 维）

    TT Operator 用 n 个 4D core 表示，每个 core 形状：
      (r_left, dim_from, dim_to, r_right)
    """

    @classmethod
    def generate_random(
        cls, key: jnp.ndarray, dims_from: Sequence[int], dims_to: Sequence[int], rs: Sequence[int]
    ) -> TTOperator:
        """
        随机生成一个 TT Operator（每个 core 元素 ~ N(0,1)）。

        参数：
          key:       JAX random key
          dims_from: 输入每维大小 [din1, din2, ..., din_n]
          dims_to:   输出每维大小 [dout1, dout2, ..., dout_n]
          rs:        内部 ranks（长度 n-1）

        返回：
          TTOperator 对象
        """
        n_dims = len(dims_from)

        # 基本一致性检查
        assert len(dims_from) == n_dims
        assert len(dims_to) == n_dims
        assert len(rs) + 1 == n_dims

        rs = [1] + list(rs) + [1]
        keys = jax.random.split(key, n_dims)

        # 每个 core 是 4D：(r_left, dim_from, dim_to, r_right)
        cores = [
            jax.random.normal(key, (rs[i], dim_from, dim_to, rs[i + 1]))
            for i, (dim_from, dim_to, key) in enumerate(zip(dims_from, dims_to, keys))
        ]

        return cls(cores)

    cores: List[jnp.ndarray]

    @property
    def full_operator(self) -> jnp.ndarray:
        """
        把 TT Operator 还原成完整算子（一个巨大的高维张量/矩阵）。

        逐 core einsum：
          res:  (..., r)
          core: (r, i, j, R)
          -> (..., i, j, R)

        最终 squeeze 掉两端 rank=1。
        结果形状：
          (din1, dout1, din2, dout2, ..., din_n, dout_n)
        （具体排列取决于 einsum 的写法，这里是按每个维度生成一对 (i,j)）
        """
        res = self.cores[0]
        for core in self.cores[1:]:
            res = jnp.einsum('...r,rijR->...ijR', res, core) # Einstein求和约定
        return jnp.squeeze(res, (0, -1)) 

    def reverse(self):
        """
        反转 operator 的 core 顺序。

        这里作者留了句注释：
          "idk, what should I do with axes 1 and 2."
        意思是：输入/输出物理轴 (dim_from, dim_to) 是否要交换、怎么交换，
        其实取决于你希望 reverse 后代表什么数学对象（是反转维度顺序？还是转置算子？）。

        当前实现：
          - core 顺序翻转
          - 把 rank 轴对调：把 axis 0 和 axis 3 互换
          - axis 1/2（输入/输出物理轴）保持不变
        """
        return TTOperator(
            [jnp.moveaxis(core, (0, 1, 2, 3), (3, 1, 2, 0)) for core in self.cores[::-1]]
        )


def transpose_core(core: jnp.ndarray) -> jnp.ndarray:
    """
    对 TT core 做“左右 rank 交换”。

    输入 core 形状：(r_left, dim, r_right)
    输出形状：(r_right, dim, r_left)

    用途：
      TT.reverse() 翻转 core 顺序时，需要把连接方向也翻过来。
    """
    return jnp.moveaxis(core, (0, 1, 2), (2, 1, 0))


def subtract(lhs: TT, rhs: TT) -> TT:
    """
    计算两个 TT 的差：lhs - rhs，并返回一个新的 TT。

    关键点（非常重要）：
    - 一般 TT 直接相减后，结果的 TT rank 会变大
    - 这里用的是经典的“block-diagonal 拼接”构造法：
        - 第一个 core：在右 rank 方向拼接 [lhs, -rhs]
        - 中间 core：做 2x2 的块对角拼接（lhs 在左上，rhs 在右下）
        - 最后一个 core：在左 rank 方向拼接 [lhs; rhs]

    这样保证：
      TT(full) = TT(lhs) - TT(rhs)

    代价：
      ranks 会变成原来的大约“相加”（更准确：中间 rank 变成 r1+r2）。
    """
    assert lhs.n_dims == rhs.n_dims

    # 只有 1 个维度时，TT 就是一个 (1, dim, 1) 的 core，直接相减即可
    if lhs.n_dims == 1:
        return TT([lhs.cores[0] - rhs.cores[0]])

    # 第一个 core：沿着右 rank 维（axis=-1）拼接
    # lhs: (1, d1, r1)  rhs: (1, d1, r1')
    # -> (1, d1, r1+r1')
    first = jnp.concatenate([lhs.cores[0], -rhs.cores[0]], axis=-1)

    # 最后一个 core：沿着左 rank 维（axis=0）拼接
    # lhs: (r_{n-1}, dn, 1)  rhs: (r'_{n-1}, dn, 1)
    # -> (r_{n-1}+r'_{n-1}, dn, 1)
    last = jnp.concatenate([lhs.cores[-1], rhs.cores[-1]], axis=0)

    # 中间 core：做块对角拼接（2x2 block）
    # 对每个位置 k：
    #   [ c1   0 ]
    #   [ 0   c2 ]
    inner = [
        jnp.concatenate(
            [
                # 上半块： [c1, 0]
                jnp.concatenate([c1, jnp.zeros((c1.shape[0], c1.shape[1], c2.shape[2]))], axis=-1),
                # 下半块： [0, c2]
                jnp.concatenate([jnp.zeros((c2.shape[0], c2.shape[1], c1.shape[2])), c2], axis=-1),
            ],
            axis=0,  # 沿左 rank 方向拼接成上下两块
        )
        for c1, c2 in zip(lhs.cores[1:-1], rhs.cores[1:-1])
    ]

    return TT([first] + inner + [last])
