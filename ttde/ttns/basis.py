"""
这个文件在干什么（超白话版）：
--------------------------------
它用“分段多项式”的方式，在给定的 knots（节点）上构造 B-spline 基函数，
并且提供这些基函数的：
  - 函数值计算（evaluate）
  - 积分（∫）
  - 两两内积（L2 内积：∫ f(x) g(x) dx）
  - 截断到 x 的内积（∫_{-∞}^{x} f(x) g(x) dx 但这里只在支撑区间内算）

核心思路：
  1) 一个多项式段 PolyPiece：在区间 [l, r) 上是一个多项式，区间外为 0。
  2) 一个分段多项式 PiecewisePoly：由很多 PolyPiece 组成，整体是它们的和。
  3) build_spline_on_knots(knots)：对一段局部 knots 构造一个 B-spline（用 Cox–de Boor 递推）
  4) SplineOnKnots：对整组 knots，把每个起点的局部 knots 都做一遍 build_spline_on_knots，
     从而得到一批（batch）B-spline 基函数。

注意：
  - 这里用的是 JAX，所以大量使用 vmap 来向量化
  - flax.struct.dataclass 让对象成为 pytree，方便 jit/grad/vmap
  - q 是 spline 的阶数相关参数（这里的代码里 q = len(knots)-2）
"""

from __future__ import annotations

from flax import struct
from jax import numpy as jnp, vmap, lax

from ttde.utils import tree_stack
from ttde.utils.polynomial import poly_shift, poly_x, poly_definite_int, Polynomial  # , poly_int


def poly_const(const: float) -> jnp.ndarray:
    """
    返回常数多项式 p(x)=const 的系数表示。

    这里多项式系数采用“降幂”格式：
      [c] 表示 c
      [a, b] 表示 a*x + b
      [a, b, c] 表示 a*x^2 + b*x + c
    """
    return jnp.array([const])


@struct.dataclass
class PolyPiece: 
    """
    一个“分段多项式”的最小单位：在区间 [l, r) 上等于某个多项式 poly(x)，区间外为 0。

    字段：
      l, r: 该段的生效区间（左闭右开）
      poly: 多项式系数（降幂格式），用于 jnp.polyval
    """
    l: float
    r: float
    poly: Polynomial

    def __call__(self, x: float):
        """
        计算该段在 x 处的函数值：
          - 若 x ∈ [l, r)，返回 polyval(poly, x)
          - 否则返回 0
        """
        return jnp.where((self.l <= x) & (x < self.r), jnp.polyval(self.poly, x), 0.0)

    def integral(self) -> float:
        """
        返回该段在自身区间上的积分：
          ∫_{l}^{r} poly(x) dx
        """
        return poly_definite_int(self.poly, self.l, self.r)

    def integral_up_to(self, x: float) -> float:
        """
        返回该段从 l 积到 min(x, r) 的积分（并处理 x 在左侧的情况）：

          若 x <= l  -> 0
          若 l < x < r -> ∫_{l}^{x} poly(t) dt
          若 x >= r -> ∫_{l}^{r} poly(t) dt

        这里通过 up_to = min(x, r) 实现“截断到区间右端”。
        """
        up_to = jnp.minimum(x, self.r)
        return jnp.where(self.l < x, poly_definite_int(self.poly, self.l, up_to), 0.0)

    @classmethod
    def l2_of_product(cls, lhs: PolyPiece, rhs: PolyPiece) -> float:
        """
        计算两个 PolyPiece 的 L2 内积（其实就是重叠区间上乘积的积分）：

          ⟨lhs, rhs⟩ = ∫ lhs(x) * rhs(x) dx

        因为每段区间外都为 0，所以只有它们区间的交集会贡献：
          L = max(lhs.l, rhs.l)
          R = min(lhs.r, rhs.r)

        若 L < R：
          积分 ∫_{L}^{R} (lhs.poly(x)*rhs.poly(x)) dx
        否则交集为空，返回 0。
        """
        L = jnp.maximum(lhs.l, rhs.l)
        R = jnp.minimum(lhs.r, rhs.r)

        return jnp.where(L < R, poly_definite_int(jnp.polymul(lhs.poly, rhs.poly), L, R), 0.0)

    @classmethod
    def l2_up_to_of_product(cls, lhs: PolyPiece, rhs: PolyPiece, x: float) -> float:
        """
        计算“截断到 x”的 L2 内积（只积到 x 为止）：

          ⟨lhs, rhs⟩_up_to(x) = ∫_{-∞}^{x} lhs(t) rhs(t) dt

        但由于 lhs/rhs 只在各自区间内非零，等价于：
          在 [max(l_lhs, l_rhs), min(r_lhs, r_rhs, x)] 上积分乘积。

        若重叠且未被 x 截断到空，则返回乘积积分，否则 0。
        """
        L = jnp.maximum(lhs.l, rhs.l)
        R = jnp.minimum(lhs.r, jnp.minimum(rhs.r, x))

        return jnp.where(L < R, poly_definite_int(jnp.polymul(lhs.poly, rhs.poly), L, R), 0.0)


@struct.dataclass
class PiecewisePoly: # 这个类做一些分段多项式的运算. 
    """
    一个“分段多项式函数”：由很多 PolyPiece 组成，整体值是各段相加。

    pieces: 这里标注为 batch，意思是它其实是一批 PolyPiece（很多段），
            通常 shape 类似 (num_pieces, ...) 的 pytree 结构。
    """
    pieces: PolyPiece  # batch

    def __call__(self, x: float):
        """
        计算分段多项式在 x 的值：对每个 piece 求值然后求和。
        使用 vmap 把 PolyPiece.__call__ 向量化到所有 piece 上。
        """
        return vmap(PolyPiece.__call__, in_axes=(0, None))(self.pieces, x).sum()

    def integral(self):
        """
        计算整体积分：对每段 piece 在自身区间积分，然后求和。
        """
        return vmap(PolyPiece.integral)(self.pieces).sum()

    def integral_up_to(self, x: float) -> float:
        """
        计算整体从 -∞ 到 x 的积分（实际上只在各 piece 支撑上积分）：
          sum_i ∫_{piece_i.l}^{min(x, piece_i.r)} piece_i.poly(t) dt
        """
        return vmap(PolyPiece.integral_up_to, in_axes=(0, None))(self.pieces, x).sum()

    @classmethod
    def l2_of_product(cls, lhs: PiecewisePoly, rhs: PiecewisePoly):
        """
        计算两个 PiecewisePoly 的 L2 内积：
          ⟨lhs, rhs⟩ = ∫ lhs(x) rhs(x) dx

        因为 lhs 和 rhs 都是“很多段相加”，所以：
          lhs = Σ_i li(x), rhs = Σ_j rj(x)
          ⟨lhs, rhs⟩ = Σ_i Σ_j ⟨li, rj⟩

        这里用双层 vmap 构造一个矩阵 [i,j] = PolyPiece.l2_of_product(li, rj)，然后 sum。
        """
        l2_with_each_other_func = vmap(vmap(PolyPiece.l2_of_product, in_axes=(0, None)), in_axes=(None, 0))
        return l2_with_each_other_func(lhs.pieces, rhs.pieces).sum()

    @classmethod
    def l2_up_to(cls, lhs: PiecewisePoly, rhs: PiecewisePoly, x: float):
        """
        计算两个 PiecewisePoly 的“截断到 x”的 L2 内积：
          ⟨lhs, rhs⟩_up_to(x) = ∫_{-∞}^{x} lhs(t) rhs(t) dt
        同样展开成 Σ_i Σ_j 的 piece-wise 乘积积分（但每对都积到 x 截断）。
        """
        l2_up_to_with_each_other_func = vmap(
            vmap(PolyPiece.l2_up_to_of_product, in_axes=(0, None, None)),
            in_axes=(None, 0, None),
        )
        return l2_up_to_with_each_other_func(lhs.pieces, rhs.pieces, x).sum()


def build_spline_on_knots(knots: jnp.ndarray) -> PiecewisePoly: 
    """
    在一段局部 knots 上构造一个 B-spline 基函数（以分段多项式形式返回）。

    输入：
      knots: 一段局部结点，长度为 (q+2)
             这里 q = len(knots)-2，所以 knots 长度决定了 spline 的“阶数/次数递推深度”。

    输出：
      PiecewisePoly：由 (q+1) 段 PolyPiece 组成（对应每个区间 [k_j, k_{j+1})）。
    
    递推方式：
      先构造 k=0 的分段常数基函数（每个 i 对应一个区间是 1，其余是 0），
      然后用 Cox–de Boor 递推：
        B_{i,k}(x) = w_{i,k}(x)*B_{i,k-1}(x) + (1 - w_{i+1,k}(x))*B_{i+1,k-1}(x)
      其中 w_{i,k}(x) = (x - knots[i]) / (knots[i+k] - knots[i])
    """
    # 这里 q 决定递推的层数；knots 长度应为 q+2
    q = len(knots) - 2

    # 多项式 x
    x = poly_x()

    def w(i: int, k: int):
        """
        构造权重函数 w_{i,k}(x) = (x - knots[i]) / (knots[i+k] - knots[i])
        - poly_shift(x, knots[i]) 表示 (x - knots[i]) 的多项式系数
        - 再除以一个标量分母
        """
        return poly_shift(x, knots[i]) / (knots[i + k] - knots[i])

    # b_prev[i][j] 表示第 i 个基函数在第 j 段区间上的 PolyPiece（初始为 k=0）
    # k=0 的基函数就是“指示函数”：
    #   第 i 个基函数在区间 [knots[i], knots[i+1]) 上为 1，其他区间为 0
    b_prev = [
        [
            PolyPiece(knots[j], knots[j + 1], poly_const(1.0) if i == j else poly_const(0.0))
            for j in range(q + 1)
        ]
        for i in range(q + 1)
    ]

    # Cox–de Boor 递推：从 k=1 递推到 k=q
    for k in range(1, q + 1):
        b_next = []
        # 递推后基函数个数减少：q-k+1 个
        for i in range(q - k + 1):
            # left = B_{i,k-1} 的分段表示
            # right= B_{i+1,k-1} 的分段表示
            left, right = b_prev[i: i + 2]

            # w_left  = w_{i,k}(x)
            # w_right = 1 - w_{i+1,k}(x)
            w_left = w(i, k)
            w_right = jnp.polysub(poly_const(1.0), w(i + 1, k))

            # ith_b[piece] 保存 B_{i,k} 在第 piece 段区间上的多项式
            ith_b = []
            for piece in range(q + 1):
                # B_{i,k} = w_left * B_{i,k-1} + w_right * B_{i+1,k-1}
                # 因为都是多项式，所以是多项式乘法 + 加法
                poly = jnp.polyadd(
                    jnp.polymul(left[piece].poly, w_left),
                    jnp.polymul(right[piece].poly, w_right),
                )
                # 区间 l,r 沿用 left[piece]（它们的分段划分是一致的）
                ith_b.append(PolyPiece(left[piece].l, left[piece].r, poly))

            b_next.append(ith_b)

        b_prev = b_next

    # 最终只取 b_prev[0]：对应从这段局部 knots 构出来的“第一个”基函数的分段表达
    # tree_stack 用来把 list[PolyPiece] 变成 batch pytree（每个字段被 stack）
    return PiecewisePoly(tree_stack(b_prev[0]))


@struct.dataclass
class SplineOnKnots:
    """
    一个“整套 B-spline 基函数集合”。

    字段：
      splines: 一批 PiecewisePoly（batch），每一个是一个基函数
      knots:   全局 knots
      q:       阶数相关参数（这里作为静态字段，不参与 pytree 追踪；pytree_node=False）
    """
    splines: PiecewisePoly  # batch
    knots: jnp.ndarray
    q: int = struct.field(pytree_node=False)

    @classmethod
    def from_uniform_knots(cls, l: float, r: float, n: int, q: int) -> SplineOnKnots:
        """
        用均匀 knots 构造一套 spline 基函数。

        参数：
          l, r: 目标区间（数据大致覆盖范围）
          n:    基函数数量相关参数（最终 dim = n）
          q:    递推层数/阶数相关参数

        实现：
          先计算步长 h = (r-l)/(n-q)，然后把 knots 向左右各扩 q 个步长
          得到长度为 (n+q+1) 的 knots。
        """
        h = (r - l) / (n - q)
        return cls.from_knots(q, jnp.linspace(l - h * q, r + h * q, n + q + 1))

    @classmethod
    def from_knots(cls, q: int, knots: jnp.ndarray) -> SplineOnKnots:
        """
        从给定 knots 构造 spline 基函数集合。

        做法：
          - 对每个起点 s，取一段局部 knots: knots[s : s+q+2]
          - 对这段局部 knots 用 build_spline_on_knots 构造一个基函数（PiecewisePoly）
          - 所有起点一起 vmap，得到一批基函数

        这里的 batch_of_knots shape 大致是 (num_splines, q+2)。
        """
        # start_inds 是每个局部切片的起点索引（列向量形状 (B,1)）
        start_inds = jnp.arange(len(knots) - q - 1)[:, None]

        # dynamic_slice 需要 (start_indices, slice_sizes)
        # vmap 后得到每个起点的局部 knots（长度 q+2）
        batch_of_knots = vmap(lax.dynamic_slice, in_axes=(None, 0, None))(knots, start_inds, [q + 2])

        return SplineOnKnots(
            splines=vmap(build_spline_on_knots)(batch_of_knots),
            knots=knots,
            q=q,
        )

    @property
    def dim(self):
        """
        返回基函数个数（也就是 spline 空间的维度）。
        对 B-spline 来说，常见公式是：dim = len(knots) - q - 1
        """
        return len(self.knots) - self.q - 1

    @property
    def left_zero_bound(self):
        """基函数整体为 0 的最左边界（knots 最左端）。"""
        return self.knots[0]

    @property
    def right_zero_bound(self):
        """基函数整体为 0 的最右边界（knots 最右端）。"""
        return self.knots[-1]

    def __call__(self, x):
        """
        计算所有基函数在 x 的取值，返回一个向量 shape=(dim,)（或 batch 形状）。

        注意这里用了 abs：
          - 理论上 B-spline 非负
          - 由于多项式运算/浮点误差可能出现极小负值，所以取绝对值兜底
        """
        return jnp.abs(vmap(PiecewisePoly.__call__, in_axes=(0, None))(self.splines, x))

    def integral(self) -> jnp.ndarray:
        """
        返回每个基函数的积分值（对整个实数轴，但实际只在支撑区间非零）。
        输出 shape=(dim,)
        """
        return vmap(PiecewisePoly.integral)(self.splines)

    def integral_up_to(self, x: float) -> jnp.ndarray:
        """
        返回每个基函数从 -∞ 积到 x 的积分值（实际只在支撑区间内累积）。
        输出 shape=(dim,)
        """
        return vmap(PiecewisePoly.integral_up_to, in_axes=(0, None))(self.splines, x)

    def l2_integral(self) -> jnp.ndarray:
        """
        计算 Gram 矩阵（两两 L2 内积）：
          G[i,j] = ∫ spline_i(x) * spline_j(x) dx

        输出 shape=(dim, dim)
        """
        l2_with_each_other_func = vmap(vmap(PiecewisePoly.l2_of_product, in_axes=(0, None)), in_axes=(None, 0))
        return l2_with_each_other_func(self.splines, self.splines)

    def l2_up_to(self, x: float):
        """
        计算“截断到 x”的 Gram 矩阵：
          G_x[i,j] = ∫_{-∞}^{x} spline_i(t) * spline_j(t) dt
        输出 shape=(dim, dim)
        """
        return vmap(vmap(PiecewisePoly.l2_up_to, in_axes=(0, None, None)), in_axes=(None, 0, None))(
            self.splines, self.splines, x
        )


def create_space_uniform_knots(xs: jnp.ndarray, n: int, q: int):
    """
    根据数据 xs 的范围，生成一组“均匀 knots”。

    参数：
      xs: 数据点（用来估计区间范围）
      n:  期望的 spline 空间维度相关参数
      q:  spline 阶数相关参数（用于向左右扩展 q 个步长）

    返回：
      一个 knots 数组，长度为 (n+q+1)

    逻辑：
      l = xs.min(), r = xs.max()
      h = (r-l)/(n-q)
      knots = linspace(l - q*h, r + q*h, n+q+1)
    """
    l, r = xs.min(), xs.max()
    h = (r - l) / (n - q)
    return jnp.linspace(l - h * q, r + h * q, n + q + 1)
