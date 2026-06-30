"""从已拟合的单父 TTNS 密度中采样（条件逆 CDF）。

方案 A（max-plus 逐层传播）需要"从上一层拟合好的 TTNS 密度中抽样"，再经 max-plus
传播得到下一层样本。仓库原本只有"合成数据生成器"，没有"从 TTNS 密度采样"的能力，
本文件补上这一新组件。

核心思路（树结构条件采样，根→叶逐节点）：
- 单父 TTNS 表示密度 $q(x)=\langle T,\bigotimes_k b_k(x_k)\rangle$，关于每个物理腿线性。
- 按前序（父在子前）遍历节点 $t$：把已采样节点替换成 $b(x_j)$、未采样节点（除 $t$）
  替换成基积分 $\int b_k$（即积掉），节点 $t$ 的腿保持开放，收缩网络即得到 $t$ 的
  一维"未归一化条件密度系数" $c_t\in\mathbb{R}^{\dim}$，条件密度 $f_t(x)=\langle c_t,b_t(x)\rangle$。
- 在网格上算出 $f_t$，截断到非负、归一化成 CDF，逆 CDF 抽样得到 $x_t$。

说明：线性 TTNS 不保证处处非负，故采样时对条件密度做 clamp(>=0) 再归一化，这是
采样的实用近似（与 L2 拟合的密度可能略有偏差，但保证抽样合法）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import jax
import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[1]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from ttde.ttns.ttns_opt import TTNSOpt, batch_eval_rank1_ttns, _children_from_parent, _root_from_parent  # noqa: E402


def _preorder(parent: Sequence[int]) -> List[int]:
    """前序遍历（父节点先于子节点），用于条件采样顺序。"""
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)
    order: List[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        # 逆序压栈让遍历顺序更自然（不影响正确性）
        for c in reversed(children[node]):
            stack.append(c)
    return order


def _basis_eval_dim(bases, dim: int, xs: jnp.ndarray) -> jnp.ndarray:
    """对单个维度 `dim` 的基函数，在一批标量 `xs[K]` 上求值，返回 [K, basis_dim]。"""
    base_t = jax.tree_util.tree_map(lambda a: a[dim], bases)
    return vmap(type(base_t).__call__, in_axes=(None, 0))(base_t, xs)


def sample_ttns(
    ttns: TTNSOpt,
    bases,
    parent: Sequence[int],
    key: jnp.ndarray,
    n: int,
    grid_size: int = 400,
    eps: float = 1e-12,
    return_debug: bool = False,
):
    """从单父 TTNS 密度抽 `n` 个样本，返回 [n, n_dims]。

    bases: 按维度 batch 的 SplineOnKnots。
    grid_size: 每个一维条件密度逆 CDF 的网格分辨率。
    """
    n_dims = len(parent)
    basis_dim = int(ttns.cores[_root_from_parent(parent)].shape[1])

    basis_integrals = vmap(type(bases).integral)(bases)  # [n_dims, basis_dim]
    knots = np.asarray(bases.knots)  # [n_dims, n_knots]

    # 预备每个维度的网格与基函数取值矩阵 B[dim] = [grid_size, basis_dim]
    grids: List[np.ndarray] = []
    B_mats: List[jnp.ndarray] = []
    for d in range(n_dims):
        lo, hi = float(knots[d, 0]), float(knots[d, -1])
        g = jnp.linspace(lo, hi, grid_size)
        grids.append(np.asarray(g))
        B_mats.append(_basis_eval_dim(bases, d, g))  # [grid_size, basis_dim]

    # V[:, k, :]：每个节点当前用于收缩的向量；未采样默认 = 基积分（积掉）
    V = jnp.broadcast_to(basis_integrals[None, :, :], (n, n_dims, basis_dim))
    xs_out = np.zeros((n, n_dims), dtype=np.float64)

    eye = jnp.eye(basis_dim)

    def cond_coeffs(V_cur: jnp.ndarray, t: int) -> jnp.ndarray:
        """节点 t 的条件密度系数 c[n, basis_dim]：对每个基向量求一次网络收缩。"""
        def eval_with_onehot(onehot):  # onehot [basis_dim]
            Vt = V_cur.at[:, t, :].set(jnp.broadcast_to(onehot, (n, basis_dim)))
            return batch_eval_rank1_ttns(ttns, Vt, parent)  # [n]
        return vmap(eval_with_onehot)(eye).T  # [n, basis_dim]

    cond_coeffs_jit = jax.jit(cond_coeffs, static_argnums=(1,))
    debug = {"neg_mass_frac": {}, "cond_mean": {}}

    for t in _preorder(parent):
        key, k_t = jax.random.split(key)
        c = cond_coeffs_jit(V, t)  # [n, basis_dim]
        B_t = B_mats[t]  # [grid_size, basis_dim]
        f_raw = c @ B_t.T  # [n, grid_size] 未归一化条件密度
        f = jnp.clip(f_raw, a_min=0.0)
        if return_debug:
            pos = f.sum(axis=1)
            neg = jnp.clip(-f_raw, a_min=0.0).sum(axis=1)
            debug["neg_mass_frac"][t] = float(jnp.mean(neg / (pos + neg + eps)))
        mass = f.sum(axis=1, keepdims=True)
        # 若全被截断（mass≈0），退化为网格均匀分布
        f = jnp.where(mass < eps, jnp.ones_like(f), f)
        pmf = f / f.sum(axis=1, keepdims=True)
        cdf = jnp.cumsum(pmf, axis=1)  # [n, grid_size]

        grid_t = jnp.asarray(grids[t])
        if return_debug:
            debug["cond_mean"][t] = np.asarray((pmf * grid_t[None, :]).sum(axis=1))
        u = jax.random.uniform(k_t, (n,))
        # 逐样本逆 CDF + 桶内线性插值
        idx = vmap(lambda row, uu: jnp.searchsorted(row, uu))(cdf, u)
        idx = jnp.clip(idx, 1, grid_size - 1)
        c_lo = jnp.take_along_axis(cdf, (idx - 1)[:, None], axis=1)[:, 0]
        c_hi = jnp.take_along_axis(cdf, idx[:, None], axis=1)[:, 0]
        frac = jnp.where(c_hi > c_lo, (u - c_lo) / (c_hi - c_lo), 0.0)
        x_lo = grid_t[idx - 1]
        x_hi = grid_t[idx]
        x_t = x_lo + frac * (x_hi - x_lo)

        xs_out[:, t] = np.asarray(x_t)
        # 用采样值更新 V[:, t, :] = b_t(x_t)
        V = V.at[:, t, :].set(_basis_eval_dim(bases, t, jnp.asarray(x_t)))

    if return_debug:
        return jnp.asarray(xs_out), debug
    return jnp.asarray(xs_out)


def marginal_coeffs(ttns: TTNSOpt, bases, parent: Sequence[int], dim: int) -> jnp.ndarray:
    r"""单变量边缘密度系数 $c$：把除 `dim` 外所有变量积掉，留 `dim` 开放。

    边缘密度 $q_{dim}(x)=\langle c, b_{dim}(x)\rangle$。用于采样器正确性验证。
    """
    basis_integrals = vmap(type(bases).integral)(bases)  # [n_dims, basis_dim]
    basis_dim = basis_integrals.shape[1]
    V = basis_integrals[None, :, :]  # [1, n_dims, basis_dim]
    eye = jnp.eye(basis_dim)

    def eval_with_onehot(onehot):
        Vt = V.at[:, dim, :].set(onehot[None, :])
        return batch_eval_rank1_ttns(ttns, Vt, parent)[0]

    return vmap(eval_with_onehot)(eye)  # [basis_dim]
