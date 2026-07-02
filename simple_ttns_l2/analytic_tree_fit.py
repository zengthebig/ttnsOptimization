"""clarify.md 全解析链的核心：把上层 TTNS(P_X) **不采样**地拟合成下层树 TTNS(P_Y)。

思路(全解析，无采样)：
1. 用 Scheme B(`UpperForest`)解析算出 Y 层每个节点的 **marginal 密度** $p_v$、
   任意两节点的 **相关**(Hoeffding)与所选树边的 **两两联合密度** $p_{vu}$。
2. 用解析相关矩阵建 **Chow-Liu 最大生成树** → Y 层拓扑 `parent`。
3. 目标 = p_Y 的 **树投影**(tree graphical model)：
   $$p_{tree}(y)=p_{root}(y_{root})\prod_{v\ne root}p(y_v\mid y_{pa(v)}),$$
   其中条件核由解析 $p_{vu}/p_u$ 给出。树 TTNS 只能表达树结构，故树投影即最优目标
   (=Chow-Liu 的 KL 投影)。
4. **解析 L2 拟合**：$L(T)=\int q^2-2\,\mathbb{E}_{p_{tree}}[q]$。$\int q^2$ 用现成
   `quadratic_form_ttns`；交叉项 $\mathbb{E}_{p_{tree}}[q]$ 用 **树消息传递** 精确算
   (逐 grid 点复用 `_eval_rank1_local` 做局部收缩，保留父键开放)。整条链无采样。

与方案 A 的关键区别：A 采 Y 样本再拟合树 TTNS(样本噪声估边缘/边)；本法解析精确算
同样的 节点边缘+树边(无噪声)。二者同为树 TTNS，故本法应 ≥ A，且不采样。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jax
import numpy as np
import optax
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[1]
TTNSDE_ROOT = REPO_ROOT / "TTNSDE"
if str(TTNSDE_ROOT) not in sys.path:
    sys.path.insert(0, str(TTNSDE_ROOT))

from ttde.ttns.ttns_opt import (  # noqa: E402
    TTNSOpt,
    _children_from_parent,
    _root_from_parent,
    _eval_rank1_local,
    quadratic_form_ttns,
    batch_eval_rank1_ttns,
)

from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1  # noqa: E402
from simple_ttns_l2.objective import integral_q_ttns, normalize_ttns_by_integral  # noqa: E402
from simple_ttns_l2.ttns_sampler import _basis_eval_dim  # noqa: E402
from simple_ttns_l2.maxplus_pipeline import DelayParams, propagate_layer  # noqa: E402
from simple_ttns_l2.chow_liu import estimate_chow_liu_tree  # noqa: E402
from simple_ttns_l2.maxplus_cdf import moments_from_marginal, cov_hoeffding  # noqa: E402
from simple_ttns_l2.maxplus_cdf_forest import (  # noqa: E402
    UpperForest, _max_spanning_tree, _preorder_tree, block_joint_cdf,
)
from simple_ttns_l2.layered_forest import BlockModel, sample_forest  # noqa: E402


# ------------------------------------------------------------------ DAG 结构分块


def _ancestor_sources(spec) -> Dict[int, frozenset]:
    """每个节点的祖先源集合(无父节点=源)。按拓扑序累积。"""
    anc: Dict[int, frozenset] = {}
    for node in spec.topo_order:
        ps = spec.parents(node)
        if not ps:
            anc[node] = frozenset([node])
        else:
            s: set = set()
            for p in ps:
                s |= anc[p]
            anc[node] = frozenset(s)
    return anc


def structural_blocks(spec, li: int) -> List[List[int]]:
    """第 li 层按 **共享祖先源** 并查集分块(纯 DAG 结构, 无需数据/统计)。

    两节点共享任一祖先源 → 边际相关 → 同块。返回层内**局部索引**分组。
    """
    nodes = list(spec.layers[li])
    K = len(nodes)
    anc = _ancestor_sources(spec)
    par = list(range(K))

    def find(a: int) -> int:
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def union(a: int, b: int) -> None:
        par[find(a)] = find(b)

    src2idx: Dict[int, List[int]] = {}
    for i, v in enumerate(nodes):
        for s in anc[v]:
            src2idx.setdefault(s, []).append(i)
    for idxs in src2idx.values():
        for k in range(1, len(idxs)):
            union(idxs[0], idxs[k])
    comps: Dict[int, List[int]] = {}
    for i in range(K):
        comps.setdefault(find(i), []).append(i)
    return [sorted(c) for c in comps.values()]


def _pair_mi(p_vw: np.ndarray, p_v: np.ndarray, p_w: np.ndarray, s_grid: np.ndarray) -> float:
    """两两互信息 $\\iint p_{vw}\\log\\frac{p_{vw}}{p_v p_w}$(copula 信息量, 优于 |corr|)。"""
    outer = np.clip(p_v[:, None] * p_w[None, :], 1e-15, None)
    pj = np.clip(p_vw, 0.0, None)
    ratio = np.where(pj > 1e-15, pj / outer, 1.0)
    integrand = np.where(pj > 1e-15, pj * np.log(ratio), 0.0)
    return float(np.trapz(np.trapz(integrand, s_grid, axis=1), s_grid))


# ------------------------------------------------------------------ 解析统计量


def _marg_density(F_v: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """marginal CDF → 归一化密度(clip 负值 + 单位积分)。"""
    p = np.clip(np.gradient(F_v, s_grid), 0.0, None)
    Z = np.trapz(p, s_grid)
    return p / max(Z, 1e-12)


def _pair_density(F_vw: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """配对 CDF $F_{vw}(s,t)$ → 归一化联合密度 $\\partial^2 F/\\partial s\\partial t$。"""
    p = np.gradient(np.gradient(F_vw, s_grid, axis=0), s_grid, axis=1)
    p = np.clip(p, 0.0, None)
    Z = np.trapz(np.trapz(p, s_grid, axis=1), s_grid)
    return p / max(Z, 1e-12)


@dataclass
class AnalyticLayer:
    nodes: List[int]              # 全局 id，列序
    parent: List[int]            # Chow-Liu 树 parent（局部索引，根 parent=root/-1）
    s_grid: np.ndarray           # [G]
    p_marg: np.ndarray           # [K, G]
    corr: np.ndarray             # [K, K]
    pcond: Dict[int, np.ndarray]  # v(非根) -> [G_v, G_u] 条件核 p(y_v|y_pa)


def analytic_layer_target(
    upper: UpperForest, spec, li: int, params: DelayParams,
    s_max: float, n_s: int = 100, n_s_pair: int = 80,
) -> AnalyticLayer:
    """解析算出第 li 层的树目标：节点边缘密度 + Chow-Liu 树 + 树边条件核。"""
    cur = list(spec.layers[li])
    K = len(cur)
    parents = {v: list(spec.parents(v)) for v in cur}
    s_grid = np.linspace(0.0, s_max, n_s)

    F = {v: upper.marginal_cdf(parents[v], s_grid, params) for v in cur}
    p_marg = np.stack([_marg_density(F[v], s_grid) for v in cur], axis=0)  # [K, G]

    # 相关矩阵（配对 CDF 粗网格）→ 最大生成树
    if K == 1:
        return AnalyticLayer(cur, [-1], s_grid, p_marg, np.eye(1), {})
    sp = np.linspace(0.0, s_max, n_s_pair)
    Fp = {v: upper.marginal_cdf(parents[v], sp, params) for v in cur}
    var_p = {v: moments_from_marginal(Fp[v], sp)[1] for v in cur}
    corr = np.eye(K)
    for a in range(K):
        for b in range(a + 1, K):
            Fvw = upper.pair_cdf(parents[cur[a]], parents[cur[b]], sp, sp, params)
            cov = cov_hoeffding(Fvw, Fp[cur[a]], Fp[cur[b]], sp, sp)
            corr[a, b] = corr[b, a] = cov / np.sqrt(var_p[cur[a]] * var_p[cur[b]])

    tree_parent = _max_spanning_tree(np.abs(corr))  # 根=0, parent[0]=-1

    # 树边条件核：p(y_v | y_pa) = p_vu(y_v,y_pa) / (∫ p_vu dy_v)
    pcond: Dict[int, np.ndarray] = {}
    for v in range(K):
        u = tree_parent[v]
        if u == -1:
            continue
        Fvu = upper.pair_cdf(parents[cur[v]], parents[cur[u]], s_grid, s_grid, params)  # [G_v, G_u]
        p_vu = _pair_density(Fvu, s_grid)
        col = np.trapz(p_vu, s_grid, axis=0)  # [G_u] = 边际(近似 p_u)
        pcond[v] = p_vu / np.clip(col[None, :], 1e-12, None)
    return AnalyticLayer(cur, tree_parent, s_grid, p_marg, corr, pcond)


# ------------------------------------------------------------------ 解析 L2 拟合


def _build_layer_bases(s_grid: np.ndarray, K: int, q: int, m: int):
    """在层公共支撑 [0, s_max] 上建每维相同的 spline 基（无采样，仅用网格范围定 knots）。"""
    grid_tiled = jnp.asarray(np.tile(s_grid[:, None], (1, K)))
    return build_bases(grid_tiled, q, m)


def _cross_term_fn(parent: Sequence[int], Bg: List[jnp.ndarray],
                   pcond: Dict[int, jnp.ndarray], p_root: jnp.ndarray, delta: float):
    """返回 cross(cores) = E_{p_tree}[q]，树消息传递(可微)。

    Bg[v]:[G,m] 节点 v 基在网格上取值；pcond[v]:[G_v,G_u] 条件核；p_root:[G] 根边缘密度。
    """
    children = _children_from_parent(parent)
    root = _root_from_parent(parent)

    def node_Gv(core, Bg_v, child_msgs: List[jnp.ndarray]) -> jnp.ndarray:
        """Gv[rp, G] = Σ_i core[rp,i,·] Bg_v[g,i] Π_c child_msgs[c][g,·]，逐 g 复用局部收缩。"""
        if child_msgs:
            out = vmap(lambda vec, *cvs: _eval_rank1_local(core, vec, list(cvs)))(Bg_v, *child_msgs)
        else:
            out = vmap(lambda vec: _eval_rank1_local(core, vec, []))(Bg_v)
        return out.T  # [rp, G]

    def cross(cores) -> jnp.ndarray:
        msgs: Dict[int, jnp.ndarray] = {}
        # 后序
        order: List[int] = []
        stack = [(root, 0)]
        while stack:
            node, idx = stack.pop()
            ch = children[node]
            if idx < len(ch):
                stack.append((node, idx + 1))
                stack.append((ch[idx], 0))
            else:
                order.append(node)
        for node in order:
            cms = [msgs[c] for c in children[node]]  # each [G_node, rc]
            Gv = node_Gv(cores[node], Bg[node], cms)  # [rp, G_node]
            if node == root:
                return jnp.sum(Gv[0] * p_root * delta)
            # msg[g_u, rp] = Σ_{g_v} Gv[rp,g_v] pcond[node][g_v,g_u] * delta
            msgs[node] = (Gv @ pcond[node]).T * delta
        raise RuntimeError("root not reached")

    return cross


def _init_rank1(layer: AnalyticLayer, bases, gram: jnp.ndarray, Bg: List[jnp.ndarray],
                rank: int, key, noise: float) -> TTNSOpt:
    """按各节点解析边缘密度做 rank1 初始化(独立乘积 ≈ Π p_v)。"""
    K = len(layer.nodes)
    delta = float(layer.s_grid[1] - layer.s_grid[0])
    gram_np = np.asarray(gram)
    vecs = np.zeros((K, gram_np.shape[1]))
    for v in range(K):
        m_marg = np.asarray(Bg[v]).T @ layer.p_marg[v] * delta  # ∫ b_i p_v
        vecs[v] = np.linalg.solve(gram_np[v] + 1e-8 * np.eye(gram_np.shape[1]), m_marg)
    t0 = TTNSOpt.from_rank1_vectors(jnp.asarray(vecs), layer.parent, rank)
    keys = jax.random.split(key, len(t0.cores))
    cores = [c + jax.random.normal(k, c.shape) * noise for c, k in zip(t0.cores, keys)]
    return TTNSOpt(tuple(cores))


def analytic_block_target(
    upper: UpperForest, spec, gids: Sequence[int], params: DelayParams,
    s_max: float, n_s: int = 100, n_s_pair: int = 80, use_mi: bool = True,
) -> AnalyticLayer:
    """解析算出**一个结构块**(全局 id = `gids`)的树目标：边缘密度 + Chow-Liu 树 + 条件核。

    与 `analytic_layer_target` 相同，但只作用于给定节点子集(一个 DAG 结构块)，且树边权
    默认用 **解析 MI**(copula 信息量)替换 |corr|(`use_mi=True`)。
    """
    cur = list(gids)
    K = len(cur)
    parents = {v: list(spec.parents(v)) for v in cur}
    s_grid = np.linspace(0.0, s_max, n_s)

    F = {v: upper.marginal_cdf(parents[v], s_grid, params) for v in cur}
    p_marg = np.stack([_marg_density(F[v], s_grid) for v in cur], axis=0)  # [K, G]

    if K == 1:
        return AnalyticLayer(cur, [-1], s_grid, p_marg, np.eye(1), {})

    sp = np.linspace(0.0, s_max, n_s_pair)
    Fp = {v: upper.marginal_cdf(parents[v], sp, params) for v in cur}
    pmarg_p = {v: _marg_density(Fp[v], sp) for v in cur}
    var_p = {v: moments_from_marginal(Fp[v], sp)[1] for v in cur}
    W = np.eye(K)  # 树边权（MI 或 |corr|）
    for a in range(K):
        for b in range(a + 1, K):
            Fvw = upper.pair_cdf(parents[cur[a]], parents[cur[b]], sp, sp, params)
            if use_mi:
                p_vw = _pair_density(Fvw, sp)
                w = _pair_mi(p_vw, pmarg_p[cur[a]], pmarg_p[cur[b]], sp)
            else:
                cov = cov_hoeffding(Fvw, Fp[cur[a]], Fp[cur[b]], sp, sp)
                w = abs(cov) / np.sqrt(var_p[cur[a]] * var_p[cur[b]])
            W[a, b] = W[b, a] = w

    tree_parent = _max_spanning_tree(W)  # 根=0, parent[0]=-1

    pcond: Dict[int, np.ndarray] = {}
    for v in range(K):
        u = tree_parent[v]
        if u == -1:
            continue
        Fvu = upper.pair_cdf(parents[cur[v]], parents[cur[u]], s_grid, s_grid, params)  # [G_v, G_u]
        p_vu = _pair_density(Fvu, s_grid)
        col = np.trapz(p_vu, s_grid, axis=0)  # [G_u] = 边际(近似 p_u)
        pcond[v] = p_vu / np.clip(col[None, :], 1e-12, None)
    return AnalyticLayer(cur, tree_parent, s_grid, p_marg, W, pcond)


# ------------------------------------------------------------------ 完整联合目标（层间完整传播）


@dataclass
class AnalyticJointTarget:
    nodes: List[int]        # 全局 id，列序
    parent: List[int]       # Chow-Liu 树 parent（局部索引，q 结构，根 parent=-1）
    s_grid: np.ndarray      # [G_j] 联合网格（每维相同）
    p_marg: np.ndarray      # [K, G_j] 由联合边缘化得（供 rank1 初始化/根）
    p_joint: np.ndarray     # [G_j]*K 完整块联合密度（拟合目标）


def block_joint_density(F_joint: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """块完整联合 CDF → K 阶混合差分得联合密度（clip 负值 + 单位积分归一化）。"""
    K = F_joint.ndim
    p = F_joint
    for ax in range(K):
        p = np.gradient(p, s_grid, axis=ax)
    p = np.clip(p, 0.0, None)
    delta = float(s_grid[1] - s_grid[0])
    Z = p.sum() * (delta ** K)
    return p / max(Z, 1e-12)


def analytic_block_target_joint(
    upper: UpperForest, spec, gids: Sequence[int], params: DelayParams,
    s_max: float, n_s: int = 100, n_s_pair: int = 80, n_s_joint: int = 40,
    use_mi: bool = True,
) -> AnalyticJointTarget:
    """解析算出一个结构块的**完整联合目标**：q 树结构（解析 MI）+ 完整 K 维联合密度 $p_Y$。

    树结构复用 `analytic_block_target`（仅用于确定单父 TTNS 的拓扑）；目标密度改为
    `block_joint_cdf` → `block_joint_density` 的完整块联合（在 `n_s_joint` 网格上）。"""
    cur = list(gids)
    K = len(cur)
    base = analytic_block_target(upper, spec, cur, params, s_max,
                                 n_s=n_s, n_s_pair=n_s_pair, use_mi=use_mi)
    s_grid = np.linspace(0.0, s_max, n_s_joint)

    if K == 1:
        F = upper.marginal_cdf(list(spec.parents(cur[0])), s_grid, params)
        p1 = _marg_density(F, s_grid)
        return AnalyticJointTarget(cur, [-1], s_grid, p1[None, :], p1)

    parents_list = [list(spec.parents(v)) for v in cur]
    F_joint = block_joint_cdf(upper, parents_list, s_grid, params)  # [G_j]*K
    p_joint = block_joint_density(F_joint, s_grid)                  # [G_j]*K
    # 由联合边缘化得每维边缘（供 rank1 初始化/根）
    delta = float(s_grid[1] - s_grid[0])
    p_marg = np.zeros((K, len(s_grid)))
    for v in range(K):
        axes = tuple(a for a in range(K) if a != v)
        pv = p_joint.sum(axis=axes) * (delta ** (K - 1))
        p_marg[v] = pv / max(np.trapz(pv, s_grid), 1e-12)
    return AnalyticJointTarget(cur, list(base.parent), s_grid, p_marg, p_joint)


def _cross_term_fn_joint(parent: Sequence[int], Bg: List[jnp.ndarray],
                         p_joint: np.ndarray, delta: float):
    """返回 cross(cores) = E_{p_Y}[q] = Σ_grid p_Y·q·Δ^K（在完整 K 维网格上，可微）。"""
    K = len(parent)
    G, m = Bg[0].shape
    mesh = np.meshgrid(*[np.arange(G)] * K, indexing="ij")
    idx = [g.ravel() for g in mesh]  # each [G^K]
    V = np.zeros((G ** K, K, m))
    for v in range(K):
        V[:, v, :] = np.asarray(Bg[v])[idx[v]]
    Vj = jnp.asarray(V)
    p_flat = jnp.asarray(p_joint.ravel() * (delta ** K))
    par = list(parent)

    def cross(cores) -> jnp.ndarray:
        qv = batch_eval_rank1_ttns(TTNSOpt(tuple(cores)), Vj, par)  # [G^K]
        return jnp.sum(qv * p_flat)

    return cross


def _fit_analytic_ttns_joint(
    target: AnalyticJointTarget, key, q: int, m: int, rank: int,
    lr: float, steps: int, init_noise: float, log_every: int = 0, label: str = "",
) -> Tuple[TTNSOpt, object]:
    """对完整联合目标做解析 L2 拟合：min ∫q² - 2 E_{p_Y}[q]。返回 (归一化 ttns, bases)。"""
    K = len(target.nodes)
    bases = _build_layer_bases(target.s_grid, K, q, m)
    gram = vmap(type(bases).l2_integral)(bases)
    basis_int = vmap(type(bases).integral)(bases)
    Bg = [_basis_eval_dim(bases, v, jnp.asarray(target.s_grid)) for v in range(K)]
    delta = float(target.s_grid[1] - target.s_grid[0]) if len(target.s_grid) > 1 else 1.0

    cross = _cross_term_fn_joint(target.parent, Bg, target.p_joint, delta)

    k_init, key = jax.random.split(key)
    ttns = _init_rank1(target, bases, gram, Bg, rank, k_init, init_noise)
    ttns, _ = normalize_ttns_by_integral(ttns, basis_int, target.parent)

    optimizer = optax.adam(lr)
    cores = list(ttns.cores)
    opt_state = optimizer.init(cores)

    def loss_fn(cores):
        T = TTNSOpt(tuple(cores))
        int_q2 = quadratic_form_ttns(T, gram, target.parent)
        return int_q2 - 2.0 * cross(cores)

    @jax.jit
    def step_fn(cores, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(cores)
        updates, opt_state = optimizer.update(grads, opt_state, cores)
        cores = optax.apply_updates(cores, updates)
        return cores, opt_state, loss

    for st in range(steps):
        cores, opt_state, loss = step_fn(cores, opt_state)
        if log_every and ((st + 1) % log_every == 0 or st == 0):
            print(f"  [joint-fit {label}] step {st+1}/{steps} L2={float(loss):.6f}", flush=True)

    ttns = TTNSOpt(tuple(cores))
    ttns, _ = normalize_ttns_by_integral(ttns, basis_int, target.parent)
    return ttns, bases


def _fit_analytic_ttns(
    layer: AnalyticLayer, key, q: int, m: int, rank: int,
    lr: float, steps: int, init_noise: float, log_every: int = 0, label: str = "",
) -> Tuple[TTNSOpt, object]:
    """对给定 AnalyticLayer(树目标)做解析 L2 拟合，返回 (归一化 ttns, bases)。"""
    K = len(layer.nodes)
    bases = _build_layer_bases(layer.s_grid, K, q, m)
    gram = vmap(type(bases).l2_integral)(bases)         # [K, m, m]
    basis_int = vmap(type(bases).integral)(bases)       # [K, m]
    Bg = [_basis_eval_dim(bases, v, jnp.asarray(layer.s_grid)) for v in range(K)]  # [G,m]
    delta = float(layer.s_grid[1] - layer.s_grid[0]) if len(layer.s_grid) > 1 else 1.0

    pcond_j = {v: jnp.asarray(layer.pcond[v]) for v in layer.pcond}
    root = _root_from_parent(layer.parent)
    p_root = jnp.asarray(layer.p_marg[root])
    cross = _cross_term_fn(layer.parent, Bg, pcond_j, p_root, delta)

    k_init, key = jax.random.split(key)
    ttns = _init_rank1(layer, bases, gram, Bg, rank, k_init, init_noise)
    ttns, _ = normalize_ttns_by_integral(ttns, basis_int, layer.parent)

    optimizer = optax.adam(lr)
    cores = list(ttns.cores)
    opt_state = optimizer.init(cores)

    def loss_fn(cores):
        T = TTNSOpt(tuple(cores))
        int_q2 = quadratic_form_ttns(T, gram, layer.parent)
        return int_q2 - 2.0 * cross(cores)

    @jax.jit
    def step_fn(cores, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(cores)
        updates, opt_state = optimizer.update(grads, opt_state, cores)
        cores = optax.apply_updates(cores, updates)
        return cores, opt_state, loss

    for st in range(steps):
        cores, opt_state, loss = step_fn(cores, opt_state)
        if log_every and ((st + 1) % log_every == 0 or st == 0):
            print(f"  [analytic-fit {label}] step {st+1}/{steps} L2={float(loss):.6f}", flush=True)

    ttns = TTNSOpt(tuple(cores))
    ttns, _ = normalize_ttns_by_integral(ttns, basis_int, layer.parent)
    return ttns, bases


def fit_next_layer_tree(
    upper: UpperForest, spec, li: int, params: DelayParams, key,
    s_max: float, q: int = 2, m: int = 24, rank: int = 8,
    n_s: int = 100, n_s_pair: int = 80,
    lr: float = 3e-3, steps: int = 1500, init_noise: float = 0.01,
    log_every: int = 0,
) -> Tuple[TTNSOpt, List[int], object, AnalyticLayer]:
    """全解析拟合第 li 层的**单棵**树 TTNS(不分块)。返回 (ttns, parent, bases, AnalyticLayer)。"""
    layer = analytic_layer_target(upper, spec, li, params, s_max, n_s=n_s, n_s_pair=n_s_pair)
    ttns, bases = _fit_analytic_ttns(
        layer, key, q, m, rank, lr, steps, init_noise, log_every, label=f"L{li}"
    )
    return ttns, list(layer.parent), bases, layer


def fit_next_layer_forest(
    upper: UpperForest, spec, li: int, params: DelayParams, key,
    s_max: float, q: int = 2, m: int = 24, rank: int = 8,
    n_s: int = 100, n_s_pair: int = 80,
    lr: float = 3e-3, steps: int = 1500, init_noise: float = 0.01,
    log_every: int = 0, use_mi: bool = True,
) -> List[BlockModel]:
    """全解析拟合第 li 层为 **TTNS 森林**：按 DAG 结构(共享祖先)分块，每块解析拟合成一棵树。

    块间(不共祖先)边际独立 → 层密度 = 块密度乘积 → 各块独立拟合即精确。返回 BlockModel 列表。
    """
    layer_nodes = list(spec.layers[li])
    blocks = structural_blocks(spec, li)  # 层内局部索引分组
    forest: List[BlockModel] = []
    for bi, blk in enumerate(blocks):
        gids = [layer_nodes[i] for i in blk]
        target = analytic_block_target(
            upper, spec, gids, params, s_max, n_s=n_s, n_s_pair=n_s_pair, use_mi=use_mi
        )
        k_b, key = jax.random.split(key)
        ttns, bases = _fit_analytic_ttns(
            target, k_b, q, m, rank, lr, steps, init_noise, log_every, label=f"L{li}.b{bi}"
        )
        forest.append(BlockModel(
            tuple(blk), tuple(int(g) for g in gids), tuple(target.parent), ttns, bases
        ))
    return forest


# ------------------------------------------------------------------ 多层全解析链


def wrap_as_forest(ttns: TTNSOpt, parent: Sequence[int], bases, global_ids: Sequence[int]):
    """把单棵层 TTNS 包成单块森林(供 UpperForest 作为下层的上层模型)。"""
    from simple_ttns_l2.layered_forest import BlockModel
    K = len(global_ids)
    return [BlockModel(tuple(range(K)), tuple(int(g) for g in global_ids), tuple(parent), ttns, bases)]


def fit_analytic_chain(
    forest0, spec, params: DelayParams, key, s_max0: float,
    q: int = 2, m: int = 24, rank: int = 8,
    n_s: int = 100, n_s_pair: int = 80, lr: float = 3e-3, steps: int = 1500,
    init_noise: float = 0.01, log_every: int = 0, use_mi: bool = True,
) -> Dict[int, list]:
    """clarify.md 全解析链：L0(数据森林) → L1 → ... 逐层解析拟合 **TTNS 森林**，**全程无采样**。

    每层按 DAG 结构(共享祖先)分块，块内解析拟合成树；整层森林作为下一层的上层
    `UpperForest`。s_max 逐层按 max-plus 上界增长(每跳 +edge_hi+node_hi)。返回 {li: forest}。
    """
    forests: Dict[int, list] = {0: forest0}
    s_max = s_max0
    for li in range(1, len(spec.layers)):
        s_max = s_max + (params.edge_hi + params.node_hi) + 0.3
        upper = UpperForest(forests[li - 1], q_grid=400)
        k_l, key = jax.random.split(key)
        forests[li] = fit_next_layer_forest(
            upper, spec, li, params, k_l, s_max=s_max,
            q=q, m=m, rank=rank, n_s=n_s, n_s_pair=n_s_pair,
            lr=lr, steps=steps, init_noise=init_noise, log_every=log_every, use_mi=use_mi,
        )
    return forests


# ------------------------------------------------------------------ 采样交叉项链(允许 MC 噪声)


def fit_sampled_chain(forest0, spec, params: DelayParams, key, cfg: dict) -> Dict[int, list]:
    """采样求 L2 的分层链:结构与解析链一致(DAG 结构分块),但每块目标 = **完整联合的样本**。

    每层:采上层森林 + max-plus merge → 该层样本;按 `structural_blocks` 分块;块内 chow-liu 树;
    用 `train_tree_l2`(解析 ∫q² + batch 均值交叉项)拟合。交叉项 = 蒙特卡洛无偏,天然含全部相关。
    不修改 `fit_analytic_chain`。返回 {li: forest}。
    """
    from simple_ttns_l2.experiments.fit_diamond_dag_vs_tree import train_tree_l2

    forests: Dict[int, list] = {0: forest0}
    for li in range(1, len(spec.layers)):
        k_s, key = jax.random.split(key)
        s_upper = np.asarray(sample_forest(forests[li - 1], k_s, cfg["n_fit"], grid_size=400))
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
        s_layer = propagate_layer(spec, li, s_upper, params, rng)  # [n, layer_size]
        layer_nodes = list(spec.layers[li])
        blocks = structural_blocks(spec, li)
        forest: List[BlockModel] = []
        for bi, blk in enumerate(blocks):
            gids = [layer_nodes[i] for i in blk]
            Xb = jnp.asarray(s_layer[:, blk])
            if len(blk) == 1:
                parent = [0]
            else:
                parent = [int(p) for p in
                          estimate_chow_liu_tree(np.asarray(Xb), n_bins=16, root=0).parent]
            bases = build_bases(Xb, cfg["q"], cfg["m"])
            gram = vmap(type(bases).l2_integral)(bases)
            basis_int = vmap(type(bases).integral)(bases)
            split = int(0.85 * Xb.shape[0])
            tr, val = Xb[:split], Xb[split:]
            k_i, key = jax.random.split(key)
            t0 = init_ttns_from_rank1(k_i, bases, tr, parent, cfg["rank"], cfg["init_noise"])
            best, _ = train_tree_l2(
                t0, parent, bases, tr, val, gram, basis_int,
                key=k_i, lr=cfg["lr"], train_steps=cfg["steps"], batch_sz=cfg["batch_sz"],
                normalize_every=1, log_every=cfg["log_every"], label=f"samp_L{li}.b{bi}",
                train_noise=cfg["train_noise"], early_stop_patience=cfg["early_stop_patience"],
            )
            best, _ = normalize_ttns_by_integral(best, basis_int, parent)
            forest.append(BlockModel(tuple(blk), tuple(int(g) for g in gids),
                                     tuple(parent), best, bases))
        forests[li] = forest
    return forests
