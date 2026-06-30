"""采样器正确性验证。

两类检查：
1. 解析边缘 CDF vs 抽样经验 CDF：直接验证采样器忠实于 TTNS 模型自身的边缘分布
   （与拟合质量无关，是采样器正确性的硬性检查）。
2. 数据回环：把 TTNS 拟合到强相关数据后采样，检查样本相关性符号/量级与数据一致，
   验证条件采样确实抓住了变量间依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "TTNSDE")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

jax.config.update("jax_enable_x64", True)

from ttde.ttns.ttns_opt import batch_eval_rank1_ttns  # noqa: E402
from simple_ttns_l2.train_l2 import build_bases, init_ttns_from_rank1, make_parent  # noqa: E402
from simple_ttns_l2.experiments.fit_diamond_dag_vs_tree import train_tree_l2  # noqa: E402
from simple_ttns_l2.ttns_sampler import sample_ttns, marginal_coeffs, _basis_eval_dim  # noqa: E402


def correlated_chain(key, n):
    """x0~U(0,1); x1=clip(x0+noise); x2=clip(x1+noise) —— 强正相关链。"""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    x0 = rng.uniform(0.1, 0.9, n)
    x1 = np.clip(x0 + 0.1 * rng.standard_normal(n), 0.02, 0.98)
    x2 = np.clip(x1 + 0.1 * rng.standard_normal(n), 0.02, 0.98)
    return jnp.asarray(np.stack([x0, x1, x2], axis=1))


def analytic_marginal_cdf(ttns, bases, parent, dim, xs_query):
    """模型边缘 CDF：F(x)=∑_i c_i ∫_{-∞}^x b_i / ∑_i c_i ∫ b_i。"""
    c = np.asarray(marginal_coeffs(ttns, bases, parent, dim))
    base_d = jax.tree_util.tree_map(lambda a: a[dim], bases)
    integ = np.asarray(base_d.integral())  # [basis_dim]
    Z = float(c @ integ)
    cdf_vals = []
    for x in xs_query:
        up = np.asarray(base_d.integral_up_to(float(x)))  # [basis_dim]
        cdf_vals.append(float(c @ up) / Z)
    return np.asarray(cdf_vals)


def empirical_cdf(samples_1d, xs_query):
    s = np.sort(np.asarray(samples_1d))
    return np.searchsorted(s, xs_query, side="right") / len(s)


def model_moment_vectors(bases, n_dims, grid_size=600):
    """网格求积得到每维的零/一/二阶矩向量 I,M,S：∫ x^k b_i(x) dx。"""
    knots = np.asarray(bases.knots)
    I_list, M_list, S_list = [], [], []
    for d in range(n_dims):
        lo, hi = float(knots[d, 0]), float(knots[d, -1])
        g = jnp.linspace(lo, hi, grid_size)
        dx = float((hi - lo) / (grid_size - 1))
        B = np.asarray(_basis_eval_dim(bases, d, g))  # [grid_size, basis_dim]
        gg = np.asarray(g)
        I_list.append(B.sum(0) * dx)
        M_list.append((gg[:, None] * B).sum(0) * dx)
        S_list.append((gg[:, None] ** 2 * B).sum(0) * dx)
    return np.stack(I_list), np.stack(M_list), np.stack(S_list)


def model_corr(ttns, bases, parent, n_dims):
    """用模型解析矩算出模型自身的相关矩阵（隔离拟合质量，纯验证采样器）。"""
    I, M, S = model_moment_vectors(bases, n_dims)

    def eval_with(vecs):
        V = jnp.asarray(vecs)[None, :, :]  # [1, n_dims, basis_dim]
        return float(batch_eval_rank1_ttns(ttns, V, parent)[0])

    base = I.copy()  # 全部积掉 = ∫q ≈ 1（与 M,S 同网格求积，保持一致）
    print(f"  [debug] 网格求积 ∫q ≈ {eval_with(base):.4f}")
    Ex = np.array([eval_with(_set(base, d, M[d])) for d in range(n_dims)])
    Ex2 = np.array([eval_with(_set(base, d, S[d])) for d in range(n_dims)])
    var = Ex2 - Ex ** 2
    print(f"  [debug] 模型 E[x]={Ex}  模型 std={np.sqrt(np.clip(var,0,None))}")
    corr = np.eye(n_dims)
    for a in range(n_dims):
        for b in range(a + 1, n_dims):
            v = base.copy()
            v[a] = M[a]
            v[b] = M[b]
            Exy = eval_with(v)
            cov = Exy - Ex[a] * Ex[b]
            corr[a, b] = corr[b, a] = cov / np.sqrt(max(var[a] * var[b], 1e-18))
    return corr


def _set(base, d, vec):
    v = base.copy()
    v[d] = vec
    return v


def main():
    key = jax.random.PRNGKey(0)
    k_tr, k_val, k_init, k_samp = jax.random.split(key, 4)

    train_x = correlated_chain(k_tr, 20000)
    val_x = correlated_chain(k_val, 8000)
    n_dims = train_x.shape[1]

    bases = build_bases(train_x, q=2, m=16)
    gram = vmap(type(bases).l2_integral)(bases)
    basis_integrals = vmap(type(bases).integral)(bases)
    parent = make_parent(n_dims, "chain")

    t0 = init_ttns_from_rank1(k_init, bases, train_x, parent, rank=10, noise=1e-2)
    ttns, summ = train_tree_l2(
        t0, parent, bases, train_x, val_x, gram, basis_integrals,
        key=k_init, lr=2e-3, train_steps=1200, batch_sz=512,
        normalize_every=1, log_every=200, label="chain_for_sampler",
    )

    print("\n采样中 ...", flush=True)
    samp_j, dbg = sample_ttns(ttns, bases, parent, k_samp, n=20000, grid_size=400, return_debug=True)
    samp = np.asarray(samp_j)
    data = np.asarray(train_x)
    print(f"  各节点条件密度负质量比例 = {dbg['neg_mass_frac']}")
    cm2 = dbg["cond_mean"][2]  # E[x2|x0,x1] 由密度直接算
    cm1 = dbg["cond_mean"][1]
    print(f"  corr(E[x1|x0], 采样x0) = {np.corrcoef(cm1, samp[:,0])[0,1]:.3f}")
    print(f"  corr(E[x2|x0,x1], 采样x1) = {np.corrcoef(cm2, samp[:,1])[0,1]:.3f}")
    print(f"  std(E[x2|·])={cm2.std():.3f}  样本x2总std={samp[:,2].std():.3f}  "
          f"平均条件展宽(内)~sqrt(var_x2-var_condmean)={np.sqrt(max(samp[:,2].var()-cm2.var(),0)):.3f}")

    print("\n==== 检查 1：解析边缘 CDF vs 经验 CDF ====")
    max_cdf_diff = 0.0
    for d in range(n_dims):
        lo, hi = float(np.asarray(bases.knots)[d, 0]), float(np.asarray(bases.knots)[d, -1])
        xq = np.linspace(lo, hi, 60)
        a = analytic_marginal_cdf(ttns, bases, parent, d, xq)
        e = empirical_cdf(samp[:, d], xq)
        diff = float(np.max(np.abs(a - e)))
        max_cdf_diff = max(max_cdf_diff, diff)
        print(f"  dim{d}: max|CDF_analytic - CDF_empirical| = {diff:.4f}")
    print(f"  全维最大 CDF 偏差 = {max_cdf_diff:.4f}")

    print("\n==== 检查 2：样本相关性 vs 模型解析相关性（隔离拟合质量）====")
    cs = np.corrcoef(samp.T)
    cm = model_corr(ttns, bases, parent, n_dims)
    print(f"  模型解析 corr(0,1),(1,2),(0,2) = {cm[0,1]:.3f},{cm[1,2]:.3f},{cm[0,2]:.3f}")
    print(f"  样本经验 corr(0,1),(1,2),(0,2) = {cs[0,1]:.3f},{cs[1,2]:.3f},{cs[0,2]:.3f}")
    corr_diff = float(np.max(np.abs(cm - cs)))
    print(f"  最大相关性偏差 |模型-样本| = {corr_diff:.4f}")

    print("\n==== 参考：数据 vs 样本（受拟合质量影响，仅供参考）====")
    cd = np.corrcoef(data.T)
    print(f"  data   corr(0,1),(1,2),(0,2) = {cd[0,1]:.3f},{cd[1,2]:.3f},{cd[0,2]:.3f}")
    print(f"  data/sample mean = {data.mean(0)} / {samp.mean(0)}")
    print(f"  data/sample std  = {data.std(0)} / {samp.std(0)}")

    print("\n==== 裁决：解析条件回归 E[x2|x1=v] 的斜率 ====")
    I, M, S = model_moment_vectors(bases, n_dims)
    base_d = jax.tree_util.tree_map(lambda a: a[1], bases)
    vs = np.linspace(0.2, 0.8, 7)
    ex2_given = []
    for v in vs:
        bv = np.asarray(base_d(float(v)))
        num = np.asarray(I).copy(); num[1] = bv; num[2] = M[2]
        den = np.asarray(I).copy(); den[1] = bv
        numv = float(batch_eval_rank1_ttns(ttns, jnp.asarray(num)[None], parent)[0])
        denv = float(batch_eval_rank1_ttns(ttns, jnp.asarray(den)[None], parent)[0])
        ex2_given.append(numv / denv)
    ex2_given = np.array(ex2_given)
    slope = np.polyfit(vs, ex2_given, 1)[0]
    print(f"  v={np.round(vs,2)}")
    print(f"  E[x2|x1=v]={np.round(ex2_given,3)}  斜率≈{slope:.3f}")
    print(f"  (若模型corr(1,2)=0.958 应有斜率≈{0.958*0.253/0.230:.3f}；采样cond_mean斜率≈"
          f"{np.polyfit(samp[:,1], cm2, 1)[0]:.3f})")

    print("\n==== 对比：同点解析 cond_mean(无网格无截断) vs 采样器网格 cond_mean ====")
    sub = slice(0, 2000)
    bx0 = np.asarray(_basis_eval_dim(bases, 0, jnp.asarray(samp[sub, 0])))
    bx1 = np.asarray(_basis_eval_dim(bases, 1, jnp.asarray(samp[sub, 1])))
    cm2_ana = []
    for k in range(bx0.shape[0]):
        num = np.asarray(I).copy(); num[0] = bx0[k]; num[1] = bx1[k]; num[2] = M[2]
        den = np.asarray(I).copy(); den[0] = bx0[k]; den[1] = bx1[k]
        nv = float(batch_eval_rank1_ttns(ttns, jnp.asarray(num)[None], parent)[0])
        dv = float(batch_eval_rank1_ttns(ttns, jnp.asarray(den)[None], parent)[0])
        cm2_ana.append(nv / dv)
    cm2_ana = np.array(cm2_ana)
    print(f"  解析 cond_mean 斜率(vs x1) = {np.polyfit(samp[sub,1], cm2_ana, 1)[0]:.3f}")
    print(f"  网格 cond_mean 斜率(vs x1) = {np.polyfit(samp[sub,1], cm2[sub], 1)[0]:.3f}")
    print(f"  两者均值差 = {np.mean(np.abs(cm2_ana - cm2[sub])):.4f}")

    ok_cdf = max_cdf_diff < 0.05
    ok_corr = corr_diff < 0.06
    print(f"\n结果：边缘CDF {'PASS' if ok_cdf else 'FAIL'} | 样本-模型相关性 {'PASS' if ok_corr else 'FAIL'}")
    if not (ok_cdf and ok_corr):
        sys.exit(1)


if __name__ == "__main__":
    main()
