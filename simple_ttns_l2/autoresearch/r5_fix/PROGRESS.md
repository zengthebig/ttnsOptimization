# R5 Autoresearch 进度

> 本表由 `run_attempt.py` 追加行；人工可补充「结论」列。  
> 图路径均相对于 `simple_ttns_l2/autoresearch/r5_fix/`。

| ID | 日期 | 名称 | variant | L4 LL | L4 nonpos | L4 std_ratio | commit | 结论 |
|---|---|---|---|---:|---:|---:|---|---|
| — | — | R5 λ=0 (audit 基线) | — | −13.02 | 0.57 | 0.35 | — | 对照，见 `slices_audit.json` |
| — | — | R7 sampled (audit) | — | +1.05 | ~0 | 0.97 | — | 上限参照 |
| — | — | R5 nonneg (remedy) | nonneg | +2.45 | 0.00 | — | remedy | 非负不够，相关塌缩 |
| — | — | marginal λ=0.3/1.0 | marginal_l2 | 更差 | ~1.0 | >5 | remedy | **否定** |

| 000 | 2026-07-10 | baseline R5 analytic chain | baseline | -11.30 | 0.55 | 0.95 | `autoresearch(r5): 000 baseline R5 analytic chain` | ✅ 复现失效：L4 负密度高、LL 崩溃 |

| 001 | 2026-07-10 | nonneg analytic corr penalty | nonneg_corr | 0.38 | 0.00 | 2.51 | `autoresearch(r5): 001 nonneg analytic corr penalty` | ❌ 权重过强：方差膨胀、LL 未达标、相关未改善 |

| 002 | 2026-07-10 | nonneg light corr penalty | nonneg_corr | 2.23 | 0.00 | 1.56 | `autoresearch(r5): 002 nonneg light corr penalty` | ❌ LL/nonpos 达标但方差过宽，相关更差 |

| 003 | 2026-07-10 | nonneg source blocks | nonneg | 1.58 | 0.00 | 1.41 | `autoresearch(r5): 003 nonneg source blocks` | ❌ source 分块仍过宽，相关误差更差 |

| 004 | 2026-07-10 | nonneg source rank4 | nonneg | 2.53 | 0.00 | 1.11 | `autoresearch(r5): 004 nonneg source rank4` | ❌ 主指标达标，但 L4 最强相关对塌缩 |

| 005 | 2026-07-10 | nonneg joint block k4 | joint_block | -4.65 | 0.00 | 1.70 | `autoresearch(r5): 005 nonneg joint block k4` | ❌ 低网格 joint 目标破坏边缘，相关仍塌缩 |

| 006 | 2026-07-10 | nonneg immediate rank16 short | nonneg | 1.82 | 0.00 | 0.95 | `autoresearch(r5): 006 nonneg immediate rank16 short` | ❌ 主指标达标，但 L4 最强相关对仍塌缩 |

| 007 | 2026-07-10 | hybrid sampled propagation | hybrid_sample_prop | 7.07 | 0.00 | 0.97 | `autoresearch(r5): 007 hybrid sampled propagation` | ✅ hybrid 达标：L4 相关同号且 \|Δr\|≈0.095 |

| — | 2026-07-10 | target/fit-quality audit | diagnostic | — | — | — | `artifacts/target_audit_summary_zh.md` | ✅ L1 解析 target 正确（truth/sampled/analytic r≈0.69），但解析非负 L2 fit 后 r≈0.02–0.05；失效点在拟合器而非 `UpperForest.pair_cdf` |

| 008 | 2026-07-10 | analytic target sampled MLE | analytic_mle | 5.96 | 0.00 | 0.97 | `autoresearch(r5): 008 analytic target sampled MLE` | ✅ 达标：解析 UpperForest target + 非负 MLE 保住相关，L4 r=+0.567 vs GT +0.758，说明 sample 的作用是修复块内拟合器而非替代解析传播 |

| 009 | 2026-07-10 | nonneg analytic L2 fast lr | nonneg | 5.89 | 0.00 | 0.94 | `autoresearch(r5): 009 nonneg analytic L2 fast lr` | ✅ 达标：纯解析 UpperForest + 非负解析 L2；L4 r=+0.513 vs GT +0.758。定位根因是 `an_lr=3e-5` 对 `raw**2` 参数化过小，相关通道增长太慢 |

<!-- 新 attempt 行插入此处上方 -->
