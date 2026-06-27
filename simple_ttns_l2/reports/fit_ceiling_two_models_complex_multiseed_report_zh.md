# Fit Ceiling 多 Seed 对比报告

## 目的

在 complex balanced 目标上，用多个随机种子验证 **balanced TTNS** 相对 **chain TTNS** 的切片 IAE 优势是否稳健。

- Seeds: `[313, 2602, 20260227]`
- 单 seed 配置见各 run 的 `config` 字段

## 聚合结果（balanced 目标，3 切片均值）

| 指标 | 值 |
|------|-----:|
| mean IAE (balanced TTNS) | 0.212800 |
| mean IAE (chain TTNS) | 0.423005 |
| 相对提升 (chain→balanced) | 49.69% |
| 通过 ≥20% 标准 | 是 |

## 分 seed 明细

| seed | mean IAE balanced | mean IAE chain | 提升 |
|-----:|---:|---:|---:|
| 313 | 0.210884 | 0.421025 | 49.91% |
| 2602 | 0.195346 | 0.422043 | 53.71% |
| 20260227 | 0.232168 | 0.425946 | 45.49% |

## 结论

- 3 个 seed 聚合后，匹配拓扑 TTNS 的 mean IAE 比 chain 低 **49.7%**，达到 Phase 2 M1.1 标准。
