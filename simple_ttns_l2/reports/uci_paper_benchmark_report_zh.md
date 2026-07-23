# UCI 论文级五数据集基准（协议骨架）

> **证据标签**：单 seed 初证（跑完后填表；未完成前保持本协议说明）  
> **分支 / 工作树**：`dev` @ `../TTDE-dev`  
> **入口**：`simple_ttns_l2/experiments/uci_ttde_vs_ttns.py --preset paper`  
> **提交**：`sbatch simple_ttns_l2/experiments/run_uci_paper.sbatch`

## 1. 协议

对齐 README Table 3 复现命令（单 `seed=0`）：

| 数据集 | m | TTDE rank | n_comps | batch | steps | train_noise | 论文 TTDE LL |
|---|---|---|---|---|---|---|---|
| POWER | 256 | 16 | 32 | 8192 | 10000 | 0.01 | 0.46 |
| GAS | 512 | 32 | 32 | 1024 | 100000 | 0.01 | 8.93 |
| HEPMASS | 128 | 32 | 32 | 2048 | 10000 | 0.01 | −21.34 |
| MINIBOONE | 64 | 32 | 32 | 1024 | 10000 | 0.08 | −28.77 |
| BSDS300 | 256 | 16 | 32 | 512 | 100000 | 0.01 | 143.30 |

共同项：$q=2$，`em_steps=10`，`init_noise=0.01`，`lr=0.001`；**全量训练数据**；**禁用 early stop**（`ttde_patience=0`）。

对比口径：

- **TTDE（链）**：论文 rank，追求绝对 test_LL 贴近论文。
- **TTNSDE（MI 树）**：同 `m/n_comps/steps/batch/lr/em/noise`；树 rank 默认 `power=6, gas=4, hepmass=3, miniboone=2, bsds300=2`（可用 `R_TTNS` / `R_TTNS_<DS>` 覆盖）。
- **关闭 `match_params`**，避免反解掉论文 TTDE rank。
- `n_comps=32`：跳过线性 TTNS 与切片图；保存 metrics / bars / 参数快照。

数据目录：`/home/sbzeng/2_1/research/datasets/data`。

## 2. 提交命令

```bash
cd /home/sbzeng/2_1/research/TTDE-dev
mkdir -p logs
sbatch simple_ttns_l2/experiments/run_uci_paper.sbatch
# 冒烟（非正式）：
QUICK=1 sbatch simple_ttns_l2/experiments/run_uci_paper.sbatch
```

阵列映射：`0=power, 1=gas, 2=hepmass, 3=miniboone, 4=bsds300`。

## 3. 预期产物

每个数据集：

- `simple_ttns_l2/reports/uci_ttde_vs_ttns_metrics_paper_<ds>_seed0.json`
- `simple_ttns_l2/reports/uci_<ds>_ttde_vs_ttns_bars_paper_<ds>_seed0.png`
- `simple_ttns_l2/reports/uci_<ds>_ttde_vs_ttns_params_paper_<ds>_seed0.pkl`
- `logs/uci_paper_<jobid>_<array>.out|.err`

## 4. 结果表（跑完后填）

| 数据集 | 论文 TTDE | 本跑 TTDE | 本跑 TTNSDE | Δ(TTNSDE−TTDE) | gap(本跑TTDE−论文) | 状态 |
|---|---|---|---|---|---|---|
| POWER | 0.46 | | | | | 待跑 |
| GAS | 8.93 | | | | | 待跑 |
| HEPMASS | −21.34 | | | | | 待跑 |
| MINIBOONE | −28.77 | | | | | 待跑 |
| BSDS300 | 143.30 | | | | | 待跑 |

## 5. 已知风险

- GAS / BSDS300 各 10 万步 × mixture 32，墙钟可能数天。
- 高维 MI 树 hub 可能使 TTNSDE 参数过大；只降 `r_ttns`，不改 TTDE 论文 rank。
- 训练路径为本仓库 `fit_ttde_tt` / `fit_ttde_ttns`，与原文 `ttde.train`+wandb 不完全同构，绝对 LL 可能有偏差；同时报告相对 Δ 与距论文差距。

---

*创建：2026-07-23 — 论文级五数据集 sbatch 协议。*
