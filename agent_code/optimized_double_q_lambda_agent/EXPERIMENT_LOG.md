# 优化版 Double Q(lambda) 实验日志

本文件是该 agent 的长期实验记录。算法、特征、奖励和输出文件名保留英文
标识，以便和配置文件及自动化脚本一一对应；其余说明以中文为准。

## Task 1 实验协议

| 项目 | 固定约定 |
|---|---|
| 主 agent | `optimized_double_q_lambda_agent` |
| 算法 | Double Q(lambda)、按动作 tile coding、Watkins trace 截断 |
| 特征 | `continuous-v2`（84 个连续值） |
| 奖励 | `r7_safe_credit_potential` |
| 安全掩码 | Task 1 使用 `survival-mask-v1/off` |
| 训练 | seed 11、100,000 个 action steps、CPU |
| 开发集评估 | seeds `10000--10004`，每个 seed 20 局 |
| 独立确认集 | seeds `11000--11099`，每 seed 一局；只对选出的候选执行 |
| 选模顺序 | `all-coins rate`、`mean coins`、`coins/100 steps`、完成步数、循环率 |

## T1-E01 — 已完成（2026-09-17）

| 候选 | Slurm 作业 | 状态 | 训练步数 | 最佳快照 | 捡满率 | 平均金币 | 备注 |
|---|---|---|---:|---|---:|---:|---|
| 单表 Q-learning 基线 | `472871` | 完成，40 分 42 秒 | 100,056 | `step_0100056.pkl` | 71% | 47.98 | 最佳基线，未达到 90% 门槛。 |
| Watkins Double Q(lambda) | `472872` | 完成，49 分 00 秒 | 100,008 | `step_0100008.pkl` | **96%** | **49.51** | 开发集达到 Task 1 门槛。 |

每项作业会保留实时 checkpoint，并每 25k action steps 复制一个不可变快照；
训练结束后对每个快照运行五个 seed 的冻结开发集评估，并生成
`best_task1_selection.json`。完成后在本节追加原始运行目录、快照评估 JSON、
checkpoint SHA-256 与最终指标。

### 冻结开发集结果（5 个 seed × 20 局 = 100 局）

| 模型 | 快照 action steps | 捡满率 | 平均金币 | coins/100 steps | 成功局平均完成步数 | 循环率 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 单表 Q-learning | 25,200 | 47% | 47.69 | 17.60 | 125.43 | 31% | 初期有效。 |
| 单表 Q-learning | 50,033 | 0% | 9.62 | 2.40 | — | 100% | 策略退化。 |
| 单表 Q-learning | 75,052 | 0% | 9.84 | 2.46 | — | 100% | 策略退化。 |
| 单表 Q-learning | 100,056 | 71% | 47.98 | 23.34 | 126.20 | 17% | 此基线的最佳快照。 |
| Double Q(lambda) | 25,200 | 78% | 47.60 | 16.78 | 250.91 | 82% | 已接近门槛，但 WAIT/循环偏多。 |
| Double Q(lambda) | 50,382 | 40% | 42.14 | 11.75 | 296.38 | 99% | 中期退化。 |
| Double Q(lambda) | 75,280 | 56% | 44.23 | 13.47 | 272.27 | 80% | 部分恢复，仍不如早期快照。 |
| Double Q(lambda) | 100,008 | **96%** | **49.51** | **23.90** | 199.10 | **0%** | 当前最佳；进入独立确认。 |

### 当前最佳 checkpoint

- 路径：`runs/qlambda_opt_t1_s11_j472872/checkpoints/snapshots/step_0100008.pkl`
- SHA-256：`5172c40830ea74c5ffa4344f75d2d6a5ee11141a7e2a816f26d9f544a5552827`
- CPU 推理：五个开发 seed 中 `act` p95 最大 6.98 ms，单次最大 11.84 ms，低于
  50 ms / 500 ms 的验收限制。
- 行为诊断：WAIT 率 25.27%，立即折返率 7.41%，长 WAIT 或长往返循环率均为 0%。
- 结论：本结果只证明开发集达标；仍须使用未参与选模的 `11000--11099` 共 100 局
  独立确认集，确认后才宣称通过 Task 1。

### 独立确认集（已完成，通过）

将固定使用上述 SHA-256 对应 checkpoint，在 `11000--11099` 每个 seed 一局进行
100 局只读冻结评估。Slurm 作业 `472894` 于 2026-09-17 完成，耗时 4 分 26 秒；
该集合不参与快照选择。

| 评估局数 | 捡满率 | 平均金币 | coins/100 steps | 循环率 | WAIT 率 | 立即折返率 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | **96%** | **49.58** | **25.05** | **0%** | 24.71% | 6.10% |

五个确认 seed 中 `act` p95 最大为 7.83 ms，单次最大为 11.63 ms，满足
50 ms / 500 ms 时限。确认集捡满率 ≥90%、平均金币 ≥48，故该固定 checkpoint
**正式通过 Task 1 的开发集与独立确认集验收**。后续迁移至 Task 2 时保留此文件
作为可回退的 Task 1 checkpoint，不再以训练 reward 或单局结果替代冻结验收。

## 正式 Task 1 → Task 2 训练链

此前的 `T1-E01` 使用 safety `off`，因此虽已在开发集和独立确认集达到 96%，但不能作为
使用 safety `all` 的正式 Task 2 `--resume-from` 父 run。本链从零重建 seed 11 的相同模型、
`continuous-v2` 和 `r7_safe_credit_potential` 合同，只将 safety 固定为
`survival-mask-v1/all`。

| 实验 ID | 阶段 | 状态 | 固定合同 | 验收 |
|---|---|---|---|---|
| T2-L01 | Task 1 正式父链，seed 11 | 待提交 | Double Q(lambda)、`continuous-v2`、`r7_safe_credit_potential`、safety-all | `9000--9019` 三次连续 mean score ≥48；随后 `10000--10019` 独立 stage gate。 |
| T2-P01 | Task 2 pilot，seed 11 | 等待 T2-L01 | 同一合同，`classic`、BOMB enabled、150k actions / 至少500局 | Task 2 质量门槛、Task 1 保留率与 CPU 时延均通过。 |

T2-L01 的冻结性能评估从第 200 局开始，每 50 个新增训练回合执行一次；只有连续三次
通过后才停止。其 `promotion_audit.json` 同时核验 checkpoint reload、独立打包文件、
20-seed stage gate、无效动作率及 CPU 时延。T2-P01 仅在该审计文件为 `passed=true` 后执行。

Task 2 每 25k action steps 冻结一次候选；每个候选同时在 Task 1 和 Task 2 的
`10000--10019` 上评估。选模必须满足 Task 2 quality gate（平均金币 ≥6、平均炸箱 ≥60、
自杀率 ≤5%、放弹存活率 ≥95%、每弹炸箱 ≥1.5 等）以及 Task 1 分数保留率 ≥90%。
若 T2-P01 未通过，只根据少放弹、自杀、空放、循环或 tile/Q 值诊断修改一个因素，随后从
T2-L01 父 run 重新开始 Task 2，不同时修改特征、奖励和超参数。

重试使用本地且被 Git 忽略的 `.venv-compute` 环境：由
`/home/students/ji/.local/bin/python3.10` 创建，并安装 CPU 训练依赖。

## 集群环境重试记录

| 尝试 | 作业 | 结果 | 原因与处理 |
|---|---|---|---|
| 1 | `472859`、`472860` | 训练前失败 | 登录节点 venv 指向 `/usr/local/bin/python3.10`，compute 节点不存在该解释器。 |
| 2 | `472864`、`472865` | 训练前失败 | compute venv 缺少共享实验入口会导入的 CPU PyTorch。 |
| 3 | `472871`、`472872` | 已完成 | `.venv-compute`：Python 3.10.19、NumPy 2.2.6、Pygame 2.6.1、PyTorch 2.5.1+cpu，已在 compute 验证。 |
