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

## T1-E01 — 已提交（2026-09-17）

| 候选 | Slurm 作业 | 状态 | 训练步数 | 最佳快照 | 捡满率 | 平均金币 | 备注 |
|---|---|---|---:|---|---:|---:|---|
| 单表 Q-learning 基线 | `472859` | 训练前失败 | 100,000 | — | — | — | compute 节点没有登录节点 venv 所链接的 `/usr/local/bin/python3.10`，未产生 transition。 |
| Watkins Double Q(lambda) | `472860` | 训练前失败 | 100,000 | — | — | — | 同一解释器环境问题，未产生 transition。 |

每项作业会保留实时 checkpoint，并每 25k action steps 复制一个不可变快照；
训练结束后对每个快照运行五个 seed 的冻结开发集评估，并生成
`best_task1_selection.json`。完成后在本节追加原始运行目录、快照评估 JSON、
checkpoint SHA-256 与最终指标。

重试使用本地且被 Git 忽略的 `.venv-compute` 环境：由
`/home/students/ji/.local/bin/python3.10` 创建，并安装 CPU 训练依赖。

## 集群环境重试记录

| 尝试 | 作业 | 结果 | 原因与处理 |
|---|---|---|---|
| 1 | `472859`、`472860` | 训练前失败 | 登录节点 venv 指向 `/usr/local/bin/python3.10`，compute 节点不存在该解释器。 |
| 2 | `472864`、`472865` | 训练前失败 | compute venv 缺少共享实验入口会导入的 CPU PyTorch。 |
| 3 | `472871`、`472872` | 运行中 | `.venv-compute`：Python 3.10.19、NumPy 2.2.6、Pygame 2.6.1、PyTorch 2.5.1+cpu，已在 compute 验证。 |
