# CNN Path Double DQN — Task 1 实验日志

更新日期：2026-09-15。本文是本 agent 唯一的长期实验记录；原始 JSON、逐局数据和 checkpoint 为证据，不以训练奖励或 loss 代替冻结评估。路径均相对本文。运行产物位于本地 `runs/`，不随本文提交；换机器复现需另行归档这些产物。

## 1. 当前状态与验收

2026-09-15 查询 Slurm：无活动任务。T1-E04、T1-E05 的训练和冻结选模均已完成。当前最佳为 **T1-E05 / r5 / 75,130 steps：17/100 局捡满，平均 41.78 枚金币**，未达到 Task 1 目标，尚未进入 Task 2。

- 场景：`coin-heaven`，无箱子、无对手，50 枚金币，每局最多 400 步；Task 1 禁用 BOMB。
- 开发集：环境评估 seeds `10001–10005`，每 seed 20 局，共 100 局，关闭探索和学习。
- 验收：开发集至少 90/100 局捡满；再用 reserved seeds `21000–21099`，每 seed 一局，至少 90/100 局捡满。确认集不参与选模；目前尚未运行确认集。
- 选模：依次最大化捡满率、平均金币、coins/100 steps，再最小化成功局平均完成步数、循环率（长等待率+长往返率）；完全相同时保留更早的快照。无成功局时完成步数记为 N/A，排序按正无穷处理。
- CPU：完整 `act` 的 p95/max 均需记录，验收要求 max < 0.5s；当前开发机实测不代替官方硬件验证。
- 团队迭代目标截至 2026-09-20；课程提交节点以 [课程要求](../../PROJECT_REQUIREMENTS.md) 和最新公告为准，不能将团队目标当作官方截止日期。

## 2. 固定模型、特征与训练配置

### 2.1 网络与学习算法

`path-spatial-residual-v1`：输入 `[17,17,17]`，3×3 卷积 17→64；四个无降采样残差块，dilation 分别为 1/2/4/8，每块两层卷积。拼接 self 位置处的 64 维特征、全局平均池化、全局最大池化，得到 192 维；全连接 192→128→6，输出 Q 值，无 dueling。

Double DQN：online 网络在 next-state mask 内选动作，target 网络估值；终局不 bootstrap。当前实际 learner 使用 **1-step、均匀 replay、Adam、SmoothL1Loss**；没有 D4、PER 或四步回报。target sync 按 optimizer **updates** 计数，不是每 2,000 环境步。

| 参数 | 固定值 |
|---|---:|
| gamma / learning rate | 0.95 / 0.0001 |
| batch / replay capacity | 64 / 20,000 |
| warmup / target sync | 5,000 transitions / 2,000 updates |
| gradient norm clip | 10 |
| epsilon | 1.0→0.05，前 80,000 action steps 线性衰减，之后保持 0.05 |
| 正式筛选预算 | 100,000 action steps，在回合结束检查，允许小幅超出 |
| 训练 seed | agent/experiment 11，environment 1011，official opponent 3011 |
| early stopping | 关闭 shaped-reward early stopping |
| 动作顺序 | UP、RIGHT、DOWN、LEFT、WAIT、BOMB |

物理 legal mask 排除不可进入的动作；课程 mask 额外禁用 BOMB。safety mask 是独立机制，不能把“safety off”理解成完全没有动作 mask。E04/E05 safety 为 `survival-mask-v1 / off / horizon=7 / fallback=physical_q`；E02 为 mode=all。

E05 训练硬件：GTX 1080 Ti 单 GPU，PyTorch `2.5.1+cu118`，CUDA 11.8，Python 3.10.0。CPU 推理设置 torch 单线程。其余运行的精确软硬件以各自 metadata 为准。

### 2.2 特征契约：board-path-history-v2

索引从 0 开始；坐标为 `board[channel,x,y] == field[x,y]`。距离为客观 BFS 距离场，不提供“最佳动作”标签。

| 索引 | 名称 | 含义和取值 |
|---:|---|---|
| 0 | stone_wall | 石墙，0/1 |
| 1 | crate | 箱子，0/1 |
| 2 | coin | 可见金币位置，0/1 |
| 3 | self | 唯一自身位置，0/1；按 `game_state['self'][3]` 重置 |
| 4 | opponent | 其他存活 agent 位置，0/1 |
| 5 | bomb_presence | 炸弹位置，0/1 |
| 6 | bomb_timer | `clip(timer/BOMB_TIMER,0,1)`；非炸弹格为 0 |
| 7 | current_explosion | 当前 explosion_map > 0，0/1 |
| 8 | danger_t1 | danger predictor 的第 1 时间层，0/1 |
| 9 | danger_t2 | 第 2 时间层，0/1 |
| 10 | danger_t3 | 第 3 时间层，0/1 |
| 11 | danger_t2_or_later | 第 2 层及以后任一层危险，0/1 |
| 12 | nearest_coin_distance | 到最近金币的 BFS 距离，`min(d,32)/32`；不可达/无目标=1 |
| 13 | nearest_crate_frontier_distance | 到箱子前沿可达格的 BFS 距离，同上 |
| 14 | nearest_opponent_distance | 到对手目标的 BFS 距离，同上 |
| 15 | previous_position | 历史中最近一次决策前的位置，0/1；首帧全 0 |
| 16 | recent_visit_frequency | 最近最多 16 次决策前位置的次数/16；首帧全 0 |

距离截断尺度由 `COLS+ROWS-2` 得到，官方 17×17 棋盘为 32；replay 当前按 32 量化距离，不能不改契约就换棋盘尺寸。BFS 的阻挡/目标语义沿用共享 context。replay 将二值通道 packbits，timer 按 4、距离按 32、历史频次按 16 编解码。

每个决策取特征后才记录当前位置。同一 `(round,step)` 缓存特征，最多保留两个决策状态，回调重访 old/new state 不重新推进历史；换 round 在提取第一帧之前清空历史和缓存。

版本变化：

- v1：直接继承共享 `board-v1` 的 self 通道。共享金币循环覆盖 `position`，导致有金币时 self 标在最后遍历的金币上。空间网络因而在错误位置读局部特征。
- commit `24c9f07`：增加两状态缓存，修复重复编码的历史不一致，但 **尚未修复 self 位置**。
- v2 / commit `fa75d10`：在本 agent 内清空第 3 通道并按真实 self 坐标置 1，升级 feature ID，不改共享编码器。旧 checkpoint/replay 只用于历史分析，不续训为 v2 基线。

### 2.3 奖励版本与权重

下表来自运行 metadata 的 resolved rewards。缺失项表示该奖励版本没有这一项，而非采用其他默认值。

| 项目 | r3_potential | r5_conditional_loop | r7_safe_credit_sparse |
|---|---:|---:|---:|
| coin_collected | 3 | 3 | 3 |
| step / invalid_action | -0.01 / -0.2 | -0.01 / -0.2 | -0.01 / -0.1 |
| coin_found / crate_destroyed | 0.25 / 0.1 | 0.25 / 0.1 | — / 0.2 |
| killed_opponent | 5 | 5 | 5 |
| killed_self / got_killed | -7 / -5 | -7 / -5 | -20 / -10 |
| survived_round | 0.25 | 0.25 | — |
| potential_gamma | 0.95 | 0.95 | 0.95 |
| potential_coin_weight | 0.5 | 0.5 | — |
| potential_crate_weight | 0.25 | 0.25 | 0.25 |
| potential_safety_weight | 0.25 | 0.25 | — |
| potential_danger_weight | — | — | 1 |
| conditional_loop_penalty | — | -0.08 | — |
| avoidable_wait_penalty | — | -0.04 | — |
| unsafe_bomb_penalty | — | — | -20 |
| useful_bomb_per_crate / cap | — | — | 0.2 / 3 |

Potential shaping 为 `0.95*Phi(next)-Phi(old)`，终局 next potential 为 0。金币项使用 `weight*exp(-BFS_distance/4)`；无可达金币时尝试箱子前沿项。安全项按最早危险时间计算，r7 使用负 danger potential 而非正 safety potential。Task 1 无箱子/炸弹/对手，多数相关项不触发；r7 sparse **没有金币距离 potential**。

r5 在 r3 上只增加两个窄条件：安全、最近可达金币目标未切换、实际移动回到上次位置时罚 -0.08；WAIT 且当前位置和下一时间层安全、存在朝该目标缩短距离的安全合法移动时罚 -0.04。不是所有折返或所有 WAIT 都罚。具体公式见 [奖励实现](../team_agent/rewards.py) 和 [时序条件](../learning_common/temporal_reward.py)。

## 3. 实验总表

长等待/长往返为“发生该现象的局数/总局数”；WAIT/立即折返为逐决策比例，不可混用。旧 v1 结果不参与可信最佳模型比较。

| ID | job / 状态 | feature / reward / safety | steps / rounds | 最佳评估 step | 捡满率 | 平均金币 | 长等待/长往返 | 结论 |
|---|---|---|---|---:|---:|---:|---|---|
| T1-E01 | 471424 completed，受污染 | v1 / r3 / off | 100,361 / 251 | 100,361（仅 final 评估） | 0% | 13.03 | 56% / 53% | self bug，不能判断 r3 优劣 |
| T1-E02 | 471427 completed，受污染 | v1 / r7 sparse / all | 100,000 / 250 | 100,000（仅 final 评估） | 0% | 2.63 | 31% / 46% | self bug，不能判定 r7 不适合 CNN |
| T1-E03 | 471429 cancelled | v1+cache / r3 / off | 未有完整 summary | N/A | N/A | N/A | N/A | 因输入 bug 停止，无性能结论 |
| T1-E04 | 471431 completed，可信 | v2 / r3 / off | 100,142 / 273 | 25,200 | 0% | 13.35 | 52% / 49% | 后期 WAIT 退化 |
| T1-E05 | 471439 completed，可信 | v2 / r5 / off | 100,394 / 283 | 75,130 | 17% | 41.78 | 26% / 18% | 当前最佳，100k 后退化，未达标 |
| T1-E06 | 471455 submitted | v2 / r5 / off + 4-step | 目标 100k / seed 11 | N/A | N/A | N/A | N/A | 已通过本地 smoke；等待/进行正式训练与冻结评估 |

## 4. 各轮证据与诊断

### T1-E01 — 原始 v1 + r3

- UTC：2026-09-14 18:20:10→19:23:46；Slurm elapsed 01:03:43。代码 `2eba43ae53021d539f7791bd4e5021cb7a88055f`。
- 配置：[cnn_path_task1_r3.json](../../experiments/configs/cnn_path_task1_r3.json)；seed 11，r3 权重见上表，safety off。
- 95,362 updates，final loss 0.1368398964；平均训练奖励 85.3583，最后 100 局 101.3062。训练奖励上升不构成达标证据。
- 100 局 final 冻结：捡满 0%，平均金币 13.03。结果受 self-channel bug 污染，保留但不用于奖励消融。
- 原始：[metadata](../../runs/cnn_path_t1_e1_s11_j471424/metadata.json)、[summary](../../runs/cnn_path_t1_e1_s11_j471424/training_summary.json)、[冻结汇总](../../runs/cnn_path_t1_e1_eval_j471424/cnn_path_t1_e1_eval_j471424_summary/summary.csv)、[final checkpoint](../../runs/cnn_path_t1_e1_s11_j471424/checkpoints/final.pt)。没有快照选模记录，不虚构 best_task1。

### T1-E02 — 原始 v1 + r7 sparse + safety

- UTC：2026-09-14 19:38:04→21:05:32；Slurm elapsed 01:27:37。代码 `fcd97e5aa576c4cd3f5b702f9d4359a39e64e8ac`。
- 配置：[cnn_path_task1_r7_safety.json](../../experiments/configs/cnn_path_task1_r7_safety.json)；seed 11，r7 权重见上表，safety all/horizon 7/physical_q fallback。
- 95,001 updates，final loss 0.0000662690；平均训练奖励 61.546，最后 100 局 63.52。
- 100 局 final 冻结：捡满 0%，平均金币 2.63。低 loss 不等于有效导航；输入错误和奖励/safety 同时变化，不能作为公平 r3/r7 对照。
- 原始：[metadata](../../runs/cnn_path_t1_r7safety_s11_j471427/metadata.json)、[summary](../../runs/cnn_path_t1_r7safety_s11_j471427/training_summary.json)、[冻结汇总](../../runs/cnn_path_t1_r7safety_eval_j471427/cnn_path_t1_r7safety_eval_j471427_summary/summary.csv)、[final checkpoint](../../runs/cnn_path_t1_r7safety_s11_j471427/checkpoints/final.pt)。

### T1-E03 — v1 + 两状态缓存，发现位置 bug 后取消

- UTC 开始：2026-09-14 22:30:39；Slurm `CANCELLED`，elapsed 00:16:33。代码 `24c9f071cc31d5e2a61913b062c7a9f269050edb`。
- 配置：[cnn_path_task1_e2_r3.json](../../experiments/configs/cnn_path_task1_e2_r3.json)，seed 11，r3/safety off。缓存已修复，self 仍错误。
- [metadata](../../runs/cnn_path_t1_e2_r3_s11_j471429/metadata.json) 残留 `status=running`、无 ended_at，不能据此误报仍在训练；调度状态为准。没有完整 training_summary 或冻结结果；保留日志/部分 checkpoint，仅作诊断。
- 下一步：本目录局部修复 self，升级 v2，从随机初始化重训；依赖评估 471430 随旧训练停止，不形成评估结果。

### T1-E04 — v2 + r3 可信基线

- UTC：2026-09-14 22:49:02→2026-09-15 00:00:22；代码 `fa75d10fdeb72d21043f025c5cf67c43c08a2e6e`。GPU train 01:11:24，快照评估 00:59:04，总 Slurm elapsed 02:11:02。
- 配置：[cnn_path_task1_e2_r3.json](../../experiments/configs/cnn_path_task1_e2_r3.json)；实际 expanded config 才是执行依据：seed 11，100k action budget，r3，safety off，从头训练。
- 95,143 updates；平均训练奖励 112.8489、最后 100 局 134.2925、final loss 0.0191482734、epsilon 0.05。

| 快照 steps | 捡满率 | 平均金币 | coins/100 steps | WAIT | 立即折返 | 长等待 | 长往返 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25,200（best） | 0% | 13.35 | 3.3375 | 49.1625% | 43.5100% | 52% | 49% |
| 50,313 | 0% | 10.23 | 2.5575 | 90.4875% | 4.6500% | 95% | 6% |
| 75,022 | 0% | 9.35 | 2.3375 | 59.0650% | 36.5550% | 64% | 40% |
| 100,142（final） | 0% | 9.77 | 2.4425 | 81.4700% | 11.8325% | 87% | 16% |

最佳全部达到 400 步上限，成功完成步数 N/A；CPU act p95 0.0101455688s、max 0.0194311142s，无非法动作和超时。延长这条训练轨迹没有改善，反而退化；不能从单 seed 推断所有更长训练都无效。

原始：[metadata](../../runs/cnn_path_v2_r3_s11_j471431/metadata.json)、[训练 summary](../../runs/cnn_path_v2_r3_s11_j471431/training_summary.json)、[训练曲线](../../runs/cnn_path_v2_r3_s11_j471431/training_progress.png)、[全部评估](../../runs/cnn_path_v2_r3_s11_j471431/snapshot_evaluations.json)、[选模依据](../../runs/cnn_path_v2_r3_s11_j471431/best_task1_selection.json)。

最佳：[policy_025200.pt](../../runs/cnn_path_v2_r3_s11_j471431/checkpoints/snapshots/policy_025200.pt)，复制为 [best_task1.pt](../../runs/cnn_path_v2_r3_s11_j471431/checkpoints/best_task1.pt)，SHA-256 `791401a3d16c23efc36c63351e1208999c2a2b6dabf5e85309f0568a418742f8`。保留 [final.pt](../../runs/cnn_path_v2_r3_s11_j471431/checkpoints/final.pt)，不是最佳。

### T1-E05 — v2 + r5 条件抗循环

- UTC：2026-09-15 11:34:22→12:50:02；代码 `826e221a304048f24d8574e11d03e62bbdbb040c`。GPU train 01:15:44，快照评估 00:54:31，总 Slurm elapsed 02:10:52。
- 配置：[cnn_path_task1_v2_r5.json](../../experiments/configs/cnn_path_task1_v2_r5.json)；seed 11，从头训练，100k budget，v2/同网络/同超参数/safety off，仅奖励换为 r5。n_rounds 上限 2,000，但实际由 action budget 在第 283 局终止，不是训练了 2,000 局。
- 95,395 updates；平均训练奖励 110.4836、最后 100 局 133.4748、final loss 0.0082681803、epsilon 0.05。

| 快照 steps | 捡满率 | 平均金币 | coins/100 steps | WAIT | 立即折返 | 长等待 | 长往返 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25,200 | 2% | 22.97 | 5.818578 | 31.7273% | 27.0841% | 35% | 21% |
| 50,059 | 2% | 31.32 | 7.933935 | 13.0839% | 25.4382% | 14% | 23% |
| 75,130（best） | 17% | 41.78 | 11.650540 | 19.5616% | 19.3358% | 26% | 18% |
| 100,394（final） | 0% | 27.11 | 6.777500 | 51.1675% | 16.7950% | 60% | 13% |

最佳成功局平均完成步数 156.5294，83% 局达到步数上限；朝金币距离缩短的决策比例 51.4626%。CPU act p95 0.0102385998s、max 0.0146384239s，无非法动作和超时。

结论：同输入、seed、网络和预算下，r5 最佳平均金币比 r3 最佳高 28.43，捡满率高 17 个百分点；WAIT 和立即折返明显减少。但只有一个训练 seed，尚不能声称稳定性已证实。75k→100k WAIT 上升、冻结表现下降，必须保留最佳快照，不能直接部署 final。信用分配、优化稳定性或探索变化可能参与退化，当前数据不能确定根因。

数据限制：训练 summary 的 conditional_loop/avoidable_wait 计数为 0；这不足以证明奖励未触发，不能用这些汇总字段计算训练罚项触发率。后续需核查逐局字段和汇总链路。

原始：[metadata](../../runs/cnn_path_v2_r5_s11_j471439/metadata.json)、[训练 summary](../../runs/cnn_path_v2_r5_s11_j471439/training_summary.json)、[训练曲线](../../runs/cnn_path_v2_r5_s11_j471439/training_progress.png)、[全部评估](../../runs/cnn_path_v2_r5_s11_j471439/snapshot_evaluations.json)、[选模依据](../../runs/cnn_path_v2_r5_s11_j471439/best_task1_selection.json)。各冻结 run 的 `episodes.jsonl`/官方统计和 summary 保留在相应开发评估目录，可用于逐局复核。

当前最佳：[policy_075130.pt](../../runs/cnn_path_v2_r5_s11_j471439/checkpoints/snapshots/policy_075130.pt)，复制为 [best_task1.pt](../../runs/cnn_path_v2_r5_s11_j471439/checkpoints/best_task1.pt)，SHA-256 `a2fa243b8ba6369ce779046c8e5570a3e3f5a80ca35f9c5acc9291c087f4b584`。保留 [final.pt](../../runs/cnn_path_v2_r5_s11_j471439/checkpoints/final.pt)，不将其当作最佳。

### T1-E06 — planned：r5 + 4-step return

已实现，Slurm job **471455** 已于 2026-09-15 提交到 `students` 分区，申请一张 GPU，目标 100,000 action steps、seed 11。配置为 [cnn_path_task1_v2_r5_n4.json](../../experiments/configs/cnn_path_task1_v2_r5_n4.json)，因此原始 [r5 1-step 配置](../../experiments/configs/cnn_path_task1_v2_r5.json) 和旧 checkpoint 仍可复现/评估。仅改变 return horizon：保留 v2、r5、网络、seed 11、100k budget、epsilon 日程、safety off 和其他超参数；从随机初始化开始。实现的 transition 保存实际 horizon，使用 `sum(gamma^k r[t+k]) + gamma^steps Q(s[t+steps])`；终局不足四步的尾部 transition 不 bootstrap。

本地 CPU smoke（seed 12，1 局、400 steps）完成，checkpoint 中 `n_step=4`、pending 为空，replay 同时包含终局尾部的 1/2/3 step 与正常 4 step transition；训练 summary 已记录 conditional-loop=60、avoidable-WAIT=104 及对应罚分。该 smoke 只验证数据流，不用于性能结论。

每 25k 保存快照并用同一开发集冻结选模，改善词典序指标才保留；没有改善则记录失败并恢复 1-step。通过后再决定 D4、dueling 和 seeds 11/22/33 稳定性实验。reserved 集仍保持未触碰。此条目只记录下一步，不授权/表示本次已启动训练。

### 4.1 已核验的 final checkpoint 哈希

final 与 best 分开记录；以下文件均实际计算 SHA-256，而非根据文件名推定。

| 实验 | final.pt SHA-256 |
|---|---|
| E01 | `c71da45f291314caf1e2ffbe405648b1940cfa26e9f2a79ba5fe884b03ba4e6e` |
| E02 | `f359dd1f618e6e7397ac145b9e37711959277d98d0958115b49b9d6f8f9e1ca8` |
| E04 | `489b6897a5f4442ee998870c2a7d6d7f32deade7666610cc13ce52201d214f1f` |
| E05 | `08407005e9b764612f5e2d276b31d62de84069cad56fd0a4b5839ec30ddc2871` |

E03 有部分 final 和 policy_025000 文件，但没有完整终止 summary；不将文件名 final 解释为正式训练完成或合格模型。

## 5. 验证记录（不计为正式实验）

| 主实验 | 验证 | 状态/证据 |
|---|---|---|
| E04 | 3 局 smoke train，checkpoint reload，CPU frozen smoke | Slurm 471431.0/.1 completed，分别 23s/11s；[smoke metadata](../../runs/cnn_path_v2_r3_s11_j471431_smoke/metadata.json)、[CPU eval metadata](../../runs/cnn_path_v2_r3_s11_j471431_smoke_eval/metadata.json) |
| E05 | 3 局 smoke train，checkpoint reload，CPU frozen smoke | Slurm 471439.0/.1 completed，分别 25s/11s；[smoke metadata](../../runs/cnn_path_v2_r5_s11_j471439_smoke/metadata.json)、[CPU eval metadata](../../runs/cnn_path_v2_r5_s11_j471439_smoke_eval/metadata.json) |
| E04/E05 | 正式快照 CPU frozen inference | 每个快照 100 局，结果见 snapshot_evaluations；不是 reserved 确认 |
| 日志建立 | 数字/路径/hash 复核 | 从 metadata、training_summary、snapshot_evaluations 和 CSV 回填；best 文件需与选中 snapshot SHA-256 一致 |

单元测试结果只有在实际运行并有证据后追加；不将测试文件存在视为测试通过。最终独立打包/官方环境兼容性验证和 reserved 确认尚不能从上述 smoke 自动推定完成。

## 6. 维护规则与新实验模板

- 每次正式训练分配唯一 `T1-E##`；快照属于主实验，不另分 ID。提交新配置前先填 planned 条目，结束后更新 completed/failed/cancelled。
- Markdown 只人工/agent 核验后更新，训练进程不隐式改 tracked 文档。配置和运行 metadata 分开链接，CLI override 以 expanded config 为准；记录 source_commit、source_hash、config hash，脏工作树运行另行说明。
- 保留失败和取消记录，不覆盖历史最佳。缺失数据写 N/A 和原因，不填猜测数字。数值展示允许舍入，原始 JSON 保留精度。
- 每轮只改一项；结论区分“观测”“假设”和“已验证”。不以 mean_reward、final loss 或单局满金币选模。
- 更新本文件时不提交 Slurm 输出、runs/、其他队员修改或 `.vscode/settings.json`。checkpoint/产物单独归档，并保留配置及 SHA-256。

### T1-E__ — 标题（planned/running/completed/failed/cancelled；可信/受污染）

- 假设与唯一改动：
- UTC 开始/结束、Slurm job/state/elapsed、训练/评估耗时：
- source commit/hash、配置路径/hash、expanded metadata：
- feature ID/通道变化、network ID、reward ID/完整权重差异：
- safety version/mode/horizon/fallback、动作 mask：
- agent/environment seed、设备/依赖、初始化或父 checkpoint/hash：
- 超参数、n-step/D4/PER/dueling、epsilon 日程、预算/实际 steps/rounds/updates、停止原因：
- smoke train/reload/CPU inference 与单元测试证据：

| 快照 steps / hash | 评估 seeds / 局数 | 捡满率 | 平均金币 | coins/100 steps | 成功完成步数 | WAIT/折返 | 长等待/长往返 | CPU p95/max |
|---|---|---|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

- 最佳/final 路径及 SHA-256、选模依据、逐局产物链接：
- 观测、异常/数据限制、回放诊断、尚未验证的解释：
- 与可信基线的比较、保留/回退决定、下一项单因素实验：
- reserved 确认（未运行/通过/失败；不得用于选模）：
