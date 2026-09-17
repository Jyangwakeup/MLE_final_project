# CNN Distilled Double DQN 实验日志

动态结果只记录在本文件。Task 1 验收为开发集 seeds 10001–10005、每 seed 20 局
捡满率至少 90%，随后 reserved seeds 21000–21099 每 seed 一局仍至少 90%。选模按
捡满率、平均金币、成功完成步数（越低越好）；CPU `act` max 必须小于 0.5 秒。

## 固定契约

- feature：`board-path-history-v2`，17 通道，见 `README.md`；
- teacher：`double_dqn_continuous_v2_agent/final.pt`，当前核验 SHA-256
  `74cf3fca33acb59f003949006c7401a5a9f58c1785dd8b294b548bd7bd307715`；
- 蒸馏：masked KL、temperature 1.0、最多 30 epochs、patience 5；
- 微调：1-step DDQN、r5_conditional_loop、safety off、BOMB disabled、lr 5e-5、
  batch 64、replay 20k、warmup 5k、target sync 2k、KL weight 1.0；
- 三候选：global、action-aligned、action-aligned+D4。

## T1-D01 — completed：teacher 蒸馏与 50k 微调

唯一主变化是从 E05 的随机初始化 CNN 转为团队 teacher 蒸馏，并比较动作对齐/D4
结构。训练数据 seeds 6000–6079，validation 6080–6099；结构筛选 12000–12019。
E06 已证实 4-step 不如 E05，故本轮回到 1-step，不加入 PER/Rainbow。

实现 smoke（不作为性能证据）：

- teacher capture seed 6000：115 rows，board `(115,17,17,17)`，每帧 self 标记数为 1；
- synthetic two-seed dataset 上 global head 1 epoch 可生成安全 checkpoint；
- 3 局 train 产生 1,200 transitions，checkpoint 可用 `weights_only=True` 加载；
- 冻结 reload 1 局成功；本机 30 次纯 forward：global max 0.0257s、
  action-aligned max 0.0227s、D4 orbit max 0.1739s，均低于 0.5s。

Slurm **471456 COMPLETED**，elapsed 01:41:25，`students` 单张 GTX 1080 Ti。
source commit `4a69f62f592bdc8f9a0d46a38680cbce202ddb1a`（含未提交的新实现），
runtime source hash `a4a69975a01b4e973f2108a6963fb1b74009fb4527a2b9c312e11b378cbd097d`。
数据集 12,825 rows/100 seeds，SHA-256
`9eacec7294201ca67dd266879930437cf0287eac924dfdf389d451f8d2a8dfda`。
三候选均训练 30 epochs，最佳 validation KL 分别为 0.0260455/0.0220214/0.0541214。

| 候选 | 12000–12019 | 开发 100 局 | reserved | 结论 |
|---|---|---|---|---|
| global + teacher | 95%，49.75 金币 | 97%，49.96 金币 | 96%，49.84 金币 | 最佳；保留 pretrained |
| action-aligned + teacher | 85%，48.60 金币 | 未运行 | 未运行 | 未入选 |
| action-aligned + teacher + D4 | 90%，49.70 金币 | 未运行 | 未运行 | 未入选 |

| global checkpoint | 开发捡满率 | mean coins | 成功完成步数 | WAIT | 立即折返 | 长等待/长往返 | CPU p95/max (s) |
|---|---:|---:|---:|---:|---:|---|---|
| pretrained | 97% | 49.96 | 130.6082 | 4.24% | 5.81% | 3%/0% | 0.00836/0.01435 |
| 10k 微调 | 80% | 44.37 | 130.2500 | 32.70% | 8.31% | 17%/2% | 0.00833/0.01540 |
| 25k 微调 | 72% | 43.02 | 128.7500 | 39.24% | 8.60% | 23%/3% | 0.00841/0.01379 |
| 50k 微调 | 55% | 37.21 | 127.4545 | 61.78% | 2.86% | 45%/0% | 0.00849/0.01557 |

独立 reserved 确认：96/100 捡满，mean coins 49.84，成功完成步数 131.2708，
coins/100 steps 35.09365；WAIT 0.0775%、立即折返 8.6185%、长等待/长往返均 0%，
CPU p95 0.008388s/max 0.015553s。开发和 reserved 两组均通过 90% 门槛；
仅验证训练 seed 11，不推定其他训练 seed 稳定性或 Task 2 能力。

最佳 [best_task1.pt](../../runs/cnn_distilled_t1_j471456/best_task1.pt) SHA-256
`8c148fee3fcfe8dd19622cdc26d1f0e191d5f7512fb3cbf1d6ad38ee5f77425e`。
原始 [selection.json](../../runs/cnn_distilled_t1_j471456/selection.json)、
[pretrained_selection.json](../../runs/cnn_distilled_t1_j471456/pretrained_selection.json)，
三候选逐 epoch 曲线在同目录 `pretrained_*.history.json`，逐局记录在对应 dev/reserved run。

结论：蒸馏使 Task 1 从 E05 的 17% 提升到 97%，但较低 validation KL 不保证更好
闭环导航。在线 r5 TD 更新即使带 KL=1 仍增加 WAIT 并破坏策略；因此不部署 50k final。
下一步保护该 Task 1 checkpoint，另立 Task 2 迁移实验，不覆盖本轮验收模型；
reserved 集已经使用，不能再次把它当作新实验的未见确认集。

发布验证：最佳 checkpoint 已复制到本 agent 的 `final.pt`；生成的
`final-project-agent-code.zip` SHA-256 为
`ad9448888e6271554ee9c3d5d3f0cb882932c912c84c8c325ab9fc7f3a43ace2`。
ZIP 在仅含官方顶层框架、assets、三个 random agents 和打包 agent 的临时目录中，
以 classic/3 局、无 GUI、无 teacher/数据集环境变量运行成功。

## T2-D01 — completed/failed：Task 1 KL 保留的 Task 2 迁移

Slurm job **471457** 因父 checkpoint 的 r5/r7 加载契约错误在训练前终止，不作为性能
实验。修复后的 job **471458** 于 2026-09-16 00:45:43–09:56:27 完成，elapsed
09:10:44；两条训练链均使用 `students` 分区单张 GTX 1080 Ti。source commit
`4a69f62f592bdc8f9a0d46a38680cbce202ddb1a`，runtime source hash
`839539c5329c03aeaf5b09ed4ed97185928071829afe51e4eecbbd13eeb84bf3`。

- 迁移链：T1-D01 `global` checkpoint 初始化，继续使用原 Task 1 teacher 数据作导航
  KL 正则，weight 2.0；这不是 Task 2 teacher 蒸馏。
- 从零对照：相同 global CNN 随机初始化，KL=0；其余训练预算相同。
- 两链均为 Task 2、r7、safety all、4-step、epsilon 0.30→0.05/120k、seed 11，
  实际各 200,000 stage action steps/500 局。每个 checkpoint 在开发 seeds
  10000–10019 各一局冻结评估。

下表采用“指标为行、checkpoint 为列”的固定格式；百分比均来自同一 20-seed Task 2
冻结评估。`T1 分数`另在 Task 1 场景评估，不能与 Task 2 金币数混为一项。

| 指标 | 迁移 50k | 迁移 100k | 迁移 150k | 迁移 200k | 从零 200k |
|---|---:|---:|---:|---:|---:|
| T1 平均分 | 50.00 | 50.00 | 50.00 | **50.00** | 2.40 |
| T2 平均金币/得分 | 1.45 | 1.90 | 1.20 | 2.00 | **2.50** |
| T2 得分中位数 | 0.50 | 1.50 | 0.00 | 0.50 | **2.00** |
| 全部金币率 | 0% | 0% | 0% | 0% | 0% |
| 0 金币率 | 50% | 45% | 55% | 50% | **35%** |
| 平均炸箱 | 24.35 | 29.05 | 25.05 | 34.35 | **41.85** |
| 平均炸弹数 | 10.55 | 16.70 | 11.80 | **18.45** | 14.25 |
| 零放弹局率 | 25% | **10%** | 5% | 20% | **10%** |
| 已结算炸弹存活率 | 100% | 100% | 100% | 100% | 100% |
| BOMB 动作率 | 2.64% | 4.18% | 2.95% | **4.61%** | 3.56% |
| WAIT 率 | 36.96% | 48.83% | 46.30% | 55.20% | 54.50% |
| 长 WAIT 问题率 | 40% | 50% | 50% | 75% | 85% |
| 长往返问题率 | 60% | 30% | 35% | 40% | 60% |
| 立即反向率 | 47.34% | 23.55% | 33.78% | **16.75%** | 20.78% |
| 存活率 | 100% | 100% | 100% | 100% | 100% |
| 跑满 400 步率 | 100% | 100% | 100% | 100% | 100% |
| CPU act p95 / max (ms) | 8.61 / 14.49 | 8.64 / 14.33 | 8.62 / 14.93 | 8.59 / 13.64 | 8.68 / 14.54 |

迁移 200k 在 T1 保留、金币、炸箱、自杀、炸弹存活、非法动作和 CPU 时间上达标，
但零放弹局为 20%，超过 ≤10% 门槛，因此 **D01 未通过**；流水线按预注册协议没有训练
seeds 22/33，也没有运行主验证。旧 `selection.json` 生成时，多 seed analyzer 没有聚合炸弹
字段，曾把 `survived_bomb_rate` 空值误判为失败；逐局数据复核为 368/368，现已修复聚合器
并增加回归测试。这个修复不改变零放弹局失败结论。

主要诊断是策略两极化，而非不会安全放弹：迁移 200k 有 4/20 局 WAIT≥99.25%、
0 炸弹、0 箱子，另一些局可炸 80–98 个箱子；整体 WAIT 55.2%、长 WAIT 75%。
从零模型的 Task 2 数值略高，却把 Task 1 平均分从 49.85 降至 2.4，不能作为递进课程
候选。迁移 200k checkpoint SHA-256
`d2ef48b8aca4db007a18058f85ba33629c894d4458ffaae49e414e0a4254bbb6`；从零 200k 为
`3d07eb007ca2fb7b204fddd576f972890798e0bbecaf6928146f171eba3786be`。原始结果见
[selection.json](../../runs/cnn_distilled_t2_j471458/selection.json) 及各 `*_t1/*_t2` 目录。

## T2-D02 — completed/passed：窄条件 avoidable-WAIT shaping

假设：D01 的失败由部分回合进入 WAIT 吸引子造成；当 WAIT 本身安全，同时存在安全的
金币/箱子前沿推进移动或能炸到箱子的安全 BOMB 时，额外奖励 `-0.04`。危险 WAIT、无安全
推进动作的 WAIT 不处罚，推理动作仍完全由 CNN+DDQN Q 值与既有 safety mask 决定。

唯一训练变化是 `task2_avoidable_wait_penalty: 0 → -0.04`。网络、17 通道、r7、
safety all、4-step、KL=2、teacher 数据、探索日程、seed 11 和 200k/500 局预算保持 D01
迁移链不变。先比较同一 20-seed 开发集的全部快照；seed 11 通过联合门槛后才自动训练
22/33 并运行 seeds 11000–11099 主验证。另有只读 Q/mask 诊断，不计入正式 CPU 延迟。

实现验证：17 项 analyzer/CNN targeted tests 通过；3 局 CPU smoke 完成 1,200 steps，
每局触发 48–53 次窄条件 WAIT 处罚；checkpoint reload 的冻结 CPU 评估成功。正式配置
[task2_cnn_wait_shaping.json](../../experiments/configs/task2_cnn_wait_shaping.json) SHA-256
`2f810d841895c9e00fbda1966a3aecdc9017fd962bb330d868f59111bc44e4b2`。Slurm job
**472342** 已提交；D01 卡死/正常 seed 的只读 Q/mask 对照诊断为 CPU job **472343**。
job **472342** 于 2026-09-17 完成，elapsed 16:54:33，单张 GTX 1080 Ti。三条训练
seed 均通过严格安全、Task 1 保留和 D01 炸箱下限；依预注册流程，选定 seed 22 的 200k
checkpoint 后运行 100-seed 主验证。

| 指标 | D01 迁移 200k | D02 seed 11 | D02 seed 22（选定） | D02 seed 33 | D02 主验证 100 seeds |
|---|---:|---:|---:|---:|---:|
| checkpoint steps | 200k | 200k | 200k | 200k | 200k |
| T1 平均分 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| T2 平均金币 | 2.00 | 2.80 | **4.20** | 3.65 | 2.60 |
| 平均炸箱 | 34.35 | 54.20 | **67.25** | 57.35 | 44.15 |
| 平均炸弹数 | 18.45 | 22.40 | **27.35** | 23.00 | 18.89 |
| crates / survived bomb | 1.87 | 2.42 | **2.48** | 2.49 | 2.34 |
| 零放弹局率 | 20% | 0% | 0% | 0% | 0% |
| WAIT 率 | 55.20% | 17.26% | **14.90%** | 25.69% | 22.56% |
| 长 WAIT 问题率 | 75% | 待归档 | 45% | 待归档 | 43% |
| 自杀 / 炸弹存活 | 0% / 100% | 0% / 100% | 0% / 100% | 0% / 100% | 0% / 100% |

主验证的所有硬门槛通过：Task 1 mean 50（保留率 100.3%）、Task 2 crates 44.15
（高于 D01 的 34.35）、零放弹 0%、自杀 0%、已结算炸弹存活 100%、非法动作 0%、CPU
act p95/max 为 8.88/19.11 ms。它不是 Task 2 的“捡满金币”结果：100 局平均只捡 2.60
枚，all-coins rate 仍为 0%；当前通过的是 Task 2 的安全放弹/炸箱迁移门槛。

发布 checkpoint：
[`best_task2.pt`](../../runs/cnn_distilled_t2_wait_j472342/best_task2.pt)，SHA-256
`912d2f7ae7e53c075e4af238b1e648b69c836beba0467263e06f51947e359569`；原始选择记录为
[`selection.json`](../../runs/cnn_distilled_t2_wait_j472342/selection.json)。

## T2-S01 — completed：20-seed Safety Mask 只读诊断与 paired safety-off

使用 D01 迁移 200k checkpoint，在 Task 2 seeds 10000--10019 各评估一局。诊断记录
physical/safe BOMB 可用性、BOMB veto、raw Q argmax 被 mask 拦截的动作类型、
raw-BOMB veto 和 fallback；不覆盖 Q 值、不改变动作，因而不是训练或性能消融。

先导 job **472343** 只覆盖 2 seeds，用于验证日志格式，不代替正式 20-seed 结果。
正式 CPU job **472344** 已提交，结果按以下表格回填。若 BOMB veto 和 raw-BOMB veto
均不超过 5%，保留 safety all；
超过阈值才触发同 checkpoint 的 all/off 配对冻结评估。
条件 job **472347** 依赖 472344：仅在任一正式 veto rate 大于 5% 时才运行同一
checkpoint、同一 20 seeds 的 safety-off 评估，否则只记录 `skipped`。override 仅存在于
冻结诊断进程；不会写回 checkpoint，也不能用于训练或发布。

先导的刻意选样 seeds 10001/10004 共 800 decisions：BOMB veto 13/584（2.23%），
raw-BOMB argmax veto 12/48（25%），raw argmax 任意动作 veto 8.75%，fallback 0；其中
一局 WAIT>=95%。由于该样本刻意包含卡死/正常回合，不能用 25% 估计总体发生率，
但它证明正式反事实诊断有必要。

| 指标 | D01 200k + safety all |
|---|---:|
| physical BOMB 可用次数 | 5,795 |
| safe BOMB 可用次数 | 5,362 |
| BOMB veto rate | 433/5,795（7.47%） |
| raw argmax veto rate | 1,044/8,000（13.05%） |
| raw-BOMB argmax veto rate | 421/790（53.29%） |
| safety fallback 次数 | 0 |

正式 job 472344 于 15:03 完成 8,000 decisions；WAIT 率 55.20%，7/20 局 WAIT>=95%。
两个 BOMB veto 指标均超过预注册 5% 阈值，因此 472347 已触发 safety-off 配对冻结
评估。这里的 53.29% 只说明 mask 经常改变 raw-BOMB argmax，不代表关闭 mask 更优；
只有 off 平均炸箱提高至少 10% 且自杀率仍 <=5%，才会判定值得检查 false negatives。

| 配对指标 | safety all | safety off |
|---|---:|---:|
| T2 平均金币 | 2.00 | 0.00 |
| 平均炸箱 | 34.35 | 3.55 |
| 平均炸弹数 | 18.45 | 1.00 |
| 零放弹局率 | 20% | 10% |
| 自杀率 | 0% | 90% |
| 已结算炸弹存活率 | 100% | 10% |
| 存活率 | 100% | 10% |
| 记录的 WAIT 率 | 55.20% | 91.82% |

pair job **472347** 于 13:28 完成。虽然关闭 mask 后物理 BOMB 均可用，实际策略并未
因此多放有效炸弹：平均 crates 从 34.35 降至 3.55（-89.7%），并触发 90% 自杀，远不满足
“crates +10% 且 suicide <=5%”的进一步检查门槛。结论是保留 `survival-mask-v1/all`；
它会修正大量 raw-BOMB Q argmax，但在该 checkpoint 上是必要安全约束，而非少炸箱的主因。
配对原始产物为
[`status.json`](../../runs/cnn_t2_d01_maskpair_j472347/status.json)。

## T2-D03 — skipped：r7 potential 奖励对照

仅当 D02 未通过严格门槛时运行。相对 D02 的唯一变化为
`r7_safe_credit_sparse -> r7_safe_credit_potential`；17 通道、global CNN、n=4、
avoidable-WAIT -0.04、safety all、KL=2、探索日程、训练 seeds 和 200k/500 局预算
均保持不变。其目的是检验 sparse 奖励在可见金币状态下的塑形真空，而不是同时调网络。
Slurm job **472345** 读取 D02 的完整 selection 后确认 D02 已通过三 seed 和主验证，
于 00:00:04 正常跳过；不重复训练。

| 指标 | D02 最佳 | D03 50k | D03 100k | D03 150k | D03 200k |
|---|---:|---:|---:|---:|---:|
| T1 平均分 | 50.00 | — | — | — | — |
| T2 平均金币 | 4.20 | — | — | — | — |
| 平均炸箱 | 67.25 | — | — | — | — |
| crates / survived bomb | 2.48 | — | — | — | — |
| 零放弹局率 | 0% | — | — | — | — |
| WAIT / 长 WAIT | 14.90% / 45% | — | — | — | — |
| 自杀 / 炸弹存活 | 0% / 100% | — | — | — | — |

## T2-D04 — skipped：5-step 炸箱信用

仅当 D02 和 D03 均未通过时运行。先按冻结评估排序选择 sparse/potential 中较好的
reward，然后只把 `n_step: 4 -> 5`。放弹通常在第五个环境步收到真实
`CRATE_DESTROYED`，故该实验检验 BOMB transition 是否需要直接覆盖爆炸事件；不扩大到
n=8/10，不改变 KL 或 bomb bonus。
Slurm job **472346** 依赖 D03 job 472345；D02/D03 任一通过严格门槛时自动跳过，
否则从二者 seed-11 冻结结果中选择奖励，再运行唯一的 n-step 改动。
job **472346** 于 00:00:03 正常跳过，因为 D02 已通过。

| 指标 | 最佳 n=4 | n=5 50k | n=5 100k | n=5 150k | n=5 200k |
|---|---:|---:|---:|---:|---:|
| T1 平均分 | 待选择 | 待训练 | 待训练 | 待训练 | 待训练 |
| T2 平均金币 | 待选择 | 待训练 | 待训练 | 待训练 | 待训练 |
| 平均炸箱 | 待选择 | 待训练 | 待训练 | 待训练 | 待训练 |
| crates / survived bomb | 待选择 | 待训练 | 待训练 | 待训练 | 待训练 |
| 零放弹局率 | 待选择 | 待训练 | 待训练 | 待训练 | 待训练 |
| WAIT / 长 WAIT | 待选择 | 待训练 | 待训练 | 待训练 | 待训练 |
| 自杀 / 炸弹存活 | 待选择 | 待训练 | 待训练 | 待训练 | 待训练 |

研究依据和 safety 消融阈值见
[`docs/research/cnn-task2-reward-safety.md`](../../docs/research/cnn-task2-reward-safety.md)。
`useful_bomb_per_crate` 增益和 KL 退火仍是条件分支；D02--D04 给出证据前不激活。
