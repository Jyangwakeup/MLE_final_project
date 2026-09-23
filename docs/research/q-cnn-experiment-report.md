# Q-learning 与 CNN Agent 实验整合报告

主要作者：Ji（提交前确认）
公开仓库 URL：`<REPOSITORY_URL_TO_CONFIRM>`

## 1. 范围与评价协议

本文整合本人负责的表格型 Q-learning 与 CNN Double DQN 路线。两条路线均以冻结、关闭探索的
评估结果选模，而不以训练 reward 或单局回放代替。Task 1 为 `coin-heaven` 的 50 金币导航；
Task 2 为 `classic` 的箱子、隐藏金币与安全放弹。开发集使用固定 20 seeds；独立确认集不参与
选模。CPU `act` 的课程上限为 500 ms。

“平均金币”是单局得分/收集金币数；“捡满率”表示收集该任务全部金币的局比例；“主验证”是
候选固定后才运行的未参与选择集合。三者不混用。

## 2. 方法

### Q-learning（主要作者：Ji）

主线使用 Watkins Double Q(lambda)、每动作 tile coding、`continuous-v2` 84维客观状态、
gamma 0.95、lambda 0.8 与 `survival-mask-v1/all`。r20 奖励保留安全放弹、金币/箱子信用和
窄条件 WAIT 信号。比较过连续历史、crate 三值量化、grouped tiles、反循环奖励和团队示范的
离线 1-step Double Q 更新。

### CNN Double DQN（主要作者：Ji）

CNN 路线从空间棋盘特征学习动作价值，保留完整六动作输出和安全动作 mask。path CNN 使用
17通道棋盘/路径/历史输入；distilled CNN 使用团队自有、冻结的 Task 1 teacher 生成软 Q 目标，
学生最终推理不加载 teacher 或示范数据。蒸馏是团队内部的知识迁移实验，报告中应明确说明其
数据来源与边界。

## 3. Task 1 结果

| 路线/候选 | 冻结结果 | 结论 |
|---|---|---|
| 早期 path CNN r3 | 最佳 13.35 平均金币 | 输入 bug 修复前后均不足。 |
| path CNN r5 | 17% 捡满、41.78 平均金币 | 明显改善但未通过。 |
| distilled CNN global | 开发集 97% 捡满、49.96 平均金币；reserved 100 局 96%、49.84 | CNN 主线的 Task 1 最佳。 |
| optimized Double Q(lambda) | 开发集与独立 100 局确认：96% 捡满、48.95 平均金币 | Q-learning 主线通过 Task 1。 |

CNN 与 Q-learning 的 CPU frozen-inference 均在课程 500 ms 限制以内；CNN Task 1 best 的
p95/max 为 8.39/15.55 ms，Q(lambda) 的独立确认结果也满足 50/500 ms 门槛。

## 4. Task 2 结果与失败分析

| 路线/最佳 checkpoint | 平均金币 / 9 | 平均炸箱 | 长 WAIT | 长往返 | 安全结论 |
|---|---:|---:|---:|---:|---|
| Q(lambda) r20 175.2k | **3.95** | 53.45 | 30% | 75% | 自杀 0%、放弹存活 100%，但未达标。 |
| Q crate quantized | 1.35 | 未超过基线 | 超限 | 超限 | 11/19 gates，归档。 |
| Q history input | 1.65 | 29.70 | 超限 | 85% | 修复历史输入未解决主因。 |
| Q continuous-v4 r12 | 0.95 | 18.40 | 0% | 35% | 历史/反循环奖励消融，未达标。 |
| Q grouped tiles | 0.00 | 0.00 | 5% | 65% | 5/19 gates，未改善 r20。 |
| Q team-demo 50k | 3.75 | **56.90** | 65% | 75% | 12/19 gates；提高炸箱但恶化 WAIT。 |
| distilled CNN D02 主验证 | 2.60 | 44.15 | 见日志 | 见日志 | 自杀/非法动作/零放弹均为 0%，安全与炸箱迁移通过；全金币目标未通过。 |

Task 2 的共同瓶颈不是自杀或无效放弹：两条路线均能安全炸箱，但在炸开路径后无法稳定重选
金币目标，导致跑满 400 步、长 WAIT 或往返。Q-learning 的高维状态组合使 tile 泛化不足；
CNN 的 teacher 蒸馏主要传递 Task 1 导航能力，未充分覆盖箱子后的长期目标切换。

最早的 `q_learning_agent` 和 `cnn_double_dqn_agent` 仅为历史 Task 1 基线，未作为本轮
Task 2 候选；它们以及所有专用变体的代码、配置和日志均已保留在归档目录，避免把未达标实验
误作最终提交模型。

## 5. 结论与提交建议

两种模型满足课程“至少两个不同 ML 模型”的比较要求，并保留了成功与失败证据。当前本人路线
中，Q(lambda) 是 Task 1 最可靠的已确认候选；distilled CNN 是 CNN 路线中实验最完整的候选。
两者都不应被表述为完成 Task 2 的全金币目标。

正式比赛 ZIP 只能包含一个 agent 目录，且需由团队对所有成员模型统一选择。本次仅整理本人
路线，未自动选择、复制权重、生成 ZIP 或修改队友代码。课程 PDF 报告应以本文为源稿补充团队
项目规划、全体作者署名、文献、最终公开仓库 URL 与最终候选的独立打包验证。
