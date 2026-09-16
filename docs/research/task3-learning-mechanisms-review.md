# Task 3 当前学习机制审阅

日期：2026-09-17。范围：`79d1e87` 套餐的源码与所选 checkpoint，只读分析；没有训练、修改配置或源代码。课程与实验约束见 `PROJECT_REQUIREMENTS.md`、`CONTEXT.md`、`docs/adr/0004-use-controllable-survival-for-bomb-safety.md`、ADR 0005/0006 和 `experiments/task3_plateau_stopping.json`，本笔记不复制其完整协议。

## 判断

已找到应优先修复的确定实现错误：**训练回调读取放弹前 old_state 时清除了刚记录的 own-bomb obligation，使训练的放弹后安全边界与冻结推理不一致。** 所以当前最小路径是修复生命周期、做回归并从干净 Task 2 父模型重新训练原 r7 套餐；奖励、Replay 和特征调整退为修复后的独立实验。不能把现有失败仅归因于奖励或小样本选择，也不能断言修复一定通过 Task 3。

## 1. 奖励确实偏重资源，而非官方分数等比例缩放

`r7_safe_credit_sparse` 的金币为 3、杀敌为 5、炸箱为 0.2；安全且预测能炸箱的 BOMB 另获最多 0.6，故炸箱具有预测信用与实际信用。源码是 `agent_code/team_agent/rewards.py:149`、`:292`；官方金币/杀敌为 1/5，见 `settings.py:55`。因此学到“两枚金币胜过一次击杀”并不违背当前即时事件奖励，而与官方价值比较不同。折扣、风险、终止与其他事件仍影响真正 Q 值，不能把该比例直接当行为因果证明。

该版本没有敌距、追击或封堵势；名字虽叫 sparse，仍有箱区与危险势（`rewards.py:256`、`:329`）。危险势是 `gamma*Phi(next)-Phi(old)`，不能直接等同于额外累计危险惩罚。

建议单变量候选为金币保持 3、击杀由 5 提至 15，恢复事件收益的官方相对比例；它不保证成功，也会增加 TD 尺度与探索攻击动机。需要冻结安全与保留门槛检验。优先于削弱安全 mask 或随意加追击奖励。

### 奖励迁移不是改一行配置

`agent_code/learning_common/runtime.py:113` 拒绝配置 reward ID 与 checkpoint 不一致；`:191` 同时检查 ID 与 reward specification。因此需要新 reward ID 与显式、有证据的迁移路径，不能偷偷改原 r7 定义或重标旧 Task 3 checkpoint。

最干净的实验分支起点是原始 Task 2 父模型：Task 1/2 没有对手，若唯一修改是 `killed_opponent`，其已有奖励和 n-step return 数值不变，故可在验证无杀敌事件及其他 reward 项完全相同后保留父 Replay，并保留原 Task 2 teacher。新 Task 3 Replay 必须从空开始。旧 Task 3 Replay 存的是折扣聚合标量而非原始事件分解（`agent_code/dqn_agent/model.py:175`），无法仅靠缓存标量精确重算新奖励。原始父模型、旧实验和迁移元数据应保持不可变。

## 2. Replay 是约 24/24/16，而不是 16/32/16

`model.py:69` 将所有父任务池合并随机抽样，`model.py:109` 抽 48 个父样本、16 个 Task 3 样本。只读检查所选 seed 11/22/33 checkpoints，Task 1 与 Task 2 均各有 20,000 条，所以期望配额是 **24/24/16**；每个 batch 的父任务内部数量仍随机。ADR 0005 提到的 16/32/16 是备选干预，并未由当前 `parent_fraction=0.75` 自动实现。

三个 checkpoint 的 teacher 参数逐 tensor 与对应 Task 2 parent policy 相等。代码在跨任务时取 `checkpoint['policy']`、同任务续训时取原 teacher，见 `model.py:440`。旧版错误地继承 Task 1 teacher 的问题不能再用来解释这批失败。

蒸馏只施加于父 Replay 行，计算 masked softmax KL，温度 1、系数 2；TD 对所有行计算（`model.py:361`）。这意味着保留并非硬约束：父池外状态没有蒸馏约束，共享参数仍可能改变 Task 2 行为。也不能声称 teacher 在 Task 3 状态直接强迫资源策略，因为这些行不受 KL。合理待验证假设是父 TD/KL 与稀疏 Task 3 TD 的梯度冲突、父样本覆盖不足或行为误差累积；需分任务 TD、KL、梯度方向及实际动作分歧诊断，而非仅比较混合 loss。

## 3. 84 维表示更偏导航，战斗状态覆盖很低

前 70 维含每动作最近敌人距离变化、能否到达敌人、当前放弹爆炸线内敌人数等（`agent_code/team_agent/feature_system/continuous_v1.py:22`）。新增 14 维是自身上个动作、金币目标延续、返回上一位置（`continuous_v2.py:14`），没有对手动作历史。

没有显式绝对敌距、对手出口/逃生可达空间、对手移动趋势、敌人身份或逐敌人表示；`common.py:235` 聚合最近敌人距离，`:244` 的 threatened 只是当前坐标在爆炸线内，并非预测击杀。标准 17×17 棋盘的一格距离增量为约 0.00347（`common.py:279`），网络理论上可以放大，尺度小本身不证明特征无效。

用 `miniforge3/envs/mle/bin/python`、`torch.load(weights_only=True)` 只读所选 checkpoint 的 Replay，按 `task_ids` 统计 `states[:,67]>0`（can-bomb 且当前爆炸线内有敌人）得到：

| seed / 选中轮次 | Task 3 缓存行数 | BOMB 动作比例 | 当前可威胁敌人状态比例 | Task 3 累计动作数 |
|---|---:|---:|---:|---:|
| 11 / 150 | 20,000 | 13.67% | 0.610% | 52,124 |
| 22 / 150 | 20,000 | 13.38% | 0.795% | 51,538 |
| 33 / 50 | 17,918 | 13.03% | 0.742% | 17,918 |

来源路径模式：`runs/task3_plateau_ddqn_cv2_r7_s{seed}_c{rounds:04d}_79d1e87/checkpoints/final.pt`；父路径与 hash 见 plateau 协议。每 batch 仅 16 条 Task 3，因此上述状态期望仅约 0.10–0.13 条。这是**战斗接触稀疏的代理指标**，不是成功杀敌样本比例；不能推导其余样本无战斗价值，也不能据此立即启用优先 Replay。首选补充统计实际击杀前缀、放弹成功率、每次敌人接触的出手率，再判断需要战斗样本重采样还是表征改进。

若后续确实需要特征变体，添加原始战斗事实（距离、出口数、对手相对位移），仍由网络学排序；不能把“最佳追击/杀敌动作”直接编码为规则绕过学习。新维度需要明确权重迁移和独立实验，成本高于先做奖励消融。

## 4. n-step 5 没有明显 off-by-one 错误，但并不包办战斗信用

`NStepAccumulator` 从放弹动作当步开始聚合五个 transition，终止时冲刷；每条记录自己的 `steps`，target 使用 `gamma**steps`（`agent_code/learning_common/n_step.py:16`、`model.py:361`）。因此父 Replay 主体 n=4、Task 3 主体 n=5 混存是支持的，不是错误。

`environment.py:167` 先执行动作，再更新炸弹；`settings.py:49` 初始计时 4，放弹同一 transition 降为 3，随后降为 2、1、0，第五次 transition 才爆炸（`environment.py:203`）。n=5 因而可直接给最初 BOMB 首次爆炸击杀信用，系数为 gamma^4。第六次 transition 的残留火焰击杀不直接进入该 BOMB 五步 return，仍需 bootstrap；这属于有限窗口边界，不足以宣称 bug。进一步增加 n 会改变偏差/方差和探索轨迹影响，不应作为未诊断时的第一改动。

## 5. 已确认训练 own-bomb obligation 被历史状态读取清除

探索动作在 `callbacks.py:196` 先经过 `safety_decision`，`:228` 只从最终 mask 内随机选择，未发现 epsilon 分支直接绕过 mask。

但存在明确边界：`agent_code/team_agent/safety.py:253` 只有 v1 未进入 physical fallback 才运行 v5；当 v1 已没有安全动作时，`:310` 直接返回 physical fallback，`robust_guarantee_loss` 使用默认 False。所以**零 guarantee-loss 不等于 own-bomb 期间从未退到 physical Q**。另在无安全非 BOMB 替代时，`:286` 并不禁止未证明的 BOMB；该逻辑与“只在存在安全替代时 veto”契约一致，却不构成全局安全证明。

确定调用链如下：

1. `callbacks.py:326` 在 act 末尾调用 `record_selected_action`；`action_history.py:46` 对 BOMB 设置位置和 pending=True。
2. 训练事件回调进入 `_transition`，`train.py:58` 无条件以放弹前 `old_state` 调用 `own_bomb_history_for_state`；即使 `_features_for` 缓存命中，该调用仍存在。
3. `action_history.py:83` 再次调用 `action_history_for_state(old_state)`；旧状态的 `self[2]`（放弹能力）仍为 True，于是 `:27–29` 把刚记录的位置与 pending 清空。
4. 后续 next-state mask 和下一次 act 不再进入 own-bomb pending 分支；v5 放弹后持续鲁棒控制因此失效，相关 guarantee-loss/collapse 计数也可能失去触发前提。冻结推理没有该训练事件回调，所以可呈现训练大量自杀而评估零自杀。

主分析提供的实例为 seed 22、c0150 的 `timing.jsonl`，round 106：step 259 BOMB；260–263 own_bomb_pending=False、visible=False、timer=null；263 自炸。源码链足以确认生命周期错误；单条轨迹及训练总体差异支持其实际发生，但各项性能损失有多少由该错误造成仍需修复后对照。

修复应使历史查询无副作用或只允许按状态时间单调前进，明确“选择动作后记录”与“读取动作前状态”边界，不能简单在所有 BOMB 后永久保留 pending。回归必须覆盖真实 act → game_events_occurred(old,new) → next act 顺序、Replay next mask、残留火焰清除、终止/新回合与 checkpoint 恢复；单独测试 safety_decision 的静态状态不足以防止此错误。

## 后续最小路径

1. 固定当前失败结论，修复 own-bomb 历史查询/更新边界，先做真实训练生命周期回归和旧自杀轨迹复现。当前用户请求为分析，本次没有实施修复。
2. 从相同原 Task 2 父模型起步，以原 r7、v5、n=5、84维、teacher 与 Replay 重跑开发实验，验证训练安全边界确已恢复；旧 Task 3 权重/Replay 不作为干净起点。按新实验协议保留独立确认集。
3. 修复后战斗仍不足，再独立测试 kill=15；Task 2 保留仍不足则独立测试固定 16/32/16。不要一开始同时修 bug 并修改奖励与 Replay，否则无法解释收益。

以上建议不改变现有失败实验的门槛或结论，不证明任何候选必能通过 Task 3。
