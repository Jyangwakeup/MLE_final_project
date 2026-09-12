# 往届参考仓库的对战 Agent 训练流程核查

核查日期：2026-09-12
研究对象：`IMPLEMENTATION_GUIDE.md` 唯一明确列出的往届仓库，在固定提交 [`a7fe5041b02548ce4438e502ca3adb11576bae75`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/tree/a7fe5041b02548ce4438e502ca3adb11576bae75) 上核查。
证据范围：该仓库自己的 README、源码、配置与 Git 提交记录；没有把第三方讲解或作者“第三名”的自述当作算法效果证据。

## 结论先行

往届仓库能够证明的是一个**在线、逐步更新的强化学习闭环**：游戏状态先被大量手工规划逻辑压缩成局部类别特征，训练时用 ε-greedy 选动作；每一步和终局回调根据官方事件及自定义事件塑形奖励；最终 DQN 将每条真实转移做旋转/翻转增强后放入经验回放，随后立即抽样更新策略网络，并定期同步目标网络；每 100 局保存策略模型。代码还为“一个训练实例对战若干同名非训练实例”准备了每 500 局重载权重的快照式 self-play 机制。

但该仓库**没有**训练 shell 命令、各阶段局数、对手序列、随机种子、训练日志、实验矩阵或最终选模表。因此，不能声称它按本届课程的“金币导航 → 炸箱 → 弱对手 → 强对手”四阶段实际训练过。仓库中的场景与提交历史只能证明“具备这些入口”和“代码开发大致从表格法演进到 DQN”，不能证明实际训练课程。

另一个容易被目录名误导的结论是：最终目录 `deep_learning_killer` **不是 DQN**，而是单 Q 表的 **SARSA**。它用下一状态经当前 ε-greedy 策略选出的 `a'` 构造 `r + γQ(s',a')`，并把表保存为 JSON；真正的 DQN 候选是 `feature_is_everything`（以及 `dqn`、`dqn_5tile` 变体）。证据见其 [`callbacks.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/deep_learning_killer/callbacks.py#L30-L54) 与 [`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/deep_learning_killer/train.py#L15-L49)。

## 1. 场景递进与对手配置

### 可直接证实的事实

仓库提供四个场景入口：`empty`（无箱、无金币）、`coin-heaven`（无箱、60 金币）、`loot-crate`（箱密度 0.4、40 金币）与锦标赛 `classic`（箱密度 0.75、9 金币）。这组配置确实支持由简单导航逐渐过渡到完整游戏，但配置本身不等于训练记录。该旧版本每局最多 200 步，也与本届的 400 步不同，不能直接移植其训练预算或成绩口径。[`settings.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/settings.py#L6-L35)

命令行允许通过 `--scenario` 选择场景、通过 `--agents` 显式列出至多四个 Agent，并通过 `--train N` 把前 N 个实例置于训练模式；`--my-agent X` 的便捷形式会让 X 对战三个 `rule_based_agent`。因此可以配置单人训练、固定规则对手或多个同名 Agent，但仓库没有给出作者实际运行过的命令。[`main.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/main.py#L102-L124) [`main.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/main.py#L154-L165)

最终 DQN 和表格 Agent 都在每个第 500 局的第一步调用 `setup`，源码注释为“train with itself”。对于非训练同名实例，`setup` 会重新从共享文件加载策略；训练实例则保留内存中的训练网络。因此若用“第一个同名实例训练、其余同名实例不训练”的配置，对手会约每 500 局刷新为已保存快照。这是代码支持的快照式 self-play，不是同步多智能体共同学习。[`feature_is_everything/callbacks.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/callbacks.py#L35-L54) [`feature_is_everything/callbacks.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/callbacks.py#L66-L88)

### 不能由仓库证实的内容

- 没有证据表明作者实际按 `empty → coin-heaven → loot-crate → classic` 依次训练。
- 没有证据表明训练依次使用 `peaceful_agent`、`coin_collector_agent`、`rule_based_agent`。
- 没有每阶段局数、通过门槛、是否继承同一检查点、训练/验证 seed 或对手比例。
- 没有证据表明最终权重来自上述 self-play 配置；代码注释与可配置能力不能替代运行记录。

提交历史可以作为**开发演进线索**：项目先加入 baseline，随后从 DQN 转向 Q-learning，再加入 SARSA、Double Q、SARSA(λ)，之后重新加入 DQN 及 5-tile/最终特征变体；例如提交 [`88e1140`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/commit/88e1140) 的标题是 “no more dqn start with q-learning”，提交 [`4a54224`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/commit/4a54224) 添加 DQN 训练，提交 [`54e75ae`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/commit/54e75ae) 整理出两个 final agents。**推断**：团队是先打磨表格型特征/更新，再回到神经网络近似；这不是场景课程或性能优劣的直接证据。

## 2. 最终 DQN 的状态特征

`feature_is_everything` 并不把整个 `game_state` 或地图直接喂给网络，而是输出 6 个离散槽位：自身四邻格、当前位置、放弹槽位，再 one-hot 成 34 维。前五槽类别为 `block/free/dead/coin/enemy/target`，放弹槽为 `True/False/target/KILL!`。[`callbacks.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/callbacks.py#L7-L32)

这些类别背后不是单纯局部观测，而是较强的手工规划：

- 预先构造石墙约束下的炸弹覆盖矩阵，并根据炸弹计时形成危险图。[`features.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/features.py#L23-L66)
- 从当前位置做最多 5 步的时空搜索，统计每个首步最终可达的安全位置数；对炸弹、当前爆炸、箱子被炸后的开放和首步对手占位做近似处理。[`features.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/features.py#L68-L114)
- 模拟下一步与假设放弹，用于判断当前放弹后是否存在逃生路线。[`features.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/features.py#L116-L177)
- 对安全候选动作分别做 BFS，按金币、箱子、最近敌人、其他敌人和逃生空间的手工权重打分，直接把得分最高的动作槽标为 `target`。[`features.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/features.py#L253-L342)
- 交换自身与敌人的视角，比较放弹前后对手是否从“有安全路线”变为“无安全路线”，若是则把放弹槽标为 `KILL!`。[`features.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/features.py#L344-L362)
- 最后将即时方格内容、安全搜索和上述规划目标汇成六槽离散特征。[`features.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/features.py#L364-L419)

因此其学习问题在很大程度上变成“学习接受或拒绝手工目标动作”。这也解释了目录名 `feature_is_everything`。对本项目而言，这是设计启发而不是可复制方案：本届要求明确禁止用一个直接确定“最佳动作”的特征或规则绕过模型学习，且禁止复制往届方案或权重（[`PROJECT_REQUIREMENTS.md`](../../PROJECT_REQUIREMENTS.md#L36-L48)）。本项目 `IMPLEMENTATION_GUIDE.md` 选择固定客观特征、让模型权衡危险，正是在规避这一问题。

## 3. 动作策略

动作空间固定为 `UP, RIGHT, DOWN, LEFT, WAIT, BOMB`。训练时使用 ε-greedy：以 `1-ε` 选择策略网络 argmax，以 `ε` 在六个动作中均匀随机；评估时直接取 argmax。代码没有合法动作 mask，因此探索和利用都可能输出物理非法动作，主要靠特征与 `INVALID_ACTION` 惩罚学习避免。[`callbacks.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/callbacks.py#L57-L88)

训练初始化 `ε=0.5`，每局结束乘 `0.9995`，下限检查为 `0.1`；折扣因子 `γ=0.9`。注意 ε 是按**局**衰减，不是按交互步或预先固定总预算衰减。[`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L21-L36) [`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L176-L188)

网络是小型 Dueling DQN：34 维输入依次经过 `64 → 32 → 16` 的线性层、LayerNorm、ReLU，然后拆成标量 value stream 与六维 advantage stream，按 `Q=V+A-mean(A)` 合成六个 Q 值。[`model.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/model.py#L3-L40)

## 4. 经验、增强与参数更新

每次 `game_events_occurred` 都重新提取 `(state, next_state)` 特征、追加自定义事件、计算奖励，将 `done=False` 的转移写入回放并调用一次优化；`end_of_round` 对最后动作构造 `next_state=None, done=True`，同样写入并优化一次。[`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L107-L144) [`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L147-L181)

每条真实转移并非只存一次：代码枚举 4 个旋转角与 3 个翻转组合，将状态、动作与下一状态同步变换，向容量 10,000 的 deque 追加 12 条样本。这里是 12 次循环，不是经去重验证的 D4 八种唯一对称；部分变换可能重复。[`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L29-L36) [`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L53-L64) [`symmetry.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/symmetry.py#L1-L45)

当回放至少有 64 条记录时，均匀抽一个 batch。策略网络只回归已执行动作的 Q 值；目标为 `reward + γ max_a Q_target(next)`，终止样本用 `done` 屏蔽未来价值；损失为 MSE，优化器为 AdamW、学习率 `1e-3`。目标网络初始化复制策略网络，并在 `steps_done % 1000 == 0` 时硬同步；`steps_done` 在每次 `act` 后增加。[`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L38-L50) [`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L66-L104) [`callbacks.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/callbacks.py#L82-L88)

## 5. 奖励塑形

最终 DQN 的真实事件奖励为：移动每步 `-1`，等待 `-5`，放弹 `-5`，非法动作 `-20`，收币 `+10`，真实击杀 `+50`，自杀 `-300`，被杀 `-100`。此外它根据特征追加：走向 `target` `+50`、走入 `dead` `-100`、有敌人在局部特征时放弹 `+20`、在放弹槽为 `target` 时放弹 `+50`、在 `KILL!` 时放弹 `+500`。同一步满足多个事件时奖励累加。[`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L124-L144) [`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L191-L221)

这套奖励远大于环境真实分数（旧仓库环境仍是收币 1、击杀 5），所以训练奖励不能当比赛分数解释。[`settings.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/settings.py#L47-L56) 更重要的是，`MOVE_TO_TARGET` 和 `KILL_ENEMY` 奖励来自同一套手工规划特征，尤其 `KILL! +500` 是预测性奖励而非已经发生的击杀。**推断**：它会强烈推动网络模仿规划器；仓库没有消融数据证明这些辅助奖励各自提高了正式得分。

## 6. 模型保存、加载与恢复

训练启动时，如果 `my-saved-model.pt` 存在就用 pickle 加载整个策略网络，否则新建网络；随后新建目标网络并复制策略参数。每 100 局 pickle 覆盖保存策略网络。评估模式也从同一个相对文件加载。[`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L21-L50) [`train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L183-L188) [`callbacks.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/callbacks.py#L35-L54)

这只算“权重续训”，不算完整恢复：重启会把 ε、经验回放、优化器状态、目标网络状态、步数和所有 RNG 状态重置。仓库也没有版本元数据、配置哈希、原子写入或加载后校验。因此不能逐位复现实验，也不能确认某个权重经历了哪些场景和对手。

表格型 Agent 的模式相似：训练启动读取已有 JSON Q 表或从空表开始，每步执行 TD 更新，每 100 局覆盖保存。基础 Q-learning 用下一状态最大 Q 值；SARSA 用 `choose_action` 生成下一动作；SARSA(λ) 另维护并在终局清零 eligibility traces；Double Q 维护两个 JSON 表。[`q_learning/train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/q_learning/train.py#L16-L49) [`sarsa_lambda/train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/sarsa_lambda/train.py#L16-L82) [`double_q/train.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/double_q/train.py#L16-L70)

## 7. 评估证据与缺口

框架本身支持 `--seed`、`--save-replay`、`--save-stats`、`--match-name` 和无 GUI 运行，为可复现实验提供了入口；但仓库没有保存使用这些入口得到的实验清单与原始结果。[`main.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/main.py#L112-L137)

唯一专门的评估脚本 `visualizer.py` 从字符串日志中提取得分，计算滑动平均分与“胜率”。它把所有并列最高且最高分非零的 Agent 都记为胜者；全员 0 分则无人获胜。脚本没有独占第一、并列第一的拆分，没有置信区间、固定 seed 集、死亡/击杀/自杀指标或训练/验证/测试隔离。[`visualizer.py`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/visualizer.py#L5-L57)

README 只有“第三名”一句，没有实验数据、比赛设置或可复现实验链接，故只能视为作者自述，不能据此归因某一算法、特征或奖励有效。[`README.md`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/README.md)

## 8. 对本项目训练流程的可用启发与边界

可以借鉴的抽象结构：

1. 在同一回调契约中统一完成 `state → features → action → transition → reward → update`。
2. 将安全逃生、金币、箱子和对手信息在一开始预留进稳定特征接口，跨阶段不改变输入维数。
3. 让状态、动作和下一状态同步做对称增强，但应使用经验证的八种唯一对称，并检查特征等变性。
4. 用经验回放、目标网络和终止 mask 稳定 DQN；用规则对手与策略快照形成难度更高的对战样本。
5. 将真实游戏分数与训练塑形奖励分开统计。

不应照搬的部分：

1. 不把 `target` 或 `KILL!` 这类近似直接推荐动作的规划器输出作为强特征/巨额奖励；本届课程明确限制这种绕过学习的做法。
2. 不允许随机探索或 bootstrap 在所有六动作上无条件取值；本项目应使用统一的物理合法动作 mask。
3. 不把“文件存在则续训”当完整恢复；应保存模型、目标网络、优化器、回放、探索进度和 RNG，并记录场景/对手/seed/代码版本。
4. 不把训练日志滑动平均或 README 排名当选模证据；应在冻结检查点上以独立 seed、固定对手、实际得分和明确排名口径评估。
5. 不根据该仓库的场景列表补写不存在的训练课程。当前项目自己的四阶段与评估流程应以 [`PROJECT_REQUIREMENTS.md`](../../PROJECT_REQUIREMENTS.md#L120-L142) 和 [`IMPLEMENTATION_GUIDE.md`](../../IMPLEMENTATION_GUIDE.md#L324-L356) 为依据，并由本项目实际 run metadata、逐局数据和检查点证明。

## 证据等级说明

- **事实**：固定提交的源码、配置、README 或 Git commit 直接显示。
- **推断**：由文件命名、提交顺序或可配置机制推测的开发意图；本文均显式标注，不能当运行结果。
- **未知**：固定提交未提供命令、日志、模型元数据或实验表，无法可靠重建；本文不填补这些空白。
