# Bomberman 项目实现指引：技术路线、逐文件步骤与三人分工

> 初始文档日期：2026-09-07；正式训练计划更新：2026-09-12。目标项目：本仓库。模板基线提交：`61b79ffa1f6976bd9f19eb922ca82ceed5ee7a9c`。
>
> 本文只记录团队派生的技术方案、实验计划与分工，不是课程要求的权威来源。课程硬约束、提交和报告要求统一见 [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)。
>
> 当前 Task 3 已通过独立冻结验证；Task 4 当前有效安排为文末“Task 4 共享父模型有限训练”。它替代本文较早的 Task 4 3000 局、六条课程链及 117 维阶段特征入口；旧安排和结果保留为历史记录。未写入结果栏的计划不得当作已完成结果。
>
> **证据驱动同步规则：**实现可以在明确分析并通过相关测试后改进本文的旧方案，但实现、接口约定、理由和检查项必须在同一可运行变更中同步。不得保留无说明的代码—文档差异，也不得通过改写本文掩盖实现缺陷。

## 1. 项目目标与已经确定的取舍

团队选择实现**表格型 Q-learning** 和**小型 DQN**。两者使用同一套客观特征、奖励和动作合法性规则，进行可复现的比较，并按预先确定的验证规则选择候选模型。

- 正式训练使用 CPU。Task 1 smoke 中 DQN CPU 前 20 局约 120 秒，GPU 同阶段约 130 秒；A100 路径保留作工程验证，但小网络未体现吞吐优势。
- 三人经验与投入接近，第一轮实现阶段各按约 24 小时安排，共 72 人时；无人值守训练另计。
- 优先完成正确性、原始数据、核心实验和独立交付，不预先保证比赛名次或某个胜率。
- `q_learning_agent` 与 `dqn_agent` 是两个独立可运行 Agent，共用 `team_agent` 的特征、奖励和探索协议。三人都服务于两个模型，不按“一人一个模型”分工；选模后只打包胜出的一个目录及其共享运行依赖。
- 物理合法性与时域生存性保持两个合同。当前 Double DQN 安全实验用 `survival-mask-v1/all` 对探索、训练贪心、冻结推理和 bootstrap 一致否决可证明必死的动作；模型仍在剩余动作中决策。
- 当前只迭代已表现出高金币/炸箱能力的 Double DQN，不引入规则动作标签、自我对弈或 Task 3。
- 实现期间同步记录方法、图表、失败案例与作者，供正式报告复用。

### 1.1 课程要求入口

截止日期、交付物、最终运行限制、预测试流程和报告格式只在 [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md) 中定义。开始打包、提交或报告工作前先读取该文档，本指引只描述团队如何满足这些要求。

## 2. 模板现在有什么，哪些文件先读

以下均为模板中**已经存在**的文件。保留原版实现，自己的扩展放入后续列出的新文件。

| 现有文件 | 现状或阅读重点 | 为什么要读 |
|---|---|---|
| `agent_code/tpl_agent/callbacks.py` | `setup` 创建随机动作概率；`act` 随机选动作；`state_to_features` 有 `...` 占位 | 接口示例，不是已完成的学习基线 |
| `agent_code/tpl_agent/train.py` | 长度为 3 的队列、占位事件、保存固定概率模型，没有学习更新 | 新训练逻辑需要独立补齐，不能只增大奖励或队列就称为学习 |
| `agents.py` | `AGENT_API`、回调签名、事件统计、顺序后端、计时和工作目录切换 | 保持官方兼容，理解状态和回调生命周期 |
| `environment.py` | `do_step`、`update_bombs`、`update_explosions`、`evaluate_explosions`、`send_game_events`、`end_round`、`get_state_for_agent` | 危险时序、死亡和终止处理的事实来源 |
| `items.py` | `Bomb.get_blast_coords`、`Explosion.is_dangerous` | 爆炸范围和危险阶段 |
| `settings.py` | 当前参数和 `SCENARIOS` | 避免沿用旧作业的 200 步或其他旧配置 |
| `main.py` | `--train`、`--seed`、`--scenario`、`--n-rounds`、`--save-stats`、`--save-replay` | 已有运行和导出功能，不重复造入口 |
| `events.py` | 官方事件常量 | 奖励引用常量，不用拼写相近的自定义名称混淆真实事件 |
| `test.py` | 运行一局并检查日志存在和更新 | 仅启动检查，不能证明训练有效 |
| `Dockerfile` | 官方提供的依赖环境 | 最终兼容性检查使用的基础 |

已有对手包括 `random_agent`、`peaceful_agent`、`coin_collector_agent`、`rule_based_agent`；保留其代码。框架还已有回放和 JSON 统计导出。

### 2.1 当前规则，按源码实现而不是按一般游戏印象

以下是模板基线的实现事实，用于解释团队方案，不是课程要求的第二份副本。实现和测试必须重新读取 `settings.py`、`environment.py` 与 `items.py`；课程约束的权威顺序见 [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)。

- 棋盘 17×17；每局最多 400 步；金币 1 分；击杀 5 分。
- 炸弹威力 3，计时参数 4，爆炸阶段计时参数 2；每步正式决策预算 0.5 秒。
- **石墙阻挡爆炸，箱子不阻挡爆炸传播。当前代码没有炸弹连锁引爆。**
- `coin-heaven`：无箱子，50 枚金币；`classic`：箱子密度 0.75，9 枚金币；以当前文件为准，不使用旧仓库的数量。
- 状态坐标为 `field[x, y]`。动作顺序固定为 `UP, RIGHT, DOWN, LEFT, WAIT, BOMB`。
- 一步的顺序为：收集各智能体动作并依随机顺序执行 → 收集金币 → 更新已有爆炸 → 更新炸弹 → 判断死亡 → 发送训练事件 → 检查是否结束。
- 所有智能体根据执行前的观察决定动作，但实际行动顺序随机；观察时合法的移动仍可能被别人的本步动作阻挡。

### 2.2 四个必须明确处理的细节

1. `explosion_map` 写入的是危险爆炸的 `timer - 1`。不能把这个值再机械减一次，或把旧代码的危险标签直接移植过来。
2. 自杀会产生 `KILLED_SELF`，死亡清理还会产生 `GOT_KILLED`。一次死亡只扣一次死亡奖励。
3. 存活者的最后一步可能先进入 `game_events_occurred`，再进入 `end_of_round`。不能将它分别作为非终止和终止样本各学一次。
4. 顺序后端在回调期间切换到智能体目录。所有自有输入输出路径应在初始化时解析为绝对路径；提交默认文件从 `Path(__file__).resolve().parent` 定位，不依赖开发机器路径。

另一个复现细节：官方随机、和平、金币和规则智能体的 `setup` 会调用无参数 `np.random.seed()`。仅在创建世界之前设全局种子不够，必须在所有智能体初始化完成后、第一步运行前再次按实验约定设置或恢复对手 RNG。

## 3. 技术路线与旧项目的借鉴边界

```text
官方 game_state
  → 物理合法性、爆炸时间与路径分析
  → 同一套离散特征
      → Q-learning：元组键 → 六个 Q 值
      → DQN：one-hot 向量 → 小网络 → 六个 Q 值
  → 在物理合法动作中探索或取最大 Q 值
  → 官方事件与下一状态 → 奖励 → 学习更新
```

旧项目值得学习的是危险预测、放弹后逃生搜索、紧凑表示和经验复用。其 `target` 是先人工评分选出动作，再奖励模型跟随；今年作业明确排除了“特征直接确定最佳动作”的示例，因此本项目不采用该设计。

| 特征模块可以提供 | 必须交给模型学习 |
|---|---|
| 某方向是否通行、未来何时危险 | 选择哪个动作 |
| 走一步后是否更接近可达金币 | 金币、箱子、攻击和风险的权衡 |
| 当前放弹能覆盖多少箱子 | 此时是否值得放弹 |
| 当前放弹后是否存在预测逃生路线 | 是否接受放弹及后续逃生的代价 |
| 对手的相对方向、距离和覆盖关系 | 是否追击或放弃攻击 |

不新增 `best_action`、`target_action`、综合推荐方向或“服从推荐动作”奖励。物理非法动作过滤不是安全策略：危险但合法的方向不被过滤。预测未来新出现的对手炸弹不在第一版范围内，有限预测不称为必然安全。

## 4. 模块索引与剩余交付项

本节是已实现接口的索引，不再是待建文件清单。`q_learning_agent` 和 `dqn_agent` 分别提供官方回调，`team_agent` 仅承载共享特征、危险、奖励和探索协议；`tpl_agent` 与官方核心文件保持不变。

### 4.1 智能体目录

| 文件 / 负责人 | 输入 → 输出与主要功能 | 依赖 | 完成标准 |
|---|---|---|---|
| `agent_code/team_agent/features.py` / A | `game_state` → `Features(state_key, vector, legal_mask)`；字段定义、确定性编码、坐标与合法性 | `danger.py`、NumPy | 14 项键与 40 维向量一致；不改输入；相同状态输出一致 |
| `agent_code/team_agent/danger.py` / A | 原始状态、是否假设放弹 → 未来危险、通行与各第一步逃生结果 | NumPy、标准库、官方参数 | 与第 5 节时序一致，固定局面检查通过 |
| `agent_code/q_learning_agent/callbacks.py`、`agent_code/dqn_agent/callbacks.py` / B | 官方 `setup`、`act` → 初始化状态、合法动作字符串 | 共享特征、探索协议、各自模型 | 训练 seed 和探索配置可追溯；评估不学习 |
| `agent_code/q_learning_agent/train.py`、`agent_code/dqn_agent/train.py` / B | 官方训练回调 → 去重转移、模型更新和检查点 | 特征、奖励、各自模型 | 每个 `(round, step)` 只更新一次，终止无 bootstrap，死亡不遗漏 |
| `agent_code/team_agent/rewards.py` / B | 事件序列、奖励配置 → 标量奖励和分项 | `events.py` | 重复箱子事件逐个累加；死亡只计一次；不修改原事件列表 |
| Q 表（位于 `q_learning_agent/callbacks.py`）与 `dqn_agent/model.py` / B | `Features` → 六个 Q 值；转移 → 更新 | NumPy / PyTorch | 表格和网络输入信息相同，Q 值有限，非法动作不参与 bootstrap |
| `agent_code/team_agent/exploration.py` / B | 完整配置、动作步 → 线性 ε | 标准库 | Q-learning 与 DQN 使用同一版本化公式，恢复时校验完整配置 |

### 4.2 实验与交付目录

| 文件 / 负责人 | 输入 → 输出与主要功能 | 依赖 | 完成标准 |
|---|---|---|---|
| `experiments/run.py` / C | Task、Agent、模式、配置、输出目录 → 训练或评估报告 | 官方世界类、自有与官方智能体 | `train`/`evaluate` 独立；四个 Task 自动选择场景和默认对手；各运行隔离 |
| `experiments/analyze.py` / C | 运行目录列表 → CSV 汇总和 PNG 图 | JSON、NumPy、Matplotlib | 正确区别独占第一、并列第一、零分平局；图能从原始数据生成 |
| `experiments/configs/base.json`、`formal_training.json`、`formal_training_coin3.json`、`stage_gate.json`、`stage_gate_coin3.json`、`main_validation.json`、`final_test.json` / C | 工程默认与正式训练/评估数据集 | 第 6–8 节约定 | CPU、冻结 Feature/Reward 契约、预定 seed 和局数均可校验 |
| `agent_code/legal_random_agent/` / C | 官方状态 → 从同一合法掩码均匀抽样 | 共享特征、独立 RNG | Task 1 诊断基线可固定 seed 重现，不包含学习状态 |
| `experiments/resume.py`、`devices.py` / C | 原子完整恢复、CPU/CUDA 设备解析 | 标准库、NumPy、可选 PyTorch | v4 合同和两代回退可验证；评估强制 CPU |
| 待实现：消融配置 / A、C | `v1_no_danger`、`r1_coin3_no_crate` 的独立完整课程 | 已版本化特征与奖励 | 各只改变一个因素并从零训练 |
| 待实现：打包工具和最终依赖清单 / C | 胜出 checkpoint → 单一可提交 Agent 包 | 最终模型存储格式 | 在原版框架、Docker、CPU 上独立加载后才标记完成 |
| `experiments/configs/reward_r2_balanced.json` / C | R2 奖励、评估 seeds、默认局数 | 第 6–8 节约定 | 不自动分配四阶段预算；每次运行显式选择 Task 和局数 |
| `experiments/configs/q_learning.json`、`dqn.json` / C | 两算法配置 | Reward 配置对应的完整展开值 | 第一轮除算法外信息、奖励和运行条件一致 |
| `experiments/configs/dqn_no_danger.json`、`dqn_no_crate_reward.json` / C | 两个固定消融配置 | A/B 定义消融含义 | 各只改变一个实验因素，分别重新训练 |
| `tools/package_agent.py` / C | 指定检查点和展开配置 → 可独立加载的智能体目录及 zip | 存储格式、自有目录 | 只打包选中模型，重新加载检查通过后才输出成功 |
| `requirements.txt` / C | 项目直接依赖清单 | 实际实现与兼容性结果 | 记录验证过的版本；不照抄旧项目整套依赖 |
| `agent_code/team_agent/requirements.txt` / C | 参赛包依赖 | 所选算法 | Q 表包无需 PyTorch；DQN 包包含 PyTorch 依赖 |

配置使用标准 JSON，不做隐式配置继承。实验运行保存完整展开值和源配置 SHA256。公共 `Features` 契约放在 `features.py`，各算法的转移和模型状态留在自己包内。

### 4.3 检查文件

| 检查文件 / 负责人 | 要检查的行为 | 不通过时阻止什么 |
|---|---|---|
| `tests/test_features.py` / A | 编码、缺失目标、边缘坐标、合法性、确定性和不修改输入 | 与模型集成 |
| `tests/test_danger.py` / A | 时间推进、爆炸范围、多炸弹和逃生 | 正式炸箱训练 |
| `tests/test_q_learning_agent.py`、`test_dqn_agent.py` / B | 更新、终止掩码、合法 bootstrap、存取与 seed | 模型对比 |
| `tests/test_resume.py`、`test_exploration.py` / B、C | 原子恢复、课程约束、探索边界与 v3 合同 | 长时间训练 |
| `tests/test_evaluation.py` / C | 排名口径、种子、原始记录和聚合 | 报告中的性能结论 |
| `tests/test_devices.py`、`test_formal_training.py` / C | 设备契约、正式配置、合法随机基线 | 正式开训 |

检查采用标准库 `unittest`；PyTorch 只在 DQN 检查中需要。第一版不新增测试框架。

## 5. A 的具体实现：危险预测和固定特征

### 5.1 危险预测的时间定义

`predict_danger(game_state, hypothetical_bomb=False)` 返回预测对象，包括 `danger[t, x, y]`、移动前障碍信息和五种第一步的预测可逃标记。**t=1 是本次动作执行后进行死亡判定的时刻**，不是当前观察时刻。

首版预测窗口 `H = BOMB_TIMER + EXPLOSION_TIMER + 1`，当前参数下 H=7。允许从配置选择消融，但正式功能使用完整窗口。

- 现有炸弹倒计时为 b：在未来 t=b+1 爆炸，危险持续该时刻和下一时刻。倒计时为 0 的炸弹本步动作后爆炸。
- 假设本次放弹：新炸弹先以计时 4 加入，再在本步更新中减为 3，因此在 t=5 爆炸，t=5、6 危险；不是本次放弹动作后立刻从 t=4 开始爆炸。
- 现有 `explosion_map=m`：未来 1≤t≤m 仍危险，m=0 对下一次判定没有残留危险。
- 石墙永久阻挡；箱子在爆炸结算时消失，因此从下一次移动才可进入。炸弹爆炸当步移动前仍阻挡新进入，下一步才解除。
- 搜索状态用 `(x,y,t)`。等待允许在自己已站立的炸弹格保持位置，但离开后不能在炸弹仍存在时重新进入。
- 第一版将当前其他智能体的位置在整个短期搜索中视为固定障碍；这是预测假设，不保证实际安全。并行动作产生的位置变化交由环境反馈。
- 合并多个危险时间段，不只记一个“最早爆炸时间”。不假设箱子挡火，也不实现不存在的连锁引爆。

实现顺序：单颗炸弹时间表 → 已有残留爆炸 → 多颗炸弹并集 → 按时间更新可通行区域 → 五种第一步搜索 → 假设放弹搜索。只有所有步都满足时间与通行约束，并能到达窗口末端的存活位置，才标记预测可逃。

### 5.2 固定特征版本 `v1`：14 个离散字段、40 维向量

字段从第一天全部预留。无对手阶段自然填“无对手”，进入对战阶段仍使用同一维度。A 先实现单人相关字段，再完成对手字段，不改变模型接口。

公共契约由 `features.py` 导出：`FEATURE_VERSION='v1'`、`STATE_KEY_SIZE=14`、`FEATURE_CATEGORY_COUNTS=(3,3,3,3,3,3,3,2,2,2,2,5,4,2)`、`FEATURE_DIM=40`，动作顺序使用 `ACTIONS`。模型、配置与检查点应导入或保存这些值并在加载时校验，不能另建一份不受检查的字段定义。

| 键索引 | 名称 / 类别数 | 编码定义 | one-hot 向量区间（左闭右开） |
|---|---|---|---|
| 0–3 | 上、右、下、左通行 / 各 3 类 | 0=物理不可进入；1=可进入但预测无逃生路径；2=可进入且预测可逃 | 0:3、3:6、6:9、9:12 |
| 4 | 原地危险 / 3 类 | 0=窗口内无危险；1=t=1 危险；2=t=2..H 才危险 | 12:15 |
| 5 | 放弹条件 / 3 类 | 0=物理不可放；1=可放但预测无逃生路径；2=可放且预测可逃 | 15:18 |
| 6 | 当前放弹覆盖箱子 / 3 类 | 0=0 个；1=1 个；2=2 个以上 | 18:21 |
| 7–10 | 上、右、下、左是否接近金币 / 各 2 类 | 1=合法移动后到固定金币目标的路径距离严格减少；其他为 0 | 21:23、23:25、25:27、27:29 |
| 11 | 最近对手方向 / 5 类 | 0=无；1=上；2=右；3=下；4=左 | 29:34 |
| 12 | 最近对手距离 / 4 类 | 0=无；1=曼哈顿距离 1；2=距离 2–4；3=距离≥5 | 34:38 |
| 13 | 对手爆炸覆盖 / 2 类 | 1=至少一名当前对手位于本格炸弹覆盖范围；否则 0 | 38:40 |

详细约定：

- 金币目标仅是用于测量距离的最近可达金币，不是综合动作推荐。按静态 BFS 距离选金币，同距按 `(x,y)` 排序；四个候选方向都对同一个目标计算。没有可达金币、方向非法或候选不可达时记 0。
- BFS 用当前石墙、箱子、炸弹、其他智能体为障碍；起点允许自身当前所在格。金币是可进入的终点。它测量导航距离，不参与安全过滤。
- 最近对手按曼哈顿距离、坐标、名称固定排序。方向取绝对位移较大的轴；相等时按 UP、RIGHT、DOWN、LEFT 的顺序取符合位移的方向。只有距离特征，不判断“应该追谁”。
- 箱子计数和对手覆盖均使用同一爆炸范围函数，受石墙阻挡，不受箱子阻挡；它们不是必然收益或击杀保证。
- `state_key` 为 14 个整数的元组，按安全 7 项、金币 4 项、对手 3 项分组；`vector` 直接拼接三个子特征的 one-hot，依次占 `0:21`、`21:29`、`29:40`，结果为 `float32[40]`，不是原始类别编号直接作为连续数。
- Q 表只创建实际见到的键，不枚举所有组合。记录唯一状态数及验证时未见状态比例；出现大量未见状态时先报告，再用特征消融/网络比较分析。
- `legal_mask` 为 `bool[6]`：移动依据当前占用，WAIT=True，BOMB 依据 `self` 的炸弹可用标记。危险与能否逃生不改变掩码。
- `game_state=None` 返回 None，作为终止输入；不把终止局面编码为普通全零状态。DQN 批量处理时可用零向量占位，但 `done` 必须屏蔽 bootstrap。
- 特征计算不改动输入数组、不调用全局 RNG、不保留跨回合目标。

## 6. B 的具体实现：回调、学习和存储

### 6.1 模型接口

两模型暴露相同能力：`q_values(features) -> float[6]`、`observe(transition)`、`state_dict()` 和 `load_state_dict(state)`。Q 表在 `observe` 中直接更新；DQN 在其中写入回放并按固定频率更新。

`Transition` 至少包含：`round_id`、`step_id`、`features`、`action_index`、`reward`、`next_features`、`done`。状态字段使用已复制的不可变键及独立数组，不引用之后会改变的游戏状态。

选动作规则：训练时用自有 RNG 做 ε-greedy，探索从合法动作中均匀抽样；利用时从合法最大 Q 值动作中按自有 RNG 处理并列。正式评估固定使用动作顺序中第一个并列最大值。未知 Q 表状态为六个零，不产生规则推荐动作。

### 6.2 两种更新

Q-learning：

```text
target = reward                                      if done
target = reward + gamma * max(Q(next)[next_legal])    otherwise
Q(key, action) += alpha * (target - Q(key, action))
```

DQN：40→64→64→6，隐藏层 ReLU，输出层线性。回放容量 20000，均匀采样 batch=64；累计 2000 次自身决策后开始，在每条完成转移进入模型时至多进行一次参数更新。Adam，学习率 0.0003，Huber/SmoothL1 loss；目标网络不计算梯度，每 1000 次成功参数更新硬同步。预测当前动作只 `gather` 已执行动作的 Q 值；下一状态取目标网络的合法最大值，终止样本不加下一状态价值。

Q 表 α=0.15；两模型 γ=0.95。两算法共用 `linear-v1`：决策步 0 时 ε=1.0，在 80,000 次自身决策内线性降到 0.05，之后保持 0.05。跨 Task 保留累计决策步，不重置 ε。

CPU 模式固定 PyTorch 单线程；Q 表路径延迟导入或完全不导入 PyTorch，以便导出不需要神经网络依赖的包。训练和评估速度要分别测量。

### 6.3 五个回调的职责

| 回调 | 必须完成 | 不放入这里的工作 |
|---|---|---|
| `setup(self)` | 解析配置、建自有 RNG、检查版本、准备模型；评估时加载权重 | 不隐式随机替代缺失模型 |
| `act(self, game_state)` | 计算特征、Q 值、合法动作选择；训练时记录决策计数 | 不做梯度更新或磁盘模型写入 |
| `setup_training(self)` | 新训练或恢复，初始化回放/优化器、计数和 pending 转移 | 不覆盖刚恢复的参数 |
| `game_events_occurred(...)` | 完成上一条暂存转移，暂存当前非终止候选 | 不立即把可能结束的一步永久记成非终止 |
| `end_of_round(...)` | 完成 pending/最终转移、去重、清理并写检查点 | 不把同一步再学一次 |

终止处理算法：

1. 用 `old_game_state['round'], old_game_state['step']` 标识普通回调的动作，不以 `new_game_state` 的步号猜下一次动作。
2. 普通回调到来，若已有不同步号的 pending，先按非终止提交它，再暂存当前记录和事件副本。
3. 结束回调的 key 与 pending 相同：使用结束回调的完整事件重新计算一次奖励，将同一步改成 `done=True, next_features=None`，只提交一次。事件不能简单拼接，否则金币和箱子会重复计数。
4. 结束 key 与 pending 不同：先提交之前 pending 的非终止转移，再提交最终动作的终止转移；这覆盖智能体死亡后没有普通回调的情况。
5. 保存已结束 key 防重复通知；清理 pending，防止跨回合串样本。没有实际动作的 None 输入直接忽略，不生成伪样本。

该方式令普通学习更新滞后一条转移，但不在 `act` 内训练，并避免先学非终止再额外学终止。400 步上限作为本作业有限回合任务的终止处理；因用户中断而提前保存的运行只标记为中断，不包装成正常完成实验。

### 6.4 奖励与消融开关

| 项目 | 初始值 |
|---|---:|
| `COIN_COLLECTED` 每次 | `r1`: +1；`r1_coin3`: +3 |
| `KILLED_OPPONENT` 每次 | +5 |
| `CRATE_DESTROYED` 每次 | +0.2 |
| 存在 `KILLED_SELF` 或 `GOT_KILLED` | 合计 −10，一次 |
| 每条完成转移 | −0.01 |
| `INVALID_ACTION` | 额外 −0.1 |

不对 BOMB_DROPPED、COIN_FOUND、SURVIVED_ROUND 单独加分，不给预测击杀大额奖励。事件列表可能有多次炸箱/击杀，不能转换成 set 后再统一求和；只有死亡使用布尔去重。合法动作仍可能因其他智能体先行动而失败，故保留无效动作处理。

返回奖励分项，分别记录真实游戏得分和训练奖励。`r1` 保持金币 +1；本轮正式课程使用
`r1_coin3`，它只把金币改为 +3。对应的严格单因素炸箱消融分别为
`r1_no_crate` 和 `r1_coin3_no_crate`。任何 Reward ID 或完整奖励表变化都必须从 Task 1
重新训练。

### 6.5 配置、存档与恢复

包内 `config.json` 至少包含 `algorithm`（`q_learning`/`dqn`）、`feature_version`（`v1`）、`reward_version`、`seed`、`checkpoint`、`training` 和 `evaluation`。奖励数值只在共享奖励注册表中维护；Runner 把所选版本解析后的完整奖励表写入展开配置、metadata 和 checkpoint。训练局数由每次运行显式指定，不在基础配置中按四阶段自动分配。

- 实验进程可用 `BOMBERMAN_CONFIG` 指定一个绝对配置路径、`BOMBERMAN_RUN_DIR` 指定绝对输出目录；没有这些变量时仅使用包内默认值。实验配置自包含，不做隐式多层覆盖。
- 模型路径相对包内配置解析，训练输出从 run 目录解析。配置进入运行前保存完整副本和 SHA256。
- 新训练由实验配置明确指定，不因文件不存在而推测“自动重练”；恢复模式或评估模式缺失检查点直接报错。
- Q 表保存整数键数组和浮点 Q 数组到 NPZ（读取禁用对象 pickle），附 JSON 元数据；DQN 保存 `state_dict`，不 pickle 整个模型对象。
- 完整训练检查点包含策略网络、目标网络、优化器、回放池、自有 RNG 状态、NumPy Generator 状态、CPU/CUDA Torch RNG 状态、训练设备、动作步数、更新步数、阶段与探索进度；Q 表保存对应适用状态。新建 DQN 前按模型种子设置 Torch RNG；恢复时在网络和优化器构造完成后恢复 RNG，避免初始化消耗改变后续随机序列。GPU 训练启用确定性算法，正式评估始终映射到 CPU。
- 每个完整回合的训练回调成功返回后保存一代 resume 快照。先写临时 generation，校验文件 SHA256 后原子发布，再原子更新 `latest.json`；只保留最新和上一代。最新一代损坏时自动回退上一代，并在子 run 谱系中记录原因所对应的丢失局数。
- `--resume-from` 只接受 v7 父 run 根目录，并始终创建新的子 run。只允许同 Task 或 `1→2→3→4` 的直接晋级；算法、seed、动作顺序、Feature/Reward/Safety 完整契约、设备、源码身份和 checkpoint schema 必须一致。当前完整恢复 schema 为 `training-resume-v7`；v1–v6 和旧 `final.pkl`/`final.pt` 默认只允许冻结评估。唯一例外是 `--migrate-resume-from` 可把完整 v6 Task 1 快照显式迁移为 v7 子 run，迁移事实和双方源码身份必须写入 lineage。
- 同 Task 回合边界恢复必须保持探索、n-step、回放保留和最小局数配置，动作目标只能提高；阶段动作计数、动作历史、n-step 队列及全部 RNG 原样恢复。直接进入下一 Task 时可按预注册配置重置阶段 ε、n-step 与回放/蒸馏比例，同时保留全局动作数、Q/网络、optimizer、target、分区 replay、父网络和 Agent RNG；清空回合 pending 与早停并按同一 seed 重建环境。
- 旧 `final.pkl`/`final.pt` 仍可用于冻结评估；缺少 resume schema 的旧 checkpoint 不能用于精确续训。
- 参赛导出仅保留推理权重、配置和所选算法依赖，不包含优化器、回放池、训练日志或本机绝对路径。

## 7. C 的具体实现：可复现实验和独立交付

### 7.1 先用已有能力，不解析旧仓库的字符串日志作为主数据

官方 `--save-stats` 可以导出累计数据，但现有 `by_round` 没有完整的逐智能体排名信息。`experiments/run.py` 定义 `ExperimentWorld(BombeRLeWorld)`，在 `end_round` 调用父类后追加每个智能体的 `score`、`statistics`、`dead` 和回合长度。官方核心文件保持不变。

每个进程只运行一个实验配置并独占 run 目录。正式六条链可作为六个单线程 CPU 进程并行；每条链内部必须串行，且设置 PyTorch/BLAS 单线程，避免共享 cwd、日志处理器和 CPU 超额订阅。运行器直接使用当前工作树，不创建未实现的框架副本，因此正式开训前必须冻结干净提交并记录源码身份。

### 7.2 随机性

- 自有智能体用独立 `random.Random(seed)` 或 NumPy Generator，不在 `act` 内调用全局 `random.seed`/`np.random.seed`。
- Runner 通过 `BOMBERMAN_AGENT_SEED` 把训练 seed 传给 Agent RNG 和 DQN 参数初始化；直接由官方框架启动、没有该环境变量时默认 seed 0。
- 世界地图与执行顺序使用官方世界 RNG；初始化由明确环境种子控制。
- 官方规则对手使用全局 Python/NumPy 随机数；实验进程在创建世界前设置一次，并在世界和所有智能体 setup 完成后、第一局开始前再次设置。恢复训练则在此处恢复保存的状态。后一次操作覆盖官方 setup 的无参数 `np.random.seed()`，是复现必要条件。
- 每个训练种子 S 的环境初始种子为 `1000+S`、官方对手全局种子为 `3000+S`；S 固定为 11、22、33。
- 验证和测试每个种子单独初始化一局，使用该数作为世界种子，`seed+100000` 作为官方对手全局种子。不同候选使用相同列表。
- 固定种子使给定策略和配置可重现，不保证不同策略在分歧后经历完全相同的随机调用序列。

### 7.3 输出结构

```text
runs/<run_id>/
  metadata.json          代码提交/源码哈希、展开配置、依赖、硬件、种子、模式、起止时间
  episodes.jsonl         每局每人的原始指标，一行一个回合
  training.csv           每回合训练奖励、epsilon、loss、状态数等
  training_summary.json  训练过程汇总
  training_progress.png  reward 与 epsilon 趋势图
  timing.jsonl           完整 act 的用时及超时/跳过记录
  official_stats.json    官方导出，作为交叉检查
  checkpoints/           只属于该运行的模型
  resume/                两代原子完整训练快照及 latest.json
```

`episodes.jsonl` 记录 `run_id`、阶段、回合索引、环境种子、场景、对手、各 agent 的金币/击杀/自杀/箱子/炸弹/无效动作/实际得分、存活、回合总步数。自有智能体生存步数通过只读观察每步 active_agents 记录。失败、中断、缺少记录必须显式标记。

计时在实验副本中通过后端 `get_with_time` 的只读包装收集官方测量值，不改其返回值、可用时间或动作；用它记录完整回调时间。被上一轮超时跳过的动作单独计数，不能算作零耗时成功决策。性能测量包含特征计算，不只计网络前向。

### 7.4 指标定义

- `mean_score`：实际总分均值，第一模型选择指标。
- 独占第一：分数严格高于所有其他人。
- 并列第一：最高分大于零，且与至少另一人并列。
- 全员零分：所有人得分为零，单独统计，不计作独占/并列胜利。
- 金币、击杀、自杀、炸箱、无效动作、生存率和生存步数分别统计。
- 输出完整 act 用时的中位数、P95、最大值、超过 0.5 秒次数及框架跳过次数；本机通过不等于官方硬件一定通过。
- 每个训练种子各自报告结果，再报告跨训练种子的均值与标准差。不把一个检查点的 100 局当作 100 次独立训练。

只有一名智能体的能力场景不报告“胜率”，报告金币、效率与死亡。

### 7.5 打包规则

打包工具为 `experiments/package_agent.py`，接受 Agent 名称、指定检查点与输出 ZIP 路径，创建唯一胜出 Agent 的独立目录，内置共享运行依赖、唯一选中权重、权重 SHA 清单与实测依赖版本。若胜出 Q-learning，包不应要求 PyTorch；若胜出 DQN，则必须声明经过验证的 CPU PyTorch 依赖。

生成前检查配置版本、40 维输入、六动作顺序、权重存在及加载；生成后解压到新的原版框架副本，以 `train=0` 对三个 random_agent 完成运行，再对三个 rule_based_agent 检查耗时。通过后才标记提交包就绪。

只打包一个包含 `callbacks.py` 的智能体目录，避免官方搜索到错误模型；输出名称与交付边界按 [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md) 执行。

## 8. 正式课程训练进程、门槛与选择

### 8.1 开训条件与 smoke 证据

GPU/CPU smoke 只证明工程链路，不进入正式训练谱系或模型比较。Task 1、seed 11 的留存证据为：Q-learning CPU 100 局约 349.95 秒，39,040 次动作、50 个 Q 状态；其 5 局 CPU 冻结评估共收集 158 枚金币。DQN A100 100 局约 1,183.42 秒、36,358 次 optimizer update、峰值显存约 65.2 MiB；其 5 局 CPU 冻结评估共收集 94 枚金币。DQN CPU 前 20 局约 120 秒，而 GPU 同阶段约 130 秒，因此正式训练选择 CPU。所有 smoke 均未在 Task 1 使用 `BOMB`，且未发现超时或硬错误。

已有六条 500 局 Task 1 run 保持原始 `r1 + discrete-v1` 身份，作为冻结评估证据。
它们的 v3 snapshots 不得作为 `r1_coin3` 课程的恢复来源，也不改写原始 metadata、
checkpoint 或训练记录。

正式开训前必须满足：完整测试、`py_compile` 和 `git diff --check` 通过；代码、配置和本指南冻结为干净提交；六条主链的 `source_commit`、`source_hash` 与源配置 hash 一致。专用配置 `formal_training_coin3.json` 固定 CPU、`discrete-v1`、`r1_coin3`、sampled replay 和关闭 early stopping。硬错误或行为修复改变源码后，所有受影响的公平对照链从 Task 1 重训，不跨 commit 续训。

### 8.2 六条独立课程链

Q-learning 与 DQN 各使用训练 seeds 11、22、33，共六条链。最多启动六个单线程 CPU 进程；每条链独立通过门槛并晋级，不等待其他链，但链内严格按 `1→2→3→4` 串行并用 `--resume-from` 创建子 run。

| Task | 新增局数 | 训练环境 | 本阶段能力目标 | 性能失败追加 |
|---|---:|---|---|---:|
| 1 | 500 | `coin-heaven`、无对手、禁 `BOMB` | 高效收集可见金币 | 125 |
| 2 | 1,000 | `classic`、无对手 | 炸箱、逃生和隐藏金币 | 250 |
| 3 | 1,500 | `classic`、同时面对 `peaceful_agent` 与 `coin_collector_agent` | 竞争、追击和击杀 | 375 |
| 4 | 3,000 | `classic`、三名 `rule_based_agent` | 正式四人强对战 | 750 |

Task 3 同时使用两名弱对手，是团队对官方 SHOULD 课程路线的合并实现；不是两个可互换的单对手实验。每阶段默认保存第 1 局及 10% 进度回放，每回合仍提交两代完整 resume generation。中断后按已完成局数补足本阶段预算，修复后新 run 使用 `_retryN`，不得覆盖旧目录。

新 coin +3 课程的命名模板为
`formal_<q|dqn>_discrete_v1_r1_coin3_s<seed>_t<task>_r<local-rounds>`；同配置追加可增加
`_cont<rounds>`，故障重跑增加 `_retryN`。名称只作索引，累计局数和谱系以 metadata 为准。

### 8.3 每阶段 5-seed × 20 局门槛

阶段门槛在 CPU 上使用 seeds 10000–10004，每个 seed 运行 20 局；同一候选的父子比较复用完全相同的 seed 列表和局数。

Task 1 coin3 checkpoint 与合法均匀随机基线均使用专用
`stage_gate_coin3.json`，固定 `discrete-v1 + r1_coin3` 并启用只读导航诊断。主判据仍只有
100 局平均金币相对同 seed 随机基线至少 `+2`；不得根据结果事后改变门槛。辅助诊断报告
每 100 步金币、每枚金币步数、收完 50 枚金币的比例与完成步数、`WAIT` 率、立即反向率、
缩短最近金币距离的动作率、仍可追踪旧目标时的目标切换率，以及多个等距最近金币的出现率。
这些指标只用于定位失败原因，不单独决定晋级。

聚合公式固定为：`coins_per_100_steps = 100 × Σcoins / Σround_steps`，
`steps_per_coin = Σround_steps / Σcoins`；完成步数只在收完 50 枚金币的局上取平均。
`WAIT`/立即反向率以全部诊断决策为分母，缩短距离率以动作前后距离均可比较的决策为分母，
目标切换率以旧目标仍存在的机会为分母，等距目标率以存在可达金币目标的决策为分母。

共同硬门槛：run 完整且局数正确；冻结 checkpoint 可加载；最新两代 resume generation 均通过 hash 校验；Q 表状态数增长，或 DQN optimizer updates 大于零且 loss 有限；无异常、超时或框架跳过；完整 `act` P95 < 50 ms、最大值 < 500 ms；Task 1 不出现 `BOMB`；无效动作率不超过 1%。任一硬门槛失败立即停止该链并修复。

能力门槛：Task 1 的平均金币数至少比 `legal_random_agent` 高 2；Task 2 的平均炸箱数至少比 Task 1 父模型高 0.5；Task 3/4 的平均击杀数至少比直接父模型高 0.1，或独占/并列第一率至少提高 5 个百分点。晋级模型在所有旧 Task 上的 `mean_score` 不得低于父模型的 90%；有炸弹阶段的自杀率不得比父模型高 10 个百分点，且绝对值不得超过 35%。比例均以完整评估局数为分母。

通过的链立即独立晋级，不等待其他 seed。某模型族至少 2/3 seeds 通过时不修改该模型族；
至少 2/3 seeds 失败但冻结表现已明显优于随机时，才使用表中 25% 预算按同配置续训一次。
若优势不足 2 枚或循环诊断明显异常，则跳过低价值追加，为失败模型族从零运行 seed 11 的
`discrete-v1/discrete-q-v2 × sparse r1_coin3/coin-only potential` 2×2 实验。四个变体先各训
250 局并冻结评估，前两个再用同契约子 run 补至总计 500 局；若两族都失败，先做 DQN。
仍失败则保留证据并按第 8.5 节截止策略继续。任何新 Feature、Reward 或探索配置使用新 ID
和新谱系，不得接续当前 v4 checkpoint。硬门槛失败不适用上述性能追加。

### 8.4 主验证、模型族与最终测试

完成 Task 4 后，六个主候选分别在 seeds 10000–10099 上对三名 `rule_based_agent` 做一次 100 局主验证。先将每个模型族三个训练 seed 的 checkpoint `mean_score` 取平均，在 Q-learning 与 DQN 之间选族；平分依次比较模型族平均自杀率、最差完整 `act` P95、模型族 ID 字典序。胜出族内按单 checkpoint 的 `mean_score` 选一个，平分依次使用更低自杀率、更低 `act` P95、checkpoint ID 字典序。

只冻结胜者，然后在从未用于调参的 seeds 20000–20099 上执行一次最终测试。最终测试结果不得触发新的训练、超参数修改或重新选模；验证和测试的数据隔离由 `main_validation.json` 与 `final_test.json` 固定。

### 8.5 截止策略与消融

- 9 月 12–13 日：完成前置实现、测试、指南同步和源码冻结。
- 9 月 13 日：Task 1–2 训练与阶段门槛。
- 9 月 14–15 日：Task 3–4 训练与阶段门槛。
- 9 月 16 日：选择最早合格的 Task 4 checkpoint，完成原版框架、Docker、CPU、三 random/三 rule-based 兼容检查并生成预测试包。若无合格 Task 4，依次降级为表现最佳的失败 Task 4、合格 Task 3、合格 Task 2，并明确标记保底候选。
- 9 月 17 日预测试后：DQN 的 `v1_no_danger` 与 `r1_coin3_no_crate` 分别从零运行 seed 11 的完整课程。其 100-seed `mean_score` 相对基础 DQN 的变化满足 `abs(variant-base) / max(abs(base), 1) >= 10%` 时，才复制 seeds 22/33；完成同等验证后方可参与最终模型族选择。
- 9 月 20 日：冻结最终 Agent；预留一天处理 9 月 21 日提交。

原计划未完成必须标为未完成，不能把较短预算当成相同预算比较。训练可在自有智能体死亡后按官方训练规则结束；阶段门槛、主验证和最终测试必须让整局正常结束。

### 8.6 当前有效的 Task 1–2 安全迭代（Task 3 前置门槛）

本节取代 8.2–8.5 中关于“直接沿旧六链进入 Task 3”的安排；Task 3/4 的后续课程仍保留，但只有本节联合门槛合格后才能启动。触发原因是上一轮 `b66a0c8` 结果：Q-learning 主要陷入 `WAIT`/往返，DQN Task 1 已有约 31–38 枚金币但仍循环；DQN Task 2 训练自杀率为 100%，仅约 9500 个有效动作，冻结评估约 0.05–0.10 金币、1.8–3.6 箱、15%–60% 自杀，Task 1 保留率约 39%–81%。因此本轮不再提高金币事件奖励，而修复目标表达、安全探索、炸弹信用和跨 Task 遗忘。

所有分支在 [`experiments/pre_task3_iteration.json`](experiments/pre_task3_iteration.json) 中预注册，最多三轮：

1. seed 11 对 Q-learning、DQN 分别运行 `discrete-v1/discrete-objective-v1 × r1_coin3/r5_coin_potential`，共 8 条链。每条先达到 100000 个 Task 1 动作或 500 局上限；按硬门槛、mean coins、全金币率、较低 WAIT/立即反向、act P95、run ID 顺序各保留前两名，再以同 Task 子 run 累计到 200000 动作，总局数最多 750。开发集固定 seeds 10000–10019，Task 1 晋级要求 `mean_coins ≥ 35`。
2. 对达标算法保留基础奖励链；另用数值等价的 `r6_safe_sparse` 或 `r6_safe_potential` 从零重建同 seed Task 1 父模型。Task 1 的轨迹和学习器状态必须与对应基础版一致，否则视为实现错误。两者分别进入 Task 2：至少 500 局且达到 150000 个阶段动作，最多 2000 局。每个子模型同时评估父模型在 Task 2 的基线、子模型 Task 2 能力及子模型 Task 1 回测。
3. 从对应 Task 1 父谱系重新建立子链，不续训失败的 Task 2。自杀率失败用 r7 与 4-step return；DQN 保留率失败改 75% 父回放/25% 当前回放且蒸馏 λ=2；Q 能力失败切 `double_q_agent + discrete-objective-v1`；DQN 能力失败切 `double_dqn_continuous_v2_agent + continuous-v2`。多项失败同时应用，并为每一项重复传入 `--adaptation-trigger suicide|retention|q_capability|dqn_capability`；Runner 将有序列表写入 expanded config 和 metadata。第三轮后不再增加预算或修改设计。

Task 2 联合门槛全部同时满足：`mean_coins ≥ 2`、`mean_crates ≥ 5`、相对同 seed 父模型 `mean_crates` 增加至少 0.5、Task 1 `mean_score` 保留率至少 90%、自杀率不超过 5%、无效动作率不超过 1%。排序依次为：是否全门槛通过、较低自杀率、较高 `min(coins/2, crates/5)`、金币、保留率、run ID。硬门槛仍要求 checkpoint 可加载、v5 两代 snapshot/hash 有效、Q 表非空或 DQN updates>0、loss 有限、无异常/timeout/skipped action、act P95<50 ms 且最大值<500 ms；动作上限命中但动作目标不足也是硬失败。

每项父子差异按相同 seed 配对，使用 10000 次 bootstrap、RNG seed 20260913 报告 2.5%/97.5% 分位区间。复制 seeds 22/33 后按 60 个开发回合选择单一 checkpoint，再且仅再用 seeds 11000–11099 做一次主验证并重新满足全部联合门槛。seeds 20000–20099 仍封存为最终测试，本轮不得使用。

运行命名固定为 `iter_r<round>_<agent>_<feature>_<reward>_s<seed>_t<task>_a<target>_<commit>`；已有目录追加 `_retryN`。`--target-stage-action-steps` 是阶段累计目标，`--min-rounds` 是本地最小局数，`--n-rounds` 是本次子 run 的硬上限。旧 v4 run 不能成为父节点。

2026-09-14 的预注册执行结果如下。实现冻结提交为 `8f35ca2`；首次 DQN Task 2 暴露出非法动作 `-inf` logits 在蒸馏 KL 中产生 `0×inf=NaN`，修复提交 `03fbd1b` 改用有限最小值掩码，并从零重建所有第二、三轮父谱系。当前有效的第二、三轮 run 均记录 `source_commit=03fbd1b`。两组 r6 Task 1 父模型与对应基础奖励父模型的 750 局 episode、动作、学习器参数及 RNG 状态完全一致；差异仅为奖励合同字段，满足数值等价检查。

下表的 Task 2 数值均来自 seeds 10000–10019 的 20 个冻结回合；括号内依次为 coins、crates、suicide。保留率是同一开发 seed 上子模型相对 Task 1 父模型的总 `mean_score` 比率：

| 轮次 | Agent/Feature/Reward | seed | Task 1 coins/循环 | Task 2 coins/crates/suicide | Task 1 保留率 | 联合门槛 | 失败触发/后续 |
|---|---|---:|---|---|---:|---|---|
| 1 | Q / `discrete-v1` / `r5_coin_potential` | 11 | 50.00；全金币率 100%；WAIT 0.20%，立即反向 7.07% | — | — | Task 1 通过 | 作为第二轮基础父模型 |
| 1 | DQN / `discrete-v1` / `r1_coin3` | 11 | 45.90；全金币率 75%；WAIT 25.77%，立即反向 20.60% | — | — | Task 1 通过 | 作为第二轮基础父模型 |
| 2 | Q / `discrete-v1` / `r5_coin_potential` | 11 | 50.00 | 0.05 / 2.30 / 50% | 92.00% | 失败；仅 34090/150000 动作 | `suicide,q_capability` |
| 2 | Q / `discrete-v1` / `r6_safe_potential` | 11 | 50.00 | 0.00 / 0.50 / 25% | 10.00% | 失败 | `suicide,q_capability`；冻结策略 WAIT 99.55% |
| 2 | DQN / `discrete-v1` / `r1_coin3` | 11 | 45.90 | 0.20 / 4.65 / 80% | 85.29% | 失败；仅 61050/150000 动作 | `suicide,retention,dqn_capability` |
| 2 | DQN / `discrete-v1` / `r6_safe_sparse` | 11 | 45.90 | 0.10 / 2.50 / 30% | 72.88% | 失败 | `suicide,retention,dqn_capability` |
| 3 | Double Q / `discrete-objective-v1` / `r7_safe_credit_potential` | 11 | 47.90；全金币率 90%；WAIT 3.60% | 0.60 / 16.10 / 20% | 24.95% | 失败 | 最佳 Q 失败模型；不复制、不晋级 |
| 3 | Double DQN / `continuous-v2` / `r7_safe_credit_sparse` | 11 | 49.90；全金币率 95%；WAIT 12.96% | 5.45 / 76.65 / 30% | 98.70% | 仅自杀率失败 | 最佳总体失败模型；不复制、不晋级 |

第三轮 10000 次配对 bootstrap（RNG seed 20260913）的关键 95% 区间为：Double Q Task 2 coins `0.60 [0.25,0.95]`、crates `16.10 [11.15,22.50]`、相对父模型 crates 增量 `13.40 [8.35,19.85]`、suicide `20% [5%,40%]`、Task 1 保留率 `24.95% [17.80%,33.09%]`；Double DQN 分别为 `5.45 [4.30,6.55]`、`76.65 [62.15,90.15]`、`74.80 [60.15,88.40]`、`30% [10%,50%]`、`98.70% [96.49%,100.20%]`。保留率区间上界略高于 100% 来自 bootstrap 重采样时父/子均值比率的分母波动。

所有有效第二、三轮训练 run 均通过 v5 两代 snapshot/hash、CPU checkpoint 加载、有限 loss、无异常/timeout/skipped action、invalid rate 0%、训练 `act` P95<50 ms 且最大值<500 ms 的工程检查。达到 Task 2 动作目标的安全 Q、Double Q、安全 DQN、Double DQN 分别在 1971、978、1742、742 局停止。三轮后没有模型满足 `suicide≤5%` 的全部联合门槛，因此 seeds 22/33 复制和 seeds 11000–11099 主验证均按协议未运行；seeds 20000–20099 仍未使用，Task 3 不得启动。

三轮后仍无联合合格模型时，冻结排序最高的失败模型并明确标记“无 Task 3 晋级资格”；不得用接近门槛、训练奖励或单个 seed 替代联合验收。

### 8.7 Double DQN v6 生存约束消融（当前执行协议）

第 8.6 节最强的 `continuous-v2 + r7` 候选已有 `49.90` 个 Task 1 金币、Task 2 `5.45` 金币和 `76.65` 箱，但 20 个开发 seed 中自杀 6 次。逐步重放表明六次均是安全放弹后选择了必死移动或等待，其中五次发生在放弹后的第一个动作；当时均存在安全替代动作。根因是 v5 只约束随机探索，未约束贪心、冻结推理或 Double DQN target。详细证据见 [`docs/research/task2-double-dqn-suicide-mitigation.md`](docs/research/task2-double-dqn-suicide-mitigation.md)。

当前 Safety 合同固定为 `survival-mask-v1`：`mode=all`、`horizon=7`、无安全动作时 `fallback=physical_q`。它只从物理合法集合中移除无法存活至预测终点的动作，不给出推荐方向；Q 网络仍在剩余动作中排序。统一 mask 用于 ε 随机、训练贪心、冻结推理和 replay 中 Double DQN policy argmax。每步记录 raw argmax、最终动作、介入/回退、各动作安全时长和逃生空间。领域术语见 [`CONTEXT.md`](CONTEXT.md)，取舍见 [`docs/adr/0001-veto-only-survival-mask.md`](docs/adr/0001-veto-only-survival-mask.md)。

`continuous-v3` 在远程 84 维 v2 上加入六动作各自的 `survives_horizon`、逃生余量和下一步可继续存活动作比例，把 `safe_area` 改成 `log1p(area)/log1p(board_area)`，再加入全局安全动作比例及自身炸弹承担/可见/倒计时/爆炸轴事实，总计 107 维。旧 78 维 v2 只通过内部 legacy 适配器读取冻结 checkpoint。动作和炸弹历史在回合开始清空，并进入 checkpoint。

`r8_safe_constrained` 保留 r7 sparse 的 step、coin、kill、crate、死亡、箱区/危险势和不可逃放弹惩罚，删除预测 useful-bomb 正奖励；只有“当前动作必死且存在安全替代”时即时扣 20。`KILLED_SELF` 与金币、炸箱或击杀同帧时抑制这些正奖励。Task 1 禁止放弹，因此相同 feature/seed 下 r8 与 r7 的数值和轨迹必须一致。

所有条件已在 [`experiments/task2_safety_ablation.json`](experiments/task2_safety_ablation.json) 预注册。冻结 shield 诊断把旧模型的20-seed自杀率从30%降到0%，因此在提交 `2eba43a` 建立了三条 seed 11 Task 1 链：`v2+r7+all`、`v3+r7+all`、`v3+r8+all`。三者跑满750局后分别累计180311、182000、182000动作，未达到旧200000动作目标，但最后100局平均金币分别为49.99、49.98、49.98；最后100局全金币率为99%、98%、98%。这证明动作门槛会把更快收完金币、因而更早结束回合的模型误记为训练量不足。

Task 1 因此改用 `task1-frozen-score-v1`：累计第200局开始，每50局暂停训练，对 seeds 9000–9019 各做1局关闭探索的同步评估；连续三次 `mean_score≥48` 才记为 **Task 1 分数收敛**，第三次 checkpoint 为父模型。一次失败将连续计数归零，累计1000局仍未收敛则状态为 `not_converged`。动作数继续控制 epsilon 并用于诊断，不再参与 Task 1 停止或晋级。全金币率、完成步数、WAIT、往返和时延只作诊断。

收敛后必须在未参与停止的 seeds 10000–10019 上做一次独立阶段门槛，并再次满足 `mean_score≥48` 与工程硬门槛；失败后不得继续利用这组 seeds 调参。只有阶段门槛合格的链可进入 Task 2。Task 2 仍至少500局且达到150000阶段动作、最多2000局，保持4-step、75%父回放和蒸馏 λ=2。seeds 11000–11099 与 20000–20099 分别继续保留给主验证和最终测试。

开发门槛同时要求：Task 1 `mean_score≥48`；Task 2 `mean_coins≥2`、`mean_crates≥5`、相对父模型 crates 增加至少 0.5；Task 1 保留率至少 90%；`suicide≤5%`、零放弹局率不超过 10%、存活炸弹率至少 95%、invalid 不超过 1%；checkpoint/v7 两代 snapshot 完整，loss 有限，无异常/timeout/skipped，act P95<50 ms 且最大值<500 ms。多候选依次按自杀率、零放弹率、存活炸弹率、`min(coins/2,crates/5)`、金币、保留率、时延和 run ID 排序。

seed 11 优胜配置才从零复制 seeds 22/33。三训练 seed × 20 开发 seed 汇总后只选一个 checkpoint，在 seeds 11000–11099 一次性做父 Task 1、子 Task 1、父 Task 2、子 Task 2 主验证，并用 10000 次配对 bootstrap（RNG `20260913`）报告 95% 区间。主验证失败即停止，不回看次优模型。seeds 20000–20099 保持封存，本轮不得启动 Task 3。

当前完整恢复协议为 `training-resume-v7`，在 v6 的 Safety/Feature/Reward/网络/历史/replay/RNG 合同上增加 Task 1 分数收敛配置与已提交评估历史。周期评估先引用不可变 generation/hash，再原子发布 assessment；评估失败或半成品不计数。普通续训拒绝 v6，显式迁移入口只接受完整 v6 Task 1 快照。

### 8.8 Task 2 winner 复现与主验证结果

`continuous-v2 + r7_safe_credit_sparse + survival-mask-v1/all` 在提交 `814173b` 上从零复制训练 seeds 22/33。两条 Task 1 均在第 200、250、300 局的冻结监控中连续取得 50 分，并在独立 seeds 10000–10019 上再次取得 50 分；随后两条 Task 2 均在 500 局、200000 阶段动作时完成。加上已有 seed 11，三个训练 seed 的开发集结果分别为：seed 11 `6.30` coins / `88.45` crates，seed 22 `7.15` / `94.55`，seed 33 `6.15` / `83.75`；三者自杀率均为 0%、炸弹存活率均为 100%、Task 1 保留率均为 100%。60 局汇总为 `6.53` coins、`88.92` crates，bootstrap 95% 区间分别为 `[6.05, 7.00]` 和 `[84.32, 93.37]`。

按预注册顺序选择 seed 22 checkpoint 作为 **Task 2 winner candidate**，并且只对该 checkpoint 使用 seeds 11000–11099 做一次主验证。结果为 Task 2 `7.25` coins（95% CI `[6.96, 7.53]`）、`100.26` crates（`[97.48, 103.02]`）、0% 自杀、0% 零放弹局、100% 炸弹存活；Task 1 父子均为 50 分，保留率 100%。invalid、timeout 和 skipped 均为 0，完整 act P95 为 `6.38 ms`、最大值 `20.58 ms`。全部联合门槛通过，因此该 seed 22 checkpoint 正式指定为 **Task 2 winner**。仓库内可直接加载的权重位于 [`agent_code/double_dqn_continuous_v2_agent/final.pt`](agent_code/double_dqn_continuous_v2_agent/final.pt)，机器清单与 640 局精简证据分别见 [`experiments/task2_winner.json`](experiments/task2_winner.json) 和 [`experiments/task2_winner_evaluations.csv`](experiments/task2_winner_evaluations.csv)。最终测试 seeds 20000–20099 仍未使用，本轮未启动 Task 3。

### 8.9 Task 3 弱对手试验协议

第一轮 Task 3 不修改 Feature、Reward 或 Safety：继续使用 84 维 `continuous-v2`、`r7_safe_credit_sparse` 和 `survival-mask-v1/all`，同时面对 `peaceful_agent` 与 `coin_collector_agent`。这样可以直接回答“加入弱对手训练是否有效”，不会把课程变化和奖励变化混在一起。完整预注册见 [`experiments/task3_pilot.json`](experiments/task3_pilot.json)，配置见 [`experiments/configs/task3_pilot_r7.json`](experiments/configs/task3_pilot_r7.json)，领域边界见 [`docs/adr/0004-pilot-task3-with-unchanged-r7.md`](docs/adr/0004-pilot-task3-with-unchanged-r7.md)。

Task 2 三条父链属于源码提交 `814173b`，精确 v7 恢复要求源码身份相同；因此 Task 3 训练在该提交的隔离 worktree 中执行，但读取当前预注册配置并记录其 SHA256 和预注册提交。seeds 11、22、33 各自从对应父链晋级，不共享 seed 22 的公开权重。每段新增500局，累计500、1000、1500局时分别检查；一次通过即停止该链，动作数只控制 Task 3 的 `epsilon=0.30→0.05/120000 actions`，不控制停止。训练保持4-step return、75%旧Task replay、25%当前Task replay、蒸馏系数2和CPU单线程。

开发 seeds 固定为12000–12019。每个父模型先分别在Task 1、2、3冻结评估；每个子checkpoint也分别评估Task 1、2、3，所有差值按相同训练seed与环境seed配对。Task 3必须同时满足：游戏分数比父模型增加至少0.5；击杀增加至少0.1或独占/并列第一率增加至少5个百分点；金币和炸箱分别保留父模型的90%。旧任务门槛为Task 1分数保留90%，以及Task 2金币和炸箱分别保留90%。Task 2/3自杀率不超过5%、炸弹存活率至少95%，Task 3零放弹局不超过10%；所有评估还要求invalid≤1%、act P95<50 ms、最大值<500 ms，并且没有异常、timeout或skipped action。

只有三个训练seed全部独立通过，才按Task 3游戏分数、自杀率、击杀、第一名率、最低资源/旧任务保留率、时延和run ID选出一个 **Task 3 pilot candidate**。它只是开发集候选，`qualified_for_task4=false`；本轮不使用新的100-seed主验证，不启动Task 4，也不根据失败结果自动修改奖励。

### 8.10 Task 3 弱对手试验结果

三条链都完成了累计1500局，并在最后一个检查点使用开发 seeds 12000–12019 完成父子配对评估。seed 11 和 seed 33 通过全部门槛；seed 22 未通过。因此模型族按预注册的“三个训练 seed 必须全部独立通过”规则判定失败，没有创建 `task3_pilot_candidate.json`，没有执行100-seed主验证，也没有启动Task 4。机器可读汇总和360局逐seed证据分别见 [`experiments/task3_pilot_results.json`](experiments/task3_pilot_results.json) 与 [`experiments/task3_pilot_evaluations.csv`](experiments/task3_pilot_evaluations.csv)。

| 训练 seed | 首次通过点 | Task 3 分数（父→子） | 金币（父→子） | 炸箱（父→子） | 击杀（父→子） | 第一名率（父→子） | 子模型自杀率 | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | 1500局 | 5.15→6.10 | 3.40→4.85 | 39.75→56.50 | 0.35→0.25 | 40%→55% | 0% | 通过；以第一名率提升满足战斗门槛 |
| 22 | 无 | 5.60→4.70 | 3.60→3.95 | 48.05→51.70 | 0.40→0.15 | 55%→30% | 35% | 失败：分数、战斗和安全均未通过 |
| 33 | 1500局 | 5.10→6.75 | 3.35→4.75 | 38.20→52.40 | 0.35→0.40 | 35%→50% | 5% | 通过；安全门槛取等号 |

三条链的Task 1回测均为50分，Task 2金币和炸箱均高于各自父模型，说明本轮的主要失败不是旧任务遗忘，也不是资源能力退化，而是 **seed 22在弱对手局面中的策略不稳定**：追加到1500局后分数与战斗指标反而低于父模型，自杀率仍为35%。因此当前证据不支持直接进入Task 4；下一轮应把“跨训练seed稳定降低对战自杀”作为首要问题，而不能用seed 11或33的单点成功替代模型族复现。

### 8.11 Task 3 分阶段奖励与安全迭代

下一轮使用117维 `continuous-phase-v1`。前107维保持 `continuous-v3`，后10维记录平滑的前/中/后期权重、箱子和对手消耗、分差、自身与最近对手机动性，以及后期机动性危机。局面进度为 `0.45×箱子消耗 + 0.30×对手淘汰 + 0.25×回合进度`；阶段权重连续混合，不按固定步数硬切。地图边缘停留只作诊断，因为边缘并不等于危险。

四个首轮对照固定为 `phase+r7`、`phase+r9_resource`、`phase+r9_combat`、`phase+r9_full`，训练 seeds 为11和22。r9前期重金币/炸箱，中期提高自己击杀价值，后期保留击杀并增加机动性势；全程维持危险势和 `survival-mask-v1/all`。每条先跑500局，在13000–13019冻结评估；自杀超过10%立即淘汰，前两名才追加到1000局。只有最高名在分数不低于父模型、自杀不超过10%且保留率合格时可追加到1500局。完整合同见 [`experiments/task3_phase_iteration.json`](experiments/task3_phase_iteration.json)。

Task 2父模型不能普通续训到新网络。`--transfer-task3-from` 显式读取v7父快照，把84维第一层复制进117维网络并把新增33列置零，同时重置optimizer和replay。训练专用seeds 6000–6099为每个父模型采集最多各10000条Task 1/2教师状态；每次更新使用32条Task 3 TD样本和32条冻结教师样本。迁移后的精确恢复版本为 `training-resume-v8`。

若第一轮最高名任一seed自杀超过5%，才建立 `survival-mask-v2` 对照：仍先用v1，只在自身炸弹有效时否决逃生面积低于最佳安全动作75%的动作。若只剩战斗门槛失败，则从同一Task 2父模型重新迁移并跑250局静止目标、250局不放弹随机移动目标、500局正常弱对手。第三轮只允许组合第二轮已经分别有效的两个改动，不再改数值。

优胜配置从自己的Task 2父模型新增seed 33，并让三seed在14000–14019重新确认。全部独立通过后只选一个candidate，在15000–15099做一次主验证；通过才命名 `task3_winner` 并允许进入Task 4，失败不测试第二名。Task 4入口保持获胜的117维特征、r9和Safety合同，对三名rule-based agent训练；本轮不启动Task 4，20000–20099继续封存。
### 8.9 无生存 mask 的因果炸弹信用实验

为检验能否只靠学习信号解决 Expected SARSA(lambda) 的 Task 2 自杀问题，新增
`r11_causal_bomb_credit`。该版本保持 `survival-mask-v1/mode=off`，不删除任何物理合法
动作；自杀时直接更新此前的放弹状态，炸箱奖励延迟到炸弹消失且智能体仍可放弹时结算，
并把探索从 `0.15` 线性降至 `0.02`。seed 11 从对应 Task 1 权重 warm start，训练到
150031 个阶段动作后于第 1747 局停止。

seeds 10000–10019 的关闭探索 Task 2 评估为：`0.25` coins、`7.95` crates、100%
自杀、0% 生存、0% 零放弹局、炸弹存活率 `53/69=76.81%`，invalid/timeout/skipped
均为 0，完整 act P95 `8.05 ms`、最大值 `22.30 ms`。同 seeds 的 Task 1 回测为
`49.8/50`，act P95 `19.15 ms`、最大值 `40.17 ms`。相较旧 r10 训练末期，平均回合
被显著延长，但冻结策略仍在每局最终自杀；因此该实验明确失败，不复制 seeds 22/33，
不具备 Task 3 晋级资格。结果表明单次终局/放弹回溯不足，下一轮若开展必须预注册密集
存活余量奖励或更换学习器，不得继续堆叠同类终局惩罚。

### 8.12 Task 3 分阶段迭代结果

首轮八条链均在源码 `125124f` 上完成500局，并在seeds 13000–13019完成父子配对冻结评估。四个方案都至少有一个训练seed的自杀率超过10%，因此按预注册规则全部在500局淘汰，不追加到1000局。排序最高的是 `continuous-phase-v1 + r7_safe_credit_sparse + survival-mask-v1`：它的两seed平均Task 3分数为5.50，但seed 11/22自杀率均为20%，seed 22分数仍比父模型低0.45。

| 首轮方案 | seed 11：分数增量 / 自杀 | seed 22：分数增量 / 自杀 | 最差Task 2金币/炸箱保留 | 结论 |
|---|---:|---:|---:|---|
| phase + r7 | +0.75 / 20% | −0.45 / 20% | 85% / 83% | 排名第一，但安全失败 |
| phase + r9 resource | +0.95 / 40% | −1.15 / 10% | 76% / 80% | 安全与资源保留失败 |
| phase + r9 combat | +0.95 / 15% | −1.35 / 25% | 89% / 91% | 跨seed战斗不稳定 |
| phase + r9 full | +0.90 / 20% | −1.75 / 15% | 81% / 80% | 后期势未转化为安全收益 |

最高名的自杀率超过5%，因此触发第二轮 `survival-mask-v2` 对照；特征和r7保持不变，并从两个Task 2父模型重新迁移训练500局。mask-v2把seed 11/22的自杀率从20%降到10%/15%，seed 11的Task 3分数和击杀分别提升1.85和0.15，但seed 22仍下降0.75分和0.20击杀；两seed的Task 2金币保留仅60%/67%，炸箱保留仅58%/75%。这说明余量否决方向有效，但没有解决5%安全门槛，而且产生明显旧任务资源退化。

第二轮没有满足“安全、资源和保留均通过而只剩战斗失败”的对手课程条件；mask-v2也没有单独解决安全问题，所以不存在两个可供第三轮组合的已验证有效改动。实验到此停止：不训练seed 33，不使用14000–14019或15000–15099，不创建Task 3 candidate/winner，也不进入Task 4。完整720局证据与机器判定见 [`experiments/task3_phase_evaluations.csv`](experiments/task3_phase_evaluations.csv) 和 [`experiments/task3_phase_results.json`](experiments/task3_phase_results.json)。本轮最佳失败配置是 `phase + r7 + mask-v2`，只能用于诊断，`qualified_for_task4=false`。

## 9. 核心实验如何分配和解释

| 实验 | 负责人 | 保持不变 | 唯一变化 | 要回答的问题 |
|---|---|---|---|---|
| 两模型比较 | B，C 提供评估工具 | 输入信息、奖励、合法性、场景、预算和种子列表 | Q-learning / DQN | 网络在相同信息下是否更有效 |
| 危险特征消融 | A，B 提供训练器 | DQN、奖励、预算、物理合法掩码 | 关闭危险相关输入 | 未来危险与放弹逃生信息贡献多少 |
| 炸箱奖励消融 | C，B 检查奖励 | DQN、特征、预算 | 炸箱奖励 0.2→0 | 辅助奖励是否改善实际得分或诱发无效炸箱 |
| 对称增强（扩展） | A 实现变换，B 接训练，C 评估 | 其他条件和真实交互预算 | 有无增强 | 经验复用是否提升样本效率 |

危险消融采用固定定义：四方向合法者一律编码为 2、非法者仍为 0；当前位置危险填 0；放弹条件只保留物理可用性（不可用=0，可用=1），不计算逃生标签；其他字段保持一致。仍为 40 维，另标特征版本 `v1_no_danger`，从头训练。不能仅清空一个字段，却从其他字段泄露同样危险信息。

所有消融重新训练，不能删除已训练模型的输入后只测一次。对称增强须同步变换状态、动作和合法掩码，采用八种唯一正方形对称；特征含同距排序时，优先变换原始固定局面后重新提取，检查变换是否真正等变，不默认所有特征都可机械轮换。

每个实验写五句话：假设是什么、只改了什么、用了多少训练与评估数据、结果和波动是什么、能支持或不能支持什么结论。训练 loss 与奖励不能替代实际游戏得分。

## 10. 三人平均分工与协作

### 10.1 代码截止前各 24 小时

| 成员 | 工作包 | 人时 | 交付证据 |
|---|---|---:|---|
| A | 核对规则与固定局面 | 3 | 时间语义说明、手工局面列表 |
| A | 危险时间与逃生搜索 | 8 | danger 模块与局面检查 |
| A | 特征、编码与接口 | 4 | 14 字段/40 维编码及说明 |
| A | 正确性检查与 CPU 耗时分析 | 4 | 检查记录、热点及优化前后对比 |
| A | 特征消融、分析与写作记录 | 5 | 原始结果、图、解释 |
| **A 合计** | | **24** | |
| B | Q-learning | 3 | 可检查的动作价值更新 |
| B | DQN、回放与目标网络 | 7 | 网络学习与终止掩码检查 |
| B | 回调、奖励与去重 | 5 | 重复回调、自杀双事件检查 |
| B | 检查点、恢复与学习检查 | 4 | 保存恢复一致性记录 |
| B | 两模型比较、分析与写作记录 | 5 | 两模型多种子结果 |
| **B 合计** | | **24** | |
| C | 配置与训练/评估运行工具 | 7 | 可独立运行的实验入口 |
| C | 逐局指标、统计与绘图 | 5 | 原始 JSONL、可再生图表 |
| C | 官方基线和奖励消融 | 5 | 基线与消融结果 |
| C | 打包、兼容性与提交检查 | 4 | 独立包验证记录 |
| C | 指引整合、结果索引与写作记录 | 3 | 可追溯的实验索引 |
| **C 合计** | | **24** | |

24 小时是计划而非保证。每天记录实际投入；9 月 12 日检查一次，优先转移实验运行整理任务，不在中期随意重分底层模块。每人都写自己实验的结果分析，C 不承担全部报告。

### 10.2 不等待别人完成的第一批工作

- **A**：直接构造 game_state 字典，测试坐标、爆炸时间和特征；不需要模型。
- **B**：手工构造合法 Features 和 Transition，检查奖励正负、终止和保存；不需要真实特征提取器。
- **C**：先用官方智能体验证 ExperimentWorld、指标、种子和图表；不需要学习模型。

人工输入只证明接口和更新逻辑，不证明模型已经学会游戏。模块之间存在数据依赖，但开发任务不形成 A 做完→B 开始→C 开始的串行链。

### 10.3 日常集成规则

- 每个文件由第 4 节的负责人主改，其他人通过小补丁或共同复核交接，不同时重写公共回调。
- 每天合并最小可运行版本；接口改进须有分析和测试证据，并在同一变更中同步实现、调用方与本指引，不能留下无说明的不一致。
- A/B/C 都使用同一配置和数据格式。训练输出目录不能相互覆盖，代码提交不包含临时运行副本。
- A 复核 B 使用特征的方式；B 复核 C 的训练/评估开关；C 复核 A 的时间开销与指标解释。
- 三人一起决定最终候选，但依据预先固定的验证规则，不凭一次好看的回放。

## 11. 团队内部里程碑与可执行入口

以下日期是团队内部计划，不定义课程截止时间；正式截止节点见 [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)。

| 日期 | 阶段成果 | 通过条件 |
|---|---|---|
| 9 月 12–13 日 | 前置实现、测试、指南、源码冻结 | 干净提交；六链源码/配置身份一致 |
| 9 月 13 日 | Task 1–2 与阶段门槛 | 每条链独立记录晋级或失败证据 |
| 9 月 14–15 日 | Task 3–4 与阶段门槛 | 得到最多六个 Task 4 候选 |
| 9 月 16 日 | 候选降级决策、原版/Docker/CPU 兼容检查 | 预测试压缩包独立运行 |
| 9 月 17 日后 | 两项 DQN 消融 | seed 11 完整课程；达到阈值才扩展 22/33 |
| 9 月 20 日 | 最终测试与 Agent 冻结 | 测试集只使用一次，代码/配置/参数一致 |
| 9 月 21 日 | 提交缓冲 | 仅处理交付问题，不再调参 |

先在项目根目录建立并验证统一环境：

```bash
conda env create -f environment.yml
conda run --no-capture-output -n mle python -m unittest discover -s tests -p "test_*.py"
```

并行启动六条链前，在每个训练终端设置 `OMP_NUM_THREADS=1` 与 `MKL_NUM_THREADS=1`；DQN
自身还会调用 `torch.set_num_threads(1)`。每条命令只启动一条链，绝不并行启动同一链的两个阶段。

以下以 Q-learning seed 11 演示一条链；DQN 仅将 `q_learning_agent`/`q` 替换为 `dqn_agent`/`dqn`。Task 1 必须从零开始，后续 Task 必须引用直接父 run：

```bash
conda run --no-capture-output -n mle python experiments/run.py --config experiments/configs/formal_training_coin3.json --mode train --device cpu --task 1 --agent q_learning_agent --n-rounds 500 --seed 11 --run-id formal_q_discrete_v1_r1_coin3_s11_t1_r500

conda run --no-capture-output -n mle python experiments/run.py --config experiments/configs/formal_training_coin3.json --mode train --device cpu --task 2 --agent q_learning_agent --n-rounds 1000 --seed 11 --resume-from runs/formal_q_discrete_v1_r1_coin3_s11_t1_r500 --run-id formal_q_discrete_v1_r1_coin3_s11_t2_r1000

conda run --no-capture-output -n mle python experiments/run.py --config experiments/configs/formal_training_coin3.json --mode train --device cpu --task 3 --agent q_learning_agent --n-rounds 1500 --seed 11 --resume-from runs/formal_q_discrete_v1_r1_coin3_s11_t2_r1000 --run-id formal_q_discrete_v1_r1_coin3_s11_t3_r1500

conda run --no-capture-output -n mle python experiments/run.py --config experiments/configs/formal_training_coin3.json --mode train --device cpu --task 4 --agent q_learning_agent --n-rounds 3000 --seed 11 --resume-from runs/formal_q_discrete_v1_r1_coin3_s11_t3_r1500 --run-id formal_q_discrete_v1_r1_coin3_s11_t4_r3000
```

Task 3 会固定加入 `peaceful_agent` 和 `coin_collector_agent`；Task 4 默认加入三名 `rule_based_agent`，无需手写 `--opponents`。另外五条链分别使用 Q-learning seeds 22/33 和 DQN seeds 11/22/33，可在独立终端作为单线程进程并行启动。

Task 1 合法均匀随机诊断基线与候选阶段门槛示例：

```bash
conda run --no-capture-output -n mle python experiments/run.py --config experiments/configs/stage_gate_coin3.json --mode evaluate --task 1 --agent legal_random_agent --checkpoint agent_code/legal_random_agent/baseline.json --run-id gate_legal_random_discrete_v1_r1_coin3_t1

conda run --no-capture-output -n mle python experiments/run.py --config experiments/configs/stage_gate_coin3.json --mode evaluate --task 1 --agent q_learning_agent --checkpoint runs/formal_q_discrete_v1_r1_coin3_s11_t1_r500/checkpoints/final.pkl --run-id gate_formal_q_discrete_v1_r1_coin3_s11_t1
```

主验证改用 `main_validation.json`，且 Task 4 对三名规则对手运行；冻结唯一胜者后才改用 `final_test.json`。主验证和最终测试配置分别固定 10000–10099 和 20000–20099，每个 seed 一局；阶段门槛使用“5 seeds × 20 局”。正式训练不使用 `--silence-errors`，也不在本轮指南更新时自动启动。

最终打包工具尚未实现；在其实现并通过独立目录检查前，不得把下列官方框架手工检查当成完整打包验收。选中 Q-learning 或 DQN 后将 `<selected_agent>` 替换为唯一胜出目录：

```bash
python main.py play --no-gui --agents <selected_agent> random_agent random_agent random_agent --n-rounds 10 --save-stats results/submission_random.json
python main.py play --no-gui --agents <selected_agent> rule_based_agent rule_based_agent rule_based_agent --n-rounds 10 --save-stats results/submission_rules.json
```

## 12. 验收清单

### Task 3 自身炸弹逃生实验（独立分支）

本轮从 `814173b` 的 Task 2 完整 v7 状态出发，只改变安全约束，不改变
84 维 `continuous-v2`、`r7_safe_credit_sparse`、网络或对手。v3 从放弹选择
开始跟踪自身炸弹责任；能保留两条内部节点不相交逃生路线时，只允许这些
动作，否则先退回 v1 安全集，v1 为空才退回物理合法 Q 最大动作。

先对 seed 22 的七个已知死亡局（12000、12005、12008、12011、12015、
12017、12019）做冻结复查，再在 16000–16019 对三个旧 1500 局 checkpoint
做 v1/v3 配对。任何 checkpoint 的 Task 2/3 自杀率超过 5%、出现可避免逃生
崩溃，或资源/旧任务保留低于 90%，均停止长训练且不调整双路线阈值。

冻结准入通过后，seed 11/22 从各自 Task 2 父 run 通过
`--transfer-task3-safety-from` 建立 v9 子链，每 500 局检查一次、最多 1500
局；两者通过后才训练 seed 33。只有训练后因安全指标失败时，才允许一次
48 条父任务、8 条普通 Task 3、8 条自身炸弹周期样本的预注册 replay 对照。
确认集为 17000–17019，唯一候选主验证为 18000–18099；20000–20099 不用。
完整固定合同见 `experiments/task3_escape_obligation.json`。

冻结准入结果：在七个已知 seed-22 自炸局中，v1 为 7/7 自炸，v3 降至
1/7；炸弹存活率从 93.75% 提升到 99.43%，但没有达到预注册的 0/7 自炸
硬门槛。残余失败发生在 seed 12017：第 285 步放弹前静态图判断有两条路线，
环境推进后的第 286 步却已无 H=7 可存活动作，说明只把对手当前位置当障碍
不能保证下一帧路线仍可用。实验因此按协议停止；16000–16019 完整冻结 A/B、
Task 3 长训练、安全 replay 对照、确认集和主验证均未运行，且没有
`task3_winner`。详细证据见
`experiments/task3_escape_obligation_results.json` 和
`docs/research/task3-escape-obligation-results.md`。

### Task 3 对手鲁棒逃生实验（v4）

v3 的残余 seed 12017 失败发生在放弹后的首次环境推进，因此下一轮不改变
84维特征、r7奖励、网络或训练合同。v4在准备放弹和整个自身炸弹逃生责任期
枚举对手全部物理合法动作、对手放弹及官方执行顺序；每个不同结果都必须至少
保留一条H=7逃生路径。无v4动作时依次回退v3、v1和物理Q。

固定流程为：七个已知死亡seed达到0自杀；16000–16019完成三个旧checkpoint
在Task 1/2/3上的v3/v4配对；三个checkpoint分别在16100–16199完成Task 3
冻结安全确认；之后才按seeds 11/22、再seed 33训练。训练后使用17000–17019
确认，唯一候选使用18000–18099主验证。任一冻结门槛失败立即停止，不修改
情景集合或fallback。完整合同见`experiments/task3_opponent_robust.json`。

### Task 3 可控生存实验（v5）

v4 只检查一次对手转移，仍可能允许“当前看似能逃、下一步已经无解”的放弹。
v5 因此把安全判断放回真正造成风险的 `BOMB` 动作：从放弹到自身火焰完全消失，
都必须存在可根据新局面重新选择动作的生存策略，而不是预先固定一条路线。

第一步严格模拟双方所有物理动作、放弹和官方执行顺序。后续步骤把对手可能占据的
格子与可能制造的爆炸合并成保守危险集合，再反向计算我方仍可控制的区域。这个近似
只会多拒绝动作，不会把未证明安全的动作放行。搜索超时按不安全处理，但超时本身仍
是工程失败，不能算通过。

实验继续固定 Double DQN、84维`continuous-v2`、`r7_safe_credit_sparse`、5-step
return和原Task 1/2保留机制。恢复协议为`training-resume-v11`。先回归11个历史
自炸seed，再用19100–19199进行100-seed冻结准入。准入要求自杀率不超过5%、
robust guarantee loss和搜索超时均为0、炸弹存活率至少95%、零放弹局不超过10%、
资源/战斗能力至少保留90%，并满足act P95<250ms、最大值<480ms。

准入通过后只先训练seed 22，每500局检查一次，最多1500局。若只因安全失败，允许
从Task 2父模型重建一次预注册安全replay对照；能力不足则继续到下一检查点。优胜臂
原样复制seeds 11和33。开发、确认、唯一候选主验证分别使用19200–19219、
19300–19319和19400–19499；20000–20099继续封存。本轮不启动Task 4。完整合同见
`experiments/task3_controllable_survival.json`。

冻结准入最终通过：11个历史自炸seed均为0自杀；三个旧Task 3 checkpoint在
19100–19199启用v5后自杀率均为0、炸弹存活率100%，没有搜索超时或guarantee
loss，完整act最大值为312ms。v5本身因此保留，不回滚。

首次seed-22训练暴露了独立的课程迁移错误：Task 3子模型继承的是Task 2 checkpoint
内用于上一阶段的Task 1 teacher，而不是Task 2 policy。错误版本在开发集上只保留
Task 2金币7.9%和炸箱11.3%。修复后500局模型的Task 3分数由父模型5.95提高到7.70，
击杀由0.35提高到0.55，第一名率由35%提高到55%，冻结自杀率仍为0；Task 2金币和
炸箱保留率恢复到82.9%和82.0%，但仍低于90%门槛，因此不得复制seeds 11/33。

下一步使用`experiments/task3_retention_prefix.json`：保持学习合同不变，从同一父模型
确定性重跑seed 22的前250局，并复用19200–19219进行父子Task 1/2/3冻结评估。
只有250局checkpoint通过全部联合门槛，seeds 11/33才各自固定训练250局。若250局
只失败于旧任务保留，下一轮才单独引入Task 1/2/3=`16/32/16`分层Replay，先检查
250局、最多追加至500局；不得把早停与Replay变化合并成同一个无法归因的实验。
直接父阶段teacher与最早合格checkpoint的定义见`CONTEXT.md`，取舍见ADR 0005。

预注册提交推送并确认工作树干净后，后台只运行以下入口：

```bash
conda run --no-capture-output -n mle \
  python experiments/task3_retention_prefix.py \
  --manifest experiments/task3_retention_prefix.json
```

入口依次完成训练、确定性前缀核验、六组冻结评估、配对bootstrap和门槛JSON。
退出码0表示全部通过，2表示能力/保留/安全门槛失败，1表示基础设施或产物完整性错误；
三种情况均在`runs/task3_retention_prefix_evidence_s22_<commit>/result.json`留下证据。

### Task 3冻结能力平台早停

250局复制实验表明固定训练长度不能跨seed稳定工作：seeds 11/22通过全部门槛，
但seed 33的Task 3分数从6.35降至5.45、击杀从0.45降至0.25，同时金币、炸箱和
冻结安全能力保持。这不是安全或旧任务遗忘，而是较晚更新覆盖了父模型已有的战斗
行为。因此新实验不再依据训练reward、loss或探索局分数停止。

`experiments/task3_plateau_stopping.json`预注册`task3-plateau-v1`。每个训练seed从
自己的Task 2父模型开始，每50局创建一个不可变v11子run并在19200–19219关闭探索
评估Task 1/2/3。合格点必须通过现有全部能力、保留、安全和工程门槛；Task 3均分
相对平台锚点提高至少0.25才重置两次检查的合格耐心。尚未出现合格点时，分数提高
0.25或失败门槛减少会重置三次检查的失败耐心。最大训练300局。

平台停止后选择历史最佳合格checkpoint，而不是最后一个checkpoint。达到300局仍在
改善时可选择最佳合格点进入确认，但必须标记为budget-truncated，不得称为收敛。
seed 33先运行；只有它合格，才按同一规则运行seeds 11/22。随后使用19300–19319
确认三个训练seed，并只对唯一候选使用19400–19499做主验证。20000–20099仍封存，
本流程不启动Task 4。

后台入口为：

```bash
conda run --no-capture-output -n mle \
  python experiments/task3_plateau_stopping.py \
  --manifest experiments/task3_plateau_stopping.json
```

基础设施中断后使用同一命令增加`--resume`。控制器不会覆盖半成品，而是从最后一次
完整50局子run创建不可变retry。退出码0表示主验证通过，2表示实验门槛失败，1表示
基础设施错误。术语见`CONTEXT.md`，取舍见ADR 0006。

### A：环境与特征

- [ ] 石墙挡火、箱子不挡火、无连锁引爆。
- [ ] 倒计时 0 本步爆炸；新放炸弹在本指南 t=5 爆炸；爆炸残留和消失时刻正确。
- [ ] 多炸弹不同危险时间段全部保留。
- [ ] 放弹死胡同、等待后通行、站在自己炸弹上等待、离开后重入阻挡正确。
- [ ] 输入状态不变，缺失金币/对手编码正常，40 维编码位置固定。
- [ ] 危险不会修改物理合法掩码；WAIT 保留。

### B：学习与生命周期

- [ ] 人工正奖励提高对应 Q 值，终止目标不使用未来价值。
- [ ] 下一状态最大值排除物理非法动作；DQN target 不反向传播。
- [ ] 普通回调和结束回调同一步只更新一次；最后死亡动作不遗漏。
- [ ] `KILLED_SELF` 与 `GOT_KILLED` 同时出现只扣 −10；两个箱子保留两次 +0.2。
- [ ] 下一回合 pending 清空；保存加载预测一致；恢复保留探索和更新进度。
- [ ] 评估缺失权重/版本不符明确报错，不降级成随机动作。

### C：实验与提交

- [ ] 单局小样本人工分数 `[5,3,2,0]` 是独占第一；`[5,5,0,0]` 是两人并列；`[0,0,0,0]` 单独统计。
- [ ] 每局原始记录齐全，累计得分与官方导出一致。
- [ ] 训练、验证、测试种子不混用，官方对手与自有模型 RNG 分开控制。
- [ ] 评估模式为 `train=False`，自有智能体死亡后对局继续完整结束。
- [ ] 图表从原始数据重建，三个训练种子分别可查，失败/中断未伪装成成功。
- [ ] 完整 act 最大耗时和超时/跳过次数已记录；CPU 下测试且不使用智能体多进程。
- [ ] 原版框架加单一智能体目录即可运行；无开发目录路径依赖。
- [ ] Docker 检查记录依赖与环境；预测试和最终包对应同一个已验证版本。

## 13. 报告与复现材料如何积累

### Task 3 生命周期修复与有限优化（2026-09-17）

`79d1e87` 平台早停套餐在独立确认中失败：seed11通过，seed22失败于
Task2金币保留、Task3得分与战斗提升，seed33失败于战斗提升；主验证未启动。
随后确认训练回调读取旧状态会清掉自身炸弹责任，冻结推理没有该回调。
修复采用决策前后不可变历史快照，不改变84维输入或v11回合结束快照格式。

新合同为 `experiments/task3_lifecycle_campaign.json`，入口为
`python experiments/task3_lifecycle_campaign.py`（基础设施恢复加 `--resume`）。
先测试并对三个seed分别做20局独立诊断，然后从各自Task2父模型重建Task3。
正式开发使用60个世界、三seed确认各100个世界、唯一候选主验证100个世界。
最多运行A（仅修复）、P（16/32/16 Replay）、R（新奖励版本，仅kill5→15）
和条件允许的PR。首个开发联合达标套餐进入确认，确认或主验证失败停止。
不能续用受影响的旧Task3 Replay；父Task1/2 Replay保留的历史限制须随报告声明。
预算开始时间、种子审计与原阈值写入manifest；24小时截止，不自动推送或启动Task4。
结果 JSON 和自动报告位于带源码提交与manifest身份的campaign运行目录。
本段是执行合同，不是通过结果。术语及取舍见ADR0007、ADR0008。

本轮已完成，结论为**主验证失败**，见
[`task3-lifecycle-frozen-results.md`](docs/research/task3-lifecycle-frozen-results.md)。
源码`15d4721`下仅运行A臂：seed11/22/33分别训练150/250/150局，
选中c50/c150/c50，三个种子全部通过100世界独立确认。
唯一候选seed22/c150主验证得分7.52（父5.68）、击杀增加0.13、
第一名比例增加19个百分点，但世界19489出现一次逃生塌缩，违反零异常门槛。
按原协议停止，不更换候选；该权重不能标记为Task3合格。
完整机器证据及复现命令见`experiments/results/task3_lifecycle_20260917.json`。

后续诊断确认这次主验证计数将物理回退中的WAIT误记为安全替代动作。
`escape-collapse-v2`修正该证书条件，保持策略、权重和零异常阈值不变。
按新的`experiments/task3_counter_validation.json`进行冻结验证：同三个检查点
在21000–21099确认，原唯一候选seed22/c150在全部确认通过后才进入21100–21199主验证。
不重训、不重新选模；旧失败报告保留。设计边界见ADR0009，验证结果以新运行记录为准。

修正后的冻结验证已完成：三个种子确认及seed22/c150主验证全部通过，
Task3合格，可进入Task4。主验证得分增加1.23、第一名增加12个百分点，
击杀差值为-0.01；战斗门槛由第一名提升满足。Task4尚未启动。
详见[`修正验证报告`](docs/research/task3-counter-validation-results.md)。

报告的规定结构、篇幅、署名和提交边界见 [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)。本节只记录团队的写作分工和复现材料积累方式，不把报告拆成三个互不相关的个人成果。

| 成员 | 主要写作内容 |
|---|---|
| A | 游戏与状态表示、危险预测、特征设计、特征消融及案例 |
| B | 强化学习背景、两算法、奖励/回调/训练更新、模型对比 |
| C | 团队规划、实验方法、统计口径、奖励消融、综合结果和交付 |

每张图记录对应 run_id、配置、种子、检查点和生成命令；每项结论保留可回溯的原始数据、代码版本和主要记录人。

代码截止后的进一步实验可以写入报告，但必须区分参赛冻结版本和后续版本。模型没有稳定优于基线、消融没有改善、CPU 预算导致某实验未完成，都应如实说明，不用训练奖励或一次高分代替证据。

## 14. 参考依据与维护说明

1. 本届作业要求：[`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)，它是原始课程 PDF 的项目内静态摘要；课程公告、MaMPF 和原 PDF 保持更高权威性。
2. 当前项目源码：本指南第 2 节的文件，基线提交 `61b79ffa1f6976bd9f19eb922ca82ceed5ee7a9c`。发现更高权威来源发生变化时，先复核实现差异，再决定需要修订和重新验证的范围。
3. [往届参考仓库](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/tree/a7fe5041b02548ce4438e502ca3adb11576bae75)：用于理解设计思路。第三名是作者自述，不能据此证明某项特征、奖励或算法的具体收益；本项目不复制其解题代码或权重。
4. [PyTorch 官方 DQN 教程](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)：用于理解经验回放、目标网络、价值回归与训练结构，自行适配本课程接口并引用方法来源。

所有数值参数和工时都是初始计划，不是最优参数或已完成结果。本指南更新由 C 整合，涉及特征由 A 复核、涉及学习由 B 复核。最终完成度由检查记录和实验文件决定，不由本指南中的待办描述决定。


## Task 4 共享父模型有限训练（2026-09-17）

合同为 [`task4_campaign.json`](experiments/task4_campaign.json)，控制器为
[`task4_campaign.py`](experiments/task4_campaign.py)。实施开始于 2026-09-17 16:36:29 UTC，
硬截止为次日同一时刻；第 18 小时后不启动新臂，第 23 小时后不启动新评估批次。
本轮只交付代码、本地提交和冻结证据，不打包、上传或推送。

固定父权重 `agent_code/double_dqn_continuous_v2_agent/task3_validated.pt`
SHA-256 `b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d`。
三个学习种子 11/22/33 均从此父模型显式迁移，称为共享父模型复现。
迁移入口 `--transfer-task4-from-checkpoint` 与 resume/warm-start/其他迁移互斥；
同实验续训只用 `--resume-from`。保留 policy/target/Adam、累计动作与更新、旧 Replay；
teacher 替换为父 policy；Task 4 分区、阶段计数、回合暂态清空，学习 RNG 按新种子重置。
原 Task 1–3 Replay 保留不代表已经证明其历史数据不受此前生命周期问题影响。

84 维、r7、n-step 5、v5 安全、蒸馏 2、每任务 Replay 20000、当前任务 warmup 2000、
探索 0.3→0.05/120000 动作保持不变。A 每 batch 48 旧/16 新；B 为32/32，
旧样本来自合并父池而非旧任务固定配额。训练单进程单线程，最多六个独立物理核心。

先通过完整 unittest，再执行三个单独 RNG 的20局诊断（22411/22422/22433），
不接续正式训练；父模型混合对手诊断20世界；父模型 Task 1–4 开发基线各60世界。
诊断世界22000–22019，开发22100–22159，确认22200–22299，主验证22300–22399；
全部 worktree 配置/metadata 扫描证据及环境/对手 RNG 根种子记录在 manifest，
20000–20099继续封存。逐世界正常结束、关闭探索，父子世界与对手 RNG 配对；
保留完整 act 耗时、所有回放、死亡前状态与安全失败状态。

A 先 seed22，每50局不可变检查点、至多300局；首个全门槛合格点冻结，
然后收齐 seed11/33。只有所有失败种子各有“仅得分/战斗失败”的检查点才可开 B；
B 重新从原父模型开始，最多两臂。诊断排序为失败数、得分、第一名、击杀、较早轮次。
首个三种子联合达标臂进入100新世界确认；任何确认失败即停。全部确认通过，
按得分、自杀率、第一名、击杀、最低保留率、P95、种子编号选择唯一100世界主验证候选。
主验证失败即停，不换候选。10,000次世界配对 bootstrap 保留，不增加事后显著性要求。

门槛：Task 4 得分增量≥0.5且（击杀增量≥0.1或独占/并列第一增量≥5个百分点）；
Task 1得分、Task 2金币/炸箱、Task 3得分/金币/炸箱保留≥90%；
Task 2/3/4自杀≤5%、炸弹存活≥95%，Task 3/4零放弹局≤10%；
所有任务无效动作≤1%、完整act P95≤250ms/最大≤480ms，Task 1不放弹。
超时、跳过、可避免逃生塌缩、鲁棒保证丢失、安全搜索超时均须为零。
基线策略失败允许开 A；工程失败或未解释训练自杀立即停止，交付诊断，不修改安全机制。

从干净、已提交的 Task 4 worktree 启动：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python experiments/task4_campaign.py
```

仅基础设施中断可在完整身份一致时加 `--resume`，半成品保留，重试目录独立。
`runs/task4_shared_parent_20260917_<commit>/result.json` 记录当前阶段；
只有 `status=passed` 且 `task4_qualified=true` 可报告 Task 4 通过。

### 本轮 Task 4 实际结果

源码 `013fef5` 的完整测试402项通过（1项跳过），但逻辑seed22的独立诊断
（RNG22422）在第1局第57步触发 `robust_search_timed_out`。按合同终止，
A/B正式训练、父基线、确认与主验证均未开始，Task 4未通过。
完整act为406.534ms，已观察57次决策的部分样本P95为315.344ms；
未修改安全机制或放宽门槛。详见
[`Task 4冻结结果`](docs/research/task4-frozen-results.md)。


## 执行顺序等价归并准入（2026-09-17）

独立worktree从ae86fac开始，4小时预算为17:17:31–21:17:31 UTC。
保留全部对手，仅合并固定动作组合中无依赖动作的执行顺序；移动冲突、
移动与放弹冲突及两次放弹保留相对顺序。完整有序场景与首个证据须与旧枚举一致。
独立旧参考、256个合成状态、20个历史状态完整证明及完整unittest先通过，
再在同一物理核心交替测量各10次，运行预算仍为400ms。60秒仅用于离线正确性参考。

历史新实现仍超时则止于报告；通过后冻结新源码与独立manifest，
运行22422、22411、22433三个20局诊断，使用原A参数和相同Task 3父模型。
这些是可重用工程诊断数据，不是独立选模或验证集。P95≤250ms、最大≤480ms，
原零超时/跳过/保证丢失/可避免塌缩、责任与掩码不变量保持。
诊断失败即停，不扩展对手筛选，不接续正式Task 4，不打开开发/确认/主验证/封存世界。
控制入口为`experiments/order_equivalence_admission.py`；报告只可称优化准入通过、
优化未达标或预算内未完成验证，不能称Task 4合格。

### 等价归并实际结果

源码859ee69完整测试411项通过（跳过1项），256合成状态及20历史完整证明等价。
旧超时现场第57步枚举中位耗时310.57→17.29ms，优化后历史复测零超时。
真实RNG22422诊断完成3局后，在第4局第95步再次出现安全搜索超时，
按协议终止；另外两种子未启动。结论为“优化未达标”，不是Task 4合格。
详见[等价归并优化报告](docs/research/task4-order-equivalence-results.md)。


## 等价生存性传播与Task 4恢复（2026-09-17）

用户要求继续解决耗时并让Task 4正常进行。本轮从dc72b17独立分支，
用NumPy布尔移位实现与标量五动作递推等价的反向生存性传播；
WAIT保持原来的驻留许可，移动仍须满足下一步可进入约束，边界不环绕。
保留全部对手与400ms搜索预算；40个历史状态及128个合成证明完整差分通过后，
进行新旧各10次配对测量和完整unittest。前两次失败记录保持终态。
新Task 4协议使用新的manifest、配置身份和未使用世界区段，
先重新完成三个20局工程诊断再开基线与A/B；诊断学习不续接正式训练。
奖励、父权重、保留比例实验臂、门槛和确认失败停止规则均沿用原Task 4计划。
沿用原24小时总截止2026-09-18 16:36:29 UTC，包含此前实施耗时；
不延长预算，不推送、不打包。


首轮生存性向量化诊断在第15局第131步失败，记录见
`docs/research/task4-viability-first-results.md`，不继续该运行。
后续新增“同环境整张生存性图复用”：仅在同一次决策内跨本方候选位置复用
相同地形、危险与对手环境的完整图，各位置分别查询，不复用单个布尔结论。
60个已保存状态、128个合成状态和错误位置复用反例须与旧标量证明完全一致；
新旧配对测量以首轮向量化版本为对照。新协议`cached-viability-v2`保持
原预算截止、全部门槛与阶段顺序，实验ID、配置、源码、数据区段重新登记。


### Task 3 预测试提交包（2026-09-17）

已用Task3 seed22/c150原权重和等价加速运行时制作单Agent ZIP，3项打包测试及原始导入框架中对三个random_agent的3局CPU单线程测试通过。此状态取代旧章节中“打包工具尚未实现”的描述；Docker和官方机器验证仍未完成。详情与SHA见 `docs/research/task3-pretest-submission.md`；该包不宣称Task4合格。

### 持续安全修复：严格放弹证明

在独立safety-certified-placement分支引入survival-mask-v6，唯一行为改动是所有未获得完整证明的新放弹均被否决；无安全动作时仍由网络在物理合法非放弹动作间排序。保留v5源码行为、原父权重及已交付ZIP，计数和门槛不降级。先复现世界23008，再按登记的工程数据执行冻结回归；失败即结束该次运行，保存新现场后另建修复轮次。v5 Replay与恢复状态不直接视为v6兼容，当前仅使用显式冻结评估覆盖，不启动正式训练。

### Continuous safety repair: rearming regression

The second v6 engineering sweep stopped at Task3 world24266/step184; it cannot be resumed as a passing experiment. v7 keeps all opponents and the same budgets while adding conservative future rearming and a proof horizon of at least seven transitions. First require the recorded 182/DOWN regression, complete tests and historical timing, then run the recorded world as a new frozen engineering regression; subsequent tests need a separate frozen manifest. No v5/v6 Replay or training state is silently migrated, and this work does not qualify or train Task4.

### Continuous safety repair: fixed deadlines and movement preference

v8 replaces the rolling own-bomb proof horizon with a recorded placement deadline; its paired Task1–3 retention and 60-world Task4 engineering sweep passed, but a recorded opponent death exposed a nonpending movement-protection gap. Experimental v9 prefers completed movement proofs outside own responsibility; its first whole-game regression stopped at world24025/step75 on search timeout and remains terminal. The next source uses equivalent boolean reach unions, with a preserved named-set reference and unchanged actors, actions, horizons and budgets; only after full proof/timing checks may it repeat the recorded engineering world under a new manifest. These frozen repair studies do not authorize formal Task4 training or qualification, and the user has already submitted the archive, which must not be repackaged or replaced. See `docs/research/safety-reach-grid-optimization.md` and ADR0018–0019.

## Compact exact safety admission (2026-09-17)

The separate `safety-compact-admission` worktree implements ADR0020/0021 from `3eb3ea4`. Its budget is registered in `experiments/compact_admission_budget.json`. Run complete proof/ordered-scenario tests and historical/full-act preflight before freezing; then use only `experiments/compact_admission.py --manifest experiments/compact_admission_manifest.json`. This entrance cannot start a formal Task4 campaign. Frozen regression, paired retention, fresh 400-game safety assessment and 3×20 diagnostic training must all pass, including complete-act P95≤100ms/max≤250ms; first frozen failure terminates the attempt. Old Task4 campaigns, failed worlds and submitted files remain unchanged; source/contract changes require new identities, not continuation of stopped runs.

## Task 4 300ms readmission and background campaign (2026-09-18)

The new `task4-campaign-v2` supersedes the stopped compact admission only for future work. Its candidate complete-act limits are P95≤100ms/max≤300ms; the400ms search budget and all safety/performance gates remain. The previous262.60ms failure is unchanged. Manifest `experiments/task4_300ms_campaign.json` binds verified reused evidence, the new20-state regression, three independent20-round diagnostics and distinct v5 parent/v9 candidate configurations. No diagnostic checkpoint continues formal training. See ADR0022 and `docs/research/task4-300ms-campaign.md`.

After a clean source freeze, launch `experiments/task4_background.py --manifest experiments/task4_300ms_campaign.json` with the mle Python and OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1. It performs diagnostics22/11/33, v5 baselines, conditional A/B development, confirmation and one main validation. The detached owner records PID/state/logs; the first engineering or final validation failure terminates. Terminal evidence and a report are committed locally, never pushed. An infrastructure-only resume requires the same frozen source, manifest, weights and configs; terminal attempts cannot resume. Budget starts2026-09-18T11:17:18Z, ends2026-09-19T11:17:18Z, with18/23-hour cutoffs.

## Task4 observed-reference continuation (2026-09-18)

The300ms attempt passed all three20-round candidate diagnostics but stopped at v5 reference world24614/step107. The cause was an unproved fallback BOMB at106; existing v9 already prohibits it. Protocol `task4-campaign-v3`, manifest `experiments/task4_reference_campaign.json`, observes only guarantee loss or escape collapse linked to a recorded unproved fallback placement in the reference. Raw reference counters remain unchanged. Search/decision timeouts, invalid masks, unexplained guarantee loss, other simultaneous faults and all candidate100/300ms gates still stop the attempt. New data blocks24900/25000/25100/25200 are registered. The old diagnostics are hash-verified and reused as evidence only; formalA/B start from the original Task3 parent. The existing2026-09-19T11:17:18Z deadline is unchanged. Launch `experiments/task4_background.py --manifest experiments/task4_reference_campaign.json`; terminal evidence goes under `experiments/results/task4_reference_20260918/terminal`. See ADR0023.

### Task4 specialist exploration (separate protocol)

The user-authorized E1/E2/E3 specialist experiment drops Task1–3 retention and
optimizes official Task4 score, with first-place rate second. It uses policy-only
transfer, no old Replay or teacher, and preserves v9 engineering gates. See
[the specialist guide](docs/research/task4-specialist-exploration.md) and
[ADR0024](docs/adr/0024-task4-specialist-exploration.md). It does not replace the
historical curriculum qualification criteria or certify an old failed run.

### 冻结历史对手对照实验（2026-09-19）

下一轮探索采用 [固定终点 C/S 对照协议](docs/research/task4-frozen-opponents.md) 和
[ADR 0025](docs/adr/0025-frozen-historical-opponent-control.md)。该轮不沿用旧专项实验的
早期检查点选型，也不自动更新对手池；旧课程能力保留门槛不适用于这个专项问题。
