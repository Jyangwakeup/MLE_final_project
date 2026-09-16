> 当前版本用于团队内部开发，暂以中文维护；项目后期统一翻译为英文。

# 实验接口

本目录是实验编排层。其接口来自 `PROJECT_REQUIREMENTS.md` 与
`IMPLEMENTATION_GUIDE.md`；特征和学习模块的实现细节仍分别由 A 与 B 负责。

六个 Agent 当前推荐的完整训练、续训和冻结评估命令集中记录在
[`docs/training-commands.md`](../docs/training-commands.md)。本轮冻结特征 coin3 课程使用
`formal_training_coin3.json`；其他新模型实验可选择 `r2_balanced`、`r3_potential` 等独立配置。
Task 1 的最终 reward 对照使用
`task1_ddqn_continuous_r3_baseline.json` 与
`task1_ddqn_continuous_r5_conditional_loop.json`；二者除 Reward ID 外完全相同。
通用的 `reward_r5_conditional_loop.json` 将算法和 Feature 留空，可用于任一已注册学习 Agent。

## 可选学习 Agent

实验入口使用显式 Agent contract，不再根据目录名猜测算法：

| Agent | Algorithm | Feature | Checkpoint |
| --- | --- | --- | --- |
| `q_learning_agent` | `q_learning` | 默认 `discrete-q-v2`，可显式选 `discrete-v1` | `final.pkl` |
| `double_q_compact_agent` | `double_q_learning` | `discrete-compact-v1` | `final.pkl` |
| `dqn_agent` | `dqn` | 默认 `discrete-q-v2`，可选 `discrete-v1` / `discrete-objective-v1` | `final.pt` |
| `double_dqn_continuous_agent` | `double_dqn` | `continuous-v2`，84 维安全消融基线 | `final.pt` |
| `double_dqn_continuous_v2_agent` | `double_dqn` | 新训练使用 84 维 `continuous-v2`；自动只读兼容旧 78 维 checkpoint | `final.pt` |
| `double_dqn_continuous_v3_agent` | `double_dqn` | `continuous-v3`，显式安全余量消融 | `final.pt` |
| `double_dqn_continuous_v4_agent` | `double_dqn` | `continuous-v4`，连续 WAIT 与 2–8 步周期历史 | `final.pt` |
| `rainbow_lite_v5_agent` | `rainbow_lite` | `continuous-v5`，全局箱区密度与放弹后目标恢复 | `final.pt` |
| `cnn_double_dqn_agent` | `cnn_double_dqn` | `board-v1` | `final.pt` |
| `hybrid_dueling_double_dqn_agent` | `hybrid_dueling_double_dqn` | `hybrid-v1` | `final.pt` |

`reward_r2_balanced.json` 的 algorithm/feature 留空时由所选 Agent 契约填充；显式填写但不一致会在
启动 World 前失败。Task 1 的禁炸弹是单独记录的 curriculum action mask，Feature 返回的
`legal_mask` 仍只描述物理合法性。

训练 checkpoint 保存完整可恢复状态。生成最终单目录提交包使用：

```bash
python3 -m experiments.package_agent \
  --agent hybrid_dueling_double_dqn_agent \
  --checkpoint runs/<run>/checkpoints/final.pt \
  --output dist/final-project-agent-code.zip
```

构建器校验算法、Feature schema、动作顺序、网络和超参数，并在生成目录中 vendor 公共
Feature/reward/学习运行时；不会在四个开发目录中维护四份公共源码，也不会覆盖已有 zip。

## 代码结构与协作边界

实验命令保持单一入口，但训练与评估分别开发：

```text
experiments/
  run.py                 CLI、公共 session、World 扩展和 metadata
  training.py            训练模式参数、训练调度和早停策略
  evaluation.py          单 seed / 多 seed 冻结评估调度
  analyze_training.py    训练指标汇总与训练曲线
  analyze.py             评估指标汇总与对比图
  resume.py              两代原子完整恢复及课程约束
  devices.py             CPU/CUDA 设备解析与元数据
  configs/               工程默认、正式训练、阶段门槛、验证和最终测试
  configs/reward_r2_balanced.json  R2 balanced 实验配置
```

`run.py` 是唯一的命令行入口，通过 `--mode train|evaluate` 分发到
`training.py` 或 `evaluation.py`。两种模式复用 `run_agent_session()`，从而保证 World
构造、随机种子、日志、checkpoint 路径、回放和 metadata 格式一致。不要在训练和评估
模块中各自复制 session 初始化逻辑。

团队可以按文件划分开发责任：训练负责人主要维护 `training.py` 和
`analyze_training.py`；评估负责人主要维护 `evaluation.py` 和 `analyze.py`；修改
`run.py` 的公共契约时双方共同确认。对外命令保持不变，因此拆分模块不会影响已有实验脚本。

## C 的职责

C 负责实验编排、评估、原始结果记录、分析、可复现性、消融配置和最终打包。这些
职责必须使用官方框架接口，且不修改官方核心文件。

## 已冻结接口

### A/C seam

特征接口由 `agent_code/team_agent/feature_system/` 负责；
`agent_code/team_agent/features.py` 是冻结 `discrete-v1` 的旧路径兼容层。C 可以使用：

- 当前 Agent 的 `FEATURE_ID = "discrete-v1"` 和旧别名 `FEATURE_VERSION = "v1"`；
- registry 的 `extract_features(game_state, feature_id)` 与 `get_feature_schema()`；
- 各输出 dataclass 的 `legal_mask`；
- `discrete-v1` 的 14 元 state / 40 维向量；以及
- 固定六动作顺序的 `ACTIONS`。

C 不复制这些定义、不改变特征语义，也不依赖特征实现细节。新的特征变体（包括危险
消融）必须由 A 通过相同特征接口提供并版本化。在实验配置选用计划中的
其他 Feature ID 之前，必须确认相应 Agent 模型输入结构已经实现。

### B/C seam

C 仅通过配置、环境变量、官方 agent 回调和 checkpoint 路径驱动独立的
`q_learning_agent` 与 `dqn_agent`；两者共享 `team_agent` 中的特征、奖励和探索协议。C 不访问或重新定义 `Transition`、Q-table、网络、replay buffer、
学习更新或其他学习实现细节。

C 需要 B 提供的最小信息是：

- 算法标识符；
- 明确的训练或评估模式；
- checkpoint 输入和输出路径；
- checkpoint 版本元数据；以及
- 训练指标的输出位置。

当前完整恢复 schema 为 `training-resume-v7`。Runner 只校验公开 checkpoint 合同，
不导入 `agent_code/q_learning_agent/` 或 `agent_code/dqn_agent/` 的私有学习实现。

### 运行时配置

- `BOMBERMAN_CONFIG` 指向一个绝对路径的完整实验配置。未设置时，agent 仅使用包内
  默认配置。配置必须自包含；第一版不支持隐式继承或多层合并。
- `BOMBERMAN_RUN_DIR` 指向绝对路径的运行输出目录。训练输出从该目录解析。
- `BOMBERMAN_AGENT_SEED` 与 `BOMBERMAN_EXPLORATION_SPEC` 由 Runner 设置；直接使用官方
  框架时分别回退为 seed 0 和共享 `linear-v1` 默认值。
- `BOMBERMAN_SAFETY_SPEC` 保存版本化 Safety 合同；旧 `BOMBERMAN_SAFE_EXPLORATION`
  仅映射为 `mode=exploration`，新旧配置不得同时出现。`BOMBERMAN_N_STEP`、
  `BOMBERMAN_RETENTION_SPEC` 和 `BOMBERMAN_TRAINING_BUDGET` 保存其他阶段契约。
- 包内配置中的模型路径相对 agent 包解析。导出配置与模型路径不得包含开发机器的
  绝对路径。

运行器将在执行前保存实际使用的完整配置及其 SHA256。恢复或评估模式缺少
checkpoint 时应报错，而不是训练或回退到随机动作。

### 随机种子

评估模式对 CLI 的实验种子 `S` 固定使用：

- `experiment_seed = S`；
- `environment_seed = S`，传给官方 world 的独立 RNG；
- `official_opponent_seed = S + 100000`，用于官方 agent 共享的 Python 与 NumPy RNG。

官方 agent 的共享 RNG 在创建 world 前设置一次，并在所有 agent `setup` 后、第一局
开始前再次设置一次；不会在每个 `act()` 中重置。训练模式使用
`environment_seed = 1000 + S` 和 `official_opponent_seed = 3000 + S`；未显式传入
`--seed` 时使用配置中的 seed，配置也未指定时默认使用 `11`。
同一 `S` 还用于 Agent 自有 RNG 与 DQN 参数初始化，且写入 metadata/checkpoint。

### 运行产物

每次运行独占一个目录，目标布局为：

```text
runs/<run_id>/
  metadata.json
  episodes.jsonl
  training.csv
  training_summary.json
  training_progress.png
  timing.jsonl
  official_stats.json
  checkpoints/
  resume/                  # 训练 run 的最新两代完整快照
  replays/                 # 按回放策略可选生成
```

`metadata.json` 记录展开后的配置、代码/源码身份、依赖、硬件、种子、模式、时间戳、
`algorithm`、`feature_id`、完整 `feature_schema`、`reward_id`、`action_order`、checkpoint
和终态状态；`termination` 进一步记录计划局数、实际完成局数及早停结果。JSONL 文件是
追加式原始记录。`official_stats.json` 是用于交叉检查的官方
框架导出。`checkpoints/`、`resume/` 与 `replays/` 只属于对应运行。运行目录不得覆盖已有运行。

`episodes.jsonl` 当前使用 `episode-v1` schema，每行对应一个已完整结束的 round：

- round 字段：`run_id`、`round_index`、环境 `seed`、`scenario`、`stage` 和
  `environment_seed`、`round_steps`；
- `agents` 中每名 agent 的字段：`name`、`score`、`coins`、`kills`、`suicides`、
  `crates`、`bombs`、`invalid`、`survived`、`dead`、`death_step` 和
  `death_causes`、`killed_by_self`、`killed_by_opponent`。`death_causes` 会记录命中
  该 agent 的危险爆炸来源，可区分 `self_bomb`、`opponent_bomb` 及同一步多重命中。

逐局的 `score`、`coins`、`kills`、`suicides`、`crates`、`bombs` 和 `invalid` 可按
agent 累加，并与 `official_stats.json` 的 `by_agent` 交叉检查。训练模式会持续保存
checkpoint 和训练指标；评估模式要求 checkpoint 已存在，并且不会更新模型。

`timing.jsonl` 当前使用 `timing-v1` schema，每行对应一次 agent 决策机会，记录
`run_id`、`round_index`、`step`、`agent_name`、最终执行的 `action`、agent 请求的
`requested_action`、`think_time`、是否 `timed_out`、是否因上一轮超时被 `skipped`，
以及前后的可用思考时间。配置启用 `evaluation.navigation_diagnostics` 时，还在动作已经返回后
追加只读 `navigation` 对象：最近金币距离及静态预测的动作后距离、是否缩短距离、是否
立即反向或等待、是否存在多个等距最近金币，以及旧目标仍存在时是否切换目标。诊断不读取
或修改 Agent RNG，不过滤或替换动作。

Q-table 泛化诊断通过可选的 `q_diagnostics.jsonl` 汇总。现有 `q_learning_agent` 在设置
`BOMBERMAN_RUN_DIR` 时会逐决策追加 `q_decisions` 和 `unseen_q_states`；非 Q-table
agent 或未上报的运行在分析结果中对应字段为空或为 0。

### 分析输出

`experiments/analyze.py` 只读取一个或多个运行目录中的原始
`episodes.jsonl`，每个 `(run_id, agent_name)` 输出一行可追溯的汇总，并在表尾追加
`run_id=AVERAGE` 的跨 seed 平均行，全部写入指定目录的 `summary.csv`，同时生成
`mean_score.png`。它报告总分、总分排名、平均分、总体标准差、
金币/击杀/自杀/被击杀/箱子/无效动作的逐运行累计值与均值或比例、存活率、平均存活步数、
独占第一/并列第一/全体零分平局的次数与比例、`act()` 平均/P95/最大耗时、超时和跳过次数，
以及可选的未见 Q 状态比例。Task 1 导航诊断还汇总每 100 步金币、每枚金币步数、全金币
完成步数、等待率、立即反向率、缩短距离率和目标切换率。输入缺少 `episodes.jsonl`、空文件、非法 JSON 或不符合
schema 时会明确失败。

所有 Task 的汇总还报告 score 中位数和均值的 95% 置信区间、六种动作各自的计数与占比，
以及逐局最长连续 `WAIT` 的中位数、最小值和最大值。通用
`wait_action_rate_all_actions` 直接以 `timing.jsonl` 的全部动作作为分母，因此即使没有启用
Task 1 的可选导航诊断也不会为空；原有 `wait_action_rate` 仍严格表示导航诊断决策中的等待率。
跨候选的同 seed 比较使用 `experiments/compare_evaluations.py`，先按
`(environment_seed, round_index)` 核对完整配对，再以环境 seed 为独立重复单位聚合，输出
均值差、中位数差、候选胜/平/负 seed 数和确定性 bootstrap 95% 区间。Task 1 比较
score、coins 和 round steps；Task 2 另比较 crates、
suicides 与生存指标；Task 3/4 再加入 kills 和胜局指标。

```bash
python3 -m experiments.compare_evaluations \
  --candidate-runs runs/candidate/candidate_s* \
  --reference-runs runs/reference/reference_s* \
  --output results/candidate_vs_reference
```

## 运行命令

所有命令均在项目根目录执行。框架只有两个彼此独立的模式：`train` 负责训练和训练报告，
`evaluate` 负责检查已有 checkpoint、冻结模型、测试和评估报告。运行目录禁止覆盖，重复
实验必须使用新的 `--run-id`。

实验运行显示固定 80 列的 `tqdm` 进度条，在同一行原地更新，避免 IDE 误判终端宽度后
因进度条过长而换行滚屏。训练结果同时持续写入 `training.csv`、`metadata.json` 和日志。

### GUI 与实验回放

`experiments/run.py` 面向批量训练和可复现评估，当前固定使用无 GUI 模式，以减少渲染开销
并提高训练速度。因此该入口不提供 `--gui` 参数。GUI 功能仍由项目根目录的 `main.py`
提供，没有被删除。

开发期间如需实时观察 Agent，可以使用官方运行入口：

```bash
python3 main.py play \
  --agents q_learning_agent \
  --scenario coin-heaven \
  --n-rounds 1
```
BOMBERMAN_ALLOW_BOMB=false python3 main.py play \
  --agents dqn_agent \
  --scenario coin-heaven \
  --n-rounds 1

这种方式适合人工检查行为，但不替代 `experiments/run.py --mode evaluate` 产生的固定 seed
评估结果。若要查看实验保存的某一局，使用 replay：

```bash
python3 main.py replay \
  runs/<run_id>/replays/round_00001.pt
```

只有对应运行的回放策略实际保存了该局时，replay 文件才会存在。训练模式默认使用
`sampled`，多 seed 评估默认使用 `all`，单 seed 快速评估默认使用 `none`；具体规则见下一节。

### 回放保存策略

运行器通过 `--replay-policy auto|none|failures|sampled|all` 控制回放。默认的 `auto`
会根据运行类型选择：训练使用 `sampled`，固定多 seed 评估使用 `all`，单 seed 快速测试
使用 `none`。

- `none`：不保存回放；
- `failures`：目标 Agent 死亡、自杀或产生无效动作时保存；
- `sampled`：保存第一局以及每隔 `--replay-interval` 局；
- `all`：保存每一局。

训练模式不再使用固定抽样间隔，而是按训练进度里程碑保存。默认在完成计划总局数的
10%、20%……100% 时保存，即间隔为 `ceil(n_rounds × 10%)`，并额外保留第 1 局。例如
训练 1,000 局时每 100 局保存一次，训练 10,000 局时每 1,000 局保存一次。可用
`--replay-interval` 显式覆盖进度比例生成的间隔。回放写入每个运行目录的
`replays/round_XXXXX.pt`，`replays/manifest.jsonl` 同时记录 round、seed、目标 Agent
得分以及保存原因。训练不会因此保存全部回放；阶段门槛、主验证和最终测试均采用多个
独立 seed、每 seed 多局，并保留全部评估回放以复核高分和失败案例。

查看某一局时运行：

```bash
python3 main.py replay runs/q_task1_eval/q_task1_eval_s10001/replays/round_00001.pt
```

### `run-id` 和自动后缀

`--run-id` 是你给一次实验取的唯一名称，也是 `runs/` 下输出目录名称的基础。它不是
Agent 名、模型文件名或 Task 名，建议主动把关键信息写进去，例如：

```text
q_task1_train_s11
dqn_task3_eval_v2
q_task4_rule_eval
```

运行器根据模式自动生成以下名称：

| 运行方式 | 输入的 `--run-id` | 实际目录 | 后缀来源 |
|---|---|---|---|
| 训练 | `q_task1_train` | `runs/q_task1_train/` | 不添加后缀 |
| 单 seed 评估 | `q_smoke` | `runs/q_smoke/` | 不添加后缀 |
| 多 seed 评估 | `q_task1_eval` | `runs/q_task1_eval/q_task1_eval_s10001/` 等 | `_s<seed>` 表示该目录使用的环境 seed |
| 多 seed 汇总 | `q_task1_eval` | `runs/q_task1_eval/q_task1_eval_summary/` | `_summary` 表示所有 seed 的合并报告 |

例如传入：

```bash
--seeds 10001 10002 --run-id q_task1_eval
```

会生成：

```text
runs/q_task1_eval/
  q_task1_eval_s10001/
  q_task1_eval_s10002/
  q_task1_eval_summary/
```

`_s10001` 和 `_s10002` 来自 `--seeds`；如果没有显式传入 `--seeds`，则来自
`reward_r2_balanced.json` 的 `evaluation.seeds`。`summary.csv` 中的 `run_id=AVERAGE` 是跨 seed
平均值所在的数据行，不是另一个运行目录。

训练 checkpoint 固定放在该训练目录的 `checkpoints/` 中。Q-learning 默认命名为
`final.pkl`，名称中包含 `dqn` 的 Agent 默认命名为 `final.pt`。评估不会复制或改写
checkpoint，只在 `metadata.json` 和 `fixed_evaluation.json` 中记录其绝对路径。

训练热路径默认启用不改变学习语义的 I/O 优化：训练 CSV 只增量读取新增行，逐步 timing
记录复用文件流，神经 replay 以列式 Tensor 写入 checkpoint，两代神经恢复快照在同一文件
系统上使用硬链接固定原子 checkpoint 的 inode（不支持时回退复制）。checkpoint 仍在每个
完整回合边界生成，恢复粒度与更新频率不变。

`run-id` 只能是一个目录名称，不能包含 `/`，也不能是 `.` 或 `..`。已有同名目录时运行器
会拒绝覆盖，因此重新运行时需要使用新名称，例如在末尾增加 `_v2`。

### 四个 Task

| Task | 场景 | 默认参与者 |
|---|---|---|
| `1` | `coin-heaven` | 目标 Agent，无对手 |
| `2` | `classic` | 目标 Agent，无对手 |
| `3` | `classic` | 目标 Agent、`peaceful_agent`、`coin_collector_agent` |
| `4` | `classic` | 目标 Agent、三名 `rule_based_agent`；可用 `--opponents` 覆盖 |

Task 1 和 Task 2 不接受对手；Task 3 的两个官方对手固定，是对官方 SHOULD 课程路线的
合并实现；Task 4 至少需要一个对手，正式训练和验证均使用默认三名规则对手。

正式六链由 Q-learning/DQN × seeds 11/22/33 构成，阶段新增预算依次为 500、1,000、
1,500、3,000 局；能力门槛失败只可按原预算 25% 追加一次，即 125、250、375、750 局。
各链独立晋级，链内严格串行。run-id 使用
`formal_<q|dqn>_discrete_v1_r1_coin3_s<seed>_t<task>_r<local-rounds>`，故障重跑追加 `_retryN`。

### 单独训练

先从仓库根目录创建统一训练环境：

```bash
conda env create -f environment.yml
conda run --no-capture-output -n mle python -m unittest discover -s tests
```

`mle` 使用 CUDA 11.8 版 PyTorch；DQN 训练可用 `--device cuda` 明确选择第一张
`CUDA_VISIBLE_DEVICES` 内可见的 GPU。`--device auto` 在 CUDA 可用时选择 GPU，
否则回退 CPU。Q-learning 始终使用 CPU，冻结评估也强制使用 CPU，以匹配官方环境。

本轮正式训练固定 CPU、`discrete-v1`、`r1_coin3`、关闭 early stopping。以下启动 Q-learning seed 11 的
Task 1 共 500 局；实验运行器会自动禁止放炸弹：

```bash
python3 experiments/run.py \
  --config experiments/configs/formal_training_coin3.json \
  --mode train \
  --device cpu \
  --task 1 \
  --agent q_learning_agent \
  --n-rounds 500 \
  --seed 11 \
  --run-id formal_q_discrete_v1_r1_coin3_s11_t1_r500
```

训练输出：

```text
runs/formal_q_discrete_v1_r1_coin3_s11_t1_r500/
  metadata.json
  episodes.jsonl
  timing.jsonl
  training.csv
  training_summary.json
  training_progress.png
  official_stats.json
  checkpoints/final.pkl
  resume/latest.json
  resume/generation-00000500/
```

`training.csv` 每个训练回合写一行，统一包含 `round`、`reward`、`action_steps`、
`epsilon`、`q_states`、`loss`、`updates` 和 `checkpoint`。Q-learning 使用
`q_states`，DQN 使用 `loss` 与 `updates`；不适用的列留空。`training_summary.json` 保存
最终步数、平均 reward、最近 100 局平均 reward 和最佳回合，`training_progress.png` 展示
reward 与 epsilon 的变化。

### Task 1 多 seed 训练分析

`analyze_training.py` 的 `analyze_training(run_directory)` 保留为单 run 的训练结束报告。
跨 seed 分析只读取完整 run 的 `metadata.json`、`episodes.jsonl` 和 `training.csv`，绝不
修改 `runs/`。例如六条 Task 1 正式训练记录可以这样聚合：

```bash
python experiments/analyze_training.py \
  --runs \
    runs/formal_q_v1_r1_s11_t1_r500 \
    runs/formal_q_v1_r1_s22_t1_r500 \
    runs/formal_q_v1_r1_s33_t1_r500 \
    runs/formal_dqn_v1_r1_s11_t1_r500 \
    runs/formal_dqn_v1_r1_s22_t1_r500 \
    runs/formal_dqn_v1_r1_s33_t1_r500 \
  --output results/task1_analysis \
  --rolling-window 25
```

报告目录包含逐局的 `round_metrics.csv`、每个 seed 的 `run_summary.csv`、以 seed 为
重复单位的 `algorithm_summary.csv`，以及记录输入、版本、预算、窗口和统计口径的
`analysis_metadata.json`。其 PNG 均从逐局原始记录重新生成。rolling mean 是包含当前局的
尾随窗口；算法均值和 sample SD 先在每个 seed 内汇总，**不会**把所有 round 当作独立训练
样本。Task 1 的 `score` 与 `coins` 相同，故只绘制“每局结束时的 coins”，不把它误写成
局内逐 step 的金币曲线。

输入会验证 round 对齐、有限数值、Task、feature/reward version、预算以及
`(algorithm, seed)` 唯一性。可选的 `--evaluation-runs` 会在输出目录下另存冻结评估汇总；
训练曲线仅用于诊断，是否进入下一 Task 仍必须由独立的 stage-gate evaluation 决定。

### 续训与课程晋级

`--resume-from` 指向父训练 run 的根目录，并总是写入一个新的子 run。`--n-rounds`
表示子 run 新增的局数。例如，Task 1 能力门槛失败后唯一允许的 25% 追加为 125 局：

```bash
python3 experiments/run.py \
  --config experiments/configs/formal_training_coin3.json \
  --mode train \
  --device cpu \
  --task 1 \
  --agent q_learning_agent \
  --n-rounds 125 \
  --seed 11 \
  --resume-from runs/formal_q_discrete_v1_r1_coin3_s11_t1_r500 \
  --run-id formal_q_discrete_v1_r1_coin3_s11_t1_r125_cont125
```

进入直接下一 Task 时命令相同，只需修改 `--task` 和新 `--run-id`。同 Task 恢复世界、
对手、Agent 和早停状态，回合编号连续；下一 Task 保留全部学习器状态和 epsilon 进度，
但按同一 seed 重建环境/对手随机流、从回合 1 开始并重置早停。只允许同 Task 或
`1→2→3→4`，且算法、训练/Agent seed、动作顺序、特征版本、奖励版本、完整奖励表、
训练设备、`source_commit`、`source_hash` 和 resume schema 必须一致。探索、n-step、回放保留和预算只可按直接 Task 晋级的预注册合同改变。
当前 schema 为 `training-resume-v7`；v1–v6 及冻结 final checkpoint 默认不能精确续训。完整 v6 Task 1 run 只能通过显式 `--migrate-resume-from` 建立 v7 子 run；普通 `--resume-from` 不自动迁移。
同 Task 的早停配置也必须保持一致。

每回合边界发布一代完整快照，`resume/` 只保留最新两代。最新一代校验失败时自动回退
上一代，丢失局数记录在子 run 的 `metadata.json`。`--checkpoint` 仅用于冻结评估；旧
checkpoint 仍可评估，但不能替代 `--resume-from`。

### 开发期权重初始化与严格续训

开发新奖励或 Task 2–4 行为时，可用 `--init-from-checkpoint` 从已有神经网络 checkpoint
继承**仅 policy 权重**，避免每次都重训 Task 1：

```bash
.venv/bin/python -m experiments.run \
  --config experiments/configs/reward_r5_conditional_loop.json \
  --mode train --device cpu --task 2 \
  --agent double_dqn_continuous_agent --seed 11 --n-rounds 500 \
  --init-from-checkpoint runs/ddqn_continuous_v2_r5_s11_t1_train/checkpoints/final.pt \
  --run-id dev_ddqn_continuous_v2_r5_s11_t2_warm
```

`--init-from-checkpoint` 是开发期 warm start：允许更换 reward，但要求算法、Feature schema、
动作顺序和网络结构兼容；policy 权重被复制到新 policy 和 target，而 optimizer、replay、
epsilon/action steps、随机状态、早停和回合状态全部从零开始。run metadata 的 `lineage.kind`
记录为 `warm_start`。它只支持神经网络 Agent，不能与 `--resume-from` 同时使用，也不能用于
冻结评估。

`--resume-from` 是严格续训/课程晋级：从父 run 的 resume snapshot 恢复学习器、optimizer、
replay、探索进度和必要的运行状态，并执行完整兼容性校验。正式实验链和最终结果必须使用
`--resume-from` 从同一配置链逐级训练；warm-start 结果只能作为开发诊断或迁移学习实验，
不能伪装成严格连续训练结果。

训练模式默认启用基于 reward 移动平均的双重条件早停；未提供配置时使用 window 200、
patience 100、min_rounds 300、min_delta 0.1、空 target_reward，并要求 action steps 达到
exploration 的衰减步数。可在配置中显式覆盖：

```json
"training": {
  "n_rounds": 10000,
  "early_stopping": {
    "enabled": true,
    "window": 200,
    "patience": 100,
    "min_rounds": 300,
    "min_delta": 0.1,
    "target_reward": 49.0,
    "require_exploration_complete": true
  }
}
```

`patience` 表示移动平均连续多少轮未提高至少 `min_delta` 后停止；默认还要求探索衰减完成，
并达到 `min_rounds` 和可选的 `target_reward`。省略或设为 `null` 的
`target_reward` 会允许低奖励平台触发早停。实际完成轮数和停止原因写入
`metadata.json` 的 `termination` 字段。该配置仅对 `train` 模式生效。

不得从零直接启动 Task 3。进入下一阶段必须引用直接父 run，例如 Task 1 晋级 Task 2：

```bash
python3 experiments/run.py \
  --config experiments/configs/formal_training_coin3.json \
  --mode train \
  --device cpu \
  --task 2 \
  --agent q_learning_agent \
  --n-rounds 1000 \
  --seed 11 \
  --resume-from runs/formal_q_discrete_v1_r1_coin3_s11_t1_r500 \
  --run-id formal_q_discrete_v1_r1_coin3_s11_t2_r1000
```

### 单独测试和评估

测试开始前必须用 `--checkpoint` 指定已有模型。路径不存在时立即失败，不创建测试结果目录，
也不会使用随机模型代替。下面评估已有 Q-table 在 Task 1 的表现：

```bash
python3 experiments/run.py \
  --config experiments/configs/stage_gate.json \
  --mode evaluate \
  --task 1 \
  --agent q_learning_agent \
  --checkpoint runs/formal_q_discrete_v1_r1_coin3_s11_t1_r500/checkpoints/final.pkl \
  --run-id gate_formal_q_discrete_v1_r1_coin3_s11_t1
```

该命令按配置中的全部固定 seeds 运行，并生成：

```text
runs/gate_formal_q_v1_r1_s11_t1/gate_formal_q_v1_r1_s11_t1_s10000/
...
runs/gate_formal_q_v1_r1_s11_t1/gate_formal_q_v1_r1_s11_t1_summary/
  summary.csv
  mean_score.png
  fixed_evaluation.json
```

Task 1 还必须运行使用同一合法动作掩码的确定性均匀随机诊断基线：

```bash
python3 experiments/run.py \
  --config experiments/configs/stage_gate.json \
  --config experiments/configs/reward_r2_balanced.json \
  --mode evaluate \
  --task 1 \
  --agent legal_random_agent \
  --checkpoint agent_code/legal_random_agent/baseline.json \
  --run-id gate_legal_random_t1
```

课程自带的四个非学习基线也可以作为评估主体运行。它们没有模型参数，因此不要传
`--checkpoint`；其余 episode、timing、Task 1 导航诊断及汇总输出与学习 Agent 相同：

```bash
for agent in random_agent rule_based_agent peaceful_agent coin_collector_agent; do
  python3 experiments/run.py \
    --config experiments/configs/stage_gate_coin3.json \
    --mode evaluate \
    --task 1 \
    --agent "$agent" \
    --run-id "gate_${agent}_t1"
done
```

阶段门槛使用 `stage_gate.json` 的 10000–10004、每 seed 20 局；主验证使用
`main_validation.json` 的 10000–10099，最终测试只使用一次 `final_test.json` 的
20000–20099，后二者都是每 seed 一局。
默认 seeds 和每个 seed 的测试局数来自 `reward_r2_balanced.json` 的 `evaluation`。也可用
`--seeds 10001 10002` 和 `--n-rounds 5` 临时覆盖。

### 单 Seed 快速测试

开发期间可先运行一局，检查 checkpoint 能否加载和 Agent 是否报错：

```bash
python3 experiments/run.py \
  --config experiments/configs/reward_r2_balanced.json \
  --mode evaluate \
  --task 1 \
  --agent q_learning_agent \
  --checkpoint runs/q_coin_train/checkpoints/final.pkl \
  --seed 10001 \
  --n-rounds 1 \
  --run-id q_smoke
```

单 seed 模式直接写入 `runs/q_smoke/`，并在 `runs/q_smoke/summary/` 生成该次测试的
`summary.csv` 和图。完整模型比较仍应使用上一节的固定多 seed 评估。

### Task 4 自定义对手

```bash
python3 experiments/run.py \
  --config experiments/configs/reward_r2_balanced.json \
  --mode evaluate \
  --task 4 \
  --agent dqn_agent \
  --opponents rule_based_agent q_learning_agent peaceful_agent \
  --checkpoint runs/dqn_task4_train/checkpoints/final.pt \
  --run-id dqn_task4_eval
```

### DQN

DQN 使用相同正式配置，只需更换 Agent 名称。checkpoint 扩展名为 `.pt`：

```bash
python3 experiments/run.py \
  --config experiments/configs/formal_training_coin3.json \
  --mode train \
  --device cpu \
  --task 1 \
  --agent dqn_agent \
  --n-rounds 500 \
  --seed 11 \
  --run-id formal_dqn_discrete_v1_r1_coin3_s11_t1_r500
```

### 常用参数

| 参数 | 含义 |
|---|---|
| `--mode train` | 只训练目标 Agent |
| `--mode evaluate` | 冻结并评估已有 checkpoint |
| `--task` | 选择 PDF 定义的 Task 1、2、3 或 4 |
| `--agent` | 目标 Agent 目录名 |
| `--n-rounds` | 训练局数，或每个评估 seed 的测试局数 |
| `--seed` | 单次运行 seed；评估时也会切换为单 seed 模式 |
| `--seeds` | 多 seed 评估列表 |
| `--checkpoint` | `evaluate` 模式加载的模型路径 |
| `--resume-from` | `train` 模式使用的父 run 根目录；创建新的子 run |
| `--device` | DQN 训练设备：`auto`、`cpu` 或 `cuda`；评估固定为 CPU |
| `--opponents` | 覆盖 Task 4 的默认对手列表 |
| `--run-id` | 本次实验的唯一输出名称 |
| `--replay-policy` | 回放策略；默认 `auto` |
| `--replay-interval` | `sampled` 策略的周期，默认 500 局 |

`--task 1` 会在训练和评估中自动设置 `BOMBERMAN_ALLOW_BOMB=false`，当前的
Q-learning 与 DQN agent 都会据此屏蔽炸弹。Task 2–4 会自动允许炸弹。

## 基础配置

`configs/reward_r2_balanced.json` 是完整 JSON 文档，不是继承片段。`algorithm`、`seed` 和
`checkpoint` 为 `null`，表示具体运行可通过命令行显式选择值。训练过程使用
`training-v1` CSV schema，固定评估的逐局记录使用 `episode-v1`，决策耗时使用
`timing-v1`。

`configs/formal_training.json` 保留 CPU、`discrete-v1`、`r1` 基线；
`configs/formal_training_coin3.json` 保留旧 v4 固定局数证据。`task2_safety_diagnostic.json` 对既有 v5 checkpoint 仅启用冻结 shield；三个 `safety_ablation_*_task1.json` 使用同步冻结分数停止：第200局起每50局在9000–9019上检查，连续三次 `mean_score≥48` 停止，累计上限1000局。10000–10019只用于独立晋级门槛。Task 2 仍使用150000动作/至少500局。`final_test.json` 的20000–20099继续封存。
奖励实验使用独立配置文件：

```text
experiments/configs/reward_r2_balanced.json  R2 balanced
experiments/configs/reward_r3_potential.json 状态势能塑形
```

Reward ID 和完整 spec 都是 checkpoint 与 resume 契约的一部分。更换 Reward 必须开始新的训练链，不能把不同 Reward 的 checkpoint 当作同配置续训。

| Reward ID | 定位 | 建议用途 |
|---|---|---|
| `r1` | 金币 +1、击杀 +5、炸箱 +0.2 | 冻结基线与旧实验 |
| `r1_no_crate` | 金币 +1、炸箱 0 | `r1` 的严格炸箱消融 |
| `r1_coin3` | 金币 +3、击杀 +5、炸箱 +0.2 | 本轮正式课程 |
| `r1_coin3_no_crate` | 金币 +3、炸箱 0 | coin3 课程的严格炸箱消融 |
| `r2_balanced` | 金币 +3，并重新平衡发现、炸箱、死亡、存活和非法动作 | Task 2–4 的事件奖励实验 |
| `r3_potential` | `r2_balanced` 加金币/箱子路径与安全势能 | Task 1 导航和循环问题，或需要密集反馈的实验 |
| `r5_coin_potential` | r1_coin3 + 金币势 | Task 1 受控矩阵 |
| `r6_safe_sparse/potential` | 箱区/危险势、炸弹信用与分离死亡代价 | Task 2 成对安全比较 |
| `r7_safe_credit_sparse/potential` | 不可逃放弹 −20 | 第三轮自杀失败分支 |
| `r8_safe_constrained` | r7 sparse 去掉预测放弹正奖；可避免必死动作 −20；自杀压制同帧正奖 | 生存约束精确信用消融 |

完整数值、势能公式和边界语义见
[`agent_code/team_agent/README.md`](../agent_code/team_agent/README.md#公共奖励)。不同 Reward 的
训练 reward 尺度不同，不能仅按曲线高低选模型，应统一做冻结评估。

使用默认 `r2_balanced`：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode train --task 1 \
  --agent double_q_compact_agent --seed 11 --n-rounds 2000 \
  --run-id double_q_r2_t1
```

使用 `r3_potential` 时换用对应配置，并使用新的 run ID：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json --mode train --task 1 \
  --agent double_q_compact_agent --seed 11 --n-rounds 2000 \
  --run-id double_q_r3_t1
```
