> 当前版本用于团队内部开发，暂以中文维护；项目后期统一翻译为英文。

# 实验接口

本目录是实验编排层。其接口来自 `PROJECT_REQUIREMENTS.md` 与
`IMPLEMENTATION_GUIDE.md`；特征和学习模块的实现细节仍分别由 A 与 B 负责。

## C 的职责

C 负责实验编排、评估、原始结果记录、分析、可复现性、消融配置和最终打包。这些
职责必须使用官方框架接口，且不修改官方核心文件。

## 已冻结接口

### A/C seam

特征接口由 `agent_code/team_agent/features.py` 负责。C 可以使用：

- 当前为 `v1` 的 `FEATURE_VERSION`；
- `Features` 和 `extract_features(game_state)`；
- `Features.legal_mask`；
- 40 维特征向量；以及
- 固定六动作顺序的 `ACTIONS`。

C 不复制这些定义、不改变特征语义，也不依赖特征实现细节。新的特征变体（包括危险
消融）必须由 A 通过相同特征接口提供并版本化。在实验配置选用计划中的
`v1_no_danger` 之前，必须**与 A 确认**其可用性。

### B/C seam

C 仅通过配置、环境变量、官方 agent 回调和 checkpoint 路径驱动未来统一的
`team_agent`。C 不访问或重新定义 `Transition`、Q-table、网络、replay buffer、
学习更新或其他学习实现细节。

C 需要 B 提供的最小信息是：

- 算法标识符；
- 明确的训练或评估模式；
- checkpoint 输入和输出路径；
- checkpoint 版本元数据；以及
- 训练指标的输出位置。

`team_agent` 实际消费的配置键、模式映射、checkpoint schema 与文件名、版本字段及
训练指标 schema 均为**待与 B 确认**项。在确认前，实验代码不得导入
`agent_code/q_learning_agent/` 或 `agent_code/dqn_agent/` 的私有模块。

### 运行时配置

- `BOMBERMAN_CONFIG` 指向一个绝对路径的完整实验配置。未设置时，agent 仅使用包内
  默认配置。配置必须自包含；第一版不支持隐式继承或多层合并。
- `BOMBERMAN_RUN_DIR` 指向绝对路径的运行输出目录。训练输出从该目录解析。
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
  sandbox/
  replays/                 # 按回放策略可选生成
```

`metadata.json` 记录展开后的配置、代码/源码身份、依赖、硬件、种子、模式、时间戳和
终态状态。JSONL 文件是追加式原始记录。`official_stats.json` 是用于交叉检查的官方
框架导出。`checkpoints/` 与 `sandbox/` 只属于对应运行。运行目录不得覆盖已有运行。

`episodes.jsonl` 当前使用 `episode-v1` schema，每行对应一个已完整结束的 round：

- round 字段：`run_id`、`round_index`、环境 `seed`、`scenario`、`stage` 和
  `environment_seed`、`round_steps`；
- `agents` 中每名 agent 的字段：`name`、`score`、`coins`、`kills`、`suicides`、
  `crates`、`bombs`、`invalid`、`survived`、`dead`、`death_step` 和
  `death_causes`、`killed_by_self`、`killed_by_opponent`。`death_causes` 会记录命中
  该 agent 的危险爆炸来源，可区分 `self_bomb`、`opponent_bomb` 及同一步多重命中。

逐局的 `score`、`coins`、`kills`、`suicides`、`crates`、`bombs` 和 `invalid` 可按
agent 累加，并与 `official_stats.json` 的 `by_agent` 交叉检查。运行器和逐局记录已在
C2/C3 实现；训练、checkpoint 和打包将在后续阶段实现。

`timing.jsonl` 当前使用 `timing-v1` schema，每行对应一次 agent 决策机会，记录
`run_id`、`round_index`、`step`、`agent_name`、最终执行的 `action`、agent 请求的
`requested_action`、`think_time`、是否 `timed_out`、是否因上一轮超时被 `skipped`，
以及前后的可用思考时间。

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
以及可选的未见 Q 状态比例。输入缺少 `episodes.jsonl`、空文件、非法 JSON 或不符合
schema 时会明确失败。

## 运行命令

所有命令均在项目根目录执行。框架只有两个彼此独立的模式：`train` 负责训练和训练报告，
`evaluate` 负责检查已有 checkpoint、冻结模型、测试和评估报告。运行目录禁止覆盖，重复
实验必须使用新的 `--run-id`。

实验运行显示固定 80 列的 `tqdm` 进度条，在同一行原地更新，避免 IDE 误判终端宽度后
因进度条过长而换行滚屏。训练结果同时持续写入 `training.csv`、`metadata.json` 和日志。

### 回放保存策略

运行器通过 `--replay-policy auto|none|failures|sampled|all` 控制回放。默认的 `auto`
会根据运行类型选择：训练使用 `sampled`，固定多 seed 评估使用 `all`，单 seed 快速测试
使用 `none`。

- `none`：不保存回放；
- `failures`：目标 Agent 死亡、自杀或产生无效动作时保存；
- `sampled`：保存第一局以及每隔 `--replay-interval` 局；
- `all`：保存每一局。

默认抽样间隔为 500 局，可用 `--replay-interval` 修改。回放写入每个运行目录的
`replays/round_XXXXX.pt`，`replays/manifest.jsonl` 同时记录 round、seed、目标 Agent
得分以及保存原因。训练不会因此保存全部回放；默认 5 seeds × 20 局的正式评估会保留
全部 100 局，便于复核高分和失败案例。

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
`base.json` 的 `evaluation.seeds`。`summary.csv` 中的 `run_id=AVERAGE` 是跨 seed
平均值所在的数据行，不是另一个运行目录。

训练 checkpoint 固定放在该训练目录的 `checkpoints/` 中。Q-learning 默认命名为
`final.pkl`，名称中包含 `dqn` 的 Agent 默认命名为 `final.pt`。评估不会复制或改写
checkpoint，只在 `metadata.json` 和 `fixed_evaluation.json` 中记录其绝对路径。

`run-id` 只能是一个目录名称，不能包含 `/`，也不能是 `.` 或 `..`。已有同名目录时运行器
会拒绝覆盖，因此重新运行时需要使用新名称，例如在末尾增加 `_v2`。

### 四个 Task

| Task | 场景 | 默认参与者 |
|---|---|---|
| `1` | `coin-heaven` | 目标 Agent，无对手 |
| `2` | `classic` | 目标 Agent，无对手 |
| `3` | `classic` | 目标 Agent、`peaceful_agent`、`coin_collector_agent` |
| `4` | `classic` | 目标 Agent、`rule_based_agent`；可用 `--opponents` 覆盖 |

Task 1 和 Task 2 不接受对手；Task 3 的两个官方对手固定；Task 4 至少需要一个对手，且可
混合官方 Agent、其他自有 Agent 或同一 Agent 的变体。

### 单独训练

例如在 Task 1 训练 Q-learning 10000 局并禁止放炸弹：

```bash
Q_LEARNING_ALLOW_BOMB=false python3 experiments/run.py \
  --config experiments/configs/base.json \
  --mode train \
  --task 1 \
  --agent q_learning_agent \
  --n-rounds 10000 \
  --seed 11 \
  --run-id q_coin_train
```

训练输出：

```text
runs/q_coin_train/
  metadata.json
  episodes.jsonl
  timing.jsonl
  training.csv
  training_summary.json
  training_progress.png
  official_stats.json
  checkpoints/final.pkl
```

`training.csv` 每个训练回合写一行，统一包含 `round`、`reward`、`action_steps`、
`epsilon`、`q_states`、`loss`、`updates` 和 `checkpoint`。Q-learning 使用
`q_states`，DQN 使用 `loss` 与 `updates`；不适用的列留空。`training_summary.json` 保存
最终步数、平均 reward、最近 100 局平均 reward 和最佳回合，`training_progress.png` 展示
reward 与 epsilon 的变化。

训练模式可在配置中启用基于 reward 移动平均的早停：

```json
"training": {
  "n_rounds": 10000,
  "early_stopping": {
    "enabled": true,
    "window": 100,
    "patience": 100,
    "min_rounds": 300,
    "min_delta": 0.1,
    "target_reward": 49.0
  }
}
```

`patience` 表示移动平均连续多少轮未提高至少 `min_delta` 后停止；只有达到
`min_rounds` 和可选的 `target_reward` 才会触发。省略或设为 `null` 的
`target_reward` 会允许低奖励平台触发早停。实际完成轮数和停止原因写入
`metadata.json` 的 `termination` 字段。该配置仅对 `train` 模式生效。

其他任务只需更换 `--task`。例如 Task 3：

```bash
python3 experiments/run.py \
  --config experiments/configs/base.json \
  --mode train \
  --task 3 \
  --agent q_learning_agent \
  --n-rounds 10000 \
  --seed 11 \
  --run-id q_task3_train
```

### 单独测试和评估

测试开始前必须用 `--checkpoint` 指定已有模型。路径不存在时立即失败，不创建测试结果目录，
也不会使用随机模型代替。下面评估已有 Q-table 在 Task 1 的表现：

```bash
Q_LEARNING_ALLOW_BOMB=false python3 experiments/run.py \
  --config experiments/configs/base.json \
  --mode evaluate \
  --task 1 \
  --agent q_learning_agent \
  --checkpoint agent_code/q_learning_agent/q-table.pkl \
  --run-id q_task1_eval
```

该命令按配置中的全部固定 seeds 运行，并生成：

```text
runs/q_task1_eval/q_task1_eval_s10001/
...
runs/q_task1_eval/q_task1_eval_summary/
  summary.csv
  mean_score.png
  fixed_evaluation.json
```

评估刚才训练产生的 checkpoint：

```bash
Q_LEARNING_ALLOW_BOMB=false python3 experiments/run.py \
  --config experiments/configs/base.json \
  --mode evaluate \
  --task 1 \
  --agent q_learning_agent \
  --checkpoint runs/q_coin_train/checkpoints/final.pkl \
  --run-id q_coin_eval
```

默认 seeds 和每个 seed 的测试局数来自 `base.json` 的 `evaluation`。也可用
`--seeds 10001 10002` 和 `--n-rounds 5` 临时覆盖。

### 单 Seed 快速测试

开发期间可先运行一局，检查 checkpoint 能否加载和 Agent 是否报错：

```bash
Q_LEARNING_ALLOW_BOMB=false python3 experiments/run.py \
  --config experiments/configs/base.json \
  --mode evaluate \
  --task 1 \
  --agent q_learning_agent \
  --checkpoint agent_code/q_learning_agent/q-table.pkl \
  --seed 10001 \
  --n-rounds 1 \
  --run-id q_smoke
```

单 seed 模式直接写入 `runs/q_smoke/`，并在 `runs/q_smoke/summary/` 生成该次测试的
`summary.csv` 和图。完整模型比较仍应使用上一节的固定多 seed 评估。

### Task 4 自定义对手

```bash
python3 experiments/run.py \
  --config experiments/configs/base.json \
  --mode evaluate \
  --task 4 \
  --agent dqn_agent \
  --opponents rule_based_agent q_learning_agent peaceful_agent \
  --checkpoint runs/dqn_task4_train/checkpoints/final.pt \
  --run-id dqn_task4_eval
```

### DQN

DQN 使用相同入口，只需更换 Agent 名称。checkpoint 扩展名为 `.pt`：

```bash
python3 experiments/run.py \
  --config experiments/configs/base.json \
  --mode train \
  --task 1 \
  --agent dqn_agent \
  --n-rounds 10000 \
  --seed 11 \
  --run-id dqn_coin
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
| `--opponents` | 覆盖 Task 4 的默认对手列表 |
| `--run-id` | 本次实验的唯一输出名称 |
| `--replay-policy` | 回放策略；默认 `auto` |
| `--replay-interval` | `sampled` 策略的周期，默认 500 局 |

`Q_LEARNING_ALLOW_BOMB=false` 会同时影响训练和评估。只有模型训练时确实禁用了炸弹，或
正在执行无炸弹消融实验时才应设置；否则应删掉该环境变量。

## 基础配置

`configs/base.json` 是完整 JSON 文档，不是继承片段。`algorithm`、`seed` 和
`checkpoint` 为 `null`，表示具体运行可通过命令行显式选择值。训练过程使用
`training-v1` CSV schema，固定评估的逐局记录使用 `episode-v1`，决策耗时使用
`timing-v1`。
