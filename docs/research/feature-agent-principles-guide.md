# Bomberman Feature 与 Agent 原理导读

> 目的：帮助团队成员真正理解当前代码在做什么，而不仅是知道怎样运行命令。
>
> 本文以 2026-09-12 工作区中的实现为准。课程要求以 `notes/final_project.pdf` 和
> `PROJECT_REQUIREMENTS.md` 为准；本文只解释团队设计，不替代课程文件。

## 1. 先建立一张完整地图

Bomberman Agent 每一步完成的是下面这条链路：

```text
游戏提供 game_state
        ↓
Feature extractor 把状态转成数字
        ↓
Q-table 或神经网络计算六个动作的 Q 值
        ↓
屏蔽物理非法动作
        ↓
训练时用 epsilon 探索；评估时选择最大 Q 值动作
        ↓
环境执行动作并返回 reward 和下一状态
        ↓
学习算法调整 Q-table 或神经网络参数
```

六个动作的固定顺序为：

```python
("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
```

这里有三个必须分开的概念：

1. **Feature** 描述“当前局面是什么样”。
2. **Model** 估计“每个动作长期有多好”。
3. **Reward** 定义“训练过程中什么结果值得鼓励”。

最终比较的其实是完整组合：

```text
Feature + Model + Reward + Training configuration
```

## 2. 原始 `game_state` 是什么

环境没有把游戏截图交给 Agent，而是提供一个结构化 Python 字典，主要包括：

| 字段 | 含义 |
|---|---|
| `field[x,y]` | 石墙 `-1`、空地 `0`、箱子 `1` |
| `bombs` | 每颗炸弹的位置和倒计时 |
| `explosion_map` | 当前爆炸还会持续多久 |
| `coins` | 当前可见金币坐标 |
| `self` | 自身分数、能否放弹和坐标 |
| `others` | 存活对手信息 |
| `step` | 当前回合步数 |

模型不能直接理解 Python 字典，因此 Feature System 必须把它变成固定结构的数字。

## 3. 什么叫 Feature

Feature 是从状态中提取出来、能够帮助模型区分局面的数字。

例如下面这些都是 Feature：

```text
右边能不能走：1
右边两步后是否危险：1
向右走后到金币的距离是否缩短：0.0035
现在放弹能炸几个箱子：2
```

它们不是动作规则。Feature 可以告诉模型“向右后两步会危险”，但不能直接写成：

```python
if right_is_safe:
    return "RIGHT"
```

项目允许人工设计 Feature，但最终动作必须由训练出来的模型决定。

## 4. 公共 Feature System 在代码中的角色

当前入口是：

```python
from agent_code.team_agent.feature_system import extract_features

features = extract_features(game_state, "continuous-v1")
```

公共计算集中在 `feature_system/common.py`。一次提取会计算：

- 六个动作的物理合法性；
- 正常情况下未来各时刻的危险图；
- 假设现在放弹后的危险图；
- 每个第一动作之后的时间展开逃生可能性；
- 到金币、箱子 frontier 和对手的 BFS 距离；
- 当前放弹能覆盖多少箱子和对手。

不同 Feature 方案使用相同客观计算，但把它们编码成不同形状。

## 5. 危险预测的数学含义

危险不是一个单独的真假值，而是一个带时间维度的布尔张量：

$$
D \in \{0,1\}^{(H+1)\times W\times H_b}
$$

这里最后一个 $H_b$ 表示棋盘高度，前面的 $H$ 是预测时间范围。当前代码的时间范围为：

```python
HORIZON = BOMB_TIMER + EXPLOSION_TIMER + 1
```

在当前设置下为 7。`D[t,x,y]=1` 表示位置 `(x,y)` 在未来第 `t` 步危险。

例如：

```text
时间 t：      1  2  3  4  5  6  7
某格是否危险：0  0  1  1  0  0  0
```

说明这个格子未来第三、四步危险。旧 Feature 可能只把它概括成“以后危险”，新 Feature
可以区分 `t+1`、`t+2`、`t+3`。

当前实现遵循项目框架的实际规则：石墙阻断爆炸，箱子不阻断爆炸，不额外模拟连锁提前引爆。

## 6. 时间展开逃生搜索

只知道“下一格安全”是不够的。进入一条死路时，第一步可能安全，但几步后仍会被炸死。

因此代码搜索的是“位置—时间”状态：

$$
(x_t,y_t,t)
$$

从一个固定的第一动作开始，在每个未来时刻枚举可以安全到达的位置：

$$
R_{t+1}=\{p'\mid p\in R_t,\ p'\text{ 可从 }p\text{ 到达且 }D[t+1,p']=0\}
$$

由此得到：

- `safe_next`：第一步是否安全；
- `safe_horizon`：最远能够安全存活到第几步；
- `survives_horizon`：是否能活过整个预测范围；
- `reachable_area`：最后安全时刻有多少个可能位置。

`reachable_area` 很重要，因为“只有一条勉强可逃的路线”和“有十个安全落点”虽然都能逃，
安全余量却不同。

## 7. BFS 距离为什么比直线距离更有用

Manhattan distance 为：

$$
d_M((x_1,y_1),(x_2,y_2))=|x_1-x_2|+|y_1-y_2|
$$

它不考虑墙。BFS 距离是在可通行格构成的图上计算最短路，因此能够考虑绕路和不可达情况。

新 Feature 使用距离变化：

$$
\Delta d=d_{before}-d_{after}
$$

- `Δd > 0`：动作后更接近目标；
- `Δd = 0`：没有变化；
- `Δd < 0`：动作后远离目标。

箱子不能直接进入，因此代码使用 **crate frontier**：与箱子相邻且可以站立的格子。
模型学习的是怎样接近合适的放弹位置，而不是怎样走进箱子。

## 8. 五套 Feature 方案

### 8.1 `discrete-v1`：旧的离散基线

状态由 14 个类别字段组成，理论最大状态数为：

$$
3^7\times2^4\times5\times4\times2=1,399,680
$$

同一状态还可以转成 40 维 one-hot 向量。例如一个三分类字段取类别 1 时：

```text
类别值：1
one-hot：[0, 1, 0]
```

优点是简单、可解释、与已有 checkpoint 兼容。缺点是将很多不同局面合并在一起，
例如“两步后危险”和“六步后危险”都可能成为“以后危险”。

### 8.2 `discrete-compact-v1`：对称压缩离散状态

该方案有 12 个类别字段、38 维 one-hot，理论最大状态数为 622,080。它增加精确的
`danger_t1/t2/t3`，并把金币、箱子 frontier、对手统一表示成一个当前目标。

它还使用棋盘旋转和镜像对称。直觉上，下面四种局面本质相同：

```text
金币在上 / 金币在右 / 金币在下 / 金币在左
```

如果把等价局面变换到一个规范方向，它们就能共享 Q 值。代码在允许的 D4 对称变换中选取
字典序最小的规范状态，并保存动作的双向映射：

```text
世界坐标动作 → 规范坐标动作 → 查询/更新 Q-table
规范坐标最优动作 → 世界坐标动作
```

它不是简单删除信息，而是让等价经验共享。主要适合 Double Q-learning。

### 8.3 `continuous-v1`：70 维连续向量

六个动作各占 10 维，共 60 维：

```text
legal
danger_t1
danger_t2
danger_t3
earliest_danger
safe_horizon
safe_area
coin_distance_delta
crate_frontier_distance_delta
opponent_distance_delta
```

再加 10 个全局量：目标是否可达、能否放弹、放弹后安全区域、可炸箱数、可威胁对手数、
存活对手数和回合进度。

为什么要归一化？不同量的自然尺度差异很大。例如合法性是 0/1，距离可能是几十，棋盘面积
可能是几百。把它们缩放到相近范围，可以让梯度下降更稳定。距离变化采用：

$$
\operatorname{clip}\left(\frac{d_{before}-d_{after}}{WH-1},-1,1\right)
$$

这个方案适合 MLP，因为输入已经是一行固定长度的数字：

```text
shape = (70,)
```

### 8.4 `board-v1`：12 个棋盘通道

多通道只是多个相同大小的矩阵叠在一起：

$$
X\in\mathbb{R}^{12\times W\times H_b}
$$

12 个通道分别表示墙、箱子、金币、自己、对手、炸弹、炸弹时间、当前爆炸以及未来危险。
在默认棋盘上输入形状是：

```text
(12, 17, 17)
```

它类似 RGB 图片的三个颜色通道，只是这里每一“层”表达一种游戏事实。CNN 的卷积核会在
所有输入通道的局部区域上做加权求和：

$$
y_{k,x,y}=b_k+\sum_c\sum_{i,j}W_{k,c,i,j}X_{c,x+i,y+j}
$$

训练逐渐调整权重，使某些卷积核对墙角、走廊、炸弹直线、对手附近的出口等局部空间模式
产生响应。

这个方案保留空间布局，但没有直接提供 BFS 目标距离和完整逃生摘要，因此需要更多数据让
CNN 自己学习。

### 8.5 `hybrid-v1`：棋盘与手工向量结合

该方案同时输出：

```text
board  = (12, W, H)
vector = (70,)
```

通常使用两个网络分支：

```text
12通道棋盘 → CNN  → 空间表示 ─┐
                               ├→ 拼接 → Q值
70维向量   → MLP  → 摘要表示 ─┘
```

CNN 负责发现空间形状，70 维向量直接提供危险时序、逃生搜索和目标距离。这是一种偏置与
学习能力的折中：不要求网络从零发现所有游戏规律，也不把整个棋盘压缩成少量人工摘要。

## 9. Q 值到底是什么

无论 Q-table 还是 DQN，目标都是估计动作价值函数：

$$
Q(s,a)=\mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s,a_t=a\right]
$$

它不是“下一步奖励”，而是当前做动作 `a` 后，未来折扣奖励总和的期望。`γ` 越接近 1，
模型越重视长期结果。

在推理时，先屏蔽物理非法动作，再选：

$$
a^*=\arg\max_{a\in A_{legal}}Q(s,a)
$$

危险但物理合法的动作没有被强制屏蔽。模型必须通过 Feature 和 Reward 学会避免它。

## 10. Agent 1：当前 Q-learning

当前已实现。它使用 `discrete-v1` 的 14 元 tuple 作为字典 key：

```python
q_table[state_key] -> 6个动作值
```

一步更新公式为：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
$$

- `α` 是学习率；
- `γ` 是折扣率；
- 括号中的量是 TD error。

优点是更新直观、训练快。缺点是没有见过的状态没有经验，而且状态数量增加时每个状态得到
的样本会迅速减少。

## 11. Agent 2：Compact Double Q-learning

计划方案，使用 `discrete-compact-v1`。普通 Q-learning 用同一组估计选择和评价最大动作，
容易产生过高估计。Double Q-learning 维护两张表 `A` 和 `B`。

更新 A 时：

$$
a^*=\arg\max_a Q_A(s',a)
$$

$$
Q_A(s,a)\leftarrow Q_A(s,a)+\alpha[r+\gamma Q_B(s',a^*)-Q_A(s,a)]
$$

更新 B 时交换角色。推理时可使用：

$$
Q(s,a)=Q_A(s,a)+Q_B(s,a)
$$

这个 Agent 同时研究两个变化：对称压缩状态，以及 Double Q 的过高估计修正。

## 12. Agent 3：当前 MLP-DQN

当前已实现。它把 `discrete-v1` 转成 40 维 one-hot，再输入：

```text
40 → 64 → 64 → 6
```

一个全连接层计算：

$$
h=\operatorname{ReLU}(Wx+b)
$$

与 Q-table 不同，神经网络参数可以在相似输入之间共享。当前 DQN 使用经验回放：把
`(s,a,r,s',done)` 保存到 buffer，随机抽取 batch，降低连续样本之间的相关性。

它还使用 target network 计算目标：

$$
y=r+\gamma\max_{a'}Q_{target}(s',a')
$$

policy network 拟合这个目标，并周期性复制到 target network。当前实现属于标准 DQN，
不是 Double DQN。

## 13. Agent 4：Continuous Double DQN

计划方案，使用 70 维 `continuous-v1` 和较宽 MLP，例如：

```text
70 → 128 → 128 → 6
```

Double DQN 把“选择动作”和“评价动作”分开：

$$
a^*=\arg\max_a Q_{policy}(s',a)
$$

$$
y=r+\gamma Q_{target}(s',a^*)
$$

这与普通 DQN 的关键区别是 target network 不再自己对所有动作取最大值，从而减少最大化
噪声造成的过高估计。

该方案是当前最稳妥的主力：它保留精确的动作后果，又比 CNN 更容易在有限训练预算内收敛。

## 14. Agent 5：Board CNN Double DQN

计划方案，使用 `board-v1`。CNN 不把棋盘立即展平成 3,468 个无空间关系的数字，而是通过
卷积在棋盘各处复用相同权重。这带来两个性质：

1. 参数共享：同一个“墙角模式”出现在不同位置时可被同一卷积核识别。
2. 局部性：相邻墙、炸弹和角色之间的关系容易首先被捕捉。

CNN 最后仍输出六个 Q 值，学习目标仍是 Double DQN。CNN 不是新的强化学习算法，只是
`Q(s,a)` 的函数逼近器改变了。

风险是样本需求更大，而且 replay buffer 中的棋盘张量比 70 维向量占用更多内存。

## 15. Agent 6：Hybrid Dueling Double DQN

计划方案，使用 `hybrid-v1`。CNN 和 MLP 的 embedding 拼接后进入 Dueling head。

Dueling 网络把 Q 值分解为：

$$
Q(s,a)=V(s)+A(s,a)-\frac{1}{|A|}\sum_{a'}A(s,a')
$$

- `V(s)`：当前状态总体有多好；
- `A(s,a)`：某动作相对其他动作有多好。

减去平均 Advantage 是为了解决分解不唯一：如果不约束，同时给 `V` 加一个常数并从所有
`A` 中减去同一常数，Q 值完全不变。

Dueling 可能在许多动作价值相近的状态中更有效，例如开阔安全区域；但它增加复杂度，不保证
一定优于普通 Double DQN，必须通过固定评估验证。

## 16. 当前 Reward 在数学上做什么

当前实际奖励版本是 `r1`：

| 结果 | 奖励 |
|---|---:|
| 每一步 | `-0.01` |
| 收集金币 | `+1.0` |
| 击杀对手 | `+5.0` |
| 炸毁箱子 | `+0.2` |
| 死亡 | `-10.0` |
| 非法动作 | `-0.1` |

`r1_no_crate` 当前把收集金币提高到 `+3.0`，同时把炸箱奖励改成 0。它用于研究“金币优先且
不直接奖励炸箱”的行为；因为同时改变了两个数值，所以已不是相对于 `r1` 的严格单因素比较。

另外实现了两套现行 Reward。`r1` 与 `r1_no_crate` 只保留用于解释和加载历史实验；新训练
默认使用 `r2_balanced`，需要密集路径反馈时使用 `r3_potential`。

### `r2_balanced`：重新平衡事件结果

| 结果 | 奖励 |
|---|---:|
| 每一步 | `-0.01` |
| 收集金币 | `+3.0` |
| 发现金币 | `+0.25` |
| 击杀对手 | `+5.0` |
| 炸毁箱子 | `+0.1` |
| 自杀 | `-7.0` |
| 被对手击杀 | `-5.0` |
| 存活到回合结束 | `+0.25` |
| 非法动作 | `-0.2` |

与 `r1` 相比，它把收集金币提高到 `+3.0`，降低无差别炸箱收益，把自杀和战斗中被杀
分开，并减弱 `-10` 可能造成的过度保守。发现隐藏金币得到小额奖励，因为这比普通炸箱
更接近正式目标；实际收集金币仍有明显更高价值。

### `r3_potential`：状态势能塑形

`r3_potential` 使用 `r2_balanced` 的事件部分，并加入：

$$
F(s,s')=\gamma\Phi(s')-\Phi(s)
$$

当前势能为三个有界部分之和：可达金币接近度；没有可达金币时的箱子 frontier 接近度；当前位置安全度。接近度采用指数衰减：

$$
C(d)=e^{-d/4}
$$

因此离目标越近，势能越高，但很远目标不会产生过大的数值。完整形式可概括为：

$$
\Phi(s)=0.5C(d_{coin})+0.25C(d_{crate})+0.25S(s)
$$

金币项与箱子项按目标可用性二选一；`S(s)` 来自当前位置最早危险时间。终局下一状态使用 `Phi=0`。

这种奖励比较前后状态，不直接奖励 `UP`、`RIGHT` 等动作名称。沿环路返回原状态时，势能增益会抵消，因而比“向金币走一步固定加分”更不容易被往返刷取。它仍然需要通过正式 score、自杀率和胜率验证，不能只看更高的训练 reward。

运行时使用相应配置：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json \
  --mode train --task 1 --agent double_dqn_continuous_agent \
  --seed 11 --n-rounds 1000 --run-id continuous_r2_t1

python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json \
  --mode train --task 1 --agent double_dqn_continuous_agent \
  --seed 11 --n-rounds 1000 --run-id continuous_r3_t1
```

Reward ID 是 checkpoint 契约的一部分，因此不能把 `r1` checkpoint 直接作为 `r2` 或
`r3` 的同任务续训起点。四个当前 Reward 的完整数值表以
`agent_code/team_agent/README.md` 和 `agent_code/team_agent/rewards.py` 为准。

Reward 决定 TD target，因此会改变模型学习的价值函数。训练 reward 高不等于正式得分高：
如果辅助奖励可以被反复刷取，模型可能优化错误目标。因此 reward 必须通过冻结模型后的正式
游戏指标验证。

## 17. Epsilon-greedy 为什么存在

如果训练一开始总选择当前最大 Q 值，随机初始化造成的偶然偏好可能永远不会被纠正。
epsilon-greedy 使用：

$$
a=\begin{cases}
\text{随机合法动作},&\text{概率 }\epsilon\\
\arg\max Q(s,a),&\text{概率 }1-\epsilon
\end{cases}
$$

训练初期 epsilon 大，用于探索；后期逐渐下降。正式评估时不探索，模型参数冻结。

## 18. 六个 Agent 的实验关系

| Agent | Feature | Model | 当前状态 | 核心问题 |
|---|---|---|---|---|
| 1 | `discrete-v1` state | Q-learning | 已实现 | 基线表现如何 |
| 2 | `discrete-compact-v1` | Double Q | 待实现 | 对称压缩是否帮助表格学习 |
| 3 | `discrete-v1` 40维 | 标准 DQN | 已实现 | 相同信息下神经网络是否更好 |
| 4 | `continuous-v1` 70维 | Double DQN | 待实现 | 精确动作后果是否减少状态混淆 |
| 5 | `board-v1` 12通道 | CNN Double DQN | 待实现 | 原始空间布局是否有额外价值 |
| 6 | `hybrid-v1` | Dueling Double DQN | 待实现 | 空间表示和人工摘要能否互补 |

Agent 1 与 3 是相同 Feature/Reward 下的算法比较。Agent 4 与 5 更接近“手工摘要还是棋盘
表示”的比较。Agent 6 是完整高上限组合，但若同时改变模型、Feature 和 Reward，就不能把
性能变化归因于某一个组件。

## 19. 训练、选择评估和最终测试不是一回事

### 训练

```text
train=True，参数持续更新，epsilon 探索
```

训练 reward、loss 和 Q-table 大小主要用于诊断学习过程。

### 模型选择评估

```text
train=False，checkpoint 冻结，固定 validation seeds
```

用于筛选 Feature、Reward、网络和超参数。

### 最终测试

配置完全确定后，使用没有参与调参的 seeds 和相同对手进行多局测试。核心指标包括平均得分、
金币、击杀、自杀率、存活率、胜率以及 `act` 的 p95/max 时间。

## 20. 怎样读一次训练更新

以 DQN 为例，buffer 抽到一条经验：

```text
状态 s：当前位置危险，右边存在完整逃生路径
动作 a：RIGHT
即时奖励 r：-0.01
下一状态 s'：仍存活且接近金币
done：False
```

模型先计算当前预测 `Q(s,RIGHT)`，再根据下一状态构造 target：

```text
target = -0.01 + gamma × 下一状态最佳合法动作价值
```

如果 target 高于当前预测，梯度下降会提高这类状态下 `RIGHT` 的值；如果之后死亡带来 `-10`，
此前导致死亡的动作价值会通过多步 bootstrap 逐渐降低。

模型不是一次就理解“炸弹很危险”，而是通过大量类似误差更新形成参数。

## 21. 最容易出现的理解误区

1. **Feature 多不一定好。** 无关和冗余输入会增加样本需求；Q-table 尤其怕状态组合爆炸。
2. **CNN 不等于强化学习。** CNN 只是状态到 Q 值的函数形式，训练仍由 DQN 完成。
3. **棋盘通道并非完全没有人工设计。** 人仍决定每个通道表示什么，只是不提前总结最佳动作。
4. **合法不等于安全。** legal mask 只表达动作能否执行，危险程度交给模型学习。
5. **训练 reward 不等于比赛性能。** 必须冻结模型后用官方指标评估。
6. **更复杂不保证更强。** 在截止时间和 CPU 限制下，收敛充分的小模型可能优于未收敛的 CNN。
7. **Feature 版本是模型契约。** 即使维度没变，只要字段语义改变，旧 checkpoint 也可能失效。

## 22. 阅读代码的推荐顺序

如果希望把本文与实现对应起来，建议按这个顺序阅读：

1. `feature_system/types.py`：先看不同输出的数据形状；
2. `feature_system/common.py`：理解危险、BFS 和逃生事实；
3. `continuous_v1.py`：看客观事实怎样变成 70 个数字；
4. `board_v1.py`：看字典怎样变成 12 张棋盘；
5. `discrete_compact_v1.py` 与 `symmetry.py`：理解类别压缩和动作映射；
6. `q_learning_agent/train.py`：看一次表格 TD 更新；
7. `dqn_agent/model.py`：看 replay、target 和梯度更新；
8. `callbacks.py`：看推理时怎样提取 Feature、mask 非法动作并选动作。

读每段代码时反复问四个问题：

```text
输入 shape 是什么？
输出 shape 是什么？
这个数字的客观含义是什么？
它是在描述状态，还是已经替模型做了决策？
```

能回答这四个问题，就已经掌握了这套系统最核心的原理。
