# Bomberman Agent 方案与原理导读

> 历史快照日期：2026-09-15。本文保留当时的实验合同、学习 Agent 和官方基线，
> 不再表示当前推荐入口。Rainbow Lite 后续版本与最终状态见
> [`rainbow-lite-index.md`](rainbow-lite-index.md)。

## 1. 一套 Agent 方案包含什么

项目实际比较的是完整合同，而不是孤立算法：

```text
Feature + Model/Algorithm + Reward + Safety + Training configuration
```

六动作 Q 值表示长期折扣回报：

$$Q(s,a)=\mathbb E\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s,a_t=a\right]$$

推理先应用物理合法动作约束，再按配置应用 veto-only 生存掩码，最后在剩余动作中取学习 Q 值最大者。无 horizon-survivable action 时回退到物理合法 Q argmax，不用规则选择所谓“救援动作”。

## 2. 当前全部学习 Agent

| Agent | 算法 | Feature | 当前角色 / 状态 |
|---|---|---|---|
| `q_learning_agent` | 单表 Q-learning | `discrete-q-v2` 默认；支持 v1/objective | 表格基线，已实现 |
| `dqn_agent` | 标准 DQN | 40/50/60 维离散 one-hot | 小型神经网络基线，已实现 |
| `double_q_agent` | Double Q-learning | `discrete-objective-v1` | 预注册 Task 1/2 fallback，代码归档于 `all_other_agent_code` |
| `double_q_compact_agent` | Double Q-learning | `discrete-compact-v1` | 对称压缩研究，已实现并归档 |
| `double_dqn_continuous_agent` | Double DQN MLP | `continuous-v2` 84 维 | 通用 continuous 基线，已实现 |
| `double_dqn_continuous_v2_agent` | Double DQN MLP | `continuous-v2` 84 维 | 当前 Task 2 winner，已带 `final.pt` |
| `double_dqn_continuous_v3_agent` | Double DQN MLP | `continuous-v3` 107 维 | 安全显式特征消融，已实现 |
| `cnn_double_dqn_agent` | CNN Double DQN | `board-v1` | 空间表示研究，已实现 |
| `hybrid_dueling_double_dqn_agent` | Dueling Double DQN | `hybrid-v1` | 融合研究，已实现并归档 |
| `double_q_lambda_agent` | 线性 Double Q(λ) + tile coding | `continuous-v2` | 当前竞争候选，已实现 |
| `expected_sarsa_lambda_agent` | 线性 Expected SARSA(λ) + tile coding | `continuous-v2` | 当前竞争候选，已实现 |
| `rainbow_lite_agent` | Dueling Double DQN + PER + 4-step | `continuous-v4` | Rainbow Lite 共享基础实现；完整历史版本见独立索引 |

`legal_random_agent` 是项目自建的物理合法随机基线，现归档于 `all_other_agent_code`。官方 `random_agent`、`rule_based_agent`、`peaceful_agent`、`coin_collector_agent` 是不可训练对照或课程对手，不属于候选学习方案。

## 3. 各算法为什么存在

### Q-learning 与 Double Q

单表 Q-learning 更新：

$$Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]$$

它易解释、训练开销小，但未见状态没有泛化，离散状态增长会造成样本稀疏。Double Q 用两套估计器分离动作选择和评价，降低最大化偏差；Compact 版本还利用棋盘对称共享经验。

### DQN 与 Double DQN

标准 DQN 用经验回放、Huber loss 和 target network。`dqn_agent` 网络为 `input -> 64 -> 64 -> 6`。Double DQN 的下一动作由 policy network 选择、由 target network 评价：

$$a^*=\arg\max_a Q_{policy}(s',a),\qquad y=r+\gamma Q_{target}(s',a^*)$$

continuous Agent 的典型网络为 `84/107 -> 128 -> 128 -> 6`。CNN 改变状态编码器，不改变强化学习目标。

### Dueling、PER 与 n-step

Dueling head 分解为：

$$Q(s,a)=V(s)+A(s,a)-\frac1{|A|}\sum_{a'}A(s,a')$$

Rainbow Lite 在 Dueling Double DQN 上加入 proportional prioritized replay，让高 TD-error transition 更常被抽到，并用固定 4-step return 更快传播延迟奖励。它是“lite”组合，不等同于包含所有 Rainbow 组件的完整实现。

### 资格迹线性方案

Double Q(λ) 使用两个独立 tile-coded 线性估计器及各自 eligibility trace，兼顾连续特征的泛化、Double Q 的偏差控制和延迟信用传播。Expected SARSA(λ) 的 bootstrap 是当前 epsilon-greedy 行为策略下的期望，并遵守存储的生存掩码。两者无需大型 replay，提供与深度网络不同的样本效率/稳定性对照。

## 4. 2026-09-15 时的主线方案

当前 Task 1/2 推荐配置形成三类竞争者：

| 方案 | Feature | Reward | Safety | 关键训练设置 |
|---|---|---|---|---|
| Double Q(λ) | `continuous-v2` | `r7_safe_credit_potential` | `survival-mask-v1/all`, H=7 | 1-step，tile coding + traces |
| Expected SARSA(λ) | `continuous-v2` | `r7_safe_credit_potential` | 同上 | 1-step，epsilon-greedy expectation |
| Rainbow Lite | `continuous-v3` | `r8_safe_constrained` | 同上 | PER、Dueling、4-step |
| Rainbow Lite fallback | `continuous-v3` | `r7_safe_credit_sparse` | 同上 | 与主模型相同算法，替换 Reward |

已确认的 `double_dqn_continuous_v2_agent` Task 2 winner 使用 `continuous-v2 + r7_safe_credit_sparse + survival-mask-v1/all`，捆绑 seed 22 的 `final.pt`；选择合同与结果在 `experiments/task2_winner.json`。

## 5. 探索、安全和 bootstrap 必须一致

epsilon-greedy 在训练早期以概率 $\epsilon$ 从允许动作中随机探索，否则选择学习 Q argmax。当前安全合同要求同一个生存集合用于：

- 随机探索；
- 训练时贪心选择；
- 冻结推理；
- Double DQN 或期望方法的 bootstrap target。

这避免模型用训练时不可选、评估时又可能出现的动作估值。Safety 是 veto boundary，不对保留动作排序；这与项目“最终决策必须通过训练学习”的要求保持一致。

## 6. 训练链与评估

正式课程链按 `Task 1 -> 2 -> 3 -> 4` 串行，并在 training seeds `11/22/33` 上独立重复。Task 1 禁用 `BOMB`；后续阶段恢复。跨阶段必须用同 seed 的完整 `--resume-from`，恢复模型、优化器、探索、RNG 以及算法所需的 replay/trace 状态。

当前推荐 Agent 的 Task 1 不按固定局数假装收敛，而是使用冻结得分停止：第 200 局开始，每 50 个新训练回合在 seeds `9000–9019` 做一次无探索评估；连续三次 `mean_score >= 48` 才达到 Task 1 score convergence，最多 1000 回合。之后还必须通过独立 `10000–10019` 阶段门槛和工程检查，才能获得 curriculum promotion qualification。

Task 2 每个训练 seed 至少 500 回合且达到 150,000 个阶段动作，最多 2,000 回合。唯一 Task 2 winner 必须先让三个 training-seed checkpoint 独立通过开发门槛，再按预注册排名选一个 checkpoint，且只允许一次 100-seed 主验证。

最终选模依据冻结评估的 `mean_score`、金币、箱子、击杀、自杀率、第一率、生存、无效动作率和完整 `act` 时延，而不是训练 reward 或单局回放。正式推理必须在 CPU 单线程下满足 0.5 秒动作上限；项目工程门槛更严格，要求 P95 `<50 ms`、max `<500 ms`。

## 7. Checkpoint 与打包

表格/线性 Agent 通常使用 `final.pkl`，神经 Agent 使用 `final.pt`。新 checkpoint 和 `training-resume-v7` 必须记录算法、Feature schema、Reward spec、Safety、动作顺序、训练 Task、随机状态及模型特有状态。旧 v1–v6 resume 与旧 final checkpoint 默认只允许冻结评估，不能静默当作当前完整续训来源。

最终提交只包含唯一最佳 Agent 目录及其训练参数；归档 Agent 保留在完整公开代码库用于课程要求的模型比较，但不进入最终 ZIP。

## 8. 推荐阅读顺序

1. `experiments/agent_contracts.py`：Agent 与 Feature 的合同；
2. 各 Agent `README.md` 和 `callbacks.py`：网络、默认值、加载逻辑；
3. `agent_code/learning_common/`：共享 runtime、n-step、replay、线性/tile coding；
4. `agent_code/team_agent/safety.py`：veto-only 生存掩码；
5. `experiments/configs/*task1.json`、`*task2.json`：当前主线完整配置；
6. `experiments/CURRENT_TRAINING_EVALUATION_PARAMETERS.md`：统一实验协议；
7. `docs/adr/0001-0003`：安全、Task 1 收敛、Task 2 单次验证决策。
