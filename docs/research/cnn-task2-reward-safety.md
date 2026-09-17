# CNN Task 2：奖励适配与 Safety Mask 验证

本文记录 `cnn_distilled_double_dqn_agent` 从 Task 1 迁移到 Task 2 时的奖励与
动作屏蔽决策。动态实验数字以
[`EXPERIMENT_LOG.md`](../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)
为唯一权威来源；本文只保存机制分析、实验顺序和晋级标准。

## 当前判断

`r7_safe_credit_sparse` 并非不能训练 CNN。D01 的训练末段已经出现平均 32.49 次
BOMB 和 82.25 个炸箱，而冻结贪心评估退化到 18.45 次 BOMB、34.35 个炸箱。这说明
环境确实提供了放弹和炸箱信用，主要故障在冻结策略的 Q 值排序（尤其是 WAIT 吸引子），
而不是 CNN 完全收不到 BOMB 信号。

不过 sparse 奖励有一个与 17 通道空间输入相关的塑形真空：状态出现可见金币时，
`_state_potential` 会进入金币分支；sparse 版本没有金币势能，于是金币和箱子前沿的
目标势能都为零，只剩危险势能。模型此时需要仅凭稀疏奖励重新学会“炸出金币后去捡”，
这比显式规则 agent 更难。

`r7_safe_credit_potential` 是下一项最小改动：金币势能对应输入的金币 BFS 距离通道，
箱子势能对应箱子前沿 BFS 距离通道。实现使用
`gamma * Phi(next_state) - Phi(state)`、`gamma=0.95`，并令终止状态势能为零，属于
potential-based reward shaping；在满足假设时不会改变最优策略集合。依据见
[Ng、Harada 与 Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。

## 受控实验顺序

1. D02 保持 `r7_safe_credit_sparse + avoidable WAIT -0.04 + n=4 + safety all + KL=2`，
   不在运行中改变配置。
2. 若 D02 未通过三训练 seed 和主验证的严格门槛，D03 只将奖励改为
   `r7_safe_credit_potential`。
3. 若 D02、D03 都未通过，选择两者中冻结结果更好的一种奖励，D04 只把
   `n_step` 从 4 改为 5。
4. 只有 potential+n5 仍少放弹时，才比较 `useful_bomb_per_crate: 0.2 -> 0.4`；
   炸弹数量本身不是晋级指标。
5. 上述实验结束前固定 KL=2，不同时搜索 KL 退火。

在当前环境的事件时序中，step 1 放置的炸弹通常到 step 5 才产生
`CRATE_DESTROYED`，因此 n=5 可把真实炸箱奖励直接放进该 BOMB transition 的回报；
n=4 通常只能依赖 bootstrap 间接传播。多步回报可能加速信用传播，但步数越长也会
增加 off-policy 偏差，所以不直接扩到 n=8/10。背景见
[Rainbow](https://arxiv.org/abs/1710.02298) 和
[Retrace](https://papers.nips.cc/paper/2016/file/c3992e9a68c5ae12bd18488bc579b30d-Paper.pdf)。

## Safety mask 的作用与风险

主路线使用 `survival-mask-v1/all`。行为探索、贪心选择、冻结推理、replay 中的
next mask 和 Double DQN target 都使用同一 admissible action set。这避免了行为策略
和 TD target 对“哪些动作存在”理解不同。动作消除和 shield 的一般背景见
[Action-Elimination DQN](https://proceedings.neurips.cc/paper_files/paper/2018/file/645098b086d2f9e1e0e939c27f9f2d6f-Paper.pdf)
与 [Safe RL via Shielding](https://arxiv.org/abs/1708.08611)。

mask 仍可能产生负面影响：有限 horizon 或危险模型错误会把可恢复的好动作判成危险；
过于保守的 mask 也会让网络缺少被屏蔽动作的 TD 数据。因而不能只看自杀率判断它正确。
但 D01 的自杀率为 0、368/368 个已结算炸弹均存活，且队友在相同 mask 下能达到更高
放弹量，当前没有证据把少放弹归因于 safety mask。

正式诊断在 seeds 10000--10019 对同一 checkpoint 只读记录：

- physical/safe BOMB 可用次数与 BOMB veto rate；
- physical-mask raw Q argmax 被 safety mask 拦截的比例及动作类型；
- raw-BOMB argmax veto rate、fallback、自杀、炸弹存活和炸箱效率。

诊断只记录反事实，不改变实际动作。若 `BOMB veto rate` 和
`raw-BOMB argmax veto rate` 都不超过 5%，保留 `all`，不训练 safety 消融。只有超过
阈值，才对同一 checkpoint、同一 seeds 做 `all` 与 `off` 配对冻结评估；仅当 off 使
平均炸箱至少提高 10% 且自杀率仍不超过 5%，才继续检查 mask false negatives。
不采用 exploration-only，因为它会让探索、贪心和 target 使用不同动作集合。

## 严格验收门槛

每个候选先通过安全和课程保留门槛，再比较能力：

- Task 1 mean score >= 48，父 checkpoint 保留率 >= 90%；
- Task 2 suicide <= 5%，已结算炸弹存活率 >= 95%，零放弹局率 <= 10%；
- mean crates >= D01 的 34.35；invalid <= 1%；
- CPU act p95 < 50 ms、max < 500 ms。

通过门槛后，依次比较 mean coins、all-coins rate、mean crates、
crates per survived bomb、WAIT/长 WAIT。bomb count 只用于定位策略，不作为优化目标。
同一阶段先用 seeds 11/22/33 检查训练稳定性，再在 seeds 11000--11099 做主验证；
没有通过前一阶段时，后续作业会自动跳过或终止，避免无依据扩大搜索。
