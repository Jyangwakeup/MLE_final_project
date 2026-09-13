# Q-learning Agent

表格型 Q-learning 基线，使用共享特征、奖励、合法动作掩码和线性探索协议，不依赖 PyTorch。

- 默认 Feature：`discrete-q-v2`；Task 3 前矩阵可显式选择 `discrete-v1` 或 60 维 `discrete-objective-v1`。
- 默认 Reward：`r1`；当前 Task 1 矩阵比较 `r1_coin3` 与 `r5_coin_potential`，Task 2 比较对应的 r6 安全版本。
- Checkpoint：`final.pkl`。
- Task 1 ε 为 1.0→0.10/160k 阶段动作；Task 2 重热为 0.30→0.05/120k。总动作数跨 Task 保留，阶段动作数在晋级时重置。
- 随机探索只从 H=7 可存活动作抽样；贪心、Bellman 最大值和冻结评估仍只用物理合法掩码。
- Task 1 由 Runner 禁止 `BOMB`；Task 2–4 恢复完整动作空间。

## 当前 Task 1 初筛

```bash
python experiments/run.py \
  --config experiments/configs/pre_task3_task1.json \
  --mode train --device cpu --task 1 --agent q_learning_agent \
  --feature-id discrete-objective-v1 --reward-id r1_coin3 \
  --n-rounds 500 --target-stage-action-steps 100000 --min-rounds 1 --seed 11 \
  --run-id iter_r1_q_discrete_objective_v1_r1_coin3_s11_t1_a100000_COMMIT
```

后续阶段必须使用 `--resume-from runs/<direct-parent-run>` 创建新 run。当前精确恢复协议是 `training-resume-v5`，并保存阶段/总动作数、安全探索诊断、动作历史与 n-step 状态；v1–v4 和旧 final checkpoint 只能冻结评估。第三轮能力失败才允许改用独立的 `double_q_agent`。

## 冻结评估

```bash
python experiments/run.py \
  --config experiments/configs/formal_training_coin3.json \
  --mode evaluate --task 1 --agent q_learning_agent \
  --checkpoint runs/formal_q_discrete_v1_r1_coin3_s11_t1_r500/checkpoints/final.pkl \
  --seed 10001 --n-rounds 1 --run-id q_coin3_eval_s10001
```

直接接入官方框架时 Agent seed 默认 0；若加载 checkpoint 且未显式指定 Feature，Agent 从 checkpoint 契约选择 `discrete-v1` 或 `discrete-q-v2`。正式课程与阶段门槛应始终通过 `experiments/run.py` 执行。

## 测试

```bash
conda run --no-capture-output -n mle \
  python -m unittest tests.test_q_learning_agent tests.test_features tests.test_resume
```
