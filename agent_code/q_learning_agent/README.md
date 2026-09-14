# Q-learning Agent

表格型 Q-learning 基线，使用共享特征、奖励、合法动作掩码和线性探索协议，不依赖 PyTorch。

- 默认 Feature：`discrete-q-v2`（16 个离散字段）；实验配置也可显式选择冻结的 `discrete-v1`（14 个字段）。
- 默认 Reward：`r1`；正式 coin3 课程使用 `r1_coin3`。
- Checkpoint：`final.pkl`。
- 正式探索：epsilon 从 1.0 在 80,000 个动作步内线性下降至 0.05，跨 Task 不重置。
- Task 1 由 Runner 禁止 `BOMB`；Task 2–4 恢复完整动作空间。

## 正式 Task 1

```bash
python experiments/run.py \
  --config experiments/configs/formal_training_coin3.json \
  --mode train --device cpu --task 1 --agent q_learning_agent \
  --n-rounds 500 --seed 11 \
  --run-id formal_q_discrete_v1_r1_coin3_s11_t1_r500
```

后续阶段必须使用 `--resume-from runs/<direct-parent-run>` 创建新 run。当前精确恢复协议是 `training-resume-v6`，并保存阶段/总动作数、完整 Safety 合同、动作历史与 n-step 状态；v1–v5 和旧 final checkpoint 只能冻结评估。

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
