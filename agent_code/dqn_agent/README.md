# DQN Agent

小型 DQN 基线，包含 `64 → 64 → 6` MLP、replay buffer、target network、Huber loss 和 epsilon-greedy。

- 默认 Feature：`discrete-q-v2` 的 50 维 one-hot；实验配置也可选择冻结 `discrete-v1` 的 40 维输入。
- 默认 Reward：`r1`；正式 coin3 课程使用 `r1_coin3`。
- Checkpoint：`final.pt`，保存 policy、target、optimizer、完整 replay 及 Python/Torch/CUDA RNG。
- 正式训练选择 CPU；`--device cuda` 保留用于工程验证。冻结评估强制 CPU。
- Task 1 由 Runner 禁止 `BOMB`。

## 正式 Task 1

```bash
python experiments/run.py \
  --config experiments/configs/formal_training_coin3.json \
  --mode train --device cpu --task 1 --agent dqn_agent \
  --n-rounds 500 --seed 11 \
  --run-id formal_dqn_discrete_v1_r1_coin3_s11_t1_r500
```

DQN 根据实际 feature schema 建立 40 或 50 维输入层。未配置 Feature 且没有 checkpoint 时默认 `discrete-q-v2`；冻结加载时从 checkpoint 推断并严格校验。当前精确恢复协议为 `training-resume-v4`；v1–v3 和旧 final checkpoint 仅支持冻结评估。

## 冻结评估

```bash
python experiments/run.py \
  --config experiments/configs/formal_training_coin3.json \
  --mode evaluate --task 1 --agent dqn_agent \
  --checkpoint runs/formal_dqn_discrete_v1_r1_coin3_s11_t1_r500/checkpoints/final.pt \
  --seed 10001 --n-rounds 1 --run-id dqn_coin3_eval_s10001
```

正式 CPU 环境与 GPU smoke 证据、六链预算和晋级门槛见根目录 `IMPLEMENTATION_GUIDE.md`。

## 测试

```bash
conda run --no-capture-output -n mle \
  python -m unittest tests.test_dqn_agent tests.test_devices tests.test_resume
```
