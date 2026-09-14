# DQN Agent

小型 DQN 基线，包含 `64 → 64 → 6` MLP、replay buffer、target network、Huber loss 和 epsilon-greedy。

- 默认 Feature：`discrete-q-v2` 的 50 维 one-hot；Task 3 前矩阵可选 40 维 `discrete-v1` 或 60 维 `discrete-objective-v1`。
- 默认 Reward：`r1`；当前 Task 1 矩阵比较 `r1_coin3` 与 `r5_coin_potential`，Task 2 比较对应 r6 安全版本。
- Checkpoint：`final.pt`，保存 policy、target、optimizer、按 Task 分区 replay、冻结父网络及 Python/Torch/CUDA RNG。
- 正式训练选择 CPU；`--device cuda` 保留用于工程验证。冻结评估强制 CPU。
- Task 1 由 Runner 禁止 `BOMB`。
- Task 2 当前分区至少 2000 条后，以 32 条 Task 1 + 32 条 Task 2 更新，并在父样本物理合法动作分布上做 λ=1、T=1 蒸馏。

## 当前 Task 1 初筛

```bash
python experiments/run.py \
  --config experiments/configs/pre_task3_task1.json \
  --mode train --device cpu --task 1 --agent dqn_agent \
  --feature-id discrete-objective-v1 --reward-id r5_coin_potential \
  --n-rounds 500 --target-stage-action-steps 100000 --min-rounds 1 --seed 11 \
  --run-id iter_r1_dqn_discrete_objective_v1_r5_coin_potential_s11_t1_a100000_COMMIT
```

DQN 根据实际 Feature schema 建立输入层。未配置 Feature 且没有 checkpoint 时默认 `discrete-q-v2`；冻结加载时从 checkpoint 推断并严格校验。当前精确恢复协议为 `training-resume-v8`；v7 及更早版本默认仅支持冻结评估。Task 3 的117维阶段模型通过显式入口迁移84维Task 2 policy/target，并用独立教师数据蒸馏。

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
