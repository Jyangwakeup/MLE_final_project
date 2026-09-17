# Optimized Double Q(lambda) Agent

本 agent 是独立的表格/线性 Q-learning 家族实验：共享 `continuous-v2` 的 84 维客观特征，
以每动作 tile coding 泛化，并用两个估计器降低最大化偏差。训练时启用 Watkins trace cut：
非贪心探索动作不会把旧 eligibility trace 的回报向前传播。

Task 1 使用 `r7_safe_credit_potential`、禁用 BOMB、CPU 训练。Task 2--4 保留六动作、危险、
箱子和对手特征，但必须先经 Task 1 冻结评估晋级。

训练：

```bash
python -m experiments.run --config experiments/configs/optimized_double_q_lambda_task1.json \
  --mode train --device cpu --task 1 --agent optimized_double_q_lambda_agent \
  --seed 11 --run-id qlambda_opt_s11_t1_screen
```

实验配置和结果统一记录在 `EXPERIMENT_LOG.md`。
