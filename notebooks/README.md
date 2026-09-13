# Task 1 结果 Notebook

在仓库根目录或 `notebooks/` 目录启动 JupyterLab 均可：

```bash
jupyter lab
```

打开 `task1_results.ipynb` 并选择 `mle` kernel。Notebook 默认读取六条已完成的 Task 1
训练 run，并调用 `experiments.analyze_training.analyze_training_runs()` 将完整 CSV、JSON 与
PNG 写入 `results/task1_analysis/`。它不会修改原始 `runs/`。

也可以先在命令行重建完整结果：

```bash
python experiments/analyze_training.py \
  --runs runs/formal_q_v1_r1_s11_t1_r500 runs/formal_q_v1_r1_s22_t1_r500 runs/formal_q_v1_r1_s33_t1_r500 \
         runs/formal_dqn_v1_r1_s11_t1_r500 runs/formal_dqn_v1_r1_s22_t1_r500 runs/formal_dqn_v1_r1_s33_t1_r500 \
  --output results/task1_analysis --rolling-window 25
```

训练曲线用于诊断探索和学习过程；是否晋级 Task 2 必须以独立的冻结 stage-gate evaluation
为准，不能由训练曲线代替。
