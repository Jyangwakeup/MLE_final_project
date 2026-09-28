# Expected SARSA(lambda)

This short name exposes the selected V4/R10 Task 2 warm-start model as
`agent_code/expected_sarsa`.

- Default checkpoint: `final.pkl` (Task 2)
- Source run: `expected_sarsa_lambda_v4_r10_s11_task2_warm`
- Parent run: `expected_sarsa_lambda_v4_r10_s11_task1`
- Feature contract: `continuous-v4` (126 inputs)
- Reward contract: `r10_bounded_history_anti_loop`
- Safety contract: `survival-mask-v1`

The retained historical checkpoints are:

| Task | Checkpoint | Source run |
|---|---|---|
| Task 1 | `checkpoints/task1/final.pkl` | `expected_sarsa_lambda_v4_r10_s11_task1` |
| Task 2 | `checkpoints/task2/final.pkl` | `expected_sarsa_lambda_v4_r10_s11_task2_warm` |

Task 2 was warm-started directly from the retained Task 1 checkpoint. The root
`final.pkl` is byte-identical to `checkpoints/task2/final.pkl` and is the
default inference weight.

The Task 2 checkpoint is a completed, lineage-correct historical candidate,
but it has no matching frozen promotion evaluation or promotion manifest.
Therefore it is not claimed to be eligible for Task 3. The project's later
V4/R11 no-safety Expected SARSA experiment recorded 100% suicide across 20
Task 2 seeds and made the Expected SARSA research line ineligible for Task 3;
that result must not be misattributed to this V4/R10 checkpoint.

The directory is independently runnable. Its supporting feature, reward,
safety and tile-coding utilities are bundled under `_vendor/`, so this folder
can be copied by itself into another Bomberman framework's `agent_code/`
directory. Install the packages in `requirements.txt` first.
