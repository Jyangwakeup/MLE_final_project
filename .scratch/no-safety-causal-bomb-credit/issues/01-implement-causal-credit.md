# Implement causal bomb credit

Type: task
Status: resolved

Add a versioned reward, learner update, tests, and experiment configuration for
the no-safety Expected SARSA(lambda) Task 2 run.

## Comments

- Existing `r10` artifacts must not be modified or resumed.

## Answer

Implemented `r11_causal_bomb_credit`, direct retrospective Expected
SARSA(lambda) credit for the causal bomb-placement state, delayed crate credit,
and a no-safety experiment configuration. The seed-11 run reached 150,031
stage actions after 1,747 rounds. Its frozen 20-seed Task 2 evaluation produced
0.25 coins, 7.95 crates, 100% suicide, and 76.8% bomb survival. Task 1 retained
49.8/50. The experiment is therefore a documented failure and is not eligible
for promotion.
