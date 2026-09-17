# Q-learning Task 1 optimization record

Current historical single-table baseline is `discrete-q-v2 + r4_anti_oscillation`: 47.44/50
mean coins and 60% all-coins rate. The old package-local checkpoint is not resumable under the
current reward contract, so all new results begin from cold starts.

The new candidate uses two tile-coded action-value estimators, per-action feature projections,
and eligibility traces. Its trace rule follows Watkins Q(lambda): an epsilon-selected action that
is not greedy clears prior traces before the current TD update. This avoids treating arbitrary
off-policy exploratory suffixes as though they followed the greedy target policy.

The design follows [Watkins and Dayan's Q-learning convergence conditions](https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html),
[Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html),
and [potential-based reward shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
The legal-action protocol follows the official [Gymnasium action-masking example](https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/): exploration, greedy inference,
and bootstrap maximization use the same admissible set.

| Experiment | Agent | Feature / reward | Budget | Development evaluation | Status |
|---|---|---|---:|---|---|
| T1-Q01 | `q_learning_agent` | `discrete-q-v2` / r4 | 100k actions | 100 games, seeds 10000--10004 | submitted |
| T1-Q02 | `optimized_double_q_lambda_agent` | `continuous-v2` / r7 potential | 100k actions | 100 games, same seeds | submitted |

Each job snapshots after each 25k action boundary. The selection JSON ranks frozen candidates by
all-coins rate, mean coins, coins per 100 steps, completion steps, and loop rate. No result is
considered a Task 1 pass until a selected checkpoint also reaches 90% all-coins rate on seeds
11000--11099.
