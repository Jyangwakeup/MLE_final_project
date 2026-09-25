# MLE Bomberman Final Report Outline

> Status: writing blueprint, not submission-ready prose.
>
> This outline was generated from the repository evidence with the `ml-paper-writing` workflow. Every author assignment, quantitative claim, citation, hardware statement, and repository link must be checked by the team before submission. AI-generated text must be understood, verified, and rewritten by the team members in their own words.

## A. Requirements-to-Section Compliance Matrix

| Course requirement | Report location | Planned evidence/content |
|---|---|---|
| Introduction: problem, relevance, challenge | Section 1 | Sparse rewards, dynamic hazards, opponent interaction, and the 0.5 s CPU constraint |
| Background: task and considered RL methods | Section 2 | Game rules, MDP formulation, Q-learning, Double Q, DQN, Double DQN, CNNs, reward shaping, curriculum learning, and shielding |
| Project planning and teamwork | Section 3 | Three-person A/B/C plan, shared interfaces, cross-review, and milestones |
| Deep-learning hardware | Sections 3.3 and 5.3 | CPU training for central MLP runs; selected CNN runs on a GTX 1080 Ti; A100 smoke evidence only; exact metadata still to audit |
| Methods and justified design choices | Section 4 | State representations, models, rewards, curriculum, safety layer, and evaluation protocol |
| Training process and acceleration | Section 5 | Checkpoints, resume, transfer, replay, distillation, experiment parallelism, and single-thread inference |
| Systematic experiments | Section 6 | RQ1–RQ7, including successful, failed, incomplete, and negative results |
| At least two ML models | Sections 4.2 and 6.1–6.3 | Q-learning, Double Q(λ), DQN, Double DQN, and CNN models |
| Cover all developed models | Table 1, Section 4.2, model appendix | Model-family overview plus a directory-level appendix |
| Explain final model selection | Section 6.7 | B33/Die Hardest origin, local performance, packaging verification, and limitations |
| Reproducibility | Sections 3.3, 4.6, and 5 | Seeds, configuration hashes, checkpoint hashes, frozen evaluation, paired comparisons, and CPU timing |
| Conclusion and future work | Section 7 | Answers to the research questions and unresolved safety, credit-assignment, and opponent-modeling problems |
| Author on every heading | Entire report | Replace every `[Primary author: TBD]` with a real name |
| Approximately 4,000 words per member | Section G | Repository plan has three roles; provisional total is 12,000 words |
| Public repository URL | Title page or Introduction | Candidate URL: `https://github.com/Jyangwakeup/MLE_final_project`; public visibility must be verified |
| No university logo | Template checklist | Do not include a university logo |
| Report PDF not in public repository | Submission checklist | Verify before submission |
| Team review of AI-assisted text | Writing workflow | Every member verifies and rewrites their assigned material |

Primary course source: [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md).

## B. Proposed Report Titles

1. **Learning to Survive and Compete: A Curriculum-Based Reinforcement Learning Agent for Bomberman**
2. **From Coin Collection to Competitive Play: Evidence-Driven Reinforcement Learning for Bomberman**
3. **Balancing Score, Safety, and Runtime in a Reinforcement Learning Bomberman Agent**
4. **Value Learning under Dynamic Hazards: Systematic Development of a Bomberman Agent**
5. **Die Hardest: Curriculum Learning and Survival-Constrained Action Selection in Bomberman**

Recommended title: **Balancing Score, Safety, and Runtime in a Reinforcement Learning Bomberman Agent**.

The recommended title matches the three best-supported dimensions of the project without claiming a novel reinforcement-learning algorithm.

## C. One-Sentence Thesis

> Through a staged curriculum, systematic comparisons of tabular and neural value-learning agents, and a survival-based action veto evaluated under frozen multi-seed protocols, we developed a CPU-compatible Bomberman agent while finding that greater model complexity and more specialized training did not consistently improve official game score.

## D. Core Claims

1. Structured continuous features combined with Double Q or Double DQN learned substantially stronger Task 1 navigation than several discrete and hybrid baselines, although architecture effects cannot always be isolated from feature and reward changes.
2. Task 2 performance was limited less by immediate bomb survival than by long-horizon target reselection, waiting, and cycling after crates opened new paths.
3. Survival masks substantially reduced self-destructive behavior, but increasing robustness introduced computational cost and never constituted a universal safety guarantee.
4. Corrected Task 3 evaluation showed that the selected child retained earlier-task performance and improved official score and first-place rate over its frozen parent, without demonstrating a kill-rate improvement.
5. B33/Die Hardest was selected as a practical competition candidate using local score, survival, timing, packaging, and reproducibility evidence—not because it was proven superior to every historical model under a single common protocol.

## E. Claims-to-Evidence Matrix

| Claim | Strongest evidence | Scope and qualification |
|---|---|---|
| Double Q(λ) and distilled CNN learned Task 1 | [`q-cnn-experiment-report.md`](docs/research/q-cnn-experiment-report.md), [`optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`](agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md), [`cnn-task1-task2-experiment-report.md`](docs/research/cnn-task1-task2-experiment-report.md) | Different protocols and checkpoints report slightly different means; do not merge them into one estimate |
| Reward/history changes altered waiting and cycling | [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md) | Several comparisons changed more than one factor; report associations rather than clean causality |
| Task 2 remained incomplete | [`q-cnn-experiment-report.md`](docs/research/q-cnn-experiment-report.md), [`cnn-task1-task2-experiment-report.md`](docs/research/cnn-task1-task2-experiment-report.md) | Safety and crate destruction did not imply all-coins completion |
| Safety evolved but remained incomplete | [`safety-certified-placement-v6.md`](docs/research/safety-certified-placement-v6.md), [`safety-rearming-v7.md`](docs/research/safety-rearming-v7.md), [`safety-fixed-deadline-v8.md`](docs/research/safety-fixed-deadline-v8.md), [`safety-proven-movement-v9.md`](docs/research/safety-proven-movement-v9.md) | Engineering worlds and regression corpora are not independent performance tests |
| Corrected Task 3 passed project gates | [`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md) | Main-validation score improved by 1.23; kill difference was −0.01, so do not claim improved kills |
| Several Task 4 approaches failed or showed no gain | [`task4-frozen-results.md`](docs/research/task4-frozen-results.md), [`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md), [`task4_frozen_20260919/report.md`](experiments/results/task4_frozen_20260919/report.md), [`task4_score_20260920/report.md`](experiments/results/task4_score_20260920/report.md) | Attempts used different protocols and must be summarized separately |
| Die Hardest met local practical criteria | [`die_hardest/README.md`](agent_code/die_hardest/README.md), [`SUBMISSION_MANIFEST.json`](agent_code/die_hardest/SUBMISSION_MANIFEST.json), [`verification.json`](docs/research/die-hardest-submission/verification.json) | Benchmark was local, not official tournament performance; counterexample 27155 remains unresolved |

## F. Detailed Three-Level Paper Outline

# 1. Introduction `[Primary author: TBD]` — 800 words

## 1.1 Problem and Motivation `[Primary author: TBD]` — 250 words

- **Question:** Why is learning Bomberman behavior a meaningful machine-learning problem?
- **Content:** Sparse and delayed rewards, exploration, navigation, bomb timing, adversarial occupancy, long-term credit assignment, and real-time inference.
- **Supported claim:** Strong play requires more than optimizing training reward or surviving isolated bomb placements.
- **Evidence:** [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md), [`README.md`](README.md).
- **Visual:** Refer forward to Figure 1.
- **Citations:** `[CITATION NEEDED: reinforcement learning under sparse rewards]`; `[CITATION NEEDED: sequential decision-making under safety constraints]`.
- **TODO:** Confirm whether the course template requires an abstract before the Introduction.
- **Transition:** Move from the general challenge to the concrete project objective.

### 1.1.1 Game-Specific Challenges `[Primary author: TBD]`

Introduce timed explosions, blocking, six discrete actions, the 400-step horizon, and uncertainty about future opponent actions. Do not label the complete environment formally partially observable without defining precisely what information is missing.

## 1.2 Project Objective and Research Questions `[Primary author: TBD]` — 300 words

- **Question:** What did the team attempt to determine?
- **Content:** Compare value-learning models; progress through Tasks 1–4; evaluate representation, reward, curriculum, and survival constraints.
- **Supported claim:** The project used frozen evaluations and stage gates rather than selecting a model from training reward or isolated high-scoring games.
- **Evidence:** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md), [`experiments/README.md`](experiments/README.md).
- **Visual:** Compact list of RQ1–RQ7.
- **Citations:** `[CITATION NEEDED: empirical methodology and variance in deep reinforcement learning]`.
- **TODO:** Decide whether the final report should merge RQ1 with RQ2 and RQ3 with RQ4 for space.
- **Transition:** State the thesis and contributions.

### 1.2.1 Scope and Success Criteria `[Primary author: TBD]`

Define official score, survival, retained capability, first-place measures, and CPU timing. State explicitly that local results are not official tournament results.

## 1.3 Contributions and Report Roadmap `[Primary author: TBD]` — 250 words

- **Question:** What does the report contribute?
- **Content:** A systematic comparison of tabular and neural value learners; staged Task 1–4 development; survival-veto design; evidence-backed final selection with negative results preserved.
- **Supported claim:** The project contribution is methodological and empirical rather than a novel RL algorithm.
- **Evidence:** [`Q_CNN_AGENT_INDEX.md`](agent_code/Q_CNN_AGENT_INDEX.md), [`task3-experiment-index.md`](docs/research/task3-experiment-index.md), [`die_hardest/README.md`](agent_code/die_hardest/README.md).
- **Visual:** None.
- **Citations:** None beyond the methods cited later.
- **TODO:** Insert and verify the public repository URL.
- **Transition:** Introduce the task and relevant RL concepts.

# 2. Background `[Primary author: TBD]` — 1,400 words

## 2.1 Bomberman Task and Environment `[Primary author: TBD]` — 350 words

- **Question:** What does the Agent observe, choose, and optimize?
- **Content:** `game_state`, six actions, obstacles, coins, crates, bombs, explosions, opponents, scoring, and termination.
- **Supported claim:** Physical legality and predicted future survival are distinct properties.
- **Evidence:** [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md), `settings.py`, `environment.py`, and `items.py`.
- **Visual:** An annotated game board or compact state/action diagram.
- **Citations:** Cite the course framework rather than external literature for game rules.
- **TODO:** Recheck numerical rules against the final course framework and announcements.
- **Transition:** Formalize the interaction as a sequential decision problem.

### 2.1.1 Curriculum Tasks 1–4 `[Primary author: TBD]`

Summarize coin navigation, crate destruction, weak-opponent interaction, and strong-opponent competition. Evidence: [`experiments/README.md`](experiments/README.md).

## 2.2 Reinforcement-Learning Formulation `[Primary author: TBD]` — 250 words

- **Question:** How is the game represented for value learning?
- **Content:** State/features \(s_t\), action \(a_t\), reward \(r_t\), transition, terminal flag, discounted return, and action-value function.
- **Supported claim:** The learned model ranks admissible actions; engineered features and the safety layer do not directly specify a best action.
- **Evidence:** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md), [`CONTEXT.md`](CONTEXT.md).
- **Visual:** One transition tuple.
- **Citations:** `[CITATION NEEDED: Markov decision processes]`.
- **TODO:** State where the Markov approximation is imperfect.
- **Transition:** Introduce the value-learning algorithms.

## 2.3 Value-Learning Algorithms `[Primary author: TBD]` — 450 words

- **Question:** Which algorithms were considered and why?
- **Content:** Q-learning; Double Q-learning; Watkins Double Q(λ); DQN; Double DQN; replay; target networks; n-step learning; dueling and Rainbow-lite components.
- **Supported claim:** The algorithms address overestimation, generalization, sample reuse, and delayed credit, but their benefit must be established empirically in this task.
- **Evidence:** [`experiments/README.md`](experiments/README.md), [`optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`](agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md).
- **Visual:** Model and algorithm comparison table.
- **Citations:** `[CITATION NEEDED: Q-learning]`; `[CITATION NEEDED: Double Q-learning]`; `[CITATION NEEDED: eligibility traces and Watkins Q(lambda)]`; `[CITATION NEEDED: DQN]`; `[CITATION NEEDED: Double DQN]`; `[CITATION NEEDED: dueling networks]`; `[CITATION NEEDED: Rainbow and multi-step returns]`.
- **TODO:** Include equations only for central algorithms; move peripheral variants to the model table or appendix.
- **Transition:** Explain why algorithms alone were insufficient.

### 2.3.1 Structured and Spatial Representations `[Primary author: TBD]`

Contrast discrete keys, tile-coded continuous vectors, 84-dimensional continuous representations, 117-dimensional phase representations, board CNNs, and hybrid inputs.

## 2.4 Reward, Curriculum, and Safety Concepts `[Primary author: TBD]` — 350 words

- **Question:** How were sparse feedback and dangerous actions addressed?
- **Content:** Reward shaping, curriculum learning, physical legality, horizon survivability, veto-only masking, and learned selection among admitted actions.
- **Supported claim:** The survival layer narrows the learned action set; it is not a rule-based controller that chooses the final action.
- **Evidence:** [`CONTEXT.md`](CONTEXT.md), [`safety-certified-placement-v6.md`](docs/research/safety-certified-placement-v6.md).
- **Visual:** Nested action sets: all actions → physically legal actions → survival-admitted actions → learned argmax.
- **Citations:** `[CITATION NEEDED: reward shaping]`; `[CITATION NEEDED: curriculum learning]`; `[CITATION NEEDED: safe reinforcement learning]`; `[CITATION NEEDED: action shielding or runtime safety filters]`.
- **TODO:** Avoid describing finite-horizon admission as guaranteed global safety.
- **Transition:** Move from concepts to project organization.

# 3. Project Planning `[Primary author: TBD]` — 900 words

## 3.1 Team Structure and Collaboration `[Primary author: TBD]` — 300 words

- **Question:** How did the team satisfy the shared-work requirement?
- **Content:** A handled danger/features; B handled learning/rewards/callbacks; C handled experiments/statistics/packaging; the plan also defined reciprocal review and joint selection.
- **Supported claim:** Responsibilities were modular but coupled through shared interfaces and cross-review.
- **Evidence:** Sections 9–10 of [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md).
- **Visual:** Responsibility and interface-seam diagram.
- **Citations:** None.
- **TODO:** Replace A/B/C with legal names and correct differences between planned and actual contributions.
- **Transition:** Explain the staged schedule.

### 3.1.1 Writing Ownership `[Primary author: TBD]`

Planned ownership: A—game, danger, and features; B—RL algorithms and learning; C—planning, evaluation, and synthesis. Every final subsection must name one real primary author.

## 3.2 Milestones and Decision Gates `[Primary author: TBD]` — 250 words

- **Question:** How were experiments sequenced under the deadline?
- **Content:** Task progression, stage gates, independent confirmation, one-time main validation, and packaging buffer.
- **Supported claim:** Later stages were conditional on earlier capability, safety, and engineering evidence.
- **Evidence:** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md), [`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md).
- **Visual:** Figure 1.
- **Citations:** `[CITATION NEEDED: curriculum learning]`.
- **TODO:** Replace the planned calendar with the actual calendar where the two differ.
- **Transition:** Describe reproducibility and compute planning.

## 3.3 Reproducibility, Compute, and Evidence Management `[Primary author: TBD]` — 350 words

- **Question:** How were claims made traceable?
- **Content:** Configuration hashes, checkpoint hashes, immutable run IDs, environment/opponent seeds, JSONL episodes, frozen evaluations, bootstrap intervals, and CPU timing.
- **Supported claim:** The repository distinguishes active documentation, versioned evidence, recovery archives, and disposable artifacts.
- **Evidence:** [`docs/repository-layout.md`](docs/repository-layout.md), [`experiments/README.md`](experiments/README.md).
- **Visual:** Artifact lineage from config to checkpoint to evaluation.
- **Citations:** `[CITATION NEEDED: reproducibility in deep reinforcement learning]`.
- **TODO:** Confirm actual team size; gather per-run CPU model and wall-clock time; verify GPU identity for central CNN runs; state that official Ryzen hardware was not tested during final packaging.
- **Transition:** Introduce the technical design.

# 4. Methods `[Primary author: TBD]` — 2,200 words

## 4.1 State, Actions, and Feature Evolution `[Primary author: TBD]` — 350 words

- **Question:** What information reached the learner?
- **Content:** Discrete representations; compact state; 70/78/84-dimensional continuous features; phase features; spatial CNN channels; action history.
- **Supported claim:** Representation evolved from compact navigation descriptors toward richer danger, opponent, and temporal information.
- **Evidence:** [`experiments/README.md`](experiments/README.md), [`feature-principles-guide.md`](docs/research/feature-principles-guide.md).
- **Visual:** Feature-version evolution table.
- **Citations:** `[CITATION NEEDED: feature engineering in reinforcement learning]`.
- **TODO:** Verify every dimensionality from source before finalizing the table.
- **Transition:** Show how different learners consumed these representations.

### 4.1.1 Legal Actions versus Survival Admission `[Primary author: TBD]`

Define the physical legal mask independently from the survival veto. Use the project vocabulary in [`CONTEXT.md`](CONTEXT.md).

## 4.2 Model Families `[Primary author: TBD]` — 600 words

- **Question:** Which models were developed?
- **Content:** Tabular Q-learning; Double Q; Double Q(λ); discrete DQN; continuous Double DQN; CNN Double DQN; distilled CNN; hybrid dueling; Rainbow-lite; phase models; final B33.
- **Supported claim:** The project explored both representation capacity and update-rule changes, but not every implementation produced valid comparative evidence.
- **Evidence:** [`Q_CNN_AGENT_INDEX.md`](agent_code/Q_CNN_AGENT_INDEX.md), [`experiments/README.md`](experiments/README.md), `experiments/agent_variants/MANIFEST.md`.
- **Visual:** Table 1.
- **Citations:** Use the algorithm citations introduced in Section 2.3.
- **TODO:** Generate a directory-to-status appendix so that “all developed models” is auditable.
- **Transition:** Explain shared learning machinery.

### 4.2.1 Tabular and Tile-Coded Learners `[Primary author: TBD]`

Present Q-learning, Double Q-learning, and Watkins Double Q(λ), including eligibility-trace truncation. Make clear that features do not determine the action.

### 4.2.2 Neural Value Learners `[Primary author: TBD]`

Describe MLP Double DQN, CNN, distilled CNN, hybrid dueling, and Rainbow-lite variants, including target networks and masking.

### 4.2.3 Final B33 Architecture `[Primary author: TBD]`

Describe Double DQN, 84-dimensional `continuous-v2`, `r7_safe_credit_sparse`, n-step learning, and survival-mask-v9. Evidence: [`SUBMISSION_MANIFEST.json`](agent_code/die_hardest/SUBMISSION_MANIFEST.json).

## 4.3 Learning and Exploration `[Primary author: TBD]` — 300 words

- **Question:** How were value estimates trained?
- **Content:** Replay, target updates, Huber loss, Adam, epsilon-greedy exploration, terminal masking, n-step samples, and deterministic frozen inference.
- **Supported claim:** Training and evaluation were explicitly separated; final evaluation disabled exploration and learning.
- **Evidence:** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md), [`experiments/README.md`](experiments/README.md).
- **Visual:** Algorithm box for the Double DQN update.
- **Citations:** DQN, Double DQN, experience replay, target networks, and n-step returns.
- **TODO:** Extract final B33 hyperparameters from its frozen configuration/checkpoint rather than assuming historical defaults.
- **Transition:** Explain rewards and curriculum transfer.

## 4.4 Reward Design and Curriculum Transfer `[Primary author: TBD]` — 350 words

- **Question:** How was sparse feedback converted into a training signal?
- **Content:** Official score versus shaped reward; reward versions; coin potential; crate reward; loop penalties; safety credit; Task 1–4 transfer.
- **Supported claim:** Shaped rewards improved some behavioral metrics but did not consistently improve official score.
- **Evidence:** [`reward-principles-guide.md`](docs/research/reward-principles-guide.md), [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md), [`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md).
- **Visual:** Reward-version comparison table.
- **Citations:** `[CITATION NEEDED: reward shaping]`; `[CITATION NEEDED: potential-based reward shaping]`.
- **TODO:** Mark which comparisons were true single-variable experiments.
- **Transition:** Introduce the separate survival constraint.

## 4.5 Survival-Mask Evolution `[Primary author: TBD]` — 400 words

- **Question:** How did the safety layer evolve?
- **Content:** v1 horizon survival; escape obligations; opponent transitions; controllable survival; certified bomb placement; opponent rearming; fixed deadline; proven movement preference; v9 compact computation.
- **Supported claim:** Each revision addressed a concrete counterexample, but stronger semantics and more opponent scenarios increased runtime cost.
- **Evidence:** [`task3-experiment-index.md`](docs/research/task3-experiment-index.md), [`safety-certified-placement-v6.md`](docs/research/safety-certified-placement-v6.md), [`safety-rearming-v7.md`](docs/research/safety-rearming-v7.md), [`safety-fixed-deadline-v8.md`](docs/research/safety-fixed-deadline-v8.md), [`safety-proven-movement-v9.md`](docs/research/safety-proven-movement-v9.md).
- **Visual:** Figure 2 or a safety-version timeline.
- **Citations:** `[CITATION NEEDED: safe reinforcement learning]`; `[CITATION NEEDED: action shielding]`.
- **TODO:** Keep counterexample 27155 explicit.
- **Transition:** Define how methods were evaluated.

## 4.6 Evaluation and Statistical Protocol `[Primary author: TBD]` — 200 words

- **Question:** What made a result admissible?
- **Content:** Development/confirmation/main splits; fixed seeds; parent-child pairing; 10,000-resample bootstrap; official score; first-place definitions; safety and latency gates.
- **Supported claim:** Candidate selection used frozen game metrics rather than training loss or training reward.
- **Evidence:** [`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md), [`experiments/README.md`](experiments/README.md).
- **Visual:** Evaluation-split table.
- **Citations:** `[CITATION NEEDED: paired bootstrap confidence intervals]`.
- **TODO:** Never treat incompatible seed sets as paired samples.
- **Transition:** Describe training in practice.

# 5. Training `[Primary author: TBD]` — 1,300 words

## 5.1 Stage-Wise Training Curriculum `[Primary author: TBD]` — 350 words

- **Question:** How did training progress from navigation to competition?
- **Content:** Task 1 coin navigation; Task 2 crates and hidden coins; Task 3 peaceful and coin-collector opponents; Task 4 rule-based opponents.
- **Supported claim:** Promotion required retained earlier abilities plus stage-specific performance, safety, and engineering metrics.
- **Evidence:** [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md), [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md).
- **Visual:** Figure 1.
- **Citations:** `[CITATION NEEDED: curriculum learning]`.
- **TODO:** Distinguish course-recommended stages from team-defined gate thresholds.
- **Transition:** Explain checkpoint movement between stages.

## 5.2 Checkpointing, Resume, and Transfer `[Primary author: TBD]` — 250 words

- **Question:** How was training continued reproducibly?
- **Content:** Atomic snapshots; policy-only initialization; strict resume; explicit Task 2→3 and Task 3→4 transfers; replay and RNG state.
- **Supported claim:** Resume and transfer were separate contracts, preventing incompatible checkpoints from being silently continued.
- **Evidence:** [`experiments/README.md`](experiments/README.md), [`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md).
- **Visual:** Checkpoint-lineage diagram.
- **Citations:** Reproducibility sources from Section 3.3.
- **TODO:** Recover the exact public evidence lineage of B33.
- **Transition:** Discuss compute and runtime.

## 5.3 Compute and Training Efficiency `[Primary author: TBD]` — 250 words

- **Question:** What compute was used and why?
- **Content:** CPU training for small networks; GPU CNN runs; one Torch thread for inference; vectorized and cached safety search.
- **Supported claim:** GPU did not automatically improve throughput for the small MLP setup, while final evaluation was CPU-only.
- **Evidence:** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md), [`cnn_path_double_dqn_agent/EXPERIMENT_LOG.md`](experiments/agent_variants/cnn_path_double_dqn_agent/EXPERIMENT_LOG.md).
- **Visual:** Compute summary table.
- **Citations:** None.
- **TODO:** Consolidate exact hardware, elapsed time, and peak memory for each central experiment. Do not generalize GTX 1080 Ti details to all CNN runs without checking metadata.
- **Transition:** Explain specialized training mechanisms.

## 5.4 Distillation and Replay Strategies `[Primary author: TBD]` — 200 words

- **Question:** How were earlier capabilities preserved?
- **Content:** Frozen Task 1 teacher, masked Q-value distillation, task-partitioned replay, historical replay, and policy-only specialist transfer.
- **Supported claim:** Distillation preserved navigation in some CNN transfers, but later TD fine-tuning could degrade it.
- **Evidence:** [`cnn-task1-task2-experiment-report.md`](docs/research/cnn-task1-task2-experiment-report.md), [`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md).
- **Visual:** Teacher–student schematic.
- **Citations:** `[CITATION NEEDED: knowledge distillation]`; `[CITATION NEEDED: catastrophic forgetting or continual learning]`.
- **TODO:** Do not imply that distillation guaranteed retention.
- **Transition:** Close with failures that redirected training.

## 5.5 Training Failures and Corrective Decisions `[Primary author: TBD]` — 250 words

- **Question:** Which failures materially changed the plan?
- **Content:** State-input bug, invalid evaluation starts, looping, Task 2 suicides, callback lifecycle defect, safety-search timeout, and interrupted Task 4 campaigns.
- **Supported claim:** Failed and incomplete runs were retained and excluded from positive conclusions.
- **Evidence:** [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md), [`task3-lifecycle-frozen-results.md`](docs/research/task3-lifecycle-frozen-results.md), [`task4-frozen-results.md`](docs/research/task4-frozen-results.md).
- **Visual:** Failure → diagnosis → decision table.
- **Citations:** None.
- **TODO:** Select failures that answer a research question rather than writing a chronological log.
- **Transition:** Present results by research question.

# 6. Experiments and Results `[Primary author: TBD]` — 4,500 words

## 6.1 RQ1: Which Models Learned Task 1 Navigation? `[Primary author: TBD]` — 600 words

- **Hypothesis:** Richer continuous or spatial representations and bias-reducing learners should outperform elementary discrete baselines.
- **Compared variants:** Q-learning, DQN, Double Q, continuous Double DQN, hybrid dueling, optimized Double Q(λ), CNN, and distilled CNN.
- **Controls/confounders:** Several early comparisons differed in representation, reward, and training budget; label them exploratory.
- **Data:** Early experiments primarily used seeds 10001–10005 × 20 rounds; later models used separately documented development and reserved sets.
- **Primary metrics:** Mean coins, all-coins rate, round length, waiting, and loops.
- **Results:** Early continuous DDQN reached 49.52 mean coins and 86% completion; optimized Double Q(λ) and distilled CNN later reached approximately 96% completion on independent or reserved evaluations.
- **Uncertainty:** Different reports contain different checkpoint/protocol means, including 48.95, 49.51, and 49.58. Present these as separate experiments until reconciled.
- **Interpretation:** Structured representation and algorithmic changes produced strong navigation, but no single comparison isolates architecture alone.
- **Evidence:** [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md), [`q-cnn-experiment-report.md`](docs/research/q-cnn-experiment-report.md), [`optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`](agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md).
- **Visual:** Table 2 and the Task 1 panel of Figure 3.
- **Transition:** Investigate why competent navigators still stalled.

### 6.1.1 Failed or Invalid Task 1 Evidence `[Primary author: TBD]`

Separate hybrid’s poor result, invalid R2 evaluation starts, and CNN input bugs from valid model comparisons.

## 6.2 RQ2: Did Reward and History Reduce Waiting and Cycling? `[Primary author: TBD]` — 500 words

- **Hypothesis:** Idle/loop penalties and history features should improve completion efficiency.
- **Compared variants:** Q R4 old versus idle; Double Q R3 versus R4; later conditional-loop proposals.
- **Confounders:** Some comparisons changed reward, features, training length, or source revision simultaneously.
- **Results:** Q all-coins rate rose from 31% to 60% and long-wait rate fell from 58% to 19%; Double Q long-wait fell to zero while ping-pong increased to 25%.
- **Interpretation:** The failure mode shifted; reducing WAIT did not eliminate looping or prove that one reward term caused the change.
- **Evidence:** [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md).
- **Visual:** Behavior comparison table; do not label it a clean causal ablation.
- **Transition:** Test whether navigation transferred to crates and bombs.

## 6.3 RQ3: How Well Did Q and CNN Agents Transfer to Task 2? `[Primary author: TBD]` — 650 words

- **Hypothesis:** A strong navigator plus bomb-survival features should learn crate destruction and hidden-coin collection.
- **Compared variants:** Optimized Double Q(λ), crate/history variants, team-demonstration variant, distilled CNN D01/D02.
- **Metrics:** Mean coins out of nine, crates, zero-bomb rate, survival, suicide, WAIT, and loops.
- **Results:** Q(λ) r20 reached 3.95 coins and 53.45 crates; CNN D02 main validation reached 2.60 coins and 44.15 crates, with zero self-suicide but zero all-coins completion.
- **Interpretation:** Immediate survival and bombing were learned more reliably than long-term goal reselection after map changes.
- **Limitation:** Q and CNN routes were not all compared under one identical protocol.
- **Evidence:** [`q-cnn-experiment-report.md`](docs/research/q-cnn-experiment-report.md), [`cnn-task1-task2-experiment-report.md`](docs/research/cnn-task1-task2-experiment-report.md).
- **Visual:** Table 2 and a representative failure trajectory.
- **Transition:** Motivate the survival-mask experiments.

### 6.3.1 Distillation Success and TD Fine-Tuning Regression `[Primary author: TBD]`

Report Task 1 preservation and the observed fall to 55% completion after additional TD fine-tuning. Do not generalize beyond the recorded run.

## 6.4 RQ4: What Did Survival Constraints Improve and Cost? `[Primary author: TBD]` — 650 words

- **Hypothesis:** Vetoing actions without finite-horizon survival evidence should reduce avoidable self-deaths.
- **Compared variants:** v1, v3, v4, v5, and engineering revisions v6–v9.
- **Metrics:** Suicide rate, bomb survival, guarantee loss, escape collapse, search timeout, P95/max action time, and retained score.
- **Results:** Later revisions produced zero observed self-deaths in several registered engineering corpora, while some strong-opponent searches exceeded latency or internal proof budgets.
- **Interpretation:** Stronger safety semantics corrected concrete failures but created a safety–latency trade-off.
- **Limitation:** Zero failures on finite corpora do not prove universal safety; opponent deaths and counterexample 27155 remain.
- **Evidence:** [`task3-experiment-index.md`](docs/research/task3-experiment-index.md), [`safety-certified-placement-v6.md`](docs/research/safety-certified-placement-v6.md), [`safety-fixed-deadline-v8.md`](docs/research/safety-fixed-deadline-v8.md), [`task4-frozen-results.md`](docs/research/task4-frozen-results.md).
- **Visual:** Figure 4 and the safety-version table.
- **Transition:** Evaluate whether safe Task 3 learning improved competition.

## 6.5 RQ5: Did Task 3 Improve Competition While Retaining Earlier Skills? `[Primary author: TBD]` — 700 words

- **Hypothesis:** Task 3 learning should raise score or first-place rate without eroding Task 1/2 performance.
- **Compared variants:** Frozen Task 2 parent and three Task 3 child seeds; selected seed22/c150.
- **Data:** Confirmation worlds 21000–21099; main-validation worlds 21100–21199; 10,000 paired bootstrap resamples.
- **Results:** Confirmation score gains were +1.92, +1.23, and +2.56. Main validation improved score from 5.91 to 7.14 and first-place rate from 41% to 53%.
- **Uncertainty:** Score difference CI [0.27, 2.14]. Kill difference was −0.01 with CI [−0.16, 0.14].
- **Interpretation:** The child improved overall competitive outcome without evidence of increased kills.
- **Correction history:** Explain the original escape-collapse count, its diagnosis, and the metric-only correction without hiding the original failed report.
- **Evidence:** [`task3-lifecycle-frozen-results.md`](docs/research/task3-lifecycle-frozen-results.md), [`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md).
- **Visual:** Table 3 with parent, child, paired difference, and CI.
- **Transition:** Ask whether stronger opponents yielded further improvement.

## 6.6 RQ6: Why Did Task 4 Attempts Fail or Show No Gain? `[Primary author: TBD]` — 700 words

- **Hypotheses:** Replay mixtures, score-specialist training, simplified rewards, higher gamma, historical-opponent exposure, and later score optimization could improve performance against three rule-based opponents.
- **Results:**
  - The initial shared-parent campaign stopped on a search timeout before formal training.
  - Specialist E1/E2/E3 produced no final score improvement; the selected difference was −0.52 with CI [−1.055, 0].
  - Historical-opponent exposure failed its initial promotion rule.
  - The later score campaign stopped on a worker failure after partial evidence.
- **Interpretation:** Neither simplifying reward nor adding same-lineage historical opponents reliably improved rule-based-opponent performance within the registered protocols.
- **Limitations:** These are separate bounded experiments; failure does not prove that the ideas can never work.
- **Evidence:** [`task4-frozen-results.md`](docs/research/task4-frozen-results.md), [`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md), [`task4_frozen_20260919/report.md`](experiments/results/task4_frozen_20260919/report.md), [`task4_score_20260920/report.md`](experiments/results/task4_score_20260920/report.md).
- **Visual:** Table 4.
- **Transition:** Explain why B33 was still selected as the competition package.

## 6.7 RQ7: Why Was B33 / Die Hardest Selected? `[Primary author: TBD]` — 700 words

- **Candidate:** Task4 B, seed33/c200.
- **Method:** Double DQN, 84-dimensional `continuous-v2`, `r7_safe_credit_sparse`, survival-mask-v9.
- **Benchmark:** 1,000 local worlds against three rule-based agents.
- **Results:** Mean official score 4.231; first place including ties 50.2%; sole first place 37.8%; survival 94.7%; zero observed framework timeouts.
- **Packaging evidence:** After renaming, 180 historical observations produced identical features, masks, Q values, and actions; six paired games reproduced complete behavior.
- **Engineering:** Die Hardest P95 14.83 ms and maximum 44.60 ms in packaging-equivalence tests; peak RSS approximately 319.6 MiB.
- **Selection claim:** Die Hardest was a tested, reproducible, packageable competition candidate with favorable local score and survival—not a universally best model.
- **Critical limitation:** The 27155 safety-contract counterexample remains unresolved; Docker and official tournament hardware were not verified during final packaging.
- **Evidence:** [`die_hardest/README.md`](agent_code/die_hardest/README.md), [`SUBMISSION_MANIFEST.json`](agent_code/die_hardest/SUBMISSION_MANIFEST.json), [`die-hardest-submission/REPORT.md`](docs/research/die-hardest-submission/REPORT.md), [`verification.json`](docs/research/die-hardest-submission/verification.json).
- **Visual:** Table 5.
- **TODO:** Recover and cite the complete original B33 training/selection report from the archive. The active final-package evidence proves identity and benchmark properties more strongly than it documents the full candidate-selection contest.
- **Transition:** Synthesize the findings without overstating them.

### 6.7.1 Packaging and Behavioral Equivalence `[Primary author: TBD]`

Separate renaming/package verification from the earlier 1,000-world benchmark. Do not claim that the renamed Agent reran the 1,000 worlds.

### 6.7.2 Remaining Safety and External-Validity Limits `[Primary author: TBD]`

State the local-hardware limitation, local opponents, unresolved 27155 counterexample, and lack of official tournament evidence.

# 7. Conclusion `[Primary author: TBD]` — 900 words

## 7.1 Answers to the Research Questions `[Primary author: TBD]` — 400 words

- **Question:** What was learned?
- **Content:** Strong Task 1 navigation was achievable; Task 2 target reselection remained difficult; survival masks helped but cost compute; Task 3 improved score; Task 4 refinements did not consistently improve score; B33 was selected pragmatically.
- **Evidence:** Cross-reference Tables 2–5.
- **Visual:** None.
- **Citations:** None.
- **TODO:** Keep every conclusion proportional to its evidence.
- **Transition:** State limitations.

## 7.2 Limitations `[Primary author: TBD]` — 250 words

- Nonuniform protocols across model families.
- Limited independent seeds for several exploratory experiments.
- Local rather than official tournament evaluation.
- Incomplete Task 2 objective.
- Reward-credit mismatch for delayed or posthumous kills.
- Finite safety corpora and unresolved counterexample 27155.
- No official CPU/Docker validation during final packaging acceptance.
- **Evidence:** [`die-hardest-submission/REPORT.md`](docs/research/die-hardest-submission/REPORT.md), [`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md).

## 7.3 Future Work `[Primary author: TBD]` — 250 words

- Repair and independently retest counterexample 27155.
- Improve long-term target reselection after crate destruction.
- Improve temporal credit assignment for delayed and posthumous kills.
- Compare principal model families under one controlled protocol.
- Add independent training seeds and stronger held-out opponents.
- Profile the final Agent on official-equivalent hardware.
- Explore learned opponent models only while preserving runtime bounds.
- End on the broader lesson that disciplined evaluation mattered as much as model capacity.

## G. Word-Budget Table

The repository contains a three-person work plan, so this outline provisionally assumes \(N=3\), or approximately 12,000 words. Confirm the actual registered membership.

| Section | Words | Share |
|---|---:|---:|
| 1. Introduction | 800 | 6.7% |
| 2. Background | 1,400 | 11.7% |
| 3. Project Planning | 900 | 7.5% |
| 4. Methods | 2,200 | 18.3% |
| 5. Training | 1,300 | 10.8% |
| 6. Experiments and Results | 4,500 | 37.5% |
| 7. Conclusion | 900 | 7.5% |
| **Total** | **12,000** | **100%** |

For an actual team size of \(N\), scale each allocation by:

\[
\text{new allocation} = \text{listed allocation} \times \frac{N}{3}.
\]

## H. Figure and Table Plan

| Item | Question answered | Data/source | Design | Comparability warning | Uncertainty |
|---|---|---|---|---|---|
| Figure 1: Curriculum and selection pipeline | How did a model reach final selection? | Implementation guide, Task 3 validation, final manifest | Task 1→4 flow with train/dev/confirm/main/package gates | Show planned and completed branches with different styles | No error bars |
| Figure 2: Agent inference pipeline | How is an action selected? | Die Hardest source, `CONTEXT.md`, safety reports | State→84 features→Double DQN Q values→legal mask→survival veto→action | Do not show the mask as ranking actions | No error bars |
| Figure 3: Capability progression | How did capability change by task? | Task 1/2 reports, Task 3 validation, final benchmark | Small multiples by Task | Do not draw a single continuous line across incompatible protocols | CIs where paired data exist |
| Figure 4: Score–safety–latency trade-off | What did stronger safety cost? | v6–v9 and Task 4 reports | Score or survival versus P95/max latency | Distinguish engineering and validation worlds | CIs only where raw paired results support them |
| Table 1: Model inventory | What was built? | Experiment README, manifests, model index | Family, input, learner, reward, task, status, evidence | “Implemented” is not “validated” | N/A |
| Table 2: Task 1/2 models | Which models transferred? | Q/CNN reports | Coins, completion, crates, loops, suicide, protocol | Separate exploratory and independently confirmed results | Report sample count and seeds |
| Table 3: Task 3 parent–child | Was Task 3 improvement retained? | Corrected validation report/JSON | Parent, child, difference, CI for Tasks 1–3 | Footnote the original failed count | Paired bootstrap CI |
| Table 4: Task 4 attempts | What failed and why? | Four Task 4 reports | Protocol, hypothesis, reached stage, outcome, stop reason | An unrun stage is “not run,” never zero | Include supplied CIs |
| Table 5: Die Hardest | Why was it selected? | README, verification, manifest | Score, wins, survival, timing, memory, package identity | Separate the 1,000-world benchmark from six-game equivalence testing | Give sample sizes; do not invent a CI |

## I. Model Inventory and Coverage Table

| Family or representative Agent | Representation | Evidence status | Report coverage |
|---|---|---|---|
| Tabular Q-learning | `discrete-v1` / `discrete-q-v2` | Valid exploratory Task 1 evidence; later surpassed | Sections 4.2, 6.1, 6.2 |
| Double Q-learning | Compact discrete | Valid exploratory Task 1 evidence | Sections 4.2, 6.1, 6.2 |
| Watkins Double Q(λ) | 84-D continuous with tile coding | Independent Task 1 confirmation; Task 2 below target | Sections 4.2.1, 6.1, 6.3 |
| Discrete DQN | Discrete features | Valid poor/medium Task 1 evidence plus failed evaluations | Sections 4.2.2, 6.1 |
| Continuous Double DQN | 70/78/84-D continuous | Strong early Task 1; central Task 2–4 lineage | Sections 4.2.2, 6.1–6.7 |
| Phase Double DQN | 117-D phase features | Task 3 experiments failed safety/retention gates | Sections 4.1, 5.5, 6.5 context |
| CNN Double DQN | Board/spatial channels | Early input issues; later valid Task 1 evidence | Sections 4.2.2, 6.1 |
| Distilled CNN Double DQN | 17-channel CNN plus frozen teacher | Strong Task 1; partial Task 2 transfer | Sections 5.4, 6.1, 6.3 |
| Hybrid Dueling Double DQN | Board tensor plus structured vector | Poor valid output plus process errors | Sections 4.2, 6.1.1 |
| Rainbow-lite | Continuous features plus selected Rainbow components | Implemented; valid-result coverage must be audited | Table 1 or appendix; use `[EVIDENCE GAP]` if no result is admissible |
| Safety ablation/no-safety variants | Same learner with altered admission | Diagnostic and ablation evidence | Sections 4.5, 6.4 |
| Task 4 E1/E2/E3 specialists | 84-D Double DQN | Completed bounded experiment; no final score gain | Section 6.6 |
| Historical-opponent C/S arms | 84-D Double DQN | Initial screening completed; S failed promotion | Section 6.6 |
| Task 4 score L/P/K variants | 84-D Double DQN | Partial campaign; terminal failure | Section 6.6 |
| B33 / Die Hardest | 84-D Double DQN plus survival-mask-v9 | Final frozen submission candidate | Sections 4.2.3 and 6.7 |

Before writing, generate an appendix from `agent_code/` and `experiments/agent_variants/MANIFEST.md`. This is necessary because the course requires all developed models to be described, including variants without successful results.

## J. Citation Search Checklist

Every entry must be verified from the original paper or an authoritative publisher record.

- `[CITATION NEEDED: Markov decision processes]`
- `[CITATION NEEDED: Q-learning]`
- `[CITATION NEEDED: Double Q-learning and maximization bias]`
- `[CITATION NEEDED: Watkins Q(lambda) and eligibility traces]`
- `[CITATION NEEDED: Deep Q-Networks]`
- `[CITATION NEEDED: Double DQN]`
- `[CITATION NEEDED: experience replay]`
- `[CITATION NEEDED: target networks]`
- `[CITATION NEEDED: multi-step temporal-difference learning]`
- `[CITATION NEEDED: dueling network architectures]`
- `[CITATION NEEDED: Rainbow DQN]`
- `[CITATION NEEDED: reward shaping]`
- `[CITATION NEEDED: potential-based reward shaping]`
- `[CITATION NEEDED: curriculum learning]`
- `[CITATION NEEDED: knowledge distillation]`
- `[CITATION NEEDED: catastrophic forgetting in sequential RL]`
- `[CITATION NEEDED: safe reinforcement learning]`
- `[CITATION NEEDED: action shielding or runtime enforcement]`
- `[CITATION NEEDED: reproducibility and variance in deep RL]`
- `[CITATION NEEDED: paired bootstrap confidence intervals]`

Mandatory citation workflow:

1. Search for the exact paper.
2. Verify it through at least two authoritative sources.
3. Check that the cited claim actually appears in the paper.
4. Fetch BibTeX programmatically.
5. If verification fails, retain `[CITATION NEEDED]` rather than inventing a reference.

## K. Missing Information and TODOs

1. Confirm the actual number and legal names of team members.
2. Replace all A/B/C and `[Primary author: TBD]` labels.
3. Confirm that `https://github.com/Jyangwakeup/MLE_final_project` is publicly accessible.
4. Confirm that the final report PDF will not be committed to the public repository.
5. Recover the complete original B33 training and candidate-selection report from the archive.
6. Extract final B33 network hyperparameters from its frozen configuration/checkpoint.
7. Produce a complete model-directory inventory, including abandoned variants.
8. Resolve whether Double Q(λ) figures 48.95, 49.51, and 49.58 refer to different formal runs/checkpoints.
9. Consolidate exact training hardware and wall-clock time for central experiments.
10. Confirm GPU identities for all CNN runs; only selected logs explicitly identify a GTX 1080 Ti.
11. Distinguish CPU training, GPU training, GPU smoke tests, and CPU inference.
12. Add official tournament results if available; otherwise state that they were unavailable.
13. Confirm current template and submission instructions against course announcements and MaMPF.
14. Determine whether appendices count toward the course word budget.
15. Add only verified bibliography entries.
16. Regenerate publication figures directly from raw machine-readable results.
17. Confirm whether Docker validation exists elsewhere; the final packaging report says it was not performed during that acceptance run.
18. Have every member verify and rewrite their assigned sections in their own words.

## L. Overclaiming and Reproducibility Risks

- Do not call the survival mask a universal or formal safety guarantee.
- Do not claim that counterexample 27155 was fixed.
- Do not equate zero observed self-deaths with proof of safety.
- Do not equate training reward with official score.
- Do not merge evaluations with different worlds, checkpoints, or protocols.
- Do not call exploratory comparisons causal when reward, features, and budget changed together.
- Do not report unstarted Task 4 phases as zero-valued results.
- Do not present regression worlds as independent test data.
- Do not hide the original Task 3 failed report after the metric correction.
- Do not claim that Task 3 increased kills; the main-validation kill difference was −0.01.
- Do not claim that renamed Die Hardest reran the 1,000-world benchmark.
- Do not claim final superiority over every repository model under a common protocol.
- Do not present the six packaging-equivalence games as a performance estimate.
- Do not infer official-hardware compatibility solely from local CPU timing.
- Do not describe same-lineage historical opponents as independent unseen opponents.
- Do not describe incomplete or interrupted training as convergence.
- Do not treat the Git remote as proof that the repository is publicly visible.

## M. Final Self-Check

| Check | Status |
|---|---|
| Exactly seven required top-level report sections | Covered |
| Introduction includes problem, value, and challenge | Covered |
| Background includes the task and considered RL methods | Covered |
| Project Planning includes teamwork, schedule, and compute | Covered; real names and actual hardware still require confirmation |
| Methods explains representations, models, rewards, safety, and metrics | Covered |
| Training explains curriculum, resume, transfer, replay, and distillation | Covered |
| Experiments and Results has the largest allocation | Covered: 37.5% |
| Conclusion includes findings, limitations, and future work | Covered |
| Every section/subsection has an author field | Covered with placeholders |
| At least two ML models are compared | Covered |
| All developed model families are represented | Covered at family level; directory-level appendix still required |
| Final model selection is explained | Covered, with B33 lineage evidence gap flagged |
| Failures and negative results are included | Covered |
| Public repository URL is included | Candidate URL identified; public visibility requires verification |
| Approximately 4,000 words per member | Planned for three members; actual membership requires confirmation |
| University logo is excluded | Must be enforced in the final template |
| Report PDF is excluded from the public repository | Must be checked at submission |
| Citations are verified rather than invented | No BibTeX has been generated; citation slots only |
| Safety and tournament claims remain calibrated | Covered |
| AI-generated material is treated as a draft requiring team review | Required before submission |

## Recommended Writing Order

Write and verify sections in this order:

1. Section 6: Experiments and Results
2. Section 4: Methods
3. Section 5: Training
4. Section 2: Background
5. Section 3: Project Planning
6. Section 1: Introduction
7. Section 7: Conclusion
8. Abstract, if required by the final template

This order grounds the narrative in admissible evidence before writing the framing sections.

---

# MLE Bomberman 期末报告框架（中文版）

> 状态：写作蓝图，并非可直接提交的报告正文。
>
> 本框架依据仓库证据并遵循 `ml-paper-writing` 工作流整理。提交前，团队必须核验作者分工、定量结论、引用、硬件信息和仓库链接。AI 生成内容必须由团队成员理解、验证，并改写为自己的表达。

## A. 课程要求与章节映射

| 课程要求 | 报告位置 | 计划覆盖内容 |
|---|---|---|
| Introduction：问题、价值和挑战 | 第 1 章 | 稀疏奖励、动态危险、对手交互和 0.5 秒 CPU 时限 |
| Background：任务与考虑过的 RL 方法 | 第 2 章 | 游戏规则、MDP、Q-learning、Double Q、DQN、Double DQN、CNN、奖励塑形、课程学习和安全屏蔽 |
| Project Planning 与团队协作 | 第 3 章 | 三人 A/B/C 分工、共享接口、交叉复核和里程碑 |
| 深度学习硬件 | 第 3.3、5.3 节 | 核心 MLP 实验以 CPU 为主；部分 CNN 使用 GTX 1080 Ti；A100 仅有 smoke 证据；精确元数据仍需核验 |
| 方法及设计理由 | 第 4 章 | 状态表示、模型、奖励、课程、安全层和评估协议 |
| 训练过程及加速 | 第 5 章 | checkpoint、resume、transfer、replay、蒸馏、实验并行和单线程推理 |
| 系统实验 | 第 6 章 | RQ1–RQ7，包含成功、失败、不完整和负面结果 |
| 至少两个机器学习模型 | 第 4.2、6.1–6.3 节 | Q-learning、Double Q(λ)、DQN、Double DQN 和 CNN |
| 覆盖全部开发模型 | 表 1、第 4.2 节、模型附录 | 模型族概览，加目录级完整清单 |
| 说明最终模型选择依据 | 第 6.7 节 | B33/Die Hardest 的来源、本地表现、打包验证和局限 |
| 可复现性 | 第 3.3、4.6、5 章 | seeds、配置哈希、checkpoint 哈希、冻结评估、配对比较和 CPU 计时 |
| 结论与未来工作 | 第 7 章 | 回答研究问题，并说明尚未解决的安全、信用分配和对手建模问题 |
| 每个标题标注主要作者 | 全文 | 将所有 `[主要作者：待定]` 替换为真实姓名 |
| 每位成员约 4,000 words | G 节 | 仓库计划包含三个角色，暂按约 12,000 words 分配 |
| 公开代码仓库 URL | 标题页或 Introduction | 候选 URL：`https://github.com/Jyangwakeup/MLE_final_project`；必须确认公开可访问 |
| 不使用大学 Logo | 模板检查表 | 明确禁止 |
| 报告 PDF 不进入公开仓库 | 提交检查表 | 提交前检查 |
| 团队复核 AI 辅助文本 | 写作流程 | 每位成员核验并改写本人负责的内容 |

课程要求的主要来源：[`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)。

## B. 报告标题候选

1. **Learning to Survive and Compete: A Curriculum-Based Reinforcement Learning Agent for Bomberman**
2. **From Coin Collection to Competitive Play: Evidence-Driven Reinforcement Learning for Bomberman**
3. **Balancing Score, Safety, and Runtime in a Reinforcement Learning Bomberman Agent**
4. **Value Learning under Dynamic Hazards: Systematic Development of a Bomberman Agent**
5. **Die Hardest: Curriculum Learning and Survival-Constrained Action Selection in Bomberman**

推荐标题：**Balancing Score, Safety, and Runtime in a Reinforcement Learning Bomberman Agent**。

该标题覆盖项目证据最充分的三个维度：正式得分、安全行为和运行时开销，同时不宣称提出了新的强化学习算法。

## C. 一句话中心论点

> 本项目通过分阶段课程学习、表格型与神经网络价值学习 Agent 的系统比较，以及在冻结多种子协议下评估的生存动作否决机制，开发出能够在 CPU 环境运行的 Bomberman Agent；实验同时表明，更高的模型复杂度和更专门化的训练并不会稳定提高正式游戏得分。

## D. 核心主张

1. 结构化连续特征结合 Double Q 或 Double DQN，在 Task 1 导航中明显优于若干离散与混合基线；但由于部分实验同时改变特征、奖励和预算，不能总把收益单独归因于模型架构。
2. Task 2 的主要瓶颈与其说是即时炸弹生存，不如说是箱子打开新路径后的长期目标重选、等待和循环行为。
3. Survival mask 明显减少了自毁行为，但更强的鲁棒性增加了计算成本，而且从未构成全局、普适的安全保证。
4. 修正后的 Task 3 评估显示，选中子模型在保留旧任务能力的同时，提高了相对冻结父模型的正式得分和第一名比例，但没有证明击杀率提高。
5. B33/Die Hardest 是依据本地得分、生存、计时、打包和复现证据选出的实用参赛候选，而不是在统一协议下被证明优于所有历史模型。

## E. 主张—证据矩阵

| 主张 | 最强证据 | 适用范围与限定 |
|---|---|---|
| Double Q(λ) 和蒸馏 CNN 学会了 Task 1 | [`q-cnn-experiment-report.md`](docs/research/q-cnn-experiment-report.md)、[`optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`](agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md)、[`cnn-task1-task2-experiment-report.md`](docs/research/cnn-task1-task2-experiment-report.md) | 不同协议和 checkpoint 的均值略有差异，不能合并成一个估计 |
| 奖励与历史信息改变了等待和循环行为 | [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md) | 多个比较同时改变多个变量，只能报告关联，不能写成严格因果 |
| Task 2 未完整解决 | [`q-cnn-experiment-report.md`](docs/research/q-cnn-experiment-report.md)、[`cnn-task1-task2-experiment-report.md`](docs/research/cnn-task1-task2-experiment-report.md) | 安全炸箱和炸箱数提高不等于完成全部金币目标 |
| 安全机制持续演进但仍不完整 | [`safety-certified-placement-v6.md`](docs/research/safety-certified-placement-v6.md)、[`safety-rearming-v7.md`](docs/research/safety-rearming-v7.md)、[`safety-fixed-deadline-v8.md`](docs/research/safety-fixed-deadline-v8.md)、[`safety-proven-movement-v9.md`](docs/research/safety-proven-movement-v9.md) | 工程世界和回归语料不是独立性能测试 |
| 修正后的 Task 3 通过项目门槛 | [`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md) | 主验证得分提高 1.23，但击杀差为 −0.01，不能宣称击杀能力提高 |
| 多种 Task 4 方法失败或没有观察到收益 | [`task4-frozen-results.md`](docs/research/task4-frozen-results.md)、[`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md)、[`task4_frozen_20260919/report.md`](experiments/results/task4_frozen_20260919/report.md)、[`task4_score_20260920/report.md`](experiments/results/task4_score_20260920/report.md) | 各实验协议不同，必须分别解释 |
| Die Hardest 满足本地实用性要求 | [`die_hardest/README.md`](agent_code/die_hardest/README.md)、[`SUBMISSION_MANIFEST.json`](agent_code/die_hardest/SUBMISSION_MANIFEST.json)、[`verification.json`](docs/research/die-hardest-submission/verification.json) | 这是本地 benchmark，不是官方锦标赛成绩；27155 仍未解决 |

## F. 详细三级论文框架

# 1. Introduction `[主要作者：待定]` — 800 words

## 1.1 问题与研究动机 `[主要作者：待定]` — 250 words

- **核心问题：** 为什么学习 Bomberman 行为是有意义的机器学习问题？
- **主要内容：** 稀疏与延迟奖励、探索、导航、炸弹时序、对手占位、长期信用分配和实时推理。
- **可支持主张：** 良好表现不能仅依靠提高训练 reward 或在孤立场景中躲过自己的炸弹。
- **证据：** [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)、[`README.md`](README.md)。
- **图表：** 前引 Figure 1。
- **引用：** `[待补引用：稀疏奖励强化学习]`；`[待补引用：安全约束下的序列决策]`。
- **TODO：** 确认课程模板是否要求在 Introduction 前提供 Abstract。
- **衔接：** 从一般挑战过渡到项目的具体目标。

### 1.1.1 游戏特有挑战 `[主要作者：待定]`

介绍定时爆炸、阻挡、六种离散动作、400 步上限和未来对手动作的不确定性。除非严格定义未观测信息，否则不要直接把完整环境称为 POMDP。

## 1.2 项目目标与研究问题 `[主要作者：待定]` — 300 words

- **核心问题：** 团队试图回答什么？
- **主要内容：** 比较价值学习模型；推进 Task 1–4；研究表示、奖励、课程和生存约束。
- **可支持主张：** 项目使用冻结评估和阶段门槛，而不是依据训练 reward 或单局高分选模。
- **证据：** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)、[`experiments/README.md`](experiments/README.md)。
- **图表：** RQ1–RQ7 的紧凑列表。
- **引用：** `[待补引用：深度强化学习实证方法与方差]`。
- **TODO：** 根据篇幅决定是否合并 RQ1/RQ2、RQ3/RQ4。
- **衔接：** 引出中心论点和贡献。

### 1.2.1 范围与成功标准 `[主要作者：待定]`

定义正式得分、生存、旧任务保留、第一名指标和 CPU 计时。明确本地结果不是官方锦标赛结果。

## 1.3 贡献与全文结构 `[主要作者：待定]` — 250 words

- **核心问题：** 本报告的贡献是什么？
- **主要内容：** 表格型与神经网络价值学习器的系统比较；Task 1–4 分阶段开发；生存否决设计；包含负面结果的证据化最终选模。
- **可支持主张：** 贡献主要是方法论和实证经验，而非新的 RL 算法。
- **证据：** [`Q_CNN_AGENT_INDEX.md`](agent_code/Q_CNN_AGENT_INDEX.md)、[`task3-experiment-index.md`](docs/research/task3-experiment-index.md)、[`die_hardest/README.md`](agent_code/die_hardest/README.md)。
- **TODO：** 插入并核验公开仓库 URL。
- **衔接：** 进入任务和强化学习背景。

# 2. Background `[主要作者：待定]` — 1,400 words

## 2.1 Bomberman 任务与环境 `[主要作者：待定]` — 350 words

- **核心问题：** Agent 观察什么、选择什么、优化什么？
- **主要内容：** `game_state`、六种动作、障碍、金币、箱子、炸弹、爆炸、对手、计分和终止条件。
- **可支持主张：** 物理合法性和预测未来生存是两个不同概念。
- **证据：** [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)、`settings.py`、`environment.py`、`items.py`。
- **图表：** 标注棋盘或状态—动作示意图。
- **TODO：** 提交前根据最终课程框架和公告复核数值规则。
- **衔接：** 将交互形式化为序列决策问题。

### 2.1.1 Task 1–4 课程任务 `[主要作者：待定]`

概述金币导航、炸箱与隐藏金币、弱对手交互和强对手竞争。证据：[`experiments/README.md`](experiments/README.md)。

## 2.2 强化学习形式化 `[主要作者：待定]` — 250 words

- **核心问题：** 游戏如何表示为价值学习问题？
- **主要内容：** 状态/特征 \(s_t\)、动作 \(a_t\)、奖励 \(r_t\)、转移、终止标记、折扣回报和动作价值函数。
- **可支持主张：** 学习模型在准入动作中排序；特征和安全层不直接给出最佳动作。
- **证据：** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)、[`CONTEXT.md`](CONTEXT.md)。
- **图表：** 一条 transition tuple。
- **引用：** `[待补引用：马尔可夫决策过程]`。
- **TODO：** 说明 Markov 近似在哪些方面不完整。
- **衔接：** 进入价值学习算法。

## 2.3 价值学习算法 `[主要作者：待定]` — 450 words

- **核心问题：** 项目考虑了哪些算法，为什么？
- **主要内容：** Q-learning、Double Q-learning、Watkins Double Q(λ)、DQN、Double DQN、经验回放、目标网络、n-step、dueling 和 Rainbow-lite 组件。
- **可支持主张：** 不同算法分别针对高估、泛化、经验复用和延迟信用，但其收益必须在本任务中实证检验。
- **证据：** [`experiments/README.md`](experiments/README.md)、[`optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`](agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md)。
- **图表：** 模型与算法比较表。
- **引用：** Q-learning、Double Q-learning、eligibility traces、DQN、Double DQN、dueling、Rainbow 原始论文，全部待核验。
- **TODO：** 正文只保留核心算法方程，边缘变体移入表格或附录。
- **衔接：** 说明为什么只改变算法还不够。

### 2.3.1 结构化与空间表示 `[主要作者：待定]`

对比离散键、tile-coded 连续向量、84 维连续表示、117 维阶段表示、棋盘 CNN 和混合输入。

## 2.4 奖励、课程与安全概念 `[主要作者：待定]` — 350 words

- **核心问题：** 项目如何处理稀疏反馈和危险动作？
- **主要内容：** 奖励塑形、课程学习、物理合法性、预测时域生存、只否决的 mask，以及在剩余动作中的学习选择。
- **可支持主张：** 生存层缩小学习动作集合，但不是直接选择最终动作的规则控制器。
- **证据：** [`CONTEXT.md`](CONTEXT.md)、[`safety-certified-placement-v6.md`](docs/research/safety-certified-placement-v6.md)。
- **图表：** 全部动作 → 物理合法动作 → 生存准入动作 → learned argmax。
- **引用：** 奖励塑形、课程学习、安全强化学习和 action shielding 原始文献，全部待核验。
- **TODO：** 不得将有限时域准入称为全局安全保证。
- **衔接：** 从理论概念转向项目组织。

# 3. Project Planning `[主要作者：待定]` — 900 words

## 3.1 团队结构与协作 `[主要作者：待定]` — 300 words

- **核心问题：** 团队如何满足共同参与要求？
- **主要内容：** A 负责危险/特征，B 负责学习/奖励/回调，C 负责实验/统计/打包；同时设置交叉复核和共同选模。
- **可支持主张：** 分工按模块组织，但通过共享接口和交叉复核保持协作。
- **证据：** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md) 第 9–10 节。
- **图表：** 职责和接口 seam 图。
- **TODO：** 用真实姓名替换 A/B/C，并按实际贡献修正计划分工。
- **衔接：** 说明时间规划和阶段门槛。

### 3.1.1 报告写作分工 `[主要作者：待定]`

原计划：A 负责游戏、危险和特征；B 负责 RL 算法与训练；C 负责规划、评估和综合。最终每个小节必须标注一位真实主要作者。

## 3.2 里程碑与决策门槛 `[主要作者：待定]` — 250 words

- **核心问题：** 团队如何在截止日前安排实验？
- **主要内容：** Task 晋级、开发门槛、独立确认、一次主验证和打包缓冲。
- **可支持主张：** 后续阶段必须以前置能力、安全和工程证据为条件。
- **证据：** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)、[`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md)。
- **图表：** Figure 1。
- **TODO：** 最终报告应写实际时间线，而不只写原始计划。
- **衔接：** 说明复现性和计算资源规划。

## 3.3 可复现性、计算资源与证据管理 `[主要作者：待定]` — 350 words

- **核心问题：** 如何让实验结论可追溯？
- **主要内容：** 配置哈希、checkpoint 哈希、不可变 run ID、环境/对手 seed、JSONL、冻结评估、bootstrap 区间和 CPU timing。
- **可支持主张：** 仓库区分活跃文档、版本化证据、恢复归档和一次性产物。
- **证据：** [`docs/repository-layout.md`](docs/repository-layout.md)、[`experiments/README.md`](experiments/README.md)。
- **图表：** config → checkpoint → evaluation 的证据链。
- **引用：** `[待补引用：深度强化学习可复现性]`。
- **TODO：** 确认实际团队人数；汇总 CPU 型号、GPU 型号、墙钟时间和峰值资源；明确最终打包未在官方 Ryzen 硬件上验证。
- **衔接：** 进入具体技术设计。

# 4. Methods `[主要作者：待定]` — 2,200 words

## 4.1 状态、动作与特征演进 `[主要作者：待定]` — 350 words

- **核心问题：** 学习器获得了哪些信息？
- **主要内容：** 离散表示、compact state、70/78/84 维连续特征、阶段特征、CNN 通道和动作历史。
- **可支持主张：** 表示从紧凑导航描述逐步扩展到危险、对手和时序信息。
- **证据：** [`experiments/README.md`](experiments/README.md)、[`feature-principles-guide.md`](docs/research/feature-principles-guide.md)。
- **图表：** Feature version 演进表。
- **引用：** `[待补引用：强化学习特征工程]`。
- **TODO：** 最终表格中的每个维度都要从源码复核。
- **衔接：** 说明不同学习器如何使用这些表示。

### 4.1.1 物理合法动作与生存准入 `[主要作者：待定]`

依据 [`CONTEXT.md`](CONTEXT.md) 明确定义 physical legal action、horizon-survivable action、survival mask 和 learned action choice。

## 4.2 模型族 `[主要作者：待定]` — 600 words

- **核心问题：** 团队开发了哪些模型？
- **主要内容：** Tabular Q、Double Q、Double Q(λ)、离散 DQN、连续 Double DQN、CNN Double DQN、蒸馏 CNN、Hybrid Dueling、Rainbow-lite、phase model 和最终 B33。
- **可支持主张：** 项目同时探索表示能力和更新规则，但不是每个实现都有有效比较结果。
- **证据：** [`Q_CNN_AGENT_INDEX.md`](agent_code/Q_CNN_AGENT_INDEX.md)、[`experiments/README.md`](experiments/README.md)、`experiments/agent_variants/MANIFEST.md`。
- **图表：** Table 1。
- **TODO：** 生成目录—状态附录，确保“全部开发模型”可以审计。
- **衔接：** 说明共同学习机制。

### 4.2.1 表格型与 Tile-Coded 学习器 `[主要作者：待定]`

介绍 Q-learning、Double Q-learning、Watkins Double Q(λ) 和 trace 截断。强调特征不直接决定动作。

### 4.2.2 神经网络价值学习器 `[主要作者：待定]`

介绍 MLP Double DQN、CNN、蒸馏 CNN、Hybrid Dueling 和 Rainbow-lite，以及目标网络、经验回放和动作 mask。

### 4.2.3 最终 B33 架构 `[主要作者：待定]`

介绍 Double DQN、84 维 `continuous-v2`、`r7_safe_credit_sparse`、n-step 和 survival-mask-v9。证据：[`SUBMISSION_MANIFEST.json`](agent_code/die_hardest/SUBMISSION_MANIFEST.json)。

## 4.3 学习与探索 `[主要作者：待定]` — 300 words

- **核心问题：** 动作价值如何训练？
- **主要内容：** Replay、目标网络更新、Huber loss、Adam、ε-greedy、终止屏蔽、n-step 和确定性冻结推理。
- **可支持主张：** 训练和评估明确分离；最终评估关闭探索和学习。
- **证据：** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)、[`experiments/README.md`](experiments/README.md)。
- **图表：** Double DQN update 算法框。
- **TODO：** 从冻结配置或 checkpoint 中提取最终 B33 超参数，不要直接套用历史默认值。
- **衔接：** 进入奖励和课程迁移。

## 4.4 奖励设计与课程迁移 `[主要作者：待定]` — 350 words

- **核心问题：** 如何将稀疏反馈转化为训练信号？
- **主要内容：** 正式得分与 shaped reward、奖励版本、金币势函数、炸箱奖励、循环惩罚、安全信用和 Task 1–4 transfer。
- **可支持主张：** 奖励塑形改善了部分行为指标，但没有稳定提高正式得分。
- **证据：** [`reward-principles-guide.md`](docs/research/reward-principles-guide.md)、[`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md)、[`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md)。
- **图表：** 奖励版本比较表。
- **引用：** 奖励塑形和 potential-based shaping 原始文献。
- **TODO：** 明确标出哪些比较是真正单变量实验。
- **衔接：** 引出生存约束。

## 4.5 Survival Mask 演进 `[主要作者：待定]` — 400 words

- **核心问题：** 安全层如何演进？
- **主要内容：** v1 时域生存、逃生责任、对手转移、可控生存、严格放弹证明、对手恢复放弹、固定 deadline、已证明移动偏好和 v9 紧凑计算。
- **可支持主张：** 每个版本都针对具体反例，但更强语义和更多对手场景增加了运行时成本。
- **证据：** [`task3-experiment-index.md`](docs/research/task3-experiment-index.md)、[`safety-certified-placement-v6.md`](docs/research/safety-certified-placement-v6.md)、[`safety-rearming-v7.md`](docs/research/safety-rearming-v7.md)、[`safety-fixed-deadline-v8.md`](docs/research/safety-fixed-deadline-v8.md)、[`safety-proven-movement-v9.md`](docs/research/safety-proven-movement-v9.md)。
- **图表：** Figure 2 或安全版本时间线。
- **引用：** 安全强化学习和 action shielding 原始文献。
- **TODO：** 保留 27155 未解决限制。
- **衔接：** 定义评估方法。

## 4.6 评估与统计协议 `[主要作者：待定]` — 200 words

- **核心问题：** 什么结果可以用于正式结论？
- **主要内容：** 开发/确认/主验证划分、固定 seeds、父子配对、10,000 次 bootstrap、正式得分、第一名定义、安全和延迟门槛。
- **可支持主张：** 选模依据是冻结游戏指标，而不是训练 loss 或 reward。
- **证据：** [`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md)、[`experiments/README.md`](experiments/README.md)。
- **图表：** 数据划分表。
- **引用：** `[待补引用：配对 bootstrap 置信区间]`。
- **TODO：** 不得把不兼容 seed 集合当成配对样本。
- **衔接：** 进入训练实践。

# 5. Training `[主要作者：待定]` — 1,300 words

## 5.1 分阶段训练课程 `[主要作者：待定]` — 350 words

- **核心问题：** 训练如何从导航推进到完整竞争？
- **主要内容：** Task 1 金币导航、Task 2 箱子与隐藏金币、Task 3 和平/金币 Agent、Task 4 三个规则 Agent。
- **可支持主张：** 晋级要求同时满足旧任务保留、当前任务表现、安全和工程指标。
- **证据：** [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)、[`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)。
- **图表：** Figure 1。
- **TODO：** 区分课程建议的四阶段与团队自定门槛。
- **衔接：** 说明 checkpoint 如何跨阶段移动。

## 5.2 Checkpoint、Resume 与 Transfer `[主要作者：待定]` — 250 words

- **核心问题：** 如何可复现地继续训练？
- **主要内容：** 原子快照、policy-only 初始化、严格 resume、Task 2→3 和 Task 3→4 显式迁移、Replay 与 RNG 状态。
- **可支持主张：** Resume 和 transfer 是不同合同，防止不兼容 checkpoint 被静默续训。
- **证据：** [`experiments/README.md`](experiments/README.md)、[`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md)。
- **图表：** Checkpoint lineage 图。
- **TODO：** 恢复 B33 的完整公开证据链。
- **衔接：** 讨论计算资源和运行效率。

## 5.3 计算资源与训练效率 `[主要作者：待定]` — 250 words

- **核心问题：** 使用了什么计算资源，为什么？
- **主要内容：** 小型网络的 CPU 训练、CNN 的 GPU 训练、单线程 Torch 推理、向量化/缓存的安全搜索。
- **可支持主张：** 对小型 MLP，GPU 不一定带来吞吐收益；正式评估始终是 CPU-only。
- **证据：** [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)、[`cnn_path_double_dqn_agent/EXPERIMENT_LOG.md`](experiments/agent_variants/cnn_path_double_dqn_agent/EXPERIMENT_LOG.md)。
- **图表：** 计算资源汇总表。
- **TODO：** 汇总核心实验的硬件、墙钟时间和峰值内存；未核验 metadata 前，不把 GTX 1080 Ti 推广到所有 CNN 实验。
- **衔接：** 说明专门训练机制。

## 5.4 蒸馏与 Replay 策略 `[主要作者：待定]` — 200 words

- **核心问题：** 如何保留早期任务能力？
- **主要内容：** 冻结 Task 1 teacher、masked Q-value distillation、按任务划分的 Replay、历史 Replay 和 policy-only 专项迁移。
- **可支持主张：** 蒸馏在部分 CNN transfer 中保留导航，但额外 TD fine-tuning 仍可能导致退化。
- **证据：** [`cnn-task1-task2-experiment-report.md`](docs/research/cnn-task1-task2-experiment-report.md)、[`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md)。
- **图表：** Teacher–student 示意图。
- **引用：** 知识蒸馏与灾难性遗忘原始文献。
- **TODO：** 不得宣称蒸馏保证能力保留。
- **衔接：** 总结改变训练方向的失败。

## 5.5 训练失败与纠正决策 `[主要作者：待定]` — 250 words

- **核心问题：** 哪些失败实质性改变了开发计划？
- **主要内容：** 输入 bug、无效评估启动、循环、Task 2 自杀、callback 生命周期缺陷、安全搜索超时和中断的 Task 4 campaign。
- **可支持主张：** 失败与不完整运行被保留，并从正面结论中排除。
- **证据：** [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md)、[`task3-lifecycle-frozen-results.md`](docs/research/task3-lifecycle-frozen-results.md)、[`task4-frozen-results.md`](docs/research/task4-frozen-results.md)。
- **图表：** 失败 → 诊断 → 决策表。
- **TODO：** 只选能回答研究问题的失败，不写成时间流水账。
- **衔接：** 按研究问题呈现实验结果。

# 6. Experiments and Results `[主要作者：待定]` — 4,500 words

## 6.1 RQ1：哪些模型学会了 Task 1 导航？ `[主要作者：待定]` — 600 words

- **假设：** 更丰富的连续/空间表示和降低高估偏差的学习器应优于基础离散模型。
- **比较对象：** Q-learning、DQN、Double Q、continuous Double DQN、Hybrid Dueling、优化 Double Q(λ)、CNN、蒸馏 CNN。
- **混杂因素：** 早期比较经常同时改变表示、奖励和预算，只能标为探索性比较。
- **数据：** 早期实验主要使用 seeds 10001–10005，每个 20 局；后期模型使用各自记录的开发集和 reserved set。
- **主要指标：** 平均金币、全金币率、回合长度、等待和循环。
- **结果：** 早期 continuous DDQN 达到 49.52 平均金币和 86% 完成率；后期优化 Double Q(λ) 与蒸馏 CNN 在独立/保留评估中约有 96% 完成率。
- **不确定性：** 48.95、49.51、49.58 来自不同 checkpoint 或协议，核清前必须作为不同实验报告。
- **解释：** 结构化表示和算法改动共同带来强导航，但没有一个比较可以单独隔离模型架构效果。
- **证据：** [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md)、[`q-cnn-experiment-report.md`](docs/research/q-cnn-experiment-report.md)、[`optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`](agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md)。
- **图表：** Table 2、Figure 3 的 Task 1 面板。
- **衔接：** 分析导航器为什么仍会停滞。

### 6.1.1 失败或无效的 Task 1 证据 `[主要作者：待定]`

将 Hybrid 的低性能、R2 评估启动失败和 CNN 输入 bug 与有效模型比较分开报告。

## 6.2 RQ2：奖励与历史信息是否减少等待和循环？ `[主要作者：待定]` — 500 words

- **假设：** Idle/loop 惩罚和历史特征应提高完成效率。
- **比较对象：** Q R4 old vs. idle、Double Q R3 vs. R4，以及后续 conditional-loop 方案。
- **混杂因素：** 部分比较同时改变奖励、特征、训练时长或源码版本。
- **结果：** Q 全金币率从 31% 增至 60%，长等待从 58% 降至 19%；Double Q 长等待降至 0%，但乒乓循环升至 25%。
- **解释：** 失败模式发生了转移；减少 WAIT 不等于消除循环，也不能证明某个奖励项产生了纯因果收益。
- **证据：** [`experiment-log-runs-1-2.md`](docs/experiment-log-runs-1-2.md)。
- **图表：** 行为比较表，不标成严格因果消融。
- **衔接：** 检验导航能力能否迁移到炸箱任务。

## 6.3 RQ3：Q 与 CNN 路线如何迁移到 Task 2？ `[主要作者：待定]` — 650 words

- **假设：** 强导航器结合炸弹生存特征，应能学会炸箱和收集隐藏金币。
- **比较对象：** 优化 Double Q(λ)、crate/history 变体、team-demo 变体、蒸馏 CNN D01/D02。
- **指标：** 9 枚金币中的平均金币、炸箱、零放弹率、生存、自杀、WAIT 和循环。
- **结果：** Q(λ) r20 达到 3.95 coins、53.45 crates；CNN D02 主验证为 2.60 coins、44.15 crates，自杀为零但全金币率为零。
- **解释：** 模型比长期地图变化后的目标重选更容易学会即时生存和炸箱。
- **限制：** Q 与 CNN 路线并非全部在同一训练协议下比较。
- **证据：** [`q-cnn-experiment-report.md`](docs/research/q-cnn-experiment-report.md)、[`cnn-task1-task2-experiment-report.md`](docs/research/cnn-task1-task2-experiment-report.md)。
- **图表：** Table 2 和代表性失败轨迹。
- **衔接：** 引出生存 mask 实验。

### 6.3.1 蒸馏成功与 TD Fine-Tuning 退化 `[主要作者：待定]`

报告 Task 1 保留，以及额外 TD fine-tuning 后完成率降至 55% 的观察；不要外推到其他设置。

## 6.4 RQ4：生存约束改善了什么，又付出了什么代价？ `[主要作者：待定]` — 650 words

- **假设：** 否决没有有限时域生存证据的动作，应减少可避免的自杀。
- **比较对象：** v1、v3、v4、v5，以及 v6–v9 工程修订。
- **指标：** 自杀率、炸弹存活、保证丢失、逃生塌缩、搜索超时、P95/max act 时间和能力保留。
- **结果：** 后期版本在多个登记工程语料中实现零观察自杀，但部分强对手搜索超过延迟或内部证明预算。
- **解释：** 更强安全语义修复了具体反例，却产生安全—延迟权衡。
- **限制：** 有限语料上的零失败不是普适安全证明；对手致死和 27155 仍然存在。
- **证据：** [`task3-experiment-index.md`](docs/research/task3-experiment-index.md)、[`safety-certified-placement-v6.md`](docs/research/safety-certified-placement-v6.md)、[`safety-fixed-deadline-v8.md`](docs/research/safety-fixed-deadline-v8.md)、[`task4-frozen-results.md`](docs/research/task4-frozen-results.md)。
- **图表：** Figure 4、安全版本表。
- **衔接：** 检验安全的 Task 3 学习是否改善对战。

## 6.5 RQ5：Task 3 是否在保留旧能力的同时改善竞争表现？ `[主要作者：待定]` — 700 words

- **假设：** Task 3 学习应提高得分或第一名率，同时保留 Task 1/2 能力。
- **比较对象：** 冻结 Task 2 父模型、三个 Task 3 子 seed，以及最终 seed22/c150。
- **数据：** confirmation worlds 21000–21099；main validation 21100–21199；10,000 次配对 bootstrap。
- **结果：** 三个确认得分增量为 +1.92、+1.23、+2.56；主验证得分从 5.91 提高到 7.14，第一名率从 41% 提高到 53%。
- **不确定性：** 得分差 CI [0.27, 2.14]；击杀差 −0.01，CI [−0.16, 0.14]。
- **解释：** 子模型改善了总体竞争结果，但没有证据支持击杀能力提高。
- **修正历史：** 说明原 escape-collapse 计数、诊断和只修改指标判定的修正，同时保留原失败报告。
- **证据：** [`task3-lifecycle-frozen-results.md`](docs/research/task3-lifecycle-frozen-results.md)、[`task3-counter-validation-results.md`](docs/research/task3-counter-validation-results.md)。
- **图表：** Table 3，列出父、子、配对差和 CI。
- **衔接：** 讨论强对手阶段是否继续提高表现。

## 6.6 RQ6：Task 4 尝试为何失败或未显示收益？ `[主要作者：待定]` — 700 words

- **假设：** Replay 比例、得分专项训练、简化奖励、更高 gamma、历史对手训练和后续得分优化可能提升对三个规则 Agent 的表现。
- **结果：**
  - 初始 shared-parent campaign 在正式训练前因搜索超时停止。
  - E1/E2/E3 专项实验未获得最终得分提升；选中差值为 −0.52，CI [−1.055, 0]。
  - 历史对手训练未通过最初晋级规则。
  - 后续 score campaign 在部分证据产生后因 worker failure 停止。
- **解释：** 在登记协议内，简化奖励和加入同源历史对手都没有稳定提高规则对手得分。
- **限制：** 这些是不同的有限实验；失败不能证明方法永远无效。
- **证据：** [`task4-frozen-results.md`](docs/research/task4-frozen-results.md)、[`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md)、[`task4_frozen_20260919/report.md`](experiments/results/task4_frozen_20260919/report.md)、[`task4_score_20260920/report.md`](experiments/results/task4_score_20260920/report.md)。
- **图表：** Table 4。
- **衔接：** 解释为什么最终仍选择 B33。

## 6.7 RQ7：为什么选择 B33 / Die Hardest？ `[主要作者：待定]` — 700 words

- **候选：** Task4 B seed33/c200。
- **方法：** Double DQN、84 维 `continuous-v2`、`r7_safe_credit_sparse`、survival-mask-v9。
- **Benchmark：** 本地 1,000 个世界，对手为三个 rule-based agents。
- **结果：** 平均正式得分 4.231；含并列第一 50.2%；独占第一 37.8%；生存率 94.7%；观察到的框架超时为零。
- **打包证据：** 重命名前后 180 条历史观察的特征、mask、Q 值和动作完全一致；六局配对游戏完整复现行为。
- **工程指标：** 打包等价性测试中，Die Hardest P95 为 14.83 ms、最大 44.60 ms；峰值 RSS 约 319.6 MiB。
- **选模结论：** Die Hardest 是经过测试、可复现、可独立打包且本地得分/生存良好的参赛候选，不是普遍最优模型。
- **关键限制：** 27155 安全合同反例仍未解决；最终打包验收没有重新验证 Docker 或官方锦标赛硬件。
- **证据：** [`die_hardest/README.md`](agent_code/die_hardest/README.md)、[`SUBMISSION_MANIFEST.json`](agent_code/die_hardest/SUBMISSION_MANIFEST.json)、[`die-hardest-submission/REPORT.md`](docs/research/die-hardest-submission/REPORT.md)、[`verification.json`](docs/research/die-hardest-submission/verification.json)。
- **图表：** Table 5。
- **TODO：** 从归档恢复并引用完整 B33 训练/选模报告。当前活跃文件更充分地证明身份与 benchmark，而不是完整候选竞争过程。
- **衔接：** 在不夸大结果的前提下综合发现。

### 6.7.1 打包与行为等价性 `[主要作者：待定]`

把重命名/打包验证与此前 1,000 世界 benchmark 分开。不能写成重命名后的 Agent 重新运行了 1,000 局。

### 6.7.2 剩余安全与外部有效性限制 `[主要作者：待定]`

明确本地硬件、本地对手、27155 未解决和缺少官方锦标赛证据。

# 7. Conclusion `[主要作者：待定]` — 900 words

## 7.1 对研究问题的回答 `[主要作者：待定]` — 400 words

- **主要内容：** Task 1 强导航可实现；Task 2 的目标重选仍困难；survival mask 有效但增加计算成本；Task 3 提高得分；Task 4 修订没有稳定提高得分；B33 是实用性选模结果。
- **证据：** 交叉引用 Tables 2–5。
- **TODO：** 所有结论强度必须与证据强度一致。
- **衔接：** 转入局限。

## 7.2 局限 `[主要作者：待定]` — 250 words

- 不同模型族的协议不完全统一。
- 若干探索实验的独立训练种子有限。
- 使用本地而非官方锦标赛评估。
- Task 2 目标未完成。
- 延迟/死后击杀的奖励信用分配不准确。
- 安全语料有限，27155 仍未解决。
- 最终打包验收没有官方 CPU/Docker 验证。
- **证据：** [`die-hardest-submission/REPORT.md`](docs/research/die-hardest-submission/REPORT.md)、[`task4_exploration_20260919/report.md`](experiments/results/task4_exploration_20260919/report.md)。

## 7.3 未来工作 `[主要作者：待定]` — 250 words

- 修复并独立复测 27155。
- 改善炸箱后长期目标重选。
- 改善延迟和死后击杀的时序信用分配。
- 在统一受控协议下比较主要模型族。
- 增加独立训练 seeds 和更强的 held-out opponents。
- 在接近官方的硬件上 profiling。
- 仅在保持运行时上限的前提下探索学习型对手模型。
- 以“严谨评估与模型容量同样重要”作为全文收束。

## G. 字数预算

仓库中记录的是三人计划，因此暂按 \(N=3\)、总计约 12,000 words 分配。必须确认实际注册成员数。

| 章节 | 建议字数 | 占比 |
|---|---:|---:|
| 1. Introduction | 800 | 6.7% |
| 2. Background | 1,400 | 11.7% |
| 3. Project Planning | 900 | 7.5% |
| 4. Methods | 2,200 | 18.3% |
| 5. Training | 1,300 | 10.8% |
| 6. Experiments and Results | 4,500 | 37.5% |
| 7. Conclusion | 900 | 7.5% |
| **总计** | **12,000** | **100%** |

若实际团队人数为 \(N\)，每章按下式缩放：

\[
\text{新字数} = \text{表中建议字数} \times \frac{N}{3}.
\]

## H. 图表计划

| 图表 | 要回答的问题 | 数据来源 | 设计 | 不可直接比较的内容 | 不确定性 |
|---|---|---|---|---|---|
| Figure 1：课程与选模流水线 | 模型如何走到最终选择？ | Implementation guide、Task 3 验证、最终 manifest | Task 1→4，并标出 train/dev/confirm/main/package gates | 计划分支和实际完成分支必须使用不同线型 | 不需要误差条 |
| Figure 2：Agent 推理流水线 | 动作如何选出？ | Die Hardest 源码、`CONTEXT.md`、安全报告 | State→84 features→Double DQN→legal mask→survival veto→action | 不得画成 safety mask 对动作排序 | 不需要误差条 |
| Figure 3：Task 1–4 能力演进 | 能力如何随任务变化？ | Task 1/2 报告、Task 3 验证、最终 benchmark | 按 Task 分面 | 不得跨不兼容协议画一条连续曲线 | 有配对数据时给 CI |
| Figure 4：得分—安全—延迟权衡 | 更强安全性的代价是什么？ | v6–v9 与 Task 4 报告 | 得分/生存对 P95/max latency | 工程世界和验证世界使用不同标记 | 仅在配对数据支持时给 CI |
| Table 1：模型清单 | 开发了什么？ | Experiments README、manifest、model index | 模型族、输入、学习器、奖励、任务、状态、证据 | “已实现”不等于“已验证” | 不适用 |
| Table 2：Task 1/2 模型 | 哪些模型成功迁移？ | Q/CNN 报告 | 金币、完成率、炸箱、循环、自杀、协议 | 探索结果与独立确认分开 | 报告样本量和 seeds |
| Table 3：Task 3 父子模型 | Task 3 是否改善并保留能力？ | 修正验证报告/JSON | 父、子、差值、CI | 脚注原失败计数 | 配对 bootstrap CI |
| Table 4：Task 4 尝试 | 哪些尝试失败，为什么？ | 四份 Task 4 报告 | 协议、假设、到达阶段、结果、停止原因 | 未运行阶段写“未运行”，不能写 0 | 使用已有 CI |
| Table 5：Die Hardest | 为什么选择它？ | README、verification、manifest | 得分、胜率、生存、时间、内存、包身份 | 1,000 局 benchmark 与六局等价性测试分开 | 给样本量，不编造 CI |

## I. 模型清单与正文覆盖

| 模型族或代表 Agent | 表示 | 证据状态 | 正文位置 |
|---|---|---|---|
| Tabular Q-learning | `discrete-v1` / `discrete-q-v2` | 有效 Task 1 探索结果，后来被超越 | 4.2、6.1、6.2 |
| Double Q-learning | Compact discrete | 有效 Task 1 探索结果 | 4.2、6.1、6.2 |
| Watkins Double Q(λ) | 84 维连续特征与 tile coding | Task 1 独立确认；Task 2 未达目标 | 4.2.1、6.1、6.3 |
| Discrete DQN | 离散特征 | 有效中低性能 Task 1 证据及失败评估 | 4.2.2、6.1 |
| Continuous Double DQN | 70/78/84 维连续特征 | 早期 Task 1 强；Task 2–4 核心谱系 | 4.2.2、6.1–6.7 |
| Phase Double DQN | 117 维阶段特征 | Task 3 安全/保留门槛失败 | 4.1、5.5、6.5 背景 |
| CNN Double DQN | 棋盘/空间通道 | 早期输入问题；后期有有效 Task 1 证据 | 4.2.2、6.1 |
| Distilled CNN Double DQN | 17 通道 CNN 与冻结 teacher | Task 1 强；Task 2 部分迁移 | 5.4、6.1、6.3 |
| Hybrid Dueling Double DQN | 棋盘张量与结构化向量 | 有效低性能结果，另有流程问题 | 4.2、6.1.1 |
| Rainbow-lite | 连续特征与部分 Rainbow 组件 | 已实现；有效结果覆盖仍需审计 | Table 1/附录；缺证据时写 `[证据缺口]` |
| Safety ablation/no-safety | 同一学习器，不同准入规则 | 诊断与消融证据 | 4.5、6.4 |
| Task 4 E1/E2/E3 | 84 维 Double DQN | 完成有限实验；无最终得分收益 | 6.6 |
| 历史对手 C/S | 84 维 Double DQN | 完成初筛；S 未晋级 | 6.6 |
| Task 4 score L/P/K | 84 维 Double DQN | 部分 campaign，终止失败 | 6.6 |
| B33 / Die Hardest | 84 维 Double DQN + survival-mask-v9 | 最终冻结参赛候选 | 4.2.3、6.7 |

正式写作前，应根据 `agent_code/` 和 `experiments/agent_variants/MANIFEST.md` 生成目录级附录，覆盖所有已开发模型，包括没有成功结果的变体。

## J. 待检索引用清单

以下每项必须根据原始论文或权威出版记录核验：

- `[待补引用：马尔可夫决策过程]`
- `[待补引用：Q-learning]`
- `[待补引用：Double Q-learning 与最大化偏差]`
- `[待补引用：Watkins Q(lambda) 与 eligibility traces]`
- `[待补引用：Deep Q-Networks]`
- `[待补引用：Double DQN]`
- `[待补引用：experience replay]`
- `[待补引用：target networks]`
- `[待补引用：multi-step temporal-difference learning]`
- `[待补引用：dueling network architectures]`
- `[待补引用：Rainbow DQN]`
- `[待补引用：reward shaping]`
- `[待补引用：potential-based reward shaping]`
- `[待补引用：curriculum learning]`
- `[待补引用：knowledge distillation]`
- `[待补引用：sequential RL 中的 catastrophic forgetting]`
- `[待补引用：safe reinforcement learning]`
- `[待补引用：action shielding/runtime enforcement]`
- `[待补引用：深度强化学习的可复现性与方差]`
- `[待补引用：paired bootstrap confidence intervals]`

强制引用流程：

1. 搜索准确论文。
2. 使用至少两个权威来源核验论文身份。
3. 确认论文确实支持被引用的主张。
4. 通过程序获取 BibTeX。
5. 无法核验时保留 `[待补引用]`，不得编造。

## K. 缺失信息与 TODO

1. 确认实际团队人数和法定姓名。
2. 将 A/B/C 和所有 `[主要作者：待定]` 替换为真实姓名。
3. 确认 `https://github.com/Jyangwakeup/MLE_final_project` 可公开访问。
4. 确认最终报告 PDF 不会进入公开仓库。
5. 从归档恢复完整 B33 训练与候选选择报告。
6. 从冻结配置/checkpoint 提取 B33 的最终网络超参数。
7. 生成完整模型目录清单，包括被放弃的变体。
8. 核对 Double Q(λ) 的 48.95、49.51、49.58 是否对应不同正式运行/checkpoint。
9. 汇总核心实验的训练硬件和墙钟时间。
10. 核对所有 CNN 运行的 GPU；目前只有部分日志明确记录 GTX 1080 Ti。
11. 区分 CPU 训练、GPU 训练、GPU smoke 和 CPU inference。
12. 如果已有官方锦标赛成绩则加入，否则明确写不可用。
13. 根据最新课程公告和 MaMPF 确认模板与提交要求。
14. 确认附录是否计入课程字数。
15. 只添加经过核验的参考文献。
16. 从机器可读原始结果重新生成论文图。
17. 核对是否存在其他 Docker 验证；最终打包验收报告说明该轮未验证 Docker。
18. 每位成员必须核验并改写本人负责的章节。

## L. 夸大结论与复现风险

- 不得把 survival mask 称为普适或形式化安全保证。
- 不得声称 27155 已修复。
- 不得把零观察自杀等同于安全证明。
- 不得把训练 reward 等同于正式得分。
- 不得合并世界、checkpoint 或协议不同的评估。
- 多变量同时变化时，不得声称单个因素具有因果收益。
- 未启动的 Task 4 阶段必须写“未运行”，不能写成零结果。
- 回归世界不能作为独立测试数据。
- 指标修正后仍须保留原 Task 3 失败报告。
- 不得声称 Task 3 提高击杀；主验证击杀差为 −0.01。
- 不得声称重命名后的 Die Hardest 重新运行了 1,000 世界。
- 不得声称最终模型在统一协议下优于仓库全部模型。
- 六局打包等价性检查不能当作性能估计。
- 本地 CPU timing 不能自动证明官方硬件兼容。
- 同源历史对手不能称为独立未知对手。
- 未完成或中断训练不能描述为收敛。
- Git remote 不能单独证明仓库是公开的。

## M. 最终自检表

| 检查项 | 状态 |
|---|---|
| 恰好七个规定的一级报告章节 | 已覆盖 |
| Introduction 包含问题、价值和挑战 | 已覆盖 |
| Background 包含任务和考虑过的 RL 方法 | 已覆盖 |
| Project Planning 包含协作、时间和计算资源 | 已覆盖；真实姓名和实际硬件待确认 |
| Methods 解释表示、模型、奖励、安全和指标 | 已覆盖 |
| Training 解释课程、resume、transfer、replay 和蒸馏 | 已覆盖 |
| Experiments and Results 占最大篇幅 | 已覆盖：37.5% |
| Conclusion 包含发现、局限和未来工作 | 已覆盖 |
| 每个 section/subsection 有作者字段 | 已用占位符覆盖 |
| 至少比较两个机器学习模型 | 已覆盖 |
| 覆盖全部开发模型族 | 模型族层面已覆盖；仍需目录级附录 |
| 解释最终模型选择 | 已覆盖；B33 完整谱系证据缺口已标出 |
| 包含失败与负面结果 | 已覆盖 |
| 包含公开仓库 URL | 已识别候选 URL；公开可见性待确认 |
| 每位成员约 4,000 words | 已按三人规划；实际人数待确认 |
| 不使用大学 Logo | 最终模板必须落实 |
| 报告 PDF 不进入公开仓库 | 提交前必须检查 |
| 引用经过核验而非编造 | 当前只保留 citation slots，未生成 BibTeX |
| 安全和锦标赛结论强度适当 | 已覆盖 |
| AI 内容被视为需团队复核的草稿 | 提交前必须落实 |

## 推荐写作顺序

1. 第 6 章：Experiments and Results
2. 第 4 章：Methods
3. 第 5 章：Training
4. 第 2 章：Background
5. 第 3 章：Project Planning
6. 第 1 章：Introduction
7. 第 7 章：Conclusion
8. Abstract（如果最终模板要求）

这个顺序先用可核验证据固定论文内容，再编写 Introduction 和 Conclusion 等框架性章节。
