# Bomberman Learning Context

This context defines the project-specific language used to separate game legality, predicted survival, and learned action choice.

## Language

**Physical legal action**:
An action the current game state permits under collision, occupancy, bomb availability, and curriculum rules. Physical legality makes no claim about future survival.
_Avoid_: Valid move, safe move

**Horizon-survivable action**:
A physical legal action for which at least one continuation remains alive through the complete configured prediction horizon.
_Avoid_: Good action, optimal action

**Avoidable fatal action**:
A physical legal action that is not horizon-survivable in a state containing at least one horizon-survivable alternative.
_Avoid_: Bad move, unsafe action

**Survival mask**:
The subset of physical legal actions that are horizon-survivable. It is a veto boundary, not a ranking or recommendation among the remaining actions.
_Avoid_: Safety policy, rule-based controller

**Safety intervention**:
A decision where the learner's highest-valued physical legal action is vetoed by the survival mask and another learned action is selected.
_Avoid_: Rule action, corrected action

**No-safe-action fallback**:
A decision state with no horizon-survivable physical legal action, in which the learner selects its highest-valued physical legal action without a safety veto.
_Avoid_: Random fallback, rescue action

**Coin capability assessment**:
A frozen, exploration-free Task 1 evaluation over the dedicated convergence seeds. Its primary result is mean game score; Task 1 requires every episode's score to equal its collected-coin count.
_Avoid_: Training reward, recent training score

**Task 1 score convergence**:
Three consecutive coin capability assessments whose mean score is at least 48, with assessments separated by 50 newly trained rounds.
_Avoid_: Action budget reached, training completed

**Curriculum promotion qualification**:
Task 1 score convergence followed by a passing, independent stage-gate evaluation and all engineering checks. Convergence alone does not authorize promotion.
_Avoid_: Converged, completed

**Task 2 winner**:
The single Task 2 checkpoint whose configuration passed independent development gates for all three training seeds and then passed the one permitted main validation. A development-selected checkpoint is only a Task 2 winner candidate.
_Avoid_: Best training run, winner configuration, winner candidate

**Task 3 parent baseline**:
A frozen evaluation of one training seed's direct Task 2 parent on the Task 1, Task 2, and Task 3 development worlds before Task 3 learning starts.
_Avoid_: Global baseline, published winner baseline

**Task 3 pilot checkpoint**:
A Task 3 checkpoint produced at a preregistered 500-round assessment boundary. It remains experimental until its training seed passes every capability, retention, safety, and engineering gate.
_Avoid_: Converged model, Task 3 winner

**Task 3 pilot candidate**:
The single development-selected checkpoint from a Task 3 configuration that passed every gate independently for all three training seeds. It is not qualified for Task 4 until a later independent main validation passes.
_Avoid_: Task 3 winner, Task 4 parent

**Situation progress**:
An observable scalar made from 45% crate depletion, 30% opponent depletion, and 25% round progress. It describes the board state rather than imposing fixed turn-number phases.
_Avoid_: Training progress, hard phase switch

**Reward phase mixture**:
The continuous triangular early/middle/late weights derived from situation progress. They sum to one and mix event values and state potentials without choosing an action.
_Avoid_: Reward schedule by training step, scripted strategy

**Mobility reserve**:
The normalized size of the best reachable H=7 safe endpoint set. It is a factual safety capacity, not a direction recommendation.
_Avoid_: Escape action, safe route

**Mobility crisis**:
A state with at most one horizon-survivable action and a safe endpoint ratio no greater than 10%. The late-phase metric counts these states; geometric edge occupancy is only diagnostic.
_Avoid_: Edge tile, certain death

**Task 2-to-phase transfer**:
The explicit v7-to-v8 operation that copies the 84 parent input columns into a 117-input network, initializes 33 new columns to zero, and resets optimizer and replay. It is not exact resume.
_Avoid_: Curriculum resume, checkpoint continuation

**Task 3 winner candidate**:
The single checkpoint selected only after one configuration passes the new-seed confirmation independently for training seeds 11, 22, and 33.
_Avoid_: Task 3 winner, best training reward

**Task 3 winner**:
The sole Task 3 winner candidate that also passes the one permitted 100-seed main validation. Only this designation may set `qualified_for_task4=true`.
_Avoid_: Pilot candidate, runner-up
