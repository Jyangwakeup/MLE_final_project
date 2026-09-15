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

**Own-bomb escape obligation**:
The interval from selecting `BOMB` until that bomb is gone and every tile in
its corresponding flame has cleared. During this interval the Agent must keep
the strongest available escape redundancy; the obligation resets at a round
boundary.
_Avoid_: Bomb timer only, permanent caution mode

**Robust-survivable action**:
A horizon-survivable action with at least two internally vertex-disjoint paths
through the time-expanded safety graph. It is a veto criterion, never a
recommended movement direction.
_Avoid_: Best escape action, scripted route

**Avoidable escape collapse**:
A transition into post-bomb no-safe-action fallback after an earlier decision
in the same own-bomb cycle had at least one safe alternative. It is counted
once per bomb cycle.
_Avoid_: Every death, unavoidable trap

**Opponent transition scenario**:
One possible next game state produced when every Agent chooses from its
physical legal actions and the environment applies one permitted execution
order, followed by the official world update. It includes resulting positions,
new bombs, explosions, destroyed crates, and deaths.
_Avoid_: Predicted opponent move, most likely move

**Controllable survival set**:
The actions for which a feedback strategy remains alive throughout the current
own-bomb danger interval under every modeled opponent transition. Later actions
may depend on the newly observed state; this is not one fixed route.
_Avoid_: Predetermined escape path, recommended action

**Opponent-robust-survivable action**:
An action admitted by the controllable survival set. v5 proves this against a
conservative superset of official opponent occupancy and bomb danger, so it may
reject extra actions but may not accept an unproved action.
_Avoid_: Likely-safe action, opponent prediction

**Robust bomb-placement veto**:
The refusal to place a bomb when placement lacks a completed controllable
survival proof and a v1-safe non-bomb alternative exists. It assigns the safety
decision to the action that creates the future obligation.
_Avoid_: Bomb penalty, rule-selected movement

**Own-bomb danger interval**:
The placement transition through the last dangerous flame produced by that
bomb, while also respecting all other timed explosions. The official framework
does not implement chain reactions, so the model must not invent them.
_Avoid_: Countdown only, chain-reaction window

**Fail-closed robust evaluation**:
If the survival proof exceeds its internal budget or cannot be completed, the
candidate is not marked robust. A timeout is recorded and fails experiment
admission even if rejecting the action happened to prevent death.
_Avoid_: Timeout success, safe by default

**Robust guarantee loss**:
During an active own-bomb obligation, no action passes v5 and execution falls
back to the v1 survival set (or ultimately physical Q). It is distinct from a
death and must be zero at the safety gates.
_Avoid_: Ordinary intervention, every fallback

**Safety regression corpus**:
The immutable eleven environment seeds containing seven legacy Task 3
self-deaths and four n-step-5 Task 3 self-deaths. It is used before any fresh
evaluation seeds, never for model selection.
_Avoid_: Development set, training seeds

**Frozen safety admission**:
The one-time 100-seed evaluation of unchanged weights under v5. Passing requires
the fixed safety, capability-retention, bomb-use, invalid-action, and latency
gates before any new Task 3 training is allowed.
_Avoid_: Training result, Task 3 winner

**Coin capability assessment**:
A frozen, exploration-free Task 1 evaluation over the dedicated convergence seeds. Its primary result is mean game score; Task 1 requires every episode's score to equal its collected-coin count.
_Avoid_: Training reward, recent training score

**Task 1 score convergence**:
Three consecutive coin capability assessments whose mean score is at least 48, with assessments separated by 50 newly trained rounds.
_Avoid_: Action budget reached, training completed

**Curriculum promotion qualification**:
Task 1 score convergence followed by a passing, independent stage-gate evaluation and all engineering checks. Convergence alone does not authorize promotion.
_Avoid_: Converged, completed
