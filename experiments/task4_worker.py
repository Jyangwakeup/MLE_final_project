"""Run the existing runner with fail-fast Task 4 observation auditing."""
from __future__ import annotations
import json
from pathlib import Path
import pickle
import sys
from collections import deque

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import run

AGENT = 'double_dqn_continuous_v2_agent'
ACTIONS = ('UP','RIGHT','DOWN','LEFT','WAIT','BOMB')


def safety_failure(state, action, safety, *, elapsed, skipped=False, timed_out=False):
    if skipped or timed_out:
        return 'act_timeout_or_skip'
    if elapsed is None or elapsed > .48:
        return 'act_max_exceeded'
    if not safety or safety.get('diagnostic_version') != 'escape-collapse-v2':
        return 'missing_safety_diagnostic'
    for key in ('robust_search_timed_out','robust_guarantee_loss','avoidable_escape_collapse'):
        if safety.get(key):
            return key
    if not state['self'][2] and not safety.get('own_bomb_pending'):
        return 'lost_bomb_responsibility'
    if (action not in ACTIONS or safety.get('selected_action') != action
            or not safety['physical_mask'][ACTIONS.index(action)]
            or not safety['decision_mask'][ACTIONS.index(action)]):
        return 'decision_mask_mismatch'
    return None


def main(argv=None):
    original = run.ExperimentWorld._append_timing
    original_end = run.ExperimentWorld.end_round
    traces = {}
    def audited(world, agent, action, requested_action, think_time, skipped,
                timed_out, available_before, available_after, game_state):
        original(world,agent,action,requested_action,think_time,skipped,timed_out,
                 available_before,available_after,game_state)
        if agent.name != AGENT:
            return
        owner = agent.backend.runner.fake_self
        safety = getattr(owner,'last_safety_diagnostic',{})
        failure = safety_failure(game_state,action,safety,elapsed=think_time,
                                 skipped=skipped,timed_out=timed_out)
        key = world._experiment_run_id
        ring = traces.setdefault(key,deque(maxlen=20))
        ring.append(dict(state=game_state,action=action,safety=safety,think_time=think_time))
        if failure:
            world._timing_file.flush()
            directory=world._timing_path.parent
            (directory/'task4_failure_states.pkl').write_bytes(pickle.dumps(list(ring)))
            (directory/'task4_safety_failure.json').write_text(json.dumps(dict(
                reason=failure,round=int(world.round),step=int(world.step),
                action=action,think_time=think_time,safety=safety),indent=2)+'\n')
            raise RuntimeError('Task 4 safety engineering failure: '+failure)
    def audited_end(world):
        original_end(world)
        key=world._experiment_run_id
        if any(a.name==AGENT and a.dead for a in world.agents):
            directory=world._timing_path.parent
            (directory/f'task4_death_states_round{world.round:04d}.pkl').write_bytes(
                pickle.dumps(list(traces.get(key,[]))))
        traces.pop(key,None)
    run.ExperimentWorld.end_round = audited_end
    run.ExperimentWorld._append_timing = audited
    try:
        return run.main(argv)
    finally:
        run.ExperimentWorld._append_timing = original
        run.ExperimentWorld.end_round = original_end

if __name__=='__main__':
    raise SystemExit(main())
