"""Fail-fast observation of every owned neural seat, with unmodified v9 decisions."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import pickle
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import run
from experiments.task4_worker import AGENT, safety_failure
from experiments.task4_score_transfer import VERSION, validate_resume, validate_archive
from experiments.task4_transfer import sha256


def owned_failure(state, action, safety, *, learner, elapsed, skipped=False, timed_out=False):
    # The 100/300ms candidate margins apply to the learner. Frozen opponents
    # retain the actual framework timeout/skip and identical proof invariants;
    # their full timing distribution is reported without an extra selection gate.
    checked_elapsed=elapsed if learner or elapsed is None else 0.
    failure=safety_failure(state,action,safety,elapsed=checked_elapsed,skipped=skipped,timed_out=timed_out)
    if learner and elapsed is not None and elapsed>.300:return failure or 'act_max_exceeded_300ms'
    return failure


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--frozen-manifest',required=True)
    parser.add_argument('--match-kind',choices=['training','rules','heldout','engineering'],required=True)
    parser.add_argument('--role',choices=['candidate','reference','archive'],required=True)
    args,remaining=parser.parse_known_args(argv)
    manifest=json.loads(Path(args.frozen_manifest).read_text())
    if manifest['schema_version']!=VERSION:raise ValueError('Wrong frozen-opponent protocol')
    config_path=Path(remaining[remaining.index('--config')+1]).resolve()
    allowed=([manifest['reference_config']] if args.role=='reference' else [manifest['archive_config']] if args.role=='archive' else list(manifest['arm_configs'].values()))
    if config_path not in [(ROOT/p).resolve() for p in allowed]:raise ValueError('Role/config mismatch')
    config=json.loads(config_path.read_text())
    if (ROOT/config['frozen_opponents']['manifest']).resolve()!=Path(args.frozen_manifest).resolve():
        raise ValueError('Worker manifest/config mismatch')
    mode=remaining[remaining.index('--mode')+1]
    if (mode=='train') != (args.match_kind=='training'):raise ValueError('Opponent role/mode mismatch')
    if args.role=='reference':
        if mode!='evaluate' or sha256(remaining[remaining.index('--checkpoint')+1])!=manifest['parent']['checkpoint_sha256']:
            raise ValueError('Reference checkpoint/mode mismatch')
    elif mode=='evaluate':
        import torch
        payload=torch.load(remaining[remaining.index('--checkpoint')+1],map_location='cpu',weights_only=True)
        if args.role=='archive':validate_archive(payload,remaining[remaining.index('--checkpoint')+1],manifest)
        else:validate_resume(payload.get('transfer_contract'),root=ROOT,config_path=config_path,seed=payload['agent_seed'])
    if args.role=='archive' and mode!='evaluate':raise ValueError('Archived controls are evaluation-only')
    os.environ['BOMBERMAN_FROZEN_MATCH_KIND']=args.match_kind
    original=run.ExperimentWorld._append_timing;original_end=run.ExperimentWorld.end_round
    traces={}
    def neural(agent):
        delegate=getattr(agent.backend.runner.fake_self,'frozen_delegate',None)
        return agent.name==AGENT or (delegate is not None and delegate.kind=='neural')
    def audited(world,agent,action,requested_action,think_time,skipped,timed_out,available_before,available_after,game_state):
        original(world,agent,action,requested_action,think_time,skipped,timed_out,available_before,available_after,game_state)
        if not neural(agent):return
        owner=agent.backend.runner.fake_self;safety=getattr(owner,'last_safety_diagnostic',{})
        ring=traces.setdefault((world._experiment_run_id,agent.name),deque(maxlen=20))
        ring.append(dict(state=game_state,action=action,safety=safety,think_time=think_time))
        failure=owned_failure(game_state,action,safety,learner=agent.name==AGENT,elapsed=think_time,skipped=skipped,timed_out=timed_out)
        if (agent.name==AGENT and args.role=='reference' and world._environment_seed in manifest['diagnostic_seeds']
                and int(world.step) in (1,10,20,40,60,80,100,150,200,250,300,350)):
            snapshot=getattr(owner,'_decision_snapshot',None)
            if snapshot is not None:
                record=dict(world=world._environment_seed,step=int(world.step),features=snapshot.vector.tolist(),
                            mask=snapshot.mask.tolist(),
                            q=owner.model.q_values(snapshot.vector.copy()).tolist(),action=action)
                with (world._timing_path.parent/'q_probes.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        if failure:
            world._timing_file.flush()
            directory=world._timing_path.parent
            (directory/f'failure_states_{agent.name}.pkl').write_bytes(pickle.dumps(list(ring)))
            (directory/'task4_safety_failure.json').write_text(json.dumps(dict(reason=failure,agent=agent.name,
                model_id=getattr(owner,'frozen_model_id','learner'),round=int(world.round),step=int(world.step),safety=safety,think_time=think_time),indent=2)+'\n')
            raise RuntimeError('Frozen opponent engineering failure: '+agent.name+': '+failure)
    def audited_end(world):
        was_running=world.running
        original_end(world)
        if not was_running:return
        for agent in world.agents:
            if not neural(agent):continue
            ring=traces.pop((world._experiment_run_id,agent.name),[])
            if agent.dead:
                (world._timing_path.parent/f'death_{agent.name}_round{world.round:04d}.pkl').write_bytes(pickle.dumps(list(ring)))
            if agent.statistics.get('suicides',0):
                (world._timing_path.parent/'task4_safety_failure.json').write_text(json.dumps(dict(
                    reason='unexplained_self_death',agent=agent.name,round=int(world.round)),indent=2)+'\n')
                raise RuntimeError('Unexplained self death: '+agent.name)
    run.ExperimentWorld._append_timing=audited;run.ExperimentWorld.end_round=audited_end
    try:return run.main(remaining)
    finally:
        run.ExperimentWorld._append_timing=original;run.ExperimentWorld.end_round=original_end

if __name__=='__main__':raise SystemExit(main())
