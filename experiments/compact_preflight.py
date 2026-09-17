"""Complete offline proofs and fixed-core measurements before freezing admission."""
import json,pickle,hashlib,subprocess,sys,time,gc,os,logging
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tests.test_order_equivalence import native
from tests.reference_grid_survival import controllable_survival_actions as reference
from agent_code.team_agent.controllable_survival import controllable_survival_actions as solve
from agent_code.team_agent.opponent_transitions import ACTIONS
from tests.test_compact_survival import synthetic_states


def corpus():
    worktrees=[l[9:] for l in subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True).splitlines() if l.startswith('worktree ')]
    files=[];seen_files=set();states={}
    for tree in worktrees:
        for directory,dirs,names in os.walk(tree):
            dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.venv','node_modules')]
            for name in names:
                if 'failure' not in name or 'states' not in name or not name.endswith(('.json','.pkl')):continue
                path=Path(directory)/name;data=path.read_bytes();sha=hashlib.sha256(data).hexdigest();files.append({'path':str(path),'sha256':sha})
                if sha in seen_files:continue
                seen_files.add(sha)
                rows=pickle.loads(data) if name.endswith('.pkl') else json.loads(data)
                if not isinstance(rows,list):continue
                for row in rows:
                    if not isinstance(row,dict) or 'state' not in row:continue
                    state=row['state'];state['field']=np.asarray(state['field']);state['explosion_map']=np.asarray(state['explosion_map']);state['self']=(*state['self'][:3],tuple(state['self'][3]));state['others']=[(*a[:3],tuple(a[3])) for a in state['others']];state['bombs']=[(tuple(p),t) for p,t in state['bombs']];state['coins']=[tuple(p) for p in state['coins']]
                    key=hashlib.sha256(json.dumps(native(state),sort_keys=True).encode()).hexdigest()
                    states.setdefault(key,dict(state=state,origin=str(path),sha256=key))
    return files,list(states.values())


def main():
    out=ROOT/'runs/compact_preflight_02';out.mkdir(parents=True,exist_ok=False)
    files,rows=corpus();(out/'corpus.json').write_text(json.dumps({'files':files,'states':[{'sha256':r['sha256'],'origin':r['origin'],'step':r['state']['step']} for r in rows]},indent=2)+'\n')
    print('unique historical states',len(rows),flush=True)
    samples=[];kw=dict(remaining_steps=7,budget_ms=60000,consider_opponent_rearming=True)
    worst=None;maximum=-1
    for index,row in enumerate(rows):
        state=row['state'];expected=reference(state,ACTIONS,**kw);stats={};observed=solve(state,ACTIONS,work_stats=stats,**kw)
        assert not expected.timed_out and not observed.timed_out
        assert asdict(expected)==asdict(observed),(index,row['origin'],expected,observed)
        for repeat in range(11):
            for name,fn in ((('old',reference),('new',solve)) if repeat%2==0 else (('new',solve),('old',reference))):
                cpu=time.process_time();start=time.perf_counter();r=fn(state,ACTIONS,remaining_steps=7,budget_ms=400,consider_opponent_rearming=True);elapsed=time.perf_counter()-start
                item=dict(index=index,version=name,repeat=repeat,warmup=repeat==0,seconds=elapsed,cpu_seconds=time.process_time()-cpu,timed_out=r.timed_out)
                if name=='new':assert not r.timed_out,item
                samples.append(item)
                if name=='new' and elapsed>maximum:maximum=elapsed;worst=state
        with (out/'measurements.jsonl').open('a') as f:
            for item in samples[-22:]:f.write(json.dumps(item)+'\n')
        with (out/'work_counts.jsonl').open('a') as f:f.write(json.dumps({'index':index,**stats})+'\n')
        if index%20==0:print('historical',index,'max new ms',maximum*1000,flush=True)
    pauses=[];started={}
    def observer(phase,info):
        g=info['generation']
        if phase=='start':started[g]=time.perf_counter()
        else:pauses.append({'generation':g,'seconds':time.perf_counter()-started.pop(g)})
    gc.callbacks.append(observer)
    try:
        for repeat in range(500):
            pauses.clear();cpu=time.process_time();start=time.perf_counter();r=solve(worst,ACTIONS,remaining_steps=7,budget_ms=400,consider_opponent_rearming=True)
            item=dict(repeat=repeat,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,gc=list(pauses),timed_out=r.timed_out)
            with (out/'tail.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            assert not r.timed_out,item
    finally:gc.callbacks.remove(observer)
    # Full callback timings only where no historical obligation reconstruction is
    # needed; pending lifecycle is separately exercised in complete-world tests.
    from agent_code.double_dqn_continuous_v2_agent import callbacks
    from agent_code.learning_common.action_history import init_action_history
    spec=json.loads((ROOT/'experiments/configs/safety_proven_movement_v9.json').read_text())['safety']
    os.environ.update(BOMBERMAN_CHECKPOINT=str(ROOT/'agent_code/double_dqn_continuous_v2_agent/task3_validated.pt'),BOMBERMAN_SAFETY_SPEC=json.dumps(spec),BOMBERMAN_AGENT_SEED='22')
    owner=SimpleNamespace(train=False,logger=logging.getLogger('compact-preflight'));callbacks.setup(owner)
    full_rows=[r['state'] for r in rows if bool(r['state']['self'][2])]
    full_rows += [s for _,s in synthetic_states(128) if s['self'][2]]
    timings=[]
    for index,state in enumerate(full_rows):
        for repeat in range(11):
            init_action_history(owner);start=time.perf_counter();action=callbacks.act(owner,state);elapsed=time.perf_counter()-start
            if repeat:timings.append(elapsed)
            assert not owner.last_safety_diagnostic['robust_search_timed_out']
            with (out/'full_act.jsonl').open('a') as f:f.write(json.dumps(dict(index=index,repeat=repeat,seconds=elapsed,action=action))+'\n')
    timings.sort();p95=timings[int(np.ceil(.95*len(timings)))-1];peak=max(timings)
    assert p95<=.1 and peak<=.25,(p95,peak)
    result=dict(status='passed',historical_states=len(rows),full_act_samples=len(timings),act_p95_ms=1000*p95,act_max_ms=1000*peak,search_max_ms=maximum*1000,affinity=sorted(os.sched_getaffinity(0)),search_budget_ms=400,offline_budget_ms=60000)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':main()
