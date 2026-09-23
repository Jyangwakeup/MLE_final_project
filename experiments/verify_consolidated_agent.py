"""Verify frozen Die Hardest using only main and consolidated archive paths."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
SCORE=ROOT/'archive/worktrees/MLE_final_project_task4_score'
DIAG=SCORE/'.scratch/task4-diagnosis'
HISTORY=ROOT/'.scratch/die-hardest-submission'
ZIP_SHA='0f64069f1ebdf7e7097780686514d123b245e6f71c1920e76ab60372303ac928'


def norm(value):
    if isinstance(value,dict):return {k:norm(v) for k,v in value.items() if k not in ('seconds','time')}
    if isinstance(value,list):return [norm(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True)
    result={'status':'running','source_root':str(SCORE),'checks':[]};start=time.monotonic()
    env={k:v for k,v in os.environ.items() if not k.startswith(('BOMBERMAN_','PYTHON'))}
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy')
    def run(label,argv,cwd,expected=0):
        r=subprocess.run([sys.executable,*map(str,argv)],cwd=cwd,env=env,capture_output=True,text=True,timeout=60)
        (out/(label+'.log')).write_text(r.stdout+r.stderr)
        if r.returncode!=expected:raise RuntimeError((label,r.returncode,r.stderr[-1000:]))
        result['checks'].append(label)
    try:
        frozen=json.loads((ROOT/'docs/research/branch-integration/frozen-files.json').read_text())
        for name,h in frozen.items():assert hashlib.sha256((ROOT/'agent_code'/name).read_bytes()).hexdigest()==h
        source=DIAG/'b33-submission-check/candidate-not-approved/final-project-agent-code.zip'
        # The real maintained CLI must work with the new source location.
        run('rebuild',['experiments/package_die_hardest.py','--source',source],ROOT)
        assert hashlib.sha256((ROOT/'output/die-hardest-submission/final-project-agent-code.zip').read_bytes()).hexdigest()==ZIP_SHA
        framework=out/'official-framework';framework.mkdir()
        raw=subprocess.check_output(['git','archive','c4ecff8','bomberman_rl'],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:t.extractall(framework,filter='data')
        for q in (framework/'bomberman_rl').iterdir():q.rename(framework/q.name)
        (framework/'bomberman_rl').rmdir();(framework/'logs').mkdir(exist_ok=True)
        shutil.copytree(ROOT/'agent_code/die_hardest',framework/'agent_code/die_hardest',ignore=shutil.ignore_patterns('__pycache__'))
        # Historical script bytes remain archived. Only these throwaway adapters change paths.
        trace=(HISTORY/'die_hardest/trace_rename.py').read_text()
        old="D=Path('/export/data/sfan/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-submission-check')"
        assert old in trace
        trace=trace.replace(old,'D=Path('+repr(str(DIAG/'b33-submission-check'))+')')
        (framework/'trace.py').write_text(trace)
        run('trace',['trace.py'],framework)
        actual=json.loads((framework/'trace.json').read_text());expected=json.loads((HISTORY/'double_dqn_continuous_v2_agent/trace.json').read_text())
        assert len(actual['rows'])==180 and actual['rows']==expected['rows'] and actual['threads']==1
        result['trace_rows_equal']=180
        safety=(HISTORY/'die_hardest/safety_check.py').read_text()
        safety=safety.replace("ROOT = Path('/export/data/sfan/MLE_final_project_task4_score')",'ROOT = Path('+repr(str(SCORE))+')')
        safety=safety.replace("sys.path.insert(0, '/export/data/sfan/MLE_final_project/.scratch/die-hardest-submission/die_hardest')",'sys.path.insert(0, '+repr(str(framework))+')')
        (framework/'safety.py').write_text(safety)
        run('known-27155',['safety.py','--start-step','73','--output',out/'safety.json'],framework,expected=1)
        safety_result=json.loads((out/'safety.json').read_text());assert safety_result['failure_steps']==[77]
        result['known_unfixed_failure_steps']=[77]
        report=json.loads((DIAG/'b33-random1000/summary.json').read_text());assert report
        result['historical_1000_summary_readable']=True
        shutil.copy2(HISTORY/'die_hardest/play_one.py',framework/'play_one.py');result['games']=[]
        for index,opponent in [(0,'random_agent'),(3,'rule_based_agent')]:
            env['TEST_OPPONENT']=opponent;target=out/('game-'+opponent)
            run(opponent,['play_one.py',HISTORY/f'config-{index}.json',target],framework)
            current=json.loads((target/'result.json').read_text());acts=json.loads((target/'acts.json').read_text())
            old=json.loads((HISTORY/f'game-{index}-die_hardest/result.json').read_text());old_acts=json.loads((HISTORY/f'game-{index}-die_hardest/acts.json').read_text())
            assert current['status']=='completed' and norm(current['final'])==norm(old['final']) and norm(acts)==norm(old_acts)
            log=(framework/'logs/game.log').read_text();(target/'game.log').write_text(log)
            assert 'exceeded think time' not in log and 'Skipping agent' not in log
            assert not any(a[k] for a in acts for k in ['robust_search_timed_out','robust_guarantee_loss','avoidable_escape_collapse'])
            times=sorted(a['seconds'] for a in acts);p95=times[int(.95*(len(times)-1))]*1000;maximum=max(times)*1000
            assert p95<=100 and maximum<=300 and current['peak_rss_bytes']<8*1024**3
            result['games'].append({'opponent':opponent,'decisions':len(acts),'p95_ms':p95,'max_ms':maximum,'peak_rss_bytes':current['peak_rss_bytes'],'historical_behavior_equal':True})
        result['status']='passed'
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        result['seconds']=time.monotonic()-start;(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
