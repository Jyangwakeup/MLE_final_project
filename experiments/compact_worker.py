"""Admission worker: same safety checks, stricter complete-act latency margin."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import task4_worker


def main(argv=None):
    original=task4_worker.safety_failure
    def checked(state,action,safety,*,elapsed,skipped=False,timed_out=False):
        reason=original(state,action,safety,elapsed=elapsed,skipped=skipped,timed_out=timed_out)
        if reason:return reason
        if elapsed>.25:return 'compact_act_max_exceeded'
        return None
    task4_worker.safety_failure=checked
    try:return task4_worker.main(argv)
    finally:task4_worker.safety_failure=original

if __name__=='__main__':raise SystemExit(main())
