#!/usr/bin/env python3
"""One existing cron entry: execute at most one task, then accept its evidence."""
import argparse
import json
from pathlib import Path
import sys
from runner import Runner, atomic_json, now
from main_control import Control, encoded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root)
    try:
        status = Runner(root).tick()
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if status['status'] == 'busy':
            # The active parent owns task execution and heartbeat transport.
            # Do not race its Git fetch/push from overlapping cron invocations.
            return 0
        control = Control(root)
        processed = control.process()
        print(json.dumps(processed, ensure_ascii=False), flush=True)
        state, _ = control.read()
        current = dict(worker_id='server2', mode=state['mode'], active_job_id=state['active_job_id'],
                       runner_status=status['status'], pause_requested=state['pause_requested'])
        if state.get('extension'):
            current['extension'] = {k: state['extension'].get(k) for k in ('scope_id','status')}
        path = 'workers/server2-main-status.json'
        raw = control.runner.read(control.runner.ref('results_branch'), path)
        previous = json.loads(raw) if raw else {}
        previous.pop('observed_at', None)
        if previous != current:
            control.runner.publish({path: encoded(dict(current, observed_at=now()))})
    except Exception as error:
        # Durable, bounded diagnosis. Retry transmission, never retry the job.
        failure = dict(status='main_service_error', error=type(error).__name__+': '+str(error),
                       new_jobs_paused=True)
        atomic_json(root/'main-service-error.json', failure)
        runner = Runner(root)
        runner.pause(failure['error'])
        try:
            runner.publish({'workers/server2-main-error.json': encoded(failure)})
        except Exception:
            pass
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
