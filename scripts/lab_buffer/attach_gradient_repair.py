#!/usr/bin/env python3
"""Attach the single evidence-bound preflight repair to an idle existing Runner.

Run from a clean pinned detached checkout. Never touches the main checkout,
cron, input files or processes; CPU checks precede the narrowly scoped upgrade.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from runner import Runner, atomic_json, digest, now
from main_control import Control, encoded
from gradient_bridge import CONTROLLED, CONTROLLED_REPAIR, SCOPE, task

ROOT = Path(__file__).resolve().parents[2]
INSTALLED = 'b8e40bd251980509aa48f8cca5d700d0fbee3311'
FAILED_CODE = 'e225ebd9547aa6eb89a23e4a4f909ca4e9e0608d'
FAILED_ARTIFACT = 'e87e19e408001f2d04bf6932e64d591d57f8927de8031038f779d2b157f2415f'


def failure_evidence(r):
    raw=r.read(r.ref('results_branch'),'jobs/'+CONTROLLED+'.json')
    result=json.loads(raw)
    path='artifacts/'+CONTROLLED+'/outputs/controlled/compact.json'
    artifact_raw=r.read(r.ref('results_branch'),path)
    request_raw=r.read(r.ref('requests_branch'),'jobs/'+CONTROLLED+'.json')
    artifact=json.loads(artifact_raw)
    if (result['status']!='failed' or result['exit_code']!=1 or not result.get('finished_at')
            or result['actual_code_sha']!=FAILED_CODE or result['request_sha256']!=digest(request_raw)
            or result['artifacts_sha256'].get(path)!=FAILED_ARTIFACT or digest(artifact_raw)!=FAILED_ARTIFACT
            or artifact['optimizer_step_count_evidence'] is not None
            or 'positive hard training H' not in artifact['mandatory_checks_log_tail']
            or 'subprocess.run' not in artifact['traceback']):
        raise ValueError('Not the exact terminal synthetic positive-H preflight failure')
    return result


def attach(root):
    root=Path(root).resolve();r=Runner(root);c=Control(root)
    code=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','HEAD','--'],check=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'PAUSED.json').exists():raise ValueError('Uncertainty pause requires independent reconciliation')
        state,_=c.read()
        if (state['active_job_id']==CONTROLLED_REPAIR and state['code_sha']==code
                and state['extension'].get('controlled_repair_authorization')):
            return dict(status='already_attached',job_id=CONTROLLED_REPAIR,code_sha=code)
        if (state['active_job_id']!=CONTROLLED or state['mode'] not in ('needs_diagnosis','awaiting_runner_upgrade')
                or not state['pause_requested'] or state.get('lease')
                or state['extension'].get('scope_id')!=SCOPE or not state['extension'].get('logs_analysis_receipt')
                or state['extension'].get('repair_delivery',{}).get('code_sha')!=code
                or r.config.get('main_scope_revision')!=2):
            raise ValueError('Require paused diagnosed zero-update failure and no live controller lease')
        failure_evidence(r)
        for path in (root/'state').glob('*.json'):
            result=json.loads(path.read_text())['result']
            if result['status'] in ('starting','running'):
                raise ValueError('A local task may still be running; no upgrade')
        d=json.loads((root/'main-deployment.json').read_text())
        if Path(d['runner_root']).resolve()!=root:raise ValueError('Deployment root mismatch')
        before={}
        for name in ('runner.py','main_control.py','main_service.py','gradient_logs.py','gradient_bridge.py'):
            expected=digest(subprocess.check_output(['git','-C',str(ROOT),'show',INSTALLED+':scripts/lab_buffer/'+name]))
            actual=digest((root/'bin'/name).read_bytes());before[name]=actual
            if actual!=expected:raise ValueError('Unexpected installed helper; no overwrite: '+name)
            if name!='gradient_bridge.py' and digest((ROOT/'scripts/lab_buffer'/name).read_bytes())!=actual:
                raise ValueError('Repair unexpectedly changes another helper: '+name)
        token=c.claim('server-explicit-positive-H-fixture-repair')
        if token is None:raise ValueError('Another controller owns the lease')
        checks=root/'diagnostic-repair-checks'/(code+'.log');checks.parent.mkdir(exist_ok=True)
        try:
            with checks.open('x') as log:
                subprocess.run([d['python'],'-B',str(ROOT/'tests/incremental/test_history_gradient_diagnostic.py'),'--require-torch'],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            backup=root/'scope-upgrade-backups'/('positive-H-fixture-'+code);backup.mkdir(exist_ok=False)
            shutil.copyfile(root/'bin/gradient_bridge.py',backup/'gradient_bridge.py')
            shutil.copyfile(root/'runner-code-sha.txt',backup/'runner-code-sha.txt')
            target=root/'bin/gradient_bridge.py';pending=target.with_suffix('.new')
            shutil.copyfile(ROOT/'scripts/lab_buffer/gradient_bridge.py',pending);os.replace(pending,target)
            (root/'runner-code-sha.txt').write_text(code+'\n')
            state,parent=c.read()
            if state['lease']['token']!=token:raise ValueError('Repair lease lost')
            request=task(d,code,job_id=CONTROLLED_REPAIR);raw=encoded(request)
            if r.read(r.ref('requests_branch'),'jobs/'+CONTROLLED_REPAIR+'.json') is not None:
                raise ValueError('Immutable repair job already exists')
            authorization=dict(failed_job_id=CONTROLLED,failed_code_sha=FAILED_CODE,
                failed_artifact_sha256=FAILED_ARTIFACT,diagnostic_steps_spent=0,remaining_step_budget=60,
                diagnosis='Tiny Plain fixture had H=0; deterministic measured positive-H token permutations fix only the test fixture.',
                mandatory_real_model_checks_passed=True,checks_log_sha256=digest(checks.read_bytes()),
                code_sha=code,attached_at=now(),installed_before=before,backup=str(backup))
            state['extension'].update(status='controlled_queued',controlled_repair_authorization=authorization,
                diagnostic_optimizer_steps_spent=0,installed_code_sha=code)
            blocker=state['extension'].pop('dispatch_blocker',None)
            if blocker:
                state['extension']['previous_dispatch_blocker']=dict(blocker,resolved_at=now(),resolved_by_code_sha=code)
            state['extension']['repair_delivery'].update(status='attached',checks_log_sha256=authorization['checks_log_sha256'])
            state.update(mode='queued',pause_requested=False,lease=None,active_job_id=CONTROLLED_REPAIR,
                code_sha=code,request_sha256=digest(raw))
            c.write(state,parent,{'jobs/'+CONTROLLED_REPAIR+'.json':raw})
            evidence=dict(status='attached',job_id=CONTROLLED_REPAIR,code_sha=code,
                authorization=authorization,main_workspace_written=False,cron_changed=False,
                beauty_optimizer_steps_during_attachment=0,existing_processes_terminated=False)
            r.publish({'workers/server2-gradient-repair-attach.json':encoded(evidence)})
            return evidence
        except BaseException as error:
            state,parent=c.read()
            if (state.get('lease') or {}).get('token')==token:
                state.update(mode='needs_diagnosis',pause_requested=True,lease=None)
                state['extension']['status']='repair_attachment_failed'
                c.write(state,parent)
            evidence=dict(status='repair_attachment_failed',code_sha=code,error=repr(error),
                checks_log_tail=checks.read_text(errors='replace')[-14000:] if checks.exists() else None,
                no_automatic_retry=True,beauty_optimizer_steps_during_attachment=0)
            atomic_json(root/'gradient-repair-attach-error.json',evidence)
            r.publish({'workers/server2-gradient-repair-attach.json':encoded(evidence)})
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    print(json.dumps(attach(parser.parse_args().root),indent=2))
