"""Upgrade the existing idle Runner for the single authorized joint study."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from runner import Runner,atomic_json,digest,now
from main_control import Control,encoded
from convergence_bridge import SCOPE,JOB,task

ROOT=Path(__file__).resolve().parents[2]
INSTALLED='3d901876e12ea344d34a59773ca3d03c812235fa'
HELPERS=('runner.py','main_control.py','main_service.py','gradient_bridge.py','gradient_logs.py','convergence_bridge.py')


def attach(root,expected_code):
    root=Path(root).resolve();r=Runner(root);c=Control(root)
    code=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if code!=expected_code:raise ValueError('Attachment checkout differs from the delivered full SHA')
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','HEAD','--'],check=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'PAUSED.json').exists():raise ValueError('Existing pause requires reconciliation; no upgrade')
        state,_=c.read()
        if state.get('extension',{}).get('scope_id')==SCOPE and state.get('code_sha')==code:
            return dict(status='already_attached',job_id=JOB,code_sha=code)
        pending=state.get('pending_scope',{})
        if (state['mode']!='plan_completed' or state['pause_requested'] or state.get('lease')
                or pending.get('scope_id')!=SCOPE or pending.get('code_sha')!=code
                or pending.get('max_optimizer_steps')!=30400
                or state.get('extension',{}).get('status')!='completed'):
            raise ValueError('Require completed prior plan and exact pending study authorization')
        if digest((ROOT/'docs/lab_buffer/joint_convergence_plan.json').read_bytes())!=pending.get('manifest_sha256'):
            raise ValueError('Pinned study manifest differs from pending authorization')
        for path in (root/'state').glob('*.json'):
            value=json.loads(path.read_text())
            if value['result']['status'] in ('starting','running') or value.get('pending_upload'):
                raise ValueError('Unfinished local task or delivery; no upgrade')
        if (root/'runner-code-sha.txt').read_text().strip()!=INSTALLED:
            raise ValueError('Installed Runner version changed; preserve it')
        before={}
        for name in HELPERS[:-1]:
            expected=digest(subprocess.check_output(['git','-C',str(ROOT),'show',INSTALLED+':scripts/lab_buffer/'+name]))
            actual=digest((root/'bin'/name).read_bytes());before[name]=actual
            if actual!=expected:raise ValueError('Installed helper differs from known version: '+name)
        d=json.loads((root/'main-deployment.json').read_text())
        if Path(d['runner_root']).resolve()!=root:raise ValueError('Deployment root mismatch')
        future=Path(d['future_dir']).resolve()
        protected={p:v for p,v in d['input_sha256'].items() if Path(p).resolve()!=future and future not in Path(p).resolve().parents}
        new=dict(schema_version=1,scope_id=SCOPE,runner_root=str(root),code_sha=code,
            e1_dir=d['e1_dir'],retuned_dir=d['retuned_dir'],protected_input_sha256=protected,
            original_deployment_sha256=digest((root/'main-deployment.json').read_bytes()),
            max_optimizer_steps=30400,output_policy='independent exact-job worktree only')
        for path,value in protected.items():
            h=__import__('hashlib').sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=value:raise ValueError('Bound original input changed: '+path)
        token=c.claim('server-explicit-joint-convergence-upgrade')
        if token is None:raise ValueError('Controller lease busy')
        checks=root/'joint-convergence-checks'/(code+'.log');checks.parent.mkdir(exist_ok=True)
        backup=root/'scope-upgrade-backups'/('joint-convergence-'+code)
        changed=False
        try:
            print(json.dumps(dict(status='real_model_preflight_running',checks_log=str(checks),beauty_optimizer_steps=0)),flush=True)
            with checks.open('x') as log:
                subprocess.run([r.config['python'],'-B',str(ROOT/'tests/incremental/test_joint_convergence.py'),'--require-torch'],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            if (root/'PAUSED.json').exists():
                raise ValueError('A new pause appeared during checks; preserve old helpers')
            backup.mkdir(parents=True,exist_ok=False)
            for name in HELPERS[:-1]:shutil.copy2(root/'bin'/name,backup/name)
            for name in ('config.json','runner-code-sha.txt'):shutil.copy2(root/name,backup/name)
            changed=True
            for name in HELPERS:
                target=root/'bin'/name;temporary=target.with_suffix('.new')
                shutil.copyfile(ROOT/'scripts/lab_buffer'/name,temporary);os.replace(temporary,target)
            atomic_json(root/'convergence-deployment.json',new)
            config=copy.deepcopy(r.config);config['main_scope_revision']=3;atomic_json(root/'config.json',config)
            (root/'runner-code-sha.txt').write_text(code+'\n')
            state,parent=c.read()
            if (state.get('lease') or {}).get('token')!=token:raise ValueError('Lost study attachment lease')
            if r.read(r.ref('requests_branch'),'jobs/'+JOB+'.json') is not None:raise ValueError('Immutable study job already exists')
            state.setdefault('completed_scope_history',[]).append(copy.deepcopy(state['extension']))
            state['completed_scope_history'][-1]['controller_plan']=copy.deepcopy(state.get('plan',[]))
            state['plan']=['calibrate_concurrency','train_only_parameter_screen','faithful_candidate_extensions',
                'conditional_second_seed_confirmation','verify_report_and_stop']
            state['extension']=dict(scope_id=SCOPE,status='queued',max_optimizer_steps=30400,optimizer_steps_spent=0,
                real_model_preflight_passed=True,checks_log_sha256=digest(checks.read_bytes()),
                code_sha=code,source_manifest=pending['manifest_sha256'],attached_at=now(),backup=str(backup))
            request=task(root,code);raw=encoded(request)
            state.update(mode='queued',pause_requested=False,lease=None,active_job_id=JOB,code_sha=code,request_sha256=digest(raw))
            state.pop('pending_scope',None)
            c.write(state,parent,{'jobs/'+JOB+'.json':raw})
            evidence=dict(status='attached',scope_id=SCOPE,job_id=JOB,code_sha=code,
                max_optimizer_steps=30400,max_parallel=2,real_model_checks_passed=True,
                checks_log_sha256=digest(checks.read_bytes()),installed_before=before,backup=str(backup),
                cron_changed=False,main_workspace_written=False,original_inputs_modified=False,
                beauty_optimizer_steps_during_attachment=0,processes_terminated=False)
            r.publish({'workers/server2-joint-convergence-attach.json':encoded(evidence)})
            return evidence
        except BaseException as error:
            if changed:
                # Keep the known new exact gate: no job is permitted without its
                # matching queued control record. Preserve backup; never blind rollback.
                r.pause('Joint attachment incomplete; reconcile backup/control before retry: '+repr(error))
            state,parent=c.read()
            if (state.get('lease') or {}).get('token')==token:
                state.update(lease=None,pause_requested=True,mode='needs_diagnosis')
                state['joint_attachment_failure']=dict(error=repr(error),code_sha=code,changed_helpers=changed)
                c.write(state,parent)
            evidence=dict(status='attachment_failed',code_sha=code,error=repr(error),changed_helpers=changed,
                checks_log_tail=checks.read_text(errors='replace')[-12000:] if checks.exists() else None,
                beauty_optimizer_steps=0,no_automatic_retry=True)
            atomic_json(root/'joint-convergence-attach-error.json',evidence)
            try:r.publish({'workers/server2-joint-convergence-attach.json':encoded(evidence)})
            except Exception:pass
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--expected-code-sha',required=True)
    a=p.parse_args();print(json.dumps(attach(a.root,a.expected_code_sha),indent=2))
