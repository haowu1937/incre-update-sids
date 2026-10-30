"""Archive only the diagnosed multiseed pause; never install code or launch a task."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

JOB='cf-letter-multiseed-resume-001'
SOURCE='cf-letter-multiseed-001'
REASON='ValueError: Task failed/interrupted; preserve evidence and diagnose before any new job'


def reconcile(root,code):
    root=Path(root).resolve();sys.path.insert(0,str(root/'bin'))
    from runner import Runner,FINAL,atomic_json,now
    if not re.fullmatch('[0-9a-f]{40}',code):raise ValueError('Require full delivered SHA')
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=Runner(root);r.env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
        for key in ('GIT_TRACE','GIT_TRACE_CURL','GIT_CURL_VERBOSE'):r.env.pop(key,None)
        states=[]
        for f in (root/'state').glob('*.json'):
            s=json.loads(f.read_text());v=s['result'];states.append(v)
            if v['status'] not in FINAL or s.get('pending_upload'):raise ValueError('Local task active, uncertain or undelivered; preserve pause')
            if v['job_id']==JOB:return dict(status='already_recorded',job_id=JOB,task_status=v['status'],changed=False)
        paused=root/'PAUSED.json';pause_raw=paused.read_bytes() if paused.exists() else None
        if pause_raw is not None:
            record=json.loads(pause_raw)
            if record['reason']!=REASON:raise ValueError('Different pause reason; no archival')
            latest=max(states,key=lambda x:x.get('finished_at',''))
            if latest['job_id']!=SOURCE or record['time']<latest['finished_at']:raise ValueError('Pause does not follow the exact failed source')
            err=json.loads((root/'main-service-error.json').read_text())
            if err.get('error')!=REASON or not err.get('new_jobs_paused'):raise ValueError('Local failure evidence differs')
        r.fetch()
        if r.git('merge-base','--is-ancestor',code,r.ref('code_branch'),check=False).returncode:raise ValueError('Delivered code is not on configured code branch')
        proof=json.loads(r.read(code,'docs/experiments/cf_letter_multiseed_failure.json'))
        old=json.loads((root/'state'/(SOURCE+'.json')).read_text())['result']
        encode=lambda x:(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        if hashlib.sha256(encode(old)).hexdigest()!=proof['result_sha256']:raise ValueError('Original terminal task changed')
        for path,digest in proof['artifacts_sha256'].items():
            local=root/'workspaces'/SOURCE/path.removeprefix('artifacts/'+SOURCE+'/')
            if local.is_symlink() or hashlib.sha256(local.read_bytes()).hexdigest()!=digest:raise ValueError('Original returned evidence changed')
        state=json.loads(r.read(r.ref('requests_branch'),'control/main.json'))
        request_raw=r.read(r.ref('requests_branch'),'jobs/'+JOB+'.json');request=json.loads(request_raw)
        if (state['lease'] or state['pause_requested'] or state['mode']!='queued' or state['active_job_id']!=JOB
            or state['code_sha']!=code or request['code_sha']!=code or state['request_sha256']!=hashlib.sha256(request_raw).hexdigest()):
            raise ValueError('Require exact queued continuation with no live lease')
        # Installed v1 validates the manifest, reservation and input read roots.
        from task_runtime import load_contract
        contract,data,scope=load_contract(r,state,request,request_raw)
        if contract['optimizer_step_reservation']!=6000 or scope['family_actual_step_cap']!=40000:raise ValueError('Continuation budget changed')
        if state['receipts'][SOURCE]['status']!='needs_attention':raise ValueError('Original failed receipt must remain')
        for key in ('code_branch','requests_branch','results_branch'):
            branch=r.config[key]
            # Same-head dry-run checks write authentication without publishing.
            r.git('push','--dry-run','origin',r.ref(key)+':refs/heads/'+branch)
        archive=None
        if pause_raw is not None:
            if not paused.exists() or paused.read_bytes()!=pause_raw:raise ValueError('Pause changed during reconciliation')
            archive=root/'resolved-pauses'/('multiseed-resume-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            archive.mkdir(parents=True,exist_ok=False);paused.rename(archive/'PAUSED.json')
            # Only the independently verified matching error is archived.
            (root/'main-service-error.json').rename(archive/'main-service-error.json')
        evidence=dict(status='ready_for_existing_cron',job_id=JOB,code_sha=code,archive=str(archive) if archive else None,
            source_steps_preserved=34000,new_step_cap=6000,family_step_cap=40000,runner_upgraded=False,cron_changed=False,
            main_workspace_written=False,processes_terminated=False,task_launched=False,time=now())
        atomic_json(root/'multiseed-resume-reconciliation.json',evidence)
        return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--expected-code-sha',required=True)
    a=p.parse_args();print(json.dumps(reconcile(a.root,a.expected_code_sha),indent=2))
