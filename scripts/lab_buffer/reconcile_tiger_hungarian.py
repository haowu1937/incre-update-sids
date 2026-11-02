"""Archive only the exact diagnosed LR-export pause; no installation or launch."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
JOB='tiger-hungarian-001'
SOURCE='cf-letter-gpt-lr-001'
REASON='ValueError: Task failed/interrupted; preserve evidence and diagnose before any new job'


def reconcile(root,code):
    root=Path(root).resolve();sys.path.insert(0,str(root/'bin'))
    from runner import Runner,FINAL,atomic_json,now
    if not re.fullmatch('[0-9a-f]{40}',code):raise ValueError('Require complete delivered SHA')
    if (ROOT/'docs/experiments/tiger_hungarian.json').is_symlink():raise ValueError('Unsafe manifest')
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=Runner(root);r.env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
        for name in ('GIT_TRACE','GIT_TRACE_CURL','GIT_CURL_VERBOSE'):r.env.pop(name,None)
        states=[]
        for path in (root/'state').glob('*.json'):
            s=json.loads(path.read_text());v=s['result'];states.append(v)
            if v['status'] not in FINAL or s.get('pending_upload'):raise ValueError('Task active/uncertain/undelivered; preserve pause')
            if v['job_id']==JOB:return dict(status='already_recorded',job_id=JOB,task_status=v['status'],changed=False)
        paused=root/'PAUSED.json';pause_raw=paused.read_bytes() if paused.exists() else None
        if pause_raw:
            record=json.loads(pause_raw);latest=max(states,key=lambda v:v.get('finished_at',''))
            if record['reason']!=REASON or latest['job_id']!=SOURCE or record['time']<latest['finished_at']:raise ValueError('Pause is not the exact old LR-export failure')
            error=json.loads((root/'main-service-error.json').read_text())
            if error.get('error')!=REASON or not error.get('new_jobs_paused'):raise ValueError('Different service error; preserve evidence')
        r.fetch()
        if r.git('merge-base','--is-ancestor',code,r.ref('code_branch'),check=False).returncode:raise ValueError('Delivery not on configured code branch')
        raw_manifest=r.read(code,'docs/experiments/tiger_hungarian.json');p=json.loads(raw_manifest)
        if raw_manifest!=(ROOT/'docs/experiments/tiger_hungarian.json').read_bytes():raise ValueError('Local/delivered manifest differs')
        old=json.loads((root/'state'/(SOURCE+'.json')).read_text())['result'];encode=lambda v:(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        if hashlib.sha256(encode(old)).hexdigest()!=p['previous_failure']['result_sha256'] or old['code_sha']!=p['previous_failure']['code_sha'] or old['status']!='failed':raise ValueError('Original failed terminal evidence differs')
        compact=root/'workspaces'/SOURCE/'outputs/cf-gpt-lr/compact.json'
        if hashlib.sha256(compact.read_bytes()).hexdigest()!=p['previous_failure']['compact_sha256'] or json.loads(compact.read_text())['optimizer_steps_spent']!=10000:raise ValueError('Original10000-step evidence differs')
        state=json.loads(r.read(r.ref('requests_branch'),'control/main.json'));request_raw=r.read(r.ref('requests_branch'),'jobs/'+JOB+'.json');request=json.loads(request_raw)
        if state['lease'] or state['pause_requested'] or state['mode']!='queued' or state['active_job_id']!=JOB or state['code_sha']!=code or request['code_sha']!=code or state['request_sha256']!=hashlib.sha256(request_raw).hexdigest():raise ValueError('Require exact new queued TIGER scope')
        from task_runtime import load_contract
        contract,_,scope=load_contract(r,state,request,request_raw)
        if contract['optimizer_step_reservation']!=6000 or scope['max_optimizer_steps']!=6000 or scope['future_passes_new']!=3 or state['receipts'][SOURCE]['status']!='needs_attention':raise ValueError('New scope or preserved failure differs')
        for key in ('code_branch','requests_branch','results_branch'):r.git('push','--dry-run','origin',r.ref(key)+':refs/heads/'+r.config[key])
        archive=None
        if pause_raw:
            if paused.read_bytes()!=pause_raw:raise ValueError('Pause changed during reconciliation')
            archive=root/'resolved-pauses'/('tiger-hungarian-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));archive.mkdir(parents=True,exist_ok=False)
            paused.rename(archive/'PAUSED.json');(root/'main-service-error.json').rename(archive/'main-service-error.json')
        result=dict(status='ready_for_existing_cron',job_id=JOB,code_sha=code,archive=str(archive) if archive else None,
            old_LR_steps_preserved=10000,new_TIGER_step_cap=6000,runner_upgraded=False,cron_changed=False,main_workspace_written=False,processes_terminated=False,task_launched=False,time=now())
        atomic_json(root/'tiger-hungarian-reconciliation.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--expected-code-sha',required=True);args=parser.parse_args()
    print(json.dumps(reconcile(args.root,args.expected_code_sha),indent=2))
