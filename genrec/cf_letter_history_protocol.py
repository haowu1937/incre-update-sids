"""Training-only direction diagnostic: immutable inputs, branches and step scope."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil

from genrec.cf_letter_protocol import encoded,read,safe_file,sha

MANIFEST='docs/experiments/cf_letter_history_direction.json'
SCOPE='cf-letter-history-direction-v1'
JOB='cf-letter-history-direction-001'


def cases(p):
    return [dict(name=o['name']+'_'+str(lr).replace('.','p').replace('-','m')+'_'+objective,
        optimizer=o['name'],lr=lr,objective=objective,steps=p['steps_per_branch'])
        for o in p['optimizers'] for lr in o['lrs'] for objective in p['objectives']]


def plan(root):
    p=read(Path(root)/MANIFEST)
    if (p['scope_id']!=SCOPE or p['job_id']!=JOB or p['optimizer_step_cap']!=60 or p['steps_per_branch']!=5
            or len(cases(p))!=12 or p['allowed_query_splits']!=['base_train','update_train']
            or p['temperature']!=.5 or p['max_grad_norm']!=.1 or p['fixed_anchor_queries']!=128
            or p['full_anchor_steps']!=[0,1,5] or p['gradient_probe_batches']!=4
            or any(p[k] is not False for k in ('full_training_authorized','lambda_sweep_authorized','validation_evaluation','gpt_training','holdout_evaluation','future_evaluation','sasrec_training','retry'))):
        raise ValueError('Unsupported bounded history-direction scope')
    if sum(c['steps'] for c in cases(p))!=60:raise ValueError('Diagnostic branch budget differs')
    return p


def submission(code_sha,deployment_sha,root,repository):
    root=Path(root);p=plan(repository);digest=lambda v:hashlib.sha256(encoded(v)).hexdigest()
    if p.get('execution_authorized') is not True:raise ValueError('Direction proposal has no explicit execution authorization')
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,
        created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_history_direction.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-direction'],
        artifacts=['outputs/task-runtime.json',*[f'outputs/cf-direction/{n}.json' for n in ('compact','data_audit','gradient_calibration','sgd_curves','adamw_curves')]])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=60,max_parallel=1,
        authorization='Hao: continue toward matched Plain/Historical on fixed E1; first isolate the new direct-codebook historical gradient via bounded H-only/RQ-only/Mixed direction checks before a lambda sweep or full experiments.',
        allowed_query_splits=p['allowed_query_splits'],predecessor_read_only=[p['predecessor'],p['refresh_predecessor']],stop_after_direction_diagnostic=True)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),
        manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),
        preflight=[['python','-u','-B','tests/incremental/test_cf_letter_base.py','--require-torch'],['python','-u','-B','tests/incremental/test_cf_letter_history_direction.py','--require-torch']],
        verifier='verify_cf_letter_history_direction.py',optimizer_step_reservation=60,input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)


def transfer_refresh(root,output,p):
    root=Path(root).resolve();source=root/'workspaces'/p['refresh_predecessor']['job_id']/'outputs/cf-refresh'
    if source!=source.resolve():raise ValueError('Refreshed dependency has symlink aliases')
    target=Path(output)/'inputs/refresh';target.mkdir(parents=True);bound={}
    for name,value in {**p['refresh_predecessor']['files_sha256'],'compact.json':p['refresh_predecessor']['compact_sha256']}.items():
        path=safe_file(source,name)
        if sha(path)!=value:raise ValueError('Accepted refresh dependency changed: '+name)
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        if sha(dest)!=value:raise ValueError('Copied refresh dependency differs')
        bound[str(path)]=value
    report=read(target/'compact.json')
    if report['research_outcome']!='feature_refresh_measured' or report['code_sha']!=p['refresh_predecessor']['code_sha']:raise ValueError('Wrong accepted CF refresh')
    return bound
