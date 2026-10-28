"""Three matched, training-only 1000-step codebook arms; generic Runner contract."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil
import numpy as np
from genrec.cf_letter_protocol import encoded,read,safe_file,sha

MANIFEST='docs/experiments/cf_letter_history_long.json'
SCOPE='cf-letter-history-long-v1'
JOB='cf-letter-history-long-001'


def cases(p):
    return [dict(name=objective,optimizer='adamw',lr=p['lr'],objective=objective,steps=p['steps_per_branch']) for objective in p['objectives']]


def plan(root):
    p=read(Path(root)/MANIFEST)
    if (p['scope_id']!=SCOPE or p['job_id']!=JOB or p['optimizer_step_cap']!=3000 or p['steps_per_branch']!=1000
            or p['objectives']!=['h_only','rq_only','mixed'] or p['allowed_query_splits']!=['base_train','update_train']
            or p['temperature']!=.5 or p['max_grad_norm']!=.1 or p['fixed_anchor_queries']!=128
            or p['full_anchor_steps']!=[0,1000] or p['gradient_probe_batches']!=4
            or p['fixed_monitor_every_steps']!=10 or p['full_monitor_every_epochs']!=10
            or p['anchor_batch_size']!=128 or p['content_batch_size']!=4096 or p['lr']!=1e-4 or p['weight_decay']!=0.
            or p['expected_training_items']!=11691 or p['anchor_size']!=4096 or p['old_catalog_items']!=10924
            or p['optimizers']!=[dict(name='adamw',lrs=[1e-4],betas=[.9,.999],eps=1e-8,weight_decay=0.)]
            or any(p[k] is not False for k in ('full_training_authorized','lambda_sweep_authorized','validation_evaluation','gpt_training','holdout_evaluation','future_evaluation','sasrec_training','retry'))):
        raise ValueError('Unsupported bounded history-long scope')
    return p


def submission(code_sha,deployment_sha,root,repository):
    root=Path(root);p=plan(repository);digest=lambda v:hashlib.sha256(encoded(v)).hexdigest()
    if p.get('execution_authorized') is not True:raise ValueError('Long diagnostic has no explicit execution authorization')
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,
        created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_history_long.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-history-long'],
        artifacts=['outputs/task-runtime.json',*[f'outputs/cf-history-long/{n}.json' for n in ('compact','data_audit','gradient_calibration','h_only_curves','rq_only_curves','mixed_curves')]])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=3000,max_parallel=1,authorization=p['authorization'],
        allowed_query_splits=p['allowed_query_splits'],predecessor_read_only=[p['predecessor'],p['refresh_predecessor']],stop_after_diagnostic=True)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),
        manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),
        preflight=[['python','-u','-B','tests/incremental/test_cf_letter_base.py','--require-torch'],['python','-u','-B','tests/incremental/test_cf_letter_history_long.py','--require-torch']],
        verifier='verify_cf_letter_history_long.py',optimizer_step_reservation=3000,input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)


def indices_sha(indices):
    return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


def tail_stability(monitors,settings):
    rows=[v for v in monitors if v['scheduled_epoch_monitor']][-settings['tail_monitors']:]
    result=dict(monitors=len(rows),steps=[v['step'] for v in rows],sufficient=len(rows)==settings['tail_monitors'])
    if not result['sufficient']:return dict(result,passed=False)
    def series(values):
        x=np.asarray(values,dtype=float);mean=float(x.mean());scale=max(abs(mean),1e-12)
        drift=abs(float(np.polyfit(np.arange(len(x)),x,1)[0]))*(len(x)-1)/scale
        return dict(mean=mean,relative_drift=drift,cv=float(x.std()/scale),relative_range=float(np.ptp(x)/scale),absolute_range=float(np.ptp(x)))
    rq=series([v['rq']['total_loss'] for v in rows]);h=series([v['anchor_train']['H'] for v in rows]);mse=series([v['rq']['reconstruction_mse'] for v in rows])
    rq_ok=rq['relative_drift']<=settings['rq_relative_drift_max'] and rq['cv']<=settings['rq_cv_max']
    h_ok=(h['absolute_range']<=settings['near_zero_H'] if h['mean']<=settings['near_zero_H'] else
        h['relative_drift']<=settings['H_relative_drift_max'] and h['relative_range']<=settings['H_relative_range_max'])
    return dict(result,rq=rq,H=h,reconstruction_mse=mse,rq_passed=rq_ok,H_passed=h_ok,passed=rq_ok and h_ok)


def paired_assessment(branches,settings):
    arms={v['case']['objective']:v for v in branches}
    if set(arms)!=set(('h_only','rq_only','mixed')) or any(not b['capacity_feasible'] or b['optimizer_steps']!=b['case']['steps'] for b in arms.values()):return dict(eligible=False,reason='incomplete_or_capacity_failure')
    rq=arms['rq_only']['stability'];mixed=arms['mixed']['stability']
    if not rq['sufficient'] or not mixed['sufficient']:return dict(eligible=False,reason='insufficient_fixed_full_monitors')
    reduction=1-mixed['H']['mean']/rq['H']['mean'] if rq['H']['mean']>settings['near_zero_H'] else None
    rq_increase=mixed['rq']['mean']/rq['rq']['mean']-1
    mse_increase=mixed['reconstruction_mse']['mean']/rq['reconstruction_mse']['mean']-1
    passed=(reduction is not None and reduction>=settings['mixed_H_reduction_min'] and rq_increase<=settings['mixed_rq_increase_max'] and mse_increase<=settings['mixed_mse_increase_max'] and mixed['passed'])
    return dict(eligible=True,passed=passed,tail_H_relative_reduction_vs_rq_only=reduction,tail_native_rq_relative_increase=rq_increase,tail_hard_mse_relative_increase=mse_increase,mixed_tail_stable=mixed['passed'])


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
