"""Bounded optimizer ablation, training-only gate and immutable v1 task contract."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil
import json
import os
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,encoded,safe_file
from genrec.cf_letter_history_long_protocol import indices_sha,save_anchors,load_anchors,tail_stability,paired_assessment,transfer_refresh

MANIFEST='docs/experiments/cf_letter_optimizer.json'
SCOPE='cf-letter-optimizer-v1'
JOB='cf-letter-optimizer-001'


def cases(p):
    return [dict(v,optimizer='adamw',objective='h_only',steps=p['steps_per_branch'],phase='screen') for v in p['screen_matrix']]


def pair_cases(selected,p):
    if selected is None:return []
    return [dict(name='pair_'+o,optimizer='adamw',objective=o,steps=p['steps_per_branch'],phase='pair',lr=selected['case']['lr'],beta1=selected['case']['beta1']) for o in ('rq_only','mixed')]


def plan(root):
    p=read(Path(root)/MANIFEST)
    expected=[('h_lr3e5_b09',3e-5,.9),('h_lr1e5_b09',1e-5,.9),('h_lr1e4_b0',1e-4,0.),('h_lr3e5_b0',3e-5,0.),('h_lr1e5_b0',1e-5,0.)]
    if (p['scope_id']!=SCOPE or p['job_id']!=JOB or p['optimizer_step_cap']!=7000 or p['steps_per_branch']!=1000 or p['parallel']!=2
            or [(v['name'],v['lr'],v['beta1']) for v in p['screen_matrix']]!=expected
            or p['allowed_query_splits']!=['base_train','update_train'] or p['fixed_history_weight']!=0.0006386606341359484
            or p['beta2']!=.999 or p['eps']!=1e-8 or p['weight_decay']!=0. or p['temperature']!=.5 or p['max_grad_norm']!=.1
            or p['content_batch_size']!=4096 or p['anchor_batch_size']!=128 or p['anchor_size']!=4096 or p['expected_training_items']!=11691
            or p['fixed_monitor_every_steps']!=10 or p['full_monitor_every_epochs']!=10
            or any(p[k] is not False for k in ('lambda_sweep_authorized','validation_evaluation','gpt_training','holdout_evaluation','future_evaluation','sasrec_training','retry'))):
        raise ValueError('Unsupported optimizer-study scope')
    return p


def screen_gate(result,initial,p):
    s=result['stability'];eligible=(result['optimizer_steps']==p['steps_per_branch'] and result['capacity_feasible'] and s['sufficient'] and s['passed'])
    if not s['sufficient']:return dict(eligible=False,reason='insufficient_scheduled_monitors')
    reduction=1-s['H']['mean']/initial['H'];rq=s['rq']['mean']/initial['rq']-1;mse=s['reconstruction_mse']['mean']/initial['mse']-1
    eligible=bool(eligible and reduction>=p['screening']['H_reduction_from_initial_min'] and rq<=p['screening']['rq_relative_increase_max'] and mse<=p['screening']['mse_relative_increase_max'])
    return dict(eligible=eligible,H_relative_reduction_from_initial=reduction,native_rq_relative_increase=rq,hard_mse_relative_increase=mse)


def selection(results,initial,p):
    matrix=[dict(name=v['case']['name'],**screen_gate(v,initial,p)) for v in results]
    choices=[(v['stability']['H']['mean'],v['stability']['rq']['mean'],i,v) for i,v in enumerate(results) if matrix[i]['eligible']]
    chosen=min(choices,key=lambda v:v[:3])[-1] if choices else None
    return dict(candidates=matrix,selected_name=chosen['case']['name'] if chosen else None,selected_case=chosen['case'] if chosen else None,
        pair_authorized=chosen is not None,rule=p['screening']['selection'])


def transfer_control(root,output,p):
    info=p['accepted_control'];source=Path(root)/'workspaces'/info['job_id']/'outputs/cf-history-long';target=Path(output)/'inputs/control';target.mkdir(parents=True)
    if source.resolve()!=source:raise ValueError('Control source alias')
    bound={}
    for name,value in info['files_sha256'].items():
        path=safe_file(source,name)
        if sha(path)!=value:raise ValueError('Accepted control changed')
        shutil.copyfile(path,target/name)
        if sha(target/name)!=value:raise ValueError('Copied control changed')
        bound[str(path)]=value
    report=read(target/'compact.json')
    if report['code_sha']!=info['code_sha'] or report['execution_status']!='completed' or report['optimizer_steps_spent']!=3000 or report['history_weight']!=p['fixed_history_weight']:raise ValueError('Wrong accepted control')
    branch=next(v for v in report['branches'] if v['case']['objective']=='h_only')
    branch['case']=dict(name=info['control_name'],optimizer='adamw',objective='h_only',steps=p['steps_per_branch'],phase='reused_control',lr=info['lr'],beta1=info['beta1'])
    return bound,branch,dict(H=report['initial_full_anchor_train']['H'],rq=report['initial_rq']['total_loss'],mse=report['initial_rq']['reconstruction_mse'])


def counters(output):
    values={};uncertain=[]
    for path in (Path(output)/'arms').glob('*/*/optimizer_steps.json'):
        v=read(path);name=path.parent.name;values[name]=max(v['optimizer_steps_completed'],v.get('optimizer_step_intent') or 0)
        if v.get('optimizer_step_intent') is not None:uncertain.append(str(path))
    return sum(values.values()),values,uncertain


def can_launch(active,observed,memory,p,stopped=False):
    return not stopped and len(active)<p['parallel'] and not (set(observed)-set(active)) and memory<=p['gpu_memory_limit_mib']


def curve_groups(p):
    names=[v['name'] for v in cases(p)]
    return dict(curves_screen_1=names[:2],curves_screen_2=names[2:4],curves_screen_3=names[4:],curves_pair=['pair_rq_only','pair_mixed'])


def publish_curves(output,p):
    output=Path(output)
    for name,arms in curve_groups(p).items():
        value={arm:read(output/(arm+'_curves.json')) for arm in arms if (output/(arm+'_curves.json')).exists()}
        raw=(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
        if len(raw)>65536:raise ValueError('Grouped curve exceeds immutable mailbox limit')
        path=output/(name+'.json');pending=path.with_suffix('.tmp');pending.write_bytes(raw);os.replace(pending,path)


def submission(code_sha,deployment_sha,root,repository):
    root=Path(root);p=plan(repository);digest=lambda v:hashlib.sha256(encoded(v)).hexdigest()
    if p['execution_authorized'] is not True:raise ValueError('No optimizer-study execution authorization')
    names=['compact','data_audit','selection',*curve_groups(p)]
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_optimizer.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-optimizer'],artifacts=['outputs/task-runtime.json',*[f'outputs/cf-optimizer/{n}.json' for n in names]])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=7000,max_parallel=2,authorization=p['authorization'],allowed_query_splits=p['allowed_query_splits'],
        predecessor_read_only=[p['predecessor'],p['refresh_predecessor'],p['accepted_control']],screen_steps=5000,conditional_pair_steps=2000,stop_after_study=True)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),
        manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),preflight=[['python','-u','-B','tests/incremental/test_cf_letter_base.py','--require-torch'],['python','-u','-B','tests/incremental/test_cf_letter_history_long.py','--require-torch'],['python','-u','-B','tests/incremental/test_cf_letter_optimizer.py','--require-torch']],verifier='verify_cf_letter_optimizer.py',optimizer_step_reservation=7000,input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)
