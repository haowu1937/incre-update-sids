"""Four explicitly requested fresh paired arms; old screening is not reclassified."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from genrec.cf_letter_protocol import read,save,sha,encoded,safe_file

JOB='cf-letter-optimizer-pair-001'
SCOPE='cf-letter-optimizer-pair-v1'
MANIFEST='docs/experiments/cf_letter_optimizer_pair.json'


def cases(p):
    return [dict(name=v['name']+'_'+o,lr=v['lr'],beta1=v['beta1'],optimizer='adamw',objective=o,
                 steps=p['steps_per_branch'],phase='explicit_pair',setting=v['name'])
            for v in p['pair_settings'] for o in ('rq_only','mixed')]


def plan(root):
    p=read(Path(root)/MANIFEST)
    if (p['job_id']!=JOB or p['scope_id']!=SCOPE or p['optimizer_step_cap']!=4000 or p['steps_per_branch']!=1000 or p['parallel']!=2
        or p['pair_settings']!=[dict(name='lr1e5_b09',lr=1e-5,beta1=.9),dict(name='lr1e5_b0',lr=1e-5,beta1=0.)]
        or p['fixed_history_weight']!=0.0006386606341359484 or p['allowed_query_splits']!=['base_train','update_train']
        or p['beta2']!=.999 or p['eps']!=1e-8 or p['weight_decay']!=0 or p['max_grad_norm']!=.1 or p['temperature']!=.5
        or p['content_batch_size']!=4096 or p['anchor_batch_size']!=128 or p['anchor_size']!=4096 or p['expected_training_items']!=11691
        or p['fixed_monitor_every_steps']!=10 or p['full_monitor_every_epochs']!=10
        or p['canonical_assignment_backend']!='cuda:0' or p['gpu_verification_authorized'] is not True
        or p['previous_screen_eligible'] is not False or p['previous_scientific_acceptance']!='blocked'
        or p['stability']!=dict(tail_monitors=8,rq_relative_drift_max=.02,rq_cv_max=.03,H_relative_drift_max=.02,H_relative_range_max=.1,
                               near_zero_H=1e-8,mixed_H_reduction_min=.05,mixed_rq_increase_max=.02,mixed_mse_increase_max=.02)
        or any(p[k] is not False for k in ('lambda_sweep_authorized','validation_evaluation','gpt_training','holdout_evaluation','future_evaluation','sasrec_training','retry'))):
        raise ValueError('Unsupported explicit paired scope')
    return p


def assess(results,p):
    rows={v['case']['objective']:v for v in results}
    if set(rows)!={'rq_only','mixed'} or any(not v['capacity_feasible'] or v['optimizer_steps']!=p['steps_per_branch'] for v in rows.values()):
        return dict(eligible=False,reason='incomplete_or_capacity_failure')
    rq=rows['rq_only']['stability'];mixed=rows['mixed']['stability'];s=p['stability']
    if not rq['sufficient'] or not mixed['sufficient']:return dict(eligible=False,reason='insufficient_fixed_full_monitors')
    reduction=1-mixed['H']['mean']/rq['H']['mean'] if rq['H']['mean']>s['near_zero_H'] else None
    q=mixed['rq']['mean']/rq['rq']['mean']-1;m=mixed['reconstruction_mse']['mean']/rq['reconstruction_mse']['mean']-1
    return dict(eligible=True,passed=bool(reduction is not None and reduction>=s['mixed_H_reduction_min'] and q<=s['mixed_rq_increase_max']
        and m<=s['mixed_mse_increase_max'] and mixed['passed']),tail_H_relative_reduction_vs_rq_only=reduction,
        tail_native_rq_relative_increase=q,tail_hard_mse_relative_increase=m,mixed_tail_stable=mixed['passed'])


def assessments(results,p):
    return {s['name']:assess([v for v in results if v['case']['setting']==s['name']],p) for s in p['pair_settings']}


def curve_groups(p):
    return {'curves_'+s['name']:[v['name'] for v in cases(p) if v['setting']==s['name']] for s in p['pair_settings']}


def publish_curves(output,p):
    output=Path(output)
    for name,arms in curve_groups(p).items():
        value={arm:read(output/(arm+'_curves.json')) for arm in arms if (output/(arm+'_curves.json')).exists()}
        raw=(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
        if len(raw)>65536:raise ValueError('Grouped paired curves exceed mailbox bound')
        pending=output/(name+'.tmp');pending.write_bytes(raw);os.replace(pending,output/(name+'.json'))


def transfer_audit(root,output,p):
    info=p['audit_predecessor'];source=Path(root)/'workspaces'/info['job_id']/'outputs/cf-optimizer-audit'
    if source.resolve()!=source:raise ValueError('Audit evidence alias')
    target=Path(output)/'inputs/assignment_audit';target.mkdir(parents=True);bound={}
    for name,value in info['files_sha256'].items():
        path=safe_file(source,name)
        if sha(path)!=value:raise ValueError('Assignment-audit evidence changed')
        shutil.copyfile(path,target/name)
        if sha(target/name)!=value:raise ValueError('Copied assignment audit differs')
        bound[str(path)]=value
    r=read(target/'compact.json')
    if (r['job_id']!=info['job_id'] or r['code_sha']!=info['code_sha'] or r['execution_status']!='completed'
        or r['optimizer_steps_spent']!=0 or not r['source_files_unchanged'] or r['monitors_checked']!=170
        or r['gpu_probed_monitors']!=4 or r['gpu_reproduction_mismatch_monitors']!=0 or r['original_acceptance']!='blocked'):
        raise ValueError('Assignment-audit prerequisite not satisfied')
    return bound


def submission(code_sha,deployment_sha,root,repository):
    root=Path(root);p=plan(repository);digest=lambda x:hashlib.sha256(encoded(x)).hexdigest()
    if p['execution_authorized'] is not True:raise ValueError('No paired execution authorization')
    names=['compact','data_audit','paired_assessment',*curve_groups(p)]
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,created_at=datetime.now(timezone.utc).isoformat(),
        timeout_seconds=86400,argv=['python','-u','-B','run_cf_letter_optimizer_pair.py','--deployment',str(root/'main-deployment.json'),
        '--output','outputs/cf-optimizer-pair'],artifacts=['outputs/task-runtime.json',*[f'outputs/cf-optimizer-pair/{n}.json' for n in names]])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=4000,max_parallel=2,authorization=p['authorization'],
        allowed_query_splits=p['allowed_query_splits'],predecessor_read_only=[p['predecessor'],p['refresh_predecessor'],p['audit_predecessor']],
        explicit_settings=p['pair_settings'],steps_per_arm=1000,old_screen_gate_passed=False,stop_after_study=True)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),
        scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),
        preflight=[['python','-u','-B','tests/incremental/test_cf_letter_base.py','--require-torch'],
                   ['python','-u','-B','tests/incremental/test_cf_letter_history_long.py','--require-torch'],
                   ['python','-u','-B','tests/incremental/test_cf_letter_optimizer.py','--require-torch'],
                   ['python','-u','-B','tests/incremental/test_cf_letter_optimizer_pair.py','--require-torch']],
        verifier='verify_cf_letter_optimizer_pair.py',optimizer_step_reservation=4000,input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)
