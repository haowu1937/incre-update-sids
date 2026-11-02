"""One-factor GPT LR ablation on Update validation; no new Future evaluation."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from genrec.cf_letter_protocol import read,save,sha,encoded,safe_file
from genrec.cf_letter_downstream_protocol import select
from genrec.cf_letter_multiseed_protocol import interval

JOB='cf-letter-gpt-lr-001'
SCOPE='cf-letter-gpt-lr-v1'
MANIFEST='docs/experiments/cf_letter_gpt_lr.json'
ARTIFACTS=['compact','data_audit','baseline','runs','curves','comparisons']
METHODS=['plain','historical']
SOURCE='cf-letter-multiseed-resume-001'
SOURCE_CODE='34f1fecb8c62a3481e553ddede22f7686a084843'
OLD_CODE='9f4bebccfa8cc8617b7e4c29ee6e1f4f8dc6e451'


def plan(root):
    root=Path(root);p=read(root/MANIFEST)
    if (p['job_id']!=JOB or p['scope_id']!=SCOPE or p['seeds']!=list(range(2027,2032)) or p['methods']!=METHODS
        or p['new_lr']!=3e-5 or p['baseline_lr']!=1e-4 or p['steps_per_arm']!=1000 or p['optimizer_step_cap']!=10000
        or p['selection_steps']!=[0,100,250,500,1000] or not p['execution_authorized'] or p['future_passes']!=0
        or p['cohort_seed']!=2026 or p['parallel']!=2 or p['retry'] or p['quantizer_training'] or p['automatic_lr_selection'] or p['baseline_original_steps']!=2000 or p['topk']!=[5,10,20]
        or p['source_job_id']!=SOURCE or p['source_code_sha']!=SOURCE_CODE or sha(root/p['template']['path'])!=p['template']['sha256']):
        raise ValueError('Unsupported fixed GPT learning-rate scope')
    return p


def cases(p):
    return [dict(name=f's{s}_{m}',seed=s,stage='gpt',source=f's{s}_{m}',method=m,steps=p['steps_per_arm'],phase='gpt_update') for s in p['seeds'] for m in METHODS]


def settings(root,p,seed=None,baseline=False):
    q=read(Path(root)/p['template']['path']);q.pop('zero_step_repair_of',None)
    q.update(job_id=JOB,scope_id=SCOPE,seed=p['cohort_seed'] if seed is None else seed,lr=p['baseline_lr'] if baseline else p['new_lr'],
        steps_per_arm=p['steps_per_arm'],optimizer_step_cap=p['optimizer_step_cap'],selection_steps=p['selection_steps'],parallel=p['parallel'],
        wall_seconds=p['wall_seconds'],gpu_memory_limit_mib=p['gpu_memory_limit_mib'],
        arms=[{k:v for k,v in c.items() if k not in ('steps','phase')} for c in cases(p)])
    return q


def source_files(p):
    files={'gpt/inputs/base/gpt/selected.pth':'inputs/base/gpt/selected.pth','gpt/inputs/base/item_sids.npy':'inputs/base/item_sids.npy'}
    for c in cases(p):
        n=c['name'];files[f'gpt/mappings/{n}.npy']=f'mappings/{n}.npy'
        names=['summary.json','metrics.jsonl','initial.pth','selected.pth','full_update_validation.jsonl']
        names += [f'{kind}_{step}.{suffix}' for step in p['selection_steps'] for kind,suffix in [('candidate','pth'),('probe','jsonl')]]
        for name in names:files[f'gpt/arms/{n}/{name}']=f'baseline/arms/{n}/{name}'
    return files


def check_source(root,p):
    root=Path(root);source=root/'workspaces'/SOURCE/'outputs/cf-multiseed-resume'
    if source.resolve()!=source or source.is_symlink():raise ValueError('Source alias')
    record=read(root/'state'/(SOURCE+'.json'))
    if record['pending_upload'] or hashlib.sha256(encoded(record['result'])).hexdigest()!=p['source_result_sha256']:raise ValueError('Accepted source terminal result differs')
    if record['result']['status']!='completed' or record['result']['actual_code_sha']!=SOURCE_CODE:raise ValueError('Source not completed')
    branch=read(root/'config.json')['requests_branch']
    receipt_raw=subprocess.check_output(['git','--git-dir='+str(root/'transport.git'),'show','refs/remotes/origin/'+branch+':receipts/'+SOURCE+'.json'],timeout=10)
    receipt=json.loads(receipt_raw)
    if hashlib.sha256(receipt_raw).hexdigest()!=p['source_receipt_sha256'] or receipt['status']!='verified_and_reported' or receipt['actual_optimizer_steps']!=6000 or receipt['code_sha']!=SOURCE_CODE:
        raise ValueError('Source scientific acceptance differs')
    compact=safe_file(source,'compact.json');index=safe_file(source,'local_artifact_index.json');runtime=safe_file(source.parent,'task-runtime.json')
    if sha(compact)!=p['source_compact_sha256'] or sha(index)!=p['source_index_sha256'] or sha(runtime)!=p['source_runtime_sha256']:raise ValueError('Accepted source binding differs')
    r=read(compact);rt=read(runtime)
    if r['execution_status']!='completed' or r['optimizer_steps_spent']!=40000 or r['uncertain_step_records'] or not r['original_inputs_unchanged'] or rt['checks_completed']!=5 or rt['status']!='completed':raise ValueError('Accepted source scope differs')
    hashes=read(index);bound={str(compact):sha(compact),str(index):sha(index),str(runtime):sha(runtime)}
    # Exact whitelist: no old Future predictions, raw Future data or holdout reads.
    for name in source_files(p):
        f=safe_file(source,name)
        if name not in hashes or sha(f)!=hashes[name]:raise ValueError('Required source file changed: '+name)
        bound[str(f)]=hashes[name]
    return source,hashes,bound


def transfer(root,output,p):
    source,index,bound=check_source(root,p);output=Path(output)
    size=sum((source/n).stat().st_size for n in source_files(p))
    if shutil.disk_usage(output).free<size+12*1024**3:raise RuntimeError('Need private source copy plus12GiB training space')
    for name,destination in source_files(p).items():
        f=safe_file(source,name);target=output/destination;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('Fresh private source required')
        shutil.copyfile(f,target)
        if sha(target)!=index[name] or (f.stat().st_dev,f.stat().st_ino)==(target.stat().st_dev,target.stat().st_ino):raise ValueError('Private copy differs or aliases source')
    save(output/'source_bindings.json',{target:index[name] for name,target in source_files(p).items()})
    return bound


def baseline_result(output,c,p,q):
    d=Path(output)/'baseline/arms'/c['name'];s=read(d/'summary.json');curve=[v for v in s['curve'] if v['step']<=p['steps_per_arm']]
    chosen=select(curve,q)
    if s['optimizer_steps']!=p['baseline_original_steps'] or chosen['step']!=s['selected_update_step'] or sha(d/'selected.pth')!=s['selected_sha256'] or chosen['checkpoint_sha256']!=s['selected_sha256']:
        raise ValueError('Baseline prefix would require new validation inference; no silent substitution')
    return dict(case=c,lr=p['baseline_lr'],original_optimizer_steps=p['baseline_original_steps'],reused_prefix_steps=p['steps_per_arm'],selected_update_step=chosen['step'],
        selected_sha256=s['selected_sha256'],mapping_sha256=s['mapping_sha256'],selected_probe=chosen['metrics'],full_update_validation=s['full_update_validation'])


def comparisons(baseline,training,p):
    names=[c['name'] for c in cases(p)]
    if list(baseline)!=names or list(training)!=names:raise ValueError('All paired seeds required')
    r=dict(stage='Update_validation_only',seeds=p['seeds'],future_evaluated=False,automatic_lr_selection=False,groups={},paired={})
    metrics=[f'{m}@{k}' for m in ('recall','ndcg') for k in p['topk']]
    # Probe trajectories remain complete in curves.json. Seed intervals use the
    # full Update-validation report, fitting within the existing64KiB mailbox.
    for cohort in ('full_update_validation',):
        r['groups'][cohort]={};r['paired'][cohort]={}
        for den in ('full_denominator','conditional_on_servable'):
            groups={}
            for label,data in [('lr1e4',baseline),('lr3e5',training)]:
                for method in METHODS:groups[label+'_'+method]={k:interval(data[f's{s}_{method}'][cohort][den][k] for s in p['seeds']) for k in metrics}
            pairs={}
            for label,data in [('lr1e4',baseline),('lr3e5',training)]:
                pairs[label+'_historical_minus_plain']={k:interval(data[f's{s}_historical'][cohort][den][k]-data[f's{s}_plain'][cohort][den][k] for s in p['seeds']) for k in metrics}
            for method in METHODS:pairs['lr3e5_minus_lr1e4_'+method]={k:interval(training[f's{s}_{method}'][cohort][den][k]-baseline[f's{s}_{method}'][cohort][den][k] for s in p['seeds']) for k in metrics}
            r['groups'][cohort][den]=groups;r['paired'][cohort][den]=pairs
    return r


def curves_data(output,p):
    output=Path(output)
    curves={}
    for label,source in [('lr1e4',output/'baseline/arms'),('lr3e5',output/'arms')]:
        for c in cases(p):
            d=source/c['name']
            if not (d/'summary.json').exists():continue
            curve=[v for v in read(d/'summary.json')['curve'] if v['step']<=p['steps_per_arm']]
            loss=[json.loads(line) for line in (d/'metrics.jsonl').read_text().splitlines()][:p['steps_per_arm']]
            curves[label+'/'+c['name']]=dict(probe=[dict(step=v['step'],full_denominator=v['metrics']['full_denominator'],mean_sequence_nll_servable=v['metrics']['mean_sequence_nll_servable']) for v in curve],
                training_loss_per100=[dict(first_step=i+1,last_step=min(i+100,len(loss)),mean_token_ce=statistics.mean(v['token_ce'] for v in loss[i:i+100])) for i in range(0,len(loss),100)])
    return curves


def publications(output,p,q=None):
    output=Path(output);q=settings(Path(__file__).resolve().parents[1],p) if q is None else q
    def ready(c):
        folder=output/'baseline/arms'/c['name']
        return all((folder/n).exists() for n in ['summary.json','selected.pth','metrics.jsonl',*[f'candidate_{s}.pth' for s in p['selection_steps']]])
    baseline={c['name']:baseline_result(output,c,p,q) for c in cases(p) if ready(c)}
    training={c['name']:read(output/'arms'/c['name']/'summary.json') for c in cases(p) if (output/'arms'/c['name']/'summary.json').exists()}
    save(output/'baseline.json',baseline);save(output/'runs.json',{n:{k:v for k,v in d.items() if k!='curve'} for n,d in training.items()})
    save(output/'curves.json',curves_data(output,p))
    save(output/'comparisons.json',comparisons(baseline,training,p) if len(training)==len(cases(p)) else {})
    return baseline,training


def submission(code_sha,deployment_sha,root,repository):
    p=plan(repository);root=Path(root);digest=lambda v:hashlib.sha256(encoded(v)).hexdigest()
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_gpt_lr.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-gpt-lr'],
        artifacts=['outputs/task-runtime.json',*[f'outputs/cf-gpt-lr/{n}.json' for n in ARTIFACTS]])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=10000,max_parallel=2,authorization=p['authorization'],seeds=p['seeds'],methods=METHODS,
        new_lr=3e-5,baseline_lr=1e-4,source_job_id=SOURCE,source_compact_sha256=p['source_compact_sha256'],future_passes=0,quantizer_training=False,stop_after_study=True)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),
        manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),preflight=[['python','-u','-B',f'tests/incremental/test_cf_letter_{n}.py','--require-torch'] for n in ('base','downstream','gpt_lr')],
        verifier='verify_cf_letter_gpt_lr.py',optimizer_step_reservation=10000,input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)
