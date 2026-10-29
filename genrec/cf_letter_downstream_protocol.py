"""Five fixed I0 mappings, causal data, locked selection and paired Future arithmetic."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,safe_file,encoded,encode_rows,metrics
from genrec.future_protocol import validate_stage,check_registry_prefixes

JOB='cf-letter-gpt-future-repair-001'
SCOPE='cf-letter-gpt-future-repair-v1'
MANIFEST='docs/experiments/cf_letter_downstream_repair.json'
NAMES=['ft_old','plain_b09','historical_b09','plain_b0','historical_b0']


def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def plan(root):
    p=read(Path(root)/MANIFEST)
    if (p['job_id']!=JOB or p['scope_id']!=SCOPE or p['execution_authorized'] is not True
        or p['optimizer_step_cap']!=10000 or p['steps_per_arm']!=2000 or p['selection_steps']!=[0,100,250,500,1000,2000]
        or [v['name'] for v in p['arms']]!=NAMES or [v['source'] for v in p['arms']]!=[None,'lr1e5_b09_rq_only','lr1e5_b09_mixed','lr1e5_b0_rq_only','lr1e5_b0_mixed']
        or p['lr']!=1e-4 or p['weight_decay']!=.01 or p['betas']!=[.9,.999] or p['eps']!=1e-8 or p['max_grad_norm']!=1.
        or p['batch_size']!=256 or p['history']!=20 or p['seed']!=2026 or p['probe_size']!=512 or p['old_catalog_items']!=10924
        or p['beam']!=20 or p['topk']!=[5,10,20] or p['parallel']!=2 or p['selection_metric']!='ndcg@10'
        or p['gpt']!=dict(d_model=768,layers=4,heads=6,ffn=1024,dropout=.1,user_buckets=8000)
        or p['quantizer']!=dict(dim=64,books=3,capacity=512,commitment=.25,outer_weight=1.)
        or p['expected_queries']!=dict(update_train=28023,update_val=3280,future_test=34189) or p['eval_batch_size']!=16
        or any(p[k] is not False for k in ('quantizer_training','sasrec_training','lambda_sweep','retry','holdout_evaluation'))):
        raise ValueError('Unsupported downstream scope')
    return p


def cases(p):return [dict(v,steps=p['steps_per_arm'],phase='gpt_update') for v in p['arms']]


def verify_zero_step_failure(root,p):
    proof=p['zero_step_repair_of'];source=Path(root)/'workspaces'/proof['job_id']/'outputs';path=safe_file(source,'task-runtime.json')
    if sha(path)!=proof['runtime_sha256']:raise ValueError('Original preflight-failure evidence changed')
    r=read(path)
    if (r['status']!='failed' or r['job_id']!=proof['job_id'] or r['code_sha']!=proof['code_sha']
        or r['command_started'] is not False or r['checks_completed']!=1 or (source/'cf-downstream').exists()):raise ValueError('Repair requires proven zero real GPT steps')
    return {str(path):proof['runtime_sha256']}


def future_table(training,future,p):
    """Use the configured cutoffs for both tiny preflight and production reports."""
    keys=[f'{metric}@{k}' for metric in ('recall','ndcg') for k in p['topk']]
    columns=['Mapping','Actual GPT Update steps','Selected Update step','Coverage',*[k.replace('recall','Recall').replace('ndcg','NDCG') for k in keys]]
    lines=['| '+' | '.join(columns)+' |','| --- | '+' | '.join(['---:']*(len(columns)-1))+' |']
    for arm in p['arms']:
        name=arm['name'];f=future[name];m=f['full_denominator']
        values=[name,training[name]['optimizer_steps'],f['selected_update_step'],f['coverage'],*[m[k] for k in keys]]
        lines.append('| '+' | '.join(map(str,values))+' |')
    return lines


def select(curve,p):
    if [v['step'] for v in curve]!=p['selection_steps']:raise ValueError('Incomplete selection curve')
    return max(curve,key=lambda v:v['metrics']['full_denominator'][p['selection_metric']])


def check_mapping(mapping,p):
    k=p['quantizer']['capacity'];a=np.asarray(mapping)
    if a.shape!=(p['old_catalog_items'],4) or not np.issubdtype(a.dtype,np.integer):raise ValueError('Wrong I0 mapping shape/type')
    raw=a-np.arange(4)*k
    if (np.any(raw<0) or np.any(raw>=k) or len(set(map(tuple,a.tolist())))!=len(a)):raise ValueError('Invalid positional full SID')


def transfer(root,output,p):
    """Only immutable accepted inputs; every consumed source hash is rechecked later."""
    root=Path(root);output=Path(output);bound={}
    base=root/'workspaces'/p['predecessor']['job_id']/'outputs/cf-base'
    pair=root/'workspaces'/p['pair_predecessor']['job_id']/'outputs/cf-optimizer-pair'
    if base.resolve()!=base or pair.resolve()!=pair:raise ValueError('Source alias')
    def copy(source,name,destination,expected):
        path=safe_file(source,name)
        if sha(path)!=expected:raise ValueError('Accepted dependency changed: '+name)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        if sha(destination)!=expected:raise ValueError('Private dependency copy differs')
        bound[str(path)]=expected
    for name in ('gpt/selected.pth','item_sids.npy'):
        copy(base,name,output/'inputs/base'/name,p['predecessor']['files_sha256'][name])
    copy(base,'compact.json',output/'inputs/base/compact.json',p['predecessor']['compact_sha256'])
    copy(pair,'compact.json',output/'inputs/pair/compact.json',p['pair_predecessor']['compact_sha256'])
    copy(pair,'local_artifact_index.json',output/'inputs/pair/local_artifact_index.json',p['pair_predecessor']['index_sha256'])
    index=read(output/'inputs/pair/local_artifact_index.json');report=read(output/'inputs/pair/compact.json')
    if (report['code_sha']!=p['pair_predecessor']['code_sha'] or report['execution_status']!='completed'
        or report['optimizer_steps_spent']!=4000 or report['uncertain_step_records']):raise ValueError('Wrong accepted paired source')
    base_report=read(output/'inputs/base/compact.json')
    if base_report['code_sha']!=p['predecessor']['code_sha'] or base_report['stages']['gpt']['selected_step']!=p['predecessor']['gpt_step']:raise ValueError('Wrong Base GPT')
    from genrec.cf_letter_refresh_protocol import versioned_mapping
    old=np.load(output/'inputs/base/item_sids.npy',allow_pickle=False);check_mapping(old,p)
    for arm in p['arms']:
        if arm['source'] is None:mapping=old
        else:
            prefix='arms/'+arm['source']+'/'+arm['source']+'/'
            for filename in ('item_sids_1000.npy','semantic_codes_1000.npy','last.pth'):
                copy(pair,prefix+filename,output/'inputs/pair'/arm['source']/filename,index[prefix+filename])
            mapping=np.load(output/'inputs/pair'/arm['source']/'item_sids_1000.npy',allow_pickle=False)
            raw=np.load(output/'inputs/pair'/arm['source']/'semantic_codes_1000.npy',allow_pickle=False)
            rebuilt,capacity=versioned_mapping(old,raw,p['quantizer']['capacity'])
            if not capacity['full_sid_allocation_feasible'] or not np.array_equal(mapping,rebuilt):raise ValueError('Frozen full-SID ledger differs')
        check_mapping(mapping,p);dest=output/'mappings'/arm['name'];dest.parent.mkdir(parents=True,exist_ok=True);np.save(str(dest)+'.npy',mapping)
    return bound


def metadata(deployment,p):
    d=read(deployment);directory=Path(d['protocol_dir'])
    if directory.resolve()!=(Path(d['project'])/'cache/incremental/Beauty/beauty_20261007T104452Z').resolve():raise ValueError('Unexpected protocol input')
    bound={str(Path(deployment).resolve()):sha(deployment)}
    for name in ('summary.json','protocol_manifest.json','registries.json'):
        path=safe_file(directory,name);value=sha(path)
        if d['input_sha256'].get(str(path))!=value:raise ValueError('Deployment metadata changed')
        bound[str(path)]=value
    summary=read(directory/'summary.json');manifest=read(directory/'protocol_manifest.json');registries=read(directory/'registries.json')
    if summary['execution_status']!='completed' or summary['protocol_sha256']!=manifest['protocol_sha256']:raise ValueError('Unprepared temporal data')
    check_registry_prefixes(registries,manifest)
    old=registries['base_train']['id2item'][1:]
    if len(old)!=p['old_catalog_items']:raise ValueError('Common I0 identity/count changed')
    for name in ('protocol_manifest.json','registries.json'):
        if summary['artifact_files_sha256'][name]!=bound[str(directory/name)]:raise ValueError('Metadata summary differs')
    return d,directory,summary,manifest,registries,bound


def valid_lock(lock,p):
    if (not isinstance(lock,dict) or lock.get('job_id')!=p['job_id'] or lock.get('future_used_for_selection') is not False
        or list(lock.get('arms',{}))!=[v['name'] for v in p['arms']]
        or any(v['selected_step'] not in p['selection_steps'] or len(v['selected_sha256'])!=64 or len(v['mapping_sha256'])!=64 for v in lock['arms'].values())):
        raise ValueError('All five selections must be locked before Future')


def load_stage(deployment,p,stage,lock=None):
    if stage not in ('update_train','update_val','future_test'):raise ValueError('Unapproved split')
    if stage=='future_test':valid_lock(lock,p)
    d,directory,summary,manifest,registries,bound=metadata(deployment,p)
    path=safe_file(directory,stage+'.jsonl');value=sha(path)
    if d['input_sha256'].get(str(path),summary['artifact_files_sha256'].get(path.name))!=value or summary['artifact_files_sha256'].get(path.name)!=value:raise ValueError('Stage checksum changed')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows)!=p['expected_queries'][stage]:raise ValueError('Wrong stage query count')
    validate_stage(rows,stage,manifest,registries)
    registry=registries['base_train']['item2id'];arrays=encode_rows(rows,registry,p['history'],p['gpt']['user_buckets'])
    arrays['raw_users']=np.asarray([r['user'] for r in rows]);arrays['source_rows']=np.arange(len(rows),dtype=np.int64)
    bound[str(path)]=value
    return arrays,bound


def load_training(deployment,p):
    train,bound=load_stage(deployment,p,'update_train');val,other=load_stage(deployment,p,'update_val');bound.update(other)
    if set(train['samples'])&set(val['samples']):raise ValueError('Train/validation sample overlap')
    all_train=len(train['targets']);keep=np.flatnonzero(train['targets']>0)
    if not len(keep):raise ValueError('No common-I0 Update training queries')
    train={k:v[keep] for k,v in train.items()}
    probe=np.asarray(sorted(sorted(range(len(val['targets'])),key=lambda i:hashlib.sha256((str(p['seed'])+':'+str(val['samples'][i])).encode()).hexdigest())[:p['probe_size']]),dtype=np.int64)
    return train,val,probe,dict(input_sha256=bound,all_update_train_queries=all_train,eligible_update_train_queries=len(keep),
        excluded_update_train_queries=all_train-len(keep),probe_sample_ids=val['samples'][probe].tolist(),catalog_items=p['old_catalog_items'],
        training_sample_ids_sha256=digest(train['samples'].tolist()))


def save_arrays(path,data):np.savez(path,**data)
def load_arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def aggregate(records,p):
    for row in records:
        if any(type(i) is not int or not 1<=i<=p['old_catalog_items'] for i in row['predictions']):raise ValueError('Prediction outside fixed I0')
    result=metrics(records,p['topk']);result.update(coverage=result['servable_queries']/len(records),
        examples_with_all_history_unavailable=sum(r['all_history_unavailable'] for r in records),
        num_beams=p['beam'],nll_target_tokens=4,history_slots=p['history'],target_eos=False)
    return result


def paired(before,after,p):
    if [(v['sample_id'],v['user'],v['target']) for v in before]!=[(v['sample_id'],v['user'],v['target']) for v in after]:raise ValueError('Future cohorts differ')
    def values(rows):
        return np.asarray([[float(r['target'] in r['predictions'][:k]) if metric=='recall' else
            (1/np.log2(r['predictions'].index(r['target'])+2) if r['target'] in r['predictions'][:k] else 0.)
            for metric in ('recall','ndcg') for k in p['topk']] for r in rows])
    delta=values(after)-values(before);users={}
    for row,change in zip(before,delta):
        u=users.setdefault(row['user'],[np.zeros(delta.shape[1]),0]);u[0]+=change;u[1]+=1
    sums=np.asarray([v[0] for v in users.values()]);counts=np.asarray([v[1] for v in users.values()]);rng=np.random.default_rng(p['seed']);draws=[]
    for _ in range(p['bootstrap_replicates']):
        take=rng.integers(0,len(users),len(users));draws.append(sums[take].sum(0)/counts[take].sum())
    baseline=values(before).mean(0);mean=delta.mean(0);ci=np.quantile(draws,[.025,.975],axis=0)
    keys=[f'{m}@{k}' for m in ('recall','ndcg') for k in p['topk']]
    return dict(queries=len(before),users=len(users),bootstrap_replicates=p['bootstrap_replicates'],bootstrap_unit='user; query-weighted clusters',
        metrics={k:dict(absolute_difference=float(mean[i]),relative_change=float(mean[i]/baseline[i]) if baseline[i] else None,ci95=ci[:,i].tolist()) for i,k in enumerate(keys)})


def submission(code_sha,deployment_sha,root,repository):
    root=Path(root);p=plan(repository)
    names=['compact','data_audit','training','selection_locked','future','comparisons']
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_downstream.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-downstream'],
        artifacts=['outputs/task-runtime.json',*[f'outputs/cf-downstream/{n}.json' for n in names]])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=10000,max_parallel=2,authorization=p['authorization'],
        allowed_query_splits=['update_train','update_val','future_test'],catalog='common Base train I0,10924 items',
        predecessor_read_only=[p['predecessor'],p['pair_predecessor']],future_passes=5,stop_after_study=True)
    scope.update(zero_step_repair_of=p['zero_step_repair_of'],family_actual_step_cap=10000)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),
        manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),preflight=[['python','-u','-B','tests/incremental/test_cf_letter_base.py','--require-torch'],
        ['python','-u','-B','tests/incremental/test_cf_letter_downstream.py','--require-torch']],verifier='verify_cf_letter_downstream.py',optimizer_step_reservation=10000,
        input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)
