"""Pure data, accepted-predecessor and full-SID contracts for the CF refresh diagnostic."""
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime,timezone

import numpy as np

from genrec.cf_letter_protocol import encode_rows,encoded,read,save,safe_file,sha

MANIFEST='docs/experiments/cf_letter_refresh_diagnostic.json'
SCOPE='cf-letter-refresh-diag-v1'
JOB='cf-letter-refresh-diag-001'


def submission(code_sha,deployment_sha,root,repository):
    root=Path(root);p=plan(repository)
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,
        created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_refresh.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-refresh'],
        artifacts=['outputs/task-runtime.json','outputs/cf-refresh/compact.json','outputs/cf-refresh/data_audit.json','outputs/cf-refresh/curves.json'])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=12240,max_parallel=1,
        authorization='Hao: continue the proposed next stage only: new Base anchor reference, shared SASRec CF refresh and frozen-quantizer feature-only diagnosis; 30 cumulative epochs <=12240 steps. No Plain/Historical/GPT fitting, holdout or Future scoring.',
        allowed_query_splits=p['allowed_query_splits'],predecessor_read_only=p['predecessor'],stop_after_diagnostic=True)
    digest=lambda v:hashlib.sha256(encoded(v)).hexdigest()
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),
        scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),
        preflight=[['python','-u','-B','tests/incremental/test_cf_letter_base.py','--require-torch'],['python','-u','-B','tests/incremental/test_cf_letter_refresh.py','--require-torch']],
        verifier='verify_cf_letter_refresh.py',optimizer_step_reservation=12240,input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)


def plan(root):
    p=read(Path(root)/MANIFEST)
    if (p['scope_id']!=SCOPE or p['job_id']!=JOB or p['optimizer_step_cap']!=12240
            or p['sasrec']['step_cap']!=12240 or p['sasrec']['epochs']!=30
            or p['allowed_query_splits']!=['base_train','update_train','update_val']
            or any(p[k] is not False for k in ('quantizer_training','gpt_training','historical_training','holdout_evaluation','future_evaluation','retry'))):
        raise ValueError('Unsupported refresh diagnostic scope')
    return p


def anchor_indices(rows,size,seed):
    users=sorted({r['user'] for r in rows},key=lambda u:hashlib.sha256((str(seed)+':user:'+str(u)).encode()).hexdigest())
    train=set(users[:len(users)//2])
    indices=[i for i,r in enumerate(rows) if r['user'] in train]
    indices=sorted(indices,key=lambda i:hashlib.sha256((str(seed)+':sample:'+str(rows[i]['sample_id'])).encode()).hexdigest())[:size]
    if len(indices)!=size:raise ValueError('Insufficient exact anchor_train cohort')
    return np.asarray(sorted(indices),dtype=np.int64)


def transfer_predecessor(root,output,p):
    """Read only the explicitly authorized, hash-pinned accepted Base artifacts.

    The generic host binds the task manifest; this task performs the transitive
    accepted-artifact check/copy. It neither changes host read roots nor installs
    a dependency adapter. Every copied dependency is inside the new job output.
    """
    root=Path(root).resolve();source=root/'workspaces'/p['predecessor']['job_id']/'outputs/cf-base'
    if source.resolve()!=source:raise ValueError('Accepted predecessor tree must not contain symlink aliases')
    target=Path(output)/'inputs/base';target.mkdir(parents=True)
    bound={}
    for name,expected in p['predecessor']['files_sha256'].items():
        path=safe_file(source,name)
        if sha(path)!=expected:raise ValueError('Accepted Base artifact changed: '+name)
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        if sha(dest)!=expected:raise ValueError('Copied Base dependency differs')
        bound[str(path)]=expected
    for name,key in (('compact.json','compact_sha256'),('data_audit.json','audit_sha256')):
        path=safe_file(source,name)
        if sha(path)!=p['predecessor'][key]:raise ValueError('Accepted Base metadata changed')
        shutil.copyfile(path,target/name);bound[str(path)]=p['predecessor'][key]
    report=read(target/'compact.json')
    if report['research_outcome']!='base_chain_accepted' or report['code_sha']!=p['predecessor']['code_sha']:
        raise ValueError('Require accepted exact Base predecessor')
    return bound


def load_data(deployment,p):
    d=read(deployment);directory=Path(d['protocol_dir']).resolve()
    if directory!=(Path(d['project'])/'cache/incremental/Beauty/beauty_20261007T104452Z').resolve():
        raise ValueError('Unexpected temporal preparation')
    names=['summary.json','protocol_manifest.json','registries.json']+[s+'.jsonl' for s in p['allowed_query_splits']]
    binding={}
    for name in names:
        path=safe_file(directory,name);value=sha(path)
        if d['input_sha256'].get(str(path))!=value:raise ValueError('Deployment-bound source changed: '+name)
        binding[str(path)]=value
    summary=read(directory/'summary.json');manifest=read(directory/'protocol_manifest.json')
    if summary['execution_status']!='completed' or summary['protocol_sha256']!=manifest['protocol_sha256']:
        raise ValueError('Temporal protocol not completed')
    if any(summary['artifact_files_sha256'][name]!=binding[str(directory/name)] for name in names if name!='summary.json'):
        raise ValueError('Temporal source checksum differs')
    registries=read(directory/'registries.json');old=registries['base_train']['id2item'][1:]
    if len(old)!=p['old_catalog_items']:raise ValueError('Original Base catalog changed')
    if any(registries['base_train']['item2id'][item]!=i+1 for i,item in enumerate(old)):raise ValueError('Original item row identities changed')
    raw={};seen=set()
    for split in p['allowed_query_splits']:
        with (directory/(split+'.jsonl')).open() as stream:rows=[json.loads(line) for line in stream if line.strip()]
        if len(rows)!=p['expected_queries'][split]:raise ValueError('Query count differs: '+split)
        upper=manifest['cutoffs_exclusive']['update' if split=='update_val' else split]
        lower=manifest['windows'][split]['first_timestamp']
        if any(not lower<=r['target_time']<upper for r in rows):raise ValueError('Target outside allowed window')
        ids={r['sample_id'] for r in rows}
        if len(ids)!=len(rows) or seen&ids:raise ValueError('Duplicate or overlapping queries')
        seen|=ids;raw[split]=rows
    # New rows are allocated solely from actually used training inputs/positives.
    allowed=set(old)
    for split in ('base_train','update_train'):
        for row in raw[split]:allowed.update(row['item_seq'][-p['history']-1:])
    candidates=registries['update_train']['id2item'][1:]
    if not allowed.issubset(candidates):raise ValueError('Training vocabulary exceeds observed training catalog')
    old_set=set(old);items=old+[item for item in candidates if item in allowed and item not in old_set]
    mapping={item:i+1 for i,item in enumerate(items)}
    arrays={s:encode_rows(rows,mapping,p['history'],p['gpt']['user_buckets']) for s,rows in raw.items()}
    for split in ('base_train','update_train'):
        if np.any(arrays[split]['targets']==0):raise ValueError('Unknown fitting positive')
    cumulative={k:np.concatenate([arrays['base_train'][k],arrays['update_train'][k]]) for k in arrays['base_train']}
    val=arrays.get('update_val');probe=[]
    if val is not None:
        val['targets']=np.where(val['targets']<=len(old),val['targets'],0)
        probe=sorted(sorted(range(len(val['targets'])),key=lambda i:hashlib.sha256((str(p['seed'])+':'+str(val['samples'][i])).encode()).hexdigest())[:p['probe_size']])
    anchors=anchor_indices(raw['base_train'],p['anchor_size'],p['seed'])
    old_map={item:i+1 for i,item in enumerate(old)}
    anchor_data=encode_rows([raw['base_train'][i] for i in anchors],old_map,p['history'],p['gpt']['user_buckets'])
    anchor_path=safe_file(d['e1_dir'],'anchor_train.jsonl')
    if d['input_sha256'].get(str(anchor_path))!=sha(anchor_path):raise ValueError('Original anchor_train source changed')
    with anchor_path.open() as stream:previous=[json.loads(line) for line in stream if line.strip()]
    if len(previous)!=len(anchors) or any(a['source_row']!=int(i) or any(a[k]!=raw['base_train'][int(i)][k] for k in ('sample_id','user','item_seq','time_seq','target_time')) for a,i in zip(previous,anchors)):
        raise ValueError('Existing anchor_train identities do not match')
    binding[str(anchor_path)]=sha(anchor_path);binding[str(Path(deployment).resolve())]=sha(deployment)
    positive=np.bincount(cumulative['targets'],minlength=len(items)+1)
    history=np.bincount(cumulative['history'].reshape(-1),minlength=len(items)+1)
    return dict(train=cumulative,val=val,probe=np.asarray(probe,dtype=np.int64),anchors=anchor_data,
        anchor_users=np.asarray([raw['base_train'][i]['user'] for i in anchors]),anchor_source_rows=anchors),dict(
        input_sha256=binding,protocol_sha256=manifest['protocol_sha256'],items=items,old_items=old,
        allowed_query_splits=p['allowed_query_splits'],probe_sample_ids=val['samples'][probe].tolist() if val is not None else [],
        anchor_sample_ids=anchor_data['samples'].tolist(),positive_exposure=positive.tolist(),history_exposure=history.tolist())


def versioned_mapping(previous,codes,capacity):
    previous=np.asarray(previous,dtype=np.int64);codes=np.asarray(codes,dtype=np.int64)
    if previous.shape!=(len(codes),4) or codes.shape[1]!=3 or np.any(codes<0) or np.any(codes>=capacity):
        raise ValueError('Invalid old-catalog allocation')
    raw=previous-np.arange(4)*capacity
    if np.any(raw<0) or np.any(raw>=capacity) or len(set(map(tuple,previous.tolist())))!=len(previous):raise ValueError('Invalid previous full SIDs')
    reserved={};current={};required={};out=[]
    for row in raw:
        prefix=tuple(int(v) for v in row[:3]);reserved[prefix]=max(reserved.get(prefix,0),int(row[3])+1)
    cursor=dict(reserved)
    for old,row in zip(raw,codes):
        prefix=tuple(int(v) for v in row);current[prefix]=current.get(prefix,0)+1
        if np.array_equal(old[:3],row):suffix=int(old[3])
        else:suffix=cursor.get(prefix,0);cursor[prefix]=suffix+1
        required[prefix]=max(required.get(prefix,0),suffix+1)
        out.append([*row,suffix])
    overflow=[dict(semantic_code=list(k),current_items=current[k],reserved_width=reserved.get(k,0),required_capacity=v) for k,v in required.items() if v>capacity]
    feasible=not overflow
    stats=dict(items=len(codes),unique_semantic_codes=len(current),maximum_current_collision_bucket=max(current.values()),
        maximum_required_suffix_capacity=max(required.values()),suffix_capacity=capacity,full_sid_allocation_feasible=feasible,overflow_buckets=overflow[:20],overflow_bucket_count=len(overflow))
    result=np.asarray(out,dtype=np.int64)+np.arange(4)*capacity if feasible else None
    if feasible and len(set(map(tuple,result.tolist())))!=len(result):raise ValueError('Versioned full SIDs are not unique')
    return result,stats


def damage(records,reps=500,seed=2026):
    old=np.asarray([r['old_sequence_nll'] for r in records]);new=np.asarray([r['new_sequence_nll'] for r in records]);delta=new-old
    if not len(old) or not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('Invalid anchor NLL')
    clusters={}
    for row,value in zip(records,delta):
        acc=clusters.setdefault(row['user'],[0.,0.,0]);acc[0]+=float(value);acc[1]+=float(max(value,0));acc[2]+=1
    groups=np.asarray(list(clusters.values()));rng=np.random.default_rng(seed);samples=[]
    for _ in range(reps):
        selected=groups[rng.integers(0,len(groups),len(groups))];samples.append((selected[:,:2].sum(0)/selected[:,2].sum()).tolist())
    return dict(queries=len(old),users=len(groups),mean_old_sequence_nll=float(old.mean()),mean_new_sequence_nll=float(new.mean()),
        mean_delta_sequence_nll=float(delta.mean()),H=float(np.maximum(delta,0).mean()),mean_delta_token_ce=float(delta.mean()/4),
        improved_fraction=float(np.mean(delta<-1e-6)),worsened_fraction=float(np.mean(delta>1e-6)),
        changed_history_fraction=float(np.mean([r['history_changed'] for r in records])),
        user_cluster_delta_ci95=np.quantile(np.asarray(samples)[:,0],[.025,.975]).tolist(),
        user_cluster_H_ci95=np.quantile(np.asarray(samples)[:,1],[.025,.975]).tolist(),bootstrap_replicates=reps)
