"""One diagnosed continuation, exactly6000 new steps within the40000-step family."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from genrec.cf_letter_protocol import read,save,sha,encoded,safe_file
from genrec.cf_letter_multiseed_protocol import plan as original_plan,settings,gpt_cases,rq_cases,ARTIFACTS
from run_cf_letter_multiseed import total_counters

JOB='cf-letter-multiseed-resume-001'
SCOPE='cf-letter-multiseed-resume-v1'
MANIFEST='docs/experiments/cf_letter_multiseed_resume.json'
SOURCE_JOB='cf-letter-multiseed-001'
SOURCE_CODE='9f4bebccfa8cc8617b7e4c29ee6e1f4f8dc6e451'
MISSING=['s2031_ft_old','s2031_plain','s2031_historical']


def plan(root):
    root=Path(root);m=read(root/MANIFEST)
    if (m['job_id']!=JOB or m['scope_id']!=SCOPE or m['new_step_cap']!=6000 or m['family_step_cap']!=40000
        or m['missing_gpt_arms']!=MISSING or m['source_job_id']!=SOURCE_JOB or m['source_code_sha']!=SOURCE_CODE
        or not m['execution_authorized'] or m['source_manifest_sha256']!=sha(root/'docs/experiments/cf_letter_multiseed.json')
        or m['failure_evidence_sha256']!=sha(root/'docs/experiments/cf_letter_multiseed_failure.json')):
        raise ValueError('Continuation scope/source changed')
    p=original_plan(root);p.update(job_id=JOB,scope_id=SCOPE)
    return m,p


def expected_counts(p,complete=False):
    return {**{'rq/'+c['name']:p['rq_steps'] for c in rq_cases(p)},
        **{'gpt/'+c['name']:p['gpt_steps'] for c in gpt_cases(p) if complete or c['name'] not in MISSING}}


def source_check(root,m,p,*,hash_files=True):
    source=Path(root)/'workspaces'/SOURCE_JOB/'outputs/cf-multiseed'
    if source.is_symlink() or not source.is_dir():raise ValueError('Missing/unsafe original output')
    proof=read(Path(__file__).resolve().parents[1]/'docs/experiments/cf_letter_multiseed_failure.json')
    state=read(Path(root)/'state'/(SOURCE_JOB+'.json'));result=state['result']
    if hashlib.sha256(encoded(result)).hexdigest()!=proof['result_sha256'] or state['pending_upload']:
        raise ValueError('Original terminal result differs or is not delivered')
    if result['status']!='failed' or result['code_sha']!=SOURCE_CODE or result['actual_code_sha']!=SOURCE_CODE or result['exit_code']!=1:
        raise ValueError('Not the diagnosed terminal task')
    for name,digest in proof['artifacts_sha256'].items():
        relative=name.removeprefix('artifacts/'+SOURCE_JOB+'/outputs/')
        file=source.parent/'task-runtime.json' if relative=='task-runtime.json' else safe_file(source,relative.removeprefix('cf-multiseed/'))
        if sha(file)!=digest:raise ValueError('Original returned artifact differs')
    c=read(source/'compact.json');runtime=read(source.parent/'task-runtime.json')
    if (c['execution_status']!='failed' or c['code_sha']!=SOURCE_CODE or c['optimizer_steps_spent']!=34000
        or c['uncertain_step_records'] or not c['original_inputs_unchanged'] or c['future_names']
        or 'CalledProcessError(128' not in c['error'] or "'fetch'" not in c['error']
        or runtime['checks_completed']!=4 or not runtime['command_started'] or runtime['status']!='failed'):
        raise ValueError('Original failure/budget differs')
    n,counts,intents=total_counters(source)
    if intents or counts!=expected_counts(p) or n!=34000:raise ValueError('Original durable step distribution differs')
    for name in MISSING:
        if (source/'gpt/arms'/name).exists():raise ValueError('A missing arm has unexpected prior state')
    if (any((source/'gpt'/n).exists() for n in ('selection_locked.json','future_opened.json','future_prepared.npz'))
        or any((source/'gpt/arms').glob('*/future_queries.jsonl')) or read(source/'future.json') or read(source/'statistics.json')):
        raise ValueError('Future already opened or source partially evaluated')
    index=read(source/'local_artifact_index.json')
    if sha(source/'local_artifact_index.json')!=c['local_artifact_index_sha256']:raise ValueError('Original index hash differs')
    if hash_files:
        cache={}
        for name,digest in index.items():
            f=safe_file(source,name);st=f.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
            if key not in cache:cache[key]=sha(f)
            if cache[key]!=digest:raise ValueError('Original indexed file changed: '+name)
        for f,digest in read(source/'data_audit.json')['input_sha256'].items():
            if sha(f)!=digest:raise ValueError('Original input changed')
    return source,index


def immutable_reuse(name):
    return name.startswith(('rq/','gpt/arms/','gpt/mappings/','gpt/inputs/')) or name in ('gpt/prepared.npz','gpt/binding.json')


def copy_indexed(source,output,index):
    """Copy once per old inode; hardlinks only between the new private copies."""
    output=Path(output);cache={}
    for name,digest in index.items():
        f=safe_file(source,name)
        if any(part.is_symlink() for part in [f,*list(f.parents)[:len(Path(name).parts)-1]]):raise ValueError('Symlink in source path')
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('Duplicate copy target')
        st=f.stat();key=(st.st_dev,st.st_ino)
        if key in cache:os.link(cache[key],target)
        else:shutil.copy2(f,target);cache[key]=target
        if target.stat().st_ino==st.st_ino and target.stat().st_dev==st.st_dev:raise ValueError('Copy aliases original input')
    for name,digest in index.items():
        if sha(output/name)!=digest:raise ValueError('Copied source hash differs')
    save(output/'provenance/reused_files.json',{n:v for n,v in index.items() if immutable_reuse(n)})


def check_reused(output):
    cache={}
    for name,digest in read(Path(output)/'provenance/reused_files.json').items():
        f=safe_file(output,name);st=f.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha(f)
        if cache[key]!=digest:raise ValueError('Reused checkpoint/evidence changed: '+name)


def submission(code_sha,deployment_sha,root,repository):
    m,p=plan(repository);root=Path(root);digest=lambda x:hashlib.sha256(encoded(x)).hexdigest()
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,
        created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_multiseed_resume.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-multiseed-resume'],
        artifacts=['outputs/task-runtime.json',*[f'outputs/cf-multiseed-resume/{n}.json' for n in ARTIFACTS]])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=6000,max_parallel=2,authorization=m['authorization'],
        source_job_id=SOURCE_JOB,source_code_sha=SOURCE_CODE,source_steps=34000,family_actual_step_cap=40000,
        missing_gpt_arms=MISSING,future_passes=15,stop_after_study=True,automatic_retry=False)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),
        scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),
        preflight=[['python','-u','-B',f'tests/incremental/test_cf_letter_{n}.py','--require-torch'] for n in ('base','optimizer_pair','downstream','multiseed','multiseed_resume')],
        verifier='verify_cf_letter_multiseed_resume.py',optimizer_step_reservation=6000,input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)
