"""Immutable TIGER-only post-codebook alignment scope and bounded publications."""
import hashlib
from pathlib import Path
from datetime import datetime,timezone
from genrec.cf_letter_protocol import read,save,sha,encoded,safe_file
from genrec.hungarian_sids import canonical

JOB='tiger-hungarian-001'
SCOPE='tiger-hungarian-v1'
MANIFEST='docs/experiments/tiger_hungarian.json'
NAMES=['compact','audit','mapping','training','anchors','future','comparisons']
ROOT=Path(__file__).resolve().parents[1]


def plan():
    p=read(ROOT/MANIFEST)
    if (p['job_id']!=JOB or p['scope_id']!=SCOPE or p['optimizer_step_cap']!=6000 or p['steps_per_arm']!=2000
        or p['new_training_arms']!=['plain_unmapped','plain_hungarian','historical_hungarian'] or p['seed']!=2026
        or p['selection_steps']!=[0,100,250,500,1000,2000] or p['translation_positions']!=[0,1,2,3]
        or p['lr']!=.003 or p['future_passes_new']!=3 or p['automatic_retry']):raise ValueError('Unsupported TIGER Hungarian scope')
    return p


def cases(p):return [dict(name=n,steps=p['steps_per_arm'],phase='tiger_update') for n in p['new_training_arms']]


def bounded(path,value):
    # Check before replacing any successful publication, including on failure.
    if len(encoded(value))>65536:raise ValueError('Publication must be a bounded summary: '+str(path))
    save(path,value)


def audit_publication(output,full):
    output=Path(output);save(output/'audit_full.json',full)
    value=dict(full_audit_path='audit_full.json',full_audit_sha256=sha(output/'audit_full.json'),
        input_count=len(full['input_sha256']),input_manifest_sha256=canonical(full['input_sha256']),
        protocol_sha256=full['protocol_sha256'],probe_queries=len(full['probe_sample_ids']),probe_sample_ids_sha256=canonical(full['probe_sample_ids']),
        train_sample_ids_sha256=full['train_sample_ids_sha256'],fit_catalog=full['fit_catalog'],
        source_acceptance=full['source_acceptance'],future_used_for_matching_or_selection=False)
    bounded(output/'audit.json',value);return value


def source(root,p):
    root=Path(root);d=read(root/'main-deployment.json')
    for key,name in [('e1_dir','e1_dir'),('retuned_dir','retuned_dir'),('future_dir','previous_future_dir')]:
        if str(Path(d[key]).resolve())!=p[name]:raise ValueError('Installed exact TIGER source path differs: '+key)
    accepted=root/'workspaces'/p['source_acceptance']['job_id']/'outputs/main-acceptance.json'
    if sha(accepted)!=p['source_acceptance']['artifact_sha256']:raise ValueError('Accepted TIGER evidence changed')
    a=read(accepted)
    if a['execution_status']!='completed' or a['summary_sha256']!=p['source_acceptance']['summary_sha256']:raise ValueError('TIGER acceptance differs')
    future=Path(d['future_dir']);report=read(future/'summary.json')
    if sha(future/'summary.json')!=p['source_acceptance']['summary_sha256'] or report['config']!=p['previous_future_config']:raise ValueError('Accepted historical Future report changed')
    if report['training']['update_optimizer_steps']!=p['steps_per_arm'] or not report['selection_locked_before_future']:raise ValueError('Reused control budget/lock differs')
    protected={str(root/'main-deployment.json'):sha(root/'main-deployment.json'),str(accepted):sha(accepted),str(future/'summary.json'):sha(future/'summary.json'),**d['input_sha256']}
    retuned=Path(d['retuned_dir']);rq=read(retuned/'summary.json')
    if (sha(retuned/'summary.json')!=report['binding']['retuned_summary_sha256'] or sha(retuned/'request.json')!=report['binding']['retuned_request_sha256']
        or sha(retuned/'historical/last.pth')!=report['binding']['historical_rqvae_sha256'] or rq['settings']['max_steps']!=p['codebook_steps']):raise ValueError('Exact accepted retuned codebook source differs')
    protected.update(rq['protected_input_sha256'])
    for name in ('summary.json','request.json','compact.json','pair_checks.json'):protected[str(retuned/name)]=sha(retuned/name)
    for arm in ('plain','historical'):
        summary=read(retuned/arm/'summary.json');protected[str(retuned/arm/'summary.json')]=sha(retuned/arm/'summary.json')
        for name,h in summary['artifact_files_sha256'].items():
            path=safe_file(retuned/arm,name)
            if sha(path)!=h:raise ValueError('Retuned codebook artifact changed')
            protected[str(path)]=h
    for name,h in report['artifact_files_sha256'].items():
        path=safe_file(future,name)
        if sha(path)!=h:raise ValueError('Accepted Future artifact changed: '+name)
        protected[str(path)]=h
    for path,h in protected.items():
        if sha(path)!=h:raise ValueError('Original source changed: '+path)
    return d,report,protected


def project_values(output,p):
    output=Path(output);training={};future={}
    for name in p['arms']:
        d=output/('reused' if name=='historical_unmapped' else 'arms')/name
        if (d/'summary.json').exists():
            s=read(d/'summary.json');training[name]={k:s[k] for k in ('update_optimizer_steps','processed_examples','best_update_optimizer_step','best_update_validation','full_update_validation','exact_base_warm_start','selected_checkpoint_reload_equal')}
            training[name]['validation_history']=[dict(step=v['step'],full_denominator=v['metrics']['full_denominator'],mean_sequence_nll_servable=v['metrics']['mean_sequence_nll_servable']) for v in s['validation_history']]
        if (d/'future_summary.json').exists():
            s=read(d/'future_summary.json');future[name]={k:v for k,v in s.items() if k not in ('slices','split_stats')}
    return training,future


def projections(output,p):
    training,future=project_values(output,p)
    bounded(Path(output)/'training.json',training);bounded(Path(output)/'future.json',future)
    return training,future


def submission(code,repository=ROOT,root='/home/hao/frec/git-lab-buffer'):
    p=plan();digest=lambda x:hashlib.sha256(encoded(x)).hexdigest();root=Path(root)
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code,created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_tiger_hungarian.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/tiger-hungarian'],
        artifacts=['outputs/task-runtime.json',*[f'outputs/tiger-hungarian/{n}.json' for n in NAMES]])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=6000,max_parallel=2,authorization=p['authorization'],
        new_training_arms=p['new_training_arms'],steps_per_arm=2000,reused_control='historical_unmapped',future_passes_new=3,quantizer_training=False,seed=2026,stop_after_study=True)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code,request_sha256=digest(request),scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),
        manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),preflight=[['python','-u','-B','tests/incremental/test_tiger_hungarian.py','--require-torch']],
        verifier='verify_tiger_hungarian.py',optimizer_step_reservation=6000,input_sha256={str(root/'main-deployment.json'):p['deployment_sha256']})
    return dict(request=request,scope=scope,contract=contract)
