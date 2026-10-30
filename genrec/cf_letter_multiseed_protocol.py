"""Fixed five-new-seed, three-arm study; conditional on accepted C0/E1/Base GPT."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import math
import statistics
from genrec.cf_letter_protocol import read, save, sha, encoded, safe_file

JOB = 'cf-letter-multiseed-001'
SCOPE = 'cf-letter-multiseed-v1'
MANIFEST = 'docs/experiments/cf_letter_multiseed.json'
ARTIFACTS = ['compact', 'data_audit', 'rq_results', 'runs', 'selections', 'future', 'statistics']
METHODS = ['ft_old', 'plain', 'historical']


def plan(root):
    p = read(Path(root) / MANIFEST)
    if (p['job_id'] != JOB or p['scope_id'] != SCOPE or not p['execution_authorized']
        or p['seeds'] != [2027, 2028, 2029, 2030, 2031] or p['cohort_seed'] != 2026
        or p['optimizer_step_cap'] != 40000 or p['rq_steps'] != 1000 or p['gpt_steps'] != 2000
        or p['parallel'] != 2 or p['future_passes'] != 15 or p['primary_metric'] != 'ndcg@10'
        or p['methods'] != METHODS or p['retry'] or p['holdout_evaluation']
        or p['full_pipeline_retraining'] or p['seed_selection']):
        raise ValueError('Unsupported five-seed scope')
    for key in ('rq_template', 'gpt_template'):
        if sha(Path(root) / p[key]['path']) != p[key]['sha256']:
            raise ValueError('Accepted recipe template changed')
    return p


def rq_cases(p):
    return [dict(name=f's{seed}_{method}', seed=seed, stage='rq', lr=1e-5, beta1=.9,
                 optimizer='adamw', objective='rq_only' if method == 'plain' else 'mixed',
                 steps=p['rq_steps'], phase='multiseed_rq', method=method)
            for seed in p['seeds'] for method in METHODS[1:]]


def gpt_cases(p):
    return [dict(name=f's{seed}_{method}', seed=seed, stage='gpt', source=None if method == 'ft_old' else f's{seed}_{method}',
                 method=method, steps=p['gpt_steps'], phase='gpt_update')
            for seed in p['seeds'] for method in METHODS]


def settings(root, p, stage, seed=None):
    q = read(Path(root) / p[stage + '_template']['path'])
    q.update(job_id=p['job_id'], scope_id=p['scope_id'], seed=p['cohort_seed'] if seed is None else seed,
             optimizer_step_cap=len(p['seeds']) * (2*p['rq_steps'] if stage == 'rq' else 3*p['gpt_steps']),
             wall_seconds=p['wall_seconds'], parallel=p['parallel'], gpu_memory_limit_mib=p['gpu_memory_limit_mib'])
    if stage == 'rq':
        q.update(steps_per_branch=p['rq_steps'], pair_settings=[dict(name='lr1e5_b09', lr=1e-5, beta1=.9)])
    else:
        q.pop('zero_step_repair_of', None)
        q.update(steps_per_arm=p['gpt_steps'], arms=[{k:v for k,v in c.items() if k not in ('steps','phase')} for c in gpt_cases(p)])
    return q


def comparison_pairs(p):
    return [(f's{s}_{before}', f's{s}_{after}') for s in p['seeds']
            for before,after in [('plain','historical'), ('ft_old','plain'), ('ft_old','historical')]]


def interval(values):
    values = list(map(float, values))
    critical = {2:12.706204736432095, 5:2.7764451051977987}
    if len(values) not in critical or not all(math.isfinite(v) for v in values):
        raise ValueError('Require every predeclared seed, with finite observations')
    mean = statistics.mean(values); sd = statistics.stdev(values)
    width = critical[len(values)] * sd / math.sqrt(len(values))
    return dict(n=len(values), mean=mean, sd=sd, ci95=[mean-width,mean+width], seed_values=values)


def summarize(future, p, topk):
    expected = [c['name'] for c in gpt_cases(p)]
    if list(future) != expected:raise ValueError('Missing, extra or reordered seed/arm')
    keys = [f'{m}@{k}' for m in ('recall','ndcg') for k in topk]
    result = dict(seeds=p['seeds'], pilot_seed_excluded=2026, statistical_unit='seed',
                  ci_method='two-sided Student-t; sample SD; df=n-1', primary_metric=p['primary_metric'], groups={}, paired={})
    for denominator in ('full_denominator','conditional_on_servable'):
        result['groups'][denominator] = {m:{k:interval(future[f's{s}_{m}'][denominator][k] for s in p['seeds']) for k in keys} for m in METHODS}
        result['paired'][denominator] = {
            after+'_minus_'+before:{k:interval(future[f's{s}_{after}'][denominator][k]-future[f's{s}_{before}'][denominator][k] for s in p['seeds']) for k in keys}
            for before,after in [('plain','historical'),('ft_old','plain'),('ft_old','historical')]}
    return result


def dependencies(output, q):
    """Read-only copied Base/refresh evidence, used by outer scientific acceptance."""
    for folder, info, metadata in [('base',q['predecessor'],{'compact.json':'compact_sha256','data_audit.json':'audit_sha256'}),
                                   ('refresh',q['refresh_predecessor'],{'compact.json':'compact_sha256'})]:
        for name,digest in info['files_sha256'].items():
            if sha(safe_file(Path(output)/'inputs'/folder,name)) != digest:raise ValueError('Copied accepted dependency differs')
        for name,key in metadata.items():
            if sha(safe_file(Path(output)/'inputs'/folder,name)) != info[key]:raise ValueError('Copied accepted metadata differs')


def submission(code_sha, deployment_sha, root, repository):
    root=Path(root);p=plan(repository);digest=lambda v:hashlib.sha256(encoded(v)).hexdigest()
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,
        created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_multiseed.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-multiseed'],
        artifacts=['outputs/task-runtime.json',*[f'outputs/cf-multiseed/{n}.json' for n in ARTIFACTS]])
    rq=settings(repository,p,'rq')
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=40000,max_parallel=2,
        authorization=p['authorization'],seeds=p['seeds'],cohort_seed=2026,methods=METHODS,
        rq_steps=10000,gpt_steps=30000,future_passes=15,
        predecessor_read_only=[rq['predecessor'],rq['refresh_predecessor'],rq['audit_predecessor']],stop_after_study=True)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,request_sha256=digest(request),
        scope_path='task-scopes/'+SCOPE+'.json',scope_sha256=digest(scope),manifest=dict(path=MANIFEST,sha256=sha(Path(repository)/MANIFEST)),
        preflight=[['python','-u','-B',f'tests/incremental/test_cf_letter_{name}.py','--require-torch'] for name in ('base','optimizer_pair','downstream','multiseed')],
        verifier='verify_cf_letter_multiseed.py',optimizer_step_reservation=40000,input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)
