"""Scientific acceptance for new explicitly requested pairs; old receipt stays blocked."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,safe_file
from genrec.cf_letter_refresh_protocol import load_data
from genrec.cf_letter_optimizer_protocol import counters
from genrec.cf_letter_optimizer_pair_protocol import JOB,SCOPE,plan,cases,assessments,curve_groups
from verify_cf_letter_optimizer import verify_arm
ROOT=Path(__file__).resolve().parent


def production_gpu():
    """Explicit task-authorized read-only GPU verification, not a Runner change."""
    from run_cf_letter_base import gpu
    began=time.monotonic()
    while True:
        live,_=gpu()
        if not live:break
        if time.monotonic()-began>600:raise RuntimeError('Foreign GPU prevents scientific reload; preserve results')
        time.sleep(5)
    # Generic verifier defaults to CPU. This pinned manifest explicitly authorizes
    # production-backend reload; set visibility before the first torch import.
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from genrec.cf_letter_base_training import seed_all
    torch.set_num_threads(1);seed_all(2026)
    if not torch.cuda.is_available():raise RuntimeError('Production CUDA assignment verification unavailable')
    torch.cuda.set_device(0)
    def guard():
        live,memory=gpu()
        if live-{os.getpid()} or memory>24576:raise RuntimeError('Foreign GPU or memory guard during reload; preserve results')
    return guard


def verify(context,*,assignment_device='cuda:0'):
    c=read(context);output=Path(c['workspace'])/'outputs/cf-optimizer-pair';p=plan(ROOT)
    r=read(output/'compact.json');audit=read(output/'data_audit.json')
    if (r['stage']!=SCOPE or r['job_id']!=JOB or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed'
        or r['optimizer_step_cap']!=p['optimizer_step_cap'] or not r['original_inputs_unchanged'] or r['uncertain_step_records']
        or r['fixed_history_weight']!=p['fixed_history_weight'] or r['previous_screen_eligible'] is not False
        or r['previous_scientific_acceptance']!='blocked' or r['canonical_assignment_backend']!=p['canonical_assignment_backend']
        or assignment_device!=p['canonical_assignment_backend']
        or any(r[k] is not False for k in ('gpt_training_performed','sasrec_training_performed','validation_evaluated','holdout_evaluated','future_evaluated'))):
        raise ValueError('Explicit paired identity/scope differs')
    for name,value in r['returned_artifact_sha256'].items():
        if sha(safe_file(output,name+'.json'))!=value:raise ValueError('Returned paired artifact differs')
    if sha(output/'local_artifact_index.json')!=r['local_artifact_index_sha256']:raise ValueError('Local index differs')
    cache={}
    for name,value in read(output/'local_artifact_index.json').items():
        path=safe_file(output,name);stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in cache:cache[key]=sha(path)
        if cache[key]!=value:raise ValueError('Local paired artifact changed')
    for path,value in audit['input_sha256'].items():
        if sha(path)!=value:raise ValueError('Read-only source input changed')
    for name,value in p['audit_predecessor']['files_sha256'].items():
        if sha(output/'inputs/assignment_audit'/name)!=value:raise ValueError('Assignment prerequisite changed')
    for name,value in {**p['predecessor']['files_sha256'],'compact.json':p['predecessor']['compact_sha256'],'data_audit.json':p['predecessor']['audit_sha256']}.items():
        if sha(output/'inputs/base'/name)!=value:raise ValueError('Copied Base dependency differs')
    for name,value in {**p['refresh_predecessor']['files_sha256'],'compact.json':p['refresh_predecessor']['compact_sha256']}.items():
        if sha(output/'inputs/refresh'/name)!=value:raise ValueError('Copied refresh dependency differs')
    deployment=next(v for v in c['contract']['input_sha256'] if v.endswith('/main-deployment.json'));data,binding=load_data(deployment,p)
    if (audit['allowed_query_splits']!=p['allowed_query_splits'] or audit['training_items']!=p['expected_training_items'] or audit['anchor_train_queries']!=p['anchor_size']
        or audit['prepared_sha256']!=sha(output/'prepared.npz')):raise ValueError('Training-only data binding differs')
    results=r['pair_results'];expected=cases(p)
    if len(results)>4 or any(v['case']!=expected[i] for i,v in enumerate(results)):raise ValueError('Unexpected paired matrix/order')
    # New tasks use strict exact canonical equality on the production backend.
    # CPU cross-backend equality is not substituted for training's hard mapping.
    guard=production_gpu() if assignment_device!='cpu' else None
    verified=[]
    for result in results:
        name=result['case']['name'];arm=verify_arm(output/'arms'/name,data,binding,p,result['case'],r['code_sha'],assignment_device=assignment_device,gpu_guard=guard)
        if {k:v for k,v in arm.items() if k!='monitors'}!=result:raise ValueError('Root pair summary differs')
        if read(output/(name+'_curves.json'))!=read(output/'arms'/name/(name+'_curves.json')):raise ValueError('Returned pair curve differs')
        verified.append(arm)
    for name,arms in curve_groups(p).items():
        expected_curves={arm:read(output/(arm+'_curves.json')) for arm in arms if (output/(arm+'_curves.json')).exists()}
        if read(output/(name+'.json'))!=expected_curves:raise ValueError('Lossless grouped pair curve differs')
    actual=assessments(results,p)
    if actual!=r['paired_assessment'] or actual!=read(output/'paired_assessment.json'):raise ValueError('Paired efficacy arithmetic differs')
    journal=read(output/'launch_journal.json')
    if journal!=[v['case'] for v in results] or len({v['name'] for v in journal})!=len(journal) or sum(v['steps'] for v in journal)>p['optimizer_step_cap']:raise ValueError('Prospective step claims differ')
    total,steps,intents=counters(output)
    if intents or total!=r['optimizer_steps_spent'] or total!=sum(v['optimizer_steps'] for v in results) or total>p['optimizer_step_cap'] or steps!=r['stage_step_counts']:raise ValueError('Durable paired budget differs')
    if not r['resource_stop_reason'] and (len(results)!=4 or r['research_outcome']!='explicit_two_setting_pairs_completed'):raise ValueError('Incomplete unguarded study')
    # Same C0/E1, references, sampler seeds and per-step sequence were independently
    # checked in verify_arm; this explicitly also compares endpoint sampler states.
    if len(verified)==4:
        import torch
        endpoints=[]
        for arm in verified:
            checkpoint=torch.load(output/'arms'/arm['case']['name']/arm['case']['name']/'last.pth',map_location='cpu',weights_only=False)
            endpoints.append((checkpoint['sampler']['processed'],checkpoint['anchor_sampler']['processed']))
        if len(set(endpoints))!=1:raise ValueError('Matched sampling endpoint differs')
    lines=['### Explicit two-setting RQ-only / Mixed verification','',f'Job `{JOB}`; code `{r["code_sha"]}`. New codebook steps: {total} / 4000. Frozen C0/E1/Base GPT; fresh reset optimizer for every arm.',
        'Hao explicitly authorized both lr=1e-5/beta1 settings after the original screening found no eligible candidate. This is a new paired scope; the old 2% stability gate and blocked scientific receipt remain unchanged.',
        '', '| Setting | Objective | Steps | Final full H | Tail H mean | Tail native RQ | Tail hard MSE | Joint tail stable |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for b in results:
        s=b['stability'];lines.append('| '+' | '.join(str(x) for x in (b['case']['setting'],b['case']['objective'],b['optimizer_steps'],b.get('final_full_H'),s.get('H',{}).get('mean'),s.get('rq',{}).get('mean'),s.get('reconstruction_mse',{}).get('mean'),s['passed']))+' |')
    lines+=['',f'Fixed Mixed lambda={p["fixed_history_weight"]}; beta2=0.999, epsilon=1e-8, wd=0, clipping=0.1, temperature=0.5. No lambda calibration, STE/residual/native-RQ change or H-only retraining.',
        '', 'Paired assessment: `'+json.dumps(actual,sort_keys=True)+'`.', '',
        'Acceptance reaggregates raw query NLL/H and every-step suffix capacity, checks all scheduled checkpoint indices exactly on production CUDA with the original full-catalog and old-item chunk shapes, verifies native RQ/MSE and CPU initial/final GPT likelihoods, and checks hashes, fresh optimizer state, sampler order and step counts. GPU reload is read-only and authorized by this task manifest. These are training-side finite-window results, not asymptotic convergence or downstream gains. No holdout, validation ranking, Future, SASRec/GPT training, automatic extension, retry or next experiment.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=total,report_markdown='\n'.join(lines))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--context',required=True);a.add_argument('--output',required=True)
    args=a.parse_args();save(args.output,verify(args.context))
