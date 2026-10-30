"""Read-only production-backend RQ and CPU GPT acceptance plus seed statistics."""
import argparse
import json
from pathlib import Path
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,safe_file
from genrec.cf_letter_multiseed_protocol import JOB,SCOPE,ARTIFACTS,plan,settings,rq_cases,gpt_cases,comparison_pairs,summarize,dependencies
from genrec.cf_letter_refresh_protocol import load_data,versioned_mapping
from genrec.cf_letter_optimizer_protocol import load_anchors
from run_cf_letter_multiseed import total_counters
ROOT=Path(__file__).resolve().parent


def verify(context,*,assignment_device='cuda:0',configuration=None,directory=None,job_id=JOB,scope_id=SCOPE,rq_code_sha=None,gpt_code_shas=None):
    c=read(context);p=configuration if configuration is not None else plan(ROOT)
    output=Path(directory) if directory is not None else Path(c['workspace'])/'outputs/cf-multiseed';r=read(output/'compact.json');rq=output/'rq';gpt=output/'gpt'
    if (r['job_id']!=job_id or r['stage']!=scope_id or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed'
        or not r['original_inputs_unchanged'] or r['uncertain_step_records'] or not r['selection_locked_before_future']
        or r['future_used_for_selection'] or any(r[k] for k in ('sasrec_training_performed','base_training_performed','holdout_evaluated'))
        or r['seeds']!=p['seeds'] or r['stop_new_stages_reason']):raise ValueError('Multi-seed scope/status differs')
    for n,v in r['returned_artifact_sha256'].items():
        if n not in ARTIFACTS or sha(safe_file(output,n+'.json'))!=v:raise ValueError('Returned artifact differs')
    if set(r['returned_artifact_sha256'])!=set(ARTIFACTS)-{'compact'}:raise ValueError('Missing bounded artifact')
    if sha(output/'local_artifact_index.json')!=r['local_artifact_index_sha256']:raise ValueError('Local index changed')
    cache={}
    for n,v in read(output/'local_artifact_index.json').items():
        f=safe_file(output,n);st=f.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha(f)
        if cache[key]!=v:raise ValueError('Bound local file changed')
    audit=read(output/'data_audit.json')
    for f,v in audit['input_sha256'].items():
        if sha(f)!=v:raise ValueError('Original accepted input changed')
    q=settings(ROOT,p,'rq');dependencies(rq,q)
    for n,v in q['audit_predecessor']['files_sha256'].items():
        if sha(rq/'inputs/assignment_audit'/n)!=v:raise ValueError('Production assignment audit changed')
    deployment=next(k for k in c['contract']['input_sha256'] if k.endswith('/main-deployment.json'))
    data,binding=load_data(deployment,q);saved=load_anchors(rq/'prepared.npz')
    if sha(rq/'prepared.npz')!=audit['rq_prepared_sha256'] or sha(gpt/'data_audit.json')!=audit['gpt_data_audit_sha256']:raise ValueError('Prepared cohort/audit differs')
    if any(not np.array_equal(saved['anchors'][k],v) for k,v in data['anchors'].items()):raise ValueError('Fixed2026 anchor cohort differs')
    if audit['cohort_seed']!=p['cohort_seed'] or read(rq/'launch_journal.json')!=rq_cases(p):raise ValueError('RQ seed order/reservations differ')
    from verify_cf_letter_optimizer_pair import production_gpu
    guard=production_gpu() if assignment_device!='cpu' else None
    from verify_cf_letter_optimizer import verify_arm
    results=read(output/'rq_results.json')
    if list(results)!=[v['name'] for v in rq_cases(p)]:raise ValueError('Incomplete RQ seed matrix')
    previous=np.load(rq/'inputs/base/item_sids.npy',allow_pickle=False)
    for case in rq_cases(p):
        name=case['name'];params=settings(ROOT,p,'rq',case['seed']);folder=rq/'arms'/name
        if results[name]!=read(folder/'arm_result.json'):raise ValueError('RQ publication differs')
        result=verify_arm(folder,data,binding,params,case,rq_code_sha if rq_code_sha is not None else r['code_sha'],assignment_device=assignment_device,gpu_guard=guard)
        if result['optimizer_steps']!=p['rq_steps'] or not result['capacity_feasible']:raise ValueError('Infeasible/incomplete seed')
        print(json.dumps(dict(verification='rq_complete',seed=case['seed'],arm=name)),flush=True)
    def gpt_sources(folder,params):
        for n,v in q['predecessor']['files_sha256'].items():
            if sha(folder/'inputs/base'/n)!=v:raise ValueError('GPT Base input changed')
        for case in gpt_cases(p):
            actual=np.load(folder/'mappings'/(case['name']+'.npy'),allow_pickle=False)
            if case['source'] is None:expected=previous
            else:
                source=rq/'arms'/case['source']/case['source'];codes=np.load(source/f'semantic_codes_{p["rq_steps"]}.npy',allow_pickle=False)
                expected,capacity=versioned_mapping(previous,codes,params['quantizer']['capacity'])
                if not capacity['full_sid_allocation_feasible'] or not np.array_equal(expected,np.load(source/f'item_sids_{p["rq_steps"]}.npy',allow_pickle=False)):
                    raise ValueError('Wrong final RQ map/suffix')
            if not np.array_equal(actual,expected):raise ValueError('GPT did not consume its own final seed map')
    # Existing independently tested acceptance rechecks all query records, fitting
    # traces, checkpoints, optimizer resets, all15 lock and CPU likelihood samples.
    from verify_cf_letter_downstream import verify as verify_gpt
    qg=settings(ROOT,p,'gpt')
    verdict=verify_gpt(context,settings=qg,directory=gpt,job_id=job_id,scope_id=scope_id,
                       dependency_check=gpt_sources,comparison_pairs=comparison_pairs(p),checkpoint_code_shas=gpt_code_shas)
    training=read(gpt/'training.json');future=read(gpt/'future.json');stats=summarize(future,p,qg['topk'])
    stats['per_seed_user_cluster_comparisons']=read(gpt/'comparisons.json')
    if stats!=read(output/'statistics.json') or future!=read(output/'future.json'):raise ValueError('Seed statistics/Future publication differs')
    if read(output/'runs.json')!={n:{k:v for k,v in d.items() if k!='curve'} for n,d in training.items()}:raise ValueError('Training publication differs')
    expected_selections=dict(arms={n:dict(selected_update_step=d['selected_update_step'],selected_sha256=d['selected_sha256'],mapping_sha256=d['mapping_sha256'],
        probe_curve=[dict(step=v['step'],ndcg10=v['metrics']['full_denominator'][qg['selection_metric']]) for v in d['curve']]) for n,d in training.items()},lock_sha256=sha(gpt/'selection_locked.json'))
    if expected_selections!=read(output/'selections.json'):raise ValueError('Selection publication differs')
    total,steps,intents=total_counters(output)
    expected={**{'rq/'+c['name']:p['rq_steps'] for c in rq_cases(p)},**{'gpt/'+c['name']:p['gpt_steps'] for c in gpt_cases(p)}}
    if intents or steps!=expected or total!=p['optimizer_step_cap'] or total!=r['optimizer_steps_spent'] or steps!=r['stage_step_counts']:
        raise ValueError('Global40000-step prospective accounting differs')
    if r['future_names']!=[c['name'] for c in gpt_cases(p)] or len(future)!=p['future_passes']:raise ValueError('Missing full Future pass')
    primary=stats['groups']['full_denominator'];paired=stats['paired']['full_denominator']['historical_minus_plain'][p['primary_metric']]
    lines=['### Five-seed CF/LETTER Update and Future comparison','',
        f'Job `{job_id}`; code `{r["code_sha"]}`. Seeds {p["seeds"]}; family steps={total}, full Future passes={len(future)}. Codebook beta1=0.9; fixed accepted C0/E1/Base GPT. Pilot2026 and beta1=0 remain historical, excluded from this mean.',
        '', '| Method | Future Recall@10 mean | Seed95%CI | Future NDCG@10 mean | Seed95%CI |', '| --- | ---: | --- | ---: | --- |']
    for method,v in primary.items():
        a=v['recall@10'];b=v['ndcg@10'];lines.append(f'| {method} | {100*a["mean"]:.6f}% | [{100*a["ci95"][0]:.6f}%, {100*a["ci95"][1]:.6f}%] | {b["mean"]:.9f} | [{b["ci95"][0]:.9f}, {b["ci95"][1]:.9f}] |')
    lines+=['', 'Primary paired Historical-minus-Plain Future NDCG@10: `'+json.dumps(paired)+'`.',
        '', 'Student-t95%CI across five seeds (df4), sample SD, equal seed weighting; paired differences are computed within each seed first. Per-seed user-cluster bootstrap intervals are separate evidence. Future has already been observed; this is fixed-recipe repeatability, conditional on one fixed Base/E1, not a fresh blind test or full-pipeline seed study.',
        '', 'Acceptance rechecked all per-update RQ hard SID/suffix evidence and monitoring checkpoints on the original production backend, frozen original targets and teacher likelihoods; all GPT reset/sampling/selection and Future raw queries/paired bootstrap/statistics were independently checked from saved artifacts. Inputs are unchanged. Actual2000 GPT steps and validation-selected steps remain separate. No H convergence claim, parameter selection on Future, additional seed or automatic next task.']
    return dict(status='verified',job_id=job_id,contract_sha256=c['contract_sha256'],actual_optimizer_steps=total,report_markdown='\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();save(a.output,verify(a.context))
