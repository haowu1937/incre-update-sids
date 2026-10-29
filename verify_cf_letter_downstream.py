"""Read-only five-arm scientific acceptance and report; never selects on Future."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,safe_file,gpt_arrays
from genrec.cf_letter_downstream_protocol import JOB,SCOPE,plan,cases,load_training,load_stage,valid_lock,check_mapping,select,aggregate,paired,load_arrays,future_table,verify_zero_step_failure
from run_cf_letter_downstream import counters
ROOT=Path(__file__).resolve().parent


def verify(context):
    c=read(context);output=Path(c['workspace'])/'outputs/cf-downstream';p=plan(ROOT);r=read(output/'compact.json')
    if (r['stage']!=SCOPE or r['job_id']!=JOB or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed'
        or not r['original_inputs_unchanged'] or not r['selection_locked_before_future'] or r['future_used_for_selection'] is not False
        or any(r[k] is not False for k in ('quantizer_training_performed','sasrec_training_performed','holdout_evaluated'))):raise ValueError('Downstream identity/scope differs')
    for name,value in r['returned_artifact_sha256'].items():
        if sha(safe_file(output,name+'.json'))!=value:raise ValueError('Returned downstream artifact differs')
    if sha(output/'local_artifact_index.json')!=r['local_artifact_index_sha256']:raise ValueError('Local index changed')
    cache={}
    for name,value in read(output/'local_artifact_index.json').items():
        path=safe_file(output,name);stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in cache:cache[key]=sha(path)
        if cache[key]!=value:raise ValueError('Local checkpoint/query/array changed')
    audit=read(output/'data_audit.json')
    for path,value in audit['input_sha256'].items():
        if sha(path)!=value:raise ValueError('Original source changed')
    if 'zero_step_repair_of' in p:
        root=Path(c['workspace']).parent.parent
        verify_zero_step_failure(root,p)
    for name in ('gpt/selected.pth','item_sids.npy'):
        if sha(output/'inputs/base'/name)!=p['predecessor']['files_sha256'][name]:raise ValueError('Copied Base changed')
    pairdir=output/'inputs/pair'
    if sha(pairdir/'compact.json')!=p['pair_predecessor']['compact_sha256'] or sha(pairdir/'local_artifact_index.json')!=p['pair_predecessor']['index_sha256']:raise ValueError('Paired source identity changed')
    source_index=read(pairdir/'local_artifact_index.json');old_map=np.load(output/'inputs/base/item_sids.npy',allow_pickle=False)
    from genrec.cf_letter_refresh_protocol import versioned_mapping
    for case in cases(p):
        m=np.load(output/'mappings'/(case['name']+'.npy'),allow_pickle=False);check_mapping(m,p)
        if case['source'] is None:
            if not np.array_equal(m,old_map):raise ValueError('FT-old remapped old SIDs')
        else:
            prefix='arms/'+case['source']+'/'+case['source']+'/'
            for name in ('item_sids_1000.npy','semantic_codes_1000.npy','last.pth'):
                if sha(pairdir/case['source']/name)!=source_index[prefix+name]:raise ValueError('Copied1000-step source changed')
            expected,capacity=versioned_mapping(old_map,np.load(pairdir/case['source']/'semantic_codes_1000.npy',allow_pickle=False),p['quantizer']['capacity'])
            if not capacity['full_sid_allocation_feasible'] or not np.array_equal(m,expected):raise ValueError('Current hard SID differs')
    deployment=next(v for v in c['contract']['input_sha256'] if v.endswith('/main-deployment.json'))
    train,val,probe,binding=load_training(deployment,p)
    if any(audit[k]!=binding[k] for k in ('all_update_train_queries','eligible_update_train_queries','excluded_update_train_queries','probe_sample_ids','catalog_items','training_sample_ids_sha256')):raise ValueError('Common fitting/probe cohort changed')
    prepared=load_arrays(output/'prepared.npz')
    for label,arrays in [('train',train),('val',val)]:
        if any(not np.array_equal(prepared[label+'_'+k],v) for k,v in arrays.items()):raise ValueError('Prepared arrays changed')
    if not np.array_equal(prepared['probe'],probe):raise ValueError('Prepared probe changed')
    lock=read(output/'selection_locked.json');valid_lock(lock,p);lock_sha=sha(output/'selection_locked.json')
    opened=read(output/'future_opened.json')
    if lock.get('code_sha')!=r['code_sha'] or lock_sha!=r['selection_lock_sha256'] or opened['selection_lock_sha256']!=lock_sha or opened['time']<lock['created_at']:raise ValueError('Future preceded global lock')
    future,_=load_stage(deployment,p,'future_test',lock)
    if set(future['samples'])&(set(train['samples'])|set(val['samples'])):raise ValueError('Future overlaps fitting/selection')
    future_saved=load_arrays(output/'future_prepared.npz')
    if any(not np.array_equal(future_saved[k],v) for k,v in future.items()):raise ValueError('Future encoding changed')
    training=read(output/'training.json');fut=read(output/'future.json');names=[v['name'] for v in cases(p)]
    if list(training)!=names or list(fut)!=names or read(output/'launch_journal.json')!=cases(p):raise ValueError('Five-arm order or claims differ')
    import torch
    from genrec.cf_letter_base_training import Sampler
    from genrec.cf_letter_downstream_training import model_from
    torch.set_num_threads(1);base=torch.load(output/'inputs/base/gpt/selected.pth',map_location='cpu',weights_only=False)
    if base['code_sha']!=p['predecessor']['code_sha'] or base['step']!=p['predecessor']['gpt_step']:raise ValueError('Wrong frozen Base initialization')
    def records(path,arrays,indices):
        rows=[json.loads(line) for line in path.read_text().splitlines()];expected=np.asarray(indices)
        if len(rows)!=len(expected):raise ValueError('Missing evaluation queries')
        for row,i in zip(rows,expected):
            if (row['source_row']!=int(arrays['source_rows'][i]) or row['sample_id']!=str(arrays['samples'][i]) or row['user']!=str(arrays['raw_users'][i])
                or row['target']!=int(arrays['targets'][i]) or row['all_history_unavailable']!=bool(np.all(arrays['history'][i]==0))):raise ValueError('Raw query identity/mask differs')
        return rows,aggregate(rows,p)
    future_rows={};processed=[]
    for case in cases(p):
        name=case['name'];directory=output/'arms'/name;summary=training[name]
        if summary!=read(directory/'summary.json') or summary['case']!=case:raise ValueError('Training publication differs')
        initial=torch.load(directory/'initial.pth',map_location='cpu',weights_only=False)
        if initial['step']!=0 or initial['optimizer']['state'] or any(not torch.equal(initial['model'][k],v) for k,v in base['model'].items()):raise ValueError('Warm start/reset differs')
        group=initial['optimizer']['param_groups'][0]
        if group['lr']!=p['lr'] or group['weight_decay']!=p['weight_decay'] or list(group['betas'])!=p['betas'] or group['eps']!=p['eps']:raise ValueError('GPT optimizer differs')
        sampler=Sampler(len(train['targets']),p['batch_size'],p['seed']);trace=[json.loads(v) for v in (directory/'metrics.jsonl').read_text().splitlines()]
        if len(trace)!=p['steps_per_arm']:raise ValueError('Training trace incomplete')
        for step,row in enumerate(trace,1):
            ids,_=sampler.take()
            if (row['step']!=step or row['source_rows']!=train['source_rows'][ids].tolist() or row['epoch']!=sampler.epochs
                or row['processed_examples']!=sampler.processed or not math.isfinite(row['token_ce']) or not math.isfinite(row['gradient_norm'])):raise ValueError('Paired fitting order/trace differs')
        curve=summary['curve'];best=select(curve,p)
        for point in curve:
            rows,metric=records(directory/('probe_'+str(point['step'])+'.jsonl'),val,probe)
            if metric!=point['metrics'] or sha(directory/('candidate_'+str(point['step'])+'.pth'))!=point['checkpoint_sha256']:raise ValueError('Probe selection arithmetic/hash differs')
        selected=directory/'selected.pth';mapping=np.load(output/'mappings'/(name+'.npy'),allow_pickle=False)
        if (best['step']!=summary['selected_update_step'] or summary['selected_probe']!=best['metrics']
            or sha(selected)!=best['checkpoint_sha256'] or sha(selected)!=summary['selected_sha256']
            or sha(output/'mappings'/(name+'.npy'))!=summary['mapping_sha256']
            or lock['arms'][name]!=dict(selected_step=best['step'],selected_sha256=summary['selected_sha256'],mapping_sha256=summary['mapping_sha256'])):raise ValueError('Locked selection differs')
        last=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
        if last['step']!=p['steps_per_arm'] or last['sampler']['processed']!=sampler.processed or summary['processed_examples']!=sampler.processed:raise ValueError('GPT endpoint accounting differs')
        processed.append(sampler.processed)
        _,metric=records(directory/'full_update_validation.jsonl',val,np.arange(len(val['targets'])))
        if metric!=summary['full_update_validation']:raise ValueError('Full Update metrics differ')
        rows,metric=records(directory/'future_queries.jsonl',future,np.arange(len(future['targets'])));future_rows[name]=rows
        if any(fut[name][k]!=value for k,value in metric.items()) or fut[name]!=read(directory/'future_summary.json'):raise ValueError('Future aggregation differs')
        if (fut[name]['selection_lock_sha256']!=lock_sha or fut[name]['model_sha256']!=sha(selected) or fut[name]['mapping_sha256']!=summary['mapping_sha256']
            or fut[name]['selected_update_step']!=best['step'] or fut[name]['selection_locked_before_future'] is not True):raise ValueError('Future used wrong checkpoint/map')
        model,saved=model_from(selected,p,'cpu');model.eval();model.requires_grad_(False)
        if saved['step']!=best['step'] or saved['code_sha']!=r['code_sha'] or saved['mapping_sha256']!=summary['mapping_sha256']:raise ValueError('Selected checkpoint metadata differs')
        # Recompute fixed likelihoods with current-arm history AND target SID.
        for arrays,path in [(val,directory/'full_update_validation.jsonl'),(future,directory/'future_queries.jsonl')]:
            indices=np.flatnonzero(arrays['targets']>0)[:16];x,m,y=gpt_arrays(arrays,mapping,p['quantizer']['capacity'],p['gpt']['user_buckets'])
            raw={q['sample_id']:q for q in map(json.loads,path.read_text().splitlines())}
            with torch.no_grad():actual=model.nll(torch.tensor(x[indices]),torch.tensor(m[indices]),torch.tensor(y[indices])).numpy()
            if not np.allclose(actual,[raw[str(arrays['samples'][i])]['nll'] for i in indices],rtol=2e-5,atol=2e-4):raise ValueError('Independent CPU selected-model NLL differs')
        del model,saved,last,initial
    if len(set(processed))!=1:raise ValueError('Unequal fitting exposure')
    total,steps,intents=counters(output)
    if intents or total!=p['optimizer_step_cap'] or total!=r['optimizer_steps_spent'] or steps!=r['stage_step_counts'] or any(v!=p['steps_per_arm'] for v in steps.values()):raise ValueError('Actual GPT budget differs')
    pairs={}
    for before,after in [('plain_b09','historical_b09'),('plain_b0','historical_b0'),*[('ft_old',name) for name in names if name!='ft_old']]:pairs[after+'_minus_'+before]=paired(future_rows[before],future_rows[after],p)
    if pairs!=read(output/'comparisons.json'):raise ValueError('Paired user bootstrap differs')
    lines=['### CF/GPT Update adaptation and locked Future evaluation','',f'Job `{JOB}`; code `{r["code_sha"]}`. Actual new GPT updates={total}; five independent arms, common I0={p["old_catalog_items"]} items, four current SID digits,20 raw history events. No RQ/SASRec fitting. All selections locked before Future.',
        '',*future_table(training,fut,p)]
    lines+=['','Recall values in this appended table are fractions. The controller maintains the separate CF/GPT block in the primary table with Recall percentages. These results are not directly interchangeable with the old Llama/TIGER extended-catalog protocol.',
        '', 'Each GPT restores the exact Base weights with reset AdamW, lr1e-4,wd0.01,beta=(0.9,0.999),batch256,clip1,seed2026. Full fixed512-query Update-validation NDCG@10 selects0/100/250/500/1000/2000, earliest tie; actual fitting always completes2000. Current-arm history and target use the same frozen full-SID mapping. All original Future queries remain in the full denominator; outside-I0 targets are misses, with common served metrics also saved. No seen-item filtering or Future tuning.',
        '', 'Primary paired Future comparisons (Historical minus matched Plain), user-cluster bootstrap95% intervals:',
        *['`'+key+'`: `'+json.dumps(pairs[key],sort_keys=True)+'`.' for key in ('historical_b09_minus_plain_b09','historical_b0_minus_plain_b0')],
        '', 'Acceptance checked raw metrics/query identities, map and source hashes, full fitting order and counters, global selection lock, earliest-tie selection and independent CPU selected-model likelihood samples. Single engineering seed; no convergence claim or cross-seed confirmation. Earlier negative RQ/H results and the old blocked screening receipt remain unchanged. Report and stop; no automatic next experiment.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=total,report_markdown='\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();save(a.output,verify(a.context))
