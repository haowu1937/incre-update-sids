"""Read-only Update-validation LR acceptance, exact pairing and CPU GPT likelihoods."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,safe_file,gpt_arrays
from genrec.cf_letter_gpt_lr_protocol import JOB,SCOPE,ARTIFACTS,OLD_CODE,SOURCE_CODE,plan,cases,settings,check_source,source_files,baseline_result,curves_data,comparisons
from genrec.cf_letter_downstream_protocol import load_training,load_arrays,aggregate,select,check_mapping
from run_cf_letter_downstream import counters
ROOT=Path(__file__).resolve().parent


def verify(context,*,configuration=None,recipe=None):
    c=read(context);p=plan(ROOT) if configuration is None else configuration;q=settings(ROOT,p) if recipe is None else recipe
    output=Path(c['workspace'])/'outputs/cf-gpt-lr';r=read(output/'compact.json');root=Path(c['workspace']).parent.parent
    if (r['stage']!=SCOPE or r['job_id']!=JOB or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed'
        or not r['original_inputs_unchanged'] or r['uncertain_step_records'] or r['stop_new_stages_reason']
        or r['new_lr']!=p['new_lr'] or r['baseline_lr']!=p['baseline_lr'] or r['seeds']!=p['seeds']
        or any(r[k] for k in ('quantizer_training_performed','sasrec_training_performed','base_training_performed','holdout_evaluated','future_evaluated'))):raise ValueError('LR scope/status differs')
    if set(r['returned_artifact_sha256'])!=set(ARTIFACTS)-{'compact'}:raise ValueError('Missing LR publication')
    for n,h in r['returned_artifact_sha256'].items():
        if sha(safe_file(output,n+'.json'))!=h:raise ValueError('Publication hash differs')
    if sha(output/'local_artifact_index.json')!=r['local_artifact_index_sha256']:raise ValueError('Local index differs')
    cache={}
    for n,h in read(output/'local_artifact_index.json').items():
        f=safe_file(output,n);st=f.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha(f)
        if cache[key]!=h:raise ValueError('Local model/query record differs')
    source,index,bound=check_source(root,p)
    expected_bindings={dest:index[name] for name,dest in source_files(p).items()}
    if read(output/'source_bindings.json')!=expected_bindings:raise ValueError('Source whitelist differs')
    for name,h in expected_bindings.items():
        if sha(safe_file(output,name))!=h:raise ValueError('Copied source changed')
    audit=read(output/'data_audit.json')
    for f,h in audit['input_sha256'].items():
        if sha(f)!=h:raise ValueError('Original input changed')
    if any('future_test.jsonl' in f or 'anchor_holdout' in f for f in audit['input_sha256']):raise ValueError('Forbidden data bound in LR task')
    deployment=next(f for f in c['contract']['input_sha256'] if f.endswith('/main-deployment.json'))
    train,val,probe,binding=load_training(deployment,q)
    for k in ('all_update_train_queries','eligible_update_train_queries','excluded_update_train_queries','probe_sample_ids','catalog_items','training_sample_ids_sha256'):
        if audit[k]!=binding[k]:raise ValueError('Fitting or fixed selection cohort changed')
    arrays=load_arrays(output/'prepared.npz')
    for tag,values in [('train',train),('val',val)]:
        if any(not np.array_equal(arrays[tag+'_'+k],v) for k,v in values.items()):raise ValueError('Prepared data differs')
    if not np.array_equal(arrays['probe'],probe) or sha(output/'prepared.npz')!=audit['prepared_sha256']:raise ValueError('Prepared probe differs')
    if any(output.rglob('future_queries.jsonl')) or any(output.rglob('future_prepared.npz')):raise ValueError('Unexpected Future output')
    matrix=cases(p);names=[v['name'] for v in matrix]
    if read(output/'launch_journal.json')!=matrix:raise ValueError('Prospective LR claims differ')
    total,steps,intents=counters(output)
    if (total!=p['optimizer_step_cap'] or total!=r['optimizer_steps_spent'] or intents or steps!={n:p['steps_per_arm'] for n in names}
        or steps!=r['stage_step_counts'] or set(output.rglob('optimizer_steps.json'))!={output/'arms'/n/'optimizer_steps.json' for n in names}):raise ValueError('New-only LR budget differs')
    training={n:read(output/'arms'/n/'summary.json') for n in names};baseline={c['name']:baseline_result(output,c,p,q) for c in matrix}
    if read(output/'baseline.json')!=baseline or read(output/'runs.json')!={n:{k:v for k,v in d.items() if k!='curve'} for n,d in training.items()}:raise ValueError('Summary publication differs')
    if curves_data(output,p)!=read(output/'curves.json') or comparisons(baseline,training,p)!=read(output/'comparisons.json'):raise ValueError('Curves/paired seed arithmetic differs')
    import torch
    from genrec.cf_letter_base_training import Sampler
    from genrec.cf_letter_downstream_training import model_from
    torch.set_num_threads(1);base=torch.load(output/'inputs/base/gpt/selected.pth',map_location='cpu',weights_only=False)
    if base['code_sha']!=q['predecessor']['code_sha'] or base['step']!=q['predecessor']['gpt_step']:raise ValueError('Base identity differs')
    def rows(path,indices,expected):
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if len(records)!=len(indices):raise ValueError('Incomplete Update evaluation')
        for row,i in zip(records,indices):
            if (row['sample_id']!=str(val['samples'][i]) or row['source_row']!=int(val['source_rows'][i]) or row['user']!=str(val['raw_users'][i])
                or row['target']!=int(val['targets'][i]) or row['all_history_unavailable']!=bool(np.all(val['history'][i]==0))):raise ValueError('Update query identity or mask differs')
        if aggregate(records,q)!=expected:raise ValueError('Raw Update aggregation differs')
        return records
    exposures=[]
    for case in matrix:
        name=case['name'];mapping=np.load(output/'mappings'/(name+'.npy'),allow_pickle=False);check_mapping(mapping,q)
        new_trace=None
        for label,folder,lr,summary,expected_code in [
            ('new',output/'arms'/name,p['new_lr'],training[name],r['code_sha']),
            ('baseline',output/'baseline/arms'/name,p['baseline_lr'],read(output/'baseline/arms'/name/'summary.json'),p.get('fixture_source_code',SOURCE_CODE if case['seed']==2031 else OLD_CODE))]:
            initial=torch.load(folder/'initial.pth',map_location='cpu',weights_only=False)
            if initial['step']!=0 or initial['code_sha']!=expected_code or initial['optimizer']['state'] or any(not torch.equal(v,initial['model'][k]) for k,v in base['model'].items()):raise ValueError('Warm start/reset differs')
            group=initial['optimizer']['param_groups'][0]
            if group['lr']!=lr or group['weight_decay']!=q['weight_decay'] or list(group['betas'])!=q['betas'] or group['eps']!=q['eps']:raise ValueError('Optimizer recipe differs')
            trace=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            expected_len=p['steps_per_arm'] if label=='new' else p['baseline_original_steps']
            if len(trace)!=expected_len:raise ValueError('Training trace incomplete')
            prefix=trace[:p['steps_per_arm']];sampler=Sampler(len(train['targets']),q['batch_size'],case['seed'])
            for i,row in enumerate(prefix,1):
                ids,_=sampler.take()
                if (row['step']!=i or row['source_rows']!=train['source_rows'][ids].tolist() or row['epoch']!=sampler.epochs
                    or row['processed_examples']!=sampler.processed or not math.isfinite(row['token_ce']) or not math.isfinite(row['gradient_norm'])):raise ValueError('Paired sampling/trace differs')
            if label=='new':new_trace=prefix;exposures.append(sampler.processed)
            elif [v['source_rows'] for v in prefix]!=[v['source_rows'] for v in new_trace]:raise ValueError('Old/new fitting order differs')
            curve=[v for v in summary['curve'] if v['step']<=p['steps_per_arm']];selected=select(curve,q)
            for point in curve:
                rows(folder/f'probe_{point["step"]}.jsonl',probe,point['metrics'])
                if sha(folder/f'candidate_{point["step"]}.pth')!=point['checkpoint_sha256']:raise ValueError('Candidate hash differs')
            if (summary['selected_update_step']!=selected['step'] or summary['selected_probe']!=selected['metrics'] or summary['selected_sha256']!=selected['checkpoint_sha256']
                or sha(folder/'selected.pth')!=summary['selected_sha256'] or sha(output/'mappings'/(name+'.npy'))!=summary['mapping_sha256']):raise ValueError('Earliest-tie selection/map differs')
            full=rows(folder/'full_update_validation.jsonl',np.arange(len(val['targets'])),summary['full_update_validation'])
            model,saved=model_from(folder/'selected.pth',q,'cpu');model.eval();model.requires_grad_(False)
            if saved['code_sha']!=expected_code or saved['step']!=selected['step'] or saved['mapping_sha256']!=summary['mapping_sha256']:raise ValueError('Selected metadata differs')
            ids=np.flatnonzero(val['targets']>0)[:16];x,m,y=gpt_arrays(val,mapping,q['quantizer']['capacity'],q['gpt']['user_buckets'])
            with torch.no_grad():nll=model.nll(torch.tensor(x[ids]),torch.tensor(m[ids]),torch.tensor(y[ids])).numpy()
            if not np.allclose(nll,[full[i]['nll'] for i in ids],rtol=2e-5,atol=2e-4):raise ValueError('Independent CPU Update NLL differs')
            if label=='new':
                last=torch.load(folder/'last.pth',map_location='cpu',weights_only=False)
                if summary['case']!=case or last['code_sha']!=r['code_sha'] or last['step']!=p['steps_per_arm'] or last['sampler']['processed']!=sampler.processed or summary['processed_examples']!=sampler.processed:raise ValueError('New endpoint accounting differs')
            del model,saved,initial
    if len(set(exposures))!=1:raise ValueError('Unequal paired exposure')
    check_source(root,p)
    stats=read(output/'comparisons.json');groups=stats['groups']['full_update_validation']['full_denominator'];cutoff=int(q['selection_metric'].split('@')[1]);lines=[
        '### GPT learning-rate ablation: Update validation only','',
        f'Job `{JOB}`; code `{r["code_sha"]}`. New GPT updates={total}; five paired seeds2027-2031, frozen accepted Plain/Historical maps. lr3e-5 trains1000 steps per arm; lr1e-4 reuses the first1000 steps of the accepted2000-step trajectories. No RQ/Base/SASRec training or new Future evaluation.',
        '', f'| GPT lr / method | Update-val Recall@{cutoff} mean | Update-val NDCG@{cutoff} mean | NDCG seed95%CI |', '| --- | ---: | ---: | --- |']
    for name,v in groups.items():
        n=v[f'ndcg@{cutoff}'];lines.append(f'| {name} | {100*v[f"recall@{cutoff}"]["mean"]:.6f}% | {n["mean"]:.9f} | [{n["ci95"][0]:.9f}, {n["ci95"][1]:.9f}] |')
    lines+=['','Each checkpoint is selected only on the same fixed512-query Update-validation NDCG@10 probe at0/100/250/500/1000, earliest tie. Full3280 validation is reporting only and overlaps that probe; it is not an independent test. Report all paired seed intervals and loss/probe curves, without automatically choosing a new learning rate. This dataset’s Future was viewed previously; new downstream claims require a separately authorized fixed-parameter evaluation. No positive-result gate, additional parameter sweep, seed or automatic next task.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=total,report_markdown='\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();save(a.output,verify(a.context))
