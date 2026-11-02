"""Read-only canonical RQ/Hungarian, TIGER sampling and Future acceptance."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,safe_file
from genrec.hungarian_sids import canonical
from genrec.tiger_hungarian_protocol import JOB,SCOPE,NAMES,plan,cases,source,project_values
from run_cf_letter_downstream import counters


def close(a,b):
    if isinstance(b,dict):
        for k,v in b.items():close(a[k],v)
    elif isinstance(b,list):
        if len(a)!=len(b):raise ValueError('Different evidence length')
        for x,y in zip(a,b):close(x,y)
    elif isinstance(b,float):
        if not math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-5):raise ValueError('Measured scalar differs')
    elif a!=b:raise ValueError('Evidence identity differs')


def verify(context_path,*,configuration=None,fixture=False):
    c=read(context_path);p=plan() if configuration is None else configuration;output=Path(c['workspace'])/'outputs/tiger-hungarian';r=read(output/'compact.json')
    if r['execution_status']!='completed' or r['job_id']!=p['job_id'] or r['code_sha']!=c['request']['code_sha'] or r['future_passes_new']!=3 or not r['original_inputs_unchanged'] or not r['selection_locked_before_new_future']:raise ValueError('Task identity/status differs')
    if any(r[k] for k in ('rqvae_training_performed','base_tiger_training_performed','holdout_evaluated','future_used_for_matching_or_selection')):raise ValueError('Unauthorized fitting/data use')
    if set(r['returned_artifact_sha256'])!=set(NAMES)-{'compact'}:raise ValueError('Missing publication')
    for name,h in r['returned_artifact_sha256'].items():
        path=output/(name+'.json')
        if sha(path)!=h or path.stat().st_size>65536:raise ValueError('Publication hash/size differs')
    if sha(output/'local_artifact_index.json')!=r['local_artifact_index_sha256']:raise ValueError('Index changed')
    public_training,public_future=project_values(output,p)
    if read(output/'training.json')!=public_training or read(output/'future.json')!=public_future:raise ValueError('Public/local summaries differ')
    cache={}
    for name,h in read(output/'local_artifact_index.json').items():
        path=safe_file(output,name);st=path.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha(path)
        if cache[key]!=h:raise ValueError('Indexed model/raw record changed')
    audit=read(output/'audit.json');full=read(output/'audit_full.json')
    if sha(output/'audit_full.json')!=audit['full_audit_sha256'] or canonical(full['input_sha256'])!=audit['input_manifest_sha256'] or canonical(full['probe_sample_ids'])!=audit['probe_sample_ids_sha256']:raise ValueError('Bounded/full audit differs')
    for path,h in full['input_sha256'].items():
        if sha(path)!=h:raise ValueError('Original input changed')
    deployment=next(v for v in c['contract']['input_sha256'] if v.endswith('/main-deployment.json'));root=Path(read(deployment)['runner_root'])
    if not fixture:source(root,p)
    import torch
    from genrec import tiger_hungarian_experiment as e
    from genrec import future_evaluation as f
    from genrec.future_protocol import summarize_records
    from genrec.temporal_tiger import make_batch
    device=torch.device('cpu' if fixture else 'cuda:0');torch.set_num_threads(1)
    ex,directory,_=e.context(deployment,p);raw,manifest,registries,capacities=e.build_maps(ex,directory,device)
    maps,mapping=e.mapped_versions(raw,{k:list(v) for k,v in ex.tokenizer.item2tokens.items()},capacities)
    saved_mapping=read(output/'mapping.json')
    if saved_mapping['translations']!=mapping or saved_mapping['fit_items']!=len(ex.tokenizer.item2tokens) or saved_mapping['fit_catalog']!='old Base train only':raise ValueError('Hungarian cooccurrence/bijection differs')
    for name,value in maps.items():
        for stage,mapping in value.items():
            if mapping!=read(output/'maps'/(name+'_'+stage+'.json')) or sha(output/'maps'/(name+'_'+stage+'.json'))!=saved_mapping['maps_sha256'][name][stage]:raise ValueError('Translated map differs')
    config=read(output/'config.json')
    expected=read(Path(read(deployment)['future_dir'])/'config.json')['config']
    if {k:v for k,v in config.items() if k!='device'}!={k:v for k,v in expected.items() if k!='device'}:raise ValueError('Original TIGER recipe differs')
    for name in p['arms']:
        token=e.tokenizer(output,name,'train');data,train,val,base_ids,probe=e.splits(ex,token,config)
        if canonical(probe)!=audit['probe_sample_ids_sha256'] or canonical([v['sample_id'] for v in train])!=audit['train_sample_ids_sha256']:raise ValueError('Fitting/selection samples changed')
    # Recompute all fixed-original-target anchor H on canonical mappings.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        measured=e.anchor_scores(ex,maps,config,Path(tmp),device)
        for name,value in measured.items():
            for key in ('queries','H','mean_delta_nll','mean_old_nll','mean_new_nll'):close(read(output/'anchors.json')[name][key],value[key])
    total,steps,intents=counters(output)
    if total!=p['optimizer_step_cap'] or total!=r['optimizer_steps_spent'] or steps!={name:p['steps_per_arm'] for name in p['new_training_arms']} or intents or steps!=r['stage_step_counts']:raise ValueError('New-only budget differs')
    if read(output/'launch_journal.json')!=cases(p) or set(output.rglob('optimizer_steps.json'))!={output/'arms'/n/'optimizer_steps.json' for n in p['new_training_arms']}:raise ValueError('Missing/extra optimizer reservation')
    lock=read(output/'selection_locked.json')
    if list(lock['arms'])!=p['arms'] or lock['future_used_for_selection']:raise ValueError('Global lock differs')
    samples=[];all_records={};training={};future={}
    future_rows=f.load_stage(ex,'future_test',base_ids+[v['sample_id'] for v in train+val])
    expected_rows=[(i,v['sample_id'],v['user'],v['item_seq'][-1],v['target_servable']) for i,v in enumerate(future_rows)]
    for name in p['arms']:
        folder=output/('reused' if name=='historical_unmapped' else 'arms')/name;s=read(folder/'summary.json');training[name]=s
        if s['update_optimizer_steps']!=p['steps_per_arm'] or not s['exact_base_warm_start'] or not s['selected_checkpoint_reload_equal']:raise ValueError('Training budget/warmstart differs')
        curve=s['validation_history']
        if [v['step'] for v in curve]!=p['selection_steps']:raise ValueError('Selection candidates differ')
        chosen=max(curve,key=lambda v:v['metrics']['full_denominator'][config['selection_metric']])
        if chosen['step']!=s['best_update_optimizer_step'] or chosen['metrics']!=s['best_update_validation']:raise ValueError('Not earliest maximum probe')
        entry=lock['arms'][name]
        if sha(folder/'best.pth')!=entry['selected_sha256'] or entry['selected_step']!=chosen['step'] or sha(output/'maps'/(name+'_deployment.json'))!=entry['mapping_sha256']:raise ValueError('Selected lock changed')
        last=torch.load(folder/'last.pth',map_location='cpu',weights_only=False);selected=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
        if last['sampler']['optimizer_steps']!=p['steps_per_arm'] or last['validation_history']!=curve or selected['optimizer_step']!=chosen['step'] or selected['config']!=config:raise ValueError('Actual checkpoint/selection metadata differs')
        if name!='historical_unmapped':
            initial=torch.load(folder/'initial.pth',map_location='cpu',weights_only=False)
            if initial['optimizer']['state'] or any(not torch.equal(v,initial['state_dict'][k]) for k,v in ex.teacher['state_dict'].items()):raise ValueError('Initial Base weights/reset differ')
            group=initial['optimizer']['param_groups'][0]
            if group['initial_lr']!=config['lr'] or group['weight_decay']!=config['weight_decay'] or list(group['betas'])!=[.9,.999] or group['eps']!=1e-8:raise ValueError('Optimizer recipe differs')
            trace=[json.loads(v) for v in (folder/'sampling.jsonl').read_text().splitlines()];loss=[json.loads(v) for v in (folder/'train_metrics.jsonl').read_text().splitlines()]
            if len(trace)!=p['steps_per_arm'] or [v['step'] for v in loss]!=list(range(1,p['steps_per_arm']+1)) or any(not math.isfinite(v['token_ce']) or not math.isfinite(v['gradient_norm']) for v in loss):raise ValueError('Training trace incomplete')
            token=e.tokenizer(output,name,'train');arrays=e.splits(ex,token,config)[0]['update_train']['arrays'];generator=torch.Generator().manual_seed(config['seed']+1);order=None;cursor=processed=epochs=0
            for step,row in enumerate(trace,1):
                if order is None or cursor==len(arrays['labels']):order=torch.randperm(len(arrays['labels']),generator=generator);cursor=0;epochs+=1
                end=min(cursor+config['batch_size'],len(order));ids=order[cursor:end].numpy();cursor=end;processed+=len(ids)
                if row!=dict(step=step,indices=ids.tolist(),source_rows=arrays['source_rows'][ids].tolist()):raise ValueError('Actual paired sampling order differs')
            if last['sampler']['processed_examples']!=processed or last['sampler']['cursor']!=cursor or last['sampler']['epochs_started']!=epochs or not torch.equal(last['sampler']['order'],order) or not torch.equal(last['generator'],generator.get_state()) or s['processed_examples']!=processed:raise ValueError('Sampler endpoint differs')
            samples.append(last)
        token=e.tokenizer(output,name,'train');probe_split=e.splits(ex,token,config)[0]['update_val'];model=f.TIGER(config,None,token).to(device);model.load_state_dict(selected['state_dict'],strict=True);model.eval().requires_grad_(False)
        close(f.rank_validation(model,config,token,probe_split,device),s['best_update_validation']);del model
        records=[json.loads(v) for v in (folder/'future_queries.jsonl').read_text().splitlines()];all_records[name]=records
        if [(v['source_row'],v['sample_id'],v['user'],v['target_raw_item_id'],v['target_servable']) for v in records]!=expected_rows:raise ValueError('Future identities/availability differ')
        keys=[f'{m}@{k}' for m in config['metrics'] for k in config['topk']];summary=read(folder/'future_summary.json');future[name]=summary
        close(summary,summarize_records(records,keys))
        if name!='historical_unmapped':
            predictions=[json.loads(v) for v in (folder/'future_predictions.jsonl').read_text().splitlines()]
            if [v['source_row'] for v in predictions]!=[i for i,v in enumerate(records) if v['target_servable']]:raise ValueError('Prediction sidecar availability differs')
            catalog=set(maps[name]['deployment'])
            for row in predictions:
                record=records[row['source_row']];pred=row['predictions'];target=record['target_raw_item_id']
                if len(pred)!=max(config['topk']) or len(set(pred))!=len(pred) or not set(pred)<=catalog:raise ValueError('Illegal/duplicate predictions')
                for k in config['topk']:
                    hit=target in pred[:k];expected_ndcg=1/math.log2(pred.index(target)+2) if hit else 0.
                    close(record['metrics'][f'recall@{k}'],float(hit));close(record['metrics'][f'ndcg@{k}'],expected_ndcg)
            if (folder/'future_queries.jsonl').stat().st_mtime< (output/'selection_locked.json').stat().st_mtime:raise ValueError('New Future precedes global selection lock')
        # Independent CPU likelihoods for fixed served Future samples.
        token=e.tokenizer(output,name,'deployment');arrays=f.encode(future_rows,token,registries['update'])['arrays'];ids=np.arange(min(16,len(arrays['labels'])))
        model=f.TIGER(config,None,token).to('cpu');model.load_state_dict(selected['state_dict'],strict=True);model.eval().requires_grad_(False)
        batch=make_batch(arrays,ids,torch.device('cpu'))
        with torch.no_grad():nll=torch.nn.functional.cross_entropy(model(batch).logits.float().transpose(1,2),batch['labels'],reduction='none').sum(1).numpy()
        if not np.allclose(nll,[records[int(arrays['source_rows'][i])]['sequence_nll'] for i in ids],rtol=2e-5,atol=2e-4):raise ValueError('Independent CPU TIGER NLL differs')
        del model,selected
    # The reused accepted control has the same actual fitting exposure/order end.
    reused=torch.load(output/'reused/historical_unmapped/last.pth',map_location='cpu',weights_only=False)
    for last in samples:
        for key in ('optimizer_steps','cursor','processed_examples','epochs_started'):
            if last['sampler'][key]!=reused['sampler'][key]:raise ValueError('Matched historical/new exposure differs')
        if not torch.equal(last['sampler']['order'],reused['sampler']['order']) or not torch.equal(last['generator'],reused['generator']):raise ValueError('Historical/new sampler endpoint differs')
    if e.comparisons(output,p)!=read(output/'comparisons.json'):raise ValueError('Paired user-bootstrap differs')
    for path,h in full['input_sha256'].items():
        if sha(path)!=h:raise ValueError('Source changed during verification')
    cutoff=10 if 10 in config['topk'] else max(config['topk'])
    lines=['### TIGER post-codebook Hungarian experiment','',f'Job`{p["job_id"]}`; code`{r["code_sha"]}`. New TIGER updates={total}; reused historical-unmapped{p["steps_per_arm"]}-step control. RQ weights frozen; seed2026. Full Future denominator includes unavailable targets as misses.','',
        f'| Method | Actual TIGER steps | Selected step | Future Recall@{cutoff} | Future NDCG@{cutoff} |','| --- | ---: | ---: | ---: | ---: |']
    for name in p['arms']:
        m=future[name]['full_denominator'];lines.append(f'| {name} | {p["steps_per_arm"]} | {training[name]["best_update_optimizer_step"]} | {100*m[f"recall@{cutoff}"]:.6f}% | {m[f"ndcg@{cutoff}"]:.9f} |')
    lines+=['','Same fixed Update-validation probe/earliest-tie selection; all four selected models/maps locked before three new Future passes. Hungarian maximizes Base-train item token cooccurrence, with a bijective suffix rename and no suffix reallocation. Training-side anchor H uses old targets; downstream history and targets both use current-arm mappings. Paired500-draw user-cluster intervals are single-seed query uncertainty. Future had been viewed previously; no configuration tuning, LETTER work or automatic next task.']
    return dict(status='verified',job_id=p['job_id'],contract_sha256=c['contract_sha256'],actual_optimizer_steps=total,report_markdown='\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    # Installed generic verifier masks CUDA by default. This task explicitly
    # authorizes strict canonical read-only GPU reload, after foreign GPU wait.
    from run_cf_letter_base import gpu
    began=time.monotonic()
    while gpu()[0]:
        if time.monotonic()-began>72000:raise RuntimeError('Foreign GPU still busy; no termination')
        time.sleep(2)
    os.environ['CUDA_VISIBLE_DEVICES']='0';save(args.output,verify(args.context))
