"""Task-local TIGER adapters; source models and original ledgers stay immutable."""
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
from genrec.cf_letter_protocol import read,save,sha
from genrec.hungarian_sids import fit,transform,canonical,bucket_sizes
from genrec.tiger_hungarian_protocol import source,audit_publication,bounded,projections


def context(deployment,p):
    import evaluate_historical_future as old
    d=read(deployment)
    return old.load_context(SimpleNamespace(e1_dir=d['e1_dir'],retuned_dir=d['retuned_dir']))


def build_maps(ex,directory,device):
    import torch
    from genrec import future_evaluation as f
    from gensid import historical_incremental_content as h
    feature_dir=Path(ex.plain['features_dir']);feature_report=f.read_json(feature_dir/'summary.json')
    f.verify_files(feature_dir,feature_report,['update/'+n for n in ('embeddings.npy','feature_rows.json','registry.json')])
    registries=f.read_json(Path(ex.e1['protocol_dir'])/'registries.json');manifest=f.read_json(Path(ex.e1['protocol_dir'])/'protocol_manifest.json')
    f.check_registry_prefixes(registries,manifest);registry=f.read_json(feature_dir/'update/registry.json')
    values=np.load(feature_dir/'update/embeddings.npy',allow_pickle=False);rows=f.read_json(feature_dir/'update/feature_rows.json');n=len(ex.features)
    expected=[dict(raw_item_id=raw,gr_item_id=i+1,gensid_global_item_id=i,feature_row=i) for i,raw in enumerate(registries['update']['id2item'][1:])]
    if registry!=registries['update'] or rows!=expected or not np.array_equal(values[:n],ex.features.numpy()) or not np.isfinite(values).all():raise ValueError('Deployment feature/registry prefix differs')
    out={}
    for arm in ('plain','historical'):
        folder=Path(directory)/arm;saved=read(folder/'summary.json');f.verify_files(folder,saved,['last.pth','allocation_registry.json','item2tokens.json','semantic_assignments.jsonl'])
        checkpoint=torch.load(folder/'last.pth',map_location='cpu',weights_only=False);model=h.build_model(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval().requires_grad_(False)
        codes=h.catalog_codes(model,ex).tolist()
        if f.read_jsonl(folder/'semantic_assignments.jsonl')!=[dict(**row,semantic_code=code) for row,code in zip(ex.feature_rows,codes)]:raise ValueError('Retuned canonical assignment differs')
        raw=registry['id2item'][1:];train=h.versioned_suffixes(ex.previous,raw[:n],codes)
        if train!=read(folder/'allocation_registry.json'):raise ValueError('Retuned full allocation differs')
        _,train_map=h.maps_from_ledger(train)
        if train_map!=read(folder/'item2tokens.json'):raise ValueError('Retuned train tokens differ')
        added=[];batch=ex.plain['config']['added_item_assignment_eval_batch_size']
        with torch.no_grad():
            for start in range(n,len(values),batch):added.extend(model.get_indices(torch.from_numpy(values[start:start+batch]).to(device),None,use_sk=False).cpu().tolist())
        deployment=h.versioned_suffixes(ex.previous,raw,codes+added)
        if deployment['entries'][:n]!=train['entries']:raise ValueError('Extension changed training SIDs')
        _,deployment_map=h.maps_from_ledger(deployment)
        out[arm]=dict(train={k:list(v) for k,v in train_map.items()},deployment={k:list(v) for k,v in deployment_map.items()})
        del model
    return out,manifest,registries,train['semantic_capacities']+[train['suffix_capacity']]


def mapped_versions(raw,old,capacities):
    maps={};audit={}
    for method in ('plain','historical'):
        translation=fit(old,raw[method]['train'],capacities);audit[method]=translation
        for tag in ('unmapped','hungarian'):
            name=method+'_'+tag;maps[name]={stage:(transform(value,translation) if tag=='hungarian' else copy.deepcopy(value)) for stage,value in raw[method].items()}
            if {k:maps[name]['deployment'][k] for k in maps[name]['train']}!=maps[name]['train']:raise ValueError('Mapped extension changed training prefix')
            if tag=='hungarian' and transform(maps[name]['deployment'],translation,True)!=raw[method]['deployment']:raise ValueError('Inverse translation differs')
    return maps,audit


def tokenizer(output,name,stage):
    from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
    meta=read(Path(output)/'maps/meta.json');return FrozenSIDTokenizer(read(Path(output)/'maps'/(name+'_'+stage+'.json')),meta['capacities'],meta['max_history'])


def splits(ex,token,config):
    import evaluate_historical_future as old
    from genrec import future_evaluation as f
    args=SimpleNamespace(probe_size=config['selection_probe_size'])
    return old.make_splits(f,ex,token,read(Path(ex.e1['protocol_dir'])/'registries.json'),args,config)


def anchor_scores(ex,maps,config,output,device):
    import torch
    from torch.nn import functional as F
    from gensid import historical_incremental_content as h
    from genrec.temporal_tiger import make_batch
    teacher=h.frozen_teacher(ex,device);indices=ex.cohorts['anchor_train'];old=ex.arrays['base_train'];old_nll=[]
    def nll(arrays):
        result=[]
        with torch.no_grad():
            for first in range(0,len(indices),config['eval_batch_size']):
                batch=make_batch(arrays,np.asarray(indices[first:first+config['eval_batch_size']]),device)
                result.extend(F.cross_entropy(teacher(batch).logits.float().transpose(1,2),batch['labels'],reduction='none').sum(1).cpu().tolist())
        return np.asarray(result)
    old_nll=nll(old);results={}
    for name,value in maps.items():
        changed=h.reencode_history(old,ex.rows['base_train'],value['train']);new=nll(changed);delta=new-old_nll
        records=[dict(source_row=int(old['source_rows'][i]),user=ex.rows['base_train'][int(old['source_rows'][i])]['user'],old_nll=float(a),new_nll=float(b)) for i,a,b in zip(indices,old_nll,new)]
        path=Path(output)/'anchors'/(name+'.jsonl');path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
        results[name]=dict(queries=len(indices),H=float(np.maximum(delta,0).mean()),mean_delta_nll=float(delta.mean()),mean_old_nll=float(old_nll.mean()),mean_new_nll=float(new.mean()),records_sha256=sha(path))
    h.check_teacher(teacher,ex);del teacher
    return results


def prepare(deployment,output,p,device,*,fixture=False):
    import torch
    output=Path(output);root=Path(read(deployment)['runner_root']);d,previous,bound=source(root,p) if not fixture else (read(deployment),p['fixture_previous'],{})
    ex,directory,_=context(deployment,p);raw,manifest,registries,capacities=build_maps(ex,directory,device)
    maps,audit=mapped_versions(raw,{k:list(v) for k,v in ex.tokenizer.item2tokens.items()},capacities)
    config=copy.deepcopy(previous['config']);config['device']=str(device)
    if not fixture and {k:v for k,v in config.items() if k!='device'}!={k:v for k,v in p['previous_future_config'].items() if k!='device'}:raise ValueError('TIGER recipe changed')
    save(output/'config.json',config);(output/'maps').mkdir()
    save(output/'maps/meta.json',dict(capacities=capacities,max_history=manifest['max_history']))
    for name,value in maps.items():
        for stage,mapping in value.items():save(output/'maps'/(name+'_'+stage+'.json'),mapping)
    # Canonical RQ reload is performed on production CUDA before translation.
    bounded(output/'mapping.json',dict(fit_catalog='old Base train only',fit_items=len(ex.tokenizer.item2tokens),translations=audit,
        source_codebooks_unchanged=True,train_items=len(raw['plain']['train']),deployment_items=len(raw['plain']['deployment']),
        maps_sha256={name:{stage:sha(output/'maps'/(name+'_'+stage+'.json')) for stage in value} for name,value in maps.items()},
        maximum_semantic_bucket={name:max(bucket_sizes(value['train'])) for name,value in maps.items()}))
    _,train,val,_,probe=splits(ex,tokenizer(output,'plain_unmapped','train'),config)
    full=dict(input_sha256=bound,protocol_sha256=manifest['protocol_sha256'],probe_sample_ids=probe,
        train_sample_ids_sha256=canonical([r['sample_id'] for r in train]),fit_catalog='old Base train only',source_acceptance=p['source_acceptance'])
    audit_publication(output,full)
    reused=output/'reused/historical_unmapped';reused.mkdir(parents=True)
    previous_dir=Path(d['future_dir'])
    for name in ('summary.json','best.pth','last.pth'):shutil.copyfile(previous_dir/'historical'/name,reused/name)
    for old,new in [('historical_finetuned.json','future_summary.json'),('historical_finetuned_queries.jsonl','future_queries.jsonl')]:shutil.copyfile(previous_dir/old,reused/new)
    if read(previous_dir/'train_ledger.json')!=read(directory/'historical/allocation_registry.json'):raise ValueError('Reused Historical training mapping differs')
    from genrec.temporal_sids import maps_from_ledger
    if {k:list(v) for k,v in maps_from_ledger(read(previous_dir/'deployment_ledger.json'))[1].items()}!=maps['historical_unmapped']['deployment']:raise ValueError('Reused Historical deployment map differs')
    bounded(output/'anchors.json',anchor_scores(ex,maps,config,output,device))
    save(output/'prepared.json',dict(status='completed',mapping_sha256=sha(output/'mapping.json'),audit_sha256=sha(output/'audit.json')))
    projections(output,p)


def train(deployment,output,p,name,device):
    import torch
    from unittest.mock import patch
    from genrec import future_evaluation as f
    from genrec.cf_letter_base_training import Counter as StepCounter
    from genrec.temporal_tiger import cpu_state
    output=Path(output);config=read(output/'config.json');ex,_,_=context(deployment,p);token=tokenizer(output,name,'train')
    data,train_rows,_,_,probe=splits(ex,token,config)
    if canonical(probe)!=read(output/'audit.json')['probe_sample_ids_sha256']:raise ValueError('Fixed probe changed')
    folder=output/'arms'/name;counter=StepCounter(folder,p['steps_per_arm']);factory=f.Trainer.for_temporal;batch_fn=f.make_batch
    def instrument(conf,model,tok):
        trainer=factory(conf,model,tok)
        torch.save(dict(state_dict=cpu_state(model),optimizer=trainer.optimizer.state_dict(),config=conf),folder/'initial.pth')
        step=trainer.optimizer.step
        trainer.optimizer.step=lambda *a,**kw:counter.update(SimpleNamespace(step=lambda:step(*a,**kw)))
        trainer.optimizer.step._wrapped_by_lr_sched=True
        return trainer
    def batch(arrays,indices,dev):
        if arrays is data['update_train']['arrays']:
            with (folder/'sampling.jsonl').open('a') as handle:handle.write(json.dumps(dict(step=counter.step+1,indices=np.asarray(indices).tolist(),source_rows=arrays['source_rows'][indices].tolist()))+'\n')
        return batch_fn(arrays,indices,dev)
    args=SimpleNamespace(max_steps=p['steps_per_arm'],log_every=100,save_every=50)
    with patch.object(f.Trainer,'for_temporal',side_effect=instrument),patch.object(f,'make_batch',side_effect=batch):
        result=f.fit_arm(ex,name,token,data,config,dict(job_id=p['job_id'],arm=name,audit_sha256=sha(output/'audit_full.json'),mapping_sha256=sha(output/'maps'/(name+'_train.json'))),folder,device,args)
    save(folder/(name+'_curves.json'),result);save(folder/'arm_result.json',dict(result=result));return result


def evaluate_future(deployment,output,p,name,device):
    import torch
    from genrec import future_evaluation as f
    output=Path(output);lock=read(output/'selection_locked.json')
    for arm,item in lock['arms'].items():
        folder=output/('reused' if arm=='historical_unmapped' else 'arms')/arm
        if sha(folder/'best.pth')!=item['selected_sha256'] or sha(output/'maps'/(arm+'_deployment.json'))!=item['mapping_sha256']:raise ValueError('Global model/mapping lock changed')
    ex,_,_=context(deployment,p);config=read(output/'config.json');token=tokenizer(output,name,'deployment');train_token=tokenizer(output,name,'train')
    _,train,val,base_ids,_=splits(ex,train_token,config)
    rows=f.load_stage(ex,'future_test',base_ids+[r['sample_id'] for r in train+val]);registries=read(Path(ex.e1['protocol_dir'])/'registries.json')
    data=f.encode(rows,token,registries['update']);folder=output/'arms'/name;saved=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
    model=f.TIGER(config,None,token).to(device);model.load_state_dict(saved['state_dict'],strict=True);model.eval().requires_grad_(False)
    generate=model.generate;reverse={tuple(v):k for k,v in token.item2tokens.items()};predictions=[]
    def capture(batch,**kwargs):
        out=generate(batch,**kwargs)
        predictions.extend([[reverse[tuple(tokens)] for tokens in group] for group in out['preds'].cpu().tolist()]);return out
    model.generate=capture
    metrics=f.future_metrics(model,config,token,data,rows,registries,Counter(r['item_seq'][-1] for r in train),device,folder/'future_queries.jsonl')
    served=[int(i) for i in data['arrays']['source_rows']]
    if len(predictions)!=len(served):raise ValueError('Prediction sidecar size differs')
    (folder/'future_predictions.jsonl').write_text(''.join(json.dumps(dict(source_row=i,predictions=v))+'\n' for i,v in zip(served,predictions)))
    if any(not torch.equal(v.cpu(),saved['state_dict'][k]) for k,v in model.state_dict().items()):raise ValueError('Inference changed selected weights')
    metrics.update(arm=name,model_sha256=sha(folder/'best.pth'),mapping_sha256=sha(output/'maps'/(name+'_deployment.json')),
        update_optimizer_steps=read(folder/'summary.json')['best_update_optimizer_step'],selection_locked_before_future=True,selection_lock_sha256=sha(output/'selection_locked.json'))
    save(folder/'future_summary.json',metrics);return metrics


def compare_records(before,after,reps=500,seed=2026):
    if [(r['sample_id'],r['user'],r['target_raw_item_id'],r['target_servable']) for r in before]!=[(r['sample_id'],r['user'],r['target_raw_item_id'],r['target_servable']) for r in after]:raise ValueError('Different Future identities')
    keys=list(before[0]['metrics']);groups={}
    for a,b in zip(before,after):
        value=groups.setdefault(a['user'],[np.zeros(len(keys)),0]);value[0]+=np.asarray([b['metrics'][k]-a['metrics'][k] for k in keys]);value[1]+=1
    totals=np.asarray([v[0] for v in groups.values()]);counts=np.asarray([v[1] for v in groups.values()]);rng=np.random.default_rng(seed);draws=[]
    for _ in range(reps):
        ids=rng.integers(0,len(groups),len(groups));draws.append(totals[ids].sum(0)/counts[ids].sum())
    delta=totals.sum(0)/counts.sum();ci=np.quantile(draws,[.025,.975],axis=0)
    return dict(queries=len(before),users=len(groups),bootstrap_replicates=reps,bootstrap_unit='user; query-weighted clusters',
        metrics={k:dict(absolute_difference=float(delta[i]),ci95=ci[:,i].tolist()) for i,k in enumerate(keys)})


def comparisons(output,p):
    output=Path(output);records={}
    for name in p['arms']:
        folder=output/('reused' if name=='historical_unmapped' else 'arms')/name
        records[name]=[json.loads(v) for v in (folder/'future_queries.jsonl').read_text().splitlines()]
    pairs=[('plain_hungarian','plain_unmapped'),('historical_hungarian','historical_unmapped'),('historical_hungarian','plain_hungarian'),('historical_unmapped','plain_unmapped')]
    return {a+'_minus_'+b:compare_records(records[b],records[a],p['bootstrap_replicates'],p['seed']) for a,b in pairs}
