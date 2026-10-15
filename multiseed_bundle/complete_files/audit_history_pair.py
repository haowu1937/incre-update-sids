"""Independent audit and re-evaluation of the frozen-encoder Plain/History pair.

Hard evaluation rebuilds full SID input IDs from raw rows and manually shifts
old decoder labels. It uses log-softmax/gather, not FullHistory, reencode_history,
paired_losses, inputs_embeds, or the training cross-entropy implementation.
Training is rerun by the immutable original engine only after preflight passes.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import history_aware_frozen_encoder as f

FORMAT='independent_history_pair_audit_v1'
NLL_TOLERANCE=1e-4


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def load_pair(args):
    ex=f.h.experiment(args);reports=[];checkpoints=[]
    for directory,mode in zip((Path(args.plain_dir),Path(args.history_dir)),('plain','history-aware')):
        r=f.read_json(directory/'summary.json')
        if (r.get('stage')!=f.FORMAT or r.get('execution_status')!='completed' or r.get('mode')!=mode
                or r['binding']!=f.binding(ex)):
            raise ValueError('Frozen pair source/data/mode binding mismatch: '+mode)
        f.verify_files(directory,r,['last.pth','config.json','anchor_train_tensor_rows.json',
            'anchor_train_hard_losses.jsonl','semantic_assignments.jsonl','new_item2tokens.json','new_allocation_registry.json'])
        cp=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
        if (cp.get('format')!=f.FORMAT or cp['config']!=r['config'] or cp['binding']!=r['binding']
                or cp['sampler_state']['step']!=r['update_optimizer_steps']):
            raise ValueError('Pair checkpoint/report mismatch: '+mode)
        if f.read_json(directory/'anchor_train_tensor_rows.json')!=ex.cohorts['anchor_train']:
            raise ValueError('Training cohort differs: '+mode)
        if any(not torch.equal(v,ex.parent['state_dict'][k]) for k,v in cp['state_dict'].items() if k.startswith('encoder.')):
            raise ValueError('Frozen encoder differs from Base: '+mode)
        reports.append(r);checkpoints.append(cp)
    configs=[copy.deepcopy(r['config']) for r in reports]
    if configs[0]['historical_v2']['history_gradient_ratio']!=0 or configs[1]['historical_v2']['history_gradient_ratio']<=0:
        raise ValueError('Expected Plain rho=0 and History rho>0')
    for c in configs:c['historical_v2'].pop('history_gradient_ratio');c['frozen_encoder'].pop('mode')
    if configs[0]!=configs[1]:raise ValueError('Content or historical-forward settings differ')
    if not equal(checkpoints[0]['sampler_state'],checkpoints[1]['sampler_state']):
        raise ValueError('Pair content sampler order/epoch/cursor/steps differ')
    if not torch.equal(checkpoints[0]['old_anchor_nll'],checkpoints[1]['old_anchor_nll']):
        raise ValueError('Pair cached old teacher NLL differs')
    # Base train is all servable in this protocol. Guard raw-row vs tensor-row assumptions.
    sources=ex.arrays['base_train']['source_rows']
    if not np.array_equal(sources,np.arange(len(ex.rows['base_train']))):
        raise ValueError('Base raw/tensor rows are not identity; E1 cohort conversion needs explicit repair')
    train_users={ex.rows['base_train'][i]['user'] for i in ex.cohorts['anchor_train']}
    hold_users={ex.rows['base_train'][i]['user'] for i in ex.cohorts['anchor_holdout']}
    if train_users & hold_users:raise ValueError('Training/holdout users overlap')
    return ex,reports,checkpoints


def manual_codes(model,ex):
    """Rebuild native greedy RQ assignments, including its float32 STE forward arithmetic."""
    codes=[];device=next(model.parameters()).device;model.eval()
    with torch.no_grad():
        for name,(left,right) in ex.groups.items():
            size=ex.old_batch if name=='base_train' else ex.plain['config']['added_item_assignment_eval_batch_size']
            for start in range(left,right,size):
                residual=model.encoder(ex.features[start:min(start+size,right)].to(device));levels=[]
                for layer in model.rq.vq_layers:
                    table=layer.embedding.weight
                    distances=residual.square().sum(1,keepdim=True)+table.square().sum(1)[None,:]-2*residual@table.t()
                    index=distances.argmin(1);levels.append(index)
                    picked=table[index]
                    quantized=residual+(picked-residual).detach()
                    residual=residual-quantized
                codes.extend(torch.stack(levels,dim=1).cpu().tolist())
    return codes


def manual_mapping(previous,feature_rows,codes):
    """Independent persistent-slot allocation; no training allocation helper."""
    old={r['raw_item_id']:r for r in previous['entries']};next_slot={}
    capacities=previous['semantic_capacities'];suffix_capacity=previous['suffix_capacity']
    for r in previous['entries']:
        key=tuple(r['semantic_code']);next_slot[key]=max(next_slot.get(key,0),r['collision_suffix']+1)
    result={}
    if len(feature_rows)!=len(codes):raise ValueError('Manual item/code count mismatch')
    for row,code in zip(feature_rows,codes):
        raw=row['raw_item_id'];key=tuple(code);prior=old.get(raw)
        if len(code)!=3 or any(not 0<=c<cap for c,cap in zip(code,capacities)):
            raise ValueError('Manual semantic code outside capacity')
        if prior is not None and list(code)==prior['semantic_code']:slot=prior['collision_suffix']
        else:slot=next_slot.get(key,0);next_slot[key]=slot+1
        if slot>=suffix_capacity:raise ValueError('Manual suffix capacity exhausted')
        offset=1;tokens=[]
        for c,cap in zip([*code,slot],[*capacities,suffix_capacity]):tokens.append(c+offset);offset+=cap
        result[raw]=tokens
    if len({tuple(v) for v in result.values()})!=len(result):raise ValueError('Manual full SIDs are not unique')
    return result


def endpoint(directory,cp,ex,device):
    model=f.build_model(cp['config']).to(device);model.load_state_dict(cp['state_dict'],strict=True)
    codes=manual_codes(model,ex)
    expected=f.read_jsonl(Path(directory)/'semantic_assignments.jsonl')
    if codes!=[r['semantic_code'] for r in expected]:raise ValueError('Manual/native SID assignments differ')
    for known,r in zip(ex.feature_rows,expected):
        if any(known[k]!=r[k] for k in known):raise ValueError('Export raw/global/feature row mismatch')
    mapping=manual_mapping(ex.previous,ex.feature_rows,codes)
    if mapping!=f.read_json(Path(directory)/'new_item2tokens.json'):
        raise ValueError('Manual/full SID allocation differs from exported mapping')
    del model
    return mapping


def raw_batch(ex,split,selected,mapping):
    """Build old/new histories and OLD labels directly from raw example fields."""
    arrays=ex.arrays[split];width=arrays['input_ids'].shape[1];eos=ex.tokenizer.eos_token
    old_ids=np.zeros((len(selected),width),dtype=np.int64);new_ids=old_ids.copy();labels=[];identities=[]
    old_mapping=ex.tokenizer.item2tokens
    for j,index in enumerate(selected):
        source=int(arrays['source_rows'][index]);raw=ex.rows[split][source]
        history=raw['item_seq'][:-1];available=raw['history_available']
        if len(history)!=len(available) or len(history)*4+1>width:raise ValueError('History/availability length mismatch')
        for k,(item,allow) in enumerate(zip(history,available)):
            if allow:
                old_ids[j,k*4:k*4+4]=old_mapping[item];new_ids[j,k*4:k*4+4]=mapping[item]
        old_ids[j,4*len(history)]=eos;new_ids[j,4*len(history)]=eos
        target=[*old_mapping[raw['item_seq'][-1]],eos];labels.append(target)
        identities.append(dict(sample_id=raw['sample_id'],user=raw['user'],source_row=source,tensor_row=int(index),
            target_raw_item_id=raw['item_seq'][-1],target_time=raw['target_time'],target_labels=target,
            history_raw_items=list(history),history_available=list(available)))
    labels=np.asarray(labels,dtype=np.int64);mask=(old_ids!=0).astype(np.int64)
    if (not np.array_equal(old_ids,arrays['input_ids'][selected])
            or not np.array_equal(mask,arrays['attention_mask'][selected])
            or not np.array_equal(labels,arrays['labels'][selected])):
        raise ValueError('Raw reconstruction differs from frozen input/mask/OLD labels')
    if not np.array_equal(mask,new_ids!=0):raise ValueError('Reencoding changed history availability')
    return old_ids,new_ids,mask,labels,identities


def cohort_hash(ex,split,selected):
    old_map={k:list(v) for k,v in ex.tokenizer.item2tokens.items()}
    old,new,mask,labels,identities=raw_batch(ex,split,np.asarray(selected),old_map)
    return canonical_hash(dict(identities=identities,input_ids=old.tolist(),mask=mask.tolist(),labels=labels.tolist()))


@torch.no_grad()
def independent_records(teacher,ex,split,selected,mapping,batch_size,device,progress=False):
    rows=[]
    for start in range(0,len(selected),batch_size):
        chosen=np.asarray(selected[start:start+batch_size]);old,new,mask,labels,identities=raw_batch(ex,split,chosen,mapping)
        end=int(np.where(mask.any(axis=0))[0][-1])+1
        labels_t=torch.as_tensor(labels,device=device);decoder=torch.zeros_like(labels_t);decoder[:,1:]=labels_t[:,:-1]
        scores=[]
        for ids in (old,new):
            logits=teacher(dict(input_ids=torch.as_tensor(ids[:,:end],device=device),
                attention_mask=torch.as_tensor(mask[:,:end],device=device),decoder_input_ids=decoder,use_cache=False)).logits.float()
            per_token=-torch.log_softmax(logits,-1).gather(2,labels_t.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(per_token).all():raise FloatingPointError('Independent NLL is not finite')
            scores.append(per_token.double().sum(1).cpu().numpy())
        for j,identity in enumerate(identities):
            delta=float(scores[1][j]-scores[0][j])
            rows.append(dict(**identity,old_sequence_nll=float(scores[0][j]),new_sequence_nll=float(scores[1][j]),
                delta_sequence_nll=delta,positive_damage=max(delta,0.),history_changed=bool(np.any(old[j]!=new[j]))))
        if progress and (start//batch_size+1)%32==0:print('Independent '+split+' '+str(min(start+batch_size,len(selected)))+'/'+str(len(selected)),flush=True)
    return rows


def positive_summary(records):
    return dict(queries=len(records),positive_damage=float(np.mean([r['positive_damage'] for r in records])),
        mean_delta_sequence_nll=float(np.mean([r['delta_sequence_nll'] for r in records])),
        worsened_fraction=float(np.mean([r['delta_sequence_nll']>1e-6 for r in records])))


def check_saved(records,directory):
    saved=f.read_jsonl(Path(directory)/'anchor_train_hard_losses.jsonl')
    if len(records)!=len(saved):raise ValueError('Saved hard-loss query count differs')
    errors=[]
    for raw,known in zip(records,saved):
        if any(raw[k]!=known[k] for k in ('sample_id','user','source_row','tensor_row','target_labels','history_changed')):
            raise ValueError('Saved anchor identity/history/OLD target differs')
        errors.extend(abs(raw[k]-known[k]) for k in ('old_sequence_nll','new_sequence_nll','positive_damage'))
    maximum=max(errors)
    if maximum>NLL_TOLERANCE:raise ValueError('Independent token-ID NLL differs from saved embedding-path NLL: '+str(maximum))
    return maximum


def manual_rq_loss(model,data):
    residual=model.encoder(data);quantized=0;losses=[]
    for layer in model.rq.vq_layers:
        w=layer.embedding.weight
        d=residual.square().sum(1,keepdim=True)+w.square().sum(1)[None,:]-2*residual@w.t()
        word=w[d.argmin(1)]
        losses.append((word-residual.detach()).square().mean()+layer.mu*(word.detach()-residual).square().mean())
        q=residual+(word-residual).detach();quantized=quantized+q;residual=residual-q
    recon=(model.decoder(quantized)-data).square().mean()
    return recon+model.quant_loss_weight*torch.stack(losses).mean(),recon


def audit_gradients(ex,cp,device):
    model=f.build_model(cp['config']).to(device);model.load_state_dict(cp['state_dict'],strict=True);model.eval()
    data=ex.features[:min(cp['config']['batch_size'],len(ex.features))].to(device)
    values=model(data,None,use_sk=False)
    original=model.compute_loss(values[0],values[1],None,values[3],dense_out_codebooks=values[4],x_res_list=values[5],xs=data)[0]
    parameters=list(model.named_parameters())
    native=torch.autograd.grad(original,[p for _,p in parameters],allow_unused=True)
    rebuilt,recon=manual_rq_loss(model,data)
    manual=torch.autograd.grad(rebuilt,[p for _,p in parameters],allow_unused=True,retain_graph=True)
    max_relative=0.
    for (name,p),a,b in zip(parameters,native,manual):
        if (a is None)!=(b is None):raise ValueError('Independent RQ gradient presence differs: '+name)
        if a is not None:
            error=float((a-b).double().norm());norm=float(a.double().norm())
            relative=error/max(norm,1e-30);max_relative=max(max_relative,relative)
            if error>1e-7 and relative>1e-3:raise ValueError('Independent RQ gradient differs: '+name)
    if abs(float(original.detach())-float(rebuilt.detach()))>1e-7:raise ValueError('Independent RQ objective differs')
    recon_grads=torch.autograd.grad(recon,[p for _,p in parameters],allow_unused=True)
    book_recon=sum(float(v.double().norm()) for (name,_),v in zip(parameters,recon_grads) if name.startswith('rq.') and v is not None)
    if book_recon!=0:raise ValueError('Unexpected reconstruction gradient into codebooks under native STE')
    subset=copy.copy(ex);subset.cohorts=dict(ex.cohorts,anchor_train=ex.cohorts['anchor_train'][:min(32,len(ex.cohorts['anchor_train']))])
    teacher=f.h.frozen_teacher(ex,device)
    scales=cp['distance_scales']
    history=f.FullHistory(model,teacher,subset,cp['config'],scales,progress_every=0)
    codes,_,_,mapping=f.catalog(model,ex)
    measured,grads=history.measure(codes,mapping,backward=True)
    if any(p.grad is not None for p in model.parameters()):raise ValueError('History autograd unexpectedly wrote parameter .grad buffers')
    old,ids,embeds=history.inputs(history.packs[0],codes,mapping,True)
    logits=teacher(dict(inputs_embeds=embeds,attention_mask=old['attention_mask'],labels=old['labels'],use_cache=False)).logits.float()
    raw_nll=-torch.log_softmax(logits,-1).gather(2,old['labels'].unsqueeze(-1)).sum()
    route=torch.autograd.grad(raw_nll,[p for _,p in parameters],allow_unused=True)
    if any(v is not None for (name,_),v in zip(parameters,route) if name.startswith(('encoder.','decoder.'))):
        raise ValueError('Historical surrogate unexpectedly reaches encoder or decoder')
    f.h.check_teacher(teacher,ex)
    mix,stats=f.normalized_mix([torch.ones_like(x)*1e-4 for x in grads],grads,.1,1e-12)
    if any(s['effective_gradient_ratio']>.100001 for s in stats):raise ValueError('Wrong gradient mixing magnitude')
    return dict(independent_content_loss_and_gradients_equal=True,max_relative_RQ_gradient_difference=max_relative,
        reconstruction_gradient_into_codebooks=book_recon,history_gradient_route='codebooks only; content encoder detached',
        historical_probe_queries=measured['queries'],historical_gradients_finite=all(bool(torch.isfinite(g).all()) for g in grads),
        internal_commitment_mu=[layer.mu for layer in model.rq.vq_layers],outer_quant_loss_weight=model.quant_loss_weight,
        quantizer_reduction='mean of per-level codebook+mu*commitment losses',
        STE_is_a_surrogate_not_a_guaranteed_hard_H_descent=True)


def paired(records,reference,seed):
    if len(records)!=len(reference):raise ValueError('Paired cohort sizes differ')
    clusters={};diffs=[]
    for a,b in zip(records,reference):
        if any(a[k]!=b[k] for k in ('sample_id','user','source_row','tensor_row','target_labels','history_raw_items','history_available')):
            raise ValueError('Paired anchor contents differ')
        if abs(a['old_sequence_nll']-b['old_sequence_nll'])>NLL_TOLERANCE:raise ValueError('Paired old reference differs')
        diff=np.array([a['positive_damage']-b['positive_damage'],a['new_sequence_nll']-b['new_sequence_nll']])
        diffs.append(diff);entry=clusters.setdefault(a['user'],np.zeros(3));entry[:2]+=diff;entry[2]+=1
    groups=np.array(list(clusters.values()));rng=np.random.default_rng(seed);draws=[]
    for _ in range(500):
        total=groups[rng.integers(0,len(groups),len(groups))].sum(0);draws.append(total[:2]/total[2])
    ci=np.quantile(draws,[.025,.975],axis=0).T.tolist() if len(groups)>1 else [None,None]
    means=np.mean(diffs,axis=0)
    return dict(history_minus_plain_positive_damage=float(means[0]),positive_damage_cluster_ci95=ci[0],
        history_minus_plain_mean_sequence_nll=float(means[1]),sequence_nll_cluster_ci95=ci[1],
        users=len(groups),bootstrap_replicates=500,bootstrap_unit='user; row-weighted paired mean')


def run(args):
    started=time.perf_counter();out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite independent audit')
    ex,reports,cps=load_pair(args);device=f.h.device_from(args.device)
    if args.batch_size!=reports[0]['config']['historical_v2']['anchor_batch_size']:
        raise ValueError('Audit batch size must equal saved reference batch size')
    f.seed_all(reports[0]['config']['seed']);teacher=f.h.frozen_teacher(ex,device)
    maps=[endpoint(d,cp,ex,device) for d,cp in zip((args.plain_dir,args.history_dir),cps)]
    gradients=audit_gradients(ex,cps[1],device) if args.command=='preflight' else None
    tasks=[('anchor_train','base_train',ex.cohorts['anchor_train'])]
    if args.command=='compare':
        tasks += [('anchor_holdout','base_train',ex.cohorts['anchor_holdout']),
                  ('base_validation','base_val',list(range(len(ex.arrays['base_val']['labels']))))]
    pending={};groups={};cohort_hashes={};maximum_error=0.
    for name,split,selected in tasks:
        records=[independent_records(teacher,ex,split,selected,mapping,args.batch_size,device,args.progress) for mapping in maps]
        if name=='anchor_train':
            for rows,directory,cp in zip(records,(args.plain_dir,args.history_dir),cps):
                maximum_error=max(maximum_error,check_saved(rows,directory))
                error=max(abs(r['old_sequence_nll']-float(old)) for r,old in zip(rows,cp['old_anchor_nll']))
                if error>NLL_TOLERANCE:raise ValueError('Independent old NLL differs from training cache')
        cohort_hashes[name]=cohort_hash(ex,split,selected)
        groups[name]=dict(plain=positive_summary(records[0]),history_aware=positive_summary(records[1]),
            paired=paired(records[1],records[0],ex.e1['cohort_seed']))
        for arm,rows in zip(('plain','history'),records):pending[name+'_'+arm+'.jsonl']=rows
        print(name+': '+json.dumps(groups[name],allow_nan=False),flush=True)
    reproduced={}
    if args.command=='compare' and args.previous_plain_dir and args.previous_history_dir:
        for arm,directory,cp in zip(('plain','history'),(args.previous_plain_dir,args.previous_history_dir),cps):
            previous=torch.load(Path(directory)/'last.pth',map_location='cpu',weights_only=False)
            if previous.get('format')!=f.FORMAT or previous['config']!=cp['config'] or previous['binding']!=cp['binding']:
                raise ValueError('Previous run is not the same experiment configuration')
            reproduced[arm]={k:equal(previous[k],cp[k]) for k in ('state_dict','optimizer','sampler_state')}
    f.h.check_teacher(teacher,ex)
    out.mkdir(parents=True)
    for name,rows in pending.items():
        with (out/name).open('w') as handle:
            for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
    result=dict(stage=FORMAT,execution_status='passed' if args.command=='preflight' else 'completed',
        command=args.command,plain_dir=str(Path(args.plain_dir).resolve()),history_dir=str(Path(args.history_dir).resolve()),
        binding=reports[0]['binding'],audit_source_sha256=f.sha256(Path(__file__)),
        update_optimizer_steps=reports[0]['update_optimizer_steps'],rho=reports[1]['config']['historical_v2']['history_gradient_ratio'],
        same_raw_anchors_and_old_targets=True,cohort_hashes=cohort_hashes,train_holdout_user_disjoint=True,
        max_independent_vs_saved_NLL_error=maximum_error,independent_NLL_tolerance=NLL_TOLERANCE,
        manual_SID_assignments_and_suffix_allocation_equal=True,encoder_unchanged_both=True,
        gradient_audit=gradients,groups=groups,reproduction=reproduced,
        rq_mse={arm:r['final_train_catalog_metrics']['reconstruction_loss'] for arm,r in zip(('plain','history'),reports)},
        holdout_evaluated=args.command=='compare',holdout_used_for_training=False,ranking_evaluated=False,future_test_used=False,
        same_compute=False,real_data_execution_location='user machine when run by user',runtime_seconds=time.perf_counter()-started,
        artifact_files_sha256={p.name:f.sha256(p) for p in out.iterdir() if p.is_file()})
    f.write_json(out/'summary.json',result)
    print('Independent '+args.command+' passed.',flush=True)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('preflight','compare'):
        a=sub.add_parser(name);a.add_argument('--e1-dir',required=True);a.add_argument('--plain-dir',required=True)
        a.add_argument('--history-dir',required=True);a.add_argument('--output',required=True)
        a.add_argument('--device',default='cuda');a.add_argument('--batch-size',type=int,default=32);a.add_argument('--progress',action='store_true')
        if name=='compare':a.add_argument('--previous-plain-dir');a.add_argument('--previous-history-dir')
    return p


if __name__=='__main__':run(parser().parse_args())
