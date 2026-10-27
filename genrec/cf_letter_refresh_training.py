"""Frozen-GPT anchor reference, cumulative SASRec refresh and feature-only damage."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from genrec.cf_letter_models import SASRec,CollaborativeRQ,GenerativeRetriever
from genrec.cf_letter_base_training import Sampler,Counter,checkpoint,evaluate,evaluating,fixed_loss,gradients,log,tensor
from genrec.cf_letter_protocol import gpt_arrays,read,save,sha
from genrec.cf_letter_refresh_protocol import versioned_mapping,damage


def expanded_sasrec(saved,count,p,device):
    model=SASRec(count,p['history'],p['sasrec']).to(device);old=saved['model'];state=copy.deepcopy(old)
    with torch.no_grad():
        rows=model.items.weight.detach().clone();rows[:len(old['items.weight'])].copy_(old['items.weight'].to(device))
    state['items.weight']=rows;model.load_state_dict(state)
    equal=all(torch.equal(value.detach().cpu(),old[name].cpu())
        for name,value in model.state_dict().items() if name!='items.weight')
    equal=equal and torch.equal(model.items.weight[:len(old['items.weight'])].detach().cpu(),old['items.weight'].cpu())
    if not equal:raise ValueError('SASRec warm start changed original weights/rows')
    return model


def teacher(output,p,device):
    source=torch.load(Path(output)/'inputs/base/gpt/selected.pth',map_location='cpu',weights_only=False)
    if source['step']!=p['predecessor']['gpt_step'] or source['code_sha']!=p['predecessor']['code_sha']:
        raise ValueError('Wrong selected Base GPT')
    model=GenerativeRetriever(p['history'],p['quantizer']['capacity'],p['gpt']).to(device)
    model.load_state_dict(source['model']);model.eval();model.requires_grad_(False)
    return model,source['model']


def paired_nll(model,anchors,old_map,new_map,p,device):
    old,mask,targets=gpt_arrays(anchors,old_map,p['quantizer']['capacity'],p['gpt']['user_buckets'])
    new,new_mask,_=gpt_arrays(anchors,new_map,p['quantizer']['capacity'],p['gpt']['user_buckets'])
    if not np.array_equal(mask,new_mask) or not np.array_equal(old[:,0],new[:,0]):raise ValueError('Historical mask/user token changed')
    before=[];after=[]
    with evaluating(model):
        for first in range(0,len(old),p['eval_batch_size']):
            take=slice(first,first+p['eval_batch_size']);m=tensor(mask[take],device);y=tensor(targets[take],device)
            before.extend(model.nll(tensor(old[take],device),m,y).cpu().tolist())
            after.extend(model.nll(tensor(new[take],device),m,y).cpu().tolist())
    return np.asarray(before),np.asarray(after),np.any(old!=new,axis=1)


def records(data,old,new,changed):
    return [dict(source_row=int(source),sample_id=str(sample),user=str(user),old_sequence_nll=float(a),
        new_sequence_nll=float(b),history_changed=bool(c)) for source,sample,user,a,b,c in zip(
        data['anchor_source_rows'],data['anchors']['samples'],data['anchor_users'],old,new,changed)]


def write_rows(path,rows):
    with Path(path).open('x') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def frozen_quantizer(output,p,device):
    source=torch.load(Path(output)/'inputs/base/quantizer/selected.pth',map_location='cpu',weights_only=False)
    old=np.load(Path(output)/'inputs/base/cf_embeddings.npy',allow_pickle=False)
    model=CollaborativeRQ(tensor(old,device),p['quantizer']).to(device);model.load_state_dict(source['model']);model.requires_grad_(False);model.eval()
    if not np.array_equal(model.features.cpu().numpy(),old):raise ValueError('Frozen quantizer CF reference differs')
    return model


@torch.no_grad()
def quantize(model,features,device):
    values=[];restored=[];native=0.;quant=0.
    for start in range(0,len(features),1024):
        chunk=features[start:start+1024]
        x=tensor(chunk,device);r=x.clone();codes=[];reconstruction=torch.zeros_like(x)
        for book in model.books:
            index=model.nearest(r,book);codes.append(index);reconstruction+=book[index];r-=book[index]
        loss,_,q,_=model(x);native+=float(loss)*len(x);quant+=float(q)*len(x)
        values.append(torch.stack(codes,dim=1).cpu().numpy());restored.append((reconstruction-x).square().sum().item())
    return np.concatenate(values),dict(items=len(features),total_loss=native/len(features),quantization_loss=quant/len(features),
        reconstruction_mse=sum(restored)/(len(features)*model.features.shape[1]))


def baseline(output,data,p,binding,device):
    directory=Path(output)/'baseline';directory.mkdir()
    model,state=teacher(output,p,device);mapping=np.load(Path(output)/'inputs/base/item_sids.npy',allow_pickle=False)
    q=frozen_quantizer(output,p,device);codes,qmetrics=quantize(q,q.features,device)
    if not np.array_equal(codes,np.load(Path(output)/'inputs/base/quantizer/semantic_codes.npy',allow_pickle=False)):
        raise ValueError('Old GPU semantic assignments do not reproduce')
    again,capacity=versioned_mapping(mapping,codes,p['quantizer']['capacity'])
    if not np.array_equal(mapping,again) or not capacity['full_sid_allocation_feasible']:raise ValueError('Same-map suffix allocation changed Base SIDs')
    a,b,changed=paired_nll(model,data['anchors'],mapping,mapping,p,device)
    rows=records(data,a,b,changed);result=damage(rows,p['bootstrap_replicates'],p['seed'])
    if np.max(np.abs(b-a))>1e-6 or result['H']>1e-6 or changed.any():raise ValueError('Same-map hard H is not zero')
    np.save(directory/'old_anchor_nll.npy',a);write_rows(directory/'anchor_reference.jsonl',rows)
    unchanged=all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
    if not unchanged or any(v.requires_grad or v.grad is not None for v in model.parameters()):raise ValueError('Frozen Base teacher changed')
    report=dict(stage='baseline',execution_status='completed',optimizer_steps=0,anchor_train=result,same_map_H_zero=True,
        teacher_unchanged=True,old_assignments_reproduced=True,old_full_sids_reproduced=True,quantizer=qmetrics)
    save(directory/'summary.json',report);return report


def refresh(output,data,p,binding,device):
    directory=Path(output)/'sasrec';directory.mkdir();count=len(binding['items']);old_count=len(binding['old_items'])
    source=torch.load(Path(output)/'inputs/base/sasrec/selected.pth',map_location='cpu',weights_only=False)
    if source['step']!=p['predecessor']['sasrec_step'] or source['code_sha']!=p['predecessor']['code_sha']:raise ValueError('Wrong selected Base SASRec')
    model=expanded_sasrec(source,count,p,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    if optimizer.state:raise ValueError('Refresh optimizer must be reset')
    sampler=Sampler(len(data['train']['targets']),p['batch_size'],p['seed']);counter=Counter(directory,p['optimizer_step_cap'])
    checkpoint(directory/'initial.pth',model,optimizer,sampler,counter,p,binding)
    negative_counts=np.zeros(count+1,dtype=np.int64);monitor=np.arange(min(p['probe_size'],len(data['train']['targets'])))
    def score():return evaluate(model,data['val'],data['probe'],p,device,'sasrec',catalog_size=old_count)
    def train_loss():return fixed_loss(model,data['train'],monitor,device,'sasrec')
    initial=score();curve=[dict(step=0,epoch=0,training_probe_loss=train_loss(),validation=initial)]
    best=curve[0];checkpoint(directory/'selected.pth',model,optimizer,sampler,counter,p,binding);began=time.monotonic()
    while sampler.epochs<p['sasrec']['epochs'] and counter.step<p['optimizer_step_cap']:
        indices,done=sampler.take();y=tensor(data['train']['targets'][indices],device)
        neg=torch.randint(1,count,y.shape,device=device);neg=neg+(neg>=y)
        negative_counts+=np.bincount(neg.cpu().numpy(),minlength=count+1)
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=model.loss(tensor(data['train']['history'][indices],device),y,neg);norm=gradients(model,loss,p);counter.update(optimizer)
        if counter.step%20==0 or done:log(directory,counter,sampler,loss=float(loss),gradient_norm=norm)
        if done and sampler.epochs%p['sasrec']['eval_every_epochs']==0:
            entry=dict(step=counter.step,epoch=sampler.epochs,training_probe_loss=train_loss(),validation=score());curve.append(entry);save(directory/'curve.json',curve)
            print(json.dumps(entry),flush=True)
            if entry['validation']['full_denominator']['ndcg@10']>best['validation']['full_denominator']['ndcg@10']:
                best=entry;checkpoint(directory/'selected.pth',model,optimizer,sampler,counter,p,binding)
        if done:checkpoint(directory/'last.pth',model,optimizer,sampler,counter,p,binding)
    if sampler.epochs!=p['sasrec']['epochs']:raise ValueError('Cumulative epoch budget inconsistent with step cap')
    checkpoint(directory/'last.pth',model,optimizer,sampler,counter,p,binding)
    selected=torch.load(directory/'selected.pth',map_location=device,weights_only=False);model.load_state_dict(selected['model']);model.eval()
    selected_probe=evaluate(model,data['val'],data['probe'],p,device,'sasrec',directory/'selected_probe_queries.jsonl',catalog_size=old_count)
    if selected_probe['full_denominator']!=best['validation']['full_denominator']:raise ValueError('Refresh selected ranking reload differs')
    full=evaluate(model,data['val'],np.arange(len(data['val']['targets'])),p,device,'sasrec',directory/'full_update_validation_queries.jsonl',catalog_size=old_count)
    features=model.items.weight[1:].detach().cpu().numpy().astype('float32');np.save(Path(output)/'refreshed_cf_embeddings.npy',features)
    if not np.array_equal(features,selected['model']['items.weight'][1:].cpu().numpy()) or not np.isfinite(features).all():raise ValueError('Refreshed CF export differs')
    save(directory/'negative_exposure.json',dict(negative_counts=negative_counts.tolist()))
    result=dict(stage='sasrec',execution_status='completed',optimizer_steps=counter.step,epochs=sampler.epochs,processed_examples=sampler.processed,
        training_items=count,old_catalog_items=old_count,new_cf_rows=count-old_count,selected_step=best['step'],selected_epoch=best['epoch'],
        warm_start_old_weights_equal=True,optimizer_reset=True,selected_checkpoint_reload_equal=True,cf_export_matches_selected=True,
        selected_probe=selected_probe,full_update_validation=full,curve=curve,runtime_seconds=time.monotonic()-began,
        cf_embeddings_sha256=sha(Path(output)/'refreshed_cf_embeddings.npy'),selected_checkpoint_sha256=sha(directory/'selected.pth'))
    save(directory/'summary.json',result);return result


def feature_only(output,data,p,binding,device):
    directory=Path(output)/'feature_only';directory.mkdir();n=len(binding['old_items'])
    model,state=teacher(output,p,device);q=frozen_quantizer(output,p,device)
    features=np.load(Path(output)/'refreshed_cf_embeddings.npy',allow_pickle=False)
    codes,all_metrics=quantize(q,features,device);old_codes,old_metrics=quantize(q,features[:n],device)
    np.save(directory/'semantic_codes.npy',old_codes);previous=np.load(Path(output)/'inputs/base/item_sids.npy',allow_pickle=False)
    mapping,capacity=versioned_mapping(previous,old_codes,p['quantizer']['capacity'])
    initial=np.load(Path(output)/'inputs/base/cf_embeddings.npy',allow_pickle=False).astype('float64');new=features[:n].astype('float64')
    delta=np.linalg.norm(new-initial,axis=1);cos=(new*initial).sum(1)/(np.linalg.norm(new,axis=1)*np.linalg.norm(initial,axis=1))
    raw_before=previous[:,:3]-np.arange(3)*p['quantizer']['capacity']
    result=dict(stage='feature_only',execution_status='completed',optimizer_steps=0,capacity=capacity,
        old_item_quantizer=old_metrics,all_cf_training_items_quantizer=all_metrics,
        old_embedding_drift=dict(mean_l2=float(delta.mean()),mean_relative_l2=float((delta/np.linalg.norm(initial,axis=1)).mean()),mean_cosine=float(cos.mean()),norm_percentiles={str(v):float(np.percentile(np.linalg.norm(new,axis=1),v)) for v in (0,25,50,95,100)}),
        semantic_changed_fraction=float(np.mean(np.any(raw_before!=old_codes,axis=1))),
        old_item_codebook_entries_used=[int(len(np.unique(old_codes[:,i]))) for i in range(3)],
        per_level_changed_fraction=np.mean(raw_before!=old_codes,axis=0).tolist(),codebooks_unchanged=True,teacher_unchanged=True,
        new_item_sids_allocated=False,anchor_train=None,outcome='capacity_infeasible' if mapping is None else 'measured')
    if mapping is not None:
        np.save(directory/'item_sids.npy',mapping)
        a,b,changed=paired_nll(model,data['anchors'],previous,mapping,p,device)
        reference=np.load(Path(output)/'baseline/old_anchor_nll.npy',allow_pickle=False)
        if not np.allclose(a,reference,rtol=0,atol=1e-6):raise ValueError('Frozen anchor reference changed between stages')
        rows=records(data,a,b,changed);write_rows(directory/'anchor_train_paired_losses.jsonl',rows)
        result['anchor_train']=damage(rows,p['bootstrap_replicates'],p['seed'])
        result['full_sid_changed_fraction']=float(np.mean(np.any(mapping!=previous,axis=1)))
    if not all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items()) or any(v.grad is not None or v.requires_grad for v in model.parameters()):
        raise ValueError('Teacher changed during feature-only diagnosis')
    save(directory/'summary.json',result);return result
