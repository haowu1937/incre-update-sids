"""Hard-forward/soft-backward CF-codebook diagnostic; finite-window training diagnostic."""
import json
import os
import time
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from genrec.cf_letter_base_training import Counter,Sampler,rng,tensor
from genrec.cf_letter_models import CollaborativeRQ
from genrec.cf_letter_protocol import gpt_arrays,read,save,sha
from genrec.cf_letter_refresh_protocol import damage,versioned_mapping
from genrec.cf_letter_refresh_training import teacher,frozen_quantizer,quantize,records,write_rows
from genrec.cf_letter_history_long_protocol import cases,indices_sha,tail_stability,paired_assessment


@torch.no_grad()
def distance_scales(model,features,cap=4096):
    values=[[] for _ in model.books]
    for x in features[:cap].split(512):
        r=x.clone()
        for i,book in enumerate(model.books):
            d=r.square().sum(-1,keepdim=True)+book.square().sum(-1)[None]-2*r@book.T
            values[i].append(d.std(1,unbiased=False).cpu());r-=book[d.argmin(-1)]
    return [max(float(torch.cat(v).median()),1e-8) for v in values]


def surrogate_tokens(model,features,hard_codes,table,scales,temperature):
    """Exactly hard embeddings forward; suffix allocation has no surrogate gradient."""
    residual=features;out=[];capacity=len(model.books[0])
    for i,book in enumerate(model.books):
        d=residual.square().sum(-1,keepdim=True)+book.square().sum(-1)[None]-2*residual@book.T
        probability=F.softmax(-d/(scales[i]*temperature),dim=-1)
        soft=probability@table[i*capacity:(i+1)*capacity]
        hard=F.embedding(hard_codes[:,i]+i*capacity,table)
        out.append(hard+(soft-soft.detach()))
        soft_word=probability@book;hard_word=book[hard_codes[:,i]].detach()
        residual=residual-(hard_word+(soft_word-soft_word.detach()))
    return torch.stack(out,dim=1)


def gradient_stats(rq,hg,weight,displacement=None):
    def one(indices):
        a=sum(float(rq[i].double().square().sum()) for i in indices);b=sum(float(hg[i].double().square().sum()) for i in indices)
        dot=sum(float((rq[i].double()*hg[i].double()).sum()) for i in indices);an=math.sqrt(a);bn=math.sqrt(b)
        result=dict(rq_gradient_norm=an,history_gradient_norm=bn,weighted_history_over_rq=weight*bn/an if an else None,
            cosine=max(-1.,min(1.,dot/(an*bn))) if an and bn else None)
        if displacement is not None:result.update(update_norm=math.sqrt(sum(float(displacement[i].double().square().sum()) for i in indices)),history_dot_actual_update=sum(float((hg[i].double()*displacement[i].double()).sum()) for i in indices))
        return result
    return dict(all=one(range(len(rq))),**{'book_'+str(i+1):one([i]) for i in range(len(rq))})


def public_curves(result,trajectory):
    return dict(case=result['case'],fixed=[dict(step=v['step'],H=v['post_update_fixed_H']) for v in trajectory if v.get('post_update_fixed_H') is not None],
        full=[dict(step=v['step'],epoch=v['epoch'],scheduled_epoch_monitor=v['scheduled_epoch_monitor'],H=v['anchor_train']['H'],
            mean_delta_sequence_nll=v['anchor_train']['mean_delta_sequence_nll'],rq=v['rq'],old_item_reconstruction_mse=v['old_item_reconstruction_mse'],
            capacity={k:v['capacity'][k] for k in ('full_sid_allocation_feasible','maximum_current_collision_bucket','maximum_required_suffix_capacity')},
            gradients=v['gradients']['all']) for v in result['monitors']],stability=result['stability'])


def save_checkpoint(path,model,optimizer,content,anchor,counter,settings,binding):
    value=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
        sampler=content.state_dict(),anchor_sampler=anchor.state_dict(),rng=rng(),step=counter.step,settings=settings,binding=binding,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'))
    path=Path(path);pending=path.with_suffix('.tmp');torch.save(value,pending);os.replace(pending,path)


class Engine:
    def __init__(self,output,data,binding,p,device):
        self.output=Path(output);self.data=data;self.binding=binding;self.p=p;self.device=device;self.n=len(binding['old_items'])
        self.teacher,self.teacher_state=teacher(output,p,device);old=frozen_quantizer(output,p,device)
        self.scales=distance_scales(old,old.features,p['distance_scale_items_cap']);save(self.output/'distance_scales.json',self.scales)
        features=np.load(self.output/'inputs/refresh/refreshed_cf_embeddings.npy',allow_pickle=False)
        if features.shape!=(len(binding['items']),p['quantizer']['dim']) or not np.isfinite(features).all():raise ValueError('Unexpected fixed E1 shape')
        self.model=CollaborativeRQ(tensor(features,device),p['quantizer']).to(device)
        state=old.state_dict();state['features']=self.model.features.detach().clone();self.model.load_state_dict(state)
        if any(not torch.equal(a,b) for a,b in zip(self.model.books,old.books)):raise ValueError('Codebooks did not warm start exactly')
        self.initial={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        self.previous=np.load(self.output/'inputs/base/item_sids.npy',allow_pickle=False)
        self.reference=np.load(self.output/'inputs/refresh/baseline/old_anchor_nll.npy',allow_pickle=False)
        self.old_ids,self.mask,self.targets=gpt_arrays(data['anchors'],self.previous,p['quantizer']['capacity'],p['gpt']['user_buckets'])
        generator=torch.Generator().manual_seed(p['seed']+77)
        self.anchor_order=torch.randperm(len(self.reference),generator=generator).numpy()
        self.fixed=self.anchor_order[:p['fixed_anchor_queries']]
        sampler=Sampler(len(features),p['content_batch_size'],p['seed']);self.content=[sampler.take()[0] for _ in range(p['gradient_probe_batches'])]
        save(self.output/'sampling.json',dict(fixed_anchor_indices=self.fixed.tolist(),gradient_anchor_order=self.anchor_order[:p['gradient_probe_batches']*p['fixed_anchor_queries']].tolist(),calibration_content_sha256=[indices_sha(v) for v in self.content],content_seed=p['seed'],anchor_seed=p['seed']+78))
        self.initial_codes,self.initial_mapping,self.initial_capacity=self.mapping()
        if self.initial_mapping is None or not np.array_equal(self.initial_mapping,np.load(self.output/'inputs/refresh/feature_only/item_sids.npy',allow_pickle=False)) or not np.array_equal(self.initial_codes,np.load(self.output/'inputs/refresh/feature_only/semantic_codes.npy',allow_pickle=False)):raise ValueError('Initial C0(E1) assignments differ from accepted diagnostic')
        initial=self.output/'initial';initial.mkdir();self.initial_full=self.score(np.arange(len(self.reference)),self.initial_mapping,initial/'full_anchor.jsonl')
        self.initial_fixed=self.score(self.fixed,self.initial_mapping,initial/'fixed_anchor.jsonl')
        if abs(self.initial_full['H']-p['refresh_predecessor']['expected_initial_full_H'])>1e-6 or self.initial_fixed['H']<=0:raise ValueError('Require reproduced positive-H initial state')
        with torch.no_grad():
            old=[]
            for first in range(0,len(self.reference),p['eval_batch_size']):
                take=slice(first,first+p['eval_batch_size'])
                old.extend(self.teacher.nll(tensor(self.old_ids[take],device),tensor(self.mask[take],device),tensor(self.targets[take],device)).cpu().tolist())
            if not np.allclose(old,self.reference,rtol=0,atol=1e-6):raise ValueError('Original Base reference NLL does not reproduce')
            h,nll,hard,embeds=self.historical(self.fixed,self.initial_codes,self.initial_mapping)
            direct=self.teacher.nll(hard,tensor(self.mask[self.fixed],device),tensor(self.targets[self.fixed],device))
            if not torch.allclose(nll,direct,rtol=2e-6,atol=2e-5) or abs(float(h)-self.initial_fixed['H'])>1e-5:raise ValueError('STE hard forward does not match direct-ID hard H')
        self.initial_rq=self.rq_monitor(initial/'catalog_codes.npy')
        self.steps=0

    @torch.no_grad()
    def mapping(self):
        codes,_=quantize(self.model,self.model.features[:self.n],self.device)
        mapping,capacity=versioned_mapping(self.previous,codes,self.p['quantizer']['capacity'])
        return codes,mapping,capacity

    @torch.no_grad()
    def rq_monitor(self,path):
        result,codes=self.model.audit();np.save(path,codes.cpu().numpy());return result

    def score(self,indices,mapping,path):
        selected={k:v[indices] for k,v in self.data['anchors'].items()}
        x,mask,_=gpt_arrays(selected,mapping,self.p['quantizer']['capacity'],self.p['gpt']['user_buckets'])
        if not np.array_equal(mask,self.mask[indices]):raise ValueError('Anchor mask changed')
        new=[]
        with torch.no_grad():
            for first in range(0,len(indices),self.p['eval_batch_size']):
                take=slice(first,first+self.p['eval_batch_size']);local=indices[take]
                new.extend(self.teacher.nll(tensor(x[take],self.device),tensor(mask[take],self.device),tensor(self.targets[local],self.device)).cpu().tolist())
        changed=np.any(x!=self.old_ids[indices],axis=1);old=self.reference[indices];new=np.where(changed,np.asarray(new),old)
        subset=dict(anchors=selected,anchor_users=self.data['anchor_users'][indices],anchor_source_rows=self.data['anchor_source_rows'][indices])
        values=records(subset,old,new,changed);write_rows(path,values)
        return damage(values,self.p['bootstrap_replicates'],self.p['seed'])

    def historical(self,indices,codes,mapping):
        selected={k:v[indices] for k,v in self.data['anchors'].items()}
        x,mask,_=gpt_arrays(selected,mapping,self.p['quantizer']['capacity'],self.p['gpt']['user_buckets'])
        hard=tensor(x,self.device);targets=tensor(self.targets[indices],self.device);table=self.teacher.model.get_input_embeddings().weight.detach()
        joined=torch.cat([hard,targets],dim=1);embeds=F.embedding(joined,table)
        history=tensor(selected['history'],self.device);positions=torch.nonzero(history>0,as_tuple=False)
        unique,inverse=torch.unique(history[history>0],return_inverse=True)
        values=surrogate_tokens(self.model,self.model.features[unique-1],tensor(codes,self.device)[unique-1],table,self.scales,self.p['temperature'])
        flat=(positions[:,0,None]*joined.shape[1]+1+4*positions[:,1,None]+torch.arange(3,device=self.device)).reshape(-1)
        embeds=embeds.reshape(-1,embeds.shape[-1]).index_copy(0,flat,values[inverse].flatten(0,1)).view_as(embeds)
        nll=self.teacher.nll(hard,tensor(mask,self.device),targets,embeds)
        changed=tensor(np.any(x!=self.old_ids[indices],axis=1),self.device)
        delta=torch.where(changed,nll-tensor(self.reference[indices],self.device).to(nll.dtype),torch.zeros_like(nll))
        return delta.relu().mean(),nll,hard,embeds

    def gradients(self,content,anchors,codes,mapping):
        parameters=list(self.model.parameters());loss,recon,quant,_=self.model(self.model.features[tensor(content,self.device)])
        rq=torch.autograd.grad(loss,parameters,allow_unused=True)
        h,nll,hard,embeds=self.historical(anchors,codes,mapping)
        hg=torch.autograd.grad(h,parameters,allow_unused=True)
        unused=[g is None for g in hg]
        rq=[torch.zeros_like(v) if g is None else g.detach() for v,g in zip(parameters,rq)]
        hg=[torch.zeros_like(v) if g is None else g.detach() for v,g in zip(parameters,hg)]
        if not all(torch.isfinite(g).all() for g in rq+hg):raise ValueError('Nonfinite diagnostic gradient')
        changed=np.any(hard.detach().cpu().numpy()!=self.old_ids[anchors],axis=1)
        measured=np.where(changed,nll.detach().cpu().numpy().astype(float),self.reference[anchors])
        return rq,hg,dict(rq_loss=float(loss),reconstruction_loss=float(recon),quantization_loss=float(quant),pre_update_damage_H=float(h),unused_history_books=unused,
            train_anchor_new_nll=measured.tolist())

    def calibrate(self):
        probes=[];ratios=[];b=self.p['fixed_anchor_queries']
        for i in range(self.p['gradient_probe_batches']):
            indices=self.anchor_order[i*b:(i+1)*b]
            rq,hg,values=self.gradients(self.content[i%len(self.content)],indices,self.initial_codes,self.initial_mapping)
            stats=gradient_stats(rq,hg,1.)['all'];probe=dict(probe=i,**values,gradients=gradient_stats(rq,hg,1.));probes.append(probe)
            if values['pre_update_damage_H']>0 and stats['history_gradient_norm']>0 and stats['rq_gradient_norm']>0:
                ratios.append(stats['rq_gradient_norm']/stats['history_gradient_norm'])
        weight=.1*float(np.median(ratios)) if len(ratios)==self.p['gradient_probe_batches'] else None
        result=dict(probes=probes,history_weight=weight,rule='0.1 * median(g_Q norm / g_H norm); all four training probes must be finite nonzero',all_history_gradient_zero=all(v['gradients']['all']['history_gradient_norm']==0 for v in probes),temperature=self.p['temperature'],distance_scales=self.scales)
        save(self.output/'gradient_calibration.json',result);return result

    def full_monitor(self,directory,step,epoch,codes,mapping,capacity,stats,scheduled):
        full=self.score(np.arange(len(self.reference)),mapping,directory/f'full_{step}.jsonl')
        rq=self.rq_monitor(directory/f'catalog_codes_{step}.npy')
        with torch.no_grad():
            restored=sum(book[tensor(codes[:,i],self.device)] for i,book in enumerate(self.model.books))
            old_mse=float((restored-self.model.features[:self.n]).square().mean())
        semantic=float(np.mean(np.any(codes!=self.initial_codes,axis=1)))
        sid=float(np.mean(np.any(mapping!=self.initial_mapping,axis=1)))
        np.save(directory/f'item_sids_{step}.npy',mapping.astype(np.int16))
        return dict(step=step,epoch=epoch,scheduled_epoch_monitor=scheduled,anchor_train=full,rq=rq,
            old_item_reconstruction_mse=old_mse,semantic_churn_from_feature_only=semantic,full_sid_churn_from_feature_only=sid,
            capacity=capacity,gradients=stats)

    def branch(self,case,weight):
        self.model.load_state_dict(self.initial);self.model.zero_grad(set_to_none=True)
        directory=self.output/case['name'];directory.mkdir();began=time.monotonic()
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=case['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
        counter=Counter(directory,case['steps']);sampler=Sampler(len(self.model.features),self.p['content_batch_size'],self.p['seed'])
        anchors=Sampler(len(self.reference),self.p['anchor_batch_size'],self.p['seed']+78)
        def snapshot(name):save_checkpoint(directory/name,self.model,optimizer,sampler,anchors,counter,self.p,self.binding)
        snapshot('initial.pth');snapshot('last_valid.pth')
        result=dict(case=case,optimizer_steps=0,capacity_feasible=True,initial_H=self.initial_full['H'],initial_rq=self.initial_rq,monitors=[],outcome='completed')
        trajectory=[];codes=self.initial_codes;mapping=self.initial_mapping
        for step in range(1,case['steps']+1):
            if self.steps>=self.p['optimizer_step_cap']:raise ValueError('Global diagnostic budget exhausted')
            content,done=sampler.take();batch,_=anchors.take()
            rq,hg,values=self.gradients(content,batch,codes,mapping);before=[v.detach().clone() for v in self.model.parameters()]
            objective=case['objective'];optimizer.zero_grad(set_to_none=True)
            for parameter,a,b in zip(self.model.parameters(),rq,hg):parameter.grad=b.clone() if objective=='h_only' else a.clone() if objective=='rq_only' else a+weight*b
            actual_loss=values['pre_update_damage_H'] if objective=='h_only' else values['rq_loss'] if objective=='rq_only' else values['rq_loss']+weight*values['pre_update_damage_H']
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.p['max_grad_norm'])
            if not torch.isfinite(norm):raise ValueError('Nonfinite clipped gradient')
            counter.update(optimizer);self.steps+=1;result['optimizer_steps']=counter.step
            displacement=[v.detach()-a for v,a in zip(self.model.parameters(),before)]
            stats=gradient_stats(rq,hg,weight,displacement);codes,mapping,capacity=self.mapping()
            np.save(directory/f'semantic_codes_{step}.npy',codes.astype(np.int16))
            entry=dict(step=step,epoch=sampler.epochs,processed_content_examples=sampler.processed,anchor_epoch=anchors.epochs,processed_anchor_examples=anchors.processed,
                content_indices_sha256=indices_sha(content),anchor_indices_sha256=indices_sha(batch),**values,objective_loss=actual_loss,
                gradient_norm_before_clip=float(norm),clip_coefficient=min(1.,self.p['max_grad_norm']/(float(norm)+1e-6)),gradient_components=stats,capacity=capacity)
            if mapping is None:
                snapshot('rejected.pth');entry['post_update_fixed_H']=None;result.update(capacity_feasible=False,outcome='capacity_infeasible')
            else:
                entry['full_sid_changed_fraction']=float(np.mean(np.any(mapping!=self.initial_mapping,axis=1)))
                if step%self.p['fixed_monitor_every_steps']==0 or step==case['steps']:
                    fixed=self.score(self.fixed,mapping,directory/f'fixed_{step}.jsonl');entry['post_update_fixed_H']=fixed['H']
                scheduled=done and sampler.epochs%self.p['full_monitor_every_epochs']==0
                if scheduled or step==case['steps']:
                    monitor=self.full_monitor(directory,step,sampler.epochs,codes,mapping,capacity,stats,scheduled)
                    result['monitors'].append(monitor);entry['full_anchor_train_H']=monitor['anchor_train']['H'];snapshot(f'monitor_{step}.pth')
                snapshot('last_valid.pth')
            trajectory.append(entry)
            with (directory/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(entry,allow_nan=False)+'\n');handle.flush()
            save(directory/'progress.json',entry)
            if step%10==0 or not result['capacity_feasible']:
                print(json.dumps(dict(branch=case['name'],step=step,epoch=sampler.epochs,train_batch_H=values['pre_update_damage_H'],fixed_H=entry.get('post_update_fixed_H'),full_H=entry.get('full_anchor_train_H'))),flush=True)
            if not result['capacity_feasible']:break
        snapshot('last.pth')
        result.update(last_fixed_H=next((v['post_update_fixed_H'] for v in reversed(trajectory) if v.get('post_update_fixed_H') is not None),None),
            final_full_H=result['monitors'][-1]['anchor_train']['H'] if result['monitors'] else None,
            final_full_monitor_step=result['monitors'][-1]['step'] if result['monitors'] else None,
            changed_any_full_sid=any(v.get('full_sid_changed_fraction',0)>0 for v in trajectory),
            stability=tail_stability(result['monitors'],self.p['stability']),runtime_seconds=time.monotonic()-began,
            clipped_steps=sum(v['clip_coefficient']<1 for v in trajectory),zero_history_gradient_steps=sum(v['gradient_components']['all']['history_gradient_norm']==0 for v in trajectory))
        save(directory/'trajectory.json',trajectory);save(directory/'summary.json',result)
        save(self.output/(case['objective']+'_curves.json'),public_curves(result,trajectory))
        return result

    def run(self):
        calibration=self.calibrate();branches=[]
        if calibration['history_weight'] is None:outcome='no_usable_history_gradient'
        else:
            for case in cases(self.p):
                if (self.output/'stop-new-branches.json').exists():break
                result=self.branch(case,calibration['history_weight']);branches.append(result)
                save(self.output/'branch_progress.json',dict(completed_branches=len(branches),optimizer_steps=self.steps))
            outcome='three_arm_diagnostic_completed' if len(branches)==3 else 'resource_stopped'
            if any(not v['capacity_feasible'] for v in branches):outcome='capacity_infeasible'
        if not all(torch.equal(v.cpu(),self.teacher_state[k]) for k,v in self.teacher.state_dict().items()) or any(v.grad is not None or v.requires_grad for v in self.teacher.parameters()):raise ValueError('Frozen GPT changed')
        if not torch.equal(self.model.features.cpu(),self.initial['features']):raise ValueError('Fixed E1 changed')
        return dict(outcome=outcome,initial_full_anchor_train=self.initial_full,initial_fixed_anchor=self.initial_fixed,initial_rq=self.initial_rq,initial_capacity=self.initial_capacity,
            history_weight=calibration['history_weight'],branches=[{k:v for k,v in b.items() if k!='monitors'} for b in branches],optimizer_steps_spent=self.steps,
            paired_assessment=paired_assessment(branches,self.p['stability']),teacher_unchanged=True,cf_embeddings_unchanged=True,
            same_start_and_sampling=True,old_reference_reproduced=True,lambda_sweep_performed=False)
