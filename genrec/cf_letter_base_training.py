"""Base-only training adapter; original temporal/TIGER training stays untouched."""
import contextlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from genrec.cf_letter_models import SASRec,CollaborativeRQ,GenerativeRetriever,make_trie
from genrec.cf_letter_protocol import allocate,gpt_arrays,metrics,read,save,sha


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:torch.cuda.set_rng_state_all(value['cuda'])


@contextlib.contextmanager
def evaluating(model):
    before=rng();training=model.training;model.eval()
    try:
        with torch.no_grad():yield
    finally:model.train(training);restore_rng(before)


class Sampler:
    def __init__(self,n,batch,seed):
        self.n=n;self.batch=batch;self.generator=torch.Generator().manual_seed(seed)
        self.order=torch.randperm(n,generator=self.generator);self.cursor=0;self.epochs=0;self.processed=0

    def take(self):
        if self.cursor==self.n:self.order=torch.randperm(self.n,generator=self.generator);self.cursor=0
        end=min(self.cursor+self.batch,self.n);ids=self.order[self.cursor:end].numpy();self.cursor=end;self.processed+=len(ids)
        done=end==self.n
        if done:self.epochs+=1
        return ids,done

    def state_dict(self):
        return dict(order=self.order.clone(),cursor=self.cursor,epochs=self.epochs,processed=self.processed,generator=self.generator.get_state())

    def load_state_dict(self,state):
        self.order=state['order'].clone();self.cursor=state['cursor'];self.epochs=state['epochs'];self.processed=state['processed'];self.generator.set_state(state['generator'])


class Counter:
    def __init__(self,directory,cap):
        self.path=Path(directory)/'optimizer_steps.json';self.cap=cap;self.step=0
        if self.path.exists():raise ValueError('Existing stage counter; never silently restart')
        self.write(None)

    def write(self,intent):
        save(self.path,dict(optimizer_steps_completed=self.step,optimizer_step_intent=intent,cap=self.cap))

    def update(self,optimizer):
        if self.step>=self.cap:raise ValueError('Optimizer budget exhausted')
        self.write(self.step+1);optimizer.step();self.step+=1;self.write(None)


def tensor(value,device):
    return torch.as_tensor(value,device=device)


def checkpoint(path,model,optimizer,sampler,counter,settings,binding):
    value=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),rng=rng(),step=counter.step,
        settings=settings,binding=binding,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'))
    path=Path(path);pending=path.with_suffix('.tmp');torch.save(value,pending);os.replace(pending,path)


def gradients(model,loss,p):
    if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['max_grad_norm'])
    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
    return float(norm)


def log(directory,counter,sampler,**values):
    record=dict(step=counter.step,epochs=sampler.epochs,processed_examples=sampler.processed,**values)
    with (Path(directory)/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    save(Path(directory)/'progress.json',record)
    if counter.step%100==0:print(json.dumps(record),flush=True)


def fixed_loss(model,data,indices,device,role,encoded=None):
    with evaluating(model):
        if role=='sasrec':
            targets=data['targets'][indices];n=model.items.num_embeddings-1
            negative=(targets-1+1+np.arange(len(indices))%(n-1))%n+1
            return float(model.loss(tensor(data['history'][indices],device),tensor(targets,device),tensor(negative,device)))
        prefix,mask,targets=encoded
        return float(model.nll(tensor(prefix[indices],device),tensor(mask[indices],device),tensor(targets[indices],device)).mean()/4)


def evaluate(model,data,indices,p,device,role,directory=None,mapping=None,catalog_size=None):
    chosen=[int(i) for i in indices if data['targets'][i]>0];all_records={}
    if role=='gpt':
        prefix,mask,target=gpt_arrays(data,mapping,p['quantizer']['capacity'],p['gpt']['user_buckets'])
        trie=make_trie(mapping.tolist());reverse={tuple(row):i+1 for i,row in enumerate(mapping.tolist())}
    with evaluating(model):
        for start in range(0,len(chosen),p['eval_batch_size']):
            idx=np.asarray(chosen[start:start+p['eval_batch_size']])
            if role=='sasrec':
                hidden=model(tensor(data['history'][idx],device));scores=hidden@model.items.weight[1:catalog_size+1 if catalog_size is not None else None].T
                predictions=scores.topk(max(p['topk']),dim=-1).indices+1
                nll=torch.nn.functional.cross_entropy(scores,tensor(data['targets'][idx]-1,device),reduction='none')
                predicted=predictions.cpu().tolist()
            else:
                ids=tensor(prefix[idx],device);attention=tensor(mask[idx],device)
                nll=model.nll(ids,attention,tensor(target[idx],device))
                seq,_=model.beam(ids,attention,trie,p['beam'])
                predicted=[[reverse[tuple(row)] for row in values] for values in seq.cpu().tolist()]
            for j,i in enumerate(idx):
                all_records[int(i)]=dict(source_row=int(i),sample_id=str(data['samples'][i]),target=int(data['targets'][i]),
                    predictions=predicted[j],nll=float(nll[j]))
    records=[all_records.get(int(i),dict(source_row=int(i),sample_id=str(data['samples'][i]),target=0,predictions=[],nll=None)) for i in indices]
    result=metrics(records,p['topk']);result['nll_target_tokens']=1 if role=='sasrec' else 4
    result['examples_with_all_history_unavailable']=int(sum(np.all(data['history'][i]==0) for i in indices))
    result['num_beams']=None if role=='sasrec' else p['beam']
    if directory:
        with Path(directory).open('x') as stream:
            for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
    return result


def train_predictor(role,output,data,p,binding,device,mapping=None):
    directory=Path(output)/role;directory.mkdir()
    count=len(binding['items']);config=p[role]
    model=(SASRec(count,p['history'],config) if role=='sasrec' else
        GenerativeRetriever(p['history'],p['quantizer']['capacity'],config)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    sampler=Sampler(len(data['base_train']['targets']),p['batch_size'],p['seed'])
    counter=Counter(directory,config['step_cap']);training=data['base_train'];val=data['base_val'];probe=data['probe']
    encoded=gpt_arrays(training,mapping,p['quantizer']['capacity'],p['gpt']['user_buckets']) if role=='gpt' else None
    monitor=np.arange(min(p['probe_size'],len(training['targets'])))
    initial_loss=fixed_loss(model,training,monitor,device,role,encoded)
    initial=evaluate(model,val,probe,p,device,role,mapping=mapping)
    checkpoint(directory/'selected.pth',model,optimizer,sampler,counter,p,binding)
    best=dict(step=0,epoch=0,metrics=initial);curve=[dict(step=0,epoch=0,training_probe_loss=initial_loss,validation=initial)]
    began=time.monotonic()
    while counter.step<config['step_cap'] and sampler.epochs<config['epochs']:
        ids,epoch_done=sampler.take();model.train();optimizer.zero_grad(set_to_none=True)
        if role=='sasrec':
            y=tensor(training['targets'][ids],device)
            neg=torch.randint(1,count,y.shape,device=device);neg=neg+(neg>=y)
            loss=model.loss(tensor(training['history'][ids],device),y,neg)
        else:
            x,m,y=encoded
            loss=model.nll(tensor(x[ids],device),tensor(m[ids],device),tensor(y[ids],device)).mean()/4
        norm=gradients(model,loss,p);counter.update(optimizer)
        if counter.step%20==0 or epoch_done:log(directory,counter,sampler,loss=float(loss),gradient_norm=norm)
        if epoch_done and (sampler.epochs%config['eval_every_epochs']==0 or sampler.epochs==config['epochs']):
            result=evaluate(model,val,probe,p,device,role,mapping=mapping)
            train_loss=fixed_loss(model,training,monitor,device,role,encoded)
            curve.append(dict(step=counter.step,epoch=sampler.epochs,training_probe_loss=train_loss,validation=result))
            save(directory/'curve.json',curve)
            print(json.dumps(dict(stage=role,step=counter.step,epoch=sampler.epochs,validation=result)),flush=True)
            if result['full_denominator']['ndcg@10']>best['metrics']['full_denominator']['ndcg@10']:
                checkpoint(directory/'selected.pth',model,optimizer,sampler,counter,p,binding)
                best=dict(step=counter.step,epoch=sampler.epochs,metrics=result)
        if epoch_done or counter.step%1000==0:checkpoint(directory/'last.pth',model,optimizer,sampler,counter,p,binding)
    # Always preserve the terminal training state separately from selected weights.
    checkpoint(directory/'last.pth',model,optimizer,sampler,counter,p,binding)
    selected=torch.load(directory/'selected.pth',map_location=device,weights_only=False)
    model.load_state_dict(selected['model']);model.eval()
    if not all(torch.equal(v.cpu(),selected['model'][k].cpu()) for k,v in model.state_dict().items()):raise ValueError('Selected checkpoint reload differs')
    final_loss=fixed_loss(model,training,monitor,device,role,encoded)
    selected_probe=evaluate(model,val,probe,p,device,role,directory/'selected_probe_queries.jsonl',mapping)
    for key in ('full_denominator','conditional_on_servable','mean_sequence_nll_servable'):
        a,b=best['metrics'][key],selected_probe[key]
        if isinstance(a,dict):
            if a!=b:raise ValueError('Selected checkpoint ranking does not reproduce')
        elif not math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-6):raise ValueError('Selected checkpoint NLL does not reproduce')
    full=evaluate(model,val,np.arange(len(val['targets'])),p,device,role,directory/'full_validation_queries.jsonl',mapping)
    ready={}
    if role=='sasrec':
        features=model.items.weight[1:].detach().cpu().numpy().astype('float32')
        np.save(Path(output)/'cf_embeddings.npy',features)
        ready['export_matches_selected_weights']=bool(np.array_equal(features,selected['model']['items.weight'][1:].cpu().numpy()))
        ready['nonzero_item_vectors']=bool(np.all(np.linalg.norm(features,axis=1)>0))
    else:
        ids=[int(i) for i in probe if val['targets'][i]>0][:16]
        x,m,y=gpt_arrays(val,mapping,p['quantizer']['capacity'],p['gpt']['user_buckets'])
        px=tensor(x[ids],device);ma=tensor(m[ids],device);ta=tensor(y[ids],device)
        before=model.nll(px,ma,ta).detach()
        model.zero_grad(set_to_none=True);model.requires_grad_(False)
        embedding=model.model.get_input_embeddings()(torch.cat([px,ta],dim=1)).detach().requires_grad_(True)
        via_embed=model.nll(px,ma,ta,embedding)
        ready['hard_id_embedding_logits_equal']=bool(torch.allclose(before,via_embed,atol=2e-5,rtol=2e-6))
        via_embed.mean().backward()
        ready['history_embedding_gradient_nonzero']=bool(torch.isfinite(embedding.grad).all() and embedding.grad[:,:px.shape[1]].norm()>0)
        ready['teacher_parameters_frozen']=all(v.grad is None for v in model.parameters())
        save(directory/'reload_samples.json',dict(source_rows=ids,nll=before.cpu().tolist()))
    if not all(ready.values()):raise ValueError('Base export/GR-input readiness failed')
    report=dict(stage=role,execution_status='completed',optimizer_steps=counter.step,processed_examples=sampler.processed,
        selected_step=best['step'],selected_epoch=best['epoch'],initial_training_probe_loss=initial_loss,
        selected_training_probe_loss=final_loss,selected_probe=selected_probe,full_base_validation=full,
        checkpoint_reload_equal=True,readiness=ready,curve=curve,parameters=sum(v.numel() for v in model.parameters()),
        runtime_seconds=time.monotonic()-began,selected_checkpoint_sha256=sha(directory/'selected.pth'),last_checkpoint_sha256=sha(directory/'last.pth'),
        quality_gate=best['step']>0 and final_loss<initial_loss and full['full_denominator']['recall@10']>0)
    save(directory/'summary.json',report)
    return report


def train_quantizer(output,p,binding,device):
    directory=Path(output)/'quantizer';directory.mkdir()
    features=np.load(Path(output)/'cf_embeddings.npy',allow_pickle=False)
    if features.dtype!=np.float32 or features.shape!=(len(binding['items']),p['quantizer']['dim']) or not np.isfinite(features).all():raise ValueError('Malformed CF export')
    feature_hash=sha(Path(output)/'cf_embeddings.npy')
    model=CollaborativeRQ(tensor(features,device),p['quantizer']).to(device);model.initialize()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    sampler=Sampler(len(features),p['quantizer']['batch_size'],p['seed']);counter=Counter(directory,p['quantizer']['steps'])
    initial,_=model.audit();curve=[dict(step=0,**initial)];repairs=[0]*3
    checkpoint(directory/'last_valid.pth',model,optimizer,sampler,counter,p,binding)
    allocate(model.assignments().cpu().numpy(),p['quantizer']['capacity'])
    began=time.monotonic()
    while counter.step<p['quantizer']['steps']:
        indices,_=sampler.take();optimizer.zero_grad(set_to_none=True)
        loss,recon,q,_=model(model.features[tensor(indices,device)])
        norm=gradients(model,loss,p);counter.update(optimizer)
        fixed=model.repair();repairs=[a+b for a,b in zip(repairs,fixed)]
        codes=model.assignments().cpu().numpy()
        try:mapping,collision=allocate(codes,p['quantizer']['capacity'])
        except ValueError:
            checkpoint(directory/'rejected.pth',model,optimizer,sampler,counter,p,binding)
            raise
        checkpoint(directory/'last_valid.pth',model,optimizer,sampler,counter,p,binding)
        if counter.step%20==0:log(directory,counter,sampler,loss=float(loss),reconstruction_loss=float(recon),quantization_loss=float(q),gradient_norm=norm,dead_reinitialized=fixed,**collision)
        if counter.step%p['quantizer']['monitor_every']==0:
            values,_=model.audit();curve.append(dict(step=counter.step,**values));save(directory/'curve.json',curve)
            checkpoint(directory/'last.pth',model,optimizer,sampler,counter,p,binding)
    checkpoint(directory/'selected.pth',model,optimizer,sampler,counter,p,binding)
    before=model.assignments();saved=torch.load(directory/'selected.pth',map_location=device,weights_only=False)
    model.load_state_dict(saved['model']);after=model.assignments()
    if not torch.equal(before,after) or sha(Path(output)/'cf_embeddings.npy')!=feature_hash:raise ValueError('Quantizer reload or frozen CF export differs')
    final,codes=model.audit();mapping,collision=allocate(codes.cpu().numpy(),p['quantizer']['capacity'])
    np.save(Path(output)/'item_sids.npy',mapping)
    np.save(directory/'semantic_codes.npy',codes.cpu().numpy())
    k=p['quantizer']['capacity']
    save(Path(output)/'token_manifest.json',dict(semantic_books=3,book_capacity=k,suffix_capacity=k,sid_length=4,
        positional_offsets=[0,k,2*k,3*k],user_offset=4*k,user_buckets=p['gpt']['user_buckets'],pad_token=4*k+p['gpt']['user_buckets']+9,
        history_slots=p['history'],target_tokens=4,target_eos=False,item_padding_row=0,raw_position_ids=True))
    report=dict(stage='quantizer',execution_status='completed',optimizer_steps=counter.step,processed_examples=sampler.processed,
        initial=initial,final=final,collision=collision,dead_code_reinitializations=repairs,curve=curve,
        cf_embeddings_sha256=feature_hash,checkpoint_reload_assignments_equal=True,
        selected_checkpoint_sha256=sha(directory/'selected.pth'),mapping_sha256=sha(Path(output)/'item_sids.npy'),
        zero_predictor_mse=float(np.square(features.astype('float64')).mean()),
        mean_predictor_mse=float(np.square(features.astype('float64')-features.astype('float64').mean(0)).mean()),
        norm_percentiles={str(q):float(np.percentile(np.linalg.norm(features.astype('float64'),axis=1),q)) for q in (0,25,50,75,95,99,100)},
        runtime_seconds=time.monotonic()-began)
    report['quality_gate']=final['reconstruction_mse']<report['mean_predictor_mse'] and collision['unique_full_sids']==len(features)
    save(directory/'summary.json',report);return report
