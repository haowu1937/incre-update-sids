"""Fixed-budget GPT adaptation and current-map recommendation; no quantizer updates."""
import gc
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from genrec.cf_letter_models import GenerativeRetriever,make_trie
from genrec.cf_letter_base_training import seed_all,Sampler,Counter,tensor,evaluating,checkpoint,gradients
from genrec.cf_letter_protocol import gpt_arrays,read,save,sha
from genrec.cf_letter_downstream_protocol import aggregate,select


def model_from(path,p,device,base=False):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if base and (saved['step']!=p['predecessor']['gpt_step'] or saved['code_sha']!=p['predecessor']['code_sha']):raise ValueError('Wrong Base GPT warm start')
    model=GenerativeRetriever(p['history'],p['quantizer']['capacity'],p['gpt']).to(device);model.load_state_dict(saved['model'])
    if not all(torch.equal(v.cpu(),saved['model'][k]) for k,v in model.state_dict().items()):raise ValueError('Checkpoint reload differs')
    return model,saved


def save_weights(path,model,step,p,mapping_sha):
    save_dict=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},step=step,settings=p,
        mapping_sha256=mapping_sha,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'))
    path=Path(path);pending=path.with_suffix('.tmp');torch.save(save_dict,pending);os.replace(pending,path)


def score(model,data,indices,mapping,p,device,path,progress=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    x,mask,y=gpt_arrays(data,mapping,p['quantizer']['capacity'],p['gpt']['user_buckets'])
    trie=make_trie(mapping.tolist());reverse={tuple(row):i+1 for i,row in enumerate(mapping.tolist())};indices=np.asarray(indices,dtype=np.int64)
    records=[]
    with evaluating(model),path.open('x') as handle:
        for first in range(0,len(indices),p['eval_batch_size']):
            batch=indices[first:first+p['eval_batch_size']];valid=batch[data['targets'][batch]>0];values={}
            if len(valid):
                ids=tensor(x[valid],device);attention=tensor(mask[valid],device)
                nll=model.nll(ids,attention,tensor(y[valid],device)).cpu().tolist()
                sequence,_=model.beam(ids,attention,trie,p['beam'])
                for i,n,seq in zip(valid,nll,sequence.cpu().tolist()):values[int(i)]=(float(n),[reverse[tuple(row)] for row in seq])
            for i in batch:
                nll,predictions=values.get(int(i),(None,[]))
                row=dict(source_row=int(data['source_rows'][i]),sample_id=str(data['samples'][i]),user=str(data['raw_users'][i]),
                    target=int(data['targets'][i]),predictions=predictions,nll=nll,all_history_unavailable=bool(np.all(data['history'][i]==0)))
                handle.write(json.dumps(row,allow_nan=False)+'\n');records.append(row)
            if progress and first%(p['eval_batch_size']*50)==0:progress(first+len(batch),len(indices))
    return aggregate(records,p)


def train_arm(output,case,train,val,probe,p,device,observe):
    output=Path(output);directory=output/'arms'/case['name'];directory.mkdir(parents=True,exist_ok=True)
    seed_all(p['seed']);model,base=model_from(output/'inputs/base/gpt/selected.pth',p,device,True)
    model.requires_grad_(True);mapping_path=output/'mappings'/(case['name']+'.npy');mapping=np.load(mapping_path,allow_pickle=False);mapping_sha=sha(mapping_path)
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],betas=tuple(p['betas']),eps=p['eps'],weight_decay=p['weight_decay'])
    if optimizer.state:raise ValueError('Optimizer must reset')
    sampler=Sampler(len(train['targets']),p['batch_size'],p['seed']);counter=Counter(directory,p['steps_per_arm'])
    checkpoint(directory/'initial.pth',model,optimizer,sampler,counter,p,{'mapping_sha256':mapping_sha})
    x,mask,y=gpt_arrays(train,mapping,p['quantizer']['capacity'],p['gpt']['user_buckets']);curve=[];began=time.monotonic()
    for step in range(p['steps_per_arm']+1):
        if step:
            ids,done=sampler.take();model.train();optimizer.zero_grad(set_to_none=True)
            loss=model.nll(tensor(x[ids],device),tensor(mask[ids],device),tensor(y[ids],device)).mean()/4
            norm=gradients(model,loss,p);counter.update(optimizer)
            entry=dict(step=step,epoch=sampler.epochs,processed_examples=sampler.processed,
                source_rows=train['source_rows'][ids].tolist(),token_ce=float(loss),gradient_norm=norm)
            with (directory/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(entry,allow_nan=False)+'\n');handle.flush()
            if step%100==0:
                checkpoint(directory/'last.pth',model,optimizer,sampler,counter,p,{'mapping_sha256':mapping_sha})
                print(json.dumps(dict(phase='gpt_update',arm=case['name'],step=step,token_ce=float(loss),processed_examples=sampler.processed)),flush=True)
            if step%25==0:observe()
        if step in p['selection_steps']:
            path=directory/('candidate_'+str(step)+'.pth');save_weights(path,model,step,p,mapping_sha)
            metric=score(model,val,probe,mapping,p,device,directory/('probe_'+str(step)+'.jsonl'))
            curve.append(dict(step=step,metrics=metric,checkpoint_sha256=sha(path)));save(directory/'curve.json',curve)
    checkpoint(directory/'last.pth',model,optimizer,sampler,counter,p,{'mapping_sha256':mapping_sha})
    selected=select(curve,p);chosen=directory/('candidate_'+str(selected['step'])+'.pth');shutil.copyfile(chosen,directory/'selected.pth')
    saved=torch.load(directory/'selected.pth',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.eval();model.requires_grad_(False)
    if any(not torch.equal(v.cpu(),saved['model'][k]) for k,v in model.state_dict().items()):raise ValueError('Selected GPT reload differs')
    full=score(model,val,np.arange(len(val['targets'])),mapping,p,device,directory/'full_update_validation.jsonl')
    summary=dict(case=case,optimizer_steps=counter.step,processed_examples=sampler.processed,selected_update_step=selected['step'],
        selected_sha256=sha(directory/'selected.pth'),mapping_sha256=mapping_sha,selected_probe=selected['metrics'],full_update_validation=full,
        exact_base_warm_start=True,optimizer_reset=True,selected_reload_equal=True,curve=curve,runtime_seconds=time.monotonic()-began)
    save(directory/'summary.json',summary);save(directory/(case['name']+'_curves.json'),summary)
    del model,optimizer,base,saved;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return summary


def future_arm(output,case,data,p,device,lock_sha,observe):
    output=Path(output);directory=output/'arms'/case['name'];summary=read(directory/'summary.json');selected=directory/'selected.pth'
    if sha(selected)!=summary['selected_sha256']:raise ValueError('Locked selected GPT changed')
    mapping_path=output/'mappings'/(case['name']+'.npy')
    if sha(mapping_path)!=summary['mapping_sha256']:raise ValueError('Locked map changed')
    model,_=model_from(selected,p,device);model.eval();model.requires_grad_(False);mapping=np.load(mapping_path,allow_pickle=False);began=time.monotonic()
    def progress(done,total):
        observe();print(json.dumps(dict(phase='future',arm=case['name'],queries_done=done,all_queries=total)),flush=True)
    result=score(model,data,np.arange(len(data['targets'])),mapping,p,device,directory/'future_queries.jsonl',progress)
    if sha(selected)!=summary['selected_sha256'] or sha(mapping_path)!=summary['mapping_sha256']:raise ValueError('Future modified selected inputs')
    result.update(arm=case['name'],model_sha256=summary['selected_sha256'],mapping_sha256=summary['mapping_sha256'],
        selected_update_step=summary['selected_update_step'],selection_lock_sha256=lock_sha,selection_locked_before_future=True,
        history_protocol='prequential raw history,20 events before availability masking; unchanged interior positions',runtime_seconds=time.monotonic()-began)
    save(directory/'future_summary.json',result);del model;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return result
