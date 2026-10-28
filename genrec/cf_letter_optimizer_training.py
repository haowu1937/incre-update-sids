"""Task-local optimizer variants; unchanged accepted hard-forward/STE mechanism."""
import json
import time
import numpy as np
import torch
from genrec.cf_letter_history_long import Engine,save_checkpoint,gradient_stats,public_curves
from genrec.cf_letter_base_training import Counter,Sampler
from genrec.cf_letter_protocol import save
from genrec.cf_letter_history_long_protocol import indices_sha,tail_stability


class OptimizerEngine(Engine):
    def branch(self,case,weight):
        self.model.load_state_dict(self.initial);self.model.zero_grad(set_to_none=True)
        directory=self.output/case['name'];directory.mkdir();began=time.monotonic()
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=case['lr'],betas=(case['beta1'],self.p['beta2']),eps=1e-8,weight_decay=0.)
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
        save(self.output/(case['name']+'_curves.json'),public_curves(result,trajectory))
        return result


    def run_case(self,case):
        result=self.branch(case,self.p['fixed_history_weight'])
        if not all(torch.equal(v.cpu(),self.teacher_state[k]) for k,v in self.teacher.state_dict().items()) or any(v.grad is not None or v.requires_grad for v in self.teacher.parameters()):raise ValueError('Frozen teacher changed')
        if not torch.equal(self.model.features.cpu(),self.initial['features']):raise ValueError('Fixed E1 changed')
        return dict(result={k:v for k,v in result.items() if k!='monitors'},initial_full_anchor_train=self.initial_full,initial_fixed_anchor=self.initial_fixed,
            initial_rq=self.initial_rq,initial_capacity=self.initial_capacity,teacher_unchanged=True,cf_embeddings_unchanged=True)
