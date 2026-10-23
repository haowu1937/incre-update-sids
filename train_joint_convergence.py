"""One train-only joint-study arm. Old source-bound drivers remain unchanged."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
from joint_convergence import lr_at, read_plan, save, sha

FORMAT = 'joint_convergence_arm_v1'


def sampler_identity(state):
    scalar = {k:v for k,v in state.items() if not hasattr(v, 'numpy')}
    digest=hashlib.sha256(json.dumps(scalar,sort_keys=True,allow_nan=False).encode())
    for name in sorted(state):
        if hasattr(state[name],'numpy'):
            digest.update(name.encode());digest.update(state[name].cpu().numpy().tobytes())
    return digest.hexdigest()


class Arm:
    def __init__(self, ex, case, directory, plan, binding, device='cuda:0', resume=False):
        import numpy as np
        import torch
        from gensid import historical_incremental_content as h
        self.np,self.torch,self.h=np,torch,h
        self.ex,self.case,self.directory,self.plan,self.binding=ex,case,Path(directory),plan,binding
        self.device=device;self.directory.mkdir(parents=True,exist_ok=resume)
        if (self.directory/'failure.json').exists():
            raise ValueError('Preserved failed arm cannot resume automatically')
        self.config=copy.deepcopy(ex.plain['config'])
        self.config.update(device=device,lr=case['lr'],seed=case['seed'],weight_decay=0.,
            compatibility_weight=case['history_weight'],temperature=plan['temperature'],
            max_grad_norm=plan['max_grad_norm'],anchor_batch_size=plan['anchor_batch_size'],
            batch_size=plan['batch_size'],anchor_sampler_seed=case['seed']+1,
            history_input_condition='new_full_sid')
        h.seed_all(case['seed'])
        self.model=h.build_model(self.config).to(device)
        self.model.load_state_dict(ex.parent['state_dict'],strict=True)
        if not all(torch.equal(v.cpu(),ex.parent['state_dict'][k]) for k,v in self.model.state_dict().items()):
            raise ValueError('Base warm start mismatch')
        self.scales=h.distance_scales(self.model,ex.features[:ex.groups['base_train'][1]],ex.old_batch)
        self.teacher=h.frozen_teacher(ex,device)
        self.trainer=h.build_trainer(self.model,self.config,self.directory)
        self.generator=torch.Generator(device='cpu').manual_seed(case['seed'])
        self.state=dict(step=0,cursor=0,order=None,epoch=0,processed_examples=0,
            generator_state=self.generator.get_state())
        self.raw_items=[r['raw_item_id'] for r in ex.feature_rows]
        self.curve=[];self.initial=None;prior=None
        if resume:
            prior=torch.load(self.directory/'last.pth',map_location='cpu',weights_only=False)
            if (prior['format']!=FORMAT or prior['config']!=self.config or prior['binding']!=binding
                    or prior['case']!=case or prior['distance_scales']!=self.scales):
                raise ValueError('Resume code/config/input/case/scales differ')
            summary=json.loads((self.directory/'summary.json').read_text())
            if summary['status']!='completed' or sha(self.directory/'last.pth')!=summary['checkpoint_sha256']:
                raise ValueError('Only a verified completed stage may be extended')
            self.model.load_state_dict(prior['state_dict'],strict=True)
            self.trainer.optimizer.load_state_dict(prior['optimizer'])
            self.state=copy.deepcopy(prior['sampler_state'])
            self.generator.set_state(self.state['generator_state'])
            if prior['scheduler']['last_step']!=self.state['step']:
                raise ValueError('Scheduler position does not match sampler')
            self.curve=prior['curve'];self.initial=prior['initial']
        self.penalty=h.HistoryPenalty(self.model,self.teacher,ex,self.config,self.scales,
            prior['anchor_sampler_state'] if prior else None)
        self.fixed=np.arange(min(plan['fixed_anchor_queries'],len(self.penalty.selected)))
        # Reuse the production hard-token scorer, without its positive-H-only diagnostic constructor.
        from diagnose_history_gradients import Engine
        self.hard_scores=lambda ledger,ids: Engine.hard_scores(self,ledger,ids)
        base=h.build_model(self.config).to(device);base.load_state_dict(ex.parent['state_dict'],strict=True)
        codes=h.catalog_codes(base,ex)
        if not np.array_equal(codes[:ex.groups['base_train'][1]],ex.old_export):
            raise ValueError('Base assignments not reproduced')
        self.initial_ledger=h.versioned_suffixes(ex.previous,self.raw_items,codes.tolist())
        del base
        if prior:
            h.restore_rng(prior['rng_state'])
        else:
            self.measure(epoch_complete=False)
            self.initial=copy.deepcopy(self.curve[0])
        self.last_valid=self.snapshot()

    def snapshot(self):
        return dict(format=FORMAT,case=self.case,config=self.config,binding=self.binding,
            distance_scales=self.scales,state_dict=copy.deepcopy(self.model.state_dict()),
            optimizer=copy.deepcopy(self.trainer.optimizer.state_dict()),sampler_state=copy.deepcopy(self.state),
            anchor_sampler_state=self.penalty.snapshot(),rng_state=self.h.rng_state(),
            scheduler=dict(name='warmup_cosine_constant_tail',last_step=self.state['step']),
            initial=copy.deepcopy(self.initial),curve=copy.deepcopy(self.curve))

    def restore(self, cp):
        self.model.load_state_dict(cp['state_dict'],strict=True)
        self.trainer.optimizer.load_state_dict(cp['optimizer'])
        self.state=copy.deepcopy(cp['sampler_state'])
        if 'generator_state' in self.state:self.generator.set_state(self.state['generator_state'])
        self.penalty.state=copy.deepcopy(cp['anchor_sampler_state'])
        self.penalty.generator.set_state(self.penalty.state['generator_state'])
        self.h.restore_rng(cp['rng_state'])
        self.curve=copy.deepcopy(cp['curve'])

    def checkpoint(self, path, state=None):
        from train_retuned_historical import atomic_checkpoint
        atomic_checkpoint(self.torch,path,state if state is not None else self.last_valid)

    def measure(self, epoch_complete):
        h=self.h;mode=self.model.training;rng=h.rng_state()
        anchor_before=sampler_identity(self.penalty.snapshot())
        try:
            with self.torch.no_grad():
                rq,groups,codes=h.evaluate_groups(self.model,self.ex.features,self.ex.groups,self.ex.old_batch,
                    self.config['added_item_assignment_eval_batch_size'])
                capacity=h.suffix_capacity_audit(self.ex.previous,self.raw_items,codes.tolist())
                if not capacity['full_sid_allocation_feasible']:raise ValueError('Monitor SID capacity exhausted')
                ledger=h.versioned_suffixes(self.ex.previous,self.raw_items,codes.tolist())
                full=self.hard_scores(ledger,self.np.arange(len(self.penalty.selected)))
                fixed=self.hard_scores(ledger,self.fixed)
                old,new=self.initial_ledger['entries'],ledger['entries']
                changes=dict(semantic_changed=sum(a['semantic_code']!=b['semantic_code'] for a,b in zip(old,new)),
                    full_sid_changed=sum(a['semantic_code']!=b['semantic_code'] or a['collision_suffix']!=b['collision_suffix'] for a,b in zip(old,new)),
                    suffix_only_changed=sum(a['semantic_code']==b['semantic_code'] and a['collision_suffix']!=b['collision_suffix'] for a,b in zip(old,new)))
                row=dict(step=self.state['step'],epoch=self.state['epoch'],epoch_complete=epoch_complete,
                    rq=rq,groups=groups,anchor_train=full,fixed_anchor=fixed,capacity=capacity,
                    collisions=h.collision_stats(codes,self.config['model']['num_emb_list']),sid_changes=changes)
                self.curve.append(row)
                save(self.directory/'curve.json',self.curve)
                print(json.dumps(dict(arm=self.case['arm_id'],monitor=row),allow_nan=False),flush=True)
        finally:
            self.model.train(mode);h.restore_rng(rng)
        if sampler_identity(self.penalty.snapshot())!=anchor_before:
            raise ValueError('Monitoring advanced training anchor sampler')

    def fit(self, target):
        from diagnose_history_gradients import gradient_groups
        torch,h=self.torch,self.h
        start=self.state['step'];began=time.monotonic();n=len(self.ex.features)
        if not start < target <= self.plan['final_steps']:raise ValueError('Invalid extension target')
        named=list(self.model.named_parameters());params=[p for _,p in named]
        counter=self.directory/'optimizer_steps.json'
        save(counter,dict(optimizer_steps_completed=start,optimizer_step_intent=None))
        self.checkpoint(self.directory/'last.pth')
        status='completed';failure=None
        try:
            self.model.train()
            while self.state['step']<target:
                if self.state['order'] is None or self.state['cursor']==n:
                    self.state.update(order=torch.randperm(n,generator=self.generator),cursor=0,epoch=self.state['epoch']+1)
                end=min(n,self.state['cursor']+self.config['batch_size'])
                ids=self.state['order'][self.state['cursor']:end]
                step=self.state['step']+1;probe=step==1 or step%self.plan['gradient_every']==0 or end==n
                x=self.ex.features[ids].to(self.device);optimizer=self.trainer.optimizer
                for group in optimizer.param_groups:group['lr']=lr_at(step,self.case['lr'],self.plan)
                optimizer.zero_grad(set_to_none=True)
                out,rq_loss,_,dense,codes,residuals=self.model(x,None,use_sk=False)
                rq,_,recon,quant=self.model.compute_loss(out,rq_loss,ids.to(self.device),dense,
                    dense_out_codebooks=codes,x_res_list=residuals,xs=x)
                with torch.set_grad_enabled(self.case['history_weight']>0 or probe):historical=self.penalty()
                rg=hg=before=None
                if probe:
                    rg=torch.autograd.grad(rq,params,retain_graph=True,allow_unused=True)
                    hg=torch.autograd.grad(historical,params,retain_graph=True,allow_unused=True) if historical.requires_grad else [None]*len(params)
                    before=[p.detach().clone() for p in params]
                objective=rq+self.case['history_weight']*historical
                if not torch.isfinite(objective):raise ValueError('Nonfinite objective')
                objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,self.plan['max_grad_norm'],error_if_nonfinite=True)
                save(counter,dict(optimizer_steps_completed=step-1,optimizer_step_intent=step))
                optimizer.step()
                save(counter,dict(optimizer_steps_completed=step,optimizer_step_intent=None))
                self.state.update(step=step,cursor=end,processed_examples=self.state['processed_examples']+len(ids),
                    generator_state=self.generator.get_state())
                metric=dict(step=step,epoch=self.state['epoch'],processed_examples=self.state['processed_examples'],
                    rq_loss=float(rq.detach()),reconstruction_loss=float(recon.detach()),quantization_loss=float(quant.detach()),
                    H_before=float(historical.detach()),weighted_H=self.case['history_weight']*float(historical.detach()),
                    total_loss=float(objective.detach()),lr=optimizer.param_groups[0]['lr'],
                    global_gradient_norm_before_clip=float(norm),clip_coefficient=min(1.,self.plan['max_grad_norm']/(float(norm)+1e-6)),
                    gradient_clipped=float(norm)>self.plan['max_grad_norm'],**self.penalty.last_metrics)
                if probe:
                    displacement=[p.detach()-v for p,v in zip(params,before)]
                    metric['gradient_components']=gradient_groups(torch,named,rg,hg,self.case['history_weight'],displacement)
                codes=h.catalog_codes(self.model,self.ex)
                audit=h.suffix_capacity_audit(self.ex.previous,self.raw_items,codes.tolist())
                metric['post_update_capacity']=audit
                with (self.directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
                if not audit['full_sid_allocation_feasible']:
                    self.checkpoint(self.directory/'rejected_candidate.pth',self.snapshot())
                    status='stopped_capacity_infeasible';raise ValueError('Post-update suffix capacity exhausted')
                if end==n or step==target:self.measure(epoch_complete=end==n)
                self.last_valid=self.snapshot()
                if step%self.plan['save_every']==0 or end==n or step==target:self.checkpoint(self.directory/'last.pth')
                if probe:print(json.dumps(dict(arm=self.case['arm_id'],training=metric),allow_nan=False),flush=True)
            h.check_teacher(self.teacher,self.ex)
            self.checkpoint(self.directory/('checkpoint-'+str(target)+'.pth'))
            saved=torch.load(self.directory/'last.pth',map_location='cpu',weights_only=False)
            reloaded=h.build_model(self.config).to(self.device);reloaded.load_state_dict(saved['state_dict'],strict=True)
            if not self.np.array_equal(h.catalog_codes(reloaded,self.ex),h.catalog_codes(self.model,self.ex)):
                raise ValueError('Checkpoint reload assignment mismatch')
            del reloaded
        except BaseException as error:
            if status!='stopped_capacity_infeasible':status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            failure=dict(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),
                optimizer_step_count=json.loads(counter.read_text()),last_valid_step=self.last_valid['sampler_state']['step'])
            self.restore(self.last_valid);self.checkpoint(self.directory/'last.pth')
            save(self.directory/'failure.json',failure)
        finally:
            self.trainer.tb_writer.close()
        report=dict(status=status,case=self.case,target=target,first_step=start,valid_step=self.state['step'],
            invocation_job_id=os.environ.get('LAB_BUFFER_ARM_JOB_ID'),
            optimizer_step_count=json.loads(counter.read_text()),initial=self.initial,curve=self.curve,
            config=self.config,binding=self.binding,content_sampler_sha256=sampler_identity(self.state),
            anchor_sampler_sha256=sampler_identity(self.penalty.snapshot()),
            anchor_queries_processed=self.penalty.state['processed_queries'],
            processed_examples=self.state['processed_examples'],teacher_unchanged=status=='completed',
            checkpoint_reload_assignments_equal=status=='completed',checkpoint_sha256=sha(self.directory/'last.pth'),
            runtime_seconds=time.monotonic()-began,failure=failure,
            optimizer_groups=[{k:v for k,v in group.items() if k!='params'} for group in self.trainer.optimizer.param_groups],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if str(self.device).startswith('cuda') else None,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if str(self.device).startswith('cuda') else None)
        if status=='completed':
            report['archived_checkpoint_sha256']=sha(self.directory/('checkpoint-'+str(target)+'.pth'))
        save(self.directory/('summary-'+str(target)+'.json'),report);save(self.directory/'summary.json',report)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',required=True);p.add_argument('--output',required=True)
    p.add_argument('--case',required=True);p.add_argument('--target',type=int,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();directory=Path(a.output).resolve();root=Path(__file__).resolve().parent
    try:
        import torch
        from gensid import historical_incremental_content as h
        plan=read_plan(root);d=json.loads(Path(a.deployment).read_text());case=json.loads(a.case)
        from joint_convergence import matrix,identifier
        allowed=matrix(plan)+matrix(plan,plan['confirmation_seed'])
        if case not in allowed or case['arm_id']!=identifier(case['lr'],case['history_weight'],case['seed']):
            raise ValueError('Arm outside declared matrix')
        if root not in directory.parents or directory==root or os.environ.get('LAB_BUFFER_JOB_ID')!='history-joint-convergence-001':
            raise ValueError('Require owned fresh bundle output')
        torch.set_num_threads(1);torch.cuda.init();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
        ex=h.experiment(SimpleNamespace(e1_dir=d['e1_dir']))
        if (len(ex.features)!=11707 or tuple(ex.features.shape)!=(11707,4096) or len(ex.cohorts['anchor_train'])!=4096
                or ex.plain['config']['model']['layers']!=[512,256,128]
                or ex.plain['config']['model']['e_dim']!=64
                or ex.plain['config']['model']['num_emb_list']!=[256]*3
                or ex.plain['config']['model']['quant_loss_weight']!=.25
                or ex.plain['config']['model']['bn'] or ex.plain['config']['model']['dropout_prob']):
            raise ValueError('Beauty input/model differs from the reviewed study')
        binding=dict(original_context=ex.binding,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],
            deployment_sha256=sha(a.deployment),manifest_sha256=sha(root/'docs/lab_buffer/joint_convergence_plan.json'))
        arm=Arm(ex,case,directory,plan,binding,resume=a.resume)
        report=arm.fit(a.target)
        raise SystemExit(0 if report['status']=='completed' else 2 if report['status']=='stopped_capacity_infeasible' else 1)
    except SystemExit:raise
    except BaseException as error:
        directory.mkdir(parents=True,exist_ok=True)
        save(directory/'launch-failure.json',dict(status='failed',case=json.loads(a.case),error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
