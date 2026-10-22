"""Training-only hard-SID direction diagnostic from one completed Historical checkpoint.

No lambda/STE changes, holdout/validation/Future scoring, teacher fitting or
production checkpoint updates. Each predeclared branch resets the same weights.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0'); os.environ.setdefault('USE_FLAX', '0')
ROOT = Path(__file__).resolve().parent
STAGE = 'history_gradient_controlled_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    pending=p.with_suffix(p.suffix+'.tmp')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(pending,p)


def task_output(deployment_path,output):
    d=json.loads(Path(deployment_path).read_text());output=Path(output).resolve()
    job_id=os.environ.get('LAB_BUFFER_JOB_ID')
    if (job_id not in ('history-gradient-controlled-001','history-gradient-controlled-repair-001')
            or ROOT != (Path(d['runner_root'])/'workspaces'/str(job_id)).resolve()
            or ROOT not in output.parents or output.exists()
            or subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=os.environ.get('LAB_BUFFER_CODE_SHA')):
        raise ValueError('Require fresh diagnostic output in the exact code-bound job workspace')
    return d,output


def cases(plan):
    return [dict(optimizer=o['name'],lr=lr,objective=name,steps=plan['steps_per_branch'])
            for o in plan['optimizers'] for lr in o['learning_rates'] for name in plan['objectives']]


def validate_plan(p):
    expected=[('sgd_reset',lr,obj,5) for lr in (1e-6,1e-5,1e-4) for obj in ('h_only','rq_only','mixed')]
    expected += [('adamw_reset',1e-4,obj,5) for obj in ('h_only','rq_only','mixed')]
    if ([(c['optimizer'],c['lr'],c['objective'],c['steps']) for c in cases(p)] != expected
            or p['max_diagnostic_optimizer_steps'] != 60 or p['history_weight'] != 1e-4
            or p['temperature'] != .5 or p['max_grad_norm'] != .1 or p['fixed_anchor_queries'] != 128
            or p['full_anchor_eval_steps'] != [0,1,5] or p['carry_original_optimizer_momentum'] is not False
            or p['holdout_evaluation'] or p['base_validation_evaluation'] or p['future_evaluation']
            or p['lambda_or_ste_changes'] or p['full_1000_step_training_authorized']):
        raise ValueError('Diagnostic budget or protocol differs from the declared training-only plan')
    return p


def gradient_groups(torch,named,rq,hg,weight,displacement=None):
    prefixes={'all':'','encoder':'encoder.','decoder':'decoder.','codebooks':'rq.',
              **{'codebook_'+str(i+1):'rq.vq_layers.'+str(i)+'.' for i in range(3)}}
    out={}
    for group,prefix in prefixes.items():
        ids=[i for i,(name,_) in enumerate(named) if name.startswith(prefix)]
        rn=sum(float(rq[i].detach().double().square().sum()) for i in ids if rq[i] is not None)
        hn=sum(float(hg[i].detach().double().square().sum()) for i in ids if hg[i] is not None)
        dot=sum(float((rq[i].detach().double()*hg[i].detach().double()).sum()) for i in ids if rq[i] is not None and hg[i] is not None)
        rn,hn=math.sqrt(rn),math.sqrt(hn)
        g=dict(rq_gradient_norm=rn,history_gradient_norm=hn,weighted_history_over_rq=weight*hn/rn if rn else None,
               cosine=max(-1.,min(1.,dot/(rn*hn))) if rn and hn else None,
               history_unused_parameters=sum(hg[i] is None for i in ids))
        if displacement is not None:
            g.update(update_norm=math.sqrt(sum(float(displacement[i].double().square().sum()) for i in ids)),
                history_dot_actual_update=sum(float((hg[i].detach().double()*displacement[i].double()).sum()) for i in ids if hg[i] is not None),
                rq_dot_actual_update=sum(float((rq[i].detach().double()*displacement[i].double()).sum()) for i in ids if rq[i] is not None))
        out[group]=g
    return out


def make_optimizer(torch,model,checkpoint,case):
    params=list(model.parameters())
    if case['optimizer']=='sgd_reset':
        return torch.optim.SGD(params,lr=case['lr'],momentum=0.,weight_decay=0.)
    if case['optimizer']!='adamw_reset':raise ValueError('Unknown diagnostic optimizer')
    groups=[]
    indices=[]
    for saved in checkpoint['optimizer']['param_groups']:
        if saved.get('weight_decay',0.) != 0 or saved.get('maximize',False):
            raise ValueError('Expected the original zero-decay minimizing AdamW recipe')
        ids=saved['params'];indices.extend(ids)
        group={k:copy.deepcopy(v) for k,v in saved.items() if k not in ('params','initial_lr')}
        group.update(params=[params[i] for i in ids],lr=case['lr']);groups.append(group)
    if sorted(indices)!=list(range(len(params))):raise ValueError('Optimizer parameter identity differs')
    # Fresh state by construction: original betas/epsilon/groups, no old moments.
    return torch.optim.AdamW(groups,lr=case['lr'])


class Engine:
    def __init__(self,ex,checkpoint,plan,output,device):
        import torch
        import numpy as np
        from gensid import historical_incremental_content as h
        self.torch,self.np,self.h=torch,np,h
        self.ex,self.cp,self.plan,self.output,self.device=ex,checkpoint,plan,Path(output),device
        self.config=copy.deepcopy(checkpoint['config']);self.config['device']=str(device)
        self.model=h.build_model(self.config).to(device)
        self.model.load_state_dict(checkpoint['state_dict'],strict=True)
        self.teacher=h.frozen_teacher(ex,device)
        self.penalty=h.HistoryPenalty(self.model,self.teacher,ex,self.config,checkpoint['distance_scales'])
        self.fixed=np.arange(min(plan['fixed_anchor_queries'],len(self.penalty.selected)),dtype=np.int64)
        self.penalty.draw=lambda:self.fixed.copy()
        self.raw_items=[r['raw_item_id'] for r in ex.feature_rows]
        generator=torch.Generator().manual_seed(plan['seed'])
        order=torch.randperm(len(ex.features),generator=generator)
        count=plan['steps_per_branch']*self.config['batch_size']
        if count>len(order):raise ValueError('Diagnostic requires five fixed nonrepeating content batches')
        self.content=[order[i*self.config['batch_size']:(i+1)*self.config['batch_size']] for i in range(plan['steps_per_branch'])]
        save(self.output/'sampling.json',dict(anchor_local_indices=self.fixed.tolist(),
            anchor_tensor_rows=self.penalty.selected[self.fixed].tolist(),content_indices=[x.tolist() for x in self.content],seed=plan['seed']))
        self.initial_state={k:v.detach().cpu().clone() for k,v in checkpoint['state_dict'].items()}
        self.total_completed_steps=0
        self.initial_codes=h.catalog_codes(self.model,ex).tolist()
        self.initial_ledger=h.versioned_suffixes(ex.previous,self.raw_items,self.initial_codes)
        self.initial_full=self.hard_scores(self.initial_ledger,np.arange(len(self.penalty.selected)))
        self.initial_fixed=self.hard_scores(self.initial_ledger,self.fixed)
        if self.initial_full['H']<=0:raise ValueError('H-only diagnostic must start at a checkpoint with positive hard training H')
        with torch.no_grad():forward=float(self.penalty())
        if abs(forward-self.initial_fixed['H'])>1e-5:raise ValueError('Production STE hard forward differs from hard token scoring')

    def hard_scores(self,ledger,local):
        torch,np,h=self.torch,self.np,self.h
        _,mapping=h.maps_from_ledger(ledger)
        ids=self.penalty.selected[local]
        arrays={k:v[ids] for k,v in self.ex.arrays['base_train'].items()}
        changed=h.reencode_history(arrays,self.ex.rows['base_train'],mapping)
        values=[];new_values=[];old_values=[]
        with torch.no_grad():
            for start in range(0,len(ids),32):
                selected=np.arange(start,min(start+32,len(ids)))
                batch=h.make_batch(changed,selected,self.device)
                # Exact old target/mask/position contract remains enforced by
                # reencode_history and the shared make_batch trimming routine.
                old_batch=h.make_batch(arrays,selected,self.device)
                if not torch.equal(batch['labels'],old_batch['labels']) or not torch.equal(batch['attention_mask'],old_batch['attention_mask']):
                    raise ValueError('Diagnostic changed targets/masks')
                logits=self.teacher(dict(**batch,use_cache=False)).logits.float()
                new=torch.nn.functional.cross_entropy(logits.transpose(1,2),batch['labels'],reduction='none').sum(1)
                old=self.penalty.old_nll[local[selected]].to(self.device)
                same=(batch['input_ids']==old_batch['input_ids']).all(1)
                delta=torch.where(same,torch.zeros_like(new),new-old)
                if not torch.isfinite(delta).all():raise ValueError('Nonfinite hard H')
                values.extend(delta.cpu().tolist());new_values.extend(new.cpu().tolist());old_values.extend(old.cpu().tolist())
        return dict(queries=len(ids),H=sum(max(v,0.) for v in values)/len(values),mean_delta_NLL=sum(values)/len(values),
                    mean_new_NLL=sum(new_values)/len(values),mean_old_NLL=sum(old_values)/len(values))

    def catalog(self):
        h=self.h
        mode=self.model.training
        try:
            metrics,_,codes=h.evaluate_groups(self.model,self.ex.features,self.ex.groups,self.ex.old_batch,
                                              self.config['added_item_assignment_eval_batch_size'])
        finally:
            self.model.train(mode)
        return metrics,codes.tolist()

    def execute(self,case):
        torch,h=self.torch,self.h
        started=time.monotonic()
        if case['steps'] != self.plan['steps_per_branch']:raise ValueError('Branch step budget mismatch')
        self.model.load_state_dict(self.initial_state,strict=True)
        if not all(torch.equal(v.cpu(),self.initial_state[k]) for k,v in self.model.state_dict().items()):
            raise ValueError('Branch did not start from identical weights')
        h.seed_all(self.plan['seed'])
        self.model.train()
        optimizer=make_optimizer(torch,self.model,self.cp,case)
        if optimizer.state:raise ValueError('Diagnostic inherited optimizer moments')
        name=case['optimizer']+'_'+format(case['lr'],'.0e')+'_'+case['objective']
        directory=self.output/name;directory.mkdir()
        named=list(self.model.named_parameters());params=[p for _,p in named]
        result=dict(case=case,status='completed',initial_weights_equal=True,optimizer_state_reset=True,
                    sampling_sha256=sha(self.output/'sampling.json'),initial_fixed=self.initial_fixed,
                    initial_full=self.initial_full,checkpoints=[],step_records=[],optimizer_steps=0)
        last_valid={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        baseline_metrics,_=self.catalog();result['initial_reconstruction']=baseline_metrics
        for index,batch_ids in enumerate(self.content):
            step=index+1
            x=self.ex.features[batch_ids].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            out,quant,_,dense,codewords,residuals=self.model(x,None,use_sk=False)
            rq,_,recon,q=self.model.compute_loss(out,quant,batch_ids.to(self.device),dense,
                dense_out_codebooks=codewords,x_res_list=residuals,xs=x)
            historical=self.penalty()
            rg=torch.autograd.grad(rq,params,retain_graph=True,allow_unused=True)
            hg=torch.autograd.grad(historical,params,retain_graph=True,allow_unused=True) if historical.requires_grad else [None]*len(params)
            objective=historical if case['objective']=='h_only' else rq if case['objective']=='rq_only' else rq+self.plan['history_weight']*historical
            before=[p.detach().clone() for p in params]
            if objective.requires_grad:objective.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,self.plan['max_grad_norm'],error_if_nonfinite=True)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise ValueError('Nonfinite diagnostic gradient')
            if self.total_completed_steps>=self.plan['max_diagnostic_optimizer_steps']:
                raise ValueError('Diagnostic global step guard reached')
            optimizer.step();result['optimizer_steps']+=1;self.total_completed_steps+=1
            save(self.output/'optimizer_steps.json',dict(completed_steps=self.total_completed_steps,branch=name,branch_step=step))
            displacement=[p.detach()-v for p,v in zip(params,before)]
            record=dict(step=step,rq_loss_before=float(rq.detach()),H_before=float(historical.detach()),
                objective_before=float(objective.detach()),global_norm_before_clip=float(norm),
                clip_coefficient=min(1.,self.plan['max_grad_norm']/(float(norm)+1e-6)),gradient_clipped=float(norm)>self.plan['max_grad_norm'],
                gradient_components=gradient_groups(torch,named,rg,hg,self.plan['history_weight'],displacement))
            codes=h.catalog_codes(self.model,self.ex).tolist()
            capacity=h.suffix_capacity_audit(self.ex.previous,self.raw_items,codes)
            record['capacity']=capacity
            if not capacity['full_sid_allocation_feasible']:
                torch.save(dict(format=STAGE,production_checkpoint=False,case=case,step=step,state_dict=self.model.state_dict()),directory/'rejected_candidate.pth')
                torch.save(dict(format=STAGE,production_checkpoint=False,case=case,step=step-1,state_dict=last_valid),directory/'last_valid.pth')
                result.update(status='stopped_capacity_infeasible',last_valid_step=step-1)
                result['step_records'].append(record)
                break
            ledger=h.versioned_suffixes(self.ex.previous,self.raw_items,codes)
            old=self.initial_ledger['entries'];new=ledger['entries']
            record['sid_changes']=dict(items=len(new),semantic_changed=sum(a['semantic_code']!=b['semantic_code'] for a,b in zip(old,new)),
                suffix_changed=sum(a['collision_suffix']!=b['collision_suffix'] for a,b in zip(old,new)),
                full_sid_changed=sum(a['semantic_code']!=b['semantic_code'] or a['collision_suffix']!=b['collision_suffix'] for a,b in zip(old,new)),
                suffix_changed_without_semantic_change=sum(a['semantic_code']==b['semantic_code'] and a['collision_suffix']!=b['collision_suffix'] for a,b in zip(old,new)))
            record['fixed_after']=self.hard_scores(ledger,self.fixed)
            if step in self.plan['full_anchor_eval_steps']:
                full=self.hard_scores(ledger,self.np.arange(len(self.penalty.selected)))
                reconstruction,_=self.catalog()
                result['checkpoints'].append(dict(step=step,full_anchor_train=full,reconstruction=reconstruction,
                                                 capacity=capacity,sid_changes=record['sid_changes']))
                print(json.dumps(dict(branch=name,step=step,H_fixed=record['fixed_after']['H'],H_full=full['H'],
                    mse=reconstruction['reconstruction_loss'],sid_changes=record['sid_changes']),allow_nan=False),flush=True)
            result['step_records'].append(record)
            with (directory/'steps.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            last_valid={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        torch.save(dict(format=STAGE,production_checkpoint=False,case=case,
            step=result.get('last_valid_step',result['optimizer_steps']),state_dict=last_valid),directory/'last_valid.pth')
        h.check_teacher(self.teacher,self.ex)
        result['teacher_unchanged']=True
        result['runtime_seconds']=time.monotonic()-started
        save(directory/'summary.json',result)
        return result


def compact(report):
    out={k:v for k,v in report.items() if k!='branches'}
    branches=[]
    for b in report['branches']:
        short={k:v for k,v in b.items() if k!='step_records'}
        short['steps']=[dict(step=r['step'],H_before=r['H_before'],fixed_after=r.get('fixed_after'),
            clip_coefficient=r['clip_coefficient'],capacity_feasible=r['capacity']['full_sid_allocation_feasible'],
            full_sid_changed=r.get('sid_changes',{}).get('full_sid_changed'),
            update_norm=r['gradient_components']['all']['update_norm'],
            H_dot_update=r['gradient_components']['all']['history_dot_actual_update'],
            RQ_dot_update=r['gradient_components']['all']['rq_dot_actual_update']) for r in b['step_records']]
        branches.append(short)
    out['branches']=branches
    return out


def run(deployment_path,output):
    started=time.monotonic()
    import torch
    import evaluate_historical_future as context
    from gensid import historical_incremental_content as h
    plan=validate_plan(json.loads((ROOT/'docs/lab_buffer/history_gradient_plan.json').read_text()))
    d,output=task_output(deployment_path,output)
    job_id=os.environ['LAB_BUFFER_JOB_ID'];code=os.environ['LAB_BUFFER_CODE_SHA']
    output.mkdir(parents=True)
    source_files=[str(ROOT/p) for p in ('diagnose_history_gradients.py','train_retuned_historical.py',
        'gensid/historical_incremental_content.py','gensid/models/rqvae.py','genrec/historical_damage.py')]
    future=Path(d['future_dir'])
    protected={p:v for p,v in d['input_sha256'].items() if future != Path(p) and future not in Path(p).parents}
    protected.update({p:sha(p) for p in source_files})
    if any(sha(p)!=v for p,v in protected.items()):raise ValueError('Immutable diagnostic input changed')
    torch.cuda.init();torch.cuda.set_device(0);device=torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats(device)
    ex,directory,original=context.load_context(SimpleNamespace(e1_dir=d['e1_dir'],retuned_dir=d['retuned_dir']))
    cp=torch.load(directory/'historical/last.pth',map_location='cpu',weights_only=False)
    if cp['sampler_state']['step']!=1000 or len(ex.cohorts['anchor_train'])!=4096:
        raise ValueError('Unexpected checkpoint/cohort for this diagnostic')
    if (cp['config']['lr']!=1e-4 or cp['config']['temperature']!=.5 or cp['config']['compatibility_weight']!=1e-4
            or cp['config']['anchor_batch_size']!=128 or cp['config']['max_grad_norm']!=.1
            or cp['config']['batch_size']!=256 or cp['config']['seed']!=2026
            or cp['config']['model']['bn'] or cp['config']['model']['dropout_prob']!=0):
        raise ValueError('Retuned Historical checkpoint config differs from the diagnosis')
    engine=Engine(ex,cp,plan,output,device)
    expected=original['evaluation']['groups']['historical']['anchor_train']['new_full_sid']['mean_positive_part_delta']
    if abs(engine.initial_full['H']-expected)>1e-5:raise ValueError('Initial full anchor_train hard H does not reproduce the accepted run')
    report=dict(stage=STAGE,execution_status='running',job_id=job_id,code_sha=code,
        original_checkpoint_sha256=sha(directory/'historical/last.pth'),plan=plan,
        initial_full_anchor_train=engine.initial_full,initial_fixed_anchor=engine.initial_fixed,
        original_training_H=expected,training_started=True,holdout_evaluated=False,
        base_validation_evaluated=False,future_evaluated=False,new_tiger_training_steps=0,
        sampler_sha256=sha(output/'sampling.json'),branches=[])
    for case in cases(plan):
        report['branches'].append(engine.execute(case))
        save(output/'progress.json',compact(report))
    report['optimizer_steps_this_task']=sum(b['optimizer_steps'] for b in report['branches'])
    if report['optimizer_steps_this_task']>60:raise ValueError('Diagnostic micro-update budget exceeded')
    if any(sha(p)!=v for p,v in protected.items()):raise ValueError('Immutable inputs changed during diagnosis')
    report.update(execution_status='completed',original_inputs_unchanged=True,teacher_unchanged=True,
        all_branches_same_start=True,all_branches_same_sampling=True,original_optimizer_momentum_used=False,
        protected_input_count=len(protected),protected_manifest_sha256=json_sha(protected))
    report.update(runtime_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
        torch_version=torch.__version__,device=str(device),
        mandatory_real_model_checks_passed=True,
        mandatory_checks_log_sha256=sha(ROOT/'outputs/gradient-checks.log'))
    save(output/'summary.json',report)
    result=compact(report);result['full_summary_sha256']=sha(output/'summary.json')
    raw=json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode())>64*1024:raise ValueError('Bounded diagnostic artifact exceeds mailbox limit')
    (output/'compact.json').write_text(raw)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deployment',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();started=time.monotonic();_,output=task_output(a.deployment,a.output)
    # Checks run in a synthetic temporary fixture, never the Beauty inputs.
    try:
        checks=ROOT/'outputs/gradient-checks.log';checks.parent.mkdir(exist_ok=True)
        with checks.open('x') as stream:
            subprocess.run([sys.executable,'-B',str(ROOT/'tests/incremental/test_history_gradient_diagnostic.py'),'--require-torch'],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=run(a.deployment,output)
        print(json.dumps(dict(stage=STAGE,execution_status='completed',optimizer_steps=result['optimizer_steps_this_task'],
            branches=len(result['branches']),initial_H=result['initial_full_anchor_train']['H'],runtime_seconds=time.monotonic()-started)),flush=True)
    except BaseException as error:
        if not output.exists():output.mkdir(parents=True)
        counter=output/'optimizer_steps.json'
        failure=dict(stage=STAGE,execution_status='failed',job_id=os.environ.get('LAB_BUFFER_JOB_ID'),
            code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),error=type(error).__name__+': '+str(error),
            traceback=traceback.format_exc(),runtime_seconds=time.monotonic()-started,
            holdout_evaluated=False,base_validation_evaluated=False,future_evaluated=False,
            optimizer_step_count_evidence=json.loads(counter.read_text()) if counter.exists() else None,
            mandatory_checks_log_tail=checks.read_text(errors='replace')[-14000:] if checks.exists() else None,
            evidence='Inspect preserved checks/progress/branch files before any new job; no automatic retry.')
        save(output/'compact.json',failure)
        raise


if __name__=='__main__':main()
