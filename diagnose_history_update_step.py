"""In-memory single-step diagnostics for existing History-aware Update v2 checkpoints.

Same next content batch and restored AdamW state for content-only and joint
proposals. Component interventions and direct history directions are diagnostics,
not training configurations. No checkpoint writes, ranking, or holdout queries.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import history_aware_update_v2 as v2


def next_content_batch(checkpoint, count):
    state = copy.deepcopy(checkpoint['sampler_state'])
    generator = torch.Generator(device='cpu').manual_seed(checkpoint['config']['seed'])
    if 'generator_state' in state:
        generator.set_state(state['generator_state'])
    if state['order'] is None or state['cursor'] == count:
        state['order'] = torch.randperm(count, generator=generator)
        state['cursor'] = 0
        state['epoch'] += 1
    end = min(count, state['cursor']+checkpoint['config']['batch_size'])
    return state['order'][state['cursor']:end].clone()


def content_gradients(model, ex, selected, device):
    model.train();model.zero_grad(set_to_none=True)
    data = ex.features[selected].to(device)
    out, quant, _, dense, books, residuals = model(data,None,use_sk=False)
    loss,aux,recon,quant_loss = model.compute_loss(out,quant,selected.to(device),dense,
        dense_out_codebooks=books,x_res_list=residuals,xs=data)
    loss.backward()
    grads = {n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in model.named_parameters()}
    return grads,dict(rq_loss=float(loss.detach()),reconstruction_loss=float(recon),quantization_loss=float(quant_loss.detach()))


def adam_proposal(model, checkpoint, grads, joint_books=None):
    """Each proposal restores both model and the same saved optimizer moments."""
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    v2.restore_rng(checkpoint['rng_state'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=checkpoint['config']['lr'],
                                  weight_decay=checkpoint['config']['weight_decay'])
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    optimizer.zero_grad(set_to_none=True)
    for name,parameter in model.named_parameters():
        value = grads[name]
        parameter.grad = value.to(parameter.device).clone() if value is not None else None
    if joint_books is not None:
        for layer,value in zip(model.rq.vq_layers,joint_books):
            layer.embedding.weight.grad = value.to(layer.embedding.weight.device).clone()
    optimizer.step()
    return v2.snapshot(model)


def component_state(base, joint, prefix):
    return {n:(joint[n] if n.startswith(prefix) else value).clone() for n,value in base.items()}


def direct_history_state(base, history_grads, relative_step, sign):
    names = ['rq.vq_layers.'+str(i)+'.embedding.weight' for i in range(len(history_grads))]
    theta_norm = sum(float(base[n].double().square().sum()) for n in names)**.5
    grad_norm = sum(float(g.double().square().sum()) for g in history_grads)**.5
    if grad_norm == 0:
        return None
    alpha = relative_step*theta_norm/grad_norm
    state = {n:v.clone() for n,v in base.items()}
    for name,gradient in zip(names,history_grads):
        state[name].add_(gradient.cpu(),alpha=sign*alpha)
    return state


def update_stats(base, candidate, history_grads):
    by_component = {}
    for label,prefix in [('encoder','encoder.'),('decoder','decoder.'),('codebooks','rq.')]:
        names = [n for n,v in base.items() if n.startswith(prefix) and v.is_floating_point()]
        before = sum(float(base[n].double().square().sum()) for n in names)**.5
        delta = sum(float((candidate[n]-base[n]).double().square().sum()) for n in names)**.5
        by_component[label] = delta/max(before,1e-30)
    projection = 0.
    for i,g in enumerate(history_grads):
        name = 'rq.vq_layers.'+str(i)+'.embedding.weight'
        projection += float((g.cpu().double()*(candidate[name]-base[name]).double()).sum())
    return dict(relative_parameter_update=by_component,
        proxy_linear_history_change_codebooks_only=projection,
        proxy_scope='local STE dot parameter change; omits encoder assignment and suffix effects')


def extra_adam_effect(base, content, joint, history_grads):
    result = []
    for i,g in enumerate(history_grads):
        name = 'rq.vq_layers.'+str(i)+'.embedding.weight'
        plain_delta = (content[name]-base[name]).double()
        extra = (joint[name]-content[name]).double()
        rn,hn = float(plain_delta.norm()),float(extra.norm())
        gradient = g.cpu().double(); gn = float(gradient.norm())
        result.append(dict(level=i+1,extra_update_relative_to_content_update=hn/rn if rn>0 else None,
            extra_update_norm=hn,content_update_norm=rn,
            extra_update_dot_history_gradient=float((extra*gradient).sum()),
            extra_update_cosine_with_negative_history_gradient=
                float((extra*(-gradient)).sum())/(hn*gn) if hn>0 and gn>0 else None))
    return result


def point(ex,directory,checkpoint,device,args):
    model = v2.build_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    base = v2.snapshot(model)
    selected = next_content_batch(checkpoint,len(ex.features))
    teacher = v2.h.frozen_teacher(ex,device)
    history = v2.FullHistory(model,teacher,ex,checkpoint['config'],checkpoint['distance_scales'],
                             progress_every=args.progress_every)
    if not torch.equal(history.old_nll(),checkpoint['old_anchor_nll']):
        raise ValueError('Old teacher reference differs from bound checkpoint')
    codes,capacity,ledger,mapping = v2.catalog(model,ex)
    if mapping is None: raise ValueError('Probe checkpoint complete SIDs are infeasible')
    v2.restore_rng(checkpoint['rng_state'])
    content_grads,content_loss = content_gradients(model,ex,selected,device)
    before,history_grads = history.measure(codes,mapping,backward=True,step=str(checkpoint['sampler_state']['step'])+' probe')
    # History autograd.grad must preserve all previously computed content grads.
    for name,p in model.named_parameters():
        stored = content_grads[name]
        if (stored is None) != (p.grad is None) or (stored is not None and not torch.equal(p.grad.cpu(),stored)):
            raise ValueError('Historical gradient altered content .grad buffer: '+name)
    rq_books = [content_grads['rq.vq_layers.'+str(i)+'.embedding.weight']
                for i in range(len(model.rq.vq_layers))]
    rq_books = [g.to(device) if g is not None else torch.zeros_like(p) for g,p in zip(rq_books,history.books)]
    rho = checkpoint['config']['historical_v2']['history_gradient_ratio']
    combined,grad_stats = v2.normalized_mix(rq_books,history_grads,rho,
                                           checkpoint['config']['historical_v2']['epsilon'])
    content = adam_proposal(model,checkpoint,content_grads)
    joint = adam_proposal(model,checkpoint,content_grads,combined)
    # Encoder and decoder Adam updates must be identical in the paired comparison.
    if any(not torch.equal(content[n],joint[n]) for n in base if n.startswith(('encoder.','decoder.'))):
        raise ValueError('Content/joint encoder or decoder updates differ')
    candidates = [('content_adam',content),('joint_adam',joint),
        ('joint_encoder_only',component_state(base,joint,'encoder.')),
        ('joint_codebooks_only',component_state(base,joint,'rq.'))]
    for name,sign in [('history_direct_descent',-1),('history_direct_reverse',1)]:
        state = direct_history_state(base,history_grads,args.direct_relative_step,sign)
        if state is not None: candidates.append((name,state))
    rows = []
    for name,state in candidates:
        model.load_state_dict(state,strict=True)
        new_codes,new_capacity,_,new_mapping = v2.catalog(model,ex)
        hard = None
        if new_mapping is not None:
            hard,_ = history.measure(new_codes,new_mapping,backward=False)
        rq,_,_ = v2.metrics(model,ex)
        row = dict(name=name,**update_stats(base,state,history_grads),
            hard=hard,hard_positive_damage_change=hard['positive_damage']-before['positive_damage'] if hard else None,
            full_sid_allocation_feasible=new_capacity['full_sid_allocation_feasible'],
            maximum_collision_bucket=new_capacity['maximum_current_collision_bucket'],
            unique_semantic_codes=len(set(map(tuple,new_codes.tolist()))),full_catalog_rq_metrics=rq,
            changed_catalog_semantic_fraction=float((new_codes!=codes).any(1).mean()),
            changed_catalog_suffix_fraction=float(np.mean([new_mapping[raw][3]!=mapping[raw][3] for raw in mapping]))
                if new_mapping is not None else None)
        rows.append(row)
        print(json.dumps(dict(checkpoint_step=checkpoint['sampler_state']['step'],candidate=name,
            hard_delta=row['hard_positive_damage_change'],feasible=row['full_sid_allocation_feasible']),allow_nan=False),flush=True)
    model.load_state_dict(base,strict=True);model.zero_grad(set_to_none=True)
    if any(not torch.equal(v.cpu(),base[n]) for n,v in model.state_dict().items()):
        raise ValueError('Probe model was not restored')
    v2.h.check_teacher(teacher,ex)
    by_name = {r['name']:r for r in rows}
    paired_delta = (by_name['joint_adam']['hard']['positive_damage']-by_name['content_adam']['hard']['positive_damage']
        if by_name['joint_adam']['hard'] is not None and by_name['content_adam']['hard'] is not None else None)
    return dict(checkpoint_dir=str(directory.resolve()),checkpoint_sha256=v2.sha256(directory/'last.pth'),
        checkpoint_step=checkpoint['sampler_state']['step'],next_content_batch_rows=selected.tolist(),
        next_content_batch_size=len(selected),content_loss_before=content_loss,
        full_anchor_train_before=before,gradient_statistics=grad_stats,candidates=rows,
        joint_minus_content_hard_positive_damage=paired_delta,
        extra_history_adam_update_by_level=extra_adam_effect(base,content,joint,history_grads),
        model_restored=True,teacher_unchanged=True,checkpoint_writes=False)


def run(args):
    started=time.perf_counter();output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError('Refusing to overwrite single-step diagnostic')
    if not np.isfinite(args.direct_relative_step) or args.direct_relative_step<=0 or args.progress_every<0:
        raise ValueError('Invalid direct step or progress interval')
    ex,device=v2.h.experiment(args),v2.h.device_from(args.device)
    loaded=[]
    for raw in args.rqvae_dirs:
        directory=Path(raw).resolve();summary=v2.read_json(directory/'summary.json')
        if summary.get('stage')!=v2.FORMAT or summary.get('execution_status')!='completed' or summary['binding']!=v2.binding(ex):
            raise ValueError('Expected a completed, source-bound v2 checkpoint')
        v2.verify_files(directory,summary,['last.pth','config.json'])
        cp=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
        if (cp.get('format')!=v2.FORMAT or cp['binding']!=summary['binding'] or cp['config']!=summary['config']
                or cp['sampler_state']['step']!=summary['update_optimizer_steps']):
            raise ValueError('Probe checkpoint config/step mismatch')
        loaded.append((directory,cp,v2.sha256(directory/'last.pth')))
    original_rng=v2.rng_state()
    results=[]
    try:
        for directory,cp,checksum in loaded:
            v2.seed_all(cp['config']['seed'])
            result=point(ex,directory,cp,device,args)
            if v2.sha256(directory/'last.pth')!=checksum: raise ValueError('Source checkpoint changed')
            results.append(result)
    finally:
        v2.restore_rng(original_rng)
    output.mkdir(parents=True)
    report=dict(stage='historical_joint_v2_single_step_diagnostic',execution_status='completed',
        e1_dir=str(ex.e1_dir.resolve()),source_binding=v2.binding(ex),
        diagnostic_source_sha256=v2.sha256(Path(__file__)),command=vars(args),points=results,
        probe_cohort='all bound anchor_train; no holdout or validation queries',holdout_used=False,
        checkpoint_writes=False,training_steps_committed=0,
        intervention_scope='component transplants and direct directions are in-memory diagnostic states',
        scope='local evidence at two existing trajectory points; not a new training run or holdout result',
        device=str(device),runtime_seconds=time.perf_counter()-started)
    v2.write_json(output/'summary.json',report)
    print('Single-step diagnostic completed; use bundle show_step.py for compact results.',flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e1-dir',required=True);p.add_argument('--rqvae-dirs',required=True,nargs='+')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--direct-relative-step',type=float,default=.001)
    p.add_argument('--progress-every',type=int,default=32)
    return p


if __name__=='__main__':run(parser().parse_args())
