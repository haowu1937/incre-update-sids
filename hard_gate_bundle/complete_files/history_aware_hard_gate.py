"""Frozen-encoder RQ-VAE Update with actual full-SID training-H candidate gating.

At each step both AdamW proposals start from the identical current model,
optimizer moments, RNG state and content batch. Commit a complete transaction.
The historical candidate must beat that step's content candidate in hard H
and satisfy a fixed relative reconstruction cap. Never select on holdout.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
import history_aware_frozen_encoder as f

FORMAT = 'historical_hard_gated_update_v1'


def read_control(directory, ex=None):
    directory=Path(directory).resolve()
    report=f.read_json(directory/'summary.json')
    if (report.get('stage')!=f.FORMAT or report.get('execution_status')!='completed'
            or report.get('mode')!='plain' or not report.get('encoder_unchanged_from_base')):
        raise ValueError('Expected completed frozen-encoder Plain reference')
    f.verify_files(directory,report,['last.pth','config.json','semantic_assignments.jsonl',
        'new_allocation_registry.json','new_item2tokens.json','anchor_train_hard_losses.jsonl',
        'anchor_train_tensor_rows.json'])
    cp=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
    if (cp.get('format')!=f.FORMAT or cp['binding']!=report['binding'] or cp['config']!=report['config']
            or cp['sampler_state']['step']!=report['update_optimizer_steps']):
        raise ValueError('Frozen Plain checkpoint/report mismatch')
    if report['config']['historical_v2']['history_gradient_ratio']!=0:
        raise ValueError('Plain reference must have rho=0')
    if ex is not None and (report['binding']!=f.binding(ex)
            or f.read_json(directory/'anchor_train_tensor_rows.json')!=ex.cohorts['anchor_train']):
        raise ValueError('Frozen Plain data/source/cohort binding mismatch')
    return report,cp


def binding(ex, plain_dir):
    directory=Path(plain_dir).resolve()
    return dict(reference=f.binding(ex), hard_gate_source_sha256=f.sha256(Path(__file__)),
        plain_summary_sha256=f.sha256(directory/'summary.json'),
        plain_checkpoint_sha256=f.sha256(directory/'last.pth'))


def transaction(model, optimizer):
    return dict(state_dict=f.snapshot(model), optimizer=copy.deepcopy(optimizer.state_dict()),rng_state=f.rng_state())


def restore_transaction(model, optimizer, state):
    model.load_state_dict(state['state_dict'],strict=True)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    f.restore_rng(state['rng_state'])
    optimizer.zero_grad(set_to_none=True)


def proposal(model, optimizer, base, gradients):
    """Restore common AdamW state; returning a candidate does not commit it."""
    restore_transaction(model,optimizer,base)
    for name,p in model.named_parameters():
        value=gradients[name]
        if value is not None:
            if not p.requires_grad:
                raise ValueError('Gradient supplied for frozen parameter: '+name)
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite proposal gradient')
            p.grad=value.to(p.device).clone()
    optimizer.step()
    return transaction(model,optimizer)


def candidate_metrics(model, ex, history):
    """Hard forward only; every candidate uses the identical full training cohort."""
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in model.state_dict().values()):
        return dict(feasible=False,reason='nonfinite_weights'),None
    try:
        totals,groups,codes=f.metrics(model,ex)
    except (ValueError,FloatingPointError) as error:
        if 'nonfinite' not in str(error).lower():raise
        return dict(feasible=False,reason='nonfinite_content_metrics'),None
    if not all(np.isfinite(v) for v in totals.values()):
        return dict(feasible=False,reason='nonfinite_content_metrics'),None
    raw=[r['raw_item_id'] for r in ex.feature_rows]
    capacity=f.h.suffix_capacity_audit(ex.previous,raw,codes.tolist())
    if not capacity['full_sid_allocation_feasible']:
        return dict(feasible=False,reason='full_sid_capacity_exhausted',rq_mse=totals['reconstruction_loss'],
            maximum_required_suffix_capacity=capacity['maximum_required_suffix_capacity']),None
    ledger=f.v2.versioned_suffixes(ex.previous,raw,codes.tolist())
    _,mapping=f.v2.maps_from_ledger(ledger)
    hard,_=history.measure(codes,mapping)
    if not all(np.isfinite(v) for v in hard.values()):
        return dict(feasible=False,reason='nonfinite_hard_history'),None
    summary=dict(feasible=True,reason='evaluated',history=hard,rq_mse=totals['reconstruction_loss'],
        unique_semantic_codes=len(set(map(tuple,codes.tolist()))),
        maximum_collision_bucket=capacity['maximum_current_collision_bucket'],
        maximum_required_suffix_capacity=capacity['maximum_required_suffix_capacity'])
    details=dict(metrics=totals,group_metrics=groups,codes=codes,capacity=capacity,ledger=ledger,mapping=mapping)
    return summary,details


def choose(content, joint, min_improvement, mse_ratio):
    """Returns None when no valid content comparator exists; caller rolls back."""
    if not content['feasible']:
        return None,'content_candidate_infeasible_rollback_stop'
    if not joint['feasible']:
        return 'content','historical_candidate_infeasible'
    if joint['rq_mse']>content['rq_mse']*mse_ratio:
        return 'content','historical_reconstruction_cap_exceeded'
    if joint['history']['positive_damage']>=content['history']['positive_damage']-min_improvement:
        return 'content','historical_hard_H_not_lower'
    return 'historical','historical_hard_H_lower_and_quality_feasible'


def save(path, model, optimizer, config, bound, sampler, work, scales, initial, initial_groups, history):
    checkpoint=transaction(model,optimizer)
    checkpoint.update(format=FORMAT,config=config,binding=bound,sampler_state=copy.deepcopy(sampler),
        work_counters=copy.deepcopy(work),distance_scales=scales,initial_metrics=initial,
        initial_group_metrics=initial_groups,old_anchor_nll=history.old_nll())
    temp=Path(str(path)+'.tmp');torch.save(checkpoint,temp);os.replace(temp,path)


def train(args):
    started=time.perf_counter();out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite hard-gate output')
    if (min(args.max_steps,args.anchor_batch_size,args.log_every,args.save_every)<1
            or not np.isfinite(args.history_gradient_ratio) or not 0<=args.history_gradient_ratio<=1
            or not np.isfinite(args.temperature) or args.temperature<=0
            or not np.isfinite(args.min_h_improvement) or args.min_h_improvement<0
            or not np.isfinite(args.rq_mse_ratio_cap) or args.rq_mse_ratio_cap<1):
        raise ValueError('Invalid hard-gate configuration')
    ex=f.h.experiment(args);device=f.h.device_from(args.device)
    control,control_cp=read_control(args.plain_dir,ex)
    if args.max_steps>control['update_optimizer_steps']:
        raise ValueError('Target budget exceeds the completed frozen Plain reference')
    config=copy.deepcopy(control['config']);config['device']=str(device)
    if config['model']['bn'] or config['model']['dropout_prob']!=0:
        raise ValueError('This intervention requires the deterministic bound MLP')
    for key,value in [('anchor_batch_size',args.anchor_batch_size),('temperature',args.temperature)]:
        if value!=control['config']['historical_v2'][key]:
            raise ValueError('Historical forward settings must match the frozen Plain reference: '+key)
    if str(device)!=control['config']['device']:
        raise ValueError('Device must match the completed frozen Plain reference')
    config['historical_v2']['history_gradient_ratio']=args.history_gradient_ratio
    config['frozen_encoder']['mode']='hard-gated'
    config['hard_gate']=dict(min_h_improvement=args.min_h_improvement,rq_mse_ratio_cap=args.rq_mse_ratio_cap,
        comparison='same-step content proposal from same current model and AdamW moments',
        objective='full anchor_train mean ReLU(new NLL-old NLL); full new SID histories',
        tie_policy='content',infeasible_content_policy='rollback both and stop',
        commit='selected model + optimizer + RNG; sampler advances once',
        holdout_used=False)
    bound=binding(ex,args.plain_dir)
    resume=torch.load(args.resume_from,map_location='cpu',weights_only=False) if args.resume_from else None
    if resume and (resume.get('format')!=FORMAT or resume['config']!=config or resume['binding']!=bound):
        raise ValueError('Hard-gate resume data/source/config/reference binding mismatch')
    if resume and resume['sampler_state']['step']>=args.max_steps:
        raise ValueError('Target must exceed resumed steps')
    f.seed_all(config['seed'])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    model=f.build_model(config).to(device);model.load_state_dict(ex.parent['state_dict'],strict=True)
    model.encoder.requires_grad_(False);model.encoder.eval()
    if not all(layer.initted for layer in model.rq.vq_layers):raise ValueError('Base codebooks uninitialized')
    initial,initial_groups,old_codes=f.metrics(model,ex)
    if not np.array_equal(old_codes[:ex.groups['base_train'][1]],ex.old_export):
        raise ValueError('Initial Base assignments differ')
    scales=f.h.distance_scales(model,ex.features[:ex.groups['base_train'][1]],ex.old_batch)
    state=dict(step=0,cursor=0,order=None,epoch=0,processed_examples=0)
    work=dict(history_gradient_queries=0,candidate_hard_queries=0,optimizer_proposals=0,
        committed_content_steps=0,committed_historical_steps=0,identical_candidates=0,
        rejected_historical_reasons={},rolled_back_rounds=0,local_guard_checks=0,
        peak_effective_gradient_ratio_by_level=[0.,0.,0.],peak_relative_parameter_update_by_level=[0.,0.,0.])
    if resume:
        model.load_state_dict(resume['state_dict'],strict=True)
        state=copy.deepcopy(resume['sampler_state']);work=copy.deepcopy(resume['work_counters'])
        initial,initial_groups=resume['initial_metrics'],resume['initial_group_metrics']
        if scales!=resume['distance_scales']:raise ValueError('Base distance scales changed')
    codes,capacity,ledger,mapping=f.catalog(model,ex)
    if mapping is None:raise ValueError('Starting checkpoint full SID capacity exhausted')
    teacher=f.h.frozen_teacher(ex,device)
    history=f.FullHistory(model,teacher,ex,config,scales,progress_every=0)
    if resume and not torch.equal(history.old_nll(),resume['old_anchor_nll']):
        raise ValueError('Cached old reference changed')
    if not torch.equal(history.old_nll(),control_cp['old_anchor_nll']):
        raise ValueError('Old teacher reference differs from frozen Plain')
    start_history,_=history.measure(codes,mapping)
    first=state['step'];out.mkdir(parents=True)
    f.write_json(out/'config.json',dict(config=config,binding=bound,command=vars(args)))
    f.write_json(out/'anchor_train_tensor_rows.json',history.selected.tolist())
    trainer=f.build_trainer(model,config,out);optimizer=trainer.optimizer
    if resume:optimizer.load_state_dict(resume['optimizer']);f.restore_rng(resume['rng_state'])
    generator=torch.Generator(device='cpu').manual_seed(config['seed'])
    if 'generator_state' in state:generator.set_state(state['generator_state'])
    stop_reason='target_step_budget_reached';times=[];last_selected=None
    save(out/'last.pth',model,optimizer,config,bound,state,work,scales,initial,initial_groups,history)
    print('Hard-gated frozen-encoder Update | steps '+str(first)+'->'+str(args.max_steps)+
        ' | full training anchors='+str(len(history.selected))+' | rho='+str(args.history_gradient_ratio),flush=True)
    with (out/'metrics.jsonl').open('w') as log:
        while state['step']<args.max_steps:
            tick=time.perf_counter();before_sampler=copy.deepcopy(state);before_generator=generator.get_state()
            common=transaction(model,optimizer)
            if state['order'] is None or state['cursor']==len(ex.features):
                state['order']=torch.randperm(len(ex.features),generator=generator)
                state['cursor']=0;state['epoch']+=1
            end=min(len(ex.features),state['cursor']+config['batch_size'])
            chosen=state['order'][state['cursor']:end]
            model.train();model.encoder.eval();optimizer.zero_grad(set_to_none=True)
            data=ex.features[chosen].to(device)
            values=model(data,None,use_sk=False)
            loss,aux,recon,quant=model.compute_loss(values[0],values[1],chosen.to(device),values[3],
                dense_out_codebooks=values[4],x_res_list=values[5],xs=data)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite content loss')
            loss.backward()
            if any(p.grad is not None or p.requires_grad for p in model.encoder.parameters()):
                raise ValueError('Frozen encoder received a gradient')
            grads={n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()}
            books=history.books
            rq=[grads['rq.vq_layers.'+str(i)+'.embedding.weight'] for i in range(len(books))]
            rq=[g if g is not None else torch.zeros_like(p) for g,p in zip(rq,books)]
            if args.history_gradient_ratio>0:
                before_history,hgrads=history.measure(codes,mapping,backward=True,step=state['step']+1)
                work['history_gradient_queries']+=len(history.selected)
            else:
                before_history=None;hgrads=[torch.zeros_like(p) for p in books]
            mixed,gstats=f.normalized_mix(rq,hgrads,args.history_gradient_ratio,1e-12)
            joint_grads=dict(grads)
            for i,g in enumerate(mixed):
                name='rq.vq_layers.'+str(i)+'.embedding.weight'
                joint_grads[name]=g if grads[name] is not None else None
            content_tx=proposal(model,optimizer,common,grads);work['optimizer_proposals']+=1
            content,content_details=candidate_metrics(model,ex,history)
            if content['feasible']:work['candidate_hard_queries']+=len(history.selected)
            identical=all(grads[n] is None and joint_grads[n] is None or
                grads[n] is not None and joint_grads[n] is not None and torch.equal(grads[n],joint_grads[n]) for n in grads)
            if identical:
                joint_tx=content_tx;joint=copy.deepcopy(content);joint_details=content_details
                work['identical_candidates']+=1
            else:
                joint_tx=proposal(model,optimizer,common,joint_grads);work['optimizer_proposals']+=1
                joint,joint_details=candidate_metrics(model,ex,history)
                if joint['feasible']:work['candidate_hard_queries']+=len(history.selected)
            selected,reason=choose(content,joint,args.min_h_improvement,args.rq_mse_ratio_cap)
            if selected is None:
                restore_transaction(model,optimizer,common);state=before_sampler;generator.set_state(before_generator)
                work['rolled_back_rounds']+=1;stop_reason=reason
                f.write_json(out/'rejected_round.json',dict(content=content,historical=joint,reason=reason))
                print('No valid content comparator; restored prior model/optimizer/sampler and stopped.',flush=True)
                break
            selected_tx=joint_tx if selected=='historical' else content_tx
            selected_metrics=joint if selected=='historical' else content
            details=joint_details if selected=='historical' else content_details
            if selected_metrics['history']['positive_damage']>content['history']['positive_damage']:
                raise AssertionError('Local hard-H selection guard failed')
            restore_transaction(model,optimizer,selected_tx)
            work['local_guard_checks']+=1
            work['committed_'+selected+'_steps']+=1
            if selected=='content':work['rejected_historical_reasons'][reason]=work['rejected_historical_reasons'].get(reason,0)+1
            state['cursor']=end;state['step']+=1;state['processed_examples']+=len(chosen)
            state['generator_state']=generator.get_state()
            relative=[]
            for i,entry in enumerate(gstats):
                key='rq.vq_layers.'+str(i)+'.embedding.weight'
                old=common['state_dict'][key]
                change=float((selected_tx['state_dict'][key]-old).double().norm())/max(float(old.double().norm()),1e-30)
                relative.append(change)
                work['peak_effective_gradient_ratio_by_level'][i]=max(work['peak_effective_gradient_ratio_by_level'][i],entry['effective_gradient_ratio'])
                work['peak_relative_parameter_update_by_level'][i]=max(work['peak_relative_parameter_update_by_level'][i],change)
            row=dict(step=state['step'],epoch=state['epoch'],processed_examples=state['processed_examples'],
                content_batch_rq_loss=float(loss.detach()),history_before_optimizer_step=before_history,
                content_candidate=content,historical_candidate=joint,selected=selected,reason=reason,
                selected_history=selected_metrics['history'],selected_RQ_MSE=selected_metrics['rq_mse'],
                identical_candidate_gradients=identical,codebook_gradients=gstats,
                actual_relative_codebook_update=relative,
                local_H_change_vs_content=selected_metrics['history']['positive_damage']-content['history']['positive_damage'],
                step_runtime_seconds=time.perf_counter()-tick)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();times.append(row['step_runtime_seconds'])
            if state['step']==first+1 or state['step']%args.log_every==0 or state['step']==args.max_steps:
                print(json.dumps(dict(step=state['step'],selected=selected,reason=reason,
                    content_H=content['history']['positive_damage'],historical_H=joint.get('history',{}).get('positive_damage'),
                    selected_H=selected_metrics['history']['positive_damage'],RQ_MSE=selected_metrics['rq_mse'],
                    cumulative_historical_accepts=work['committed_historical_steps']),allow_nan=False),flush=True)
            codes,capacity,ledger,mapping=[details[k] for k in ('codes','capacity','ledger','mapping')]
            last_selected=selected_metrics
            if state['step']%args.save_every==0:
                save(out/'last.pth',model,optimizer,config,bound,state,work,scales,initial,initial_groups,history)
    save(out/'last.pth',model,optimizer,config,bound,state,work,scales,initial,initial_groups,history)
    final,final_groups,final_codes=f.metrics(model,ex);final_h,_=history.measure(final_codes,mapping)
    if not np.array_equal(codes,final_codes):raise ValueError('Final catalog batching changed assignments')
    if last_selected is not None and (abs(final_h['positive_damage']-last_selected['history']['positive_damage'])>1e-5
            or abs(final['reconstruction_loss']-last_selected['rq_mse'])>1e-10):
        raise ValueError('Committed final state differs from selected candidate')
    if any(not torch.equal(v.cpu(),ex.parent['state_dict']['encoder.'+k]) for k,v in model.encoder.state_dict().items()):
        raise ValueError('Frozen encoder changed')
    reloaded=f.build_model(config).to(device)
    reloaded.load_state_dict(torch.load(out/'last.pth',map_location='cpu',weights_only=False)['state_dict'],strict=True)
    if not np.array_equal(f.h.catalog_codes(reloaded,ex),final_codes):raise ValueError('Reload assignments differ')
    records=f.hard_records(history,final_codes,mapping)
    if abs(np.mean([r['positive_damage'] for r in records])-final_h['positive_damage'])>1e-5:
        raise ValueError('Endpoint per-query H differs')
    with (out/'anchor_train_hard_losses.jsonl').open('w') as handle:
        for r in records:handle.write(json.dumps(r,allow_nan=False)+'\n')
    with (out/'semantic_assignments.jsonl').open('w') as handle:
        for r,c in zip(ex.feature_rows,final_codes.tolist()):handle.write(json.dumps(dict(**r,semantic_code=c))+'\n')
    f.write_json(out/'new_allocation_registry.json',ledger);f.write_json(out/'new_item2tokens.json',mapping)
    f.write_json(out/'suffix_capacity_audit.json',capacity)
    f.h.check_teacher(teacher,ex)
    if f.sha256(Path(ex.plain['base_rqvae_dir'])/'last.pth')!=ex.binding['parent_base_checkpoint_sha256']:
        raise ValueError('Base checkpoint changed')
    if binding(ex,args.plain_dir)!=bound:raise ValueError('Bound sources or Plain reference changed during run')
    report=dict(stage=FORMAT,execution_status='completed' if state['step']==args.max_steps else 'stopped',
        stop_reason=stop_reason,output=str(out),e1_dir=str(ex.e1_dir.resolve()),plain_dir=str(Path(args.plain_dir).resolve()),
        binding=bound,config=config,update_optimizer_steps=state['step'],steps_this_run=state['step']-first,
        requested_update_optimizer_steps=args.max_steps,processed_examples=state['processed_examples'],
        sampler_epochs_started=state['epoch'],initial_train_catalog_metrics=initial,initial_group_metrics=initial_groups,
        final_train_catalog_metrics=final,final_group_metrics=final_groups,
        full_anchor_train_at_run_start=start_history,final_full_anchor_train=final_h,
        collision_statistics=f.collision_stats(final_codes,config['model']['num_emb_list']),
        base_semantic_churn=f.churn(ex.old_export,final_codes[:ex.groups['base_train'][1]]),
        work_counters=work,encoder_frozen=True,encoder_unchanged_from_base=True,
        teacher_frozen=True,parent_checkpoint_unchanged=True,checkpoint_reload_assignments_equal=True,
        starting_checkpoint='exact bound Base C_old; AdamW reset on fresh run; both restored on resume',
        trained_components=['decoder','three_codebooks'],historical_gradient_components=['three_codebook_tables'],
        full_sid_export_status='completed',anchor_train_queries=len(history.selected),holdout_used=False,
        base_validation_prediction_evaluated=False,ranking_evaluated=False,future_test_used=False,
        per_step_local_guard_passed=work['local_guard_checks']==state['step'],
        independent_plain_endpoint_dominance_guaranteed=False,same_compute_as_plain=False,
        checkpoint_selection='last at fixed committed-step budget; no holdout selection',
        resume_from=str(Path(args.resume_from).resolve()) if args.resume_from else None,
        median_step_seconds_this_run=float(np.median(times)) if times else None,
        runtime_seconds=time.perf_counter()-started,device=str(device),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        artifact_files_sha256={p.name:f.sha256(p) for p in out.iterdir() if p.is_file()})
    f.write_json(out/'summary.json',report);trainer.tb_writer.close()
    print(json.dumps(dict(execution_status=report['execution_status'],steps=state['step'],
        historical_accepts=work['committed_historical_steps'],final_training_H=final_h['positive_damage'],
        RQ_MSE=final['reconstruction_loss'],full_sid_export_status='completed'),indent=2),flush=True)
    return report


def compare(args):
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite hard-gate comparison')
    directory=Path(args.rqvae_dir).resolve();report=f.read_json(directory/'summary.json')
    if report.get('stage')!=FORMAT or report.get('execution_status')!='completed':
        raise ValueError('Expected completed hard-gated endpoint')
    required=['last.pth','config.json','anchor_train_hard_losses.jsonl','metrics.jsonl',
        'new_allocation_registry.json','new_item2tokens.json','semantic_assignments.jsonl']
    f.verify_files(directory,report,required)
    cp=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
    if (cp.get('format')!=FORMAT or cp['config']!=report['config'] or cp['binding']!=report['binding']
            or cp['sampler_state']['step']!=report['update_optimizer_steps']):
        raise ValueError('Hard-gate checkpoint/report mismatch')
    plain_dir=Path(report['plain_dir']);plain,pcp=read_control(plain_dir)
    b=report['binding']
    if (b['reference']!=plain['binding'] or b['plain_summary_sha256']!=f.sha256(plain_dir/'summary.json')
            or b['plain_checkpoint_sha256']!=f.sha256(plain_dir/'last.pth')
            or b['hard_gate_source_sha256']!=f.sha256(Path(__file__))
            or b['reference']['frozen_source_sha256']!=f.sha256(Path(f.__file__))
            or b['reference']['v2_source_sha256']!=f.sha256(Path(f.v2.__file__))):
        raise ValueError('Bound Plain or source changed')
    if report['update_optimizer_steps']!=plain['update_optimizer_steps']:
        raise ValueError('Formal comparison requires the same completed Plain step budget')
    configs=[copy.deepcopy(x['config']) for x in (report,plain)]
    configs[0].pop('hard_gate')
    for c in configs:
        c['frozen_encoder'].pop('mode');c['historical_v2'].pop('history_gradient_ratio')
    if configs[0]!=configs[1]:raise ValueError('Content/forward configuration differs from Plain')
    for key in ('step','cursor','epoch','processed_examples'):
        if cp['sampler_state'][key]!=pcp['sampler_state'][key]:raise ValueError('Sampler endpoint differs: '+key)
    for key in ('order','generator_state'):
        if not torch.equal(cp['sampler_state'][key],pcp['sampler_state'][key]):raise ValueError('Sampler stream differs: '+key)
    if not torch.equal(cp['old_anchor_nll'],pcp['old_anchor_nll']):raise ValueError('Old teacher reference differs')
    for key,value in cp['state_dict'].items():
        if key.startswith('encoder.') and not torch.equal(value,pcp['state_dict'][key]):
            raise ValueError('Frozen encoders differ')
    records=[f.read_jsonl(d/'anchor_train_hard_losses.jsonl') for d in (directory,plain_dir)]
    if not records[0] or len(records[0])!=len(records[1]):raise ValueError('Training cohort size differs')
    differences=[];clusters={}
    for new,old in zip(*records):
        for key in ('sample_id','user','source_row','tensor_row','target_labels','old_sequence_nll'):
            if new[key]!=old[key]:raise ValueError('Paired old training reference differs: '+key)
        diff=[new['positive_damage']-old['positive_damage'],new['new_sequence_nll']-old['new_sequence_nll']]
        differences.append(diff);group=clusters.setdefault(new['user'],np.zeros(3));group[:2]+=diff;group[2]+=1
    for r,rows in zip((report,plain),records):
        if abs(np.mean([x['positive_damage'] for x in rows])-r['final_full_anchor_train']['positive_damage'])>1e-5:
            raise ValueError('Endpoint hard H disagrees with query records')
    rows=f.read_jsonl(directory/'metrics.jsonl')
    for row in rows:
        selected,reason=choose(row['content_candidate'],row['historical_candidate'],
            report['config']['hard_gate']['min_h_improvement'],report['config']['hard_gate']['rq_mse_ratio_cap'])
        if selected!=row['selected'] or reason!=row['reason'] or row['local_H_change_vs_content']>0:
            raise ValueError('Saved candidate decision violates the fixed local gate')
    if not report.get('per_step_local_guard_passed') or report['work_counters']['local_guard_checks']!=report['update_optimizer_steps']:
        raise ValueError('Incomplete local guard counters')
    groups=np.array(list(clusters.values()));rng=np.random.default_rng(report['config']['seed']);draws=[]
    for _ in range(500):
        total=groups[rng.integers(0,len(groups),len(groups))].sum(0);draws.append(total[:2]/total[2])
    ci=np.quantile(draws,[.025,.975],axis=0).T.tolist() if len(groups)>1 else [None,None]
    means=np.mean(differences,axis=0)
    ungated=None
    if args.ungated_dir:
        d=Path(args.ungated_dir).resolve();u=f.read_json(d/'summary.json')
        if (u.get('stage')!=f.FORMAT or u.get('execution_status')!='completed' or u.get('mode')!='history-aware'
                or u['binding']!=plain['binding'] or u['update_optimizer_steps']!=report['update_optimizer_steps']):
            raise ValueError('Optional ungated endpoint binding/mode/step mismatch')
        f.verify_files(d,u,['last.pth','config.json','anchor_train_hard_losses.jsonl'])
        uc=torch.load(d/'last.pth',map_location='cpu',weights_only=False)
        if (uc['binding']!=u['binding'] or uc['config']!=u['config']
                or uc['sampler_state']['step']!=report['update_optimizer_steps']
                or uc['config']['historical_v2']['history_gradient_ratio']!=report['config']['historical_v2']['history_gradient_ratio']):
            raise ValueError('Optional ungated checkpoint/rho mismatch')
        uconfig=copy.deepcopy(uc['config'])
        uconfig['frozen_encoder'].pop('mode');uconfig['historical_v2'].pop('history_gradient_ratio')
        if uconfig!=configs[1] or not torch.equal(uc['old_anchor_nll'],pcp['old_anchor_nll']):
            raise ValueError('Optional ungated content/forward/reference differs')
        for key in ('order','generator_state'):
            if not torch.equal(uc['sampler_state'][key],pcp['sampler_state'][key]):
                raise ValueError('Optional ungated sampler differs')
        ungated=dict(directory=str(d),history=u['final_full_anchor_train'],
            rq_mse=u['final_train_catalog_metrics']['reconstruction_loss'])
    result=dict(stage='historical_hard_gate_training_comparison',execution_status='completed',
        update_optimizer_steps=report['update_optimizer_steps'],queries=len(differences),users=len(clusters),
        rho=report['config']['historical_v2']['history_gradient_ratio'],
        rq_mse_ratio_cap=report['config']['hard_gate']['rq_mse_ratio_cap'],
        min_h_improvement=report['config']['hard_gate']['min_h_improvement'],
        binding=b,plain_dir=str(plain_dir),rqvae_dir=str(directory),
        hard_gate_checkpoint_sha256=f.sha256(directory/'last.pth'),
        plain=dict(history=plain['final_full_anchor_train'],rq_mse=plain['final_train_catalog_metrics']['reconstruction_loss'],
            collision_statistics=plain['collision_statistics']),
        hard_gated=dict(history=report['final_full_anchor_train'],rq_mse=report['final_train_catalog_metrics']['reconstruction_loss'],
            collision_statistics=report['collision_statistics']),
        ungated=ungated,hard_gated_minus_plain=dict(positive_damage=float(means[0]),mean_sequence_nll=float(means[1]),
            positive_damage_cluster_ci95=ci[0],sequence_nll_cluster_ci95=ci[1],
            rq_mse=report['final_train_catalog_metrics']['reconstruction_loss']-plain['final_train_catalog_metrics']['reconstruction_loss']),
        work_counters=report['work_counters'],median_step_seconds=report['median_step_seconds_this_run'],
        per_step_local_guard_passed=True,independent_plain_endpoint_dominance_guaranteed=False,
        same_base=True,encoder_unchanged_both=True,same_content_optimizer_and_sampler=True,same_compute=False,
        holdout_used=False,ranking_evaluated=False,future_test_used=False,
        interpretation='training-only paired diagnostic; local gate does not ensure independent Plain or holdout superiority',
        bootstrap_unit='user; row-weighted paired mean; training cohort only',bootstrap_replicates=500)
    out.mkdir(parents=True);f.write_json(out/'summary.json',result)
    print(json.dumps(dict(execution_status='completed',hard_gated_minus_plain=result['hard_gated_minus_plain']),indent=2),flush=True)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--e1-dir',required=True);t.add_argument('--plain-dir',required=True)
    t.add_argument('--output',required=True);t.add_argument('--device',default='cuda')
    t.add_argument('--max-steps',type=int,default=100);t.add_argument('--history-gradient-ratio',type=float,default=.1)
    t.add_argument('--anchor-batch-size',type=int,default=32);t.add_argument('--temperature',type=float,default=.1)
    t.add_argument('--min-h-improvement',type=float,default=1e-6);t.add_argument('--rq-mse-ratio-cap',type=float,default=1.01)
    t.add_argument('--log-every',type=int,default=10);t.add_argument('--save-every',type=int,default=10)
    t.add_argument('--resume-from')
    c=sub.add_parser('compare');c.add_argument('--rqvae-dir',required=True);c.add_argument('--output',required=True)
    c.add_argument('--ungated-dir')
    return p


if __name__=='__main__':
    args=parser().parse_args();train(args) if args.command=='train' else compare(args)
