"""Bounded historical gradient ratio experiments; immutable earlier engines unchanged.

Train loop copied from history_update_multiseed.py. Only the supported rho range,
source binding and format change. Independent evaluation can use training anchors
only; held-out evaluation is a separate, post-selection action.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import history_update_multiseed as prior
import history_aware_frozen_encoder as legacy
import audit_history_pair as audit
from history_aware_frozen_encoder import (h,snapshot,catalog,FullHistory,normalized_mix,metrics,
    build_model,build_trainer,seed_all,sha256,write_json,collision_stats,rng_state,restore_rng,
    churn,read_json,read_jsonl,verify_files,hard_records)

FORMAT='historical_weight_update_v1'
EVALUATION='historical_weight_independent_comparison_v1'


def binding(ex):
    return dict(experiment=ex.binding,weight_source_sha256=sha256(Path(__file__)),
        previous_multiseed_source_sha256=sha256(Path(prior.__file__)),
        frozen_source_sha256=sha256(Path(legacy.__file__)),v2_source_sha256=sha256(Path(legacy.v2.__file__)),
        independent_auditor_sha256=sha256(Path(audit.__file__)))


def batch_hash(previous,chosen):
    payload=np.asarray(chosen,dtype='<i8').tobytes()
    return hashlib.sha256(bytes.fromhex(previous)+payload).hexdigest()


def expected_sampler(seed,count,batch_size,steps):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    state=dict(step=0,cursor=0,order=None,epoch=0,processed_examples=0,batch_trace_sha256='0'*64)
    for _ in range(steps):
        if state['order'] is None or state['cursor']==count:
            state['order']=torch.randperm(count,generator=generator);state['cursor']=0;state['epoch']+=1
        end=min(count,state['cursor']+batch_size);chosen=state['order'][state['cursor']:end]
        state['batch_trace_sha256']=batch_hash(state['batch_trace_sha256'],chosen)
        state['cursor']=end;state['step']+=1;state['processed_examples']+=len(chosen)
    state['generator_state']=generator.get_state()
    return state


def sampler_core(state):
    return {k:state[k] for k in ('step','cursor','order','epoch','processed_examples','generator_state')}


def save_checkpoint(path,model,optimizer,config,bound,sampler,work,scales,initial,initial_groups,history):
    temp=Path(str(path)+'.tmp')
    torch.save(dict(format=FORMAT,config=config,binding=bound,state_dict=snapshot(model),
        optimizer=optimizer.state_dict(),sampler_state=copy.deepcopy(sampler),rng_state=rng_state(),
        work_counters=copy.deepcopy(work),distance_scales=scales,initial_metrics=initial,
        initial_group_metrics=initial_groups,old_anchor_nll=history.old_nll()),temp)
    os.replace(temp,path)


def train(args):
    started = time.perf_counter()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite multiseed update output: '+str(out))
    if (min(args.max_steps, args.anchor_batch_size, args.log_every, args.save_every) < 1
            or not np.isfinite(args.history_gradient_ratio) or not 0 <= args.history_gradient_ratio <= 10
            or not np.isfinite(args.temperature) or args.temperature <= 0 or args.history_progress_every < 0):
        raise ValueError('Invalid joint update settings')
    if not 0 <= args.update_seed < 2**32:
        raise ValueError('Update seed must be in [0, 2**32)')
    args = copy.copy(args)
    args.history_gradient_ratio = 0. if args.mode == 'plain' else args.history_gradient_ratio
    ex, device = h.experiment(args), h.device_from(args.device)
    if args.max_steps > ex.plain['update_optimizer_steps']:
        raise ValueError('Joint target step budget exceeds bound Plain Update budget')
    if ex.parent['config']['model']['bn'] or ex.parent['config']['model']['dropout_prob'] != 0:
        raise ValueError('This comparison requires the bound deterministic content MLP')
    config = copy.deepcopy(ex.plain['config'])
    config['device'] = str(device)
    config['seed'] = args.update_seed
    config['update_randomness'] = dict(update_seed=args.update_seed,
        base_checkpoint_seed=ex.parent['config']['seed'],anchor_cohort_seed=ex.e1['cohort_seed'],
        scope='fixed Base/teacher/anchors; change Update content sampling order only')
    config['historical_v2'] = dict(history_gradient_ratio=args.history_gradient_ratio,
        normalization='per-layer current content gradient norm; detached scaling', epsilon=1e-12,
        history_gradient_components='codebook tables only; encoder output detached',
        anchor_batch_size=args.anchor_batch_size, anchor_coverage='all anchor_train on each history gradient step; Plain uses log-step hard probes',
        temperature=args.temperature, history_input_condition='new_full_sid',
        suffix_gradient='none; deterministic complete SID allocation in forward',
        capacity_policy='rollback invalid candidate and stop at last feasible checkpoint')
    config['frozen_encoder'] = dict(mode=args.mode, content_encoder_frozen=True,
        trainable_components=['decoder', 'three_codebooks'])
    bound = binding(ex)
    resume = torch.load(args.resume_from, map_location='cpu', weights_only=False) if args.resume_from else None
    if resume and (resume.get('format') != FORMAT or resume['config'] != config or resume['binding'] != bound):
        raise ValueError('Joint resume data/source/package/config binding mismatch')
    if resume and resume['sampler_state']['step'] > args.max_steps:
        raise ValueError('Target budget must not be below resumed checkpoint steps')
    seed_all(config['seed'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config).to(device)
    model.load_state_dict(ex.parent['state_dict'], strict=True)
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    if not all(layer.initted for layer in model.rq.vq_layers):
        raise ValueError('Uninitialized Base codebooks')
    warm_equal = all(torch.equal(v.cpu(), ex.parent['state_dict'][k]) for k, v in model.state_dict().items())
    initial, initial_groups, old_codes = metrics(model, ex)
    if not np.array_equal(old_codes[:ex.groups['base_train'][1]], ex.old_export):
        raise ValueError('Base assignments changed before joint Update')
    scales = h.distance_scales(model, ex.features[:ex.groups['base_train'][1]], ex.old_batch)
    state = dict(step=0, cursor=0, order=None, epoch=0, processed_examples=0, batch_trace_sha256='0'*64)
    work = dict(post_step_probe_queries=0, history_gradient_queries=0, old_reference_queries=0, full_catalog_assignment_passes=0,
        full_sid_capacity_checks=0, optimizer_attempts=0, rolled_back_optimizer_attempts=0,
        nonzero_history_gradient_steps=0, peak_effective_gradient_ratio_by_level=[0.,0.,0.],
        peak_relative_parameter_update_by_level=[0.,0.,0.])
    if resume:
        model.load_state_dict(resume['state_dict'], strict=True)
        state = copy.deepcopy(resume['sampler_state'])
        work = copy.deepcopy(resume['work_counters'])
        initial, initial_groups = resume['initial_metrics'], resume['initial_group_metrics']
        if scales != resume['distance_scales']:
            raise ValueError('Base calibration scales differ from resumed checkpoint')
    codes, capacity, ledger, mapping = catalog(model, ex)
    if mapping is None:
        raise ValueError('Initial checkpoint does not have feasible complete SIDs')
    teacher = h.frozen_teacher(ex, device)
    history = FullHistory(model, teacher, ex, config, scales, args.history_progress_every)
    if resume and not torch.equal(history.old_nll(), resume['old_anchor_nll']):
        raise ValueError('Resumed old teacher anchor NLL differs')
    work['old_reference_queries'] += len(history.selected)
    initial_history, _ = history.measure(codes, mapping)
    first_step = state['step']
    out.mkdir(parents=True)
    write_json(out/'config.json', dict(config=config, binding=bound, command=vars(args)))
    write_json(out/'anchor_train_tensor_rows.json', history.selected.tolist())
    trainer = build_trainer(model, config, out)
    optimizer = trainer.optimizer
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        restore_rng(resume['rng_state'])
    generator = torch.Generator(device='cpu').manual_seed(config['seed'])
    if 'generator_state' in state:
        generator.set_state(state['generator_state'])
    books = history.books
    stop_reason, failed_capacity = 'target_step_budget_reached', None
    step_times = []
    save_checkpoint(out/'last.pth', model, optimizer, config, bound, state, work, scales,
                    initial, initial_groups, history)
    print('Update seed '+str(args.update_seed)+' '+args.mode+' | steps '+str(first_step)+'->'+str(args.max_steps)+
        ' | full anchors='+str(len(history.selected))+' | rho='+str(args.history_gradient_ratio)+
        ' | complete SID forward | historical encoder detached', flush=True)
    with (out/'metrics.jsonl').open('w') as log:
        while state['step'] < args.max_steps:
            step_started = time.perf_counter()
            before_state = copy.deepcopy(state)
            before_rng, before_generator = rng_state(), generator.get_state()
            before_model, before_optimizer = snapshot(model), copy.deepcopy(optimizer.state_dict())
            if state['order'] is None or state['cursor'] == len(ex.features):
                state['order'] = torch.randperm(len(ex.features), generator=generator)
                state['cursor'] = 0; state['epoch'] += 1
            end = min(len(ex.features), state['cursor']+config['batch_size'])
            chosen = state['order'][state['cursor']:end]
            data = ex.features[chosen].to(device)
            model.train()
            model.encoder.eval()
            optimizer.zero_grad(set_to_none=True)
            out_values, quant_loss, _, dense, out_books, residuals = model(data, None, use_sk=False)
            rq_loss, aux, recon, quant = model.compute_loss(out_values, quant_loss, chosen.to(device), dense,
                dense_out_codebooks=out_books, x_res_list=residuals, xs=data)
            if not torch.isfinite(rq_loss):
                raise FloatingPointError('Nonfinite RQ loss')
            rq_loss.backward()
            if any(p.requires_grad or p.grad is not None for p in model.encoder.parameters()):
                raise ValueError('Frozen encoder received a gradient')
            rq_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in books]
            # autograd.grad for H does not overwrite any existing content .grad.
            if args.history_gradient_ratio > 0:
                history_metrics, history_grads = history.measure(codes, mapping, backward=True, step=state['step']+1)
                work['history_gradient_queries'] += len(history.selected)
            else:
                history_metrics = None
                history_grads = [torch.zeros_like(p) for p in books]
            combined, gradient_stats = normalized_mix(rq_grads, history_grads, args.history_gradient_ratio, 1e-12)
            if args.history_gradient_ratio > 0:
                for parameter, value in zip(books, combined):
                    parameter.grad = value
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite joint parameter gradient')
            optimizer.step()
            work['optimizer_attempts'] += 1
            next_codes, next_capacity, next_ledger, next_mapping = catalog(model, ex)
            work['full_catalog_assignment_passes'] += 1
            work['full_sid_capacity_checks'] += 1
            if next_mapping is None:
                failed_capacity = next_capacity
                model.load_state_dict(before_model, strict=True)
                optimizer.load_state_dict(before_optimizer)
                state = before_state
                generator.set_state(before_generator); restore_rng(before_rng)
                optimizer.zero_grad(set_to_none=True)
                work['rolled_back_optimizer_attempts'] += 1
                stop_reason = 'candidate_full_sid_capacity_exhausted_rolled_back'
                print('Candidate capacity exhausted; restored last feasible model/optimizer/sampler.', flush=True)
                break
            state['cursor'] = end; state['step'] += 1
            state['processed_examples'] += len(chosen)
            state['batch_trace_sha256'] = batch_hash(state['batch_trace_sha256'], chosen)
            state['generator_state'] = generator.get_state()
            updates = []
            for i, layer in enumerate(model.rq.vq_layers):
                key = 'rq.vq_layers.'+str(i)+'.embedding.weight'
                old = before_model[key].to(device)
                updates.append(float((layer.embedding.weight.detach()-old).double().norm())/
                               max(float(old.double().norm()), 1e-30))
            codebooks = [dict(level=i+1, **entry, actual_relative_parameter_update=updates[i])
                         for i, entry in enumerate(gradient_stats)]
            work['nonzero_history_gradient_steps'] += int(any(x['history_gradient_norm'] > 0 for x in codebooks))
            for i, entry in enumerate(codebooks):
                work['peak_effective_gradient_ratio_by_level'][i] = max(
                    work['peak_effective_gradient_ratio_by_level'][i], entry['effective_gradient_ratio'])
                work['peak_relative_parameter_update_by_level'][i] = max(
                    work['peak_relative_parameter_update_by_level'][i], entry['actual_relative_parameter_update'])
            after_history = None
            if state['step'] % args.log_every == 0 or state['step'] == args.max_steps:
                after_history, _ = history.measure(next_codes, next_mapping)
                work['post_step_probe_queries'] += len(history.selected)
            row = dict(content_batch_trace_sha256=state['batch_trace_sha256'], history_after_optimizer_step=after_history, step=state['step'], epoch=state['epoch'], processed_examples=state['processed_examples'],
                rq_loss=float(rq_loss.detach()), reconstruction_loss=float(recon), quantization_loss=float(quant.detach()),
                history_before_optimizer_step=history_metrics, codebook_gradients=codebooks,
                full_sid_allocation_feasible=True, maximum_collision_bucket=next_capacity['maximum_current_collision_bucket'],
                maximum_required_suffix_capacity=next_capacity['maximum_required_suffix_capacity'],
                unique_semantic_codes=len(set(map(tuple, next_codes.tolist()))),
                history_gradient_queries_processed=work['history_gradient_queries'],
                step_runtime_seconds=time.perf_counter()-step_started)
            step_times.append(row['step_runtime_seconds'])
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            if state['step'] == first_step+1 or state['step'] % args.log_every == 0 or state['step'] == args.max_steps:
                print(json.dumps(row, allow_nan=False), flush=True)
            codes, capacity, ledger, mapping = next_codes, next_capacity, next_ledger, next_mapping
            if state['step'] % args.save_every == 0:
                save_checkpoint(out/'last.pth', model, optimizer, config, bound, state, work, scales,
                                initial, initial_groups, history)
    save_checkpoint(out/'last.pth', model, optimizer, config, bound, state, work, scales,
                    initial, initial_groups, history)
    final, final_groups, final_codes = metrics(model, ex)
    final_history, _ = history.measure(final_codes, mapping)
    restored = build_model(config).to(device)
    restored.load_state_dict(torch.load(out/'last.pth', map_location='cpu', weights_only=False)['state_dict'], strict=True)
    if not np.array_equal(h.catalog_codes(restored, ex), final_codes):
        raise ValueError('Checkpoint reload semantic assignments differ')
    if not np.array_equal(codes, final_codes):
        raise ValueError('Final content evaluation changed assignment batching')
    encoder_equal = all(torch.equal(v.cpu(), ex.parent['state_dict']['encoder.'+k])
                        for k, v in model.encoder.state_dict().items())
    if not encoder_equal:
        raise ValueError('Frozen encoder differs from Base checkpoint')
    records = hard_records(history, final_codes, mapping)
    with (out/'anchor_train_hard_losses.jsonl').open('w') as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    if abs(np.mean([r['positive_damage'] for r in records])-final_history['positive_damage']) > 1e-5:
        raise ValueError('Final hard-loss records disagree with history measurement')
    h.check_teacher(teacher, ex)
    if sha256(Path(ex.plain['base_rqvae_dir'])/'last.pth') != ex.binding['parent_base_checkpoint_sha256']:
        raise ValueError('Base source checkpoint changed')
    with (out/'semantic_assignments.jsonl').open('w') as f:
        for row, semantic in zip(ex.feature_rows, final_codes.tolist()):
            f.write(json.dumps(dict(**row, semantic_code=semantic))+'\n')
    write_json(out/'new_allocation_registry.json', ledger)
    write_json(out/'new_item2tokens.json', mapping)
    write_json(out/'suffix_capacity_audit.json', dict(saved_checkpoint=capacity, rejected_candidate=failed_capacity))
    report = dict(stage=FORMAT, execution_status='completed' if state['step'] == args.max_steps else 'stopped',
        e1_dir=str(ex.e1_dir.resolve()), output=str(out), binding=bound, config=config,
        update_seed=args.update_seed,base_checkpoint_seed=ex.parent['config']['seed'],
        anchor_cohort_seed=ex.e1['cohort_seed'],content_batch_trace_sha256=state['batch_trace_sha256'],
        requested_update_optimizer_steps=args.max_steps, update_optimizer_steps=state['step'],
        steps_this_run=state['step']-first_step, stop_reason=stop_reason,
        resume_from=str(Path(args.resume_from).resolve()) if args.resume_from else None,
        starting_checkpoint='bound C_old Base, not Plain Update or repair checkpoint',
        parent_base_checkpoint_sha256=ex.binding['parent_base_checkpoint_sha256'],
        warm_start_all_weights_equal=warm_equal, original_base_assignments_equal_before_update=True,
        optimizer_initialization='reset repository AdamW, same Plain settings; restored on resume',
        content_sampling='same uniform shuffled eligible catalog/order as Plain; original partial batches',
        processed_examples=state['processed_examples'], sampler_epochs_started=state['epoch'],
        initial_train_catalog_metrics=initial, initial_group_metrics=initial_groups,
        final_train_catalog_metrics=final, final_group_metrics=final_groups,
        original_unfrozen_plain_reference_metrics=ex.plain['final_train_catalog_metrics'],
        full_anchor_train_at_run_start=initial_history, final_full_anchor_train=final_history,
        anchor_train_queries=len(history.selected), anchor_holdout_used_for_training=False,
        base_validation_used_for_training=False, trained_components=['decoder','three_codebooks'], mode=args.mode,
        encoder_frozen=True, encoder_unchanged_from_base=encoder_equal,
        historical_gradient_components=['three_codebook_tables'], historical_encoder_output_detached=True,
        historical_gradient_estimator='hard semantic forward; soft-distance STE backward; fixed Base distance scales',
        history_input_condition='new_full_sid', collision_suffix_gradient='none', old_target_labels_fixed=True,
        checkpoint_selection='last at fixed optimizer-step budget; no holdout or validation selection',
        collision_statistics=collision_stats(final_codes, config['model']['num_emb_list']),
        base_semantic_churn=churn(ex.old_export, final_codes[:ex.groups['base_train'][1]]),
        full_sid_export_status='completed', saved_checkpoint_deployment_sid_ready=True,
        teacher_frozen=True, teacher_parameters_unchanged=True, parent_checkpoint_unchanged=True,
        checkpoint_reload_assignments_equal=True, work_counters=work,
        original_unfrozen_plain_reference_steps=ex.plain['update_optimizer_steps'],
        matched_comparison_status='run compare with separately trained frozen-encoder Plain',
        median_optimizer_step_seconds_this_run=float(np.median(step_times)) if step_times else None,
        same_compute_as_plain=False, graph_used=False, sinkhorn_used=False, content_encoder_finetuning=False,
        recommendation_training_status='not_run', future_test_status='not_run',
        historical_damage_evaluation_status='not_run', device=str(device), runtime_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        artifact_files_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'summary.json', report)
    trainer.tb_writer.close()
    print(json.dumps(dict(update_optimizer_steps=state['step'], execution_status=report['execution_status'],
        stop_reason=stop_reason, final_full_anchor_train=final_history,
        RQ_MSE=final['reconstruction_loss'], full_sid_export_status='completed'), indent=2), flush=True)
    return report


def design(config,ignore_rho=False):
    c=copy.deepcopy(config);c['seed']=None;c.pop('update_randomness',None)
    c['device']=None  # physical GPU placement is not a content/historical hyperparameter
    c['frozen_encoder'].pop('mode')
    if ignore_rho:c['historical_v2'].pop('history_gradient_ratio')
    return c


def load_pair(args):
    ex=h.experiment(args);reports=[];cps=[];formats=[]
    for directory,mode in zip((Path(args.plain_dir),Path(args.history_dir)),('plain','history-aware')):
        r=read_json(directory/'summary.json');kind=r.get('stage')
        if kind not in (FORMAT,prior.FORMAT,legacy.FORMAT) or r.get('execution_status')!='completed' or r.get('mode')!=mode:
            raise ValueError('Expected completed, ungated frozen-encoder paired endpoint: '+mode)
        expected_binding=binding(ex) if kind==FORMAT else (prior.binding(ex) if kind==prior.FORMAT else legacy.binding(ex))
        if r['binding']!=expected_binding:raise ValueError('Endpoint source/data binding mismatch')
        verify_files(directory,r,['last.pth','config.json','anchor_train_tensor_rows.json',
            'anchor_train_hard_losses.jsonl','semantic_assignments.jsonl','new_item2tokens.json','new_allocation_registry.json'])
        cp=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
        if (cp.get('format')!=kind or cp['binding']!=r['binding'] or cp['config']!=r['config']
                or cp['sampler_state']['step']!=r['update_optimizer_steps']):raise ValueError('Checkpoint/report mismatch')
        if r['update_optimizer_steps']!=args.expected_steps:
            raise ValueError('Endpoint does not match the predeclared step budget')
        seed=cp['config']['seed']
        if seed!=args.update_seed:raise ValueError('Requested Update seed differs from checkpoint')
        if kind in (FORMAT,prior.FORMAT):
            meta=cp['config']['update_randomness']
            if (meta['update_seed']!=seed or meta['base_checkpoint_seed']!=ex.parent['config']['seed']
                    or meta['anchor_cohort_seed']!=ex.e1['cohort_seed']):raise ValueError('Base/cohort/Update seed metadata differ')
        if read_json(directory/'anchor_train_tensor_rows.json')!=ex.cohorts['anchor_train']:
            raise ValueError('Training cohort differs')
        if any(not torch.equal(v,ex.parent['state_dict'][k]) for k,v in cp['state_dict'].items() if k.startswith('encoder.')):
            raise ValueError('Frozen encoder differs from Base')
        expected=expected_sampler(seed,len(ex.features),cp['config']['batch_size'],r['update_optimizer_steps'])
        if not audit.equal(sampler_core(cp['sampler_state']),sampler_core(expected)):
            raise ValueError('Actual sampler differs from the declared Update seed')
        if kind in (FORMAT,prior.FORMAT) and (cp['sampler_state']['batch_trace_sha256']!=expected['batch_trace_sha256']
                or r['content_batch_trace_sha256']!=expected['batch_trace_sha256']):
            raise ValueError('Actual per-step batch digest differs from expected order')
        reports.append(r);cps.append(cp);formats.append(kind)
    if design(cps[0]['config'],True)!=design(cps[1]['config'],True):raise ValueError('Within-seed configurations differ beyond rho/mode')
    if cps[0]['config']['historical_v2']['history_gradient_ratio']!=0 or cps[1]['config']['historical_v2']['history_gradient_ratio']<=0:
        raise ValueError('Plain must have rho=0 and History-aware rho>0')
    if not audit.equal(sampler_core(cps[0]['sampler_state']),sampler_core(cps[1]['sampler_state'])):raise ValueError('Within-seed actual content batch order differs')
    if not torch.equal(cps[0]['old_anchor_nll'],cps[1]['old_anchor_nll']):raise ValueError('Within-seed old teacher reference differs')
    if not np.array_equal(ex.arrays['base_train']['source_rows'],np.arange(len(ex.rows['base_train']))):
        raise ValueError('Base raw/tensor identity assumption fails')
    train_users={ex.rows['base_train'][i]['user'] for i in ex.cohorts['anchor_train']}
    hold_users={ex.rows['base_train'][i]['user'] for i in ex.cohorts['anchor_holdout']}
    if train_users & hold_users:raise ValueError('Training/holdout users overlap')
    return ex,reports,cps,expected['batch_trace_sha256']


def evaluate(args):
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite seed evaluation')
    ex,reports,cps,trace=load_pair(args);device=h.device_from(args.device)
    if args.batch_size!=reports[0]['config']['historical_v2']['anchor_batch_size']:
        raise ValueError('Evaluation batch size must match the saved old reference')
    seed_all(ex.e1['cohort_seed']);teacher=h.frozen_teacher(ex,device)
    mappings=[audit.endpoint(d,cp,ex,device) for d,cp in zip((args.plain_dir,args.history_dir),cps)]
    groups={};hashes={};pending={};error=0.
    tasks=[('anchor_train','base_train',ex.cohorts['anchor_train']),('anchor_holdout','base_train',ex.cohorts['anchor_holdout']),
        ('base_validation','base_val',list(range(len(ex.arrays['base_val']['labels']))))]
    if getattr(args,'train_only',False):tasks=tasks[:1]
    for name,split,chosen in tasks:
        records=[audit.independent_records(teacher,ex,split,chosen,m,args.batch_size,device,args.progress) for m in mappings]
        hashes[name]=audit.cohort_hash(ex,split,chosen)
        if name=='anchor_train':
            for rows,directory,cp in zip(records,(args.plain_dir,args.history_dir),cps):
                error=max(error,audit.check_saved(rows,directory))
                if max(abs(r['old_sequence_nll']-float(v)) for r,v in zip(rows,cp['old_anchor_nll']))>audit.NLL_TOLERANCE:
                    raise ValueError('Independent old NLL differs from cached training reference')
        groups[name]=dict(plain=audit.positive_summary(records[0]),history_aware=audit.positive_summary(records[1]),
            paired=audit.paired(records[1],records[0],ex.e1['cohort_seed']))
        for arm,rows in zip(('plain','history'),records):pending[name+'_'+arm+'.jsonl']=rows
        print('seed '+str(args.update_seed)+' '+name+': '+json.dumps(groups[name]),flush=True)
    h.check_teacher(teacher,ex)
    out.mkdir(parents=True)
    for name,rows in pending.items():
        with (out/name).open('w') as handle:
            for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
    report=dict(stage=EVALUATION,execution_status='completed',update_seed=args.update_seed,
        base_checkpoint_seed=ex.parent['config']['seed'],anchor_cohort_seed=ex.e1['cohort_seed'],
        common_binding=ex.binding,design=design(cps[1]['config']),update_optimizer_steps=reports[0]['update_optimizer_steps'],
        plain_dir=str(Path(args.plain_dir).resolve()),history_dir=str(Path(args.history_dir).resolve()),
        checkpoint_sha256={arm:sha256(Path(d)/'last.pth') for arm,d in zip(('plain','history'),(args.plain_dir,args.history_dir))},
        cohort_hashes=hashes,content_batch_trace_sha256=trace,within_seed_same_batch_order=True,
        plain_training_reused=True,plain_training_format=cps[0]['format'],history_training_format=cps[1]['format'],encoder_unchanged_both=True,
        independent_hard_ID_evaluation=True,max_independent_vs_saved_NLL_error=error,groups=groups,
        rq_mse={arm:r['final_train_catalog_metrics']['reconstruction_loss'] for arm,r in zip(('plain','history'),reports)},
        holdout_used_for_training=False,holdout_evaluated=not getattr(args,'train_only',False),ranking_evaluated=False,future_test_used=False,same_compute=False,
        evaluation_sources=dict(multiseed=sha256(Path(__file__)),auditor=sha256(Path(audit.__file__)),legacy=sha256(Path(legacy.__file__))),
        artifact_files_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'summary.json',report);return report


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--mode',choices=['plain','history-aware'],required=True)
    t.add_argument('--e1-dir',required=True);t.add_argument('--output',required=True);t.add_argument('--update-seed',type=int,required=True)
    t.add_argument('--device',default='cuda');t.add_argument('--max-steps',type=int,default=100)
    t.add_argument('--history-gradient-ratio',type=float,default=.1);t.add_argument('--anchor-batch-size',type=int,default=32)
    t.add_argument('--temperature',type=float,default=.1);t.add_argument('--log-every',type=int,default=10)
    t.add_argument('--save-every',type=int,default=10);t.add_argument('--history-progress-every',type=int,default=0);t.add_argument('--resume-from')
    e=sub.add_parser('evaluate');e.add_argument('--e1-dir',required=True);e.add_argument('--plain-dir',required=True)
    e.add_argument('--history-dir',required=True);e.add_argument('--output',required=True);e.add_argument('--update-seed',type=int,required=True)
    e.add_argument('--expected-steps',type=int,default=100);e.add_argument('--train-only',action='store_true')
    e.add_argument('--device',default='cuda');e.add_argument('--batch-size',type=int,default=32);e.add_argument('--progress',action='store_true')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    {'train':train,'evaluate':evaluate}[args.command](args)
