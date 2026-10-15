"""Matched frozen-encoder RQ-VAE Update pilot; training anchors only.

Reuse the immutable v2 full-SID forward and book-only historical gradient.
Start from the exact Base weights; freeze the content RQ-VAE encoder in both
arms. Plain optimizes RQ loss; history-aware also mixes normalized H gradients.
No holdout, Base validation prediction, or ranking evaluation is performed.
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
from torch.nn import functional as F
import history_aware_update_v2 as v2
from history_aware_update_v2 import (h, snapshot, catalog, FullHistory,
    normalized_mix, metrics, build_model, build_trainer, seed_all, sha256,
    write_json, collision_stats, rng_state, restore_rng, churn, read_json,
    read_jsonl, verify_files)

FORMAT = 'historical_frozen_encoder_update_v1'


def binding(ex):
    return dict(experiment=ex.binding, frozen_source_sha256=sha256(Path(__file__)),
        v2_source_sha256=sha256(Path(v2.__file__)))


def hard_records(history, codes, mapping):
    """Per-query endpoint H on the exact training cohort and cached old reference."""
    result = []
    with torch.no_grad():
        for pack in history.packs:
            old, ids, embeds = history.inputs(pack, codes, mapping, False)
            logits = history.teacher(dict(inputs_embeds=embeds, attention_mask=old['attention_mask'],
                labels=old['labels'], use_cache=False)).logits.float()
            nll = F.cross_entropy(logits.transpose(1,2), old['labels'], reduction='none').sum(1)
            for j, index in enumerate(pack['indices']):
                source = int(history.ex.arrays['base_train']['source_rows'][index])
                raw = history.ex.rows['base_train'][source]
                before = float(pack['old_nll'][j])
                changed = bool((ids[j] != old['input_ids'][j]).any())
                after = float(nll[j]) if changed else before
                result.append(dict(sample_id=raw['sample_id'], user=raw['user'], source_row=source,
                    tensor_row=int(index), target_labels=old['labels'][j].tolist(),
                    old_sequence_nll=before, new_sequence_nll=after,
                    positive_damage=max(after-before,0.), history_changed=changed))
    return result


def save_checkpoint(path, model, optimizer, config, bound, sampler, work, scales, initial, initial_groups, history):
    temp = Path(str(path)+'.tmp')
    torch.save(dict(format=FORMAT, config=config, binding=bound, state_dict=snapshot(model),
        optimizer=optimizer.state_dict(), sampler_state=copy.deepcopy(sampler), rng_state=rng_state(),
        work_counters=copy.deepcopy(work), distance_scales=scales, initial_metrics=initial,
        initial_group_metrics=initial_groups, old_anchor_nll=history.old_nll()), temp)
    os.replace(temp, path)


def train(args):
    started = time.perf_counter()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite frozen-encoder update output: '+str(out))
    if (min(args.max_steps, args.anchor_batch_size, args.log_every, args.save_every) < 1
            or not np.isfinite(args.history_gradient_ratio) or not 0 <= args.history_gradient_ratio <= 1
            or not np.isfinite(args.temperature) or args.temperature <= 0 or args.history_progress_every < 0):
        raise ValueError('Invalid joint update settings')
    args = copy.copy(args)
    args.history_gradient_ratio = 0. if args.mode == 'plain' else args.history_gradient_ratio
    ex, device = h.experiment(args), h.device_from(args.device)
    if args.max_steps > ex.plain['update_optimizer_steps']:
        raise ValueError('Joint target step budget exceeds bound Plain Update budget')
    if ex.parent['config']['model']['bn'] or ex.parent['config']['model']['dropout_prob'] != 0:
        raise ValueError('This comparison requires the bound deterministic content MLP')
    config = copy.deepcopy(ex.plain['config'])
    config['device'] = str(device)
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
    if resume and resume['sampler_state']['step'] >= args.max_steps:
        raise ValueError('Target budget must exceed resumed checkpoint steps')
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
    state = dict(step=0, cursor=0, order=None, epoch=0, processed_examples=0)
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
    print('Frozen-encoder '+args.mode+' | steps '+str(first_step)+'->'+str(args.max_steps)+
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
            row = dict(history_after_optimizer_step=after_history, step=state['step'], epoch=state['epoch'], processed_examples=state['processed_examples'],
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


def compare(args):
    """Compare ONLY the matched frozen-encoder training endpoints."""
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite comparison output')
    directories = [Path(args.plain_dir).resolve(), Path(args.history_dir).resolve()]
    reports, checkpoints, records = [], [], []
    for directory, mode in zip(directories, ('plain','history-aware')):
        report = read_json(directory/'summary.json')
        if (report.get('stage') != FORMAT or report.get('execution_status') != 'completed'
                or report.get('mode') != mode or not report.get('encoder_unchanged_from_base')):
            raise ValueError('Expected completed frozen-encoder '+mode+' endpoint')
        verify_files(directory, report, ['last.pth','anchor_train_hard_losses.jsonl','config.json',
            'semantic_assignments.jsonl','new_allocation_registry.json','new_item2tokens.json'])
        cp = torch.load(directory/'last.pth', map_location='cpu', weights_only=False)
        if (cp.get('format') != FORMAT or cp['binding'] != report['binding']
                or cp['config'] != report['config'] or cp['sampler_state']['step'] != report['update_optimizer_steps']):
            raise ValueError('Checkpoint/report mismatch')
        if report['binding']['frozen_source_sha256'] != sha256(Path(__file__)) or report['binding']['v2_source_sha256'] != sha256(Path(v2.__file__)):
            raise ValueError('Frozen experiment source changed')
        reports.append(report); checkpoints.append(cp)
        records.append(read_jsonl(directory/'anchor_train_hard_losses.jsonl'))
    a,b = reports; ca,cb = checkpoints
    if a['binding'] != b['binding'] or a['update_optimizer_steps'] != b['update_optimizer_steps']:
        raise ValueError('The two arms must have the same source/data binding and step budget')
    if a['config']['historical_v2']['history_gradient_ratio'] != 0 or b['config']['historical_v2']['history_gradient_ratio'] <= 0:
        raise ValueError('Expected Plain rho=0 and History-aware rho>0')
    configs = [copy.deepcopy(r['config']) for r in reports]
    for c in configs:
        c.pop('frozen_encoder')
        c['historical_v2'].pop('history_gradient_ratio')
    if configs[0] != configs[1]:
        raise ValueError('Content optimizer, sampler, or historical forward settings differ')
    for key in ('step','cursor','epoch','processed_examples'):
        if ca['sampler_state'][key] != cb['sampler_state'][key]:
            raise ValueError('Matched sampler endpoint differs: '+key)
    for key in ('order','generator_state'):
        if not torch.equal(ca['sampler_state'][key],cb['sampler_state'][key]):
            raise ValueError('Matched sampler order/generator differs')
    if not torch.equal(ca['old_anchor_nll'], cb['old_anchor_nll']):
        raise ValueError('The old teacher reference differs between arms')
    if any(not torch.equal(value,cb['state_dict'][key]) for key,value in ca['state_dict'].items() if key.startswith('encoder.')):
        raise ValueError('Frozen encoder differs between arms')
    paired, clusters = [], {}
    if not records[0] or len(records[0]) != len(records[1]):
        raise ValueError('Matched training cohort counts differ')
    for x,y in zip(*records):
        if any(x[k] != y[k] for k in ('sample_id','user','source_row','tensor_row','target_labels','old_sequence_nll')):
            raise ValueError('Matched training queries/old labels/reference differ')
        values = [y['positive_damage']-x['positive_damage'], y['new_sequence_nll']-x['new_sequence_nll']]
        paired.append(values)
        group = clusters.setdefault(x['user'], np.zeros(3))
        group[:2] += values; group[2] += 1
    means = np.mean(paired,axis=0)
    rng = np.random.default_rng(a['config']['seed']); groups = np.array(list(clusters.values()))
    draws = []
    for _ in range(500):
        total = groups[rng.integers(0,len(groups),len(groups))].sum(0)
        draws.append(total[:2]/total[2])
    ci = np.quantile(draws,[.025,.975],axis=0).T.tolist() if len(groups)>1 else [None,None]
    for r,rows in zip(reports,records):
        if abs(np.mean([x['positive_damage'] for x in rows])-r['final_full_anchor_train']['positive_damage']) > 1e-5:
            raise ValueError('Saved endpoint H differs from query records')
    result = dict(stage='frozen_encoder_matched_training_pilot',execution_status='completed',
        update_optimizer_steps=a['update_optimizer_steps'], rho=b['config']['historical_v2']['history_gradient_ratio'],
        queries=len(paired),users=len(clusters), binding=a['binding'],
        plain_dir=str(directories[0]), history_dir=str(directories[1]),
        endpoint_checkpoint_sha256={mode:sha256(d/'last.pth') for mode,d in zip(('plain','history-aware'),directories)},
        same_starting_base_checkpoint=True,same_content_optimizer_and_sampler=True,same_step_budget=True,
        encoder_frozen_and_unchanged_both=True,same_compute=False,
        plain=dict(history=a['final_full_anchor_train'],rq_mse=a['final_train_catalog_metrics']['reconstruction_loss'],
            collision_statistics=a['collision_statistics'],base_semantic_churn=a['base_semantic_churn']),
        history_aware=dict(history=b['final_full_anchor_train'],rq_mse=b['final_train_catalog_metrics']['reconstruction_loss'],
            collision_statistics=b['collision_statistics'],base_semantic_churn=b['base_semantic_churn']),
        history_aware_minus_plain=dict(positive_damage=float(means[0]),mean_sequence_nll=float(means[1]),
            rq_mse=b['final_train_catalog_metrics']['reconstruction_loss']-a['final_train_catalog_metrics']['reconstruction_loss'],
            positive_damage_cluster_ci95=ci[0],sequence_nll_cluster_ci95=ci[1]),
        bootstrap_replicates=500,bootstrap_unit='user; row-weighted paired means; training cohort only',
        interpretation='training diagnostic; training-anchor CIs do not establish held-out improvement',
        holdout_used=False,base_validation_prediction_evaluated=False,ranking_evaluated=False,
        future_test_used=False)
    out.mkdir(parents=True)
    write_json(out/'summary.json',result)
    print(json.dumps(dict(execution_status='completed',update_optimizer_steps=result['update_optimizer_steps'],
        history_aware_minus_plain=result['history_aware_minus_plain']),indent=2),flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--mode',choices=['plain','history-aware'],required=True)
    t.add_argument('--e1-dir',required=True); t.add_argument('--output',required=True)
    t.add_argument('--device',default='cuda'); t.add_argument('--max-steps',type=int,default=100)
    t.add_argument('--history-gradient-ratio',type=float,default=.1)
    t.add_argument('--anchor-batch-size',type=int,default=32); t.add_argument('--temperature',type=float,default=.1)
    t.add_argument('--log-every',type=int,default=10); t.add_argument('--save-every',type=int,default=10)
    t.add_argument('--history-progress-every',type=int,default=0); t.add_argument('--resume-from')
    c = sub.add_parser('compare')
    c.add_argument('--plain-dir',required=True);c.add_argument('--history-dir',required=True);c.add_argument('--output',required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    train(args) if args.command == 'train' else compare(args)
