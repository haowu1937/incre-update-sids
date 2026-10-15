"""History-aware joint RQ-VAE update: full anchors and normalized codebook gradients.

Starts from C_old, with the Plain Update content sampler/optimizer. RQ loss
updates all parameters; history gradients update only three codebook tables.
Complete new SIDs are used in the history forward. No holdout-based selection.
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
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import gensid.historical_incremental_content as h
from gensid.incremental_content import (build_model, build_trainer, seed_all, sha256,
    write_json, collision_stats, rng_state, restore_rng)
from gensid.plain_incremental_content import evaluate_groups, churn
from genrec.temporal_sids import read_json, read_jsonl, maps_from_ledger, verify_files
from genrec.historical_damage import versioned_suffixes, reencode_history, paired_losses, aggregate
from genrec.temporal_tiger import make_batch, evaluate as rank_evaluate
from genrec.evaluator import Evaluator

FORMAT = 'historical_joint_update_v2'


def binding(ex):
    return dict(experiment=ex.binding, joint_source_sha256=sha256(Path(__file__)))


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def catalog(model, ex):
    codes = h.catalog_codes(model, ex)
    raw = [r['raw_item_id'] for r in ex.feature_rows]
    capacity = h.suffix_capacity_audit(ex.previous, raw, codes.tolist())
    if not capacity['full_sid_allocation_feasible']:
        return codes, capacity, None, None
    ledger = versioned_suffixes(ex.previous, raw, codes.tolist())
    _, mapping = maps_from_ledger(ledger)
    return codes, capacity, ledger, mapping


def book_surrogate(model, features, codes, teacher, capacities, scales, temperature):
    """Hard semantic forward; historical encoder output and suffix are detached."""
    with torch.no_grad():
        residual = model.encoder(features)
    residual = residual.detach()
    table = teacher.t5.get_input_embeddings().weight.detach()
    result, offset = [], 1
    for i, layer in enumerate(model.rq.vq_layers):
        weights = layer.embedding.weight
        probability = F.softmax(-h.distance(residual, weights)/(scales[i]*temperature), dim=1)
        soft_token = probability@table[offset:offset+capacities[i]]
        hard_token = F.embedding(codes[:, i]+offset, table)
        result.append(hard_token+(soft_token-soft_token.detach()))
        soft_codeword = probability@weights
        hard_codeword = F.embedding(codes[:, i], weights)
        residual = residual-(hard_codeword.detach()+(soft_codeword-soft_codeword.detach()))
        offset += capacities[i]
    return torch.stack(result, dim=1)


class FullHistory:
    """Every accepted optimizer step uses ALL fixed anchor_train rows.

    Old batches/NLL are cached once. Current catalog assignments/allocation are
    shared across the batches in one step. Each graph is freed before the next.
    """
    def __init__(self, model, teacher, ex, config, scales, progress_every=32):
        self.model, self.teacher, self.ex, self.config, self.scales = model, teacher, ex, config, scales
        self.device = next(model.parameters()).device
        self.selected = np.asarray(ex.cohorts['anchor_train'], dtype=np.int64)
        self.books = [layer.embedding.weight for layer in model.rq.vq_layers]
        self.progress_every = progress_every
        self.packs = []
        raw_ids = {r['raw_item_id']: i for i, r in enumerate(ex.feature_rows)}
        size = config['historical_v2']['anchor_batch_size']
        print('Cache old anchor_train NLL | queries='+str(len(self.selected)), flush=True)
        started = time.perf_counter()
        for start in range(0, len(self.selected), size):
            selected = self.selected[start:start+size]
            old = make_batch(ex.arrays['base_train'], selected, self.device)
            with torch.no_grad():
                logits = teacher(dict(**old, use_cache=False)).logits.float()
                old_nll = F.cross_entropy(logits.transpose(1, 2), old['labels'], reduction='none').sum(1)
            globals_, positions = [], []
            for b, index in enumerate(selected):
                source = int(ex.arrays['base_train']['source_rows'][index])
                row = ex.rows['base_train'][source]
                for j, (raw, available) in enumerate(zip(row['item_seq'][:-1], row['history_available'])):
                    if available:
                        globals_.append(raw_ids[raw]); positions.append(b*old['input_ids'].shape[1]+j*4)
            unique, inverse = np.unique(globals_, return_inverse=True)
            self.packs.append(dict(indices=selected.copy(), old={k: v.cpu() for k, v in old.items()},
                old_nll=old_nll.cpu(), unique=unique, inverse=inverse,
                full_positions=(np.asarray(positions, dtype=np.int64)[:, None]+np.arange(4)).flatten(),
                semantic_positions=(np.asarray(positions, dtype=np.int64)[:, None]+np.arange(3)).flatten()))
            if progress_every and len(self.packs) % progress_every == 0:
                print('Cache anchors '+str(start+len(selected))+'/'+str(len(self.selected))+
                      ' | elapsed %.1fs' % (time.perf_counter()-started), flush=True)

    def old_nll(self):
        return torch.cat([p['old_nll'] for p in self.packs])

    def inputs(self, pack, codes, mapping, differentiable):
        old = {k: v.to(self.device) for k, v in pack['old'].items()}
        ids = old['input_ids'].clone()
        unique = pack['unique']
        if len(unique):
            tokens = torch.as_tensor([mapping[self.ex.feature_rows[i]['raw_item_id']] for i in unique],
                                     dtype=torch.long, device=self.device)
            inverse = torch.as_tensor(pack['inverse'], dtype=torch.long, device=self.device)
            ids.flatten().index_copy_(0, torch.as_tensor(pack['full_positions'], device=self.device),
                                      tokens[inverse].flatten())
        table = self.teacher.t5.get_input_embeddings().weight.detach()
        embeds = F.embedding(ids, table)
        if differentiable and len(unique):
            semantic = book_surrogate(self.model, self.ex.features[unique].to(self.device),
                torch.as_tensor(codes[unique], dtype=torch.long, device=self.device), self.teacher,
                self.ex.tokenizer.capacities[:3], self.scales, self.config['historical_v2']['temperature'])
            embeds = embeds.flatten(0, 1).index_copy(0,
                torch.as_tensor(pack['semantic_positions'], device=self.device), semantic[inverse].flatten(0, 1)).view_as(embeds)
        return old, ids, embeds

    def measure(self, codes, mapping, backward=False, step=None):
        if mapping is None:
            raise ValueError('History forward requires feasible complete SIDs')
        count = len(self.selected)
        grads = [torch.zeros_like(p) for p in self.books] if backward else None
        positive_sum = delta_sum = changed_count = 0.
        was_training = self.model.training
        self.model.eval()
        started = time.perf_counter()
        try:
            for i, pack in enumerate(self.packs):
                with torch.set_grad_enabled(backward):
                    old, ids, embeds = self.inputs(pack, codes, mapping, backward)
                    logits = self.teacher(dict(inputs_embeds=embeds, attention_mask=old['attention_mask'],
                        labels=old['labels'], use_cache=False)).logits.float()
                    new_nll = F.cross_entropy(logits.transpose(1, 2), old['labels'], reduction='none').sum(1)
                    changed = (ids != old['input_ids']).any(1)
                    # Hard identity inputs have exactly zero loss AND gradient contribution.
                    delta = torch.where(changed, new_nll-pack['old_nll'].to(self.device), torch.zeros_like(new_nll))
                    positive = F.relu(delta)
                    if backward and positive.requires_grad:
                        values = torch.autograd.grad(positive.sum()/count, self.books, allow_unused=True)
                        for accumulator, value in zip(grads, values):
                            if value is not None:
                                accumulator.add_(value.detach())
                if not torch.isfinite(delta).all():
                    raise FloatingPointError('Nonfinite historical prediction loss')
                positive_sum += float(positive.detach().sum())
                delta_sum += float(delta.detach().sum())
                changed_count += int(changed.sum())
                if backward and self.progress_every and (i+1) % self.progress_every == 0:
                    print('Update step '+str(step)+' | history batches '+str(i+1)+'/'+str(len(self.packs))+
                          ' | elapsed %.1fs' % (time.perf_counter()-started), flush=True)
        finally:
            self.model.train(was_training)
        return dict(queries=count, positive_damage=positive_sum/count, mean_delta_sequence_nll=delta_sum/count,
                    changed_history_fraction=changed_count/count), grads


def normalized_mix(rq, historical, rho, epsilon):
    """No higher-order gradient or global scalar-loss interpretation."""
    combined, statistics = [], []
    for a, b in zip(rq, historical):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError('Nonfinite codebook gradient')
        rn, hn = float(a.double().norm()), float(b.double().norm())
        alpha = rho*rn/(hn+epsilon) if rn > 0 and hn > 0 else 0.
        weighted = b*alpha
        wn = float(weighted.double().norm())
        combined.append(a+weighted)
        statistics.append(dict(rq_gradient_norm=rn, history_gradient_norm=hn, alpha=alpha,
            weighted_history_gradient_norm=wn, effective_gradient_ratio=wn/rn if rn > 0 else 0.,
            cosine=float((a.double()*b.double()).sum())/(rn*hn) if rn > 0 and hn > 0 else None))
    return combined, statistics


def metrics(model, ex):
    return evaluate_groups(model, ex.features, ex.groups, ex.old_batch,
                           ex.plain['config']['added_item_assignment_eval_batch_size'])


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
        raise FileExistsError('Refusing to overwrite joint update output: '+str(out))
    if (min(args.max_steps, args.anchor_batch_size, args.log_every, args.save_every) < 1
            or not np.isfinite(args.history_gradient_ratio) or not 0 <= args.history_gradient_ratio <= 1
            or not np.isfinite(args.temperature) or args.temperature <= 0 or args.history_progress_every < 0):
        raise ValueError('Invalid joint update settings')
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
        anchor_batch_size=args.anchor_batch_size, anchor_coverage='all anchor_train every optimizer step',
        temperature=args.temperature, history_input_condition='new_full_sid',
        suffix_gradient='none; deterministic complete SID allocation in forward',
        capacity_policy='rollback invalid candidate and stop at last feasible checkpoint')
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
    if not all(layer.initted for layer in model.rq.vq_layers):
        raise ValueError('Uninitialized Base codebooks')
    warm_equal = all(torch.equal(v.cpu(), ex.parent['state_dict'][k]) for k, v in model.state_dict().items())
    initial, initial_groups, old_codes = metrics(model, ex)
    if not np.array_equal(old_codes[:ex.groups['base_train'][1]], ex.old_export):
        raise ValueError('Base assignments changed before joint Update')
    scales = h.distance_scales(model, ex.features[:ex.groups['base_train'][1]], ex.old_batch)
    state = dict(step=0, cursor=0, order=None, epoch=0, processed_examples=0)
    work = dict(history_gradient_queries=0, old_reference_queries=0, full_catalog_assignment_passes=0,
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
    print('History-aware joint Update v2 | steps '+str(first_step)+'->'+str(args.max_steps)+
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
            optimizer.zero_grad(set_to_none=True)
            out_values, quant_loss, _, dense, out_books, residuals = model(data, None, use_sk=False)
            rq_loss, aux, recon, quant = model.compute_loss(out_values, quant_loss, chosen.to(device), dense,
                dense_out_codebooks=out_books, x_res_list=residuals, xs=data)
            if not torch.isfinite(rq_loss):
                raise FloatingPointError('Nonfinite RQ loss')
            rq_loss.backward()
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
            row = dict(step=state['step'], epoch=state['epoch'], processed_examples=state['processed_examples'],
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
        plain_train_catalog_metrics=ex.plain['final_train_catalog_metrics'],
        full_anchor_train_at_run_start=initial_history, final_full_anchor_train=final_history,
        anchor_train_queries=len(history.selected), anchor_holdout_used_for_training=False,
        base_validation_used_for_training=False, trained_components=['encoder','decoder','three_codebooks'],
        historical_gradient_components=['three_codebook_tables'], historical_encoder_output_detached=True,
        historical_gradient_estimator='hard semantic forward; soft-distance STE backward; fixed Base distance scales',
        history_input_condition='new_full_sid', collision_suffix_gradient='none', old_target_labels_fixed=True,
        checkpoint_selection='last at fixed optimizer-step budget; no holdout or validation selection',
        collision_statistics=collision_stats(final_codes, config['model']['num_emb_list']),
        base_semantic_churn=churn(ex.old_export, final_codes[:ex.groups['base_train'][1]]),
        full_sid_export_status='completed', saved_checkpoint_deployment_sid_ready=True,
        teacher_frozen=True, teacher_parameters_unchanged=True, parent_checkpoint_unchanged=True,
        checkpoint_reload_assignments_equal=True, work_counters=work,
        same_optimizer_step_budget_as_plain=state['step'] == ex.plain['update_optimizer_steps'],
        plain_optimizer_step_budget=ex.plain['update_optimizer_steps'],
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


def paired_contrast(new, plain, condition, repetitions, seed):
    if not new or len(new) != len(plain):
        raise ValueError('Paired query count mismatch')
    clusters, deltas = {}, []
    for a, b in zip(new, plain):
        if any(a[k] != b[k] for k in ('sample_id','user','target_labels','source_row')):
            raise ValueError('Paired query/old target mismatch')
        if abs(a['old_sequence_nll']-b['old_sequence_nll']) > 1e-3:
            raise ValueError('Old teacher NLL differs from E1')
        da = a[condition+'_sequence_nll']-a['old_sequence_nll']
        db = b[condition+'_sequence_nll']-b['old_sequence_nll']
        value = [a[condition+'_sequence_nll']-b[condition+'_sequence_nll'], max(da,0)-max(db,0)]
        deltas.append(value)
        entry = clusters.setdefault(a['user'], [0.,0.,0.])
        entry[0] += value[0]; entry[1] += value[1]; entry[2] += 1
    values, rng, draws = np.asarray(list(clusters.values())), np.random.default_rng(seed), []
    for _ in range(repetitions):
        total = values[rng.integers(0,len(values),len(values))].sum(0)
        draws.append(total[:2]/total[2])
    ci = np.quantile(draws,[.025,.975],axis=0) if len(values)>1 else None
    return dict(queries=len(new), users=len(values),
        mean_history_aware_minus_plain_sequence_nll=float(np.mean(deltas,axis=0)[0]),
        mean_history_aware_minus_plain_positive_damage=float(np.mean(deltas,axis=0)[1]),
        sequence_nll_cluster_ci95=ci[:,0].tolist() if ci is not None else None,
        positive_damage_cluster_ci95=ci[:,1].tolist() if ci is not None else None,
        bootstrap_unit='user; row-weighted paired means', bootstrap_replicates=repetitions)


def evaluate(args):
    started = time.perf_counter()
    out, directory = Path(args.output).resolve(), Path(args.rqvae_dir).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite joint evaluation output')
    if args.batch_size < 1:
        raise ValueError('Invalid evaluation batch size')
    ex, device = h.experiment(args), h.device_from(args.device)
    seed_all(ex.e1['cohort_seed'])
    summary = read_json(directory/'summary.json')
    if (summary.get('stage') != FORMAT or summary.get('execution_status') != 'completed'
            or summary['binding'] != binding(ex)):
        raise ValueError('Completed joint checkpoint data/source binding mismatch')
    if summary['update_optimizer_steps'] != ex.plain['update_optimizer_steps']:
        raise ValueError('Formal comparison requires the same 1000-step (bound Plain) budget')
    verify_files(directory, summary, ['last.pth','config.json','semantic_assignments.jsonl',
        'new_allocation_registry.json','new_item2tokens.json','anchor_train_tensor_rows.json'])
    cp = torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
    if (cp.get('format') != FORMAT or cp['binding'] != summary['binding'] or cp['config'] != summary['config']
            or cp['sampler_state']['step'] != summary['update_optimizer_steps']):
        raise ValueError('Joint checkpoint/config/step mismatch')
    if read_json(directory/'anchor_train_tensor_rows.json') != ex.cohorts['anchor_train']:
        raise ValueError('Joint anchor cohort mismatch')
    rq = build_model(cp['config']).to(device); rq.load_state_dict(cp['state_dict'],strict=True)
    codes, capacity, ledger, mapping = catalog(rq,ex)
    if mapping is None:
        raise ValueError('Loaded joint full SID capacity exhausted')
    if read_jsonl(directory/'semantic_assignments.jsonl') != [dict(**r,semantic_code=c) for r,c in zip(ex.feature_rows,codes.tolist())]:
        raise ValueError('Loaded joint codes differ from exported assignments')
    if ledger != read_json(directory/'new_allocation_registry.json') or mapping != read_json(directory/'new_item2tokens.json'):
        raise ValueError('Loaded joint full SID allocation mismatch')
    control = {raw:[*mapping[raw][:3],tokens[3]] for raw,tokens in ex.tokenizer.item2tokens.items()}
    variants = {split:dict(old=arrays,new_full_sid=reencode_history(arrays,ex.rows[split],mapping),
        new_semantics_old_suffix=reencode_history(arrays,ex.rows[split],control)) for split,arrays in ex.arrays.items()}
    teacher = h.frozen_teacher(ex,device)
    out.mkdir(parents=True)
    groups, contrasts = {}, {}
    tasks = [(name,'base_train',chosen) for name,chosen in ex.cohorts.items()]
    tasks.append(('base_validation','base_val',list(range(len(ex.arrays['base_val']['labels'])))))
    for name,split,chosen in tasks:
        records = paired_losses(teacher,variants[split],ex.rows[split],chosen,args.batch_size,device,name)
        with (out/(name+'_paired_losses.jsonl')).open('w') as f:
            for row in records:
                f.write(json.dumps(row,allow_nan=False)+'\n')
        plain = read_jsonl(ex.e1_dir/(name+'_paired_losses.jsonl'))
        groups[name],contrasts[name] = {},{}
        for condition in ('new_full_sid','new_semantics_old_suffix'):
            repetitions = ex.e1['groups'][name][condition]['cluster_bootstrap_replicates']
            groups[name][condition] = aggregate(records,condition,repetitions,ex.e1['cohort_seed'])
            contrasts[name][condition] = paired_contrast(records,plain,condition,repetitions,ex.e1['cohort_seed'])
        print(name+': '+json.dumps(contrasts[name]),flush=True)
    ranking = None
    if args.ranking:
        if ex.e1.get('base_validation_ranking') is None:
            raise ValueError('Bound Plain full ranking missing')
        wrapper = SimpleNamespace(model=teacher,config=ex.teacher['config'],evaluator=Evaluator(ex.teacher['config'],ex.tokenizer))
        new_rank = rank_evaluate(wrapper,ex.tokenizer,variants['base_val']['new_full_sid'],ex.audits['base_val'],
            list(range(len(ex.audits['base_val']))),device,ex.teacher['config']['eval_batch_size'])
        reference = ex.e1['base_validation_ranking']
        ranking = dict(old=reference['old'],plain=reference['new_full_sid'],history_aware=new_rank,
            history_aware_minus_plain_full_denominator={k:new_rank['full_denominator'][k]-v
                for k,v in reference['new_full_sid']['full_denominator'].items()},
            reference_rankings_reused_from_hash_bound_e1=True)
        write_json(out/'base_validation_ranking.json',ranking)
    h.check_teacher(teacher,ex)
    result = dict(stage='historical_joint_update_v2_evaluation',execution_status='completed',binding=summary['binding'],
        e1_dir=str(ex.e1_dir.resolve()),rqvae_dir=str(directory),rqvae_checkpoint_sha256=sha256(directory/'last.pth'),
        selected_tiger_sha256=ex.e1['selected_tiger_sha256'],plain_checkpoint_sha256=ex.e1['update_rqvae_sha256'],
        update_optimizer_steps=summary['update_optimizer_steps'],config=summary['config'],
        groups=groups,paired_history_aware_vs_plain=contrasts,base_validation_ranking=ranking,
        rqvae_reconstruction=dict(plain=ex.plain['final_train_catalog_metrics'],history_aware=summary['final_train_catalog_metrics']),
        churn=dict(plain=ex.plain['base_semantic_churn'],history_aware=summary['base_semantic_churn']),
        primary_cohort='anchor_holdout; unused by historical training; seen by original Base TIGER',
        history_training_condition='new_full_sid',evaluation_labels='original full SID plus EOS',
        ranking_catalog='original base_train; unavailable targets are misses in full denominator',
        same_optimizer_step_budget_as_plain=True,same_compute_as_plain=False,teacher_frozen=True,
        recommendation_training_status='not_run',future_test_status='not_run',device=str(device),
        runtime_seconds=time.perf_counter()-started,
        artifact_files_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'summary.json',result)
    print('Joint Update v2 evaluation completed. Use the bundle show_v2.py for compact results.',flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    t = sub.add_parser('train')
    t.add_argument('--e1-dir',required=True); t.add_argument('--output',required=True)
    t.add_argument('--device',default='cuda'); t.add_argument('--max-steps',type=int,default=1000)
    t.add_argument('--history-gradient-ratio',type=float,default=.1)
    t.add_argument('--anchor-batch-size',type=int,default=32); t.add_argument('--temperature',type=float,default=.1)
    t.add_argument('--log-every',type=int,default=25); t.add_argument('--save-every',type=int,default=10)
    t.add_argument('--history-progress-every',type=int,default=32); t.add_argument('--resume-from')
    e = sub.add_parser('evaluate')
    e.add_argument('--e1-dir',required=True); e.add_argument('--rqvae-dir',required=True); e.add_argument('--output',required=True)
    e.add_argument('--device',default='cuda'); e.add_argument('--batch-size',type=int,default=32)
    e.add_argument('--ranking',action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    train(args) if args.command == 'train' else evaluate(args)
