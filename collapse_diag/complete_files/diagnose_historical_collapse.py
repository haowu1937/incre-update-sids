"""Short matched traces: separate RQ/history gradients and hard SID collapse.

This is a diagnostic, not an alternative training method or paper result.
Existing bound source files and checkpoints are read, never modified.
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

import gensid.historical_incremental_content as history
from gensid.incremental_content import build_model, build_trainer, collision_stats, seed_all, sha256, write_json
from gensid.plain_incremental_content import evaluate_groups
from genrec.temporal_sids import verify_files


def group_parameters(model):
    return dict(encoder=list(model.encoder.parameters()), decoder=list(model.decoder.parameters()),
        **{'codebook_'+str(i+1): list(layer.parameters()) for i, layer in enumerate(model.rq.vq_layers)})


def gradient_contributions(model, rq_loss, penalty, weight):
    params = list(model.parameters())
    first = torch.autograd.grad(rq_loss, params, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(penalty, params, retain_graph=True, allow_unused=True)
    lookup = {id(p): i for i, p in enumerate(params)}
    groups = group_parameters(model)
    groups['all'] = params
    report = {}
    for name, group in groups.items():
        aa = bb = dot = 0.0
        for p in group:
            i = lookup[id(p)]
            a, b = first[i], second[i]
            if a is not None:
                aa += float(a.detach().double().square().sum())
            if b is not None:
                bb += float(b.detach().double().square().sum())
            if a is not None and b is not None:
                dot += float((a.detach().double()*b.detach().double()).sum())
        nr, nh = aa**0.5, bb**0.5
        report[name] = dict(rq_gradient_norm=nr, history_gradient_norm=nh,
            weighted_history_gradient_norm=weight*nh,
            weighted_history_to_rq_norm_ratio=weight*nh/nr if nr > 0 else None,
            rq_history_cosine=dot/(nr*nh) if nr > 0 and nh > 0 else None)
    return report


@torch.no_grad()
def observe(model, ex):
    codes = history.catalog_codes(model, ex)
    result = collision_stats(codes, ex.parent['config']['model']['num_emb_list'])
    entropies = []
    for i, cap in enumerate(ex.parent['config']['model']['num_emb_list']):
        count = np.bincount(codes[:, i], minlength=cap)
        probability = count[count > 0]/len(codes)
        entropies.append(float(-(probability*np.log(probability)).sum()))
    device = next(model.parameters()).device
    # Fixed content-only diagnostic sample; no held-out recommendation queries.
    values = []
    for start in range(0, min(len(ex.features), 4096), ex.old_batch):
        values.append(model.encoder(ex.features[start:min(start+ex.old_batch, len(ex.features), 4096)]
                                    .to(device)).cpu())
    encoded = torch.cat(values)
    result.update(hard_codebook_entropy_nats=entropies,
        latent_sample_items=len(encoded), latent_variance_mean=float(encoded.var(0, unbiased=False).mean()),
        latent_abs_mean=float(encoded.abs().mean()),
        all_items_same_semantic_code=result['unique_semantic_codes'] == 1,
        suffix_capacity=history.suffix_capacity_audit(ex.previous,
            [r['raw_item_id'] for r in ex.feature_rows], codes.tolist()))
    return result


@torch.no_grad()
def soft_health(model, features, hard_codes, scales, temperature):
    residual = model.encoder(features)
    result = []
    for i, layer in enumerate(model.rq.vq_layers):
        d = history.distance(residual, layer.embedding.weight)
        probability = torch.softmax(-d/(scales[i]*temperature), dim=1)
        entropy = -(probability*probability.clamp_min(1e-30).log()).sum(1)
        nearest = torch.topk(d, min(2, d.shape[1]), largest=False, dim=1).values
        result.append(dict(level=i+1, probability_entropy_mean=float(entropy.mean()),
            maximum_probability_mean=float(probability.max(1).values.mean()),
            normalized_nearest_margin_median=float(((nearest[:, -1]-nearest[:, 0])
                                                    /(scales[i]*temperature)).median()),
            hard_batch_vs_catalog_argmin_agreement=float((d.argmin(1) == hard_codes[:, i]).float().mean())))
        residual = residual-layer.embedding(hard_codes[:, i])
    return result


def update_norms(model, before):
    report = {}
    for name, params in group_parameters(model).items():
        numerator = denominator = maximum = 0.0
        for p in params:
            old = before[id(p)]
            delta = p.detach()-old
            numerator += float(delta.double().square().sum())
            denominator += float(old.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        report[name] = dict(update_l2=numerator**0.5,
            update_relative_l2=(numerator/denominator)**0.5 if denominator > 0 else None,
            maximum_absolute_update=maximum)
    return report


def run_lane(ex, device, steps, weight, anchor_batch, output, quiet=False):
    seed_all(ex.plain['config']['seed'])
    config = copy.deepcopy(ex.plain['config'])
    config.update(device=str(device), anchor_batch_size=anchor_batch,
        anchor_sampler_seed=(config['seed']+1) % 2**32, temperature=0.1,
        compatibility_weight=weight, history_input_condition='new_semantics_old_suffix')
    model = build_model(config).to(device)
    model.load_state_dict(ex.parent['state_dict'], strict=True)
    initial_metrics, _, _ = evaluate_groups(model, ex.features, ex.groups, ex.old_batch,
        ex.plain['config']['added_item_assignment_eval_batch_size'])
    scales = history.distance_scales(model, ex.features[:ex.groups['base_train'][1]], ex.old_batch)
    teacher = history.frozen_teacher(ex, device)
    penalty = history.HistoryPenalty(model, teacher, ex, config, scales)
    trainer = build_trainer(model, config, output)
    initial = observe(model, ex)
    last_observation = initial
    probes, records = {}, []
    original_loss = model.compute_loss
    original_surrogate = history.surrogate_embeddings
    def surrogate_probe(*a, **kw):
        result = original_surrogate(*a, **kw)
        probes['soft_assignment'] = soft_health(a[0], a[1], a[2], a[5], a[6])
        return result
    def loss_probe(*a, **kw):
        rq, aux, recon, quant = original_loss(*a, **kw)
        term = penalty()
        probes['gradients'] = gradient_contributions(model, rq, term, weight)
        probes['rq_loss'] = float(rq.detach())
        probes['history_penalty'] = float(term.detach())
        probes['before_parameters'] = {id(p): p.detach().clone() for p in model.parameters()}
        # Computing diagnostic history derivatives does not add them to the zero-weight optimizer.
        return (rq if weight == 0 else rq+weight*term), aux, recon, quant
    model.compute_loss = loss_probe
    history.surrogate_embeddings = surrogate_probe
    try:
        with (output/'trace.jsonl').open('w') as log:
            def on_step(state, metrics):
                nonlocal last_observation
                post = observe(model, ex)
                record = dict(step=state['step'], content_epoch=state['epoch'],
                    processed_content_items=state['processed_examples'], compatibility_weight=weight,
                    rq_loss=probes['rq_loss'], history_penalty=probes['history_penalty'],
                    weighted_history_penalty=weight*probes['history_penalty'],
                    pre_update=last_observation, post_update=post,
                    gradient_contributions=probes['gradients'], soft_assignment=probes['soft_assignment'],
                    parameter_updates=update_norms(model, probes.pop('before_parameters')),
                    anchor_queries_processed=penalty.state['processed_queries'])
                records.append(record)
                log.write(json.dumps(record, allow_nan=False)+'\n')
                log.flush()
                if not quiet:
                    print(json.dumps(dict(weight=weight, step=state['step'],
                        unique_codes=post['unique_semantic_codes'], max_bucket=post['maximum_collision_bucket'],
                        codebooks_used=[r['used'] for r in post['codebooks']],
                        history_penalty=record['history_penalty'],
                        encoder_weighted_history_to_rq_gradient_ratio=
                            probes['gradients']['encoder']['weighted_history_to_rq_norm_ratio'])), flush=True)
                last_observation = post
            sampler = trainer.fit_steps(ex.features, torch.arange(len(ex.features)), steps,
                config['batch_size'], config['seed'], None, on_step)
    finally:
        model.compute_loss = original_loss
        history.surrogate_embeddings = original_surrogate
    final_metrics, _, _ = evaluate_groups(model, ex.features, ex.groups, ex.old_batch,
        ex.plain['config']['added_item_assignment_eval_batch_size'])
    history.check_teacher(teacher, ex)
    compact = dict(compatibility_weight=weight, optimizer_steps=sampler['step'],
        content_items_processed=sampler['processed_examples'],
        initial=initial, final=last_observation, initial_rq_metrics=initial_metrics,
        final_rq_metrics=final_metrics, distance_scales=scales,
        first_complete_collapse_post_step=next((r['step'] for r in records
            if r['post_update']['all_items_same_semantic_code']), None),
        all_complete_collapse_post_steps=[r['step'] for r in records if r['post_update']['all_items_same_semantic_code']],
        first_capacity_overflow_post_step=next((r['step'] for r in records
            if not r['post_update']['suffix_capacity']['full_sid_allocation_feasible']), None),
        last_batch_gradient_contributions=records[-1]['gradient_contributions'],
        last_batch_soft_assignment=records[-1]['soft_assignment'],
        teacher_parameters_unchanged=True)
    write_json(output/'summary.json', compact)
    trainer.tb_writer.close()
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return compact, final_state


def failed_checkpoint_report(directory, ex, device):
    directory = Path(directory)
    summary = history.read_json(directory/'summary.json')
    if summary.get('stage') != 'historical_aware_update_content_rqvae' or summary['binding'] != ex.binding:
        raise ValueError('Failed checkpoint binding mismatch')
    verify_files(directory, summary, ['last.pth'])
    checkpoint = torch.load(directory/'last.pth', map_location='cpu', weights_only=False)
    if (checkpoint.get('format') != 'historical_update_rqvae_v1' or checkpoint['binding'] != ex.binding
            or checkpoint['config'] != summary['config']
            or checkpoint['sampler_state']['step'] != summary['update_optimizer_steps']):
        raise ValueError('Failed checkpoint config/step mismatch')
    model = build_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    metrics, groups, _ = evaluate_groups(model, ex.features, ex.groups, ex.old_batch,
        ex.plain['config']['added_item_assignment_eval_batch_size'])
    centered = ex.features-ex.features.mean(0, keepdim=True)
    mean_baseline = float(centered.double().square().mean())
    return dict(checkpoint_sha256=sha256(directory/'last.pth'),
        optimizer_steps=summary['update_optimizer_steps'], observation=observe(model, ex),
        full_update_catalog_metrics=metrics, group_metrics=groups, mean_embedding_baseline_mse=mean_baseline,
        reconstruction_improvement_over_mean_baseline_pct=100*(1-metrics['reconstruction_loss']/mean_baseline))


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite diagnostic output')
    if args.steps < 1 or args.anchor_batch_size < 1 or any(not np.isfinite(w) or w < 0 for w in args.weights):
        raise ValueError('Invalid steps, anchors or weights')
    if len(set(args.weights)) != len(args.weights):
        raise ValueError('Weights must be distinct')
    ex, device = history.experiment(args), history.device_from(args.device)
    if args.steps > ex.plain['update_optimizer_steps']:
        raise ValueError('Diagnostic steps exceed existing plain Update budget')
    failed = failed_checkpoint_report(args.failed_rqvae_dir, ex, device) if args.failed_rqvae_dir else None
    output.mkdir(parents=True)
    reports = {}
    for weight in args.weights:
        label = 'lambda_'+str(weight).replace('.', 'p')
        directory = output/label
        directory.mkdir()
        report, _ = run_lane(ex, device, args.steps, weight, args.anchor_batch_size, directory)
        reports[label] = report
    result = dict(stage='historical_update_collapse_diagnostic', execution_status='completed',
        e1_dir=str(Path(args.e1_dir).resolve()), source_binding=ex.binding,
        diagnostic_source_sha256=sha256(Path(__file__)), lanes=reports, failed_checkpoint=failed,
        objective_condition='new_semantics_old_suffix',
        gradient_scope='pre-optimizer batch gradients; separate raw RQ and raw history gradients',
        comparison='same Base weights, reset optimizer, content order, anchor order, learning rate and batch size',
        zero_weight_history='computed for observation; excluded exactly from optimizer loss',
        diagnostic_only=True, deployment_sid_export_performed=False,
        future_test_status='not_run', recommendation_training_status='not_run',
        device=str(device), runtime_seconds=time.perf_counter()-started)
    write_json(output/'summary.json', result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e1-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--anchor-batch-size', type=int, default=32)
    p.add_argument('--weights', type=float, nargs='+', default=[0.0, 0.001])
    p.add_argument('--failed-rqvae-dir', help='Optional: evaluate the collapsed completed checkpoint read-only')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
