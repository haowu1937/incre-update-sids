"""Local pure-history direction check, with hard codes and an STE tangent.

No optimizer or full training run. Perturbations are made in memory, restored
after every candidate, and never written as checkpoints. Training anchors only.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import gensid.historical_incremental_content as history
from gensid.incremental_content import build_model, seed_all, sha256, write_json, collision_stats
from gensid.plain_incremental_content import evaluate_groups
from genrec.temporal_tiger import make_batch
from genrec.temporal_sids import code_to_tokens, verify_files


@torch.no_grad()
def constants(model, features, codes, teacher, capacities, scales, temperature):
    residual = model.encoder(features)
    table = teacher.t5.get_input_embeddings().weight.detach()
    result, offset = [], 1
    for i, layer in enumerate(model.rq.vq_layers):
        probability = F.softmax(-history.distance(residual, layer.embedding.weight)/(scales[i]*temperature), dim=1)
        soft_codeword = probability@layer.embedding.weight
        hard_codeword = layer.embedding(codes[:, i])
        soft_token = probability@table[offset:offset+capacities[i]]
        hard_token = F.embedding(codes[:, i]+offset, table)
        result.append(dict(hard_codeword=hard_codeword.cpu(), soft_codeword=soft_codeword.cpu(),
                           hard_token=hard_token.cpu(), soft_token=soft_token.cpu()))
        residual = residual-(hard_codeword.detach()+(soft_codeword-soft_codeword.detach()))
        offset += capacities[i]
    return result


def tangent_embeddings(model, features, stored, teacher, capacities, scales, temperature):
    """Continuous local extension: base forward and derivative equal the STE.

    Hard offsets are frozen at the base point, while soft codeword and token
    functions vary continuously. This is a local numerical check, not a new
    deployable SID encoder or the fully soft model's global objective.
    """
    residual = model.encoder(features)
    table = teacher.t5.get_input_embeddings().weight.detach()
    result, offset = [], 1
    for i, layer in enumerate(model.rq.vq_layers):
        reference = {k: v.to(features.device) for k, v in stored[i].items()}
        probability = F.softmax(-history.distance(residual, layer.embedding.weight)/(scales[i]*temperature), dim=1)
        soft_token = probability@table[offset:offset+capacities[i]]
        soft_codeword = probability@layer.embedding.weight
        result.append(reference['hard_token']+(soft_token-reference['soft_token']))
        residual = residual-(reference['hard_codeword']+(soft_codeword-reference['soft_codeword']))
        offset += capacities[i]
    return torch.stack(result, dim=1)


@torch.no_grad()
def contexts(model, teacher, ex, selected, batch_size, scales, device):
    codes = history.catalog_codes(model, ex)
    raw_ids = {r['raw_item_id']: i for i, r in enumerate(ex.feature_rows)}
    table = teacher.t5.get_input_embeddings().weight.detach()
    result = []
    for start in range(0, len(selected), batch_size):
        indices = selected[start:start+batch_size]
        batch = make_batch(ex.arrays['base_train'], indices, device)
        old_nll = F.cross_entropy(teacher(dict(**batch, use_cache=False)).logits.float().transpose(1, 2),
                                  batch['labels'], reduction='none').sum(1)
        hard_ids = batch['input_ids'].clone()
        positions, globals_ = [], []
        for b, index in enumerate(indices):
            source = int(ex.arrays['base_train']['source_rows'][index])
            raw = ex.rows['base_train'][source]
            for j, (item, available) in enumerate(zip(raw['item_seq'][:-1], raw['history_available'])):
                if available:
                    global_id = raw_ids[item]
                    hard_ids[b, j*4:j*4+3] = torch.tensor(
                        code_to_tokens(codes[global_id].tolist(), ex.tokenizer.capacities[:3]), device=device)
                    globals_.append(global_id)
                    positions.append(b*hard_ids.shape[1]+j*4)
        unique, inverse = np.unique(globals_, return_inverse=True)
        unique_codes = torch.as_tensor(codes[unique], dtype=torch.long, device=device)
        features = ex.features[unique].to(device)
        result.append(dict(indices=np.asarray(indices),
            old_batch={k: v.cpu() for k, v in batch.items()}, old_nll=old_nll.cpu(), hard_ids=hard_ids.cpu(),
            base_embed=F.embedding(hard_ids, table).cpu(), unique=unique, inverse=inverse,
            flat_positions=(np.asarray(positions)[:, None]+np.arange(3)).flatten(),
            unique_codes=unique_codes.cpu(),
            stored=constants(model, features, unique_codes, teacher, ex.tokenizer.capacities[:3], scales, 0.1)))
    return result, codes


def embed_batch(model, teacher, ex, ctx, scales, device, mode):
    batch = {k: v.to(device) for k, v in ctx['old_batch'].items()}
    features = ex.features[ctx['unique']].to(device)
    if mode == 'ste':
        values = history.surrogate_embeddings(model, features, ctx['unique_codes'].to(device), teacher,
                                               ex.tokenizer.capacities[:3], scales, 0.1)
    elif mode == 'tangent':
        values = tangent_embeddings(model, features, ctx['stored'], teacher,
                                    ex.tokenizer.capacities[:3], scales, 0.1)
    else:
        raise ValueError('Unknown differentiable embedding condition')
    embeds = ctx['base_embed'].to(device)
    values = values[torch.as_tensor(ctx['inverse'], dtype=torch.long, device=device)].flatten(0, 1)
    embeds = embeds.flatten(0, 1).index_copy(0,
        torch.as_tensor(ctx['flat_positions'], dtype=torch.long, device=device), values).view_as(embeds)
    return dict(inputs_embeds=embeds, attention_mask=batch['attention_mask'], labels=batch['labels'], use_cache=False)


def objective(model, teacher, ex, packs, scales, device, mode, backward=False):
    total, delta_total, count = 0.0, 0.0, sum(len(ctx['indices']) for ctx in packs)
    if backward:
        model.zero_grad(set_to_none=True)
    for ctx in packs:
        outputs = teacher(embed_batch(model, teacher, ex, ctx, scales, device, mode))
        nll = F.cross_entropy(outputs.logits.float().transpose(1, 2),
                              ctx['old_batch']['labels'].to(device), reduction='none').sum(1)
        delta = nll-ctx['old_nll'].to(device)
        positive = F.relu(delta)
        if backward:
            (positive.sum()/count).backward()
        total += float(positive.detach().sum())
        delta_total += float(delta.detach().sum())
    grads = {k: p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p, device='cpu')
             for k, p in model.named_parameters()} if backward else None
    return dict(positive_damage=total/count, mean_delta_sequence_nll=delta_total/count, queries=count), grads


@torch.no_grad()
def hard_objective(model, teacher, ex, packs, base_codes, device):
    codes = history.catalog_codes(model, ex)
    table = teacher.t5.get_input_embeddings().weight.detach()
    total = delta_sum = changed = queries = 0
    offset = np.cumsum([1]+ex.tokenizer.capacities[:2])
    for ctx in packs:
        ids = ctx['hard_ids'].to(device).clone()
        semantic_tokens = codes[ctx['unique']]+offset[None, :]
        values = torch.as_tensor(semantic_tokens[ctx['inverse']].flatten(), dtype=torch.long, device=device)
        ids.flatten().index_copy_(0, torch.as_tensor(ctx['flat_positions'], dtype=torch.long, device=device), values)
        batch = ctx['old_batch']
        logits = teacher(dict(input_ids=ids, attention_mask=batch['attention_mask'].to(device),
                              labels=batch['labels'].to(device), use_cache=False)).logits.float()
        nll = F.cross_entropy(logits.transpose(1, 2), batch['labels'].to(device), reduction='none').sum(1)
        delta = nll-ctx['old_nll'].to(device)
        # Match training's exact-zero convention for identical original history inputs.
        same_as_old = (ids == batch['input_ids'].to(device)).all(1)
        delta = torch.where(same_as_old, torch.zeros_like(delta), delta)
        total += float(F.relu(delta).sum())
        delta_sum += float(delta.sum())
        changed += int((ids != ctx['hard_ids'].to(device)).any(1).sum())
        queries += len(ids)
    stats = collision_stats(codes, ex.parent['config']['model']['num_emb_list'])
    return dict(positive_damage=total/queries, mean_delta_sequence_nll=delta_sum/queries,
        queries=queries, changed_history_fraction_from_probe_base=changed/queries,
        changed_catalog_assignment_fraction_from_probe_base=float((codes != base_codes).any(1).mean()),
        unique_semantic_codes=stats['unique_semantic_codes'], maximum_collision_bucket=stats['maximum_collision_bucket'],
        codebooks_used=[r['used'] for r in stats['codebooks']])


def gradient_match(first, second):
    aa = bb = dd = dot = 0.0
    for key, a in first.items():
        a, b = a.double(), second[key].double()
        aa += float(a.square().sum()); bb += float(b.square().sum())
        dd += float((a-b).square().sum()); dot += float((a*b).sum())
    return dict(ste_gradient_norm=aa**0.5, tangent_gradient_norm=bb**0.5,
        relative_gradient_difference=(dd/aa)**0.5 if aa else (0.0 if dd == 0 else None),
        gradient_cosine=dot/(aa*bb)**0.5 if aa and bb else None)


@torch.no_grad()
def perturb(model, base_state, grads, scope, relative_step, sign):
    model.load_state_dict(base_state, strict=True)
    names = [k for k, _ in model.named_parameters() if
        (scope == 'joint' and (k.startswith('encoder.') or k.startswith('rq.')))
        or (scope == 'encoder' and k.startswith('encoder.'))
        or (scope == 'codebooks' and k.startswith('rq.'))]
    theta_norm = sum(float(base_state[k].double().square().sum()) for k in names)**0.5
    grad_norm = sum(float(grads[k].double().square().sum()) for k in names)**0.5
    if grad_norm == 0:
        return None
    alpha = relative_step*theta_norm/grad_norm
    for k, p in model.named_parameters():
        if k in names:
            p.add_(grads[k].to(p.device), alpha=sign*alpha)
    return dict(scope=scope, sign=sign, relative_step=relative_step,
        parameter_norm=theta_norm, gradient_norm=grad_norm, alpha=alpha,
        predicted_tangent_change=sign*alpha*grad_norm**2)


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite direction diagnostic')
    if (min(args.queries, args.batch_size) < 1 or any(not np.isfinite(x) or x <= 0 for x in args.relative_steps)):
        raise ValueError('Invalid query count, batch or positive relative steps')
    ex, device = history.experiment(args), history.device_from(args.device)
    seed_all(ex.e1['cohort_seed'])
    point_dir = Path(args.rqvae_dir or ex.e1['update_rqvae_dir'])
    summary = history.read_json(point_dir/'summary.json')
    verify_files(point_dir, summary, ['last.pth'])
    checkpoint = torch.load(point_dir/'last.pth', map_location='cpu', weights_only=False)
    if summary.get('stage') == 'E1_plain_update_content_rqvae_preparation':
        if point_dir.resolve() != Path(ex.e1['update_rqvae_dir']).resolve() or checkpoint['binding'] != ex.plain['binding']:
            raise ValueError('Expected the E1-bound plain checkpoint')
    elif summary.get('stage') == 'historical_aware_update_content_rqvae':
        if checkpoint['binding'] != ex.binding:
            raise ValueError('Historical probe checkpoint binding mismatch')
    else:
        raise ValueError('Unsupported probe checkpoint stage')
    if checkpoint['config'] != summary['config']:
        raise ValueError('Probe checkpoint configuration mismatch')
    base = build_model(ex.parent['config']).to(device)
    base.load_state_dict(ex.parent['state_dict'], strict=True)
    scales = history.distance_scales(base, ex.features[:ex.groups['base_train'][1]], ex.old_batch)
    del base
    model = build_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    teacher = history.frozen_teacher(ex, device)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    anchors = np.asarray(ex.cohorts['anchor_train'])
    chosen = np.sort(np.random.default_rng(ex.e1['cohort_seed']).choice(anchors,
                     min(args.queries, len(anchors)), replace=False))
    packs, base_codes = contexts(model, teacher, ex, chosen, args.batch_size, scales, device)
    ste_reference, ste_grad = objective(model, teacher, ex, packs, scales, device, 'ste', backward=True)
    tangent_reference, tangent_grad = objective(model, teacher, ex, packs, scales, device, 'tangent', backward=True)
    match = gradient_match(ste_grad, tangent_grad)
    hard_reference = hard_objective(model, teacher, ex, packs, base_codes, device)
    if (match['relative_gradient_difference'] is None or match['relative_gradient_difference'] > 1e-3
            or abs(ste_reference['positive_damage']-hard_reference['positive_damage']) > 1e-4):
        raise ValueError('Local tangent failed to reproduce existing STE or its hard forward')
    output.mkdir(parents=True)
    write_json(output/'selected_anchor_train_rows.json', chosen.tolist())
    candidates = []
    try:
        for scope in args.scopes:
            for step in args.relative_steps:
                for sign in (-1, 1):
                    change = perturb(model, base_state, ste_grad, scope, step, sign)
                    if change is None:
                        continue
                    with torch.no_grad():
                        tangent_result, _ = objective(model, teacher, ex, packs, scales, device, 'tangent')
                        hard_result = hard_objective(model, teacher, ex, packs, base_codes, device)
                        rq_metrics, _, _ = evaluate_groups(model, ex.features, ex.groups,
                            ex.old_batch, ex.plain['config']['added_item_assignment_eval_batch_size'])
                    record = dict(**change, hard=hard_result, tangent=tangent_result, rq_metrics=rq_metrics,
                        hard_positive_damage_change=hard_result['positive_damage']-hard_reference['positive_damage'],
                        tangent_positive_damage_change=tangent_result['positive_damage']-tangent_reference['positive_damage'])
                    candidates.append(record)
                    print(json.dumps(dict(scope=scope, relative_step=step, sign=sign,
                        hard_damage_change=record['hard_positive_damage_change'],
                        tangent_damage_change=record['tangent_positive_damage_change'],
                        changed_history_fraction=hard_result['changed_history_fraction_from_probe_base'])), flush=True)
    finally:
        model.load_state_dict(base_state, strict=True)
        model.zero_grad(set_to_none=True)
    if any(not torch.equal(v.cpu(), base_state[k]) for k, v in model.state_dict().items()):
        raise ValueError('Probe parameters were not restored')
    history.check_teacher(teacher, ex)
    if sha256(point_dir/'last.pth') != summary['artifact_files_sha256']['last.pth']:
        raise ValueError('Source checkpoint changed')
    result = dict(stage='historical_surrogate_local_direction_diagnostic', execution_status='completed',
        source_binding=ex.binding, diagnostic_source_sha256=sha256(Path(__file__)),
        probe_checkpoint_dir=str(point_dir.resolve()), probe_checkpoint_sha256=sha256(point_dir/'last.pth'),
        probe_queries=len(chosen), probe_cohort='anchor_train only; deterministic subsample',
        command=vars(args), distance_scales=scales,
        condition='new_semantics_old_suffix; original targets/masks',
        gradient_match=match, hard_reference=hard_reference, tangent_reference=tangent_reference,
        zero_surrogate_gradient=match['ste_gradient_norm'] == 0,
        candidates=candidates, source_parameters_restored=True, teacher_unchanged=True,
        optimizer_steps=0, checkpoints_written=False, holdout_used=False,
        scope='local subset diagnostic; not proof of full-training or holdout improvement',
        device=str(device), runtime_seconds=time.perf_counter()-started)
    write_json(output/'summary.json', result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e1-dir', required=True)
    p.add_argument('--rqvae-dir', help='Default: existing E1-bound plain Update checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--queries', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--relative-steps', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3])
    p.add_argument('--scopes', choices=['joint','encoder','codebooks'], nargs='+', default=['joint','encoder','codebooks'])
    return p


if __name__ == '__main__':
    run(parser().parse_args())
