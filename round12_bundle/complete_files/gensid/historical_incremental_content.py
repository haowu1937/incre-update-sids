"""Historical-compatible content update, using frozen TIGER and hard forward SIDs.

Only the three semantic assignments receive a soft-distance surrogate gradient.
Collision suffix allocation is recomputed over the entire eligible catalog at
every step and is discrete. The teacher's forward input uses exact hard full
SIDs, including suffixes. No held-out history enters the training objective.
"""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from gensid.incremental_content import (build_model, build_trainer, collision_stats,
    restore_rng, rng_state, seed_all, sha256, write_json)
from gensid.plain_incremental_content import load_inputs, evaluate_groups, churn
from genrec.historical_damage import (aggregate, historical_cohorts, load_bound_inputs,
    paired_losses, reencode_history, versioned_suffixes)
from genrec.models.TIGER.model import TIGER
from genrec.temporal_sids import read_json, read_jsonl, maps_from_ledger, verify_files
from genrec.temporal_tiger import make_batch, evaluate as rank_evaluate
from genrec.evaluator import Evaluator

SOURCES = ['gensid/historical_incremental_content.py', 'train_historical_incremental_rqvae.py',
           'evaluate_historical_update.py', 'genrec/historical_damage.py', 'measure_historical_damage.py']


def experiment(args):
    """Validate the completed E1 measurement and recover its exact cohorts."""
    e1_dir = Path(args.e1_dir)
    e1 = read_json(e1_dir/'summary.json')
    if e1.get('stage') != 'E1_historical_input_reencoding_damage' or e1.get('E1_status') != 'completed':
        raise ValueError('Expected completed E1 historical measurement')
    required = ['anchor_train.jsonl', 'anchor_holdout.jsonl',
                'anchor_train_paired_losses.jsonl', 'anchor_holdout_paired_losses.jsonl',
                'base_validation_paired_losses.jsonl']
    verify_files(e1_dir, e1, required)
    root = Path(__file__).resolve().parents[1]
    for name, value in e1['source_files_sha256'].items():
        if sha256(root/name) != value:
            raise ValueError('E1 source mismatch: '+name)
    for name, path in [('tiger', Path(e1['tiger_dir'])), ('update_rqvae', Path(e1['update_rqvae_dir'])),
                       ('sid', Path(e1['sid_dir'])), ('protocol', Path(e1['protocol_dir']))]:
        if sha256(path/'summary.json') != e1['input_summary_sha256'][name]:
            raise ValueError('E1 input summary mismatch: '+name)
    old = load_bound_inputs(SimpleNamespace(protocol_dir=e1['protocol_dir'], tiger_dir=e1['tiger_dir'],
        update_rqvae_dir=e1['update_rqvae_dir'], sid_dir=e1['sid_dir'],
        expected_tiger_sha256=e1['selected_tiger_sha256']))
    tiger_summary, teacher, sid_dir, tokenizer, arrays, audits, rows, _, _ = old
    cohorts = historical_cohorts(rows['base_train'], e1['requested_queries_per_anchor_cohort'], e1['cohort_seed'])
    for key, indices in cohorts.items():
        saved = read_jsonl(e1_dir/(key+'.jsonl'))
        expected = [dict(source_row=i, **rows['base_train'][i]) for i in indices]
        if saved != expected:
            raise ValueError('E1 historical cohort mismatch: '+key)
    plain = read_json(Path(e1['update_rqvae_dir'])/'summary.json')
    features, feature_rows, parent, parent_report, binding, groups, old_export, old_batch = load_inputs(
        e1['protocol_dir'], plain['features_dir'], plain['base_rqvae_dir'], e1['base_rqvae_sha256'])
    if sha256(Path(e1['update_rqvae_dir'])/'last.pth') != e1['update_rqvae_sha256']:
        raise ValueError('E1 plain Update checkpoint mismatch')
    binding.update(e1_summary_sha256=sha256(e1_dir/'summary.json'),
        selected_tiger_sha256=e1['selected_tiger_sha256'],
        plain_update_checkpoint_sha256=e1['update_rqvae_sha256'],
        anchor_train_sha256=sha256(e1_dir/'anchor_train.jsonl'),
        source_files_sha256={name: sha256(root/name) for name in SOURCES},
        package_versions={name: importlib.metadata.version(name)
                          for name in ('torch', 'numpy', 'transformers', 'scikit-learn')},
        python=platform.python_version())
    return SimpleNamespace(e1=e1, e1_dir=e1_dir, plain=plain, tiger_summary=tiger_summary,
        teacher=teacher, sid_dir=sid_dir, tokenizer=tokenizer, arrays=arrays, audits=audits,
        rows=rows, cohorts=cohorts, features=features, feature_rows=feature_rows,
        parent=parent, parent_report=parent_report, binding=binding, groups=groups,
        old_export=old_export, old_batch=old_batch,
        previous=read_json(sid_dir/'base/allocation_registry.json'))


def frozen_teacher(ex, device):
    model = TIGER(ex.teacher['config'], None, ex.tokenizer).to(device)
    model.load_state_dict(ex.teacher['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model


def check_teacher(model, ex):
    if (any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(not torch.equal(v.cpu(), ex.teacher['state_dict'][k])
                   for k, v in model.state_dict().items())):
        raise ValueError('Frozen teacher changed or accumulated parameter gradients')


@torch.no_grad()
def catalog_codes(model, ex):
    """Preserve original assignment batching; avoid the training loss hook."""
    device = next(model.parameters()).device
    result = []
    was_training = model.training
    model.eval()
    for name, (start, end) in ex.groups.items():
        batch = ex.old_batch if name == 'base_train' else ex.plain['config']['added_item_assignment_eval_batch_size']
        for left in range(start, end, batch):
            result.append(model.get_indices(ex.features[left:min(left+batch, end)].to(device),
                                            None, use_sk=False).cpu().numpy())
    model.train(was_training)
    return np.concatenate(result)


def distance(residual, weight):
    # Match the repository quantizer's squared-distance expression.
    return residual.square().sum(1, keepdim=True)+weight.square().sum(1).unsqueeze(0)-2*residual@weight.t()


@torch.no_grad()
def distance_scales(model, features, batch_size):
    """Fix per-level scales on Base content before any Update optimizer step."""
    values = [[] for _ in model.rq.vq_layers]
    device = next(model.parameters()).device
    for start in range(0, min(len(features), 4096), batch_size):
        residual = model.encoder(features[start:min(start+batch_size, len(features), 4096)].to(device))
        for i, layer in enumerate(model.rq.vq_layers):
            d = distance(residual, layer.embedding.weight)
            values[i].append(d.std(1, unbiased=False).cpu())
            # Reproduce the repository hard residual computation for calibration.
            quantized = layer.embedding(d.argmin(1))
            residual = residual-(residual+(quantized-residual).detach())
    return [max(float(torch.cat(x).median()), 1e-8) for x in values]


def surrogate_embeddings(model, features, hard_codes, teacher, capacities, scales, temperature):
    """Exact hard token embeddings in forward; soft assignment derivative only.

    Residual forward values follow selected hard codewords. Their backward path
    uses the probability-weighted codeword vector, allowing all encoder and
    codebook levels to receive compatibility gradients. This is a surrogate,
    not the derivative of integer argmin or of collision suffix allocation.
    """
    residual = model.encoder(features)
    embeddings, offset = [], 1
    table = teacher.t5.get_input_embeddings().weight.detach()
    for i, layer in enumerate(model.rq.vq_layers):
        weights = layer.embedding.weight
        d = distance(residual, weights)
        probability = F.softmax(-d/(scales[i]*temperature), dim=-1)
        soft_token = probability@table[offset:offset+capacities[i]]
        hard_token = F.embedding(hard_codes[:, i]+offset, table)
        embeddings.append(hard_token+(soft_token-soft_token.detach()))
        soft_codeword = probability@weights
        hard_codeword = F.embedding(hard_codes[:, i], weights)
        residual = residual-(hard_codeword.detach()+(soft_codeword-soft_codeword.detach()))
        offset += capacities[i]
    return torch.stack(embeddings, dim=1)


class HistoryPenalty:
    def __init__(self, model, teacher, ex, config, scales, state=None):
        self.model, self.teacher, self.ex, self.config = model, teacher, ex, config
        self.scales = scales
        self.device = next(model.parameters()).device
        self.selected = np.asarray(ex.cohorts['anchor_train'], dtype=np.int64)
        self.generator = torch.Generator(device='cpu').manual_seed(config['anchor_sampler_seed'])
        self.state = copy.deepcopy(state) if state else dict(cursor=0, order=None, epoch=0, processed_queries=0)
        if state:
            self.generator.set_state(state['generator_state'])
        self.old_nll = torch.empty(len(self.selected), dtype=torch.float32)
        with torch.no_grad():
            for start in tqdm(range(0, len(self.selected), config['anchor_batch_size']),
                              desc='Cache old anchor_train NLL', unit='batch'):
                ids = self.selected[start:start+config['anchor_batch_size']]
                batch = make_batch(ex.arrays['base_train'], ids, self.device)
                logits = teacher(batch).logits.float()
                self.old_nll[start:start+len(ids)] = F.cross_entropy(
                    logits.transpose(1, 2), batch['labels'], reduction='none').sum(1).cpu()
        self.last_metrics = {}

    def snapshot(self):
        result = copy.deepcopy(self.state)
        result['generator_state'] = self.generator.get_state()
        return result

    def draw(self):
        n = len(self.selected)
        if self.state['order'] is None or self.state['cursor'] == n:
            self.state['order'] = torch.randperm(n, generator=self.generator)
            self.state['cursor'] = 0
            self.state['epoch'] += 1
        end = min(n, self.state['cursor']+self.config['anchor_batch_size'])
        selected = self.state['order'][self.state['cursor']:end]
        self.state['cursor'] = end
        self.state['processed_queries'] += len(selected)
        return selected.numpy()

    def input_embeds(self, local_indices):
        codes = catalog_codes(self.model, self.ex)
        ledger = versioned_suffixes(self.ex.previous,
            [r['raw_item_id'] for r in self.ex.feature_rows], codes.tolist())
        _, mapping = maps_from_ledger(ledger)
        tensor_indices = self.selected[local_indices]
        old = make_batch(self.ex.arrays['base_train'], tensor_indices, self.device)
        hard_ids = old['input_ids'].clone()
        positions, global_ids, history_tokens = [], [], []
        by_raw = {r['raw_item_id']: i for i, r in enumerate(self.ex.feature_rows)}
        for b, index in enumerate(tensor_indices):
            source = int(self.ex.arrays['base_train']['source_rows'][index])
            raw = self.ex.rows['base_train'][source]
            for j, (item, available) in enumerate(zip(raw['item_seq'][:-1], raw['history_available'])):
                if available:
                    global_ids.append(by_raw[item])
                    positions.append((b, j*4))
                    history_tokens.append(mapping[item])
        if positions:
            position_tensor = torch.as_tensor(positions, dtype=torch.long, device=self.device)
            hard_ids[position_tensor[:, :1], position_tensor[:, 1:]+torch.arange(4, device=self.device)] = \
                torch.as_tensor(history_tokens, dtype=torch.long, device=self.device)
        table = self.teacher.t5.get_input_embeddings().weight.detach()
        embeds = F.embedding(hard_ids, table)
        if global_ids:
            unique, inverse = np.unique(global_ids, return_inverse=True)
            values = surrogate_embeddings(self.model, self.ex.features[unique].to(self.device),
                torch.as_tensor(codes[unique], dtype=torch.long, device=self.device), self.teacher,
                self.ex.tokenizer.capacities[:3], self.scales, self.config['temperature'])
            # One index_copy avoids a large per-history CopySlices autograd graph.
            flat_positions = (position_tensor[:, :1]*hard_ids.shape[1]+position_tensor[:, 1:]
                              +torch.arange(3, device=self.device)).flatten()
            selected_values = values[torch.as_tensor(inverse, dtype=torch.long, device=self.device)].flatten(0, 1)
            embeds = embeds.flatten(0, 1).index_copy(0, flat_positions, selected_values).view_as(embeds)
        return old, hard_ids, embeds

    def __call__(self):
        local_indices = self.draw()
        old, hard_ids, embeds = self.input_embeds(local_indices)
        output = self.teacher(dict(inputs_embeds=embeds, attention_mask=old['attention_mask'],
                                   labels=old['labels'], use_cache=False))
        new_nll = F.cross_entropy(output.logits.float().transpose(1, 2), old['labels'], reduction='none').sum(1)
        old_nll = self.old_nll[local_indices].to(self.device)
        changed = (hard_ids != old['input_ids']).any(1)
        # Identical inputs have exactly zero Delta, removing reference-batching roundoff.
        delta = torch.where(changed, new_nll-old_nll, new_nll-new_nll.detach())
        penalty = F.relu(delta).mean()
        self.last_metrics = dict(history_penalty=float(penalty.detach()),
            anchor_mean_delta_sequence_nll=float(delta.detach().mean()),
            anchor_changed_history_fraction=float(changed.float().mean()),
            anchor_queries=len(local_indices), anchor_sampler_epoch=self.state['epoch'])
        return penalty


def compatible_fit(trainer, features, max_steps, config, state, penalty, on_step):
    """Use the existing repository Trainer.fit_steps and optimizer unchanged."""
    if config['compatibility_weight'] == 0:
        return trainer.fit_steps(features, torch.arange(len(features)), max_steps,
            config['batch_size'], config['seed'], state, on_step)
    original = trainer.model.compute_loss
    def combined(*args, **kwargs):
        loss, aux, recon, quant = original(*args, **kwargs)
        historical = penalty()
        return loss+config['compatibility_weight']*historical, aux, recon, quant
    trainer.model.compute_loss = combined
    try:
        return trainer.fit_steps(features, torch.arange(len(features)), max_steps,
            config['batch_size'], config['seed'], state, on_step)
    finally:
        trainer.model.compute_loss = original


def save(path, trainer, config, binding, sampler, penalty, scales, initial, initial_groups):
    pending = Path(str(path)+'.tmp')
    torch.save(dict(format='historical_update_rqvae_v1', config=config, binding=binding,
        state_dict=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
        sampler_state=sampler, anchor_sampler_state=penalty.snapshot() if penalty else None,
        distance_scales=scales, rng_state=rng_state(), initial_metrics=initial,
        initial_group_metrics=initial_groups), pending)
    os.replace(pending, path)


def device_from(value):
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda') or device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CPU/CUDA device unavailable')
    return device


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite historical Update output: '+str(output))
    if (min(args.max_steps, args.anchor_batch_size, args.log_every, args.save_every) < 1
            or not np.isfinite(args.compatibility_weight) or args.compatibility_weight < 0
            or not np.isfinite(args.temperature) or args.temperature <= 0):
        raise ValueError('Invalid step/batch/compatibility weight/temperature')
    ex, device = experiment(args), device_from(args.device)
    base_config = ex.plain['config']
    if args.max_steps > ex.plain['update_optimizer_steps']:
        raise ValueError('Historical Update budget must not exceed the plain comparison budget')
    config = copy.deepcopy(base_config)
    config.update(device=str(device), compatibility_weight=args.compatibility_weight,
        temperature=args.temperature, anchor_batch_size=args.anchor_batch_size,
        anchor_sampler_seed=(base_config['seed']+1) % 2**32,
        objective='repository_RQ_loss + lambda * mean(relu(new_oldlabel_sequence_NLL - old_NLL))',
        surrogate='hard full SID forward; soft distance gradient for three semantic tokens',
        suffix_gradient='none; exact discrete version allocation recomputed each step')
    resume = torch.load(args.resume_from, map_location='cpu', weights_only=False) if args.resume_from else None
    if resume and (resume.get('format') != 'historical_update_rqvae_v1'
                   or resume['config'] != config or resume['binding'] != ex.binding):
        raise ValueError('Historical Update resume data/source/config binding mismatch')
    if resume and resume['sampler_state']['step'] >= args.max_steps:
        raise ValueError('Target update steps must exceed resumed steps')
    seed_all(config['seed'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config).to(device)
    model.load_state_dict(ex.parent['state_dict'], strict=True)
    initial, initial_groups, old_codes = evaluate_groups(model, ex.features, ex.groups,
        ex.old_batch, base_config['added_item_assignment_eval_batch_size'])
    n_base = ex.groups['base_train'][1]
    if not np.array_equal(old_codes[:n_base], ex.old_export):
        raise ValueError('Base assignment mismatch before historical Update')
    scales = distance_scales(model, ex.features[:n_base], ex.old_batch)
    teacher = frozen_teacher(ex, device)
    output.mkdir(parents=True)
    trainer = build_trainer(model, config, output)
    state = None
    if resume:
        model.load_state_dict(resume['state_dict'], strict=True)
        trainer.optimizer.load_state_dict(resume['optimizer'])
        state = resume['sampler_state']
        initial, initial_groups = resume['initial_metrics'], resume['initial_group_metrics']
        if scales != resume['distance_scales']:
            raise ValueError('Base temperature calibration differs on resume')
    penalty = HistoryPenalty(model, teacher, ex, config, scales,
        resume['anchor_sampler_state'] if resume else None) if args.compatibility_weight > 0 else None
    if resume:
        restore_rng(resume['rng_state'])
    write_json(output/'config.json', dict(config=config, binding=ex.binding, command=vars(args),
                                         distance_scales=scales))
    first_step = state['step'] if state else 0
    print(f'Historical Update ready | {len(ex.features)} content items | '
          f'{len(ex.cohorts["anchor_train"])} training anchors | frozen teacher | '
          f'lambda={args.compatibility_weight} | update steps {first_step}->{args.max_steps}', flush=True)
    with (output/'metrics.jsonl').open('w') as log:
        def on_step(current, metrics):
            metrics.update(penalty.last_metrics if penalty else dict(history_penalty=0.0, anchor_queries=0))
            metrics['weighted_history_penalty'] = config['compatibility_weight']*metrics['history_penalty']
            metrics['rq_loss'] = metrics['total_loss']-metrics['weighted_history_penalty']
            metrics['gradient_norms'] = {name: float(torch.stack([p.grad.detach().square().sum()
                for p in module.parameters() if p.grad is not None]).sum().sqrt())
                for name, module in [('encoder', model.encoder), ('decoder', model.decoder), ('codebooks', model.rq)]}
            log.write(json.dumps(metrics, allow_nan=False)+'\n')
            log.flush()
            if current['step'] == first_step+1 or current['step'] % args.log_every == 0 or current['step'] == args.max_steps:
                print(json.dumps(metrics, allow_nan=False), flush=True)
            if current['step'] % args.save_every == 0:
                save(output/'last.pth', trainer, config, ex.binding, current, penalty, scales, initial, initial_groups)
        state = compatible_fit(trainer, ex.features, args.max_steps, config, state, penalty, on_step)
    save(output/'last.pth', trainer, config, ex.binding, state, penalty, scales, initial, initial_groups)
    final, final_groups, new_codes = evaluate_groups(model, ex.features, ex.groups,
        ex.old_batch, base_config['added_item_assignment_eval_batch_size'])
    restored = build_model(config).to(device)
    restored.load_state_dict(torch.load(output/'last.pth', map_location='cpu', weights_only=False)['state_dict'])
    if not np.array_equal(new_codes, catalog_codes(restored, ex)):
        raise ValueError('Historical Update assignment reload mismatch')
    for filename, codes in [('old_semantic_assignments.jsonl', old_codes), ('semantic_assignments.jsonl', new_codes)]:
        with (output/filename).open('w') as handle:
            for row, code in zip(ex.feature_rows, codes.tolist()):
                handle.write(json.dumps(dict(**row, semantic_code=code))+'\n')
    ledger = versioned_suffixes(ex.previous, [r['raw_item_id'] for r in ex.feature_rows], new_codes.tolist())
    write_json(output/'new_allocation_registry.json', ledger)
    _, mapping = maps_from_ledger(ledger)
    write_json(output/'new_item2tokens.json', mapping)
    comparison = churn(old_codes[:n_base], new_codes[:n_base])
    write_json(output/'base_semantic_churn.json', comparison)
    check_teacher(teacher, ex)
    if sha256(Path(ex.plain['base_rqvae_dir'])/'last.pth') != ex.binding['parent_base_checkpoint_sha256']:
        raise ValueError('Base RQ-VAE changed')
    report = dict(stage='historical_aware_update_content_rqvae', execution_status='completed',
        e1_dir=str(ex.e1_dir.resolve()), protocol_sha256=ex.binding['protocol_sha256'],
        features_dir=ex.plain['features_dir'], base_rqvae_dir=ex.plain['base_rqvae_dir'],
        output=str(output), config=config, binding=ex.binding,
        training_catalog='update_train', training_items=len(ex.features), feature_dim=ex.features.shape[1],
        update_optimizer_steps=state['step'], steps_this_run=state['step']-first_step,
        processed_examples=state['processed_examples'], sampler_epochs_started=state['epoch'],
        resume_from=str(Path(args.resume_from).resolve()) if args.resume_from else None,
        checkpoint_selection='last_at_fixed_update_optimizer_step_budget; no heldout selection',
        initial_train_catalog_metrics=initial, final_train_catalog_metrics=final,
        initial_group_metrics=initial_groups, final_group_metrics=final_groups,
        base_semantic_churn=comparison, collision_statistics=collision_stats(new_codes, config['model']['num_emb_list']),
        distance_scales=scales, calibration='fixed Base content distance spread; first up to 4096 registry items',
        penalty_training_cohort='anchor_train only', anchor_train_queries=len(ex.cohorts['anchor_train']),
        anchor_holdout_used_for_training=False, base_validation_used_for_training=False,
        anchor_queries_processed=penalty.state['processed_queries'] if penalty else 0,
        teacher_frozen=True, teacher_parameters_unchanged=True, old_target_labels_fixed=True,
        hard_full_sid_forward=True, semantic_gradient_estimator=config['surrogate'],
        collision_suffix_gradient='none; allocator is discrete; full SID forward still uses its exact output',
        historical_prediction_penalty_used=args.compatibility_weight > 0,
        parent_base_checkpoint_unchanged=True, checkpoint_reload_assignments_equal=True,
        same_parent_optimizer_content_sampler_as_plain=True, recommendation_training_status='not_run',
        historical_damage_evaluation_status='not_run', future_test_status='not_run',
        E0_full_status='not_run', E1_status='completed_in_bound_input',
        graph_used=False, sinkhorn_used=False, content_encoder_finetuning=False,
        codebook_reinitialization=False, trained_components=['encoder', 'decoder', 'three_codebooks'],
        python=platform.python_version(), package_versions=ex.binding['package_versions'], device=str(device),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        runtime_seconds=time.perf_counter()-started,
        artifact_files_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'summary.json', report)
    trainer.tb_writer.close()
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e1-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--compatibility-weight', type=float, default=0.001)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--anchor-batch-size', type=int, default=32)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--resume-from')
    return p


def evaluation_parser():
    p = argparse.ArgumentParser(description='Compare historical Update to the bound plain E1 measurement')
    p.add_argument('--e1-dir', required=True)
    p.add_argument('--historical-rqvae-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--ranking', action='store_true')
    return p


def evaluate_update(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite historical evaluation output: '+str(output))
    if args.batch_size < 1:
        raise ValueError('Invalid evaluation batch size')
    ex, device = experiment(args), device_from(args.device)
    directory = Path(args.historical_rqvae_dir)
    summary = read_json(directory/'summary.json')
    if (summary.get('stage') != 'historical_aware_update_content_rqvae'
            or summary.get('execution_status') != 'completed' or summary['binding'] != ex.binding):
        raise ValueError('Historical checkpoint E1/source binding mismatch')
    verify_files(directory, summary, ['last.pth', 'semantic_assignments.jsonl', 'config.json',
                                      'new_allocation_registry.json', 'new_item2tokens.json'])
    checkpoint = torch.load(directory/'last.pth', map_location='cpu', weights_only=False)
    if (checkpoint.get('format') != 'historical_update_rqvae_v1' or checkpoint['binding'] != ex.binding
            or checkpoint['config'] != summary['config']
            or checkpoint['sampler_state']['step'] != summary['update_optimizer_steps']):
        raise ValueError('Historical checkpoint config/step mismatch')
    if summary['update_optimizer_steps'] != ex.plain['update_optimizer_steps']:
        raise ValueError('Evaluation requires the same Update step budget as plain baseline')
    seed_all(ex.e1['cohort_seed'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rq = build_model(checkpoint['config']).to(device)
    rq.load_state_dict(checkpoint['state_dict'], strict=True)
    codes = catalog_codes(rq, ex)
    exported = read_jsonl(directory/'semantic_assignments.jsonl')
    if exported != [dict(**row, semantic_code=code) for row, code in zip(ex.feature_rows, codes.tolist())]:
        raise ValueError('Historical exported codes differ from loaded checkpoint')
    ledger = versioned_suffixes(ex.previous, [r['raw_item_id'] for r in ex.feature_rows], codes.tolist())
    _, mapping = maps_from_ledger(ledger)
    if ledger != read_json(directory/'new_allocation_registry.json') or mapping != read_json(directory/'new_item2tokens.json'):
        raise ValueError('Historical full SID allocation mismatch')
    control = {raw: [*mapping[raw][:3], tokens[3]] for raw, tokens in ex.tokenizer.item2tokens.items()}
    variants = {split: dict(old=arrays,
        new_full_sid=reencode_history(arrays, ex.rows[split], mapping),
        new_semantics_old_suffix=reencode_history(arrays, ex.rows[split], control))
        for split, arrays in ex.arrays.items()}
    model = frozen_teacher(ex, device)
    output.mkdir(parents=True)
    groups, paired_comparison = {}, {}
    tasks = [(name, 'base_train', selected) for name, selected in ex.cohorts.items()]
    tasks.append(('base_validation', 'base_val', list(range(len(ex.arrays['base_val']['labels'])))))
    for name, split, selected in tasks:
        records = paired_losses(model, variants[split], ex.rows[split], selected, args.batch_size, device, name)
        with (output/(name+'_paired_losses.jsonl')).open('w') as handle:
            for row in records:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
        groups[name] = {key: aggregate(records, key, ex.e1['groups'][name][key]['cluster_bootstrap_replicates'],
                                      ex.e1['cohort_seed']) for key in ('new_full_sid', 'new_semantics_old_suffix')}
        plain_records = read_jsonl(ex.e1_dir/(name+'_paired_losses.jsonl'))
        paired = []
        if len(records) != len(plain_records):
            raise ValueError('Plain/compatible paired query count mismatch')
        for new, old in zip(records, plain_records):
            if any(new[k] != old[k] for k in ('sample_id', 'user', 'target_labels', 'source_row')):
                raise ValueError('Plain/compatible query or old-label mismatch')
            if abs(new['old_sequence_nll']-old['old_sequence_nll']) > 1e-3:
                raise ValueError('Frozen old NLL differs from E1 reference')
            paired.append(dict(user=new['user'], old_sequence_nll=old['new_full_sid_sequence_nll'],
                old_position_nll=old['new_full_sid_position_nll'],
                new_full_sid_sequence_nll=new['new_full_sid_sequence_nll'],
                new_full_sid_position_nll=new['new_full_sid_position_nll'],
                new_full_sid_delta_sequence_nll=new['new_full_sid_sequence_nll']-old['new_full_sid_sequence_nll'],
                new_full_sid_kl_sum=0.0, new_full_sid_history_changed=True))
        contrast = aggregate(paired, 'new_full_sid', ex.e1['groups'][name]['new_full_sid']['cluster_bootstrap_replicates'],
                             ex.e1['cohort_seed'])
        paired_comparison[name] = dict(queries=contrast['queries'], users=contrast['users'],
            mean_compatible_minus_plain_sequence_nll=contrast['mean_delta_sequence_nll'],
            cluster_bootstrap_ci95=contrast['cluster_bootstrap_mean_delta_ci95'],
            compatible_better_fraction=contrast['improved_fraction'],
            plain_mean_positive_damage=ex.e1['groups'][name]['new_full_sid']['mean_positive_part_delta'],
            compatible_mean_positive_damage=groups[name]['new_full_sid']['mean_positive_part_delta'])
        print(name+': '+json.dumps(paired_comparison[name]), flush=True)
    ranking = None
    if args.ranking:
        wrapper = SimpleNamespace(model=model, config=ex.teacher['config'],
                                  evaluator=Evaluator(ex.teacher['config'], ex.tokenizer))
        ranking = {key: rank_evaluate(wrapper, ex.tokenizer, variants['base_val'][key], ex.audits['base_val'],
            list(range(len(ex.audits['base_val']))), device, ex.teacher['config']['eval_batch_size'])
            for key in ('old', 'new_full_sid')}
        for denominator in ('full_denominator', 'conditional_on_servable'):
            if any(abs(ranking['old'][denominator][k]-v) > 1e-8
                   for k, v in ex.tiger_summary['full_base_validation'][denominator].items()):
                raise ValueError('Frozen Base ranking was not reproduced')
        plain_rank = ex.e1['base_validation_ranking']
        if plain_rank is None:
            raise ValueError('E1 ranking must exist for matched ranking comparison')
        ranking['plain_new_full_sid'] = plain_rank['new_full_sid']
        ranking['compatible_minus_plain_full_denominator'] = {
            k: ranking['new_full_sid']['full_denominator'][k]-v
            for k, v in plain_rank['new_full_sid']['full_denominator'].items()}
        write_json(output/'base_validation_ranking.json', ranking)
    check_teacher(model, ex)
    report = dict(stage='historical_aware_update_evaluation', execution_status='completed',
        e1_dir=str(ex.e1_dir.resolve()), historical_rqvae_dir=str(directory.resolve()),
        protocol_sha256=ex.binding['protocol_sha256'], selected_tiger_sha256=ex.e1['selected_tiger_sha256'],
        historical_rqvae_sha256=sha256(directory/'last.pth'), plain_rqvae_sha256=ex.e1['update_rqvae_sha256'],
        update_optimizer_steps=summary['update_optimizer_steps'], compatibility_weight=summary['config']['compatibility_weight'],
        groups=groups, paired_compatible_vs_plain=paired_comparison, base_validation_ranking=ranking,
        rqvae_reconstruction=dict(plain=ex.plain['final_train_catalog_metrics'],
                                 compatible=summary['final_train_catalog_metrics']),
        churn=dict(plain=ex.plain['base_semantic_churn'], compatible=summary['base_semantic_churn']),
        primary_cohort='anchor_holdout; held out from compatibility penalty, not from Base TIGER training',
        labels='original full SID plus EOS', ranking_catalog='base_train; original full SIDs',
        teacher_frozen=True, recommendation_training_status='not_run', future_test_status='not_run',
        E0_full_status='not_run', historical_aware_training_status='completed',
        device=str(device), runtime_seconds=time.perf_counter()-started,
        artifact_files_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'summary.json', report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report
