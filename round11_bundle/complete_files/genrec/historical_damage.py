"""E1 paired input-reencoding damage with frozen repository TIGER and old labels."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from genrec.evaluator import Evaluator
from genrec.models.TIGER.model import TIGER
from genrec.temporal_sids import (code_to_tokens, load_protocol, maps_from_ledger,
    prepare_split, read_json, read_jsonl, sha256, verify_files, write_json)
from genrec.temporal_tiger import evaluate, load_sids, make_batch, seed_all
from gensid.plain_incremental_content import load_inputs


def versioned_suffixes(previous, raw_items, semantic_codes):
    """Reserve all old identifiers; retain unchanged-prefix slots and append movers.

    This is a new version. The immutable Base allocator deliberately rejects
    changed semantic codes and must never be used in place for this operation.
    """
    capacities = previous['semantic_capacities']+[previous['suffix_capacity']]
    if (len(capacities) != 4 or len(raw_items) != len(semantic_codes)
            or len(raw_items) < len(previous['entries'])
            or len(set(raw_items)) != len(raw_items) or '[PAD]' in raw_items):
        raise ValueError('Invalid new-version item/capacity contract')
    old, occupied, next_slot = {}, set(), {}
    for i, entry in enumerate(previous['entries']):
        raw, prefix, suffix = entry['raw_item_id'], tuple(entry['semantic_code']), entry['collision_suffix']
        code_to_tokens([*prefix, suffix], capacities)
        if (raw in old or (*prefix, suffix) in occupied or raw_items[i] != raw
                or entry['gr_item_id'] != i+1 or entry['gensid_global_item_id'] != i):
            raise ValueError('Previous allocation order/ID/reservation mismatch')
        old[raw] = entry
        occupied.add((*prefix, suffix))
        next_slot[prefix] = max(next_slot.get(prefix, 0), suffix+1)
    entries = []
    for i, (raw, codes) in enumerate(zip(raw_items, semantic_codes)):
        code_to_tokens(codes, capacities[:3])
        prefix = tuple(codes)
        prior = old.get(raw)
        keep = prior is not None and prior['semantic_code'] == list(codes)
        suffix = prior['collision_suffix'] if keep else next_slot.get(prefix, 0)
        if suffix >= capacities[-1]:
            raise ValueError('New-version collision suffix capacity exhausted')
        if not keep:
            occupied.add((*prefix, suffix))
            next_slot[prefix] = suffix+1
        entries.append(dict(raw_item_id=raw, gr_item_id=i+1, gensid_global_item_id=i,
            feature_row=i, semantic_code=list(codes), collision_suffix=suffix,
            allocation_reason='unchanged_prefix_keep_slot' if keep else 'append_after_reserved_slots'))
    return dict(schema_version=1, semantic_capacities=capacities[:3], suffix_capacity=capacities[-1],
        entries=entries, version_policy='new snapshot; old identifiers reserved; no old snapshot mutation',
        reserved_previous_entries=previous['entries'])


def historical_cohorts(rows, size, seed):
    users = sorted({r['user'] for r in rows}, key=lambda u: hashlib.sha256(
        (str(seed)+':user:'+str(u)).encode()).hexdigest())
    if len(users) < 2:
        raise ValueError('At least two historical users required for disjoint cohorts')
    train_users = set(users[:len(users)//2])
    groups = {'anchor_train': [], 'anchor_holdout': []}
    for i, row in enumerate(rows):
        groups['anchor_train' if row['user'] in train_users else 'anchor_holdout'].append(i)
    for key, values in groups.items():
        ordered = sorted(values, key=lambda i: hashlib.sha256(
            (str(seed)+':sample:'+str(rows[i]['sample_id'])).encode()).hexdigest())
        groups[key] = sorted(ordered[:size])
        if not groups[key]:
            raise ValueError('Empty historical anchor cohort')
    return groups


def reencode_history(arrays, rows, mapping):
    """Only replace available history; preserve old labels, masks, EOS and row order."""
    changed = dict(arrays)
    ids = np.array(arrays['input_ids'], copy=True)
    for i, source in enumerate(arrays['source_rows']):
        row = rows[int(source)]
        for j, (raw, available) in enumerate(zip(row['item_seq'][:-1], row['history_available'])):
            ids[i, j*4:(j+1)*4] = mapping[raw] if available else [0]*4
    if not np.array_equal(ids != 0, arrays['attention_mask']):
        raise ValueError('Reencoding changed causal availability/masks')
    changed['input_ids'] = ids
    return changed


def aggregate(records, condition, bootstrap_reps, seed):
    if not records:
        raise ValueError('No paired historical losses')
    old = np.array([r['old_sequence_nll'] for r in records], dtype=np.float64)
    new = np.array([r[condition+'_sequence_nll'] for r in records], dtype=np.float64)
    delta = new-old
    clusters = {}
    for row, value in zip(records, delta):
        total, count = clusters.get(row['user'], (0.0, 0))
        clusters[row['user']] = (total+value, count+1)
    pairs = np.array(list(clusters.values()))
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(bootstrap_reps):
        chosen = rng.integers(0, len(pairs), len(pairs))
        bootstrap.append(float(pairs[chosen, 0].sum()/pairs[chosen, 1].sum()))
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist() if len(pairs) > 1 else None
    eps = 1e-6
    return dict(queries=len(records), users=len(clusters), mean_old_sequence_nll=float(old.mean()),
        mean_new_sequence_nll=float(new.mean()), mean_delta_sequence_nll=float(delta.mean()),
        mean_delta_token_ce=float(delta.mean()/5), median_delta_sequence_nll=float(np.median(delta)),
        delta_quantiles={str(q): float(np.quantile(delta, q)) for q in (0.05, 0.25, 0.75, 0.95, 0.99)},
        worsened_fraction=float((delta > eps).mean()), improved_fraction=float((delta < -eps).mean()),
        within_tolerance_fraction=float((np.abs(delta) <= eps).mean()), tolerance=eps,
        mean_positive_part_delta=float(np.maximum(delta, 0).mean()),
        mean_teacher_forced_positionwise_kl_old_to_new=float(np.mean([r[condition+'_kl_sum'] for r in records])),
        mean_delta_by_target_position=(np.mean([r[condition+'_position_nll'] for r in records], axis=0)
                           -np.mean([r['old_position_nll'] for r in records], axis=0)).tolist(),
        history_changed_fraction=float(np.mean([r[condition+'_history_changed'] for r in records])),
        max_abs_delta_unchanged_history=max((abs(r[condition+'_delta_sequence_nll']) for r in records
            if not r[condition+'_history_changed']), default=None),
        cluster_bootstrap_mean_delta_ci95=interval, cluster_bootstrap_replicates=bootstrap_reps,
        bootstrap_unit='user; row-weighted mean in resampled user clusters',
        positive_damage_detected=bool(interval is not None and interval[0] > eps and delta.mean() > eps))


@torch.no_grad()
def paired_losses(model, variants, rows, selected_tensor_rows, batch_size, device, label):
    records = []
    for start in tqdm(range(0, len(selected_tensor_rows), batch_size), desc=label, unit='batch'):
        indices = np.asarray(selected_tensor_rows[start:start+batch_size], dtype=np.int64)
        outputs, probabilities = {}, None
        for key in ('old', 'new_full_sid', 'new_semantics_old_suffix'):
            batch = make_batch(variants[key], indices, device)
            logits = model(batch).logits.float()
            log_probs = F.log_softmax(logits, dim=-1)
            nll = F.cross_entropy(logits.transpose(1, 2), batch['labels'], reduction='none')
            if not torch.isfinite(nll).all() or not torch.isfinite(log_probs).all():
                raise FloatingPointError('Non-finite frozen TIGER prediction')
            if key == 'old':
                probabilities = (log_probs, log_probs.exp())
                kl = torch.zeros(len(indices), device=device)
            else:
                kl = (probabilities[1]*(probabilities[0]-log_probs)).sum(dim=(1, 2)).clamp_min(0)
            outputs[key] = (nll.cpu().numpy(), kl.cpu().numpy())
        for j, tensor_row in enumerate(indices):
            source = int(variants['old']['source_rows'][tensor_row])
            raw = rows[source]
            record = dict(source_row=source, tensor_row=int(tensor_row), sample_id=raw['sample_id'],
                user=raw['user'], target_raw_item_id=raw['item_seq'][-1], target_time=raw['target_time'],
                unavailable_history_events=sum(not x for x in raw['history_available']),
                target_labels=variants['old']['labels'][tensor_row].tolist())
            for key, (nll, kl) in outputs.items():
                record[key+'_position_nll'] = nll[j].tolist()
                record[key+'_sequence_nll'] = float(nll[j].sum())
                if key != 'old':
                    record[key+'_delta_sequence_nll'] = record[key+'_sequence_nll']-record['old_sequence_nll']
                    record[key+'_kl_sum'] = float(kl[j])
                    record[key+'_history_changed'] = bool(np.any(variants[key]['input_ids'][tensor_row]
                                                                 != variants['old']['input_ids'][tensor_row]))
            records.append(record)
    return records


def load_bound_inputs(args):
    tiger_dir, update_dir = Path(args.tiger_dir), Path(args.update_rqvae_dir)
    tiger_summary, update_summary = read_json(tiger_dir/'summary.json'), read_json(update_dir/'summary.json')
    if (tiger_summary.get('stage') != 'E0_base_content_TIGER'
            or tiger_summary.get('execution_status') != 'completed'
            or tiger_summary.get('E0_base_end_to_end_status') != 'completed'):
        raise ValueError('Expected completed Base TIGER with full Base validation')
    verify_files(tiger_dir, tiger_summary, ['best.pth', 'base_validation.json', 'config.json'])
    tiger_sha = sha256(tiger_dir/'best.pth')
    if args.expected_tiger_sha256 and tiger_sha != args.expected_tiger_sha256:
        raise ValueError('Unexpected selected Base TIGER SHA')
    teacher = torch.load(tiger_dir/'best.pth', map_location='cpu', weights_only=False)
    if (teacher.get('format') != 'incremental_base_tiger_selected_v1'
            or teacher['binding'] != tiger_summary['binding'] or teacher['config'] != tiger_summary['config']
            or teacher['optimizer_step'] != tiger_summary['best_optimizer_step']):
        raise ValueError('Selected TIGER checkpoint/summary mismatch')
    root = Path(__file__).resolve().parents[1]
    for name, expected in teacher['binding']['source_files_sha256'].items():
        if sha256(root/name) != expected:
            raise ValueError('Frozen TIGER source mismatch: '+name)
    for name, expected in teacher['binding']['package_versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Frozen TIGER software mismatch: '+name)
    sid_dir = Path(args.sid_dir or tiger_summary['sid_dir'])
    sid_summary, tokenizer, arrays, audits = load_sids(sid_dir, tiger_summary['rqvae_checkpoint_sha256'])
    if sha256(sid_dir/'summary.json') != teacher['binding']['sid_summary_sha256']:
        raise ValueError('Frozen TIGER SID snapshot mismatch')
    if (update_summary.get('stage') != 'E1_plain_update_content_rqvae_preparation'
            or update_summary.get('execution_status') != 'completed'
            or update_summary['parent_base_checkpoint_sha256'] != sid_summary['rqvae_checkpoint_sha256']
            or update_summary['historical_prediction_penalty_used'] is not False):
        raise ValueError('Expected completed plain Update from the selected Base RQ-VAE')
    verify_files(update_dir, update_summary, ['last.pth', 'semantic_assignments.jsonl',
                                             'old_semantic_assignments.jsonl', 'config.json'])
    update = torch.load(update_dir/'last.pth', map_location='cpu', weights_only=False)
    if (update.get('format') != 'plain_update_rqvae_v1' or update['binding'] != update_summary['binding']
            or update['config'] != update_summary['config']
            or update['sampler_state']['step'] != update_summary['update_optimizer_steps']):
        raise ValueError('Update checkpoint/summary mismatch')
    features, feature_rows, parent, parent_report, binding, groups, old_export, old_batch = load_inputs(
        args.protocol_dir, update_summary['features_dir'], update_summary['base_rqvae_dir'],
        sid_summary['rqvae_checkpoint_sha256'])
    for key, value in binding.items():
        if update['binding'].get(key) != value:
            raise ValueError('Update source/data binding mismatch: '+key)
    for name, expected in update['binding']['source_files_sha256'].items():
        if sha256(root/name) != expected:
            raise ValueError('Plain Update source mismatch: '+name)
    protocol_summary, manifest, registries, examples = load_protocol(Path(args.protocol_dir))
    if (manifest['protocol_sha256'] != sid_summary['protocol_sha256']
            or manifest['protocol_sha256'] != update_summary['protocol_sha256']
            or manifest['protocol_sha256'] != tiger_summary['protocol_sha256']):
        raise ValueError('Historical measurement protocol mismatch')
    raw_to_global = {raw: i for i, raw in enumerate(registries['base_train']['id2item'][1:])}
    for split in ('base_train', 'base_val'):
        rebuilt, audit, stats = prepare_split(examples[split], tokenizer, raw_to_global)
        if audit != audits[split] or any(not np.array_equal(rebuilt[k], arrays[split][k]) for k in arrays[split]):
            raise ValueError('Raw history differs from frozen TIGER tensors/audit')
    old_assignments, new_assignments = (read_jsonl(update_dir/name) for name in
        ('old_semantic_assignments.jsonl', 'semantic_assignments.jsonl'))
    if len(old_assignments) != len(feature_rows) or len(new_assignments) != len(feature_rows):
        raise ValueError('Update assignment count mismatch')
    for rows in (old_assignments, new_assignments):
        for known, row in zip(feature_rows, rows):
            if any(row.get(k) != v for k, v in known.items()):
                raise ValueError('Update assignment raw/global ID mismatch')
            code_to_tokens(row['semantic_code'], tokenizer.capacities[:3])
    previous = read_json(sid_dir/'base/allocation_registry.json')
    for entry, row in zip(previous['entries'], old_assignments):
        if entry['raw_item_id'] != row['raw_item_id'] or entry['semantic_code'] != row['semantic_code']:
            raise ValueError('Original deployment codes differ from Update reference')
    if len(previous['entries']) != len(registries['base']['id2item'])-1:
        raise ValueError('Original deployment allocation count mismatch')
    new_ledger = versioned_suffixes(previous, [r['raw_item_id'] for r in new_assignments],
                                   [r['semantic_code'] for r in new_assignments])
    _, full_mapping = maps_from_ledger(new_ledger)
    # Validate unique, fixed-capacity full SIDs for the complete eligible Update catalog.
    from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
    FrozenSIDTokenizer(full_mapping, tokenizer.capacities, manifest['max_history'])
    control_mapping = {raw: [*full_mapping[raw][:3], old_tokens[3]]
                       for raw, old_tokens in tokenizer.item2tokens.items()}
    variants = {split: dict(old=arrays[split],
        new_full_sid=reencode_history(arrays[split], examples[split], full_mapping),
        new_semantics_old_suffix=reencode_history(arrays[split], examples[split], control_mapping))
        for split in arrays}
    return tiger_summary, teacher, sid_dir, tokenizer, arrays, audits, examples, variants, new_ledger


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite E1 output: '+str(output))
    if min(args.batch_size, args.anchor_size, args.bootstrap_reps) < 1 or not 0 <= args.seed < 2**32:
        raise ValueError('Invalid batch size, cohort size, bootstrap or seed')
    tiger_summary, teacher, sid_dir, tokenizer, arrays, audits, examples, variants, ledger = load_bound_inputs(args)
    cohorts = historical_cohorts(examples['base_train'], args.anchor_size, args.seed)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested evaluation device unavailable')
    seed_all(args.seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = TIGER(teacher['config'], None, tokenizer).to(device)
    model.load_state_dict(teacher['state_dict'], strict=True)
    model.eval()
    model.requires_grad_(False)
    output.mkdir(parents=True)
    write_json(output/'new_allocation_registry.json', ledger)
    _, mapping = maps_from_ledger(ledger)
    write_json(output/'new_item2tokens.json', mapping)
    for name, indices in cohorts.items():
        with (output/(name+'.jsonl')).open('w') as handle:
            for i in indices:
                handle.write(json.dumps(dict(source_row=i, **examples['base_train'][i]))+'\n')
    report_groups = {}
    tasks = [(name, 'base_train', indices) for name, indices in cohorts.items()]
    tasks.append(('base_validation', 'base_val', list(range(len(arrays['base_val']['labels'])))))
    print('E1 ready | frozen selected TIGER step '+str(teacher['optimizer_step'])+
          ' | old target labels fixed | user-disjoint historical cohorts', flush=True)
    for name, split, selected in tasks:
        records = paired_losses(model, variants[split], examples[split], selected,
                                args.batch_size, device, name)
        with (output/(name+'_paired_losses.jsonl')).open('w') as handle:
            for row in records:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
        summaries = {key: aggregate(records, key, args.bootstrap_reps, args.seed) for key in
                     ('new_full_sid', 'new_semantics_old_suffix')}
        report_groups[name] = summaries
        print(name+': '+json.dumps(summaries, allow_nan=False), flush=True)
    old_validation_nll = report_groups['base_validation']['new_full_sid']['mean_old_sequence_nll']
    expected_nll = tiger_summary['full_base_validation']['mean_sequence_nll_servable']
    if abs(old_validation_nll-expected_nll) > 1e-3:
        raise ValueError('Old-label validation NLL does not reproduce the frozen Base reference')
    ranking = None
    if args.ranking:
        wrapper = SimpleNamespace(model=model, config=teacher['config'],
                                  evaluator=Evaluator(teacher['config'], tokenizer))
        ranking = {}
        for key in ('old', 'new_full_sid'):
            ranking[key] = evaluate(wrapper, tokenizer, variants['base_val'][key], audits['base_val'],
                list(range(len(audits['base_val']))), device, teacher['config']['eval_batch_size'])
        old_ref = tiger_summary['full_base_validation']
        for denom in ('full_denominator', 'conditional_on_servable'):
            if any(abs(ranking['old'][denom][k]-v) > 1e-8 for k, v in old_ref[denom].items()):
                raise ValueError('Old ranking does not reproduce the frozen Base reference')
        ranking['delta_full_denominator'] = {k: ranking['new_full_sid']['full_denominator'][k]-v
                                             for k, v in ranking['old']['full_denominator'].items()}
        write_json(output/'base_validation_ranking.json', ranking)
    frozen_equal = all(torch.equal(value.cpu(), teacher['state_dict'][key]) for key, value in model.state_dict().items())
    if not frozen_equal or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen TIGER was modified or accumulated gradients')
    root = Path(__file__).resolve().parents[1]
    sources = ['genrec/historical_damage.py', 'measure_historical_damage.py']
    report = dict(stage='E1_historical_input_reencoding_damage', execution_status='completed',
        protocol_dir=str(Path(args.protocol_dir).resolve()), sid_dir=str(sid_dir.resolve()),
        tiger_dir=str(Path(args.tiger_dir).resolve()), update_rqvae_dir=str(Path(args.update_rqvae_dir).resolve()),
        protocol_sha256=tiger_summary['protocol_sha256'], selected_tiger_sha256=sha256(Path(args.tiger_dir)/'best.pth'),
        selected_tiger_optimizer_step=teacher['optimizer_step'],
        base_rqvae_sha256=tiger_summary['rqvae_checkpoint_sha256'],
        update_rqvae_sha256=sha256(Path(args.update_rqvae_dir)/'last.pth'),
        input_summary_sha256={name: sha256(path/'summary.json') for name, path in
            (('tiger', Path(args.tiger_dir)), ('update_rqvae', Path(args.update_rqvae_dir)), ('sid', sid_dir),
             ('protocol', Path(args.protocol_dir)))},
        source_files_sha256={name: sha256(root/name) for name in sources},
        groups=report_groups, base_validation_ranking=ranking,
        historical_damage_detected=report_groups['anchor_holdout']['new_full_sid']['positive_damage_detected'],
        detection_rule='held-out mean Delta NLL > 1e-6 and user-cluster bootstrap CI95 lower bound > 1e-6',
        labels='fixed original full SID plus EOS; teacher forcing uses the same old target prefix',
        primary_condition='new_full_sid', diagnostic_condition='new_semantics_old_suffix',
        diagnostic_condition_serving_sid=False, ranking_catalog='base_train; old full SIDs',
        unavailable_history_policy='original four zero tokens/mask retained, even if item exists in Update',
        base_validation_all_queries=len(audits['base_val']),
        base_validation_nll_queries=len(arrays['base_val']['labels']),
        base_validation_targets_without_old_catalog_labels=len(audits['base_val'])-len(arrays['base_val']['labels']),
        unavailable_target_nll_policy='undefined; exclude from NLL, count as misses in full-denominator ranking',
        cohort_selection='deterministic user split, then sample_id SHA order within each half',
        cohort_seed=args.seed, requested_queries_per_anchor_cohort=args.anchor_size,
        anchor_train_use='saved for future historical penalty; no penalty training in this run',
        anchor_holdout_use='primary damage measurement; do not use for penalty training',
        target_position_order=['semantic_1', 'semantic_2', 'semantic_3', 'collision_suffix', 'EOS'],
        kl_scope='sum of token-distribution KL at five teacher-forced positions with old target prefixes; not full item-distribution KL',
        new_full_sids_unique=len(set(map(tuple, mapping.values()))) == len(mapping),
        suffix_policy=ledger['version_policy'], old_suffix_slots_reserved=True,
        tokenizer_vocab_size=tokenizer.vocab_size, tiger_frozen=True, frozen_teacher_parameters_equal=True,
        old_validation_nll_reproduced=True, historical_aware_training_status='not_run',
        codebook_training_performed=False, recommendation_training_performed=False,
        future_test_status='not_run', E0_full_status='not_run', E1_status='completed',
        python=platform.python_version(), package_versions={name: importlib.metadata.version(name)
            for name in ('torch', 'numpy', 'transformers', 'scikit-learn')}, device=str(device),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        runtime_seconds=time.perf_counter()-started,
        artifact_files_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'summary.json', report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-dir', required=True)
    p.add_argument('--tiger-dir', required=True)
    p.add_argument('--update-rqvae-dir', required=True)
    p.add_argument('--sid-dir', help='Defaults to the frozen SID directory recorded by Base TIGER')
    p.add_argument('--expected-tiger-sha256')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--anchor-size', type=int, default=4096)
    p.add_argument('--bootstrap-reps', type=int, default=500)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--ranking', action='store_true', help='Also compare complete Base validation rankings')
    return p
