"""Two-stage codebook repair with full-training-anchor hard acceptance checks.

Run on the user's machine. Starts from the E1-bound Plain Update, freezes the
encoder/decoder, and uses the existing history STE only to propose directions.
Held-out queries never select a proposal. This adds compute beyond Plain Update.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

import gensid.historical_incremental_content as h
import check_history_gradient_direction as direction
from gensid.incremental_content import build_model, seed_all, sha256, write_json, collision_stats
from gensid.plain_incremental_content import evaluate_groups, churn
from genrec.temporal_sids import read_json, read_jsonl, maps_from_ledger, verify_files
from genrec.historical_damage import versioned_suffixes, reencode_history, paired_losses, aggregate
from genrec.temporal_tiger import evaluate as rank_evaluate
from genrec.evaluator import Evaluator

FORMAT = 'hard_checked_codebook_repair_v1'
SOURCES = ['repair_historical_codebooks.py', 'check_history_gradient_direction.py']


def binding(ex):
    root = Path(__file__).resolve().parent
    return dict(experiment=ex.binding, repair_sources={p: sha256(root/p) for p in SOURCES})


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def freeze_content_networks(model):
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('rq.vq_layers.') and name.endswith('.embedding.weight'))
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError('No repository codebook parameters found')


def verify_frozen(model, initial):
    for name, value in model.state_dict().items():
        if name.startswith(('encoder.', 'decoder.')) and not torch.equal(value.cpu(), initial[name]):
            raise ValueError('Frozen content parameter changed: '+name)
    if any(p.grad is not None for n, p in model.named_parameters() if n.startswith(('encoder.', 'decoder.'))):
        raise ValueError('Frozen content network received gradients')


def acceptance(current_damage, candidate_damage, candidate_mse, fixed_mse_cap, feasible, min_decrease):
    """Fixed initial MSE cap; hard objective must strictly improve."""
    if not feasible:
        return False, 'suffix_capacity'
    if not np.isfinite(candidate_mse) or candidate_mse > fixed_mse_cap:
        return False, 'reconstruction_cap'
    if not np.isfinite(candidate_damage) or not candidate_damage < current_damage-min_decrease:
        return False, 'hard_training_damage_not_lower'
    return True, 'accepted'


def metrics(model, ex):
    return evaluate_groups(model, ex.features, ex.groups, ex.old_batch,
                           ex.plain['config']['added_item_assignment_eval_batch_size'])


def validate_args(args):
    if (args.max_rounds < 1 or args.gradient_queries < 1 or args.batch_size < 1
            or not np.isfinite(args.mse_slack) or args.mse_slack < 0
            or not np.isfinite(args.min_decrease) or args.min_decrease <= 0
            or not args.relative_steps
            or any(not np.isfinite(v) or v <= 0 for v in args.relative_steps)):
        raise ValueError('Invalid repair budget, steps, or acceptance thresholds')


def repair(args):
    started = time.perf_counter()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite repair output: '+str(out))
    validate_args(args)
    ex, device = h.experiment(args), h.device_from(args.device)
    seed_all(ex.e1['cohort_seed'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    plain_dir = Path(ex.e1['update_rqvae_dir'])
    cp = torch.load(plain_dir/'last.pth', map_location='cpu', weights_only=False)
    if (cp.get('format') != 'plain_update_rqvae_v1' or cp['binding'] != ex.plain['binding']
            or cp['config'] != ex.plain['config']):
        raise ValueError('Plain checkpoint binding/config mismatch')
    model = build_model(cp['config']).to(device)
    model.load_state_dict(cp['state_dict'], strict=True)
    freeze_content_networks(model)
    initial_state = snapshot(model)
    initial_metrics, initial_groups, initial_codes = metrics(model, ex)
    reference_mse = initial_metrics['reconstruction_loss']
    if abs(reference_mse-ex.plain['final_train_catalog_metrics']['reconstruction_loss']) > 1e-7:
        raise ValueError('Plain reconstruction was not reproduced')
    fixed_mse_cap = reference_mse*(1+args.mse_slack)
    raw_items = [r['raw_item_id'] for r in ex.feature_rows]
    initial_capacity = h.suffix_capacity_audit(ex.previous, raw_items, initial_codes.tolist())
    if not initial_capacity['full_sid_allocation_feasible']:
        raise ValueError('Plain start point must have feasible full SID allocation')
    base = build_model(ex.parent['config']).to(device)
    base.load_state_dict(ex.parent['state_dict'], strict=True)
    scales = h.distance_scales(base, ex.features[:ex.groups['base_train'][1]], ex.old_batch)
    del base
    teacher = h.frozen_teacher(ex, device)
    anchors = np.asarray(ex.cohorts['anchor_train'], dtype=np.int64)
    selected = np.sort(np.random.default_rng(ex.e1['cohort_seed']).choice(
        anchors, min(args.gradient_queries, len(anchors)), replace=False))
    print('Cache full anchor_train hard reference | queries='+str(len(anchors)), flush=True)
    full_packs, base_codes = direction.contexts(model, teacher, ex, anchors, args.batch_size, scales, device)
    current = direction.hard_objective(model, teacher, ex, full_packs, base_codes, device)
    initial_hard = dict(current)
    out.mkdir(parents=True)
    write_json(out/'gradient_anchor_train_rows.json', selected.tolist())
    write_json(out/'verification_anchor_train_rows.json', anchors.tolist())
    proposals, accepted, rounds, work = [], 0, 0, dict(
        gradient_anchor_queries=0, hard_verification_anchor_queries=len(anchors),
        old_reference_anchor_queries=len(anchors), gradient_context_old_reference_queries=0,
        reconstruction_full_catalog_passes=1, suffix_full_catalog_checks=1)
    stop_reason = 'max_rounds'
    for round_id in range(1, args.max_rounds+1):
        rounds = round_id
        current_state = snapshot(model)
        packs, _ = direction.contexts(model, teacher, ex, selected, args.batch_size, scales, device)
        work['gradient_context_old_reference_queries'] += len(selected)
        objective, grads = direction.objective(model, teacher, ex, packs, scales, device, 'ste', backward=True)
        work['gradient_anchor_queries'] += len(selected)
        made_progress = False
        for relative_step in args.relative_steps:
            change = direction.perturb(model, current_state, grads, 'codebooks', relative_step, -1)
            if change is None:
                stop_reason = 'zero_codebook_surrogate_gradient'
                break
            candidate_metrics, _, candidate_codes = metrics(model, ex)
            work['reconstruction_full_catalog_passes'] += 1
            capacity = h.suffix_capacity_audit(ex.previous, raw_items, candidate_codes.tolist())
            work['suffix_full_catalog_checks'] += 1
            # Cheap gates first. Every accepted proposal also gets the full hard check.
            hard = None
            if (capacity['full_sid_allocation_feasible'] and np.isfinite(candidate_metrics['reconstruction_loss'])
                    and candidate_metrics['reconstruction_loss'] <= fixed_mse_cap):
                hard = direction.hard_objective(model, teacher, ex, full_packs, base_codes, device)
                work['hard_verification_anchor_queries'] += len(anchors)
            ok, reason = acceptance(current['positive_damage'],
                hard['positive_damage'] if hard else float('nan'), candidate_metrics['reconstruction_loss'],
                fixed_mse_cap, capacity['full_sid_allocation_feasible'], args.min_decrease)
            row = dict(round=round_id, **change, accepted=ok, reason=reason,
                previous_full_train_positive_damage=current['positive_damage'],
                full_train_hard=hard, rq_metrics=candidate_metrics,
                maximum_collision_bucket=capacity['maximum_current_collision_bucket'],
                maximum_required_suffix_capacity=capacity['maximum_required_suffix_capacity'],
                full_sid_allocation_feasible=capacity['full_sid_allocation_feasible'],
                gradient_subset_objective=objective)
            proposals.append(row)
            with (out/'proposals.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(dict(round=round_id, relative_step=relative_step, accepted=ok, reason=reason,
                hard_positive_damage=hard['positive_damage'] if hard else None,
                RQ_MSE=candidate_metrics['reconstruction_loss']), allow_nan=False), flush=True)
            if ok:
                current, accepted, made_progress = hard, accepted+1, True
                break
        if not made_progress:
            model.load_state_dict(current_state, strict=True)
            if stop_reason != 'zero_codebook_surrogate_gradient':
                stop_reason = 'no_acceptable_descent_proposal'
            break
        verify_frozen(model, initial_state)
    model.zero_grad(set_to_none=True)
    verify_frozen(model, initial_state)
    final_metrics, final_groups, final_codes = metrics(model, ex)
    final_hard = direction.hard_objective(model, teacher, ex, full_packs, base_codes, device)
    work['reconstruction_full_catalog_passes'] += 1
    work['hard_verification_anchor_queries'] += len(anchors)
    if (final_hard['positive_damage'] > initial_hard['positive_damage']+1e-7
            or final_metrics['reconstruction_loss'] > fixed_mse_cap
            or abs(final_hard['positive_damage']-current['positive_damage']) > 1e-7):
        raise ValueError('Final hard acceptance invariant failed')
    ledger = versioned_suffixes(ex.previous, raw_items, final_codes.tolist())
    _, mapping = maps_from_ledger(ledger)
    repair_binding = binding(ex)
    torch.save(dict(format=FORMAT, binding=repair_binding, config=cp['config'], state_dict=snapshot(model),
                    accepted_repair_steps=accepted), out/'last.pth')
    loaded = torch.load(out/'last.pth', map_location='cpu', weights_only=False)
    reload_model = build_model(loaded['config']).to(device)
    reload_model.load_state_dict(loaded['state_dict'], strict=True)
    if not np.array_equal(h.catalog_codes(reload_model, ex), final_codes):
        raise ValueError('Checkpoint reload assignment mismatch')
    if any(not torch.equal(v.cpu(), loaded['state_dict'][k]) for k, v in model.state_dict().items()):
        raise ValueError('Checkpoint reload weights mismatch')
    h.check_teacher(teacher, ex)
    if sha256(plain_dir/'last.pth') != ex.e1['update_rqvae_sha256']:
        raise ValueError('Plain source checkpoint changed')
    write_json(out/'config.json', loaded['config'])
    with (out/'semantic_assignments.jsonl').open('w') as f:
        for row, codes in zip(ex.feature_rows, final_codes.tolist()):
            f.write(json.dumps(dict(**row, semantic_code=codes))+'\n')
    write_json(out/'new_allocation_registry.json', ledger)
    write_json(out/'new_item2tokens.json', mapping)
    n_base = ex.groups['base_train'][1]
    report = dict(stage='hard_checked_codebook_repair', execution_status='completed', binding=repair_binding,
        e1_dir=str(ex.e1_dir.resolve()), starting_plain_rqvae_dir=str(plain_dir.resolve()),
        starting_plain_checkpoint_sha256=ex.e1['update_rqvae_sha256'], output=str(out), config=loaded['config'],
        repair_protocol=vars(args), rounds_attempted=rounds, accepted_repair_steps=accepted,
        proposals_checked=len(proposals), stop_reason=stop_reason,
        initial_full_anchor_train=initial_hard, final_full_anchor_train=final_hard,
        fixed_initial_reconstruction_mse=reference_mse, fixed_reconstruction_mse_cap=fixed_mse_cap,
        initial_train_catalog_metrics=initial_metrics, final_train_catalog_metrics=final_metrics,
        initial_group_metrics=initial_groups, final_group_metrics=final_groups,
        collision_statistics=collision_stats(final_codes, cp['config']['model']['num_emb_list']),
        base_semantic_churn=churn(ex.old_export, final_codes[:n_base]),
        change_from_plain=churn(initial_codes[:n_base], final_codes[:n_base]),
        full_sid_export_status='completed', teacher_frozen=True, teacher_unchanged=True,
        encoder_unchanged=True, decoder_unchanged=True, source_checkpoint_unchanged=True,
        checkpoint_reload_assignments_equal=True, trained_components=['three_codebooks'],
        history_input_condition='new_semantics_old_suffix', old_target_labels_fixed=True,
        temperature=0.1, distance_scales=scales, penalty_cohort='anchor_train',
        gradient_subset_queries=len(selected), hard_acceptance_queries=len(anchors),
        anchor_holdout_used_for_selection=False, base_validation_used_for_selection=False,
        starting_plain_update_optimizer_steps=ex.plain['update_optimizer_steps'],
        repair_optimizer_steps=0, work_counters=work,
        compute_comparison='extra repair after Plain Update; not a matched-compute baseline comparison',
        recommendation_training_status='not_run', future_test_status='not_run', graph_used=False,
        device=str(device), runtime_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        artifact_files_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'summary.json', report)
    print(json.dumps(dict(accepted_repair_steps=accepted, stop_reason=stop_reason,
        initial_hard_positive_damage=initial_hard['positive_damage'],
        final_hard_positive_damage=final_hard['positive_damage'],
        RQ_MSE=final_metrics['reconstruction_loss'], full_sid_export_status='completed'), indent=2), flush=True)
    return report


def paired_contrast(repaired, plain, condition, replicates, seed):
    if len(repaired) != len(plain) or not repaired:
        raise ValueError('Paired cohort size mismatch')
    clusters, nll_diffs, damage_diffs = {}, [], []
    for a, b in zip(repaired, plain):
        if any(a[k] != b[k] for k in ('sample_id', 'user', 'target_labels', 'source_row')):
            raise ValueError('Paired query/old-label mismatch')
        if abs(a['old_sequence_nll']-b['old_sequence_nll']) > 1e-3:
            raise ValueError('Old teacher NLL differs from bound Plain reference')
        da, db = a[condition+'_sequence_nll']-a['old_sequence_nll'], b[condition+'_sequence_nll']-b['old_sequence_nll']
        nd, hd = a[condition+'_sequence_nll']-b[condition+'_sequence_nll'], max(da, 0)-max(db, 0)
        nll_diffs.append(nd); damage_diffs.append(hd)
        value = clusters.setdefault(a['user'], [0., 0., 0.])
        value[0] += nd; value[1] += hd; value[2] += 1
    values = np.asarray(list(clusters.values()))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        totals = values[rng.integers(0, len(values), len(values))].sum(0)
        samples.append((totals[:2]/totals[2]).tolist())
    ci = np.quantile(samples, [0.025, 0.975], axis=0) if len(values) > 1 else None
    return dict(queries=len(repaired), users=len(values), mean_repair_minus_plain_sequence_nll=float(np.mean(nll_diffs)),
        mean_repair_minus_plain_positive_damage=float(np.mean(damage_diffs)),
        sequence_nll_cluster_ci95=ci[:, 0].tolist() if ci is not None else None,
        positive_damage_cluster_ci95=ci[:, 1].tolist() if ci is not None else None,
        bootstrap_unit='user; row-weighted paired means', bootstrap_replicates=replicates)


def evaluate_repair(args):
    started = time.perf_counter()
    out, directory = Path(args.output).resolve(), Path(args.repair_dir).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite repair evaluation')
    if args.batch_size < 1:
        raise ValueError('Invalid batch size')
    ex, device = h.experiment(args), h.device_from(args.device)
    seed_all(ex.e1['cohort_seed'])
    report = read_json(directory/'summary.json')
    if (report.get('stage') != 'hard_checked_codebook_repair' or report.get('execution_status') != 'completed'
            or report['binding'] != binding(ex)):
        raise ValueError('Repair checkpoint binding mismatch')
    verify_files(directory, report, ['last.pth', 'config.json', 'semantic_assignments.jsonl',
        'new_allocation_registry.json', 'new_item2tokens.json', 'gradient_anchor_train_rows.json',
        'verification_anchor_train_rows.json'])
    cp = torch.load(directory/'last.pth', map_location='cpu', weights_only=False)
    if (cp.get('format') != FORMAT or cp['binding'] != report['binding'] or cp['config'] != report['config']
            or cp['accepted_repair_steps'] != report['accepted_repair_steps']):
        raise ValueError('Repair checkpoint/config/step mismatch')
    rq = build_model(cp['config']).to(device)
    rq.load_state_dict(cp['state_dict'], strict=True)
    codes = h.catalog_codes(rq, ex)
    if read_jsonl(directory/'semantic_assignments.jsonl') != [dict(**r, semantic_code=c) for r, c in zip(ex.feature_rows, codes.tolist())]:
        raise ValueError('Exported repair semantic codes mismatch')
    ledger = versioned_suffixes(ex.previous, [r['raw_item_id'] for r in ex.feature_rows], codes.tolist())
    _, mapping = maps_from_ledger(ledger)
    if ledger != read_json(directory/'new_allocation_registry.json') or mapping != read_json(directory/'new_item2tokens.json'):
        raise ValueError('Repair full SID allocation mismatch')
    control = {raw: [*mapping[raw][:3], tokens[3]] for raw, tokens in ex.tokenizer.item2tokens.items()}
    variants = {split: dict(old=arrays, new_full_sid=reencode_history(arrays, ex.rows[split], mapping),
        new_semantics_old_suffix=reencode_history(arrays, ex.rows[split], control)) for split, arrays in ex.arrays.items()}
    teacher = h.frozen_teacher(ex, device)
    out.mkdir(parents=True)
    groups, contrasts = {}, {}
    tasks = [(name, 'base_train', chosen) for name, chosen in ex.cohorts.items()]
    tasks.append(('base_validation', 'base_val', list(range(len(ex.arrays['base_val']['labels'])))))
    for name, split, chosen in tasks:
        records = paired_losses(teacher, variants[split], ex.rows[split], chosen, args.batch_size, device, name)
        with (out/(name+'_paired_losses.jsonl')).open('w') as f:
            for row in records:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        plain = read_jsonl(ex.e1_dir/(name+'_paired_losses.jsonl'))
        groups[name], contrasts[name] = {}, {}
        for condition in ('new_full_sid', 'new_semantics_old_suffix'):
            repetitions = ex.e1['groups'][name][condition]['cluster_bootstrap_replicates']
            groups[name][condition] = aggregate(records, condition, repetitions, ex.e1['cohort_seed'])
            contrasts[name][condition] = paired_contrast(records, plain, condition, repetitions, ex.e1['cohort_seed'])
        print(name+': '+json.dumps(contrasts[name]), flush=True)
    ranking = None
    if args.ranking:
        if ex.e1.get('base_validation_ranking') is None:
            raise ValueError('Bound Plain ranking missing')
        wrapper = SimpleNamespace(model=teacher, config=ex.teacher['config'], evaluator=Evaluator(ex.teacher['config'], ex.tokenizer))
        repaired_rank = rank_evaluate(wrapper, ex.tokenizer, variants['base_val']['new_full_sid'], ex.audits['base_val'],
            list(range(len(ex.audits['base_val']))), device, ex.teacher['config']['eval_batch_size'])
        reference = ex.e1['base_validation_ranking']
        ranking = dict(old=reference['old'], plain=reference['new_full_sid'], repaired=repaired_rank,
            repair_minus_plain_full_denominator={k: repaired_rank['full_denominator'][k]-v
                for k, v in reference['new_full_sid']['full_denominator'].items()},
            reference_rankings_reused_from_hash_bound_e1=True)
        write_json(out/'base_validation_ranking.json', ranking)
    h.check_teacher(teacher, ex)
    result = dict(stage='hard_checked_codebook_repair_evaluation', execution_status='completed', binding=report['binding'],
        repair_dir=str(directory), e1_dir=str(ex.e1_dir.resolve()), repair_checkpoint_sha256=sha256(directory/'last.pth'),
        selected_tiger_sha256=ex.e1['selected_tiger_sha256'], plain_rqvae_sha256=ex.e1['update_rqvae_sha256'],
        groups=groups, paired_repair_vs_plain=contrasts, base_validation_ranking=ranking,
        rqvae_reconstruction=dict(plain=ex.plain['final_train_catalog_metrics'], repair=report['final_train_catalog_metrics']),
        primary_cohort='anchor_holdout; unused for repair selection; seen by Base TIGER training',
        accepted_repair_steps=report['accepted_repair_steps'], teacher_frozen=True,
        labels='original full SID plus EOS', ranking_catalog='base_train; original full SIDs',
        compute_comparison=report['compute_comparison'], recommendation_training_status='not_run', future_test_status='not_run',
        device=str(device), runtime_seconds=time.perf_counter()-started,
        artifact_files_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'summary.json', result)
    print('Repair evaluation completed. Use the bundle show_repair.py for compact results.', flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('repair')
    train.add_argument('--e1-dir', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--device', default='cuda')
    train.add_argument('--max-rounds', type=int, default=10)
    train.add_argument('--gradient-queries', type=int, default=512)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--relative-steps', type=float, nargs='+', default=[1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    train.add_argument('--mse-slack', type=float, default=0.01)
    train.add_argument('--min-decrease', type=float, default=1e-6)
    test = sub.add_parser('evaluate')
    test.add_argument('--e1-dir', required=True)
    test.add_argument('--repair-dir', required=True)
    test.add_argument('--output', required=True)
    test.add_argument('--device', default='cuda')
    test.add_argument('--batch-size', type=int, default=32)
    test.add_argument('--ranking', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    repair(args) if args.command == 'repair' else evaluate_repair(args)
