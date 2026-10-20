"""Paired scalar History-aware retuning; fixed Base, full SIDs and held-out evaluation."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
FORMAT = 'retuned_scalar_history_v1'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name+'.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(pending, path)


def settings(args):
    for name in ('lr', 'history_weight', 'temperature', 'max_grad_norm'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError('Expected finite positive '+name)
    for name in ('max_steps', 'anchor_batch_size', 'eval_batch_size', 'log_every', 'save_every'):
        if getattr(args, name) < 1:
            raise ValueError('Expected positive '+name)
    return {name: getattr(args, name) for name in ('lr', 'history_weight', 'temperature',
        'max_grad_norm', 'max_steps', 'anchor_batch_size', 'eval_batch_size',
        'log_every', 'save_every', 'device', 'ranking')}


def bootstrap_differences(plain, historical, repeats, seed):
    """Paired user-cluster intervals for mean NLL difference AND H difference."""
    import numpy as np
    if repeats < 1:
        raise ValueError('Expected positive bootstrap repeats')
    if len(plain) != len(historical) or not plain:
        raise ValueError('Paired evaluation query count mismatch')
    clusters = {}
    values = []
    for p, h in zip(plain, historical):
        if any(p[k] != h[k] for k in ('sample_id', 'user', 'target_labels', 'source_row')):
            raise ValueError('Paired historical queries/labels differ')
        if abs(p['old_sequence_nll']-h['old_sequence_nll']) > 1e-5:
            raise ValueError('Frozen old reference differs across arms')
        old = p['old_sequence_nll']
        pn, hn = p['new_full_sid_sequence_nll'], h['new_full_sid_sequence_nll']
        if not all(math.isfinite(v) for v in (old, pn, hn, h['old_sequence_nll'])):
            raise ValueError('Nonfinite paired NLL')
        dn, dh = hn-pn, max(hn-old, 0)-max(pn-old, 0)
        values.append((dn, dh))
        a, b, n = clusters.get(p['user'], (0., 0., 0))
        clusters[p['user']] = (a+dn, b+dh, n+1)
    matrix = np.asarray(list(clusters.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(repeats):
        sample = matrix[rng.integers(0, len(matrix), len(matrix))].sum(0)
        bootstrap.append(sample[:2]/sample[2])
    ci = np.quantile(bootstrap, [0.025, 0.975], axis=0) if len(matrix) > 1 else None
    mean = np.asarray(values).mean(0)
    return dict(queries=len(values), users=len(matrix),
        mean_historical_minus_plain_sequence_nll=float(mean[0]),
        sequence_nll_difference_ci95=ci[:, 0].tolist() if ci is not None else None,
        mean_historical_minus_plain_positive_damage=float(mean[1]),
        positive_damage_difference_ci95=ci[:, 1].tolist() if ci is not None else None,
        bootstrap_replicates=repeats, bootstrap_unit='user; query-weighted resampled clusters')


def atomic_checkpoint(torch, path, value):
    pending = Path(str(path)+'.tmp')
    torch.save(value, pending)
    os.replace(pending, path)


def gradients(torch, model, rq_loss, historical_loss, weight):
    params = list(model.named_parameters())
    tensors = [p for _, p in params]
    rq = torch.autograd.grad(rq_loss, tensors, retain_graph=True, allow_unused=True)
    hg = (torch.autograd.grad(historical_loss, tensors, retain_graph=True, allow_unused=True)
          if historical_loss is not None and historical_loss.requires_grad else [None]*len(tensors))
    result = {}
    for name, prefix in (('encoder', 'encoder.'), ('decoder', 'decoder.'), ('codebooks', 'rq.')):
        triples = [(a, b) for (key, _), a, b in zip(params, rq, hg) if key.startswith(prefix)]
        rn = math.sqrt(sum(float(a.detach().double().square().sum()) for a, _ in triples if a is not None))
        hn = math.sqrt(sum(float(b.detach().double().square().sum()) for _, b in triples if b is not None))
        dot = sum(float((a.detach().double()*b.detach().double()).sum())
                  for a, b in triples if a is not None and b is not None)
        result[name] = dict(rq_gradient_norm=rn, history_gradient_norm=hn,
            weighted_history_gradient_norm=weight*hn,
            weighted_history_over_rq=(weight*hn/rn if rn > 0 else (0. if weight*hn == 0 else None)),
            cosine=(max(-1., min(1., dot/(rn*hn))) if rn > 0 and hn > 0 else None))
    return result


def restore_valid(h, model, trainer, penalty, current, checkpoint):
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    current.clear()
    current.update(copy.deepcopy(checkpoint['sampler_state']))
    if checkpoint['anchor_sampler_state'] is not None:
        penalty.state = copy.deepcopy(checkpoint['anchor_sampler_state'])
        penalty.generator.set_state(penalty.state['generator_state'])
    h.restore_rng(checkpoint['rng_state'])


def train_arm(h, torch, ex, arm, directory, common, binding, resume):
    config = copy.deepcopy(ex.plain['config'])
    config.update(device=common['device'], lr=common['lr'],
        compatibility_weight=0. if arm == 'plain' else common['history_weight'],
        history_input_condition='new_full_sid', temperature=common['temperature'],
        anchor_batch_size=common['anchor_batch_size'], anchor_sampler_seed=(config['seed']+1) % 2**32,
        max_grad_norm=common['max_grad_norm'], recipe='scalar_retuned_v1')
    if (directory/'summary.json').exists():
        if not resume:
            raise FileExistsError(str(directory))
        saved = json.loads((directory/'summary.json').read_text())
        if (saved['execution_status'] != 'completed' or saved['config'] != config
                or saved['binding'] != binding or saved['update_optimizer_steps'] != common['max_steps']):
            raise ValueError('Completed arm binding/config mismatch')
        for name, expected in saved['artifact_files_sha256'].items():
            if sha256(directory/name) != expected:
                raise ValueError('Completed arm artifact changed: '+name)
        return saved
    if (directory/'failure.json').exists():
        raise ValueError('A failed attempt is preserved here; changed settings require a fresh run')
    directory.mkdir(parents=True, exist_ok=resume)
    h.seed_all(config['seed'])
    model = h.build_model(config).to(config['device'])
    model.load_state_dict(ex.parent['state_dict'], strict=True)
    if not all(torch.equal(v.cpu(), ex.parent['state_dict'][key]) for key, v in model.state_dict().items()):
        raise ValueError('Base weight warm start mismatch')
    initial, initial_groups, initial_codes = h.evaluate_groups(model, ex.features, ex.groups,
        ex.old_batch, config['added_item_assignment_eval_batch_size'])
    if not __import__('numpy').array_equal(initial_codes[:ex.groups['base_train'][1]], ex.old_export):
        raise ValueError('Initial Base assignment mismatch')
    scales = h.distance_scales(model, ex.features[:ex.groups['base_train'][1]], ex.old_batch)
    teacher = h.frozen_teacher(ex, config['device'])
    trainer = h.build_trainer(model, config, directory)
    current = dict(step=0, cursor=0, order=None, epoch=0, processed_examples=0)
    prior = None
    if (directory/'last.pth').exists():
        if not resume:
            raise FileExistsError('Checkpoint already exists')
        prior = torch.load(directory/'last.pth', map_location='cpu', weights_only=False)
        if (prior.get('format') != FORMAT or prior['config'] != config or prior['binding'] != binding
                or prior['arm'] != arm or prior['distance_scales'] != scales):
            raise ValueError('Retuned checkpoint data/source/config mismatch')
        model.load_state_dict(prior['state_dict'], strict=True)
        trainer.optimizer.load_state_dict(prior['optimizer'])
        current = copy.deepcopy(prior['sampler_state'])
    penalty = h.HistoryPenalty(model, teacher, ex, config, scales,
                              prior['anchor_sampler_state'] if prior and config['compatibility_weight'] > 0 else None)
    if prior:
        h.restore_rng(prior['rng_state'])
    raw_items = [row['raw_item_id'] for row in ex.feature_rows]
    first_step = current['step']
    attempt = Path(tempfile.mkdtemp(prefix='attempt_', dir=directory))
    started = time.perf_counter()
    state_for_callback = dict(step=first_step)
    latest = {}

    def snapshot(state):
        return dict(format=FORMAT, arm=arm, config=config, binding=binding,
            state_dict=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
            sampler_state=copy.deepcopy(state),
            anchor_sampler_state=penalty.snapshot() if config['compatibility_weight'] > 0 else None,
            rng_state=h.rng_state(), distance_scales=scales)

    initial_capacity = h.suffix_capacity_audit(ex.previous, raw_items, h.catalog_codes(model, ex).tolist())
    if not initial_capacity['full_sid_allocation_feasible']:
        raise ValueError('Starting checkpoint is not full-SID feasible')
    last_valid = snapshot(current)
    atomic_checkpoint(torch, directory/'last.pth', last_valid)
    original_loss, original_step = model.compute_loss, trainer.optimizer.step

    def compute(*args, **kwargs):
        rq, aux, recon, quant = original_loss(*args, **kwargs)
        historical = penalty() if config['compatibility_weight'] > 0 else None
        step = state_for_callback['step']+1
        latest.clear()
        latest.update(rq_loss=float(rq.detach()),
            history_penalty=float(historical.detach()) if historical is not None else 0., anchor_queries=0)
        if historical is not None:
            latest.update(penalty.last_metrics)
        if step == first_step+1 or step % common['log_every'] == 0:
            latest['gradient_components'] = gradients(torch, model, rq, historical, config['compatibility_weight'])
        return rq+(config['compatibility_weight']*historical if historical is not None else 0.), aux, recon, quant

    def optimizer_step(*args, **kwargs):
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), common['max_grad_norm'], error_if_nonfinite=True)
        latest['global_gradient_norm_before_clip'] = float(norm)
        latest['gradient_clipped'] = float(norm) > common['max_grad_norm']
        return original_step(*args, **kwargs)

    def on_step(state, metrics):
        nonlocal last_valid
        state_for_callback['step'] = state['step']
        codes = h.catalog_codes(model, ex)
        capacity = h.suffix_capacity_audit(ex.previous, raw_items, codes.tolist())
        collisions = h.collision_stats(codes, config['model']['num_emb_list'])
        if not capacity['full_sid_allocation_feasible']:
            candidate = snapshot(state)
            candidate.update(format='retuned_scalar_rejected_v1', full_sid_export_ready=False)
            atomic_checkpoint(torch, directory/'rejected_candidate.pth', candidate)
            failure = dict(reason='post_update_suffix_capacity_exhausted', rejected_optimizer_step=state['step'],
                last_valid_optimizer_step=last_valid['sampler_state']['step'], capacity=capacity, collisions=collisions,
                full_sid_export_ready=False, automatic_retry=False)
            restore_valid(h, model, trainer, penalty, state, last_valid)
            atomic_checkpoint(torch, directory/'last.pth', last_valid)
            failure['rollback_completed'] = True
            write_json(directory/'failure.json', failure)
            raise ValueError('Post-update full SID capacity exhausted; last valid state restored')
        metrics.update(latest, weighted_history_penalty=config['compatibility_weight']*latest['history_penalty'],
            post_update_capacity=capacity, post_update_collisions=collisions)
        # Pre-update penalty/gradient metrics and post-update assignment metrics are explicitly separate.
        with (attempt/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
        last_valid = snapshot(state)
        if state['step'] % common['save_every'] == 0 or state['step'] == common['max_steps']:
            atomic_checkpoint(torch, directory/'last.pth', last_valid)
        if state['step'] == first_step+1 or state['step'] % common['log_every'] == 0:
            print(arm+' '+json.dumps(metrics, allow_nan=False), flush=True)

    failure = None
    model.compute_loss, trainer.optimizer.step = compute, optimizer_step
    try:
        current = trainer.fit_steps(ex.features, torch.arange(len(ex.features)), common['max_steps'],
            config['batch_size'], config['seed'], current, on_step)
        atomic_checkpoint(torch, directory/'last.pth', last_valid)
    except BaseException as error:
        failure = dict(reason=str(error), error_type=type(error).__name__,
            last_valid_optimizer_step=last_valid['sampler_state']['step'], resumable=isinstance(error, KeyboardInterrupt))
        restore_valid(h, model, trainer, penalty, current, last_valid)
        atomic_checkpoint(torch, directory/'last.pth', last_valid)
        (attempt/'error.txt').write_text(traceback.format_exc())
        if not failure['resumable'] and not (directory/'failure.json').exists():
            write_json(directory/'failure.json', dict(failure, rollback_completed=True))
    finally:
        model.compute_loss, trainer.optimizer.step = original_loss, original_step
        trainer.tb_writer.close()
        write_json(attempt/'summary.json', dict(execution_status='failed' if failure else 'completed',
            first_optimizer_step=first_step, last_valid_optimizer_step=last_valid['sampler_state']['step'],
            runtime_seconds=time.perf_counter()-started, failure=failure))
    if failure:
        return dict(execution_status='interrupted' if failure['resumable'] else 'failed', arm=arm,
            failure=json.loads((directory/'failure.json').read_text()) if (directory/'failure.json').exists() else failure)
    h.check_teacher(teacher, ex)
    final, groups, codes = h.evaluate_groups(model, ex.features, ex.groups, ex.old_batch,
                                            config['added_item_assignment_eval_batch_size'])
    reload_model = h.build_model(config).to(config['device'])
    reload_model.load_state_dict(torch.load(directory/'last.pth', map_location='cpu', weights_only=False)['state_dict'], strict=True)
    if not __import__('numpy').array_equal(codes, h.catalog_codes(reload_model, ex)):
        raise ValueError('Saved checkpoint assignments differ from final model')
    del reload_model
    ledger = h.versioned_suffixes(ex.previous, raw_items, codes.tolist())
    _, mapping = h.maps_from_ledger(ledger)
    write_json(directory/'allocation_registry.json', ledger)
    write_json(directory/'item2tokens.json', mapping)
    with (directory/'semantic_assignments.jsonl').open('w') as stream:
        for row, code in zip(ex.feature_rows, codes.tolist()):
            stream.write(json.dumps(dict(**row, semantic_code=code))+'\n')
    report = dict(execution_status='completed', arm=arm, config=config, binding=binding,
        update_optimizer_steps=current['step'], processed_examples=current['processed_examples'],
        warm_start_all_weights_equal=True, initial_rq_metrics=initial, initial_group_metrics=initial_groups,
        final_rq_metrics=final, final_group_metrics=groups,
        collision_statistics=h.collision_stats(codes, config['model']['num_emb_list']),
        base_semantic_churn=h.churn(ex.old_export, codes[:ex.groups['base_train'][1]]),
        teacher_unchanged=True, checkpoint_reload_assignments_equal=True,
        historical_penalty_training_cohort='anchor_train only' if arm == 'historical' else 'none',
        anchor_queries_processed=penalty.state['processed_queries'] if arm == 'historical' else 0,
        checkpoint_selection='last fixed-budget step; no holdout selection',
        training_runtime_seconds=sum(json.loads(path.read_text())['runtime_seconds'] for path in directory.glob('attempt_*/summary.json')),
        runtime_scope='optimizer loop, gradient probes, capacity checks and checkpoints; excludes initialization and final evaluation',
        artifact_files_sha256={name: sha256(directory/name) for name in ('last.pth', 'allocation_registry.json',
                                  'item2tokens.json', 'semantic_assignments.jsonl')})
    write_json(directory/'summary.json', report)
    return report


def check_pair(torch, output, training):
    a, b = [torch.load(output/name/'last.pth', map_location='cpu', weights_only=False)
            for name in ('plain', 'historical')]
    ca, cb = copy.deepcopy(a['config']), copy.deepcopy(b['config'])
    if ca['compatibility_weight'] != 0 or cb['compatibility_weight'] <= 0:
        raise ValueError('Expected zero-weight Plain and positive-weight Historical')
    ca.pop('compatibility_weight'); cb.pop('compatibility_weight')
    if (ca != cb or a['binding'] != b['binding'] or a['optimizer']['param_groups'] != b['optimizer']['param_groups']
            or training['plain']['initial_rq_metrics'] != training['historical']['initial_rq_metrics']):
        raise ValueError('Retuned arm settings/initialization/optimizer mismatch')
    sa, sb = a['sampler_state'], b['sampler_state']
    if any(sa[k] != sb[k] for k in ('step', 'cursor', 'epoch', 'processed_examples')):
        raise ValueError('Retuned content sampler endpoint differs')
    if not all(torch.equal(sa[k], sb[k]) for k in ('order', 'generator_state')):
        raise ValueError('Retuned content sampler order/generator differs')
    return dict(execution_status='passed', same_base=True, same_content_recipe=True,
        same_optimizer_and_clipping=True, same_content_sampler_endpoint=True,
        same_update_steps=sa['step'], processed_examples=sa['processed_examples'], same_compute=False)


def evaluate(h, torch, ex, output, training, common, binding):
    directory = output/'evaluation'
    if (directory/'summary.json').exists():
        report = json.loads((directory/'summary.json').read_text())
        if report['binding'] != binding:
            raise ValueError('Evaluation binding differs')
        for name, expected in report['artifact_files_sha256'].items():
            if sha256(directory/name) != expected:
                raise ValueError('Evaluation artifact changed: '+name)
        return report
    directory.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='attempt_', dir=directory))
    teacher = h.frozen_teacher(ex, common['device'])
    groups, records_by_arm, rankings = {}, {}, {}
    if common['ranking']:
        config = ex.teacher['config']
        wrapper = SimpleNamespace(model=teacher, config=config, evaluator=h.Evaluator(config, ex.tokenizer))
        rankings['old_base'] = h.rank_evaluate(wrapper, ex.tokenizer, ex.arrays['base_val'],
            ex.audits['base_val'], list(range(len(ex.audits['base_val']))), common['device'], config['eval_batch_size'])
        for denominator in ('full_denominator', 'conditional_on_servable'):
            if any(abs(rankings['old_base'][denominator][k]-v) > 1e-8
                   for k, v in ex.tiger_summary['full_base_validation'][denominator].items()):
                raise ValueError('Frozen Base ranking was not reproduced')
    for arm in ('plain', 'historical'):
        mapping = json.loads((output/arm/'item2tokens.json').read_text())
        control = {raw: [*mapping[raw][:3], tokens[3]] for raw, tokens in ex.tokenizer.item2tokens.items()}
        variants = {split: dict(old=arrays,
            new_full_sid=h.reencode_history(arrays, ex.rows[split], mapping),
            new_semantics_old_suffix=h.reencode_history(arrays, ex.rows[split], control))
            for split, arrays in ex.arrays.items()}
        groups[arm], records_by_arm[arm] = {}, {}
        tasks = [(name, 'base_train', indices) for name, indices in ex.cohorts.items()]
        tasks.append(('base_validation', 'base_val', list(range(len(ex.arrays['base_val']['labels'])))))
        for name, split, indices in tasks:
            records = h.paired_losses(teacher, variants[split], ex.rows[split], indices,
                                      common['eval_batch_size'], common['device'], arm+' '+name)
            records_by_arm[arm][name] = records
            with (attempt/(arm+'_'+name+'.jsonl')).open('w') as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
            groups[arm][name] = {condition: h.aggregate(records, condition,
                ex.e1['groups'][name][condition]['cluster_bootstrap_replicates'], ex.e1['cohort_seed'])
                for condition in ('new_full_sid', 'new_semantics_old_suffix')}
        if common['ranking']:
            config = ex.teacher['config']
            wrapper = SimpleNamespace(model=teacher, config=config, evaluator=h.Evaluator(config, ex.tokenizer))
            ranking = h.rank_evaluate(wrapper, ex.tokenizer, variants['base_val']['new_full_sid'],
                ex.audits['base_val'], list(range(len(ex.audits['base_val']))), common['device'], config['eval_batch_size'])
            rankings[arm] = ranking
    h.check_teacher(teacher, ex)
    if common['ranking']:
        rankings['historical_minus_plain_full_denominator'] = {
            k: rankings['historical']['full_denominator'][k]-v
            for k, v in rankings['plain']['full_denominator'].items()}
    differences = {name: bootstrap_differences(records_by_arm['plain'][name], records_by_arm['historical'][name],
        ex.e1['groups'][name]['new_full_sid']['cluster_bootstrap_replicates'], ex.e1['cohort_seed'])
        for name in records_by_arm['plain']}
    report = dict(execution_status='completed', binding=binding, groups=groups,
        historical_minus_retuned_plain=differences, base_validation_ranking=rankings or None,
        primary_cohort='anchor_holdout; not used for training or parameter selection',
        evaluation_condition='new full history SID; fixed original target, mask and catalog',
        recommendation_training_status='not_run', future_test_status='not_run',
        artifact_files_sha256={str(p.relative_to(directory)): sha256(p) for p in attempt.glob('*.jsonl')})
    write_json(directory/'summary.json', report)
    return report


def run(args):
    import torch
    from gensid import historical_incremental_content as h
    common = settings(args)
    ex = h.experiment(SimpleNamespace(e1_dir=args.e1_dir))
    if common['max_steps'] > ex.plain['update_optimizer_steps']:
        raise ValueError('Retuned step budget must not exceed the measured Plain budget')
    if str(h.device_from(common['device'])) != common['device']:
        raise ValueError('Use an explicit device such as cuda:0')
    output = Path(args.output).resolve()
    for path in (Path(args.e1_dir).resolve(), Path(ex.plain['base_rqvae_dir']).resolve(), Path(ex.e1['update_rqvae_dir']).resolve()):
        if path == output or path in output.parents:
            raise ValueError('New outputs must be outside original inputs')
    binding = dict(original_context=ex.binding, retuned_source_sha256=sha256(__file__))
    request = dict(format=FORMAT, settings=common, binding=binding)
    if output.exists():
        if not args.resume or json.loads((output/'request.json').read_text()) != request:
            raise ValueError('Explicit resume with unchanged request/source is required')
    else:
        output.mkdir(parents=True)
        write_json(output/'request.json', request)
    protected = [Path(ex.plain['base_rqvae_dir'])/'last.pth', Path(ex.e1['update_rqvae_dir'])/'last.pth',
                 Path(ex.e1['tiger_dir'])/'best.pth', Path(args.e1_dir)/'summary.json']
    before = {str(p): sha256(p) for p in protected}
    if common['device'].startswith('cuda:'):
        torch.cuda.reset_peak_memory_stats(torch.device(common['device']))
    started = time.perf_counter()
    training = {}
    try:
        for arm in ('plain', 'historical'):
            training[arm] = train_arm(h, torch, ex, arm, output/arm, common, binding, args.resume)
            if training[arm]['execution_status'] != 'completed':
                report = dict(execution_status='failed', settings=common, training=training,
                    evaluation_status='not_run; requires both complete paired arms', future_test_status='not_run')
                break
        else:
            pair = check_pair(torch, output, training)
            write_json(output/'pair_checks.json', pair)
            evaluation = evaluate(h, torch, ex, output, training, common, binding)
            report = dict(execution_status='completed', settings=common, training=training,
                          pair_checks=pair, evaluation=evaluation, future_test_status='not_run')
    except Exception as error:
        (output/'error.txt').write_text(traceback.format_exc())
        report = dict(execution_status='failed', settings=common, training=training,
                      error_type=type(error).__name__, error=str(error))
    report.update(stage='paired_retuned_scalar_history', original_inputs_unchanged=all(sha256(p) == value for p, value in before.items()),
        protected_input_sha256=before, binding=binding, runtime_seconds=time.perf_counter()-started)
    report['peak_cuda_allocated_bytes'] = (torch.cuda.max_memory_allocated(torch.device(common['device']))
        if common['device'].startswith('cuda:') else None)
    report['peak_cuda_reserved_bytes'] = (torch.cuda.max_memory_reserved(torch.device(common['device']))
        if common['device'].startswith('cuda:') else None)
    if not report['original_inputs_unchanged']:
        report['execution_status'] = 'failed'
    write_json(output/'summary.json', report)
    compact = copy.deepcopy(report)
    compact.pop('binding'); compact.pop('protected_input_sha256')
    for arm in compact['training'].values():
        for key in ('binding', 'artifact_files_sha256'):
            arm.pop(key, None)
    if 'evaluation' in compact:
        for key in ('binding', 'artifact_files_sha256'):
            compact['evaluation'].pop(key, None)
    write_json(output/'compact.json', compact)
    print(json.dumps(compact, indent=2, allow_nan=False), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e1-dir', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--history-weight', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.5); p.add_argument('--anchor-batch-size', type=int, default=128)
    p.add_argument('--max-grad-norm', type=float, default=0.1); p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--log-every', type=int, default=25); p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--ranking', action='store_true'); p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    raise SystemExit(0 if result['execution_status'] == 'completed' else 1)
