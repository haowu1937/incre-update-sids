"""Bounded replay of a failed full-SID update; never writes input artifacts."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_paths(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if '=' not in line:
            raise ValueError('Malformed paths.txt')
        key, value = line.split('=', 1)
        if key in values:
            raise ValueError('Duplicate paths.txt key: '+key)
        values[key] = value
    return values


def read_trace(path):
    records = {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        step = record['step']
        if type(step) is not int or step < 1 or step in records:
            raise ValueError('Invalid or duplicate training step')
        if records and step != max(records)+1:
            raise ValueError('Noncontiguous training trace')
        records[step] = record
    if not records:
        raise ValueError('Missing successful training records')
    return records


def replay_target(saved_step, records, max_steps=50):
    last = max(records)
    target = last+1
    if (not 1 <= max_steps <= 50 or saved_step not in records or saved_step > last
            or not 1 <= target-saved_step <= max_steps):
        raise ValueError('Checkpoint/trace replay window is invalid or exceeds its bound')
    return target


class TraceMismatch(ValueError):
    pass


def compare(expected, actual, path='metrics'):
    """Compare saved observables, not an unrecorded bitwise weight trajectory."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not expected.keys() <= actual.keys():
            raise TraceMismatch(path+': missing keys')
        for key, value in expected.items():
            compare(value, actual[key], path+'.'+key)
    elif isinstance(expected, float):
        if (not isinstance(actual, (int, float)) or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-7)):
            raise TraceMismatch(f'{path}: saved={expected!r}, replay={actual!r}')
    elif type(expected) is not type(actual) or expected != actual:
        raise TraceMismatch(f'{path}: saved={expected!r}, replay={actual!r}')


def assignment_statistics(codes, capacities):
    if not codes or any(len(row) != len(capacities) for row in codes):
        raise ValueError('Invalid assignment matrix')
    for row in codes:
        if any(type(v) is not int or not 0 <= v < cap for v, cap in zip(row, capacities)):
            raise ValueError('Assignment out of bounds')
    buckets = Counter(map(tuple, codes))
    return dict(items=len(codes), unique_semantic_codes=len(buckets),
        maximum_current_collision_bucket=max(buckets.values()),
        largest_bucket_fraction=max(buckets.values())/len(codes),
        all_items_share_one_full_semantic_code=len(buckets) == 1,
        codebooks=[dict(capacity=cap, used=len(set(row[i] for row in codes)))
                   for i, cap in enumerate(capacities)],
        largest_prefix_buckets=[max(Counter(tuple(row[:i]) for row in codes).values())
                                for i in range(1, len(capacities)+1)])


def ensure_separate(output, inputs):
    for value in inputs:
        source = Path(value).resolve()
        if output == source or source in output.parents:
            raise ValueError('Diagnostic output must be outside all input directories')
    if output.exists():
        raise FileExistsError('Refusing to overwrite diagnostic output: '+str(output))


def run(args):
    # ML imports are deferred so trace/bounds/statistics checks work without Torch.
    import torch
    from gensid import historical_incremental_content as engine
    started = time.perf_counter()
    run_dir = Path(args.run_dir).resolve()
    paths = read_paths(run_dir/'paths.txt')
    history = Path(paths['HISTORY_DIR']).resolve()
    e1 = Path(paths['E1_DIR']).resolve()
    output = Path(args.output).resolve()
    ensure_separate(output, [run_dir, history, e1])
    records = read_trace(history/'metrics.jsonl')
    checkpoint_path = history/'last.pth'
    checkpoint_digest = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'historical_update_rqvae_v1':
        raise ValueError('Expected historical Update checkpoint')
    config = checkpoint['config']
    if config['history_input_condition'] != 'new_full_sid' or config['compatibility_weight'] <= 0:
        raise ValueError('Expected positive-weight full SID historical objective')
    if config['device'] != str(engine.device_from(args.device)):
        raise ValueError('Replay must use the original configured device')
    saved_step = checkpoint['sampler_state']['step']
    target = replay_target(saved_step, records, args.max_replay_steps)
    original = json.loads((history/'config.json').read_text())
    if (original['config'] != config or original['binding'] != checkpoint['binding']
            or target > original['command']['max_steps']):
        raise ValueError('Checkpoint/config/budget mismatch')
    if Path(original['command']['e1_dir']).resolve() != e1:
        raise ValueError('Original E1 directory differs from paths.txt')
    ex = engine.experiment(SimpleNamespace(e1_dir=str(e1)))
    if ex.binding != checkpoint['binding']:
        raise ValueError('Original data/source/environment binding mismatch')
    observed_paths = [checkpoint_path, history/'metrics.jsonl', history/'config.json',
        run_dir/'paths.txt', run_dir/'commit.txt', e1/'summary.json',
        Path(ex.plain['base_rqvae_dir'])/'last.pth',
        Path(ex.e1['update_rqvae_dir'])/'last.pth', Path(ex.e1['tiger_dir'])/'best.pth']
    observed_paths += [Path(__file__).resolve().parent/name for name in engine.SOURCES]
    before = {str(p): digest(p) for p in observed_paths}
    if before[str(checkpoint_path)] != checkpoint_digest:
        raise ValueError('Checkpoint changed while preparing replay')
    output.mkdir(parents=True)
    report = dict(stage='historical_full_sid_capacity_diagnostic', execution_status='started',
        original_run=str(run_dir), original_run_commit=(run_dir/'commit.txt').read_text().strip(),
        original_checkpoint_step=saved_step, original_last_completed_step=max(records),
        expected_failed_attempt=target, max_replay_steps=args.max_replay_steps,
        original_binding=checkpoint['binding'], original_config=config,
        observed_input_sha256=before, diagnostic_source_sha256=digest(__file__),
        current_attempt=None, compared_steps=[], original_run_continued=False,
        checkpoint_written=False, holdout_evaluated=False, future_evaluated=False,
        trace_comparison=dict(relative_tolerance=1e-6, absolute_tolerance=1e-7,
            evidence='logged observables only; no per-step weight references exist'))
    write_json(output/'request.json', report)
    trainer = None
    original_audit = engine.suffix_capacity_audit
    progress = dict(step=saved_step)
    try:
        engine.seed_all(config['seed'])
        model = engine.build_model(config).to(args.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        teacher = engine.frozen_teacher(ex, args.device)
        trainer = engine.build_trainer(model, config, output/'trainer')
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        penalty = engine.HistoryPenalty(model, teacher, ex, config, checkpoint['distance_scales'],
                                       checkpoint['anchor_sampler_state'])

        def audit(previous, raw_items, codes):
            capacity = original_audit(previous, raw_items, codes)
            event = dict(attempted_update_step=progress['step']+1,
                parameter_optimizer_steps=progress['step'], capacity=capacity,
                assignments=assignment_statistics(codes, config['model']['num_emb_list']))
            report['current_attempt'] = event
            if progress['step'] == saved_step:
                report['saved_checkpoint_assignments'] = event
            with (output/'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            return capacity  # The unchanged allocator still raises on exhaustion.

        def on_step(state, metrics):
            step = state['step']
            progress['step'] = step
            metrics.update(penalty.last_metrics)
            metrics['weighted_history_penalty'] = config['compatibility_weight']*metrics['history_penalty']
            metrics['rq_loss'] = metrics['total_loss']-metrics['weighted_history_penalty']
            metrics['gradient_norms'] = {name: float(torch.stack([p.grad.detach().square().sum()
                for p in module.parameters() if p.grad is not None]).sum().sqrt())
                for name, module in [('encoder', model.encoder), ('decoder', model.decoder), ('codebooks', model.rq)]}
            if step in records:
                compare(records[step], metrics)
                report['compared_steps'].append(step)
                report['last_matched_attempt'] = copy.deepcopy(report['current_attempt'])
            with (output/'replay.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            print(f'Replay {step}/{target}; matched_original={step in records}', flush=True)

        engine.suffix_capacity_audit = audit
        engine.restore_rng(checkpoint['rng_state'])  # After old-NLL cache initialization, as in resume.
        try:
            engine.compatible_fit(trainer, ex.features, target, config,
                                  copy.deepcopy(checkpoint['sampler_state']), penalty, on_step)
            report.update(execution_status='completed', outcome='overflow_not_reproduced')
        except ValueError as error:
            event = report['current_attempt']
            if (str(error) != 'New-version collision suffix capacity exhausted'
                    or not event or event['capacity']['full_sid_allocation_feasible']):
                raise
            matched = report['compared_steps'] == list(range(saved_step+1, target))
            reproduced = event['attempted_update_step'] == target and matched
            report.update(execution_status='completed' if reproduced else 'blocked',
                outcome='overflow_reproduced' if reproduced else 'overflow_at_different_step',
                allocator_error=str(error))
        engine.check_teacher(teacher, ex)
    except Exception as error:
        report.update(execution_status='blocked', outcome='trace_mismatch' if isinstance(error, TraceMismatch)
                      else 'diagnostic_error', error_type=type(error).__name__, error=str(error))
        (output/'error.txt').write_text(traceback.format_exc())
    finally:
        engine.suffix_capacity_audit = original_audit
        if trainer is not None:
            trainer.tb_writer.close()
        report['observed_inputs_unchanged'] = all(Path(p).is_file() and digest(p) == h
                                                 for p, h in before.items())
        if not report['observed_inputs_unchanged']:
            report.update(execution_status='blocked', outcome='input_changed_during_replay')
        report['runtime_seconds'] = time.perf_counter()-started
        write_json(output/'summary.json', report)
    compact = {key: value for key, value in report.items()
               if key not in ('original_binding', 'original_config', 'observed_input_sha256')}
    write_json(output/'compact.json', compact)
    print(json.dumps(compact, indent=2, allow_nan=False), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-replay-steps', type=int, choices=range(1, 51), default=50)
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    raise SystemExit(0 if result['execution_status'] == 'completed' else 1)
