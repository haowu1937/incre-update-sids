"""Future ranking for one immutable retuned History-aware RQ-VAE, with two TIGER states."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
STAGE = 'retuned_History_aware_Future_evaluation'
ROOT = Path(__file__).resolve().parent
SOURCES = ['evaluate_historical_future.py', 'train_retuned_historical.py',
    'genrec/future_evaluation.py', 'genrec/future_protocol.py', 'genrec/temporal_tiger.py',
    'genrec/trainer.py', 'genrec/models/TIGER/model.py', 'genrec/evaluator.py',
    'genrec/frozen_sid_tokenizer.py', 'genrec/temporal_sids.py', 'genrec/historical_damage.py',
    'gensid/historical_incremental_content.py', 'gensid/plain_incremental_content.py',
    'gensid/incremental_content.py', 'gensid/models/rqvae.py', 'gensid/trainer.py']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_args(args):
    if (min(args.max_steps, args.eval_batch_size, args.probe_size, args.log_every, args.save_every) < 1
            or not math.isfinite(args.lr) or args.lr <= 0 or args.warmup_steps < 0
            or not args.topk or min(args.topk) < 1 or len(set(args.topk)) != len(args.topk)
            or args.num_beams < max(args.topk) or args.num_beams < 2):
        raise ValueError('Invalid History-aware Future configuration')


def original_compact(report):
    """Exact projection used by the immutable retuned training driver."""
    result = copy.deepcopy(report)
    result.pop('binding'); result.pop('protected_input_sha256')
    for arm in result['training'].values():
        arm.pop('binding', None); arm.pop('artifact_files_sha256', None)
    for name in ('binding', 'artifact_files_sha256'):
        result['evaluation'].pop(name, None)
    return result


def validate_retuned_metadata(request, report, compact, pair):
    if (request.get('format') != 'retuned_scalar_history_v1'
            or report.get('stage') != 'paired_retuned_scalar_history'
            or report.get('execution_status') != 'completed'
            or report.get('original_inputs_unchanged') is not True
            or report['binding'] != request['binding'] or report['settings'] != request['settings']
            or report['pair_checks'] != pair or pair.get('execution_status') != 'passed'
            or report['evaluation'].get('execution_status') != 'completed'
            or original_compact(report) != compact):
        raise ValueError('Expected internally consistent, completed retuned paired experiment')
    steps = request['settings']['max_steps']
    if steps < 1 or pair['same_update_steps'] != steps:
        raise ValueError('Retuned update budget mismatch')
    for key in ('same_base', 'same_content_recipe', 'same_optimizer_and_clipping', 'same_content_sampler_endpoint'):
        if pair.get(key) is not True:
            raise ValueError('Retuned pair check did not pass: '+key)
    for arm in ('plain', 'historical'):
        training = report['training'][arm]
        if (training['execution_status'] != 'completed' or training['update_optimizer_steps'] != steps
                or training['processed_examples'] != pair['processed_examples']
                or training['teacher_unchanged'] is not True
                or training['checkpoint_reload_assignments_equal'] is not True):
            raise ValueError('Incomplete/unmatched retuned '+arm+' training')


def load_context(args):
    import torch
    import train_retuned_historical as retuned
    from gensid import historical_incremental_content as h
    from genrec.temporal_sids import read_json, verify_files
    ex = h.experiment(SimpleNamespace(e1_dir=args.e1_dir))
    directory = Path(args.retuned_dir).resolve()
    request, report, compact, pair = [read_json(directory/name)
        for name in ('request.json', 'summary.json', 'compact.json', 'pair_checks.json')]
    validate_retuned_metadata(request, report, compact, pair)
    if (request['binding']['original_context'] != ex.binding
            or request['binding']['retuned_source_sha256'] != digest(retuned.__file__)):
        raise ValueError('Retuned source/parent/cohort/environment binding mismatch')
    for arm in ('plain', 'historical'):
        saved = read_json(directory/arm/'summary.json')
        if saved != report['training'][arm]:
            raise ValueError('Retuned root/arm report mismatch')
        verify_files(directory/arm, saved, ['last.pth', 'allocation_registry.json',
                                         'item2tokens.json', 'semantic_assignments.jsonl'])
        checkpoint = torch.load(directory/arm/'last.pth', map_location='cpu', weights_only=False)
        if (checkpoint.get('format') != retuned.FORMAT or checkpoint['arm'] != arm
                or checkpoint['binding'] != request['binding'] or checkpoint['config'] != saved['config']
                or checkpoint['sampler_state']['step'] != saved['update_optimizer_steps']):
            raise ValueError('Retuned checkpoint identity/config/step mismatch')
    if retuned.check_pair(torch, directory, report['training']) != pair:
        raise ValueError('Actual retuned checkpoint pair does not reproduce saved pair checks')
    return ex, directory, report


def build_mapping(ex, directory, device):
    import numpy as np
    import torch
    from genrec import future_evaluation as f
    from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
    from gensid import historical_incremental_content as h
    feature_dir = Path(ex.plain['features_dir'])
    feature_report = f.read_json(feature_dir/'summary.json')
    f.verify_files(feature_dir, feature_report,
        ['update/'+name for name in ('embeddings.npy', 'feature_rows.json', 'registry.json')])
    protocol_dir = Path(ex.e1['protocol_dir'])
    registries = f.read_json(protocol_dir/'registries.json')
    manifest = f.read_json(protocol_dir/'protocol_manifest.json')
    f.check_registry_prefixes(registries, manifest)
    registry = f.read_json(feature_dir/'update/registry.json')
    values = np.load(feature_dir/'update/embeddings.npy', allow_pickle=False)
    rows = f.read_json(feature_dir/'update/feature_rows.json')
    n = len(ex.features)
    expected = [dict(raw_item_id=raw, gr_item_id=i+1, gensid_global_item_id=i, feature_row=i)
        for i, raw in enumerate(registries['update']['id2item'][1:])]
    if (registry != registries['update'] or rows != expected or values.dtype != np.float32
            or values.shape != (len(rows), ex.features.shape[1])
            or list(values.shape) != feature_report['stages']['update']['shape']
            or not np.isfinite(values).all() or not np.array_equal(values[:n], ex.features.numpy())):
        raise ValueError('Deployment features/registry/training prefix mismatch')
    checkpoint = torch.load(directory/'historical/last.pth', map_location='cpu', weights_only=False)
    model = h.build_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    codes = h.catalog_codes(model, ex).tolist()
    exported = f.read_jsonl(directory/'historical/semantic_assignments.jsonl')
    if exported != [dict(**row, semantic_code=code) for row, code in zip(ex.feature_rows, codes)]:
        raise ValueError('History-aware assignments changed on checkpoint reload')
    raw = registry['id2item'][1:]
    train = h.versioned_suffixes(ex.previous, raw[:n], codes)
    if train != f.read_json(directory/'historical/allocation_registry.json'):
        raise ValueError('History-aware allocation does not reproduce saved full SIDs')
    _, train_map = h.maps_from_ledger(train)
    if train_map != f.read_json(directory/'historical/item2tokens.json'):
        raise ValueError('History-aware saved token mapping mismatch')
    added = []
    batch = ex.plain['config']['added_item_assignment_eval_batch_size']
    with torch.no_grad():
        for start in range(n, len(values), batch):
            added.extend(model.get_indices(torch.from_numpy(values[start:start+batch]).to(device),
                None, use_sk=False).cpu().tolist())
    # Rebuild against the ORIGINAL Base reservations, as in the measured training version.
    deployment = h.versioned_suffixes(ex.previous, raw, codes+added)
    if deployment['entries'][:n] != train['entries']:
        raise ValueError('Deployment extension changed measured training SID addresses')
    if not all(torch.equal(value.cpu(), checkpoint['state_dict'][key])
               for key, value in model.state_dict().items()):
        raise ValueError('RQ-VAE changed during deployment export')
    _, full_map = h.maps_from_ledger(deployment)
    capacities = train['semantic_capacities']+[train['suffix_capacity']]
    tokenizers = {key: FrozenSIDTokenizer(mapping, capacities, manifest['max_history'])
                  for key, mapping in (('train', train_map), ('deployment', full_map))}
    if any(t.vocab_size != ex.tokenizer.vocab_size for t in tokenizers.values()):
        raise ValueError('SID vocabulary changed')
    del model
    return dict(train_ledger=train, deployment_ledger=deployment, tokenizers=tokenizers), manifest, registries


def make_splits(f, ex, tokenizer, registries, args, config):
    import numpy as np
    from genrec.temporal_tiger import probe_rows
    base_ids = [row['sample_id'] for rows in ex.rows.values() for row in rows]
    train = f.load_stage(ex, 'update_train', base_ids)
    val = f.load_stage(ex, 'update_val', base_ids+[row['sample_id'] for row in train])
    splits = {name: f.encode(rows, tokenizer, registries['update_train'])
              for name, rows in (('update_train', train), ('full_update_val', val))}
    full = splits['full_update_val']
    chosen = probe_rows(full['audit'], args.probe_size, config['seed'])
    audit = [copy.deepcopy(full['audit'][i]) for i in chosen]
    selected = np.asarray([r['tensor_row'] for r in audit if r['target_servable']], dtype=np.int64)
    arrays = {key: value[selected] for key, value in full['arrays'].items()}
    cursor = 0
    for row in audit:
        if row['target_servable']:
            row['tensor_row'] = cursor
            cursor += 1
    splits['update_val'] = dict(arrays=arrays, audit=audit, stats=full['stats'])
    return splits, train, val, base_ids, [full['audit'][i]['sample_id'] for i in chosen]


def compact_report(report):
    result = {key: report[key] for key in ('stage', 'execution_status', 'retuned_dir',
        'config', 'condition_checks', 'finetuned_minus_frozen', 'future_used_for_selection',
        'selection_locked_before_future', 'original_inputs_unchanged', 'rqvae_training_performed',
        'runtime_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')}
    result['training'] = {key: report['training'][key] for key in ('update_optimizer_steps',
        'processed_examples', 'best_update_optimizer_step', 'best_update_validation', 'full_update_validation',
        'exact_base_warm_start', 'selected_checkpoint_reload_equal', 'runtime_seconds')}
    result['future'] = {state: {key: value for key, value in metrics.items() if key not in ('slices', 'split_stats')}
                        for state, metrics in report['future'].items()}
    return result


def run(args):
    validate_args(args)
    import torch
    from genrec import future_evaluation as f
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError('Refusing to overwrite; use a fresh run or explicit --resume')
    if args.resume and not (output/'config.json').is_file():
        raise ValueError('Explicit resume requires an existing config.json')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested device unavailable')
    ex, directory, rq_report = load_context(args)
    inputs = [directory, Path(ex.e1['protocol_dir']).resolve(), Path(ex.plain['features_dir']).resolve(),
              ex.e1_dir.resolve(), Path(ex.e1['tiger_dir']).resolve(), Path(ex.plain['base_rqvae_dir']).resolve(),
              Path(ex.e1['update_rqvae_dir']).resolve()]
    if any(p == output or p in output.parents for p in inputs):
        raise ValueError('Outputs must be outside immutable inputs')
    protected = [directory/name for name in ('request.json', 'summary.json', 'compact.json', 'pair_checks.json')]
    for arm in ('plain', 'historical'):
        protected.extend(directory/arm/name for name in ('summary.json', 'last.pth', 'allocation_registry.json',
                                                        'item2tokens.json', 'semantic_assignments.jsonl'))
    protected.extend([ex.e1_dir/'summary.json', Path(ex.plain['base_rqvae_dir'])/'last.pth',
                      Path(ex.e1['update_rqvae_dir'])/'last.pth', Path(ex.e1['tiger_dir'])/'best.pth'])
    before = {str(p): digest(p) for p in protected}
    config = copy.deepcopy(ex.teacher['config'])
    config.update(device=str(device), lr=args.lr, warmup_steps=args.warmup_steps,
        eval_batch_size=args.eval_batch_size, topk=sorted(args.topk), num_beams=args.num_beams,
        selection_metric='ndcg@'+str(10 if 10 in args.topk else max(args.topk)),
        selection_steps=sorted({0, args.max_steps, *(s for s in args.selection_steps if 0 < s < args.max_steps)}),
        update_max_steps=args.max_steps, selection_probe_size=args.probe_size,
        checkpoint_selection='best_fixed_Update_validation_probe_full_denominator_NDCG; earliest tie')
    protocol = Path(ex.e1['protocol_dir'])
    protocol_report = f.read_json(protocol/'summary.json')
    f.verify_files(protocol, protocol_report, ['update_train.jsonl', 'update_val.jsonl', 'future_test.jsonl'])
    binding = dict(retuned_summary_sha256=digest(directory/'summary.json'),
        retuned_request_sha256=digest(directory/'request.json'),
        historical_rqvae_sha256=digest(directory/'historical/last.pth'),
        base_tiger_sha256=ex.e1['selected_tiger_sha256'], upstream=ex.binding,
        retuned_binding=rq_report['binding'], sources={name: digest(ROOT/name) for name in SOURCES},
        split_sha256={name: digest(protocol/(name+'.jsonl')) for name in ('update_train', 'update_val', 'future_test')})
    provenance = dict(config=config, binding=binding)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists():
        if f.read_json(output/'config.json') != provenance:
            raise ValueError('Future resume source/data/config binding mismatch')
    else:
        f.atomic_json(output/'config.json', provenance)
    if (output/'summary.json').exists():
        report = f.read_json(output/'summary.json')
        if report['binding']['sources'] != binding['sources'] or report['config'] != config:
            raise ValueError('Completed report source/config mismatch')
        f.verify_files(output, report, list(report['artifact_files_sha256']))
        return report
    f.seed_all(config['seed'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    mapping, manifest, registries = build_mapping(ex, directory, device)
    for key in ('train_ledger', 'deployment_ledger'):
        path = output/(key+'.json')
        if path.exists() and f.read_json(path) != mapping[key]:
            raise ValueError('Saved mapping differs from immutable checkpoint reconstruction')
        f.atomic_json(path, mapping[key])
    binding['mapping_sha256'] = {key: digest(output/(key+'.json')) for key in ('train_ledger', 'deployment_ledger')}
    splits, train_rows, val_rows, base_ids, probe_ids = make_splits(f, ex, mapping['tokenizers']['train'], registries, args, config)
    binding['validation_probe_sample_ids'] = probe_ids
    training = f.fit_arm(ex, 'historical', mapping['tokenizers']['train'], splits, config,
                         dict(binding, arm='historical'), output/'historical', device, args)
    if (training['execution_status'] != 'completed' or training['update_optimizer_steps'] != args.max_steps
            or training['exact_base_warm_start'] is not True
            or training['selected_checkpoint_reload_equal'] is not True
            or training['best_update_optimizer_step'] not in config['selection_steps']):
        raise ValueError('TIGER fine-tuning budget/warm-start/selection checks did not pass')
    selected_sha = digest(output/'historical/best.pth')
    lock = dict(binding=binding, checkpoint_sha256=selected_sha,
        update_optimizer_step=training['best_update_optimizer_step'], selection_split='update_val',
        selection_metric=config['selection_metric'], future_used_for_selection=False)
    lock_path = output/'selection_locked.json'
    if lock_path.exists() and f.read_json(lock_path) != lock:
        raise ValueError('Selected checkpoint changed after Future lock')
    f.atomic_json(lock_path, lock)
    # No Future rows are parsed until the sole fine-tuning arm is selected and locked.
    rows = f.load_stage(ex, 'future_test', base_ids+[r['sample_id'] for r in train_rows+val_rows])
    tokenizer = mapping['tokenizers']['deployment']
    split = f.encode(rows, tokenizer, registries['update'])
    frequency = Counter(r['item_seq'][-1] for r in train_rows)
    results = {}
    for state in ('frozen', 'finetuned'):
        metric_path, records_path, checksum = [output/('historical_'+state+suffix)
            for suffix in ('.json', '_queries.jsonl', '.sha256.json')]
        if all(p.exists() for p in (metric_path, records_path, checksum)):
            if f.read_json(checksum) != {p.name: digest(p) for p in (metric_path, records_path)}:
                raise ValueError('Cached Future checksum mismatch')
            results[state] = f.read_json(metric_path)
            continue
        model = f.TIGER(config, None, tokenizer).to(device)
        checkpoint = ex.teacher if state == 'frozen' else torch.load(output/'historical/best.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        evaluated = time.perf_counter()
        metrics = f.future_metrics(model, config, tokenizer, split, rows, registries, frequency, device, records_path)
        if not all(torch.equal(v.cpu(), checkpoint['state_dict'][k]) for k, v in model.state_dict().items()):
            raise ValueError('TIGER weights changed during Future inference')
        metrics.update(arm='historical', model_state=state,
            update_optimizer_steps=0 if state == 'frozen' else lock['update_optimizer_step'],
            model_sha256=ex.e1['selected_tiger_sha256'] if state == 'frozen' else selected_sha,
            runtime_seconds=time.perf_counter()-evaluated, selection_locked_before_future=True,
            teacher_frozen_during_evaluation=True, split_stats=split['stats'])
        f.atomic_json(metric_path, metrics)
        f.atomic_json(checksum, {p.name: digest(p) for p in (metric_path, records_path)})
        results[state] = metrics
        print('historical_'+state+': '+json.dumps(metrics, allow_nan=False), flush=True)
        del model
    if any(digest(path) != value for path, value in before.items()):
        raise ValueError('An immutable original input changed')
    if any(results['frozen'][key] != results['finetuned'][key]
           for key in ('all_queries', 'servable_queries', 'unavailable_targets', 'coverage', 'eligible_items')):
        raise ValueError('Future conditions use different catalogs/denominators')
    checks = dict(execution_status='passed', same_history_and_target_mapping=True,
        same_future_queries_and_catalog=True, historical_training_sids_preserved=True,
        exact_base_tiger_warm_start=training['exact_base_warm_start'],
        selected_checkpoint_reload_equal=training['selected_checkpoint_reload_equal'])
    differences = {denominator: {key: results['finetuned'][denominator][key]-value
        for key, value in results['frozen'][denominator].items()}
        for denominator in ('full_denominator', 'conditional_on_servable')}
    report = dict(stage=STAGE, execution_status='completed', retuned_dir=str(directory),
        binding=binding, config=config, training=training, condition_checks=checks, future=results,
        finetuned_minus_frozen=differences, future_used_for_selection=False,
        selection_locked_before_future=True, original_inputs_unchanged=True, rqvae_training_performed=False,
        catalog_policy='same observed Update deployment catalog; fixed historical RQ weights; append observed identities',
        evaluation_target_policy='current historical full SIDs for BOTH history and target',
        runtime_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
    f.atomic_json(output/'compact.json', compact_report(report))
    lines = ['## History-aware Future results', '',
        '| Tokenizer | TIGER | Selected Update step | '+ ' | '.join(results['frozen']['full_denominator'])+' |',
        '| --- | --- | ---: | '+' | '.join('---:' for _ in results['frozen']['full_denominator'])+' |']
    for state in ('frozen', 'finetuned'):
        m = results[state]
        lines.append('| History-aware | '+state+' | '+str(m['update_optimizer_steps'])+' | '+
            ' | '.join(f'{v:.12f}' for v in m['full_denominator'].values())+' |')
    lines.extend(['', 'Raw Recall/NDCG values; full denominator, unavailable targets count as misses.',
        'Both history and target use the same immutable History-aware full-SID mapping.',
        'Selection uses Update validation only; Future is not used for selection.', ''])
    (output/'experiment_report_fragment.md').write_text('\n'.join(lines))
    report['artifact_files_sha256'] = {p.relative_to(output).as_posix(): digest(p)
        for p in output.rglob('*') if p.is_file() and p != output/'summary.json'}
    f.atomic_json(output/'summary.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e1-dir', required=True); p.add_argument('--retuned-dir', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-steps', type=int, default=2000); p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--warmup-steps', type=int, default=200); p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--probe-size', type=int, default=512)
    p.add_argument('--selection-steps', nargs='+', type=int, default=[100, 250, 500, 1000, 2000])
    p.add_argument('--topk', nargs='+', type=int, default=[5, 10, 20]); p.add_argument('--num-beams', type=int, default=20)
    p.add_argument('--log-every', type=int, default=100); p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    report = run(parser().parse_args())
    print(json.dumps(compact_report(report), indent=2, allow_nan=False), flush=True)
