"""Paired frozen/finetuned TIGER evaluation on an untouched temporal Future split.

RQ-VAE snapshots are immutable. Both arms extend to the same observed Update
catalog. Selection uses Update validation only; Future inference starts only
after BOTH arms' selected checkpoints have been locked on disk.
"""
import argparse
from collections import Counter
import copy
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch.nn import functional as F

from genrec.evaluator import Evaluator
from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
from genrec.future_protocol import (check_registry_prefixes, sliced_metrics,
                                    summarize_records, target_group, validate_stage)
from genrec.historical_damage import versioned_suffixes
from genrec.models.TIGER.model import TIGER
from genrec.temporal_sids import (allocate_suffixes, maps_from_ledger, prepare_split,
                                 read_json, read_jsonl, sha256, verify_files, write_json)
from genrec.temporal_tiger import (catalog_prefix, cpu_state, evaluate, make_batch,
                                  restore_rng, rng_state, seed_all)
from genrec.trainer import Trainer
from gensid.incremental_content import build_model
from gensid.historical_incremental_content import catalog_codes, experiment

SOURCES = ['genrec/future_evaluation.py', 'genrec/future_protocol.py',
           'evaluate_incremental_future.py']
FORMAT = 'paired_update_tiger_v1'


def atomic_json(path, value):
    pending = path.with_name(path.name+'.tmp')
    write_json(pending, value)
    os.replace(pending, path)


def load_stage(ex, stage, seen=()):
    directory = Path(ex.e1['protocol_dir'])
    summary = read_json(directory/'summary.json')
    verify_files(directory, summary, [stage+'.jsonl'])
    manifest = read_json(directory/'protocol_manifest.json')
    registries = read_json(directory/'registries.json')
    rows = read_jsonl(directory/(stage+'.jsonl'))
    validate_stage(rows, stage, manifest, registries, seen)
    return rows


def build_maps(ex, device):
    """Preserve the measured train assignments; only encode deployment additions."""
    features_dir = Path(ex.plain['features_dir'])
    report = read_json(features_dir/'summary.json')
    names = ['update/'+name for name in ('embeddings.npy', 'feature_rows.json', 'registry.json')]
    verify_files(features_dir, report, names)
    registries = read_json(Path(ex.e1['protocol_dir'])/'registries.json')
    manifest = read_json(Path(ex.e1['protocol_dir'])/'protocol_manifest.json')
    check_registry_prefixes(registries, manifest)
    registry = read_json(features_dir/'update/registry.json')
    values = np.load(features_dir/'update/embeddings.npy', allow_pickle=False)
    rows = read_json(features_dir/'update/feature_rows.json')
    n = len(ex.features)
    expected = [dict(raw_item_id=raw, gr_item_id=i+1, gensid_global_item_id=i, feature_row=i)
                for i, raw in enumerate(registries['update']['id2item'][1:])]
    if (registry != registries['update'] or rows != expected or values.dtype != np.float32
            or values.shape != (len(rows), ex.features.shape[1])
            or list(values.shape) != report['stages']['update']['shape']
            or not np.isfinite(values).all() or not np.array_equal(values[:n], ex.features.numpy())):
        raise ValueError('Deployment features/IDs/unchanged Update training prefix mismatch')
    plain_dir = Path(ex.e1['update_rqvae_dir'])
    plain_checkpoint = torch.load(plain_dir/'last.pth', map_location='cpu', weights_only=False)
    original = {arm: read_jsonl(plain_dir/name) for arm, name in
                (('base', 'old_semantic_assignments.jsonl'), ('plain', 'semantic_assignments.jsonl'))}
    result = {}
    for arm, checkpoint in (('base', ex.parent), ('plain', plain_checkpoint)):
        model = build_model(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        codes = catalog_codes(model, ex).tolist()
        if codes != [row['semantic_code'] for row in original[arm]]:
            raise ValueError('Measured '+arm+' assignments changed on checkpoint reload')
        added = []
        batch = ex.plain['config']['added_item_assignment_eval_batch_size']
        with torch.no_grad():
            for start in range(n, len(values), batch):
                data = torch.from_numpy(values[start:start+batch]).to(device)
                added.extend(model.get_indices(data, None, use_sk=False).cpu().tolist())
        raw = registry['id2item'][1:]
        if arm == 'base':
            train_ledger = allocate_suffixes(raw[:n], codes, ex.previous['semantic_capacities'],
                ex.previous['suffix_capacity'], previous=ex.previous)
        else:
            train_ledger = read_json(ex.e1_dir/'new_allocation_registry.json')
            if ([entry['raw_item_id'] for entry in train_ledger['entries']] != raw[:n]
                    or [entry['semantic_code'] for entry in train_ledger['entries']] != codes):
                raise ValueError('Measured Plain allocation differs from checkpoint assignments')
        deployment = (allocate_suffixes(raw, codes+added, train_ledger['semantic_capacities'],
            train_ledger['suffix_capacity'], previous=train_ledger) if arm == 'base'
            else versioned_suffixes(ex.previous, raw, codes+added))
        if deployment['entries'][:n] != train_ledger['entries']:
            raise ValueError('Deployment extension changed training SID slots')
        capacities = train_ledger['semantic_capacities']+[train_ledger['suffix_capacity']]
        _, train_map = maps_from_ledger(train_ledger)
        _, full_map = maps_from_ledger(deployment)
        tokenizers = {key: FrozenSIDTokenizer(mapping, capacities, manifest['max_history'])
                      for key, mapping in (('train', train_map), ('deployment', full_map))}
        if any(t.vocab_size != ex.tokenizer.vocab_size for t in tokenizers.values()):
            raise ValueError('Vocabulary changed during catalog extension')
        result[arm] = dict(train_ledger=train_ledger, deployment_ledger=deployment,
                           tokenizers=tokenizers)
        del model
    return result, manifest, registries


def encode(rows, tokenizer, registry):
    arrays, audit, stats = prepare_split(rows, tokenizer,
        {raw: i for i, raw in enumerate(registry['id2item'][1:])})
    return dict(arrays=arrays, audit=audit, stats=stats)


def rank_validation(model, config, tokenizer, split, device):
    wrapper = SimpleNamespace(model=model, config=config, evaluator=Evaluator(config, tokenizer))
    print('Update validation | queries='+str(len(split['audit'])), flush=True)
    result = evaluate(wrapper, tokenizer, split['arrays'], split['audit'],
        list(range(len(split['audit']))), device, config['eval_batch_size'], progress=False)
    result['ranking_catalog'] = 'update_train'
    return result


def save_checkpoint(path, payload):
    pending = path.with_name(path.name+'.tmp')
    torch.save(payload, pending)
    os.replace(pending, path)


def fit_arm(ex, arm, tokenizer, splits, config, binding, directory, device, args):
    """Same repository Trainer, sampler, optimizer recipe and RNG seed in both arms."""
    directory.mkdir(exist_ok=True)
    selected_path, last_path = directory/'best.pth', directory/'last.pth'
    if (directory/'summary.json').exists():
        report = read_json(directory/'summary.json')
        if report['binding'] != binding or report['config'] != config:
            raise ValueError('Completed finetuning source/config binding mismatch')
        verify_files(directory, report, ['best.pth', 'last.pth'])
        selected = torch.load(selected_path, map_location='cpu', weights_only=False)
        if (selected.get('format') != 'paired_update_tiger_selected_v1'
                or selected['binding'] != binding or selected['config'] != config
                or selected['optimizer_step'] != report['best_update_optimizer_step']
                or report['update_optimizer_steps'] != args.max_steps):
            raise ValueError('Completed selected checkpoint/report mismatch')
        print(arm+' completed finetuning reused', flush=True)
        return report
    seed_all(config['seed'])
    model = TIGER(config, None, tokenizer).to(device)
    model.load_state_dict(ex.teacher['state_dict'], strict=True)
    if not all(torch.equal(v.cpu(), ex.teacher['state_dict'][k]) for k, v in model.state_dict().items()):
        raise ValueError('Finetuning did not warm-start from the exact selected Base TIGER')
    trainer = Trainer.for_temporal(config, model, tokenizer)
    generator = torch.Generator().manual_seed(config['seed']+1)
    sampler = dict(order=None, cursor=0, optimizer_steps=0, processed_examples=0, epochs_started=0)
    best_state, best_metrics, best_step, history, elapsed = None, None, None, [], 0.
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=False)
        if (checkpoint.get('format') != FORMAT or checkpoint['binding'] != binding
                or checkpoint['config'] != config):
            raise ValueError('Finetuning resume source/data/config mismatch')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler'])
        sampler = checkpoint['sampler']
        generator.set_state(checkpoint['generator'])
        best_state, best_metrics, best_step = (checkpoint[key] for key in ('best_state', 'best_metrics', 'best_step'))
        history, elapsed = checkpoint['validation_history'], checkpoint['elapsed_seconds']
        restore_rng(checkpoint['rng'])
    started = time.perf_counter()
    metric = config['selection_metric']

    def save_last():
        save_checkpoint(last_path, dict(format=FORMAT, binding=binding, config=config,
            state_dict=cpu_state(model), optimizer=trainer.optimizer.state_dict(),
            scheduler=trainer.scheduler.state_dict(), sampler=copy.deepcopy(sampler),
            generator=generator.get_state(), rng=rng_state(), best_state=best_state,
            best_metrics=best_metrics, best_step=best_step, validation_history=history,
            elapsed_seconds=elapsed+time.perf_counter()-started))

    def validation():
        nonlocal best_state, best_metrics, best_step
        step = sampler['optimizer_steps']
        metrics = rank_validation(model, config, tokenizer, splits['update_val'], device)
        history.append(dict(step=step, metrics=metrics))
        if best_metrics is None or metrics['full_denominator'][metric] > best_metrics['full_denominator'][metric]:
            best_state, best_metrics, best_step = cpu_state(model), metrics, step
        print(arm+' Update validation '+json.dumps(dict(step=step, best_step=best_step, metrics=metrics)), flush=True)

    if not history:
        validation()  # Step zero is an eligible checkpoint; ties retain earliest.
        save_last()

    def on_step(loss, grad_norm):
        step = sampler['optimizer_steps']
        with (directory/'train_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step, token_ce=loss, gradient_norm=grad_norm), allow_nan=False)+'\n')
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            print(f'{arm} Update TIGER {step}/{args.max_steps} | CE {loss:.6f}', flush=True)
        if step in config['selection_steps']:
            validation()
        if step % args.save_every == 0 or step == args.max_steps or step in config['selection_steps']:
            save_last()

    trainer.fit_steps(splits['update_train']['arrays'], sampler, generator, args.max_steps,
                      lambda arrays, indices: make_batch(arrays, indices, device), on_step)
    save_last()
    save_checkpoint(selected_path, dict(format='paired_update_tiger_selected_v1', binding=binding,
        config=config, state_dict=best_state, optimizer_step=best_step))
    reloaded = torch.load(selected_path, map_location='cpu', weights_only=False)
    model.load_state_dict(reloaded['state_dict'], strict=True)
    if not all(torch.equal(v.cpu(), best_state[k]) for k, v in model.state_dict().items()):
        raise ValueError('Selected finetuned checkpoint reload mismatch')
    full_validation = rank_validation(model, config, tokenizer, splits['full_update_val'], device)
    report = dict(execution_status='completed', arm=arm, binding=binding, config=config,
        update_optimizer_steps=sampler['optimizer_steps'], processed_examples=sampler['processed_examples'],
        best_update_optimizer_step=best_step, best_update_validation=best_metrics,
        full_update_validation=full_validation,
        validation_history=history, exact_base_warm_start=True, optimizer_reset_at_update_start=True,
        selected_checkpoint_reload_equal=True, future_used_for_selection=False,
        runtime_seconds=elapsed+time.perf_counter()-started,
        artifact_files_sha256={name: sha256(directory/name) for name in ('best.pth', 'last.pth')})
    atomic_json(directory/'summary.json', report)
    return report


def check_training_pair(output):
    checkpoints = [torch.load(output/arm/'last.pth', map_location='cpu', weights_only=False) for arm in ('base', 'plain')]
    a, b = checkpoints
    bindings = [copy.deepcopy(checkpoint['binding']) for checkpoint in checkpoints]
    if [bound.pop('arm') for bound in bindings] != ['base', 'plain'] or bindings[0] != bindings[1]:
        raise ValueError('Paired TIGER source/data/starting checkpoint binding differs')
    if a['config'] != b['config'] or a['optimizer']['param_groups'] != b['optimizer']['param_groups']:
        raise ValueError('Paired TIGER content training/optimizer settings differ')
    for key in ('optimizer_steps', 'cursor', 'processed_examples', 'epochs_started'):
        if a['sampler'][key] != b['sampler'][key]:
            raise ValueError('Paired TIGER sampler endpoint differs: '+key)
    if (not torch.equal(a['sampler']['order'], b['sampler']['order'])
            or not torch.equal(a['generator'], b['generator'])):
        raise ValueError('Paired TIGER sampler order/generator differs')
    result = dict(execution_status='passed', same_starting_base_tiger=True, same_optimizer_recipe=True,
        same_update_training_queries=True, same_step_budget=True, same_sampler_endpoint=True,
        update_optimizer_steps=a['sampler']['optimizer_steps'], processed_examples=a['sampler']['processed_examples'])
    atomic_json(output/'pair_checks.json', result)
    return result


@torch.no_grad()
def future_metrics(model, config, tokenizer, split, rows, registries, frequency, device, records_path):
    evaluator = Evaluator(config, tokenizer)
    keys = [f'{metric}@{k}' for metric in config['metrics'] for k in config['topk']]
    records = [dict(source_row=i, sample_id=row['sample_id'], user=row['user'],
        target_raw_item_id=row['item_seq'][-1], target_servable=row['target_servable'],
        target_group=target_group(row['item_seq'][-1], registries),
        update_train_frequency=frequency[row['item_seq'][-1]], sequence_nll=None,
        metrics=dict.fromkeys(keys, 0.)) for i, row in enumerate(rows)]
    legal = set(tokenizer.item2tokens.values())
    from tqdm import tqdm
    model.eval().requires_grad_(False)
    size = config['eval_batch_size']
    for start in tqdm(range(0, len(split['arrays']['labels']), size), desc='Future test', unit='batch'):
        indices = np.arange(start, min(start+size, len(split['arrays']['labels'])))
        batch = make_batch(split['arrays'], indices, device)
        logits = model(batch).logits.float()
        losses = F.cross_entropy(logits.transpose(1, 2), batch['labels'], reduction='none').sum(1)
        predicted = model.generate(batch, n_return_sequences=evaluator.maxk, return_scores=True,
                                    prefix_allowed_tokens_fn=catalog_prefix(tokenizer))
        sequences = predicted['sequences']
        if (sequences.shape[-1] != 6 or torch.any(sequences[:, :, 0] != 0)
                or torch.any(sequences[:, :, -1] != tokenizer.eos_token)
                or not torch.isfinite(predicted['scores']).all() or not torch.isfinite(losses).all()):
            raise ValueError('Invalid Future prediction sequence, score or NLL')
        for prediction in predicted['preds'].cpu().tolist():
            if len({tuple(tokens) for tokens in prediction}) != evaluator.maxk or any(tuple(tokens) not in legal for tokens in prediction):
                raise ValueError('Duplicate or out-of-catalog Future prediction')
        values = evaluator.calculate_metrics(predicted, batch['labels'])
        for local, index in enumerate(indices):
            row = records[int(split['arrays']['source_rows'][index])]
            row['sequence_nll'] = float(losses[local])
            row['metrics'] = {key: float(value[local]) for key, value in values.items()}
    with records_path.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
    metrics = summarize_records(records, keys)
    metrics.update(slices=sliced_metrics(records, keys), ranking_catalog='update',
        eligible_items=len(tokenizer.item2tokens), invalid_predictions=0, seen_item_filter=False,
        num_beams=config['num_beams'], decoding='catalog_prefix_beam_search',
        history_protocol='prequential raw history; unavailable four-token blocks retain position and mask',
        target_address='current arm full SID plus EOS; history and target use the same mapping')
    return metrics


def write_report_fragment(output, report):
    lines = ['## Future：Base 与 Plain，冻结及 Update 微调后', '',
        '以下为服务器实际评估输出；两组 RQ-VAE 权重均冻结，采用相同 Update 部署目录。',
        'Base 行是冻结 Base RQ-VAE 对已观察 Update 新物品追加 SID 的对照。',
        'history 与 target 同时使用本组完整 SID；不沿用 history-only 的旧 target 诊断口径。', '',
        '| RQ-VAE | TIGER 状态 | 选中 Update 步数 | Future queries | 可服务 queries | 覆盖率 | '+
        ' | '.join(key for key in next(iter(report['future']['base'].values()))['full_denominator'])+' |',
        '| --- | --- | ---: | ---: | ---: | ---: | '+
        ' | '.join('---:' for _ in next(iter(report['future']['base'].values()))['full_denominator'])+' |']
    for arm in ('base', 'plain'):
        for state in ('frozen', 'finetuned'):
            result = report['future'][arm][state]
            lines.append('| '+ ' | '.join([arm, state, str(result['update_optimizer_steps']),
                str(result['all_queries']), str(result['servable_queries']), f'{result["coverage"]:.6%}',
                *(f'{value:.9f}' for value in result['full_denominator'].values())])+' |')
    lines.extend(['', '表格为 full-denominator 指标，不可服务 target 计 misses；conditional 指标与 old/new/频次切片见 summary.json。',
        'checkpoint 由固定 Update validation probe 选择，另报告选中模型的完整 Update validation；Future 未参与选择。',
        '微调预算：'+str(report['config']['update_max_steps'])+' 步；两组内容训练 query 顺序/优化器设置/预算核对通过。',
        '频次切片使用 Update train 中可形成因果样本的 target 出现次数，阈值为 5，不使用 Future 频次。',
        '结果目录：`'+str(output)+'`。', ''])
    (output/'experiment_report_fragment.md').write_text('\n'.join(lines))


def run(args):
    started = time.perf_counter()
    if (min(args.max_steps, args.eval_batch_size, args.probe_size, args.log_every, args.save_every) < 1
            or not args.topk or min(args.topk) < 1 or len(set(args.topk)) != len(args.topk)
            or args.num_beams < max(args.topk) or args.num_beams < 2
            or args.lr <= 0 or not np.isfinite(args.lr) or args.warmup_steps < 0):
        raise ValueError('Invalid paired Future configuration')
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError('Refusing to overwrite Future run; use a new output or explicit --resume')
    if args.resume and not output.is_dir():
        raise ValueError('Resume output directory does not exist')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested evaluation device unavailable')
    ex = experiment(SimpleNamespace(e1_dir=args.e1_dir))
    root = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(ex.teacher['config'])
    config.update(device=str(device), lr=args.lr, warmup_steps=args.warmup_steps,
        eval_batch_size=args.eval_batch_size, topk=sorted(args.topk), num_beams=args.num_beams,
        selection_metric='ndcg@'+str(10 if 10 in args.topk else max(args.topk)),
        selection_steps=sorted({0, args.max_steps, *(step for step in args.selection_steps if 0 < step < args.max_steps)}),
        update_max_steps=args.max_steps, checkpoint_selection='best_fixed_Update_validation_probe_full_denominator_NDCG; earliest tie',
        selection_probe_size=args.probe_size)
    binding = dict(e1_summary_sha256=sha256(ex.e1_dir/'summary.json'),
        base_tiger_sha256=ex.e1['selected_tiger_sha256'], base_rqvae_sha256=ex.e1['base_rqvae_sha256'],
        plain_rqvae_sha256=ex.e1['update_rqvae_sha256'], protocol_sha256=ex.e1['protocol_sha256'],
        upstream=ex.binding, sources={name: sha256(root/name) for name in SOURCES},
        package_versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'transformers', 'scikit-learn')})
    protocol_dir = Path(ex.e1['protocol_dir'])
    protocol_summary = read_json(protocol_dir/'summary.json')
    verify_files(protocol_dir, protocol_summary, ['update_train.jsonl', 'update_val.jsonl', 'future_test.jsonl'])
    binding['split_sha256'] = {stage: sha256(protocol_dir/(stage+'.jsonl')) for stage in ('update_train', 'update_val', 'future_test')}
    provenance = dict(binding=binding, config=config)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists():
        if read_json(output/'config.json') != provenance:
            raise ValueError('Future resume source/data/config binding mismatch')
    else:
        atomic_json(output/'config.json', provenance)
    if (output/'summary.json').exists():
        report = read_json(output/'summary.json')
        verify_files(output, report, list(report['artifact_files_sha256']))
        return report
    seed_all(config['seed'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    maps, manifest, registries = build_maps(ex, device)
    update_train = load_stage(ex, 'update_train', [r['sample_id'] for stage in ex.rows.values() for r in stage])
    base_ids = [r['sample_id'] for stage in ex.rows.values() for r in stage]
    update_val = load_stage(ex, 'update_val', base_ids+[r['sample_id'] for r in update_train])
    splits = {}
    map_hashes = {}
    for arm, values in maps.items():
        directory = output/arm
        directory.mkdir(exist_ok=True)
        for key in ('train_ledger', 'deployment_ledger'):
            path = directory/(key+'.json')
            if path.exists() and read_json(path) != values[key]:
                raise ValueError('Saved map differs from immutable checkpoint reconstruction')
            atomic_json(path, values[key])
        map_hashes[arm] = {key: sha256(directory/(key+'.json')) for key in ('train_ledger', 'deployment_ledger')}
        tokenizer = values['tokenizers']['train']
        splits[arm] = {stage: encode(rows, tokenizer, registries['update_train'])
                       for stage, rows in (('update_train', update_train), ('update_val', update_val))}
    if (not np.array_equal(splits['base']['update_train']['arrays']['source_rows'], splits['plain']['update_train']['arrays']['source_rows'])
            or splits['base']['update_val']['audit'] != splits['plain']['update_val']['audit']):
        raise ValueError('Paired training/validation cohorts differ')
    # Fixed source-hash probe over ALL Update validation queries, not Future.
    from genrec.temporal_tiger import probe_rows
    chosen = probe_rows(splits['base']['update_val']['audit'], args.probe_size, config['seed'])
    for arm in splits:
        full = splits[arm]['update_val']
        splits[arm]['full_update_val'] = full
        audit = [copy.deepcopy(full['audit'][i]) for i in chosen]
        source_indices = [row['tensor_row'] for row in audit if row['target_servable']]
        arrays = {key: value[np.asarray(source_indices, dtype=np.int64)] for key, value in full['arrays'].items()}
        cursor = 0
        for row in audit:
            if row['target_servable']:
                row['tensor_row'] = cursor
                cursor += 1
        splits[arm]['update_val'] = dict(arrays=arrays, audit=audit, stats=full['stats'])
    binding['mapping_sha256'] = map_hashes
    # Lock mapping identity as part of each arm's independently resumable training.
    training = {arm: fit_arm(ex, arm, maps[arm]['tokenizers']['train'], splits[arm], config,
                            dict(binding, arm=arm), output/arm, device, args) for arm in ('base', 'plain')}
    pair = check_training_pair(output)
    selections = {arm: dict(checkpoint_sha256=sha256(output/arm/'best.pth'),
        update_optimizer_step=training[arm]['best_update_optimizer_step']) for arm in training}
    lock = dict(selections=selections, binding=binding, future_used_for_selection=False,
                selection_metric=config['selection_metric'], selection_split='update_val')
    lock_path = output/'selection_locked.json'
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError('Selected checkpoints changed after Future selection lock')
    atomic_json(lock_path, lock)
    # Future rows are first parsed ONLY after both arms have locked their choices.
    future_rows = load_stage(ex, 'future_test', base_ids+[r['sample_id'] for r in update_train+update_val])
    frequency = Counter(row['item_seq'][-1] for row in update_train)
    results = {}
    for arm in ('base', 'plain'):
        tokenizer = maps[arm]['tokenizers']['deployment']
        future = encode(future_rows, tokenizer, registries['update'])
        results[arm] = {}
        for state in ('frozen', 'finetuned'):
            name = arm+'_'+state
            metric_path, records_path = output/(name+'.json'), output/(name+'_queries.jsonl')
            checksum_path = output/(name+'.sha256.json')
            if metric_path.exists() and records_path.exists() and checksum_path.exists():
                saved = read_json(checksum_path)
                if saved != {path.name: sha256(path) for path in (metric_path, records_path)}:
                    raise ValueError('Cached Future result checksum mismatch')
                results[arm][state] = read_json(metric_path)
                continue
            model = TIGER(config, None, tokenizer).to(device)
            checkpoint = ex.teacher if state == 'frozen' else torch.load(output/arm/'best.pth', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval().requires_grad_(False)
            evaluated = time.perf_counter()
            metrics = future_metrics(model, config, tokenizer, future, future_rows, registries, frequency, device, records_path)
            if not all(torch.equal(v.cpu(), checkpoint['state_dict'][key]) for key, v in model.state_dict().items()):
                raise ValueError('Evaluated TIGER weights changed during Future inference')
            metrics.update(arm=arm, model_state=state, update_optimizer_steps=0 if state == 'frozen' else selections[arm]['update_optimizer_step'],
                model_sha256=ex.e1['selected_tiger_sha256'] if state == 'frozen' else selections[arm]['checkpoint_sha256'],
                runtime_seconds=time.perf_counter()-evaluated, selection_locked_before_future=True,
                teacher_frozen_during_evaluation=True, split_stats=future['stats'])
            atomic_json(metric_path, metrics)
            atomic_json(checksum_path, {path.name: sha256(path) for path in (metric_path, records_path)})
            results[arm][state] = metrics
            print(name+': '+json.dumps(metrics, allow_nan=False), flush=True)
            del model
    differences = {state: {denominator: {key: results['plain'][state][denominator][key]-value
        for key, value in results['base'][state][denominator].items()}
        for denominator in ('full_denominator', 'conditional_on_servable')}
        for state in ('frozen', 'finetuned')}
    if (sha256(Path(ex.plain['base_rqvae_dir'])/'last.pth') != ex.e1['base_rqvae_sha256']
            or sha256(Path(ex.e1['update_rqvae_dir'])/'last.pth') != ex.e1['update_rqvae_sha256']
            or sha256(Path(ex.e1['tiger_dir'])/'best.pth') != ex.e1['selected_tiger_sha256']):
        raise ValueError('An immutable input checkpoint changed')
    report = dict(stage='paired_Base_Plain_Future_evaluation', execution_status='completed',
        e1_dir=str(ex.e1_dir.resolve()), binding=binding, config=config, training=training,
        pair_checks=pair, future=results, plain_minus_base=differences,
        catalog_policy='same observed Update deployment catalog; Base RQ weights frozen and new identities appended',
        evaluation_target_policy='each arm uses its own full SID mapping for BOTH history and target',
        future_used_for_selection=False, selection_locked_before_future=True,
        rqvae_training_performed=False, historical_aware_training_status='not_run_in_this_pipeline',
        original_checkpoints_unchanged=True, runtime_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
    write_report_fragment(output, report)
    report['artifact_files_sha256'] = {path.relative_to(output).as_posix(): sha256(path)
        for path in output.rglob('*') if path.is_file() and path != output/'summary.json'}
    atomic_json(output/'summary.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e1-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--probe-size', type=int, default=512)
    p.add_argument('--selection-steps', type=int, nargs='+', default=[100, 250, 500, 1000, 2000])
    p.add_argument('--topk', type=int, nargs='+', default=[5, 10, 20])
    p.add_argument('--num-beams', type=int, default=20)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    return p
