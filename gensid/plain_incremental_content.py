"""Ordinary content RQ-VAE update; preparation for E1, without historical penalty."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from gensid.incremental_content import (SOURCE_FILES as BASE_SOURCES, build_model,
    build_trainer, collision_stats, evaluate, implementation_binding, load_base_features,
    load_checkpoint, read_json, restore_rng, rng_state, seed_all, sha256, write_json)
from genrec.temporal_sids import validate_registry, verify_files

SOURCES = BASE_SOURCES + ['gensid/plain_incremental_content.py', 'train_plain_incremental_rqvae.py',
                         'genrec/temporal_sids.py', 'genrec/frozen_sid_tokenizer.py', 'genrec/tokenizer.py']


def load_inputs(protocol_dir, features_dir, base_dir, expected_sha=None):
    protocol_dir, features_dir, base_dir = map(Path, (protocol_dir, features_dir, base_dir))
    base_features, base_rows, old_binding = load_base_features(features_dir)
    old_binding.update(implementation_binding())
    base_summary = read_json(base_dir/'summary.json')
    if (base_summary.get('stage') != 'E0_base_content_rqvae'
            or base_summary.get('execution_status') != 'completed'
            or base_summary.get('training_catalog') != 'base_train'):
        raise ValueError('Expected completed Base-only RQ-VAE')
    verify_files(base_dir, base_summary, list(base_summary['artifact_files_sha256']))
    checkpoint_sha = sha256(base_dir/'last.pth')
    if expected_sha and expected_sha != checkpoint_sha:
        raise ValueError('Unexpected Base RQ-VAE checkpoint SHA')
    checkpoint = load_checkpoint(base_dir/'last.pth')
    if checkpoint['binding'] != old_binding or base_summary['binding'] != old_binding:
        raise ValueError('Base checkpoint feature/source/package binding mismatch')
    if (checkpoint['config'] != base_summary['config']
            or checkpoint['sampler_state']['step'] != base_summary['optimizer_steps']):
        raise ValueError('Base checkpoint config/step mismatch')
    model_config = checkpoint['config']['model']
    if (len(model_config['num_emb_list']) != 3 or model_config['use_cf_loss']
            or any(model_config[k] != 0 for k in ('alpha', 'alpha_codebook', 'alpha_graph_cluster'))
            or any(model_config['sk_epsilons']) or model_config['bn']
            or model_config['dropout_prob'] != 0 or model_config['loss_type'] != 'mse'):
        raise ValueError('Expected three-layer pure-content Base model with no BN/dropout/Sinkhorn')
    protocol_summary = read_json(protocol_dir/'summary.json')
    if (protocol_summary.get('stage') != 'E0_data_preparation'
            or protocol_summary.get('execution_status') != 'completed'):
        raise ValueError('Expected completed temporal data preparation')
    verify_files(protocol_dir, protocol_summary, ['protocol_manifest.json', 'registries.json'])
    manifest, registries = (read_json(protocol_dir/name) for name in
                            ('protocol_manifest.json', 'registries.json'))
    canonical = dict(manifest)
    declared = canonical.pop('protocol_sha256')
    actual = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (actual != declared or declared != old_binding['protocol_sha256']
            or declared != protocol_summary['protocol_sha256']
            or manifest['catalog_reference']['update_train'] != 'update_train'):
        raise ValueError('Temporal protocol identity/training catalog mismatch')
    feature_summary = read_json(features_dir/'summary.json')
    names = ['update_train/'+name for name in ('embeddings.npy', 'feature_rows.json', 'registry.json')]
    verify_files(features_dir, feature_summary, names)
    values = np.load(features_dir/names[0], allow_pickle=False)
    rows, registry = read_json(features_dir/names[1]), read_json(features_dir/names[2])
    for stage in ('base_train', 'base', 'update_train'):
        validate_registry(registries[stage])
        if len(registries[stage]['id2item'])-1 != manifest['catalog_sizes'][stage]:
            raise ValueError('Protocol registry/catalog size mismatch')
    n_base = len(base_features)
    n_deployment = len(registries['base']['id2item'])-1
    if (registry != registries['update_train']
            or [r['raw_item_id'] for r in base_rows] != registries['base_train']['id2item'][1:]
            or registry['id2item'][:n_deployment+1] != registries['base']['id2item']
            or registries['base']['id2item'][:n_base+1] != registries['base_train']['id2item']
            or values.ndim != 2 or values.dtype != np.float32 or not np.isfinite(values).all()
            or len(values) != len(rows) or len(values)+1 != len(registry['id2item'])
            or list(values.shape) != feature_summary['stages']['update_train']['shape']
            or values.shape[1] != base_features.shape[1]
            or not np.array_equal(values[:n_base], base_features.numpy())):
        raise ValueError('Update feature shape/registry or unchanged Base prefix mismatch')
    for i, row in enumerate(rows):
        if row != dict(raw_item_id=registry['id2item'][i+1], gr_item_id=i+1,
                       gensid_global_item_id=i, feature_row=i):
            raise ValueError('Update feature row/item ID contract mismatch')
    assignments = [json.loads(line) for line in (base_dir/'semantic_assignments.jsonl').read_text().splitlines()]
    if (len(assignments) != n_base or any({k: r[k] for k in base_rows[i]} != base_rows[i]
                                         for i, r in enumerate(assignments))):
        raise ValueError('Base exported assignment row/ID mismatch')
    codes = np.asarray([r['semantic_code'] for r in assignments])
    collision_stats(codes, model_config['num_emb_list'])
    old_eval_batch = read_json(base_dir/'config.json')['command']['eval_batch_size']
    if type(old_eval_batch) is not int or old_eval_batch < 1:
        raise ValueError('Invalid original assignment evaluation batch size')
    groups = dict(base_train=(0, n_base),
        base_validation_first_seen=(n_base, n_deployment),
        update_train_first_seen=(n_deployment, len(values)))
    binding = dict(protocol_sha256=declared, feature_summary_sha256=sha256(features_dir/'summary.json'),
        protocol_artifact_files_sha256={name: sha256(protocol_dir/name) for name in
                                       ('protocol_manifest.json', 'registries.json')},
        update_artifact_files_sha256={name: sha256(features_dir/name) for name in names},
        parent_base_checkpoint_sha256=checkpoint_sha,
        parent_base_summary_sha256=sha256(base_dir/'summary.json'),
        parent_binding=old_binding, group_ranges={k: list(v) for k, v in groups.items()})
    return torch.from_numpy(values), rows, checkpoint, base_summary, binding, groups, codes, old_eval_batch


def evaluate_groups(model, features, groups, base_batch, other_batch):
    """Keep the original Base export batching; evaluate each stable registry slice."""
    metrics, codes = {}, []
    for name, (start, end) in groups.items():
        if end == start:
            metrics[name] = None
            continue
        result, assignment = evaluate(model, features[start:end],
            base_batch if name == 'base_train' else other_batch)
        metrics[name] = dict(items=end-start, **result)
        codes.append(assignment)
    combined = {key: sum(value[key]*value['items'] for value in metrics.values() if value)
                    /len(features) for key in ('total_loss', 'reconstruction_loss', 'quantization_loss')}
    return combined, metrics, np.concatenate(codes)


def churn(old, new):
    changed = old != new
    return dict(items=len(old), changed_items=int(changed.any(axis=1).sum()),
        unchanged_items=int((~changed.any(axis=1)).sum()),
        changed_item_fraction=float(changed.any(axis=1).mean()),
        per_level_changed_fraction=changed.mean(axis=0).tolist(),
        prefix_changed_fraction=[float(changed[:, :i].any(axis=1).mean()) for i in (1, 2, 3)],
        codeword_alignment='none', scope='three raw semantic indices; collision suffix not included')


def save(path, trainer, config, binding, sampler, initial, initial_groups):
    temp = Path(str(path)+'.tmp')
    torch.save(dict(format='plain_update_rqvae_v1', config=config, binding=binding,
        state_dict=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
        sampler_state=sampler, rng_state=rng_state(), initial_metrics=initial,
        initial_group_metrics=initial_groups), temp)
    os.replace(temp, path)


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite output: '+str(output))
    if (min(args.max_steps, args.batch_size, args.eval_batch_size, args.log_every, args.save_every) < 1
            or not np.isfinite(args.lr) or args.lr <= 0
            or not np.isfinite(args.weight_decay) or args.weight_decay < 0
            or not 0 <= args.seed < 2**32):
        raise ValueError('Invalid training/optimizer/seed settings')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA supported')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    features, rows, parent, parent_summary, binding, groups, old_export, old_batch = load_inputs(
        args.protocol_dir, args.features_dir, args.base_rqvae_dir, args.expected_base_sha256)
    root = Path(__file__).resolve().parents[1]
    binding.update(source_files_sha256={name: sha256(root/name) for name in SOURCES},
        package_versions=implementation_binding()['package_versions'], python=platform.python_version())
    config = dict(device=str(device), seed=args.seed, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, model=copy.deepcopy(parent['config']['model']),
        initialization='exact_base_weights_without_reinitialization', optimizer_initialization='reset_adamw',
        original_base_assignment_eval_batch_size=old_batch,
        added_item_assignment_eval_batch_size=args.eval_batch_size)
    resume = torch.load(args.resume_from, map_location='cpu', weights_only=False) if args.resume_from else None
    if resume and (resume.get('format') != 'plain_update_rqvae_v1'
                   or resume['config'] != config or resume['binding'] != binding):
        raise ValueError('Update resume data/source/package/config binding mismatch')
    if resume and resume['sampler_state']['step'] >= args.max_steps:
        raise ValueError('Target update steps must exceed resumed checkpoint steps')
    seed_all(args.seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config).to(device)
    model.load_state_dict(parent['state_dict'], strict=True)
    if not all(layer.initted for layer in model.rq.vq_layers):
        raise ValueError('Parent contains uninitialized codebooks')
    warm_start_equal = all(torch.equal(v.cpu(), parent['state_dict'][key])
                           for key, v in model.state_dict().items())
    initial, initial_groups, old_codes = evaluate_groups(model, features, groups, old_batch, args.eval_batch_size)
    n_base = groups['base_train'][1]
    if not np.array_equal(old_codes[:n_base], old_export):
        raise ValueError('Frozen Base assignments changed before Update training')
    output.mkdir(parents=True)
    write_json(output/'config.json', dict(config=config, binding=binding, command=vars(args)))
    trainer = build_trainer(model, config, output)
    state = None
    if resume:
        model.load_state_dict(resume['state_dict'], strict=True)
        trainer.optimizer.load_state_dict(resume['optimizer'])
        state = resume['sampler_state']
        initial, initial_groups = resume['initial_metrics'], resume['initial_group_metrics']
        restore_rng(resume['rng_state'])
    first_step = state['step'] if state else 0
    print(f'Plain Update ready | {len(features)} eligible items | {n_base} Base training items | '
          f'parent Base steps {parent_summary["optimizer_steps"]} | update steps {first_step}->{args.max_steps}', flush=True)
    with (output/'metrics.jsonl').open('w') as log:
        def on_step(current, metrics):
            log.write(json.dumps(metrics, allow_nan=False)+'\n')
            log.flush()
            if (current['step'] == first_step+1 or current['step'] % args.log_every == 0
                    or current['step'] == args.max_steps):
                print(json.dumps(metrics, allow_nan=False), flush=True)
            if current['step'] % args.save_every == 0:
                save(output/'last.pth', trainer, config, binding, current, initial, initial_groups)
        state = trainer.fit_steps(features, torch.arange(len(features)), args.max_steps,
                                  args.batch_size, args.seed, state, on_step)
    save(output/'last.pth', trainer, config, binding, state, initial, initial_groups)
    final, final_groups, new_codes = evaluate_groups(model, features, groups, old_batch, args.eval_batch_size)
    restored = build_model(config).to(device)
    restored.load_state_dict(torch.load(output/'last.pth', map_location='cpu', weights_only=False)['state_dict'])
    _, _, reloaded_codes = evaluate_groups(restored, features, groups, old_batch, args.eval_batch_size)
    if not np.array_equal(new_codes, reloaded_codes):
        raise ValueError('Update checkpoint reload changed assignments')
    comparison = churn(old_codes[:n_base], new_codes[:n_base])
    write_json(output/'base_semantic_churn.json', comparison)
    for filename, codes in (('old_semantic_assignments.jsonl', old_codes),
                             ('semantic_assignments.jsonl', new_codes)):
        with (output/filename).open('w') as handle:
            for row, code in zip(rows, codes.tolist()):
                handle.write(json.dumps(dict(**row, semantic_code=code))+'\n')
    with (output/'base_assignment_changes.jsonl').open('w') as handle:
        for row, old, new in zip(rows[:n_base], old_codes.tolist()[:n_base], new_codes.tolist()[:n_base]):
            handle.write(json.dumps(dict(**row, old_semantic_code=old, new_semantic_code=new,
                changed=old != new, changed_levels=[a != b for a, b in zip(old, new)]))+'\n')
    # Verify source checkpoint is unchanged, including across training/resume.
    if sha256(Path(args.base_rqvae_dir)/'last.pth') != binding['parent_base_checkpoint_sha256']:
        raise ValueError('Parent Base checkpoint changed during Update training')
    report = dict(stage='E1_plain_update_content_rqvae_preparation', execution_status='completed',
        protocol_dir=str(Path(args.protocol_dir).resolve()), features_dir=str(Path(args.features_dir).resolve()),
        base_rqvae_dir=str(Path(args.base_rqvae_dir).resolve()), output=str(output),
        protocol_sha256=binding['protocol_sha256'], parent_base_checkpoint_sha256=binding['parent_base_checkpoint_sha256'],
        parent_base_optimizer_steps=parent_summary['optimizer_steps'], config=config, binding=binding,
        training_catalog='update_train', training_items=len(features), feature_dim=features.shape[1],
        item_groups={name: end-start for name, (start, end) in groups.items()},
        update_optimizer_steps=state['step'], steps_this_run=state['step']-first_step,
        processed_examples=state['processed_examples'], sampler_epochs_started=state['epoch'],
        checkpoint_selection='last_at_fixed_update_optimizer_step_budget',
        resume_from=str(Path(args.resume_from).resolve()) if args.resume_from else None,
        warm_start_all_weights_equal=warm_start_equal, original_base_assignments_equal_before_update=True,
        parent_base_checkpoint_unchanged=True, checkpoint_reload_assignments_equal=True,
        initial_train_catalog_metrics=initial, final_train_catalog_metrics=final,
        initial_group_metrics=initial_groups, final_group_metrics=final_groups,
        base_semantic_churn=comparison,
        collision_statistics=collision_stats(new_codes, config['model']['num_emb_list']),
        training_sampling='uniform shuffled eligible catalog, including old item content',
        trained_components=['encoder', 'decoder', 'three_codebooks'], optimizer_reset_at_update_start=True,
        codebook_reinitialization=False, codeword_alignment='none', historical_prediction_penalty_used=False,
        graph_used=False, cf_loss_used=False, sinkhorn_used=False, content_encoder_finetuning=False,
        feature_normalization='unchanged', projection='none', base_sid_artifacts_modified=False,
        sid_export='three raw zero-based semantic indices; collisions retained',
        collision_suffix_status='not_created_for_new_version',
        historical_prediction_damage_status='not_run', recommendation_training_status='not_run',
        future_test_status='not_run', E0_full_status='not_run', E1_status='not_run',
        python=platform.python_version(), package_versions=binding['package_versions'],
        source_files_sha256=binding['source_files_sha256'], device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        memory_scope='this process, model loading through assignment reload check',
        runtime_seconds=time.perf_counter()-started,
        artifact_files_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'summary.json', report)
    trainer.tb_writer.close()
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-dir', required=True)
    p.add_argument('--features-dir', required=True)
    p.add_argument('--base-rqvae-dir', required=True)
    p.add_argument('--expected-base-sha256')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-steps', type=int, default=1000, help='Update steps only; includes resumed Update steps')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--resume-from', help='Plain Update last.pth, not Base RQ-VAE or TIGER')
    return p
