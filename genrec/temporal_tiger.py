"""Train the repository TIGER on frozen temporal SIDs; no feature/SID refitting."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
from genrec.models.TIGER.model import TIGER
from genrec.temporal_sids import (read_json, read_jsonl, sha256, validate_registry,
                                 verify_files, write_json)
from genrec.trainer import Trainer

SOURCE_FILES = ['genrec/temporal_tiger.py', 'train_incremental_tiger.py',
    'genrec/trainer.py', 'genrec/models/TIGER/model.py', 'genrec/evaluator.py',
    'genrec/model.py', 'genrec/frozen_sid_tokenizer.py', 'genrec/tokenizer.py',
    'genrec/temporal_sids.py']
MODEL_DEFAULTS = dict(num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, activation_function='relu',
    feed_forward_proj='relu')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def load_sids(directory, expected_checkpoint=None):
    """Verify all frozen artifacts and cross-check tensor rows against the audits."""
    directory = Path(directory)
    summary = read_json(directory/'summary.json')
    if (summary.get('stage') != 'E0_base_frozen_sid_preparation'
            or summary.get('execution_status') != 'completed'
            or summary.get('rqvae_frozen') is not True
            or summary.get('rqvae_training_performed') is not False
            or summary.get('old_full_sids_unchanged_on_deployment_extension') is not True):
        raise ValueError('Expected completed frozen Base SID preparation')
    if expected_checkpoint and summary['rqvae_checkpoint_sha256'] != expected_checkpoint:
        raise ValueError('Unexpected frozen RQ-VAE checkpoint')
    verify_files(directory, summary, list(summary['artifact_files_sha256']))
    contract = read_json(directory/'tokenizer_contract.json')
    if (contract != summary['tokenizer_contract'] or contract['semantic_codebooks'] != 3
            or contract['collision_suffix_digits'] != 1 or contract['n_digit'] != 4
            or contract['n_user_tokens'] != 0 or contract['padding_token'] != 0
            or contract['label_length'] != 5
            or any(contract['serving_catalog_reference'][s] != 'base_train'
                   for s in ('base_train', 'base_val'))):
        raise ValueError('Unsupported frozen tokenizer contract or validation catalog')
    mapping = read_json(directory/'base_train/item2tokens.json')
    registry = read_json(directory/'base_train/registry.json')
    validate_registry(registry)
    if set(mapping) != set(registry['id2item'][1:]):
        raise ValueError('SID/registry item membership mismatch')
    tokenizer = FrozenSIDTokenizer(mapping, contract['codebook_capacities'], contract['max_history'])
    for key in ('vocab_size', 'max_token_seq_len'):
        if getattr(tokenizer, key) != contract[key]:
            raise ValueError('Tokenizer size mismatch')
    if tokenizer.eos_token != contract['eos_token']:
        raise ValueError('EOS mismatch')
    if len(mapping) != summary['catalogs']['base_train']['items']:
        raise ValueError('Training catalog size mismatch')
    legal_sids = set(tokenizer.item2tokens.values())
    arrays, audits = {}, {}
    for split in ('base_train', 'base_val'):
        with np.load(directory/'tiger'/(split+'.npz'), allow_pickle=False) as file:
            values = {k: file[k] for k in file.files}
        audit = read_jsonl(directory/'tiger'/(split+'_audit.jsonl'))
        report = summary['tiger_splits'][split]
        n = len(values['labels'])
        if (set(values) != {'input_ids', 'attention_mask', 'labels',
                           'source_rows', 'target_global_item_ids'}
                or len(audit) != report['all_examples'] or n != report['encoded_examples']
                or values['input_ids'].shape != (n, tokenizer.max_token_seq_len)
                or values['attention_mask'].shape != values['input_ids'].shape
                or values['labels'].shape != (n, 5)
                or values['source_rows'].shape != (n,)
                or values['target_global_item_ids'].shape != (n,)
                or n == 0 or any(v.dtype.kind not in 'iu' for v in values.values())
                or not np.array_equal(values['attention_mask'], values['input_ids'] != 0)
                or np.any(values['input_ids'] >= tokenizer.vocab_size)
                or np.any(values['input_ids'] < 0)):
            raise ValueError('Malformed frozen tensors/counts/masks: '+split)
        source_rows = []
        sample_ids = set()
        for i, row in enumerate(audit):
            if (row['source_row'] != i or row['sample_id'] in sample_ids
                    or row['catalog_version'] != 'base_train'
                    or row['target_servable'] != (row['target_raw_item_id'] in mapping)):
                raise ValueError('Audit sample/catalog mismatch')
            sample_ids.add(row['sample_id'])
            if row['target_servable']:
                j = len(source_rows)
                raw = row['target_raw_item_id']
                source_rows.append(i)
                if (row['tensor_row'] != j or values['target_global_item_ids'][j]
                        != registry['item2id'][raw]-1
                        or values['labels'][j].tolist() != [*mapping[raw], tokenizer.eos_token]):
                    raise ValueError('Audit/target/label row mismatch')
                ids = values['input_ids'][j]
                eos = np.flatnonzero(ids == tokenizer.eos_token)
                if (len(eos) != 1 or eos[0] % 4 or np.any(ids[eos[0]+1:] != 0)):
                    raise ValueError('Missing/misplaced history EOS or trailing padding')
                missing = 0
                for block in ids[:eos[0]].reshape(-1, 4):
                    if np.all(block == 0):
                        missing += 1
                    elif tuple(block.tolist()) not in legal_sids:
                        raise ValueError('History item outside eligible SID catalog')
                if (missing != row['unavailable_history_events']
                        or row['all_history_unavailable'] != (missing == eos[0]//4)):
                    raise ValueError('Unavailable history mask/audit mismatch')
            elif row['tensor_row'] is not None:
                raise ValueError('Unavailable target unexpectedly has a labeled tensor')
        if (source_rows != values['source_rows'].tolist()
                or len(audit)-n != report['targets_outside_frozen_catalog']
                or split == 'base_train' and (n != len(audit)
                    or any(r['unavailable_history_events'] for r in audit))):
            raise ValueError('Audit denominator/source indices mismatch')
        arrays[split], audits[split] = values, audit
    return summary, tokenizer, arrays, audits


def make_batch(arrays, indices, device):
    """Remove trailing padding only; preserve unavailable history positions."""
    ids = arrays['input_ids'][indices]
    mask = arrays['attention_mask'][indices]
    active = np.flatnonzero(np.any(mask, axis=0))
    if not len(active):
        raise ValueError('Empty attention mask, including EOS')
    end = int(active[-1])+1
    return dict(input_ids=torch.as_tensor(ids[:, :end].astype(np.int64), device=device),
        attention_mask=torch.as_tensor(mask[:, :end].astype(np.int64), device=device),
        labels=torch.as_tensor(arrays['labels'][indices].astype(np.int64), device=device))


def probe_rows(audit, size, seed):
    # Select from ALL queries, including unavailable targets, without popularity stratification.
    ranked = sorted(range(len(audit)), key=lambda i: hashlib.sha256(
        (str(seed)+':'+str(audit[i]['sample_id'])).encode()).hexdigest())
    return sorted(ranked[:min(size, len(audit))])


def catalog_prefix(tokenizer):
    def callback(batch_id, prefix):
        try:
            return tokenizer.allowed_next_tokens(prefix.tolist())
        except ValueError:
            # Beam search may create inactive, -inf beams when a trie level has
            # fewer children than num_beams. Terminate them; validate every final beam.
            return [tokenizer.eos_token]
    return callback


@torch.no_grad()
def evaluate(trainer, tokenizer, arrays, audit, source_indices, device, batch_size,
             progress=True):
    """Repository metrics, with unavailable targets counted as misses in full ranking."""
    from tqdm import tqdm
    before_rng, was_training = rng_state(), trainer.model.training
    trainer.model.eval()
    chosen = [audit[i]['tensor_row'] for i in source_indices if audit[i]['target_servable']]
    sums = {f'{metric}@{k}': 0.0 for metric in trainer.config['metrics'] for k in trainer.config['topk']}
    nll_sum = 0.0
    legal = set(tokenizer.item2tokens.values())
    maxk = trainer.evaluator.maxk
    iterator = range(0, len(chosen), batch_size)
    if progress:
        iterator = tqdm(iterator, desc='Base validation', unit='batch')
    try:
        for start in iterator:
            indices = np.asarray(chosen[start:start+batch_size])
            batch = make_batch(arrays, indices, device)
            output = trainer.model(batch)
            losses = F.cross_entropy(output.logits.transpose(1, 2), batch['labels'], reduction='none')
            nll_sum += float(losses.sum())
            predicted = trainer.model.generate(batch, n_return_sequences=maxk, return_scores=True,
                                                prefix_allowed_tokens_fn=catalog_prefix(tokenizer))
            preds = predicted['preds'].cpu().tolist()
            sequences = predicted['sequences']
            if (sequences.shape[-1] != 6 or torch.any(sequences[:, :, 0] != 0)
                    or torch.any(sequences[:, :, -1] != tokenizer.eos_token)
                    or not torch.isfinite(predicted['scores']).all()
                    or any(len({tuple(x) for x in row}) != maxk
                           or any(tuple(x) not in legal for x in row) for row in preds)):
                raise ValueError('Invalid, duplicate, non-finite or ineligible final ranking beam')
            metrics = trainer.evaluator.calculate_metrics(predicted, batch['labels'])
            for key, value in metrics.items():
                sums[key] += float(value.sum())
    finally:
        trainer.model.train(was_training)
        restore_rng(before_rng)
    total, available = len(source_indices), len(chosen)
    if total == 0:
        raise ValueError('Empty evaluation query set')
    return dict(all_queries=total, servable_queries=available,
        unavailable_targets=total-available, decoded_queries=available,
        full_denominator={k: v/total for k, v in sums.items()},
        conditional_on_servable={k: v/available if available else None for k, v in sums.items()},
        mean_sequence_nll_servable=nll_sum/available if available else None,
        mean_token_ce_servable=nll_sum/(available*5) if available else None,
        ranking_catalog='base_train', invalid_predictions=0,
        seen_item_filter=False, decoding='catalog_prefix_beam_search', num_beams=trainer.config['num_beams'])


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run(args):
    started = time.perf_counter()
    if args.max_steps < 1 or any(getattr(args, k) < 1 for k in
            ('batch_size', 'eval_batch_size', 'probe_size', 'eval_every', 'log_every', 'save_every')):
        raise ValueError('Steps, intervals and batch sizes must be positive')
    if args.lr <= 0 or args.weight_decay < 0 or args.warmup_steps < 0:
        raise ValueError('Invalid optimizer settings')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite output: '+str(output))
    sid_dir = Path(args.sid_dir).resolve()
    summary, tokenizer, arrays, audits = load_sids(sid_dir, args.expected_rqvae_sha256)
    if (not args.topk or min(args.topk) < 1 or len(set(args.topk)) != len(args.topk)
            or args.num_beams < max(args.topk) or args.num_beams < 2
            or len(tokenizer.item2tokens) < max(args.topk)):
        raise ValueError('Invalid top-k/beam count or insufficient eligible catalog')
    model_config = dict(MODEL_DEFAULTS)
    if args.model_config:
        overrides = read_json(Path(args.model_config))
        if set(overrides)-set(MODEL_DEFAULTS):
            raise ValueError('Unknown model configuration override')
        model_config.update(overrides)
    config = dict(model_config, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, warmup_steps=args.warmup_steps, scheduler='constant_with_warmup',
        max_grad_norm=1.0, seed=args.seed, device=args.device, precision='float32',
        num_beams=args.num_beams, topk=sorted(args.topk), metrics=['recall', 'ndcg'],
        probe_size=args.probe_size, eval_batch_size=args.eval_batch_size, eval_every=args.eval_every,
        checkpoint_selection='best_probe_full_denominator_ndcg@'+str(max(args.topk)),
        trailing_padding='trim_only_after_last_active_position')
    root = Path(__file__).resolve().parents[1]
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'transformers')}
    selected_rows = probe_rows(audits['base_val'], args.probe_size, args.seed)
    binding = dict(protocol_sha256=summary['protocol_sha256'],
        rqvae_checkpoint_sha256=summary['rqvae_checkpoint_sha256'],
        sid_summary_sha256=sha256(sid_dir/'summary.json'),
        sid_artifact_files_sha256=summary['artifact_files_sha256'],
        tokenizer_contract=summary['tokenizer_contract'],
        source_files_sha256={name: sha256(root/name) for name in SOURCE_FILES},
        package_versions=versions, python=platform.python_version(),
        probe_source_rows=selected_rows)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable')
    seed_all(args.seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = TIGER(config, None, tokenizer).to(device)
    trainer = Trainer.for_temporal(config, model, tokenizer)
    generator = torch.Generator().manual_seed(args.seed+1)
    sampler = dict(order=None, cursor=0, optimizer_steps=0, processed_examples=0, epochs_started=0)
    best_state, best_metrics, best_step = None, None, None
    history = []
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        if (checkpoint.get('format') != 'incremental_base_tiger_v1'
                or checkpoint['binding'] != binding or checkpoint['config'] != config):
            raise ValueError('Resume source/data/config/package binding mismatch')
        if checkpoint['sampler']['optimizer_steps'] >= args.max_steps:
            raise ValueError('Resume target must exceed completed optimizer steps')
        model.load_state_dict(checkpoint['state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler'])
        sampler = checkpoint['sampler']
        generator.set_state(checkpoint['generator'])
        best_state, best_metrics, best_step = (checkpoint[k] for k in ('best_state_dict', 'best_metrics', 'best_step'))
        history = checkpoint['validation_history']
        restore_rng(checkpoint['rng'])
    output.mkdir(parents=True)
    write_json(output/'config.json', config)
    write_json(output/'probe.json', [dict(source_row=i, sample_id=audits['base_val'][i]['sample_id'])
                                     for i in selected_rows])
    initial_step = sampler['optimizer_steps']
    metric_key = 'ndcg@'+str(max(args.topk))

    def save_last():
        payload = dict(format='incremental_base_tiger_v1', config=config, binding=binding,
            state_dict=cpu_state(model), optimizer=trainer.optimizer.state_dict(),
            scheduler=trainer.scheduler.state_dict(), sampler=copy.deepcopy(sampler),
            generator=generator.get_state(), rng=rng_state(), best_state_dict=best_state,
            best_metrics=best_metrics, best_step=best_step, validation_history=history)
        temp = output/'last.pth.tmp'
        torch.save(payload, temp)
        os.replace(temp, output/'last.pth')

    def on_step(loss, grad_norm):
        nonlocal best_state, best_metrics, best_step
        step = sampler['optimizer_steps']
        with (output/'train_metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(step=step, loss=loss, gradient_norm=grad_norm,
                lr=trainer.optimizer.param_groups[0]['lr']), allow_nan=False)+'\n')
        if step % args.log_every == 0 or step == initial_step+1 or step == args.max_steps:
            elapsed = time.perf_counter()-started
            print(f'TIGER step {step}/{args.max_steps} | token CE {loss:.6f} | '
                  f'lr {trainer.optimizer.param_groups[0]["lr"]:.6g} | elapsed {elapsed:.1f}s', flush=True)
        if step % args.eval_every == 0 or step == args.max_steps:
            results = evaluate(trainer, tokenizer, arrays['base_val'], audits['base_val'],
                               selected_rows, device, args.eval_batch_size)
            history.append(dict(step=step, metrics=results))
            if best_metrics is None or results['full_denominator'][metric_key] > best_metrics['full_denominator'][metric_key]:
                best_state, best_metrics, best_step = cpu_state(model), results, step
            print('Validation probe: '+json.dumps(dict(step=step, best_step=best_step,
                metrics=results), allow_nan=False), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_last()

    print(f'TIGER ready | {len(arrays["base_train"]["labels"])} train queries | '
          f'{len(tokenizer.item2tokens)} eligible items | {model.n_parameters}', flush=True)
    trainer.fit_steps(arrays['base_train'], sampler, generator, args.max_steps,
        lambda a, i: make_batch(a, i, device), on_step)
    # Export the selected model independently of current training state for E1 reuse.
    torch.save(dict(format='incremental_base_tiger_selected_v1', state_dict=best_state,
        config=config, binding=binding, optimizer_step=best_step,
        validation_probe=best_metrics), output/'best.pth')
    selected = torch.load(output/'best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['state_dict'])
    reload_equal = all(torch.equal(value.cpu(), selected['state_dict'][key])
                       for key, value in model.state_dict().items())
    final_validation = None
    if not args.smoke:
        final_validation = evaluate(trainer, tokenizer, arrays['base_val'], audits['base_val'],
            list(range(len(audits['base_val']))), device, args.eval_batch_size)
        write_json(output/'base_validation.json', final_validation)
    report = dict(stage='E0_base_content_TIGER', execution_status='completed',
        sid_dir=str(sid_dir), output=str(output), protocol_sha256=summary['protocol_sha256'],
        rqvae_checkpoint_sha256=summary['rqvae_checkpoint_sha256'], config=config, binding=binding,
        model_implementation='repository genrec.models.TIGER.model.TIGER',
        model_initialization='random T5Config; no pretrained model/tokenizer download',
        parameters=sum(p.numel() for p in model.parameters()), optimizer_steps=sampler['optimizer_steps'],
        steps_this_run=sampler['optimizer_steps']-initial_step,
        processed_examples=sampler['processed_examples'], sampler_epochs_started=sampler['epochs_started'],
        resume_from=str(Path(args.resume_from).resolve()) if args.resume_from else None,
        checkpoint_selection=config['checkpoint_selection'], best_optimizer_step=best_step,
        best_validation_probe=best_metrics, validation_probe_queries=len(selected_rows),
        full_base_validation=final_validation, selected_checkpoint_reload_equal=reload_equal,
        training_catalog='base_train', ranking_catalog='base_train',
        eligible_items=len(tokenizer.item2tokens), graph_used=False, rqvae_frozen=True,
        sid_refit=False, content_encoder_finetuning=False, update_training_status='not_run',
        future_test_status='not_run', E0_base_end_to_end_status='not_run' if args.smoke else 'completed',
        E0_full_status='not_run', E1_status='not_run', execution_mode='smoke' if args.smoke else 'full_base_validation',
        python=platform.python_version(), package_versions=versions,
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        memory_scope='this process, model construction through final validation; excludes other processes',
        runtime_seconds=time.perf_counter()-started,
        artifact_files_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sid-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expected-rqvae-sha256')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--probe-size', type=int, default=512)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--num-beams', type=int, default=20)
    p.add_argument('--topk', type=int, nargs='+', default=[5, 10])
    p.add_argument('--resume-from')
    p.add_argument('--model-config', help='Optional JSON model architecture overrides for synthetic checks')
    p.add_argument('--smoke', action='store_true', help='Run probe evaluation, defer full Base validation')
    return p
