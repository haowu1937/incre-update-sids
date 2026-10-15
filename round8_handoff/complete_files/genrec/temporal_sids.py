"""Freeze Base RQ-VAE, append collision slots and prepare causal TIGER tensors."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def verify_files(directory, report, names):
    for name in names:
        if sha256(directory/name) != report['artifact_files_sha256'].get(name):
            raise ValueError('Artifact checksum mismatch: ' + name)


def validate_registry(registry):
    raw, mapping = registry['id2item'], registry['item2id']
    if (not raw or raw[0] != '[PAD]' or len(mapping) != len(raw)
            or any(mapping.get(item) != i for i, item in enumerate(raw))):
        raise ValueError('Registry raw-ID round trip mismatch')


def code_to_tokens(code, capacities):
    if len(code) != len(capacities):
        raise ValueError('Incorrect full SID length')
    tokens, offset = [], 1
    for value, capacity in zip(code, capacities):
        if type(value) is not int or not 0 <= value < capacity:
            raise ValueError('SID index outside fixed codebook capacity')
        tokens.append(value + offset)
        offset += capacity
    return tokens


def allocate_suffixes(raw_items, semantic_codes, semantic_capacities, suffix_capacity,
                      previous=None):
    """Append only, in stable registry order; never renumber an existing slot.

    This export does not update RQ-VAE codes. A future codebook-version change
    requires its own versioned allocation policy, not an in-place mutation.
    """
    capacities = list(semantic_capacities) + [suffix_capacity]
    if (len(semantic_capacities) != 3 or any(type(x) is not int or x < 1 for x in capacities)
            or len(raw_items) != len(semantic_codes) or len(set(raw_items)) != len(raw_items)
            or '[PAD]' in raw_items):
        raise ValueError('Invalid allocator items or capacities')
    ledger = copy.deepcopy(previous) if previous is not None else dict(
        schema_version=1, semantic_capacities=list(semantic_capacities),
        suffix_capacity=suffix_capacity, entries=[])
    if (ledger.get('schema_version') != 1 or ledger['semantic_capacities'] != list(semantic_capacities)
            or ledger['suffix_capacity'] != suffix_capacity):
        raise ValueError('Previous allocation capacity/schema mismatch')
    entries = ledger['entries']
    if len(raw_items) < len(entries):
        raise ValueError('Cannot shrink an append-only allocation registry')
    next_slot, occupied = {}, set()
    for i, entry in enumerate(entries):
        expected = dict(raw_item_id=raw_items[i], gr_item_id=i+1,
                        gensid_global_item_id=i, feature_row=i)
        if any(entry.get(k) != v for k,v in expected.items()) or entry['semantic_code'] != list(semantic_codes[i]):
            raise ValueError('Existing item/order/code changed during append-only extension')
        code = [*entry['semantic_code'], entry['collision_suffix']]
        code_to_tokens(code, capacities)
        prefix = tuple(entry['semantic_code'])
        key = (*prefix, entry['collision_suffix'])
        if key in occupied:
            raise ValueError('Duplicate reserved full SID in previous registry')
        occupied.add(key)
        next_slot[prefix] = max(next_slot.get(prefix, 0), entry['collision_suffix'] + 1)
    for i in range(len(entries), len(raw_items)):
        prefix = list(semantic_codes[i])
        # Validate all three semantic indices independently of the suffix.
        code_to_tokens(prefix, semantic_capacities)
        suffix = next_slot.get(tuple(prefix), 0)
        if suffix >= suffix_capacity:
            raise ValueError('Collision suffix capacity exhausted; vocabulary must not grow silently')
        entries.append(dict(raw_item_id=raw_items[i], gr_item_id=i+1,
            gensid_global_item_id=i, feature_row=i, semantic_code=prefix,
            collision_suffix=suffix))
        next_slot[tuple(prefix)] = suffix + 1
    return ledger


def maps_from_ledger(ledger):
    capacities = ledger['semantic_capacities'] + [ledger['suffix_capacity']]
    semantic, tokens = {}, {}
    for entry in ledger['entries']:
        raw = entry['raw_item_id']
        code = [*entry['semantic_code'], entry['collision_suffix']]
        semantic[raw] = code
        tokens[raw] = code_to_tokens(code, capacities)
    return semantic, tokens


def load_protocol(directory):
    summary = read_json(directory/'summary.json')
    if summary.get('stage') != 'E0_data_preparation' or summary.get('execution_status') != 'completed':
        raise ValueError('Expected completed temporal data artifacts')
    verify_files(directory, summary, ['protocol_manifest.json','registries.json',
                                      'base_train.jsonl','base_val.jsonl'])
    manifest, registries = read_json(directory/'protocol_manifest.json'), read_json(directory/'registries.json')
    canonical = dict(manifest)
    declared = canonical.pop('protocol_sha256')
    actual = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                                      separators=(',',':'), allow_nan=False).encode()).hexdigest()
    if declared != actual or declared != summary['protocol_sha256']:
        raise ValueError('Canonical protocol identity mismatch')
    if (manifest.get('validation_refit') is not False
            or manifest.get('same_timestamp_policy') != 'exclude_all_simultaneous_events_from_history'):
        raise ValueError('Expected no-refit protocol with strict simultaneous-event exclusion')
    if any(manifest['catalog_reference'][stage] != 'base_train' for stage in ('base_train','base_val')):
        raise ValueError('Base validation must use the Base training catalog')
    for stage in ('base_train','base'):
        validate_registry(registries[stage])
        if len(registries[stage]['id2item'])-1 != manifest['catalog_sizes'][stage]:
            raise ValueError('Protocol catalog size mismatch')
    ids = registries['base_train']['id2item']
    if registries['base']['id2item'][:len(ids)] != ids:
        raise ValueError('Base registry does not extend Base training IDs')
    max_history = manifest['max_history']
    if type(max_history) is not int or max_history < 1:
        raise ValueError('Invalid protocol history length')
    train_cut, base_cut = (manifest['cutoffs_exclusive'][key] for key in ('base_train','base'))
    catalog, full_catalog = set(ids[1:]), set(registries['base']['id2item'][1:])
    examples, seen = {}, set()
    for stage in ('base_train','base_val'):
        rows = read_jsonl(directory/(stage+'.jsonl'))
        if len(rows) != manifest['windows'][stage]['example_count']:
            raise ValueError('Protocol example count mismatch')
        for row in rows:
            items, times = row['item_seq'], row['time_seq']
            if (len(items) < 2 or len(items)-1 > max_history or len(times) != len(items)
                    or any(type(t) is not int for t in times)
                    or times[-1] != row['target_time']
                    or any(t >= times[-1] for t in times[:-1])
                    or times[:-1] != sorted(times[:-1])
                    or any(item not in full_catalog for item in items)):
                raise ValueError('Invalid causal history or Base item membership')
            expected_flags = [item in catalog for item in items[:-1]]
            target_servable = items[-1] in catalog
            if (row['catalog_version'] != 'base_train'
                    or row['history_available'] != expected_flags
                    or any(type(v) is not bool for v in row['history_available'])
                    or type(row['target_servable']) is not bool
                    or row['target_servable'] != target_servable):
                raise ValueError('Availability flags/catalog mismatch')
            if ((stage == 'base_train' and times[-1] >= train_cut)
                    or (stage == 'base_val' and not train_cut <= times[-1] < base_cut)):
                raise ValueError('Example outside its temporal window')
            if row['sample_id'] in seen:
                raise ValueError('Repeated causal sample ID')
            seen.add(row['sample_id'])
        if (sum(not row['target_servable'] for row in rows) != manifest['windows'][stage]['unservable_targets']
                or sum(not all(row['history_available']) for row in rows)
                    != manifest['windows'][stage]['examples_with_unavailable_history']):
            raise ValueError('Protocol availability count mismatch')
        if stage == 'base_train' and any(not row['target_servable'] or not all(row['history_available']) for row in rows):
            raise ValueError('Training must contain only eligible Base items')
        examples[stage] = rows
    return summary, manifest, registries, examples


def prepare_split(rows, tokenizer, raw_to_global):
    encoded, masks, labels, global_targets, source_rows, audit = [], [], [], [], [], []
    unavailable_histories = 0
    masked_events, fully_masked_histories = 0, 0
    for i,row in enumerate(rows):
        unavailable = sum(not value for value in row['history_available'])
        unavailable_histories += unavailable > 0
        masked_events += unavailable
        full_mask = unavailable == len(row['history_available'])
        fully_masked_histories += full_mask
        tensor_row = None
        if row['target_servable']:
            input_ids, mask, target = tokenizer._tokenize_once(row)
            tensor_row = len(encoded)
            encoded.append(input_ids); masks.append(mask); labels.append(target)
            global_targets.append(raw_to_global[row['item_seq'][-1]])
            source_rows.append(i)
        audit.append(dict(source_row=i, tensor_row=tensor_row, sample_id=row['sample_id'],
            target_raw_item_id=row['item_seq'][-1], target_servable=row['target_servable'],
            catalog_version=row['catalog_version'], unavailable_history_events=unavailable,
            all_history_unavailable=full_mask))
    if not encoded:
        raise ValueError('No eligible target examples to encode')
    arrays = dict(input_ids=np.asarray(encoded,dtype=np.int32),
        attention_mask=np.asarray(masks,dtype=np.uint8), labels=np.asarray(labels,dtype=np.int32),
        target_global_item_ids=np.asarray(global_targets,dtype=np.int32),
        source_rows=np.asarray(source_rows,dtype=np.int32))
    if arrays['input_ids'].shape[1] != tokenizer.max_token_seq_len or arrays['labels'].shape[1] != tokenizer.n_digit+1:
        raise ValueError('Incorrect TIGER tensor lengths')
    stats = dict(all_examples=len(rows), encoded_examples=len(encoded),
        targets_outside_frozen_catalog=len(rows)-len(encoded),
        examples_with_unavailable_history=int(unavailable_histories),
        unavailable_history_events=int(masked_events),
        examples_with_all_history_unavailable=int(fully_masked_histories),
        input_shape=list(arrays['input_ids'].shape), label_shape=list(arrays['labels'].shape))
    return arrays, audit, stats


def export_base_sids(protocol_dir, features_dir, rqvae_dir, output, device='cuda',
                     suffix_capacity=256, expected_checkpoint_sha256=None):
    started = time.perf_counter()
    protocol_dir, features_dir, rqvae_dir, output = map(lambda x: Path(x).resolve(),
        (protocol_dir,features_dir,rqvae_dir,output))
    pending = output.with_name(output.name+'.incomplete')
    if output.exists() or pending.exists():
        raise FileExistsError('Use a new output directory without an incomplete sibling')
    if type(suffix_capacity) is not int or suffix_capacity < 1:
        raise ValueError('Collision suffix capacity must be a positive fixed integer')
    data_report, manifest, registries, examples = load_protocol(protocol_dir)
    print('Verified Base temporal protocol and availability flags',flush=True)
    from gensid.incremental_content import (load_base_features, load_checkpoint,
        build_model, implementation_binding, seed_all)
    import torch
    features, base_rows, feature_binding = load_base_features(features_dir)
    feature_report = read_json(features_dir/'summary.json')
    rq_report = read_json(rqvae_dir/'summary.json')
    if (rq_report.get('stage') != 'E0_base_content_rqvae'
            or rq_report.get('execution_status') != 'completed'
            or rq_report.get('training_catalog') != 'base_train'
            or rq_report.get('checkpoint_reload_assignments_equal') is not True):
        raise ValueError('Expected completed Base content RQ-VAE artifacts')
    verify_files(rqvae_dir, rq_report, ['config.json','last.pth','semantic_assignments.jsonl'])
    checkpoint_sha = sha256(rqvae_dir/'last.pth')
    if expected_checkpoint_sha256 is not None and checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError('Checkpoint differs from the explicitly selected Base model')
    checkpoint = load_checkpoint(rqvae_dir/'last.pth')
    config = checkpoint['config']
    feature_binding.update(implementation_binding())
    if (checkpoint['binding'] != feature_binding or rq_report['binding'] != feature_binding
            or checkpoint['config'] != rq_report['config']
            or checkpoint['sampler_state']['step'] != rq_report['optimizer_steps']
            or rq_report['protocol_sha256'] != manifest['protocol_sha256']):
        raise ValueError('Frozen RQ-VAE data/source/package/config/protocol binding mismatch')
    if feature_report['protocol_sha256'] != manifest['protocol_sha256']:
        raise ValueError('Feature/protocol mismatch')
    if read_json(features_dir/'base_train/registry.json') != registries['base_train']:
        raise ValueError('Feature/temporal Base training registry mismatch')
    capacities = config['model']['num_emb_list']
    if len(capacities) != 3:
        raise ValueError('Exactly three semantic codebooks required')
    original = read_jsonl(rqvae_dir/'semantic_assignments.jsonl')
    if len(original) != len(base_rows):
        raise ValueError('Base assignment row count mismatch')
    original_codes = []
    for row, known in zip(original, base_rows):
        if any(row.get(key) != value for key,value in known.items()):
            raise ValueError('Base assignment item ID mismatch')
        code_to_tokens(row['semantic_code'],capacities)
        original_codes.append(row['semantic_code'])
    # Base deployment includes items first observed during Base validation.
    # They are encoded ONLY for deployment after that window; validation still
    # uses the earlier Base training catalog and its original availability flags.
    verify_files(features_dir, feature_report, ['update_train/embeddings.npy',
        'update_train/feature_rows.json','update_train/registry.json'])
    full_registry = read_json(features_dir/'update_train/registry.json')
    validate_registry(full_registry)
    n_base = len(registries['base']['id2item'])-1
    n_train = len(features)
    if full_registry['id2item'][:n_base+1] != registries['base']['id2item']:
        raise ValueError('Shared features do not preserve Base deployment item IDs')
    full_rows = read_json(features_dir/'update_train/feature_rows.json')
    if len(full_rows) != len(full_registry['id2item'])-1:
        raise ValueError('Shared feature row count mismatch')
    for i,row in enumerate(full_rows[:n_base]):
        if row != dict(raw_item_id=full_registry['id2item'][i+1], gr_item_id=i+1,
            gensid_global_item_id=i,feature_row=i):
            raise ValueError('Shared feature ID mismatch')
    shared = np.load(features_dir/'update_train/embeddings.npy', mmap_mode='r', allow_pickle=False)
    if (shared.dtype != np.float32 or shared.ndim != 2
            or shared.shape[0] != len(full_rows) or shared.shape[1] != features.shape[1]
            or list(shared.shape) != feature_report['stages']['update_train']['shape']
            or not np.array_equal(shared[:n_train], features.numpy())
            or not np.isfinite(shared[:n_base]).all()):
        raise ValueError('Invalid shared feature values or changed Base prefix')
    export_device = torch.device(device)
    if export_device.type not in ('cpu','cuda') or (export_device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested export device unavailable')
    seed_all(config['seed'])
    model = build_model(config).to(export_device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('Frozen checkpoint contains nonfinite parameters')
    if not all(layer.initted for layer in model.rq.vq_layers):
        raise ValueError('Frozen checkpoint contains uninitialized codebooks')
    original_command = read_json(rqvae_dir/'config.json')['command']
    batch_size = original_command['eval_batch_size']
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Invalid original SID export batch size')
    def encode(values):
        output_codes = []
        with torch.no_grad():
            for i in range(0,len(values),batch_size):
                codes = model.get_indices(values[i:i+batch_size].to(export_device),None,use_sk=False)
                output_codes.extend(codes.cpu().tolist())
        return output_codes
    print('Checking frozen Base assignments and encoding Base deployment additions',flush=True)
    recomputed = encode(features)
    if recomputed != original_codes:
        raise ValueError('Frozen checkpoint encoding changed original Base assignments')
    new_values = torch.from_numpy(np.array(shared[n_train:n_base],copy=True))
    all_codes = [*original_codes,*encode(new_values)]
    raw_train, raw_base = registries['base_train']['id2item'][1:], registries['base']['id2item'][1:]
    train_ledger = allocate_suffixes(raw_train,original_codes,capacities,suffix_capacity)
    base_ledger = allocate_suffixes(raw_base,all_codes,capacities,suffix_capacity,previous=train_ledger)
    train_sids, train_tokens = maps_from_ledger(train_ledger)
    base_sids, base_tokens = maps_from_ledger(base_ledger)
    if any(base_tokens[raw] != tokens for raw,tokens in train_tokens.items()):
        raise ValueError('Deployment extension renumbered an existing full SID')
    token_capacities = capacities+[suffix_capacity]
    tokenizer = FrozenSIDTokenizer(train_tokens,token_capacities,manifest['max_history'])
    raw_to_global = {raw:i for i,raw in enumerate(raw_train)}
    splits = {stage: prepare_split(rows,tokenizer,raw_to_global) for stage,rows in examples.items()}
    print('Writing fixed-capacity SID snapshots and causal TIGER tensors',flush=True)
    pending.mkdir(parents=True)
    stage_stats = {}
    for stage,ledger,sids,tokens in (
            ('base_train',train_ledger,train_sids,train_tokens), ('base',base_ledger,base_sids,base_tokens)):
        directory = pending/stage;directory.mkdir()
        write_json(directory/'allocation_registry.json',ledger)
        write_json(directory/'item2sem_ids.json',sids)
        write_json(directory/'item2tokens.json',tokens)
        write_json(directory/'registry.json',registries[stage])
        buckets = Counter(tuple(row['semantic_code']) for row in ledger['entries'])
        stage_stats[stage] = dict(items=len(ledger['entries']),unique_semantic_codes=len(buckets),
            maximum_collision_bucket=max(buckets.values()),
            unique_full_sids=len(set(map(tuple,sids.values()))),
            highest_suffix_used=max(row['collision_suffix'] for row in ledger['entries']))
    directory = pending/'tiger';directory.mkdir()
    split_stats = {}
    for stage,(arrays,audit,stats) in splits.items():
        np.savez_compressed(directory/(stage+'.npz'),**arrays)
        with (directory/(stage+'_audit.jsonl')).open('w',encoding='utf-8') as handle:
            for row in audit:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        split_stats[stage] = stats
    tokenizer_contract = dict(semantic_codebooks=3,collision_suffix_digits=1,n_digit=4,
        codebook_capacities=token_capacities,padding_token=0,eos_token=tokenizer.eos_token,
        vocab_size=tokenizer.vocab_size,n_user_tokens=0,max_history=manifest['max_history'],
        max_token_seq_len=tokenizer.max_token_seq_len,label_length=tokenizer.n_digit+1,
        serving_catalog_reference=dict(base_train='base_train',base_val='base_train'),
        base_deployment_catalog='base',base_deployment_available_after=manifest['cutoffs_exclusive']['base'],
        unavailable_history_policy='four padding IDs and attention zeros at original positions; EOS active',
        unavailable_target_policy='audit every row; exclude from labeled tensors; count as misses in full-denominator ranking',
        ranking_catalog_policy='constrain decoding to eligible full SID prefixes; never admit deployment-only items in Base validation')
    write_json(pending/'tokenizer_contract.json',tokenizer_contract)
    root = Path(__file__).resolve().parents[1]
    sources = ['genrec/temporal_sids.py','genrec/frozen_sid_tokenizer.py','genrec/tokenizer.py',
               'prepare_incremental_sids.py']
    result = dict(stage='E0_base_frozen_sid_preparation',execution_status='completed',
        protocol_dir=str(protocol_dir),features_dir=str(features_dir),rqvae_dir=str(rqvae_dir),
        protocol_sha256=manifest['protocol_sha256'],rqvae_checkpoint_sha256=checkpoint_sha,
        rqvae_optimizer_steps=rq_report['optimizer_steps'],rqvae_frozen=True,rqvae_training_performed=False,
        graph_used=False,sinkhorn_used=False,content_encoder_finetuning=False,
        update_or_future_item_assignments_created=False,
        source_binding=dict(protocol_summary_sha256=sha256(protocol_dir/'summary.json'),
            feature_summary_sha256=sha256(features_dir/'summary.json'),
            rqvae_summary_sha256=sha256(rqvae_dir/'summary.json')),
        catalogs=stage_stats,tiger_splits=split_stats,tokenizer_contract=tokenizer_contract,
        original_base_assignments_equal=True,old_full_sids_unchanged_on_deployment_extension=True,
        collision_suffix_policy='zero-based persistent slots, append in temporal registry ID order; fixed capacity, overflow fails',
        recommendation_training_status='not_run',E0_full_status='not_run',E1_status='not_run',
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=str(export_device),
        source_files_sha256={name:sha256(root/name) for name in sources},
        artifact_files_sha256={path.relative_to(pending).as_posix():sha256(path)
            for path in sorted(pending.rglob('*')) if path.is_file()},
        runtime_seconds=time.perf_counter()-started)
    write_json(pending/'summary.json',result)
    if output.exists():
        raise FileExistsError('Output was created concurrently')
    pending.rename(output)
    return result
