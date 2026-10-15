"""Fixed metadata content and stage-scoped embeddings for pure-content SIDs.

Reuse prepare_dataset.encode_sent_emb. No graph/data/model training stack.
"""
import hashlib
import json
import platform
import re
import time
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from genrec.temporal import iter_gzip_records

STAGES = ('base_train', 'update_train', 'update')
FIELDS = ('title', 'brand', 'categories', 'description')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def _new_output(output):
    output = Path(output).resolve()
    pending = output.with_name(output.name + '.incomplete')
    if output.exists() or pending.exists():
        raise FileExistsError('Use a new output directory without an incomplete sibling')
    return output, pending


def _commit(pending, output):
    if output.exists():
        raise FileExistsError('Output created concurrently')
    pending.rename(output)


def _payload_hashes(directory):
    return {str(p.relative_to(directory)): file_sha256(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def _verify_hashes(directory, expected, required):
    if not required.issubset(expected):
        raise ValueError('Incomplete artifact manifest')
    for name in sorted(required):
        if file_sha256(directory / name) != expected[name]:
            raise ValueError(f'Artifact checksum mismatch: {name}')


def _text_parts(value):
    if value is None:
        return []
    if isinstance(value, str):
        clean = ' '.join(unescape(re.sub(r'<[^>]*>', ' ', value)).split())
        return [clean] if clean else []
    if isinstance(value, (list, tuple)):
        return [part for child in value for part in _text_parts(child)]
    raise ValueError('Content fields must be strings, string lists, or null')


def content_text(record):
    """Whitelist only product fields; no raw-ID tokens or behavioural fields."""
    pieces = []
    for field in FIELDS:
        values = _text_parts(record.get(field))
        if values:
            pieces.append(field.capitalize() + ': ' + '; '.join(values))
    return '\n'.join(pieces)


def _validate_registry(registry):
    ids = registry['id2item']
    mapping = registry['item2id']
    if not ids or ids[0] != '[PAD]' or mapping.get('[PAD]') != 0:
        raise ValueError('Registry must reserve padding ID 0')
    if len(mapping) != len(ids) or any(mapping.get(raw) != i for i, raw in enumerate(ids)):
        raise ValueError('Registry raw-ID round trip mismatch')


def prepare_content(metadata_path, protocol_dir, output, missing_policy='error'):
    started = time.perf_counter()
    if missing_policy not in ('error', 'placeholder'):
        raise ValueError('Invalid missing metadata policy')
    metadata_path, protocol_dir = Path(metadata_path).resolve(), Path(protocol_dir).resolve()
    output, pending = _new_output(output)
    source = _json(protocol_dir / 'summary.json')
    if source.get('stage') != 'E0_data_preparation' or source.get('execution_status') != 'completed':
        raise ValueError('Expected completed Round 3 temporal preparation')
    required = {'registries.json', 'protocol_manifest.json', 'base_train.jsonl',
                'base_val.jsonl', 'update_train.jsonl', 'update_val.jsonl', 'future_test.jsonl'}
    _verify_hashes(protocol_dir, source['artifact_files_sha256'], required)
    if _json(protocol_dir / 'protocol_manifest.json')['protocol_sha256'] != source['protocol_sha256']:
        raise ValueError('Protocol identity mismatch')
    registries = _json(protocol_dir / 'registries.json')
    for stage in STAGES:
        _validate_registry(registries[stage])
        if registries['update']['id2item'][:len(registries[stage]['id2item'])] != registries[stage]['id2item']:
            raise ValueError('Stage registries do not preserve old item IDs')
    wanted = set(registries['update']['id2item'][1:])
    texts, seen, duplicates = {}, set(), 0
    metadata_records = 0
    for index, record in enumerate(iter_gzip_records(metadata_path), 1):
        metadata_records = index
        raw = record.get('asin')
        if raw not in wanted:
            continue
        text = content_text(record)
        if raw in seen:
            duplicates += 1
            if texts[raw] != text:
                raise ValueError(f'Conflicting allowed-field metadata for item {raw}')
        else:
            seen.add(raw)
            texts[raw] = text
    missing = wanted - seen
    empty = {raw for raw, text in texts.items() if not text}
    if (missing or empty) and missing_policy == 'error':
        raise ValueError(f'Missing metadata records: {len(missing)}; empty allowed-field text: {len(empty)}. '
                         'Inspect metadata or explicitly choose --missing-policy placeholder.')
    for raw in missing | empty:
        texts[raw] = 'Product information unavailable.'
    pending.mkdir(parents=True, exist_ok=False)
    stats = {}
    for stage in STAGES:
        raw_ids = registries[stage]['id2item'][1:]
        directory = pending / stage
        directory.mkdir()
        with (directory / 'content.jsonl').open('w', encoding='utf-8') as handle:
            for row, raw in enumerate(raw_ids):
                handle.write(json.dumps(dict(raw_item_id=raw, gr_item_id=row+1,
                    gensid_global_item_id=row, feature_row=row, text=texts[raw],
                    metadata_missing=raw in missing, allowed_text_empty=raw in empty),
                    ensure_ascii=False) + '\n')
        _write(directory / 'registry.json', registries[stage])
        stats[stage] = dict(items=len(raw_ids), missing_records=sum(raw in missing for raw in raw_ids),
                           empty_texts=sum(raw in empty for raw in raw_ids))
    result = dict(stage='E0_content_metadata', execution_status='completed',
        protocol_dir=str(protocol_dir), protocol_sha256=source['protocol_sha256'],
        reviews_file_sha256=source['reviews_file_sha256'], metadata_path=str(metadata_path),
        metadata_file_sha256=file_sha256(metadata_path), metadata_records_scanned=metadata_records,
        duplicate_eligible_metadata_records=duplicates, stages=stats, allowed_fields=list(FIELDS),
        excluded_fields='all other fields, including reviews/ratings/salesRank/related',
        missing_policy=missing_policy, encoder_status='not_run', graph_used=False,
        metadata_time_policy='static metadata without historical publication dates; offline approximation',
        catalog_policy='only items observed by update end; stage-specific prefix catalogs',
        preprocessing='HTML tag removal, entity decoding, whitespace collapse; field labels and semicolon lists',
        training_status='not_run', E0_full_status='not_run', E1_status='not_run',
        python=platform.python_version())
    result['artifact_files_sha256'] = _payload_hashes(pending)
    result['runtime_seconds'] = time.perf_counter() - started
    _write(pending / 'summary.json', result)
    _commit(pending, output)
    return result


def resolve_encoder(model, revision, local_files_only=False):
    path = Path(model).expanduser()
    if path.is_dir():
        files = [p for p in sorted(path.rglob('*')) if p.is_file()]
        if not files:
            raise ValueError('Empty local encoder directory')
        content = [(str(p.relative_to(path)), file_sha256(p)) for p in files]
        digest = hashlib.sha256(json.dumps(content, separators=(',', ':')).encode()).hexdigest()
        return str(path.resolve()), None, dict(kind='local_model_tree', sha256=digest, files=len(files))
    if revision and re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        commit = revision.lower()
    else:
        if local_files_only:
            raise ValueError('Offline Hub loading requires a full 40-character --model-revision commit')
        from huggingface_hub import HfApi
        commit = HfApi().model_info(model, revision=revision or 'main').sha
        if not commit or not re.fullmatch(r'[0-9a-fA-F]{40}', commit):
            raise ValueError('Could not resolve an immutable encoder revision')
    return model, commit, dict(kind='huggingface_commit', model=model, commit=commit)


def encode_content(content_dir, output, model='sentence-transformers/sentence-t5-base',
                   revision=None, device='cpu', batch_size=32, max_length=256,
                   expected_dim=768, local_files_only=False, *, model_factory=None,
                   encoder_resolver=None):
    started = time.perf_counter()
    if batch_size < 1 or max_length < 1 or expected_dim < 1:
        raise ValueError('Batch size, max length and expected dimension must be positive')
    content_dir = Path(content_dir).resolve()
    output, pending = _new_output(output)
    source = _json(content_dir / 'summary.json')
    if source.get('stage') != 'E0_content_metadata' or source.get('execution_status') != 'completed':
        raise ValueError('Expected completed content preparation')
    required = {f'{s}/{name}' for s in STAGES for name in ('registry.json', 'content.jsonl')}
    _verify_hashes(content_dir, source['artifact_files_sha256'], required)
    all_rows = {}
    for stage in STAGES:
        rows = [json.loads(line) for line in (content_dir / stage / 'content.jsonl').read_text().splitlines()]
        registry = _json(content_dir / stage / 'registry.json')
        _validate_registry(registry)
        if len(rows) != len(registry['id2item'])-1 or len(rows) != source['stages'][stage]['items']:
            raise ValueError('Content row count mismatch')
        for i, row in enumerate(rows):
            if (row['feature_row'] != i or row['gensid_global_item_id'] != i or row['gr_item_id'] != i+1
                    or row['raw_item_id'] != registry['id2item'][i+1] or not row['text']):
                raise ValueError('Content identity/order mismatch')
        all_rows[stage] = rows
    for stage in STAGES:
        if all_rows['update'][:len(all_rows[stage])] != all_rows[stage]:
            raise ValueError('Stage content differs from the shared immutable content')
    resolver = encoder_resolver or resolve_encoder
    model, resolved_revision, identity = resolver(model, revision, local_files_only)
    # One fixed encoder pass over the observed Update catalog; stage views
    # are copied from shared features. No encoder fine-tuning or corpus fitting.
    rows = all_rows['update']
    dataset = SimpleNamespace(n_items=len(rows)+1,
        id_mapping={'id2item': ['[PAD]'] + [r['raw_item_id'] for r in rows]},
        item2meta={r['raw_item_id']: r['text'] for r in rows})
    config = dict(metadata='sentence', sent_emb_model=model, sent_emb_revision=resolved_revision,
        sent_emb_local_files_only=local_files_only, device=device,
        sent_emb_batch_size=batch_size, sent_emb_max_length=max_length,
        sent_emb_expected_dim=expected_dim)
    pending.mkdir(parents=True, exist_ok=False)
    from prepare_dataset import encode_sent_emb
    values = encode_sent_emb(dataset, str(pending / 'content.sent_emb'), config,
                             model_factory=model_factory)
    # Avoid carrying ambiguous raw binary features downstream: .npy plus row map.
    (pending / 'content.sent_emb').unlink()
    if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
        raise ValueError('Content encoder returned nonfinite or zero-vector rows')
    stats = {}
    for stage in STAGES:
        directory = pending / stage
        directory.mkdir()
        subset = values[:len(all_rows[stage])]
        np.save(directory / 'embeddings.npy', subset)
        _write(directory / 'feature_rows.json', [{k: row[k] for k in
            ('raw_item_id', 'gr_item_id', 'gensid_global_item_id', 'feature_row')}
            for row in all_rows[stage]])
        _write(directory / 'registry.json', _json(content_dir / stage / 'registry.json'))
        norms = np.linalg.norm(subset, axis=1)
        stats[stage] = dict(shape=list(subset.shape), dtype=str(subset.dtype),
            norm_min=float(norms.min()), norm_max=float(norms.max()),
            norm_mean=float(norms.mean()))
    result = dict(stage='E0_fixed_content_embeddings', execution_status='completed',
        content_dir=str(content_dir), protocol_sha256=source['protocol_sha256'],
        metadata_file_sha256=source['metadata_file_sha256'],
        content_summary_sha256=file_sha256(content_dir / 'summary.json'),
        encoder=identity, encoder_config=config, stages=stats,
        normalization='none added by this code; preserve the selected encoder output',
        projection='none', graph_used=False, static_metadata_approximation=True,
        future_only_items_encoded=False, encoder_finetuning=False,
        training_catalogs=['base_train', 'update_train'],
        update_view='deployment catalog including update-validation-only items; not a training view',
        shared_features='encode fixed content once; Base features copied unchanged into Update views',
        training_status='not_run', E0_full_status='not_run', E1_status='not_run',
        python=platform.python_version())
    try:
        from importlib import metadata
        result['package_versions'] = {name: metadata.version(name) for name in
                                     ('numpy', 'sentence-transformers', 'transformers', 'torch')}
    except Exception:
        result['package_versions'] = 'not_available_in_synthetic_environment'
    root = Path(__file__).resolve().parents[1]
    result['source_files_sha256'] = {name: file_sha256(root/name) for name in (
        'prepare_dataset.py', 'genrec/temporal_content.py', 'prepare_incremental_content.py')}
    result['artifact_files_sha256'] = _payload_hashes(pending)
    result['runtime_seconds'] = time.perf_counter() - started
    _write(pending / 'summary.json', result)
    _commit(pending, output)
    return result
