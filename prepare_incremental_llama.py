"""Offline Llama content features using existing temporal catalogs.

Match Mitigating's text format and fixed-length pooling, not its temporal split.
Existing preparation/training modules and old feature files are never modified.
"""
import argparse
from datetime import datetime, timezone
import gzip
from importlib import metadata as packages
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from genrec.temporal import parse_record
from genrec.temporal_content import (
    STAGES, _json, _payload_hashes, _validate_registry, _verify_hashes,
    file_sha256,
)

REFERENCE = ('https://github.com/iskbaga/semantic-id-alignment/blob/main/'
             'notebooks/AmazonBeautyDownload.ipynb')
REFERENCE_BLOB_SHA = '20f09c2d081b8ddc7b7825d43ff3bf626e879521'
ROW_KEYS = ('raw_item_id', 'gr_item_id', 'gensid_global_item_id', 'feature_row')
SPLITS = ('base_train', 'base_val', 'update_train', 'update_val', 'future_test')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def mitigating_text(record):
    """Notebook preprocess: pandas null -> 'None', first category path, repr lists.

    For absent/empty categories explicitly use 'None' instead of indexing an
    absent path. No HTML cleaning, brand, review, popularity or related-item data.
    """
    title = record.get('title')
    description = record.get('description')
    title = 'None' if title is None else title
    description = 'None' if description is None else description
    paths = record.get('categories')
    if paths is None or paths == []:
        category = 'None'
    elif (isinstance(paths, list) and isinstance(paths[0], list)
          and all(isinstance(x, str) for x in paths[0])):
        category = ', '.join(paths[0])
    else:
        raise ValueError('Expected categories as nested string lists')
    return f'Title: {title}. Categories: {category}. Description: {description}.'


def fixed_length_pool(hidden_states, attention_mask):
    """Zero padded vectors and divide by padded length, NOT valid-token count.

    Works with torch tensors and numpy arrays. Inference calls this in float32.
    """
    if (len(hidden_states.shape) != 3 or len(attention_mask.shape) != 2
            or hidden_states.shape[:2] != attention_mask.shape
            or attention_mask.shape[1] == 0):
        raise ValueError('Invalid hidden-state / attention-mask shapes')
    return (hidden_states * attention_mask[..., None]).sum(1) / attention_mask.shape[1]


def checked_inputs(args):
    """Verify protocol, raw reviews, previous content identity and stage row maps."""
    protocol_dir = Path(args.protocol_dir).expanduser().resolve()
    content_dir = Path(args.content_dir).expanduser().resolve()
    source = _json(protocol_dir / 'summary.json')
    content = _json(content_dir / 'summary.json')
    if (source.get('stage') != 'E0_data_preparation'
            or source.get('execution_status') != 'completed'):
        raise ValueError('Expected completed temporal data preparation')
    if (content.get('stage') != 'E0_content_metadata'
            or content.get('execution_status') != 'completed'):
        raise ValueError('Expected completed content metadata preparation')
    _verify_hashes(protocol_dir, source['artifact_files_sha256'],
                   {'registries.json', 'protocol_manifest.json'} |
                   {s + '.jsonl' for s in SPLITS})
    protocol_sha = source['protocol_sha256']
    if (_json(protocol_dir / 'protocol_manifest.json')['protocol_sha256'] != protocol_sha
            or content['protocol_sha256'] != protocol_sha
            or content['reviews_file_sha256'] != source['reviews_file_sha256']):
        raise ValueError('Protocol/content identity mismatch')
    if file_sha256(Path(args.reviews).expanduser()) != source['reviews_file_sha256']:
        raise ValueError('Raw reviews checksum differs from temporal preparation')
    _verify_hashes(content_dir, content['artifact_files_sha256'],
                   {f'{s}/{n}' for s in STAGES for n in ('registry.json', 'content.jsonl')})
    registries = _json(protocol_dir / 'registries.json')
    rows = {}
    for stage in STAGES:
        registry = registries[stage]
        _validate_registry(registry)
        if registry != _json(content_dir / stage / 'registry.json'):
            raise ValueError('Content registry differs from protocol: ' + stage)
        if registries['update']['id2item'][:len(registry['id2item'])] != registry['id2item']:
            raise ValueError('Temporal catalogs do not preserve item-ID prefixes')
        old_rows = [json.loads(line) for line in
                    (content_dir / stage / 'content.jsonl').read_text().splitlines()]
        expected = [dict(raw_item_id=raw, gr_item_id=i+1,
                         gensid_global_item_id=i, feature_row=i)
                    for i, raw in enumerate(registry['id2item'][1:])]
        if ([{k: row[k] for k in ROW_KEYS} for row in old_rows] != expected
                or content['stages'][stage]['items'] != len(expected)):
            raise ValueError('Content item-row mapping differs from protocol: ' + stage)
        rows[stage] = expected
    metadata_path = Path(args.metadata or content['metadata_path']).expanduser().resolve()
    if file_sha256(metadata_path) != content['metadata_file_sha256']:
        raise ValueError('Metadata checksum differs from existing content preparation')
    wanted = set(registries['update']['id2item'][1:])
    texts = {}
    duplicates = 0
    opener = gzip.open if metadata_path.suffix == '.gz' else open
    with opener(metadata_path, 'rt', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = parse_record(line)
                raw = record.get('asin')
                if raw not in wanted:
                    continue
                text = mitigating_text(record)
                if raw in texts:
                    duplicates += 1
                    if texts[raw] != text:
                        raise ValueError('Conflicting metadata for ' + raw)
                texts[raw] = text
            except (ValueError, TypeError, IndexError) as exc:
                raise ValueError(f'Invalid metadata at {metadata_path}:{line_number}') from exc
    missing = wanted - texts.keys()
    if missing:
        raise ValueError(f'Metadata missing for {len(missing)} eligible items; no features written')
    binding = dict(protocol_dir=str(protocol_dir), content_dir=str(content_dir),
        protocol_sha256=protocol_sha, reviews_file_sha256=source['reviews_file_sha256'],
        protocol_summary_sha256=file_sha256(protocol_dir / 'summary.json'),
        content_summary_sha256=file_sha256(content_dir / 'summary.json'),
        metadata_path=str(metadata_path), metadata_file_sha256=content['metadata_file_sha256'],
        duplicate_metadata_records=duplicates)
    return binding, registries, rows, texts


def model_identity(directory, expected_dim):
    """Hash the local safetensors and tokenizer actually selected for loading."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('Local model directory not found: ' + str(directory))
    config = _json(directory / 'config.json')
    if config.get('model_type') != 'llama' or config.get('hidden_size') != expected_dim:
        raise ValueError('Expected a Llama config with the declared embedding dimension')
    if not (directory / 'tokenizer.model').is_file():
        raise ValueError('Slow LlamaTokenizer requires local tokenizer.model')
    index = directory / 'model.safetensors.index.json'
    if index.is_file():
        shards = set(_json(index)['weight_map'].values())
        if not shards or any(Path(n).name != n for n in shards):
            raise ValueError('Invalid local safetensors index')
    else:
        shards = {'model.safetensors'}
    names = set(shards) | {'config.json', 'tokenizer.model'}
    for name in ('model.safetensors.index.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'tokenizer.json', 'generation_config.json'):
        if (directory / name).is_file():
            names.add(name)
    hashes = {}
    for name in sorted(names):
        print('Hash local model file: ' + name, flush=True)
        hashes[name] = file_sha256(directory / name)
    return dict(kind='local_llama_safetensors', path=str(directory),
                files_sha256=hashes, model_type=config['model_type'],
                hidden_size=config['hidden_size'],
                num_hidden_layers=config.get('num_hidden_layers'),
                published_checkpoint_identity_verified=False)


class LocalLlamaEncoder:
    def __init__(self, args):
        # No SentenceTransformer/TensorFlow imports, Hub requests or remote code.
        os.environ.setdefault('USE_TF', '0')
        os.environ.setdefault('USE_FLAX', '0')
        import torch
        from transformers import LlamaModel, LlamaTokenizer
        self.torch = torch
        self.device = torch.device(args.device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU and CUDA devices are supported')
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA requested but unavailable')
            torch.cuda.set_device(self.device)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(self.device)
        dtype = {'float16': torch.float16, 'float32': torch.float32}[args.dtype]
        if self.device.type == 'cpu' and args.dtype != 'float32':
            raise ValueError('Use --dtype float32 for CPU inference')
        self.tokenizer = LlamaTokenizer.from_pretrained(
            args.model_dir, local_files_only=True, legacy=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer has neither a padding token nor an EOS token')
        self.tokenizer.padding_side = 'right'
        self.model, loading = LlamaModel.from_pretrained(
            args.model_dir, local_files_only=True, use_safetensors=True,
            torch_dtype=dtype, attn_implementation=args.attention,
            output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('Local weights do not fully initialize LlamaModel: ' + str(loading))
        self.model.config.use_cache = False
        if args.max_length > self.model.config.max_position_embeddings:
            raise ValueError('Requested length exceeds the model context window')
        self.model.requires_grad_(False).eval().to(self.device)
        self.length = args.max_length
        self.info = dict(device=str(self.device), dtype=args.dtype,
            attention=args.attention, tokenizer_class=type(self.tokenizer).__name__,
            tokenizer_legacy=True, padding_side='right',
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id, eos_token_id=self.tokenizer.eos_token_id,
            add_bos_token=self.tokenizer.add_bos_token,
            add_eos_token=self.tokenizer.add_eos_token, use_cache=False)

    def encode(self, texts):
        torch = self.torch
        inputs = self.tokenizer(texts, max_length=self.length, truncation=True,
                                padding='max_length', return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if k in ('input_ids', 'attention_mask')}
        with torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            values = fixed_length_pool(hidden.float(), inputs['attention_mask'])
        return values.cpu().numpy().astype(np.float32, copy=False)

    def memory(self):
        if self.device.type != 'cuda':
            return dict(peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None)
        return dict(peak_cuda_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device),
                    peak_cuda_reserved_bytes=self.torch.cuda.max_memory_reserved(self.device))


def run(args, *, encoder_factory=LocalLlamaEncoder):
    started = time.monotonic()
    if min(args.batch_size, args.max_length, args.expected_dim, args.log_every) < 1:
        raise ValueError('Batch size, sequence length, dimension and log interval must be positive')
    output = Path(args.output).expanduser().resolve()
    pending = output.with_name(output.name + '.incomplete')
    if output.exists():
        raise FileExistsError('Completed output already exists; use a new run directory')
    if pending.exists() != args.resume:
        raise FileExistsError('Use --resume only for an existing .incomplete output; otherwise choose a new output')
    print('Verify temporal protocol, reviews, metadata and item mappings...', flush=True)
    binding, registries, rows, texts = checked_inputs(args)
    identity = model_identity(args.model_dir, args.expected_dim)
    root = Path(__file__).resolve().parent
    sources = {name: file_sha256(root / name) for name in (
        'prepare_incremental_llama.py', 'genrec/temporal.py', 'genrec/temporal_content.py')}
    versions = {}
    for name in ('numpy', 'torch', 'transformers', 'sentencepiece', 'safetensors'):
        try:
            versions[name] = packages.version(name)
        except packages.PackageNotFoundError:
            versions[name] = 'not_installed'
    config = dict(backend='local_llama', metadata='mitigating_title_categories_description',
        model=str(Path(args.model_dir).expanduser().resolve()), expected_dim=args.expected_dim,
        max_length=args.max_length, batch_size=args.batch_size, device=args.device,
        dtype=args.dtype, attention=args.attention,
        pooling='zero_padding_then_mean_over_fixed_padded_length', pooling_accumulation='float32',
        normalization='none', projection='none', local_files_only=True)
    plan = dict(binding=binding, encoder=identity, encoder_config=config,
                source_files_sha256=sources, package_versions=versions,
                python=platform.python_version())
    if args.resume:
        if _json(pending / 'encoding_plan.json') != plan:
            raise ValueError('Resume inputs/model/code/environment/config differ; use the original configuration')
    else:
        pending.mkdir(parents=True, exist_ok=False)
        write_json(pending / 'encoding_plan.json', plan)
        write_json(pending / 'progress.json', dict(completed_rows=0))
    all_rows = rows['update']
    count = len(all_rows)
    if count == 0:
        raise ValueError('Empty eligible item catalog')
    cursor = _json(pending / 'progress.json')['completed_rows']
    if type(cursor) is not int or not 0 <= cursor <= count:
        raise ValueError('Invalid resume cursor')
    scratch = pending / 'shared_embeddings.npy'
    if args.resume and scratch.exists():
        values = np.lib.format.open_memmap(scratch, mode='r+')
        if values.shape != (count, args.expected_dim) or values.dtype != np.float32:
            raise ValueError('Partial embedding file shape/dtype differs')
        if not np.isfinite(values[:cursor]).all() or (np.linalg.norm(values[:cursor], axis=1) == 0).any():
            raise ValueError('Partial embedding rows are nonfinite or zero')
    elif cursor == 0:
        values = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.float32,
                                          shape=(count, args.expected_dim))
    else:
        raise ValueError('Partial embedding file missing')
    # Keep the shared file in completed outputs as the resumable source for views.
    text_path = pending / 'item_texts.jsonl'
    with text_path.open('w', encoding='utf-8') as handle:
        for row in all_rows:
            handle.write(json.dumps(dict(**row, text=texts[row['raw_item_id']]),
                                    ensure_ascii=False) + '\n')
    print(f'Load local Llama | items={count} | resume_row={cursor} | '
          f'length={args.max_length} | batch={args.batch_size} | {args.dtype}', flush=True)
    encoder = encoder_factory(args)
    inference_started, initial_cursor = time.monotonic(), cursor
    for start in range(cursor, count, args.batch_size):
        end = min(start + args.batch_size, count)
        batch = encoder.encode([texts[row['raw_item_id']] for row in all_rows[start:end]])
        if (batch.shape != (end-start, args.expected_dim) or batch.dtype != np.float32
                or not np.isfinite(batch).all() or (np.linalg.norm(batch, axis=1) == 0).any()):
            raise ValueError('Encoder returned invalid shape, dtype, nonfinite or zero vectors')
        values[start:end] = batch
        values.flush()
        write_json(pending / 'progress.json', dict(completed_rows=end))
        step = (start-initial_cursor) // args.batch_size + 1
        if step == 1 or step % args.log_every == 0 or end == count:
            elapsed = time.monotonic() - inference_started
            speed = (end-initial_cursor) / max(elapsed, 1e-9)
            print(f'Llama items {end}/{count} | {speed:.2f} items/s | '
                  f'elapsed={elapsed:.1f}s | eta={(count-end)/speed:.1f}s', flush=True)
    stats = {}
    shared_equal = True
    for stage in STAGES:
        directory = pending / stage
        directory.mkdir(exist_ok=True)
        subset = values[:len(rows[stage])]
        np.save(directory / 'embeddings.npy', subset, allow_pickle=False)
        write_json(directory / 'feature_rows.json', rows[stage])
        write_json(directory / 'registry.json', registries[stage])
        reloaded = np.load(directory / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        if not np.array_equal(subset, reloaded):
            raise ValueError('Saved features differ from shared rows: ' + stage)
        norms = np.linalg.norm(subset, axis=1)
        stats[stage] = dict(shape=list(subset.shape), dtype=str(subset.dtype),
                           norm_min=float(norms.min()), norm_max=float(norms.max()),
                           norm_mean=float(norms.mean()))
    result = dict(stage='E0_fixed_content_embeddings', execution_status='completed',
        created_utc=datetime.now(timezone.utc).isoformat(), **binding,
        encoder=identity, encoder_config=config, actual_encoder_runtime=encoder.info,
        stages=stats, reference_notebook=REFERENCE, reference_notebook_blob_sha=REFERENCE_BLOB_SHA,
        text_policy='Title: {title}. Categories: {first category path}. Description: {description}.',
        reference_differences=['existing temporal catalogs, not author ten-block split',
            'local weights identified by checksums; published checkpoint identity unverified',
            'explicit inference precision/attention; float32 pooling; not a bitwise reproduction',
            'absent/empty category path explicitly becomes None'],
        normalization='none', projection='none', graph_used=False,
        static_metadata_approximation=True, future_only_items_encoded=False,
        encoder_finetuning=False, shared_features_equal=shared_equal,
        training_catalogs=['base_train', 'update_train'],
        update_view='deployment catalog including update-validation-only items; not a training view',
        split_catalog_reference=dict(base_train='base_train', base_val='base_train',
            update_train='update_train', update_val='update_train', future_test='update'),
        shared_features='encode observed Update catalog once; identical prefix rows copied to stage views',
        training_status='not_run', E0_full_status='not_run', E1_status='not_run',
        python=platform.python_version(), package_versions=versions,
        source_files_sha256=sources, **encoder.memory(),
        memory_scope='this process, model construction through encoding; excludes other processes',
        resumed=args.resume, encoded_items_this_invocation=count-initial_cursor,
        runtime_seconds_this_invocation=time.monotonic()-started)
    compact = ['execution_status = completed',
        f'items_encoded_this_invocation = {count-initial_cursor} | resumed = {args.resume}',
        f'model = {identity["path"]} | dtype = {args.dtype}',
        f'pooling = padding-zeroed sum / {args.max_length} | normalization = none',
        'stage rows dimension dtype']
    compact += [f'{s} {stats[s]["shape"][0]} {stats[s]["shape"][1]} {stats[s]["dtype"]}' for s in STAGES]
    compact += ['shared_features_equal = True | future_only_items_encoded = False',
                f'peak_cuda_allocated_bytes = {result["peak_cuda_allocated_bytes"]}',
                f'runtime_seconds_this_invocation = {result["runtime_seconds_this_invocation"]:.2f}',
                'training_status = not_run', f'output = {output}']
    (pending / 'compact.txt').write_text('\n'.join(compact) + '\n', encoding='utf-8')
    result['artifact_files_sha256'] = {name: digest for name, digest in
        _payload_hashes(pending).items() if name != 'summary.json' and not name.endswith('.tmp')}
    write_json(pending / 'summary.json', result)
    if output.exists():
        raise FileExistsError('Output created concurrently')
    pending.rename(output)
    print('\n'.join(compact), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-dir', required=True)
    p.add_argument('--content-dir', required=True, help='Existing prepared metadata directory')
    p.add_argument('--reviews', required=True, help='Original reviews; checksum verification only')
    p.add_argument('--metadata', help='Override metadata path; original checksum must match')
    p.add_argument('--model-dir', required=True, help='Local HF Llama safetensors directory')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    p.add_argument('--attention', choices=('sdpa', 'eager'), default='sdpa')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--expected-dim', type=int, default=4096)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--resume', action='store_true', help='Continue matching output.incomplete')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
