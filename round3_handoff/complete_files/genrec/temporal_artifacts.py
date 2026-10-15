"""Persist real-review temporal protocols; reuse AbstractDataset's stage adapter."""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from genrec.dataset import AbstractDataset
from genrec.temporal import iter_gzip_records


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_reviews(path):
    rows = []
    for index, record in enumerate(iter_gzip_records(path), 1):
        try:
            rows.append((record['reviewerID'], record['asin'],
                         record['unixReviewTime'], record.get('overall')))
        except KeyError as exc:
            raise ValueError(f'Missing review field in record {index}: {exc}') from exc
    return rows


def prepare_artifacts(reviews_path, output, max_history=50,
                      release='2014_beauty_5core', source_url=None):
    """No downloads, model fitting, legacy cache reuse or silent overwrite."""
    started = time.perf_counter()
    reviews_path = Path(reviews_path).resolve()
    if not reviews_path.is_file():
        raise FileNotFoundError(f'Reviews file not found: {reviews_path}')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    pending = output.with_name(output.name + '.incomplete')
    if pending.exists():
        raise FileExistsError(f'Previous incomplete output exists: {pending}')
    # Read raw reviews, never the legacy all_seqs/id_mapping caches.
    reviews = load_reviews(reviews_path)
    dataset = AbstractDataset({'explicit_causal_examples': True,
                               'n_user_tokens': 0,
                               'max_item_seq_len': max_history})
    dataset.prepare_temporal_stage(reviews, stage='base')
    protocol = dataset.temporal_protocol
    users = {name: set() for name in ('base', 'update', 'future')}
    cutoffs = protocol['manifest']['cutoffs_exclusive']
    for user, _, timestamp, _ in reviews:
        stage = ('base' if timestamp < cutoffs['base'] else
                 'update' if timestamp < cutoffs['update'] else 'future')
        users[stage].add(user)
    cohorts = {
        'base_users': len(users['base']),
        'update_users': len(users['update']),
        'future_users': len(users['future']),
        'base_only_users': len(users['base'] - users['update'] - users['future']),
        'future_only_users': len(users['future'] - users['base'] - users['update']),
        'base_and_update_users': len(users['base'] & users['update']),
        'base_and_future_users': len(users['base'] & users['future']),
    }
    pending.mkdir(parents=True, exist_ok=False)
    def write_json(name, value):
        (pending / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                              allow_nan=False) + '\n', encoding='utf-8')
    write_json('protocol_manifest.json', protocol['manifest'])
    write_json('registries.json', protocol['registries'])
    for stage, rows in protocol['examples'].items():
        with (pending / f'{stage}.jsonl').open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, timeout=10)
        source_sha = commit.stdout.strip() if commit.returncode == 0 else 'not_available'
    except (OSError, subprocess.TimeoutExpired):
        source_sha = 'not_available'
    root = Path(__file__).resolve().parents[1]
    source_files = ('genrec/temporal.py', 'genrec/dataset.py',
                    'genrec/temporal_artifacts.py', 'prepare_incremental_data.py')
    summary = {
        'stage': 'E0_data_preparation', 'execution_status': 'completed',
        'dataset_release_declared': release,
        'dataset_identity_verified': False,
        'source_url_declared': source_url,
        'reviews_path': str(reviews_path),
        'reviews_file_sha256': file_sha256(reviews_path),
        'source_sha': source_sha,
        'source_files_sha256': {name: file_sha256(root / name) for name in source_files},
        'python': platform.python_version(),
        'command_argv': sys.argv,
        'resolved_config': {'max_history': max_history, 'duplicate_policy': 'drop_exact',
                            'explicit_causal_examples': True, 'n_user_tokens': 0},
        'protocol_sha256': protocol['manifest']['protocol_sha256'],
        'preprocessing': protocol['manifest']['preprocessing'],
        'windows': protocol['manifest']['windows'],
        'catalog_sizes': protocol['manifest']['catalog_sizes'],
        'user_cohorts': cohorts,
        'full_period_filtering_limitation': (
            'The declared 5-core release was filtered over the full period; '
            'temporal splitting does not undo that selection bias.'
            if release == '2014_beauty_5core' else 'synthetic_fixture'),
        'metadata_status': 'not_loaded', 'training_status': 'not_run',
        'E0_full_status': 'not_run', 'E1_status': 'not_run',
        'memory': 'not_measured',
    }
    summary['artifact_files_sha256'] = {
        file.name: file_sha256(file) for file in sorted(pending.iterdir()) if file.is_file()}
    summary['runtime_seconds'] = time.perf_counter() - started
    write_json('summary.json', summary)
    # Directory presence is the completion marker; a failed write stays .incomplete.
    if output.exists():
        raise FileExistsError(f'Output was created concurrently: {output}')
    pending.rename(output)
    return summary
