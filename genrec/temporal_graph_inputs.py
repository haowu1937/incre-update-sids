"""Export observed training-window events for the existing gensid graph path.

No graph construction, propagation, content encoding, or model training here.
Use raw retained events: causal prediction rows omit first interactions and
repeat history, so they must not be used to reconstruct an interaction graph.
"""
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

from genrec.temporal import iter_gzip_records, build_temporal_protocol


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stage_view(events, registry, lower, upper):
    """Explicit raw/GR/gensid/local identities; retain all eligible nodes."""
    items = registry['item2id']
    users = registry['user2id']
    real = sorted((gr_id, raw) for raw, gr_id in items.items() if raw != '[PAD]')
    if items.get('[PAD]') != 0 or any(gr_id <= 0 for gr_id, _ in real):
        raise ValueError('Item registry must reserve ID 0 for padding')
    if len({gr_id for gr_id, _ in real}) != len(real):
        raise ValueError('Duplicate GR item IDs')
    selected = [event for event in events
                if (lower is None or event['timestamp'] >= lower)
                and event['timestamp'] < upper]
    if not selected:
        raise ValueError('Empty training window')
    local_by_global = {gr_id - 1: local for local, (gr_id, _) in enumerate(real)}
    rows, grouped = [], defaultdict(list)
    observed = set()
    for event in selected:
        raw, user = event['item'], event['user']
        if raw not in items or user not in users:
            raise ValueError('Training event absent from the stage registry')
        global_id = items[raw] - 1
        row = dict(event, gr_item_id=items[raw], gensid_global_item_id=global_id,
                   graph_local_item_id=local_by_global[global_id],
                   gr_user_id=users[user])
        rows.append(row)
        grouped[user].append(row)
        observed.add(global_id)
    nodes = [dict(raw_item_id=raw, gr_item_id=gr_id,
                  gensid_global_item_id=gr_id - 1, graph_local_item_id=local,
                  has_training_window_interaction=(gr_id - 1 in observed))
             for local, (gr_id, raw) in enumerate(real)]
    sequences = []
    for user in sorted(grouped, key=users.__getitem__):
        user_rows = grouped[user]
        sequences.append(dict(raw_user_id=user, gr_user_id=users[user],
                              raw_item_ids=[r['item'] for r in user_rows],
                              gensid_global_item_ids=[r['gensid_global_item_id'] for r in user_rows],
                              graph_local_item_ids=[r['graph_local_item_id'] for r in user_rows],
                              timestamps=[r['timestamp'] for r in user_rows]))
    return rows, nodes, sequences


def prepare_graph_inputs(reviews_path, protocol_dir, output):
    started = time.perf_counter()
    reviews_path, protocol_dir, output = map(lambda p: Path(p).resolve(),
                                            (reviews_path, protocol_dir, output))
    pending = output.with_name(output.name + '.incomplete')
    if output.exists() or pending.exists():
        raise FileExistsError('Use a new output directory without an incomplete sibling')
    summary = json.loads((protocol_dir / 'summary.json').read_text())
    if summary.get('execution_status') != 'completed':
        raise ValueError('Temporal preparation is not complete')
    if file_sha256(reviews_path) != summary['reviews_file_sha256']:
        raise ValueError('Reviews checksum differs from Round 3')
    required = {'protocol_manifest.json', 'registries.json', 'base_train.jsonl',
                'base_val.jsonl', 'update_train.jsonl', 'update_val.jsonl', 'future_test.jsonl'}
    hashes = summary['artifact_files_sha256']
    if not required.issubset(hashes):
        raise ValueError('Incomplete Round 3 artifact manifest')
    for name in sorted(required):
        if file_sha256(protocol_dir / name) != hashes[name]:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    reviews = []
    for record in iter_gzip_records(reviews_path):
        reviews.append((record['reviewerID'], record['asin'],
                        record['unixReviewTime'], record.get('overall')))
    # Reproduce the protocol from the exact raw input and configuration.
    from genrec.temporal import normalize_reviews
    rebuilt = build_temporal_protocol(reviews,
        max_history=summary['resolved_config']['max_history'],
        duplicate_policy=summary['resolved_config']['duplicate_policy'])
    manifest = json.loads((protocol_dir / 'protocol_manifest.json').read_text())
    registries = json.loads((protocol_dir / 'registries.json').read_text())
    if (rebuilt['manifest'] != manifest or rebuilt['registries'] != registries
            or manifest['protocol_sha256'] != summary['protocol_sha256']):
        raise ValueError('Round 3 protocol cannot be reproduced from these reviews')
    events, _ = normalize_reviews(reviews, summary['resolved_config']['duplicate_policy'])
    cut = manifest['cutoffs_exclusive']
    windows = {'base': (None, cut['base_train']),
               'update': (cut['base'], cut['update_train'])}
    views = {stage: stage_view(events, registries[stage + '_train'], lower, upper)
             for stage, (lower, upper) in windows.items()}
    for stage, (rows, _, _) in views.items():
        if len(rows) != manifest['windows'][stage + '_train']['event_count']:
            raise ValueError(f'Event count mismatch: {stage}')
    pending.mkdir(parents=True, exist_ok=False)
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    def write_rows(path, rows):
        with path.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    stats = {}
    for stage, (rows, nodes, sequences) in views.items():
        directory = pending / stage
        directory.mkdir()
        write_rows(directory / 'observed_events.jsonl', rows)
        write_rows(directory / 'user_sequences.jsonl', sequences)
        write_json(directory / 'item_nodes.json', nodes)
        write_json(directory / 'registry.json', registries[stage + '_train'])
        lower, upper = windows[stage]
        stats[stage] = dict(event_count=len(rows), user_count=len(sequences),
            catalog_size=len(nodes), nodes_without_window_interactions=sum(
                not n['has_training_window_interaction'] for n in nodes),
            lower_timestamp_inclusive=lower, upper_timestamp_exclusive=upper,
            first_timestamp=rows[0]['timestamp'], last_timestamp=rows[-1]['timestamp'])
    result = dict(stage='E0_graph_input_preparation', execution_status='completed',
        python=platform.python_version(), reviews_file_sha256=summary['reviews_file_sha256'],
        protocol_sha256=summary['protocol_sha256'], protocol_dir=str(protocol_dir),
        stages=stats, update_window_policy='update_train_only_not_cumulative',
        skip_n_last_for_downstream_builder=0,
        id_contract='GR item ID 0 is padding; gensid global ID = GR ID - 1; local ID is explicit',
        same_timestamp_policy='timestamps retained; equal-time ordering is not causal; edge policy pending',
        singleton_policy='all observed events retained, including first and singleton events',
        old_nodes_policy='retain eligible old nodes even without update-window interactions',
        graph_construction_status='not_run', appnp_status='not_run',
        feature_extraction_status='not_run', training_status='not_run',
        E0_full_status='not_run', E1_status='not_run')
    result['artifact_files_sha256'] = {str(p.relative_to(pending)): file_sha256(p)
        for p in sorted(pending.rglob('*')) if p.is_file()}
    root = Path(__file__).resolve().parents[1]
    result['source_files_sha256'] = {name: file_sha256(root / name)
        for name in ('genrec/temporal.py', 'genrec/temporal_graph_inputs.py',
                     'prepare_incremental_graph_inputs.py')}
    result['runtime_seconds'] = time.perf_counter() - started
    write_json(pending / 'summary.json', result)
    if output.exists():
        raise FileExistsError('Output created concurrently')
    pending.rename(output)
    return result
