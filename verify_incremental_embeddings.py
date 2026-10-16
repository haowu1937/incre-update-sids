"""Read-only coverage check against the temporal protocol; numpy only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(features_dir, protocol_dir, dimension=4096):
    features_dir, protocol_dir = Path(features_dir), Path(protocol_dir)
    summary, protocol = read(features_dir/'summary.json'), read(protocol_dir/'summary.json')
    require(summary.get('execution_status') == 'completed', 'Feature run is not completed')
    require(protocol.get('execution_status') == 'completed', 'Protocol is not completed')
    require(summary['protocol_sha256'] == protocol['protocol_sha256'], 'Protocol identity mismatch')
    require(checksum(protocol_dir/'registries.json') ==
            protocol['artifact_files_sha256']['registries.json'], 'Protocol registry checksum mismatch')
    registries = read(protocol_dir/'registries.json')
    catalogs, report = {}, {}
    for stage in ('base_train', 'update_train', 'update'):
        for name in ('embeddings.npy', 'feature_rows.json', 'registry.json'):
            relative = f'{stage}/{name}'
            require(checksum(features_dir/relative) == summary['artifact_files_sha256'][relative],
                    'Artifact checksum mismatch: ' + relative)
        registry = registries[stage]
        ids = registry['id2item']
        require(ids[0] == '[PAD]' and len(set(ids)) == len(ids)
                and registry['item2id'] == {raw: i for i, raw in enumerate(ids)},
                'Invalid protocol item mapping: ' + stage)
        require(read(features_dir/stage/'registry.json') == registry, 'Registry mismatch: ' + stage)
        expected = [dict(raw_item_id=raw, gr_item_id=i+1, gensid_global_item_id=i, feature_row=i)
                    for i, raw in enumerate(ids[1:])]
        require(read(features_dir/stage/'feature_rows.json') == expected,
                'Missing, duplicate, reordered or mismatched item rows: ' + stage)
        values = np.load(features_dir/stage/'embeddings.npy', mmap_mode='r', allow_pickle=False)
        require(values.shape == (len(expected), dimension) and values.dtype == np.float32,
                'Embedding shape or dtype mismatch: ' + stage)
        require(list(values.shape) == summary['stages'][stage]['shape'],
                'Summary shape differs from actual array: ' + stage)
        nonfinite, zero = 0, 0
        for start in range(0, len(values), 512):
            block = values[start:start+512]
            nonfinite += int((~np.isfinite(block).all(axis=1)).sum())
            zero += int((~np.any(block != 0, axis=1)).sum())
        require(nonfinite == 0 and zero == 0, f'{stage}: nonfinite={nonfinite}, zero={zero}')
        catalogs[stage] = values
        report[stage] = dict(protocol_items=len(expected), embedding_rows=len(values),
            dimension=dimension, missing_embeddings=0, nonfinite_vectors=nonfinite, zero_vectors=zero)
    for stage in ('base_train', 'update_train'):
        require(registries['update']['id2item'][:len(registries[stage]['id2item'])] ==
                registries[stage]['id2item'], 'Catalogs do not preserve shared IDs')
        require(np.array_equal(catalogs[stage], catalogs['update'][:len(catalogs[stage])]),
                'Shared item embeddings differ: ' + stage)
    return dict(execution_status='passed', scope='all eligible items in each temporal catalog',
                stages=report, shared_item_embeddings_equal=True,
                future_only_items_required=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--protocol-dir', required=True)
    parser.add_argument('--expected-dim', type=int, default=4096)
    args = parser.parse_args()
    try:
        result = verify(args.features_dir, args.protocol_dir, args.expected_dim)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps(dict(execution_status='failed', error=str(exc)), ensure_ascii=False, indent=2))
        sys.exit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
