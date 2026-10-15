"""Build stage-specific graphs through the repository InteractionGraphBuilder."""
import json
import platform
import time
from pathlib import Path

from genrec.temporal_graph_inputs import file_sha256


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_verified_graph(directory):
    """Restore a complete graph only after payload checksum validation."""
    from gensid.graph_aware_sid.graph_utils import InteractionGraphBuilder
    directory = Path(directory)
    manifest = _json(directory / 'graph_manifest.json')
    required = {'meta.json', 'graph_df.parquet', 'node_map.parquet', 'pyg_data.pt'}
    if set(manifest['files_sha256']) != required:
        raise ValueError('Incomplete graph manifest')
    for name, expected in manifest['files_sha256'].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f'Graph checksum mismatch: {name}')
    builder = InteractionGraphBuilder.from_dump(str(directory))
    if builder.node_map['item_id'].to_list() != manifest['global_item_ids']:
        raise ValueError('Graph global item order differs from manifest')
    if builder.node_map['node_id'].to_list() != list(range(len(manifest['global_item_ids']))):
        raise ValueError('Graph node IDs must be contiguous local IDs')
    if builder.pyg_data.num_nodes != len(manifest['global_item_ids']):
        raise ValueError('Graph node count mismatch')
    return builder


def prepare_graphs(graph_inputs_dir, output):
    # Lazy imports keep the CLI/help separate from graph kernels.
    import polars as pl
    import torch
    from gensid.graph_aware_sid.graph_utils import InteractionGraphBuilder
    started = time.perf_counter()
    graph_inputs_dir, output = Path(graph_inputs_dir).resolve(), Path(output).resolve()
    pending = output.with_name(output.name + '.incomplete')
    if output.exists() or pending.exists():
        raise FileExistsError('Use a new output directory')
    source = _json(graph_inputs_dir / 'summary.json')
    if source.get('execution_status') != 'completed' or source.get('stage') != 'E0_graph_input_preparation':
        raise ValueError('Expected completed Round 4 graph inputs')
    required = {f'{stage}/{name}' for stage in ('base', 'update')
                for name in ('observed_events.jsonl', 'user_sequences.jsonl', 'item_nodes.json', 'registry.json')}
    if not required.issubset(source['artifact_files_sha256']):
        raise ValueError('Incomplete Round 4 manifest')
    for name in sorted(required):
        if file_sha256(graph_inputs_dir / name) != source['artifact_files_sha256'][name]:
            raise ValueError(f'Input checksum mismatch: {name}')
    stats = {}
    builders = {}
    for stage in ('base', 'update'):
        nodes = _json(graph_inputs_dir / stage / 'item_nodes.json')
        ids = [n['gensid_global_item_id'] for n in nodes]
        if ids != sorted(set(ids)) or any(i < 0 for i in ids):
            raise ValueError('Invalid catalog global IDs')
        if [n['graph_local_item_id'] for n in nodes] != list(range(len(ids))):
            raise ValueError('Catalog local node order mismatch')
        rows = [json.loads(line) for line in
                (graph_inputs_dir / stage / 'observed_events.jsonl').read_text().splitlines()]
        expected = source['stages'][stage]
        if len(rows) != expected['event_count'] or len(ids) != expected['catalog_size']:
            raise ValueError('Stage input count mismatch')
        for r in rows:
            local = r['graph_local_item_id']
            if local < 0 or local >= len(nodes) or nodes[local]['gensid_global_item_id'] != r['gensid_global_item_id']:
                raise ValueError('Event global/local map mismatch')
            if nodes[local]['raw_item_id'] != r['item']:
                raise ValueError('Raw item round trip mismatch')
            lo, hi = expected['lower_timestamp_inclusive'], expected['upper_timestamp_exclusive']
            if (lo is not None and r['timestamp'] < lo) or r['timestamp'] >= hi:
                raise ValueError('Event outside training window')
        df = pl.DataFrame({'user_id': [r['gr_user_id'] for r in rows],
                           'item_id': [r['gensid_global_item_id'] for r in rows],
                           'ts': [r['timestamp'] for r in rows]})
        builder = InteractionGraphBuilder(interactions_df=df, skip_n_last=0,
            graph_type='adjacent_cooc', graph_building_kwargs={'strict_timestamp_groups': True},
            make_undirected=True, add_identity_features=False, all_item_ids=ids)
        if builder.node_map['item_id'].to_list() != ids:
            raise ValueError('Builder changed catalog order')
        degree = torch.zeros(len(ids), dtype=torch.float)
        ei, ew = builder.pyg_data.edge_index, builder.pyg_data.edge_weight
        if ew.numel():
            degree.scatter_add_(0, ei[0], ew)
        unique_user_times = df.select(['user_id', 'ts']).unique().height
        stats[stage] = dict(num_nodes=len(ids),
            directed_edges_before_symmetrization=builder.graph_df.height,
            directed_edges_after_symmetrization=int(ei.shape[1]),
            isolated_nodes=int((degree == 0).sum()),
            nodes_without_window_interactions=expected['nodes_without_window_interactions'],
            interactions=len(rows), duplicate_user_timestamp_events=len(rows)-unique_user_times,
            total_directed_edge_weight=float(ew.sum()))
        builders[stage] = builder
    pending.mkdir(parents=True, exist_ok=False)
    def write(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    for stage, builder in builders.items():
        directory = pending / stage
        builder.dump(str(directory))
        graph_manifest = dict(stage=stage, protocol_sha256=source['protocol_sha256'],
            input_events_sha256=source['artifact_files_sha256'][f'{stage}/observed_events.jsonl'],
            catalog_sha256=source['artifact_files_sha256'][f'{stage}/item_nodes.json'],
            global_item_ids=builder.node_map['item_id'].to_list(),
            graph_policy='all event pairs between consecutive distinct per-user timestamp groups',
            same_timestamp_edges=False, repeat_event_multiplicity=True,
            directed_counts_then_undirected_sum=True, skip_n_last=0,
            files_sha256={p.name: file_sha256(p) for p in sorted(directory.iterdir())})
        write(directory / 'graph_manifest.json', graph_manifest)
        restored = load_verified_graph(directory)
        for attr in ('edge_index', 'edge_weight', 'x'):
            if not torch.equal(getattr(builder.pyg_data, attr), getattr(restored.pyg_data, attr)):
                raise ValueError('Graph dump/reload tensor mismatch')
    result = dict(stage='E0_graph_construction', execution_status='completed', stages=stats,
        protocol_sha256=source['protocol_sha256'], graph_inputs_dir=str(graph_inputs_dir),
        graph_type='adjacent_cooc', strict_timestamp_groups=True, make_undirected=True,
        self_pairs_between_distinct_timestamp_groups='preserved',
        graph_dump_reload_verified=True, graph_features='degree placeholder; not content embeddings',
        appnp_status='not_run', feature_extraction_status='not_run', training_status='not_run',
        E0_full_status='not_run', E1_status='not_run', python=platform.python_version(),
        torch_version=torch.__version__, polars_version=pl.__version__)
    root = Path(__file__).resolve().parents[1]
    result['source_files_sha256'] = {name: file_sha256(root / name) for name in (
        'gensid/graph_aware_sid/graph_utils.py', 'genrec/temporal_graphs.py', 'prepare_incremental_graphs.py')}
    result['artifact_files_sha256'] = {str(p.relative_to(pending)): file_sha256(p)
        for p in sorted(pending.rglob('*')) if p.is_file()}
    result['runtime_seconds'] = time.perf_counter() - started
    write(pending / 'summary.json', result)
    if output.exists():
        raise FileExistsError('Output created concurrently')
    pending.rename(output)
    return result
