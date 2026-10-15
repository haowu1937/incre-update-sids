"""Graph integration checks; synthetic data only, no encoder download."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import polars as pl
import torch
from gensid.graph_aware_sid.graph_utils import InteractionGraphBuilder, load_train_item_mask
from genrec.temporal_graph_inputs import file_sha256
from genrec.temporal_graphs import load_verified_graph, prepare_graphs


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inter = self.root / 'inter.json'
        self.inter.write_text(json.dumps({'u': [2, 7, 11]}))

    def build(self, pairs=None):
        pairs = pairs or [(1, 2, 10), (1, 7, 10), (1, 11, 20)]
        df = pl.DataFrame(pairs, schema=['user_id', 'item_id', 'ts'], orient='row')
        return InteractionGraphBuilder(interactions_df=df, skip_n_last=0,
            graph_building_kwargs={'strict_timestamp_groups': True},
            all_item_ids=[2, 7, 11])

    def fixture(self):
        source = self.root / 'inputs'
        source.mkdir()
        hashes, stats = {}, {}
        for stage in ('base', 'update'):
            directory = source / stage
            directory.mkdir()
            nodes = [dict(raw_item_id=raw, gensid_global_item_id=g,
                          graph_local_item_id=i, gr_item_id=g+1)
                     for i, (raw, g) in enumerate([('a', 2), ('b', 7), ('c', 11)])]
            rows = [dict(item=raw, timestamp=t, gr_user_id=1,
                         gensid_global_item_id=g, graph_local_item_id=i)
                    for i, (raw, g, t) in enumerate([('a', 2, 10), ('b', 7, 20)])]
            (directory / 'observed_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            for name, obj in (('item_nodes.json', nodes), ('registry.json', {})):
                (directory / name).write_text(json.dumps(obj))
            (directory / 'user_sequences.jsonl').write_text('{}\n')
            for path in directory.iterdir():
                hashes[str(path.relative_to(source))] = file_sha256(path)
            stats[stage] = dict(event_count=2, catalog_size=3,
                nodes_without_window_interactions=1,
                lower_timestamp_inclusive=None, upper_timestamp_exclusive=30)
        (source / 'summary.json').write_text(json.dumps(dict(
            execution_status='completed', stage='E0_graph_input_preparation',
            protocol_sha256='synthetic', artifact_files_sha256=hashes, stages=stats)))
        return source

    def test_zero_none_and_positive_truncation(self):
        for skip in (0, None):
            self.assertEqual(load_train_item_mask(self.inter, 12, skip).tolist(), [2, 7, 11])
        self.assertEqual(load_train_item_mask(self.inter, 12, 1).tolist(), [2, 7])
        with self.assertRaises(ValueError):
            load_train_item_mask(self.inter, 12, -1)

    def test_legacy_builder_zero_and_none_preserve_all_events(self):
        for skip in (0, None):
            builder = InteractionGraphBuilder(interactions_path=str(self.inter),
                skip_n_last=skip, all_item_ids=[2, 7, 11])
            self.assertEqual(builder.graph_df.height, 2)
        builder = InteractionGraphBuilder(interactions_path=str(self.inter),
            skip_n_last=1, all_item_ids=[2, 7, 11])
        self.assertEqual(builder.graph_df.height, 1)

    def test_simultaneous_group_has_no_internal_edges(self):
        builder = self.build()
        edges = {(r['src'], r['dst']): r['weight'] for r in builder.graph_df.to_dicts()}
        self.assertEqual(edges, {(2, 11): 1.0, (7, 11): 1.0})

    def test_repeat_multiplicity_preserved(self):
        builder = self.build([(1, 2, 10), (1, 2, 10), (1, 7, 20)])
        self.assertEqual(builder.graph_df.to_dicts(), [{'src': 2, 'dst': 7, 'weight': 2.0}])

    def test_noncontiguous_mapping_and_single_timestamp_isolation(self):
        builder = self.build([(1, 2, 10), (1, 7, 10), (2, 11, 20)])
        self.assertEqual(builder.node_map['item_id'].to_list(), [2, 7, 11])
        self.assertEqual(builder.node_map['node_id'].to_list(), [0, 1, 2])
        self.assertEqual(builder.pyg_data.num_nodes, 3)
        self.assertEqual(builder.pyg_data.edge_index.shape, (2, 0))

    def test_pre_filtered_dataframe_rejects_extra_truncation(self):
        df = pl.DataFrame({'user_id': [1], 'item_id': [2], 'ts': [10]})
        with self.assertRaises(ValueError):
            InteractionGraphBuilder(interactions_df=df, skip_n_last=2, all_item_ids=[2])

    def test_export_reload_checksums_and_no_overwrite(self):
        source = self.fixture()
        output = self.root / 'out'
        result = prepare_graphs(source, output)
        self.assertEqual(result['stages']['update']['isolated_nodes'], 1)
        self.assertTrue(result['graph_dump_reload_verified'])
        restored = load_verified_graph(output / 'base')
        self.assertTrue(torch.equal(restored.pyg_data.edge_index,
                                     torch.tensor([[0, 1], [1, 0]])))
        self.assertEqual(restored.all_item_ids, [2, 7, 11])
        self.assertEqual(result['training_status'], 'not_run')
        with self.assertRaises(FileExistsError):
            prepare_graphs(source, output)
        with (output / 'base/pyg_data.pt').open('ab') as handle:
            handle.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_verified_graph(output / 'base')

    def test_modified_stage_input_rejected_before_output(self):
        source = self.fixture()
        with (source / 'update/observed_events.jsonl').open('a') as handle:
            handle.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            prepare_graphs(source, self.root / 'out')
        self.assertFalse((self.root / 'out').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (output / 'checks.json').write_text(json.dumps(dict(round='round5', inputs='synthetic_only',
        tests_run=result.testsRun, execution_status='passed' if result.wasSuccessful() else 'failed',
        E0_full_status='not_run', E1_status='not_run'), indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
