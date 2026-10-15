"""Synthetic E1 checks using the real repository RQ-VAE and TIGER/T5."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from genrec.historical_damage import (aggregate, historical_cohorts, load_bound_inputs,
    paired_losses, parser, reencode_history, run, versioned_suffixes)
from genrec.temporal_sids import allocate_suffixes, maps_from_ledger, read_json, sha256, write_json
from genrec.models.TIGER.model import TIGER
from genrec.temporal_tiger import parser as tiger_parser, run as train_tiger
from gensid.plain_incremental_content import parser as update_parser, run as train_update

torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('round9_fixture',
            ROOT/'tests/incremental/test_temporal_tiger_round9.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.previous = module.Checks
        cls.previous.setUpClass()
        cls.fixture = tempfile.TemporaryDirectory()
        root = Path(cls.fixture.name)
        for name in ('protocol', 'features', 'rqvae'):
            shutil.copytree(Path(cls.previous.previous.fixture.name)/name, root/name)
        shutil.copytree(cls.previous.sid_dir, root/'sids')
        write_json(root/'tiny.json', dict(num_layers=1, num_decoder_layers=1, d_model=16,
            d_ff=32, num_heads=2, d_kv=8, dropout_rate=0.2))
        args = tiger_parser().parse_args(['--sid-dir', str(root/'sids'), '--output', str(root/'tiger'),
            '--device', 'cpu', '--max-steps', '20', '--batch-size', '8', '--eval-batch-size', '3',
            '--warmup-steps', '2', '--probe-size', '10', '--eval-every', '100',
            '--num-beams', '4', '--topk', '1', '3', '--model-config', str(root/'tiny.json')])
        update = update_parser().parse_args(['--protocol-dir', str(root/'protocol'),
            '--features-dir', str(root/'features'), '--base-rqvae-dir', str(root/'rqvae'),
            '--output', str(root/'update'), '--device', 'cpu', '--max-steps', '12',
            '--batch-size', '5', '--eval-batch-size', '4', '--lr', '0.005'])
        with contextlib.redirect_stdout(io.StringIO()):
            train_tiger(args)
            train_update(update)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()
        cls.previous.tearDownClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, name='e1', ranking=False):
        root = Path(self.fixture.name)
        values = ['--protocol-dir', str(root/'protocol'), '--tiger-dir', str(root/'tiger'),
            '--update-rqvae-dir', str(root/'update'), '--sid-dir', str(root/'sids'),
            '--expected-tiger-sha256', sha256(root/'tiger/best.pth'),
            '--output', str(self.root/name), '--device', 'cpu', '--batch-size', '3',
            '--anchor-size', '12', '--bootstrap-reps', '40']
        if ranking:
            values += ['--ranking']
        return parser().parse_args(values)

    def test_versioned_suffixes_reserve_old_ids_and_preserve_unchanged_prefix_slots(self):
        old = allocate_suffixes(['a','b','c'], [[0,0,0],[0,0,0],[1,0,0]], [2]*3, 4)
        untouched = copy.deepcopy(old)
        new = versioned_suffixes(old, ['a','b','c','d'], [[1,0,0],[0,0,0],[1,0,0],[0,0,0]])
        self.assertEqual(old, untouched)
        self.assertEqual([r['collision_suffix'] for r in new['entries']], [1,1,0,2])
        _, mapping = maps_from_ledger(new)
        self.assertEqual(len(set(map(tuple, mapping.values()))), 4)
        self.assertEqual(new['reserved_previous_entries'], old['entries'])

    def test_versioned_suffix_overflow_and_shrink_fail(self):
        old = allocate_suffixes(['a'], [[0,0,0]], [2]*3, 1)
        with self.assertRaisesRegex(ValueError, 'capacity exhausted'):
            versioned_suffixes(old, ['a','b'], [[1,0,0],[0,0,0]])
        with self.assertRaises(ValueError):
            versioned_suffixes(old, [], [])

    def test_bound_raw_histories_old_targets_and_masks_are_preserved(self):
        values = load_bound_inputs(self.args())
        arrays, examples, variants = values[4], values[6], values[7]
        for split, conditions in variants.items():
            for key, changed in conditions.items():
                self.assertTrue(np.array_equal(changed['labels'], arrays[split]['labels']))
                self.assertTrue(np.array_equal(changed['attention_mask'], arrays[split]['attention_mask']))
                self.assertTrue(np.array_equal(changed['source_rows'], arrays[split]['source_rows']))
                self.assertTrue(np.array_equal(changed['input_ids'] == 0, arrays[split]['input_ids'] == 0))
        old = variants['base_val']['old']['input_ids']
        new = variants['base_val']['new_full_sid']['input_ids']
        self.assertTrue(np.array_equal(old[old == 0], new[old == 0]))

    def test_identity_reencoding_actual_t5_delta_and_kl_are_zero(self):
        _, teacher, _, tokenizer, arrays, _, examples, variants, _ = load_bound_inputs(self.args())
        identity = reencode_history(arrays['base_train'], examples['base_train'], tokenizer.item2tokens)
        self.assertTrue(np.array_equal(identity['input_ids'], arrays['base_train']['input_ids']))
        model = TIGER(teacher['config'], None, tokenizer).eval()
        model.load_state_dict(teacher['state_dict'])
        model.requires_grad_(False)
        conditions = dict(old=arrays['base_train'], new_full_sid=identity, new_semantics_old_suffix=identity)
        records = paired_losses(model, conditions, examples['base_train'], list(range(6)), 3, 'cpu', 'identity')
        self.assertTrue(all(r['new_full_sid_delta_sequence_nll'] == 0 for r in records))
        self.assertTrue(all(r['new_full_sid_kl_sum'] == 0 for r in records))
        summary = aggregate(records, 'new_full_sid', 40, 2026)
        self.assertEqual(summary['mean_delta_sequence_nll'], 0)
        self.assertFalse(summary['positive_damage_detected'])

    def test_user_disjoint_cohorts_are_deterministic(self):
        _, _, _, _, _, _, examples, _, _ = load_bound_inputs(self.args())
        rows = examples['base_train']
        cohorts = historical_cohorts(rows, 12, 2026)
        self.assertEqual(cohorts, historical_cohorts(rows, 12, 2026))
        train_users = {rows[i]['user'] for i in cohorts['anchor_train']}
        holdout_users = {rows[i]['user'] for i in cohorts['anchor_holdout']}
        self.assertFalse(train_users & holdout_users)
        self.assertFalse(set(cohorts['anchor_train']) & set(cohorts['anchor_holdout']))
        self.assertEqual(len(cohorts['anchor_train']), 12)
        with self.assertRaisesRegex(ValueError, 'two historical users'):
            historical_cohorts([rows[0]], 12, 2026)

    def test_paired_aggregate_and_cluster_confidence_interval(self):
        records = []
        for user, delta in [('a',1.0), ('a',1.0), ('b',3.0), ('b',3.0)]:
            records.append(dict(user=user, old_sequence_nll=2.0, old_position_nll=[0.4]*5,
                new_full_sid_sequence_nll=2.0+delta,
                new_full_sid_position_nll=[(2.0+delta)/5]*5,
                new_full_sid_delta_sequence_nll=delta, new_full_sid_kl_sum=0.5,
                new_full_sid_history_changed=True))
        report = aggregate(records, 'new_full_sid', 100, 2026)
        self.assertEqual(report['mean_delta_sequence_nll'], 2.0)
        self.assertEqual(report['mean_positive_part_delta'], 2.0)
        self.assertEqual(report['worsened_fraction'], 1.0)
        self.assertTrue(report['positive_damage_detected'])
        self.assertGreaterEqual(report['cluster_bootstrap_mean_delta_ci95'][0], 1.0)

    def test_full_e1_real_models_ranking_hashes_and_frozen_inputs(self):
        root = Path(self.fixture.name)
        before = {str(p): sha256(p) for directory in ('sids','rqvae','tiger','update')
                  for p in (root/directory).rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            report = run(self.args(ranking=True))
        self.assertEqual(report['E1_status'], 'completed')
        self.assertEqual(report['historical_aware_training_status'], 'not_run')
        self.assertTrue(report['frozen_teacher_parameters_equal'])
        self.assertTrue(report['new_full_sids_unique'])
        self.assertTrue(report['old_validation_nll_reproduced'])
        self.assertEqual(report['base_validation_all_queries'], 10)
        self.assertEqual(report['base_validation_nll_queries'], 5)
        self.assertEqual(report['base_validation_targets_without_old_catalog_labels'], 5)
        ranking = report['base_validation_ranking']
        self.assertEqual(ranking['old']['unavailable_targets'], 5)
        self.assertEqual(ranking['new_full_sid']['unavailable_targets'], 5)
        self.assertEqual(ranking['new_full_sid']['invalid_predictions'], 0)
        for key in ranking['new_full_sid']['full_denominator']:
            self.assertAlmostEqual(ranking['new_full_sid']['full_denominator'][key],
                                   ranking['new_full_sid']['conditional_on_servable'][key]*0.5)
        for filename, expected in report['artifact_files_sha256'].items():
            self.assertEqual(sha256(self.root/'e1'/filename), expected)
        self.assertTrue(all(sha256(Path(name)) == value for name, value in before.items()))
        rows = [json.loads(x) for x in (self.root/'e1/base_validation_paired_losses.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(np.isfinite(r['new_full_sid_delta_sequence_nll']) for r in rows))

    def test_wrong_selected_teacher_fails_before_output(self):
        args = self.args()
        args.expected_tiger_sha256 = '0'*64
        with self.assertRaisesRegex(ValueError, 'TIGER SHA'):
            run(args)
        self.assertFalse(Path(args.output).exists())

    def test_corrupt_assignment_export_fails_before_output(self):
        update = self.root/'corrupt'
        shutil.copytree(Path(self.fixture.name)/'update', update)
        with (update/'semantic_assignments.jsonl').open('ab') as handle:
            handle.write(b'corrupt')
        args = self.args()
        args.update_rqvae_dir = str(update)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            run(args)
        self.assertFalse(Path(args.output).exists())

    def test_existing_e1_output_is_not_overwritten(self):
        args = self.args()
        Path(args.output).mkdir()
        with self.assertRaises(FileExistsError):
            run(args)


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False).result
    print(json.dumps(dict(round='round11', inputs='synthetic_only', tests_run=result.testsRun,
        execution_status='passed' if result.wasSuccessful() else 'failed',
        real_Beauty_E1_status='not_run_locally', historical_aware_training_status='not_run'), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
