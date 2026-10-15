"""Synthetic-only checks with the existing repository RQ-VAE and step trainer."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from gensid.plain_incremental_content import churn, load_inputs, parser, run
from gensid.incremental_content import sha256, read_json, write_json

torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('round8_fixture',
            ROOT/'tests/incremental/test_temporal_sids_round8.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.previous = module.Checks
        cls.previous.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.previous.tearDownClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ('protocol', 'features', 'rqvae'):
            shutil.copytree(Path(self.previous.fixture.name)/name, self.root/name)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, name, steps=8, resume=None):
        values = ['--protocol-dir', str(self.root/'protocol'), '--features-dir', str(self.root/'features'),
            '--base-rqvae-dir', str(self.root/'rqvae'), '--output', str(self.root/name),
            '--device', 'cpu', '--max-steps', str(steps), '--batch-size', '5',
            '--eval-batch-size', '4', '--lr', '0.005', '--log-every', '100', '--save-every', '2',
            '--expected-base-sha256', sha256(self.root/'rqvae/last.pth')]
        if resume:
            values += ['--resume-from', str(resume)]
        return parser().parse_args(values)

    def train(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.args(*args, **kwargs))

    def rehash_feature(self, name):
        summary = read_json(self.root/'features/summary.json')
        summary['artifact_files_sha256'][name] = sha256(self.root/'features'/name)
        write_json(self.root/'features/summary.json', summary)

    def test_warm_start_no_reinitialization_catalog_groups_and_preserved_parent(self):
        from gensid.models.vq import VectorQuantizer
        digest = sha256(self.root/'rqvae/last.pth')
        with patch.object(VectorQuantizer, 'init_emb', side_effect=AssertionError('reinitialization')):
            report = self.train('update', 7)
        self.assertTrue(report['warm_start_all_weights_equal'])
        self.assertTrue(report['original_base_assignments_equal_before_update'])
        self.assertTrue(report['parent_base_checkpoint_unchanged'])
        self.assertEqual(sha256(self.root/'rqvae/last.pth'), digest)
        self.assertEqual(report['training_catalog'], 'update_train')
        self.assertEqual(report['training_items'], 12)
        self.assertEqual(report['item_groups'], dict(base_train=9, base_validation_first_seen=2,
                                                    update_train_first_seen=1))
        self.assertEqual(report['parent_base_optimizer_steps'], 8)
        self.assertEqual(report['update_optimizer_steps'], 7)
        self.assertEqual(report['processed_examples'], 29)
        self.assertFalse(report['historical_prediction_penalty_used'])
        self.assertFalse(report['codebook_reinitialization'])
        self.assertTrue(report['optimizer_reset_at_update_start'])
        self.assertEqual(report['E1_status'], 'not_run')
        self.assertEqual(report['historical_prediction_damage_status'], 'not_run')

    def test_resume_exact_weights_optimizer_sampler_and_assignment_exports(self):
        self.train('continuous', 11)
        self.train('partial', 4)
        resumed = self.train('resumed', 11, self.root/'partial/last.pth')
        a = torch.load(self.root/'continuous/last.pth', weights_only=False)
        b = torch.load(self.root/'resumed/last.pth', weights_only=False)
        for key, value in a['state_dict'].items():
            self.assertTrue(torch.equal(value, b['state_dict'][key]), key)
        for key, state in a['optimizer']['state'].items():
            for field, value in state.items():
                self.assertTrue(torch.equal(value, b['optimizer']['state'][key][field]))
        for key in ('order', 'generator_state'):
            self.assertTrue(torch.equal(a['sampler_state'][key], b['sampler_state'][key]))
        self.assertTrue(torch.equal(a['rng_state']['torch'], b['rng_state']['torch']))
        self.assertEqual(a['sampler_state']['cursor'], b['sampler_state']['cursor'])
        self.assertEqual(resumed['steps_this_run'], 7)
        for name in ('semantic_assignments.jsonl', 'old_semantic_assignments.jsonl', 'base_assignment_changes.jsonl'):
            self.assertEqual((self.root/'continuous'/name).read_bytes(), (self.root/'resumed'/name).read_bytes())

    def test_actual_model_updates_with_finite_losses_and_reload(self):
        report = self.train('learn', 40)
        a = torch.load(self.root/'rqvae/last.pth', weights_only=False)
        b = torch.load(self.root/'learn/last.pth', weights_only=False)
        self.assertTrue(any(not torch.equal(v, b['state_dict'][key]) for key, v in a['state_dict'].items()))
        self.assertLess(report['final_train_catalog_metrics']['reconstruction_loss'],
                        report['initial_train_catalog_metrics']['reconstruction_loss'])
        self.assertTrue(report['checkpoint_reload_assignments_equal'])
        rows = [json.loads(x) for x in (self.root/'learn/metrics.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 40)
        self.assertTrue(all(np.isfinite(r['total_loss']) for r in rows))
        for name, expected in report['artifact_files_sha256'].items():
            self.assertEqual(sha256(self.root/'learn'/name), expected)

    def test_churn_counts_and_nested_prefixes(self):
        old = np.array([[0,0,0], [1,1,1], [0,1,0], [1,0,1]])
        new = np.array([[0,0,0], [0,1,1], [0,0,0], [1,0,0]])
        result = churn(old, new)
        self.assertEqual(result['changed_items'], 3)
        self.assertEqual(result['changed_item_fraction'], 0.75)
        self.assertEqual(result['per_level_changed_fraction'], [0.25]*3)
        self.assertEqual(result['prefix_changed_fraction'], [0.25, 0.5, 0.75])

    def test_bad_hash_and_wrong_parent_fail_before_output(self):
        with (self.root/'features/update_train/embeddings.npy').open('ab') as handle:
            handle.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.train('bad')
        self.assertFalse((self.root/'bad').exists())

    def test_changed_base_prefix_even_with_valid_hash_is_rejected(self):
        path = self.root/'features/update_train/embeddings.npy'
        values = np.load(path)
        values[0, 0] += 1
        np.save(path, values)
        self.rehash_feature('update_train/embeddings.npy')
        # Rebind the parent to isolate the row-content equality guard.
        from gensid.incremental_content import load_base_features
        binding = load_base_features(self.root/'features')[2]
        checkpoint = torch.load(self.root/'rqvae/last.pth', weights_only=False)
        checkpoint['binding']['feature_summary_sha256'] = binding['feature_summary_sha256']
        torch.save(checkpoint, self.root/'rqvae/last.pth')
        summary = read_json(self.root/'rqvae/summary.json')
        summary['binding']['feature_summary_sha256'] = binding['feature_summary_sha256']
        summary['artifact_files_sha256']['last.pth'] = sha256(self.root/'rqvae/last.pth')
        write_json(self.root/'rqvae/summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'Base prefix mismatch'):
            self.train('bad')
        self.assertFalse((self.root/'bad').exists())

    def test_expected_parent_and_changed_resume_config_are_rejected(self):
        args = self.args('bad')
        args.expected_base_sha256 = '0'*64
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA'):
            run(args)
        self.assertFalse((self.root/'bad').exists())
        self.train('partial', 2)
        args = self.args('changed', 4, self.root/'partial/last.pth')
        args.lr = 0.007
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            run(args)
        self.assertFalse((self.root/'changed').exists())

    def test_no_overwrite_and_raw_semantic_exports(self):
        report = self.train('update', 3)
        with self.assertRaises(FileExistsError):
            self.train('update', 4)
        rows = [json.loads(x) for x in (self.root/'update/semantic_assignments.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 12)
        self.assertEqual([r['gensid_global_item_id'] for r in rows], list(range(12)))
        self.assertTrue(all(len(r['semantic_code']) == 3 and all(0 <= c < 2 for c in r['semantic_code']) for r in rows))
        changes = [json.loads(x) for x in (self.root/'update/base_assignment_changes.jsonl').read_text().splitlines()]
        self.assertEqual(len(changes), 9)
        self.assertEqual(sum(r['changed'] for r in changes), report['base_semantic_churn']['changed_items'])


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False).result
    print(json.dumps(dict(round='round10', inputs='synthetic_only', tests_run=result.testsRun,
        execution_status='passed' if result.wasSuccessful() else 'failed',
        historical_prediction_damage_status='not_run', E1_status='not_run'), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
