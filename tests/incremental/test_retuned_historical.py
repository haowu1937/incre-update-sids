"""Protocol checks plus real repository RQ-VAE/TIGER paired integration checks."""
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import train_retuned_historical as runner


class Protocol(unittest.TestCase):
    def args(self):
        return runner.parser().parse_args(['--e1-dir', 'e1', '--output', 'new'])

    def test_fixed_defaults_and_invalid_settings(self):
        a = self.args()
        s = runner.settings(a)
        self.assertEqual((s['lr'], s['history_weight'], s['temperature'], s['anchor_batch_size'],
                          s['max_grad_norm'], s['max_steps']), (1e-4, 1e-4, .5, 128, .1, 1000))
        for key in ('lr', 'history_weight', 'temperature', 'max_grad_norm'):
            for value in (0., -1., math.inf, math.nan):
                b = copy.deepcopy(a)
                setattr(b, key, value)
                with self.assertRaises(ValueError):
                    runner.settings(b)
        for key in ('max_steps', 'anchor_batch_size', 'eval_batch_size', 'log_every', 'save_every'):
            b = copy.deepcopy(a)
            setattr(b, key, 0)
            with self.assertRaises(ValueError):
                runner.settings(b)

    @staticmethod
    def rows(nlls):
        return [dict(sample_id=str(i), user=str(i), target_labels=[1, 2], source_row=i,
                     old_sequence_nll=10., new_full_sid_sequence_nll=n) for i, n in enumerate(nlls)]

    def test_positive_damage_difference_is_not_nll_difference(self):
        p, h = self.rows([12., 8., 9.]), self.rows([11., 7., 8.])
        report = runner.bootstrap_differences(p, h, 100, 2026)
        self.assertEqual(report['mean_historical_minus_plain_sequence_nll'], -1.)
        self.assertAlmostEqual(report['mean_historical_minus_plain_positive_damage'], -1/3)
        self.assertEqual(report['sequence_nll_difference_ci95'], [-1., -1.])
        self.assertNotEqual(report['positive_damage_difference_ci95'], [-1., -1.])
        self.assertEqual(report, runner.bootstrap_differences(p, h, 100, 2026))

    def test_paired_queries_reference_and_finite_values(self):
        p = self.rows([11., 12.])
        for key, value in [('user', 'different'), ('target_labels', [3]), ('source_row', 5),
                           ('old_sequence_nll', 11.), ('new_full_sid_sequence_nll', math.nan)]:
            h = copy.deepcopy(p)
            h[0][key] = value
            with self.assertRaises(ValueError):
                runner.bootstrap_differences(p, h, 10, 1)
        for h, repeats in [(p[:1], 10), ([], 10), (p, 0)]:
            with self.assertRaises(ValueError):
                runner.bootstrap_differences(p, h, repeats, 1)

    def test_single_user_interval_not_claimed(self):
        p = self.rows([12., 8.])
        for row in p:
            row['user'] = 'same'
        report = runner.bootstrap_differences(p, p, 10, 1)
        self.assertEqual(report['mean_historical_minus_plain_positive_damage'], 0.)
        self.assertIsNone(report['sequence_nll_difference_ci95'])
        self.assertIsNone(report['positive_damage_difference_ci95'])


HAVE_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAVE_TORCH, 'PyTorch not installed; server launcher requires real checks')
class RealModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location('retuned_real_fixture',
            ROOT/'tests/incremental/test_historical_update_round12.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.fixture = module.Checks
        cls.fixture.setUpClass()
        cls.e1_dir = cls.fixture.e1_dir

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, name, resume=False):
        argv = ['--e1-dir', str(self.e1_dir), '--output', str(self.root/name), '--device', 'cpu',
                '--max-steps', '6', '--anchor-batch-size', '3', '--eval-batch-size', '3',
                '--log-every', '2', '--save-every', '2', '--ranking']
        return runner.parser().parse_args(argv+(['--resume'] if resume else []))

    @staticmethod
    def run_quiet(args):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(args)

    def assert_equal_nested(self, a, b):
        import numpy as np
        import torch
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            self.assertTrue(np.array_equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_equal_nested(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_equal_nested(x, y)
        else:
            self.assertEqual(a, b)

    def test_complete_pair_ranking_fixed_labels_and_resume_exactness(self):
        import torch
        from gensid import historical_incremental_content as h
        from gensid.trainer import Trainer
        original = Trainer.fit_steps
        full = self.run_quiet(self.args('full'))
        self.assertEqual(full['execution_status'], 'completed', full.get('error'))
        self.assertTrue(full['original_inputs_unchanged'])
        self.assertEqual(full['pair_checks']['same_update_steps'], 6)
        self.assertFalse(full['pair_checks']['same_compute'])
        evaluation = full['evaluation']
        self.assertEqual(evaluation['groups']['historical']['anchor_holdout']['new_full_sid']['queries'], 12)
        self.assertEqual(evaluation['base_validation_ranking']['historical']['all_queries'], 10)
        self.assertEqual(evaluation['base_validation_ranking']['historical']['invalid_predictions'], 0)
        self.assertEqual(full['training']['historical']['anchor_queries_processed'], 18)
        for name in ('plain', 'historical'):
            self.assertTrue(full['training'][name]['checkpoint_reload_assignments_equal'])
        ex = h.experiment(self.args('full'))
        for name in ('plain', 'historical'):
            path = next((self.root/'full/evaluation').glob('attempt_*/'+name+'_anchor_holdout.jsonl'))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            expected = ex.arrays['base_train']['labels'][ex.cohorts['anchor_holdout']].tolist()
            self.assertEqual([r['target_labels'] for r in records], expected)

        # Interrupt Historical after two accepted steps; Plain is complete and will be verified/reused.
        calls = 0
        def interrupted(trainer, features, ids, max_steps, batch_size, seed, state, callback):
            nonlocal calls
            calls += 1
            if calls != 2:
                return original(trainer, features, ids, max_steps, batch_size, seed, state, callback)
            def interrupt_after_save(s, metrics):
                callback(s, metrics)
                if s['step'] == 2:
                    raise KeyboardInterrupt()
            return original(trainer, features, ids, max_steps, batch_size, seed, state, interrupt_after_save)
        with patch.object(Trainer, 'fit_steps', interrupted):
            partial = self.run_quiet(self.args('resume'))
        self.assertEqual(partial['training']['historical']['execution_status'], 'interrupted')
        self.assertFalse((self.root/'resume/historical/failure.json').exists())
        resumed = self.run_quiet(self.args('resume', resume=True))
        self.assertEqual(resumed['execution_status'], 'completed', resumed.get('error'))
        for name in ('plain', 'historical'):
            a = torch.load(self.root/'full'/name/'last.pth', weights_only=False)
            b = torch.load(self.root/'resume'/name/'last.pth', weights_only=False)
            for key in ('state_dict', 'optimizer', 'sampler_state', 'anchor_sampler_state', 'rng_state'):
                self.assert_equal_nested(a[key], b[key])
        self.assertEqual(full['evaluation']['historical_minus_retuned_plain'],
                         resumed['evaluation']['historical_minus_retuned_plain'])
        # A partially written evaluation is retained and recomputed into a new attempt directory.
        (self.root/'resume/evaluation/summary.json').unlink()
        again = self.run_quiet(self.args('resume', resume=True))
        self.assertEqual(again['execution_status'], 'completed', again.get('error'))
        self.assertEqual(len(list((self.root/'resume/evaluation').glob('attempt_*'))), 2)

    def test_post_update_overflow_rolls_back_and_rejects_candidate(self):
        import torch
        from gensid import historical_incremental_content as h
        audit = h.suffix_capacity_audit
        calls = 0
        def forced_overflow(*args):
            nonlocal calls
            calls += 1
            result = audit(*args)
            if calls == 2:  # Plain initial audit succeeds; first post-optimizer audit fails.
                result.update(full_sid_allocation_feasible=False, maximum_required_suffix_capacity=257)
            return result
        with patch.object(h, 'suffix_capacity_audit', forced_overflow):
            report = self.run_quiet(self.args('overflow'))
        self.assertEqual(report['execution_status'], 'failed')
        self.assertTrue(report['original_inputs_unchanged'])
        self.assertNotIn('evaluation', report)
        directory = self.root/'overflow/plain'
        accepted = torch.load(directory/'last.pth', weights_only=False)
        candidate = torch.load(directory/'rejected_candidate.pth', weights_only=False)
        ex = h.experiment(self.args('overflow'))
        self.assertEqual(accepted['sampler_state']['step'], 0)
        self.assertEqual(candidate['sampler_state']['step'], 1)
        self.assertNotEqual(candidate['format'], runner.FORMAT)
        self.assertFalse(candidate['full_sid_export_ready'])
        self.assert_equal_nested(accepted['state_dict'], ex.parent['state_dict'])
        failure = json.loads((directory/'failure.json').read_text())
        self.assertTrue(failure['rollback_completed'])
        self.assertFalse(failure['automatic_retry'])
        retry = self.run_quiet(self.args('overflow', resume=True))
        self.assertEqual(retry['execution_status'], 'failed')
        self.assertIn('failed attempt is preserved', retry['error'])


if __name__ == '__main__':
    require = '--require-torch' in sys.argv
    if require:
        sys.argv.remove('--require-torch')
    if require and not HAVE_TORCH:
        raise SystemExit('PyTorch required for real repository RQ-VAE/TIGER checks')
    unittest.main()
