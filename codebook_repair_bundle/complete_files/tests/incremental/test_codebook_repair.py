"""Synthetic CPU data using the repository RQ-VAE and frozen TIGER."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import repair_historical_codebooks as repair
from gensid.incremental_content import build_model, sha256
torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('previous_fixture', ROOT/'tests/incremental/test_historical_update_round12.py')
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

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, name='repair', *extra):
        return repair.parser().parse_args(['repair', '--e1-dir', str(self.previous.e1_dir),
            '--output', str(self.root/name), '--device', 'cpu', '--max-rounds', '2',
            '--gradient-queries', '9', '--batch-size', '3', '--relative-steps', '0.001', '0.0001', *extra])

    def run_repair(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return repair.repair(args)

    def test_real_model_repair_full_verification_frozen_networks_reload_and_evaluation(self):
        ex = repair.h.experiment(self.args())
        before = {str(p): sha256(p) for directory in [ex.e1_dir, Path(self.previous.previous.fixture.name)]
            for p in directory.rglob('*') if p.is_file()}
        with mock.patch.object(repair.direction, 'hard_objective', wraps=repair.direction.hard_objective) as hard:
            report = self.run_repair(self.args())
            self.assertTrue(all(sum(len(pack['indices']) for pack in call.args[3]) == 12 for call in hard.call_args_list))
        self.assertEqual(report['gradient_subset_queries'], 9)
        self.assertEqual(report['hard_acceptance_queries'], 12)
        self.assertFalse(report['anchor_holdout_used_for_selection'])
        self.assertFalse(report['base_validation_used_for_selection'])
        self.assertLessEqual(report['final_full_anchor_train']['positive_damage'], report['initial_full_anchor_train']['positive_damage']+1e-7)
        self.assertLessEqual(report['final_train_catalog_metrics']['reconstruction_loss'], report['fixed_reconstruction_mse_cap'])
        cp = torch.load(self.root/'repair/last.pth', weights_only=False)
        original = torch.load(Path(ex.e1['update_rqvae_dir'])/'last.pth', weights_only=False)
        for k, v in cp['state_dict'].items():
            if k.startswith(('encoder.', 'decoder.')):
                self.assertTrue(torch.equal(v, original['state_dict'][k]))
        selected = repair.read_json(self.root/'repair/gradient_anchor_train_rows.json')
        self.assertFalse(set(selected) & set(ex.cohorts['anchor_holdout']))
        self.assertEqual(repair.read_json(self.root/'repair/verification_anchor_train_rows.json'), ex.cohorts['anchor_train'])
        for name, digest in report['artifact_files_sha256'].items():
            self.assertEqual(sha256(self.root/'repair'/name), digest)
        args = repair.parser().parse_args(['evaluate', '--e1-dir', str(ex.e1_dir), '--repair-dir', str(self.root/'repair'),
            '--output', str(self.root/'eval'), '--device', 'cpu', '--batch-size', '3', '--ranking'])
        with contextlib.redirect_stdout(io.StringIO()):
            result = repair.evaluate_repair(args)
        self.assertEqual(result['groups']['anchor_holdout']['new_full_sid']['queries'], 12)
        self.assertEqual(result['base_validation_ranking']['repaired']['all_queries'], 10)
        self.assertEqual(result['base_validation_ranking']['repaired']['unavailable_targets'], 5)
        self.assertEqual(result['base_validation_ranking']['repaired']['invalid_predictions'], 0)
        self.assertEqual(len(result['paired_repair_vs_plain']['anchor_holdout']['new_semantics_old_suffix']['positive_damage_cluster_ci95']), 2)
        self.assertTrue(all(sha256(Path(path)) == digest for path, digest in before.items()))

    def test_rejected_proposals_restore_exact_plain_weights(self):
        report = self.run_repair(self.args('rejected', '--min-decrease', '100'))
        self.assertEqual(report['accepted_repair_steps'], 0)
        self.assertEqual(report['stop_reason'], 'no_acceptable_descent_proposal')
        ex = repair.h.experiment(self.args())
        a = torch.load(self.root/'rejected/last.pth', weights_only=False)
        b = torch.load(Path(ex.e1['update_rqvae_dir'])/'last.pth', weights_only=False)
        self.assertTrue(all(torch.equal(v, b['state_dict'][k]) for k, v in a['state_dict'].items()))

    def test_frozen_networks_and_actual_codebook_gradient(self):
        ex = repair.h.experiment(self.args())
        cp = torch.load(Path(ex.e1['update_rqvae_dir'])/'last.pth', weights_only=False)
        model = build_model(cp['config'])
        model.load_state_dict(cp['state_dict'])
        repair.freeze_content_networks(model)
        before = repair.snapshot(model)
        teacher = repair.h.frozen_teacher(ex, 'cpu')
        base = build_model(ex.parent['config'])
        base.load_state_dict(ex.parent['state_dict'])
        scales = repair.h.distance_scales(base, ex.features[:ex.groups['base_train'][1]], ex.old_batch)
        packs, _ = repair.direction.contexts(model, teacher, ex, np.asarray(ex.cohorts['anchor_train']), 3, scales, 'cpu')
        _, gradient = repair.direction.objective(model, teacher, ex, packs, scales, 'cpu', 'ste', backward=True)
        self.assertGreater(sum(float(v.abs().sum()) for k, v in gradient.items() if k.startswith('rq.')), 0)
        self.assertTrue(all(p.grad is None for k, p in model.named_parameters() if k.startswith(('encoder.', 'decoder.'))))
        repair.direction.perturb(model, before, gradient, 'codebooks', .001, -1)
        repair.verify_frozen(model, before)
        self.assertTrue(any(not torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items() if k.startswith('rq.')))
        repair.h.check_teacher(teacher, ex)

    def test_acceptance_rejects_increase_overflow_and_fixed_mse_cap(self):
        self.assertTrue(repair.acceptance(.15, .149, .101, .101, True, 1e-6)[0])
        self.assertFalse(repair.acceptance(.15, .151, .1, .101, True, 1e-6)[0])
        self.assertFalse(repair.acceptance(.15, .15, .1, .101, True, 1e-6)[0])
        self.assertFalse(repair.acceptance(.15, .149, .1001, .101, False, 1e-6)[0])
        self.assertFalse(repair.acceptance(.15, .149, .1011, .101, True, 1e-6)[0])
        self.assertFalse(repair.acceptance(.15, float('nan'), .1, .101, True, 1e-6)[0])

    def test_paired_positive_damage_contrast_and_query_identity(self):
        ex = repair.h.experiment(self.args())
        rows = repair.read_jsonl(ex.e1_dir/'anchor_holdout_paired_losses.jsonl')
        result = repair.paired_contrast(rows, rows, 'new_semantics_old_suffix', 50, 2026)
        self.assertEqual(result['mean_repair_minus_plain_sequence_nll'], 0)
        self.assertEqual(result['mean_repair_minus_plain_positive_damage'], 0)
        self.assertEqual(result['positive_damage_cluster_ci95'], [0, 0])
        changed = [dict(x) for x in rows]
        changed[0]['sample_id'] = 'incorrect'
        with self.assertRaisesRegex(ValueError, 'query'):
            repair.paired_contrast(changed, rows, 'new_full_sid', 50, 2026)

    def test_invalid_arguments_existing_output_and_tamper_fail(self):
        args = self.args('invalid', '--mse-slack', '-0.1')
        with self.assertRaises(ValueError):
            repair.repair(args)
        self.assertFalse(Path(args.output).exists())
        self.root.joinpath('existing').mkdir()
        with self.assertRaises(FileExistsError):
            repair.repair(self.args('existing'))
        self.run_repair(self.args())
        with (self.root/'repair/new_item2tokens.json').open('a') as f:
            f.write(' ')
        args = repair.parser().parse_args(['evaluate', '--e1-dir', str(self.previous.e1_dir),
            '--repair-dir', str(self.root/'repair'), '--output', str(self.root/'bad_eval'), '--device', 'cpu'])
        with self.assertRaises(ValueError):
            repair.evaluate_repair(args)
        self.assertFalse(Path(args.output).exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
