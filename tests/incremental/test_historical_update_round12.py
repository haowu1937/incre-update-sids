"""Synthetic checks with repository RQ-VAE, repository Trainer and frozen TIGER."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
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
from torch.nn import functional as F
from gensid.historical_incremental_content import (HistoryPenalty, catalog_codes, check_teacher,
    distance_scales, evaluate_update, evaluation_parser, experiment, frozen_teacher, parser, run)
from gensid.incremental_content import build_model, sha256
from genrec.historical_damage import reencode_history, versioned_suffixes
from genrec.temporal_sids import maps_from_ledger
torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('round11_fixture',
            ROOT/'tests/incremental/test_historical_damage_round11.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.previous = module.Checks
        cls.previous.setUpClass()
        cls.old_test = cls.previous()
        cls.old_test.setUp()
        cls.e1_args = cls.old_test.args(ranking=True)
        with contextlib.redirect_stdout(io.StringIO()):
            module.run(cls.e1_args)
        cls.e1_dir = Path(cls.e1_args.output)

    @classmethod
    def tearDownClass(cls):
        cls.old_test.tearDown()
        cls.previous.tearDownClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, name='compatible', steps=12, weight=0.001, resume=None):
        values = ['--e1-dir', str(self.e1_dir), '--output', str(self.root/name),
            '--device', 'cpu', '--max-steps', str(steps), '--compatibility-weight', str(weight),
            '--anchor-batch-size', '3', '--log-every', '100', '--save-every', '100']
        if resume:
            values += ['--resume-from', str(resume)]
        return parser().parse_args(values)

    def train(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(args)

    def test_full_hard_forward_preserves_labels_masks_and_gradient_reaches_all_semantic_levels(self):
        ex = experiment(self.args())
        model = build_model(ex.parent['config'])
        model.load_state_dict(ex.parent['state_dict'])
        scales = distance_scales(model, ex.features[:ex.groups['base_train'][1]], ex.old_batch)
        # Evaluate a real updated RQ-VAE, rather than fake SID assignments.
        plain = torch.load(Path(ex.e1['update_rqvae_dir'])/'last.pth', weights_only=False)
        model.load_state_dict(plain['state_dict'])
        teacher = frozen_teacher(ex, 'cpu')
        config = dict(anchor_batch_size=3, anchor_sampler_seed=2027, temperature=0.1)
        penalty = HistoryPenalty(model, teacher, ex, config, scales)
        local = np.arange(len(penalty.selected))
        old, hard_ids, embeds = penalty.input_embeds(local)
        codes = catalog_codes(model, ex)
        ledger = versioned_suffixes(ex.previous, [r['raw_item_id'] for r in ex.feature_rows], codes.tolist())
        _, mapping = maps_from_ledger(ledger)
        expected = reencode_history(ex.arrays['base_train'], ex.rows['base_train'], mapping)
        self.assertTrue(np.array_equal(hard_ids.numpy(), expected['input_ids'][penalty.selected, :hard_ids.shape[1]]))
        self.assertTrue(torch.equal(embeds.detach(), teacher.t5.get_input_embeddings()(hard_ids)))
        self.assertTrue(np.array_equal(old['labels'].numpy(), ex.arrays['base_train']['labels'][penalty.selected]))
        self.assertTrue(np.array_equal(old['attention_mask'].numpy(),
            ex.arrays['base_train']['attention_mask'][penalty.selected, :hard_ids.shape[1]]))
        teacher_logits = teacher(dict(input_ids=hard_ids, attention_mask=old['attention_mask'],
                                     labels=old['labels'], use_cache=False)).logits
        logits = teacher(dict(inputs_embeds=embeds, attention_mask=old['attention_mask'],
                              labels=old['labels'], use_cache=False)).logits
        self.assertTrue(torch.equal(logits.detach(), teacher_logits.detach()))
        F.cross_entropy(logits.transpose(1, 2), old['labels']).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
        for layer in model.rq.vq_layers:
            self.assertIsNotNone(layer.embedding.weight.grad)
            self.assertGreater(float(layer.embedding.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
        check_teacher(teacher, ex)

    def test_zero_weight_exactly_reproduces_plain_update(self):
        report = self.train(self.args(weight=0))
        ex = experiment(self.args())
        old = torch.load(Path(ex.e1['update_rqvae_dir'])/'last.pth', weights_only=False)
        new = torch.load(self.root/'compatible/last.pth', weights_only=False)
        self.assertTrue(all(torch.equal(v, new['state_dict'][k]) for k, v in old['state_dict'].items()))
        self.assertTrue(torch.equal(old['sampler_state']['order'], new['sampler_state']['order']))
        self.assertEqual(old['sampler_state']['cursor'], new['sampler_state']['cursor'])
        self.assertEqual(report['anchor_queries_processed'], 0)
        self.assertEqual(report['final_train_catalog_metrics'], ex.plain['final_train_catalog_metrics'])

    def test_nonzero_resume_is_exact_and_anchor_state_is_persistent(self):
        full = self.train(self.args(name='full'))
        self.train(self.args(name='smoke', steps=3))
        resumed = self.train(self.args(name='resumed', resume=self.root/'smoke/last.pth'))
        a = torch.load(self.root/'full/last.pth', weights_only=False)
        b = torch.load(self.root/'resumed/last.pth', weights_only=False)
        self.assertTrue(all(torch.equal(v, b['state_dict'][k]) for k, v in a['state_dict'].items()))
        for key in ('cursor', 'epoch', 'processed_queries'):
            self.assertEqual(a['anchor_sampler_state'][key], b['anchor_sampler_state'][key])
        self.assertTrue(torch.equal(a['anchor_sampler_state']['order'], b['anchor_sampler_state']['order']))
        self.assertTrue(torch.equal(a['anchor_sampler_state']['generator_state'], b['anchor_sampler_state']['generator_state']))
        self.assertEqual(full['final_train_catalog_metrics'], resumed['final_train_catalog_metrics'])
        self.assertEqual(resumed['steps_this_run'], 9)
        self.assertEqual(full['anchor_queries_processed'], 36)

    def test_identity_history_has_zero_training_penalty_and_never_uses_holdout(self):
        ex = experiment(self.args())
        model = build_model(ex.parent['config'])
        model.load_state_dict(ex.parent['state_dict'])
        teacher = frozen_teacher(ex, 'cpu')
        penalty = HistoryPenalty(model, teacher, ex,
            dict(anchor_batch_size=3, anchor_sampler_seed=2027, temperature=0.1),
            distance_scales(model, ex.features[:ex.groups['base_train'][1]], ex.old_batch))
        value = penalty()
        self.assertEqual(float(value), 0)
        self.assertEqual(penalty.last_metrics['anchor_changed_history_fraction'], 0)
        self.assertFalse(set(penalty.selected) & set(ex.cohorts['anchor_holdout']))
        self.assertEqual(set(penalty.selected), set(ex.cohorts['anchor_train']))

    def test_matched_real_model_evaluation_and_all_hashes(self):
        before = {str(p): sha256(p) for directory in [self.e1_dir, Path(self.previous.fixture.name)]
                  for p in directory.rglob('*') if p.is_file()}
        self.train(self.args())
        args = evaluation_parser().parse_args(['--e1-dir', str(self.e1_dir),
            '--historical-rqvae-dir', str(self.root/'compatible'), '--output', str(self.root/'evaluation'),
            '--device', 'cpu', '--batch-size', '3', '--ranking'])
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluate_update(args)
        self.assertTrue(report['teacher_frozen'])
        self.assertEqual(report['groups']['anchor_holdout']['new_full_sid']['queries'], 12)
        self.assertEqual(report['groups']['base_validation']['new_full_sid']['queries'], 5)
        ranking = report['base_validation_ranking']
        self.assertEqual(ranking['new_full_sid']['all_queries'], 10)
        self.assertEqual(ranking['new_full_sid']['unavailable_targets'], 5)
        self.assertEqual(ranking['new_full_sid']['invalid_predictions'], 0)
        self.assertEqual(len(report['paired_compatible_vs_plain']['anchor_holdout']['cluster_bootstrap_ci95']), 2)
        for name, digest in report['artifact_files_sha256'].items():
            self.assertEqual(sha256(self.root/'evaluation'/name), digest)
        self.assertTrue(all(sha256(Path(name)) == value for name, value in before.items()))

    def test_wrong_budget_and_bad_weight_fail_before_output(self):
        args = self.args(steps=13)
        with self.assertRaisesRegex(ValueError, 'budget'):
            run(args)
        self.assertFalse(Path(args.output).exists())
        args = self.args(weight=float('nan'))
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            run(args)
        self.assertFalse(Path(args.output).exists())

    def test_resume_changed_objective_is_rejected_before_output(self):
        self.train(self.args(name='smoke', steps=3))
        args = self.args(weight=0.002, resume=self.root/'smoke/last.pth')
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            run(args)
        self.assertFalse(Path(args.output).exists())

    def test_existing_output_not_overwritten(self):
        args = self.args()
        Path(args.output).mkdir()
        with self.assertRaises(FileExistsError):
            run(args)


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False).result
    print(json.dumps(dict(round='round12', inputs='synthetic_only', tests_run=result.testsRun,
        execution_status='passed' if result.wasSuccessful() else 'failed',
        real_Beauty_historical_aware_training_status='not_run_locally'), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
