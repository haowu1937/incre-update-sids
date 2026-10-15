"""Regression: transient 300-item collision must not block input-only training."""
import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from genrec.temporal_sids import allocate_suffixes
from genrec.historical_damage import versioned_suffixes
from gensid.incremental_content import build_model
from gensid.historical_incremental_content import (HistoryPenalty, catalog_codes, distance_scales,
    experiment, frozen_teacher, run, suffix_capacity_audit, evaluation_parser, evaluate_update)
torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('original_round12_fixture',
            ROOT/'tests/incremental/test_historical_update_round12.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.previous = module.Checks
        cls.previous.setUpClass()
        cls.helper = cls.previous()
        cls.helper.setUp()

    @classmethod
    def tearDownClass(cls):
        cls.helper.tearDown()
        cls.previous.tearDownClass()

    def test_genuine_300_item_overflow_and_reserved_slot_overflow_are_distinguished(self):
        names = ['i'+str(i) for i in range(300)]
        old_codes = [[i//256, i%256, 0] for i in range(300)]
        old = allocate_suffixes(names, old_codes, [256]*3, 256)
        new = [[0, 0, 0]]*300
        with self.assertRaisesRegex(ValueError, 'capacity exhausted'):
            versioned_suffixes(old, names, new)
        audit = suffix_capacity_audit(old, names, new)
        self.assertEqual(audit['maximum_current_collision_bucket'], 300)
        self.assertEqual(audit['maximum_required_suffix_capacity'], 300)
        self.assertFalse(audit['full_sid_allocation_feasible'])
        old = allocate_suffixes(['a','b'], [[0,0,0],[0,0,0]], [2]*3, 2)
        # Only two items in the destination bucket, but two old slots reserved plus one appended.
        audit = suffix_capacity_audit(old, ['a','b','c'], [[1,0,0],[0,0,0],[0,0,0]])
        self.assertEqual(audit['maximum_current_collision_bucket'], 2)
        self.assertEqual(audit['maximum_required_suffix_capacity'], 3)
        self.assertFalse(audit['full_sid_allocation_feasible'])

    def test_fixed_old_suffix_avoids_allocator_and_keeps_real_teacher_gradient(self):
        ex = experiment(self.helper.args())
        model = build_model(ex.parent['config'])
        model.load_state_dict(ex.parent['state_dict'])
        teacher = frozen_teacher(ex, 'cpu')
        config = dict(anchor_batch_size=3, anchor_sampler_seed=2027, temperature=0.1,
                      history_input_condition='new_semantics_old_suffix')
        penalty = HistoryPenalty(model, teacher, ex, config,
            distance_scales(model, ex.features[:ex.groups['base_train'][1]], ex.old_batch))
        # A real code assignment matrix with every eligible item in one prefix.
        collapsed = np.zeros_like(catalog_codes(model, ex))
        with patch('gensid.historical_incremental_content.catalog_codes', return_value=collapsed), \
             patch('gensid.historical_incremental_content.versioned_suffixes',
                   side_effect=AssertionError('training must not allocate complete IDs')):
            old, hard, embeds = penalty.input_embeds(np.arange(len(penalty.selected)))
        self.assertTrue(torch.equal(hard[:, 3::4], old['input_ids'][:, 3::4]))
        self.assertTrue(torch.equal(embeds.detach(), teacher.t5.get_input_embeddings()(hard)))
        self.assertTrue(torch.equal(old['attention_mask'].bool(), hard != 0))
        logits = teacher(dict(inputs_embeds=embeds, attention_mask=old['attention_mask'],
                              labels=old['labels'], use_cache=False)).logits
        F.cross_entropy(logits.transpose(1, 2), old['labels']).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))

    def test_overflow_at_smoke_end_keeps_checkpoint_resumable_without_fake_export(self):
        args = self.helper.args(name='overflow_smoke', steps=3)
        import gensid.historical_incremental_content as module
        real_audit = module.suffix_capacity_audit
        def force_overflow(*a, **kw):
            report = real_audit(*a, **kw)
            report.update(full_sid_allocation_feasible=False, maximum_required_suffix_capacity=300)
            return report
        with patch.object(module, 'suffix_capacity_audit', side_effect=force_overflow), \
             contextlib.redirect_stdout(io.StringIO()):
            report = run(args)
        self.assertEqual(report['full_sid_export_status'], 'capacity_exhausted')
        self.assertFalse(report['deployment_sid_ready'])
        output = Path(args.output)
        self.assertTrue((output/'last.pth').exists())
        self.assertFalse((output/'new_item2tokens.json').exists())
        self.assertGreater(report['full_sid_infeasible_training_steps'], 0)
        resumed = self.helper.args(name='after_overflow', steps=12, resume=output/'last.pth')
        with contextlib.redirect_stdout(io.StringIO()):
            continued = run(resumed)
        self.assertEqual(continued['update_optimizer_steps'], 12)
        self.assertEqual(continued['steps_this_run'], 9)
        # Capacity-invalid outputs must not produce ranking or be called deployment-ready.
        eval_args = evaluation_parser().parse_args(['--e1-dir', str(self.helper.e1_dir),
            '--historical-rqvae-dir', str(output), '--output', str(output.parent/'invalid_eval'), '--device','cpu'])
        with self.assertRaisesRegex(ValueError, 'capacity_exhausted'):
            evaluate_update(eval_args)
        self.assertFalse(Path(eval_args.output).exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
