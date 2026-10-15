"""Actual repository models on synthetic CPU data: local direction probe."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from check_history_gradient_direction import parser, run
from gensid.incremental_content import sha256
torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('previous_fixture', ROOT/'tests/incremental/test_historical_update_round12.py')
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

    def test_existing_ste_and_continuous_tangent_match_and_inputs_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'direction'
            args = parser().parse_args(['--e1-dir', str(self.helper.e1_dir), '--output', str(output),
                '--device', 'cpu', '--queries', '12', '--batch-size', '3', '--scopes', 'joint',
                '--relative-steps', '0.0001', '0.001'])
            before = {str(p): sha256(p) for p in Path(self.previous.previous.fixture.name).rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertLess(result['gradient_match']['relative_gradient_difference'], 1e-5)
            self.assertGreater(result['gradient_match']['gradient_cosine'], 0.99999)
            self.assertEqual(len(result['candidates']), 4)
            self.assertTrue(result['source_parameters_restored'])
            self.assertTrue(result['teacher_unchanged'])
            self.assertEqual(result['optimizer_steps'], 0)
            self.assertFalse(result['holdout_used'])
            for row in result['candidates']:
                self.assertEqual(row['sign'], -1 if row['predicted_tangent_change'] < 0 else 1)
                self.assertGreater(row['hard']['unique_semantic_codes'], 0)
            # At a sufficiently small tested step, the differentiable local function descends.
            small_down = next(x for x in result['candidates'] if x['relative_step'] == 0.0001 and x['sign'] == -1)
            self.assertLess(small_down['tangent_positive_damage_change'], 0)
            self.assertTrue(all(sha256(Path(p)) == v for p, v in before.items()))
            self.assertFalse(any(output.rglob('*.pth')))

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parser().parse_args(['--e1-dir', str(self.helper.e1_dir), '--output', folder, '--device', 'cpu'])
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
