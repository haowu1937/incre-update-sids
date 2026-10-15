"""CPU-only diagnostic checks on synthetic data using actual repository models."""
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
from diagnose_historical_collapse import gradient_contributions, parser, run, run_lane
from gensid.incremental_content import build_model, build_trainer, seed_all, sha256
from gensid.historical_incremental_content import experiment
torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('fixture', ROOT/'tests/incremental/test_historical_update_round12.py')
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

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_separate_gradient_ratio_cosine_and_no_grad_accumulation(self):
        ex = experiment(self.helper.args())
        model = build_model(ex.parent['config'])
        parameter = next(model.encoder.parameters())
        rq = parameter.sum()
        penalty = -2*parameter.sum()
        result = gradient_contributions(model, rq, penalty, 0.25)
        self.assertAlmostEqual(result['encoder']['weighted_history_to_rq_norm_ratio'], 0.5)
        self.assertAlmostEqual(result['encoder']['rq_history_cosine'], -1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_probe_zero_weight_reproduces_actual_original_training_weights(self):
        ex = experiment(self.helper.args())
        config = dict(ex.plain['config'], device='cpu')
        seed_all(config['seed'])
        model = build_model(config)
        model.load_state_dict(ex.parent['state_dict'])
        reference = self.root/'reference'
        reference.mkdir()
        trainer = build_trainer(model, config, reference)
        trainer.fit_steps(ex.features, torch.arange(len(ex.features)), 6,
                          config['batch_size'], config['seed'])
        directory = self.root/'probe'
        directory.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            report, state = run_lane(ex, 'cpu', 6, 0.0, 3, directory, quiet=True)
        self.assertTrue(all(torch.equal(v, state[k]) for k, v in model.state_dict().items()))
        self.assertTrue(report['teacher_parameters_unchanged'])
        trainer.tb_writer.close()

    def test_full_matched_diagnostic_keeps_inputs_and_reports_both_lanes(self):
        ex = experiment(self.helper.args())
        before = {str(p): sha256(p) for root in [ex.e1_dir, Path(ex.plain['base_rqvae_dir'])]
                  for p in root.rglob('*') if p.is_file()}
        args = parser().parse_args(['--e1-dir', str(ex.e1_dir), '--output', str(self.root/'diagnostic'),
                                    '--device', 'cpu', '--steps', '6', '--anchor-batch-size', '3'])
        with contextlib.redirect_stdout(io.StringIO()):
            report = run(args)
        self.assertEqual(len(report['lanes']), 2)
        self.assertEqual(report['lanes']['lambda_0p0']['content_items_processed'],
                         report['lanes']['lambda_0p001']['content_items_processed'])
        self.assertEqual(report['lanes']['lambda_0p0']['initial'], report['lanes']['lambda_0p001']['initial'])
        self.assertTrue(report['diagnostic_only'])
        self.assertTrue(all(sha256(Path(p)) == value for p, value in before.items()))
        import json
        first = [json.loads((self.root/'diagnostic'/name/'trace.jsonl').read_text().splitlines()[0])
                 for name in ['lambda_0p0', 'lambda_0p001']]
        self.assertEqual(first[0]['post_update'], first[1]['post_update'])
        self.assertEqual(first[0]['history_penalty'], 0)
        self.assertEqual(first[1]['history_penalty'], 0)
        self.assertFalse(any((self.root/'diagnostic').rglob('*.pth')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
