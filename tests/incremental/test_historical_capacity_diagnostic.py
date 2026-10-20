"""Diagnostic guards locally; real RQ-VAE/TIGER replay checks when Torch exists."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import diagnose_historical_capacity as diagnostic

HAS_TORCH = importlib.util.find_spec('torch') is not None
if '--require-torch' in sys.argv:
    sys.argv.remove('--require-torch')
    if not HAS_TORCH:
        raise RuntimeError('Real replay checks require the existing Torch environment')


class ProtocolChecks(unittest.TestCase):
    def test_complete_and_partial_assignment_concentration_are_distinct(self):
        full = diagnostic.assignment_statistics([[0, 1, 0]]*5, [2]*3)
        self.assertTrue(full['all_items_share_one_full_semantic_code'])
        self.assertEqual(full['largest_bucket_fraction'], 1)
        partial = diagnostic.assignment_statistics([[0, 0, 0], [0, 0, 0], [1, 1, 0]], [2]*3)
        self.assertFalse(partial['all_items_share_one_full_semantic_code'])
        self.assertEqual(partial['unique_semantic_codes'], 2)
        self.assertEqual([x['used'] for x in partial['codebooks']], [2, 2, 1])
        self.assertEqual(partial['largest_prefix_buckets'], [2, 2, 2])

    def test_invalid_assignment_indices_fail(self):
        for codes in ([], [[-1, 0, 0]], [[0, 2, 0]], [[0, 0]], [[0., 0, 0]]):
            with self.assertRaises(ValueError):
                diagnostic.assignment_statistics(codes, [2]*3)

    def test_replay_is_bounded_and_targets_first_uncompleted_step(self):
        records = {i: dict(step=i) for i in range(400, 419)}
        self.assertEqual(diagnostic.replay_target(400, records), 419)
        for saved, limit in ((399, 50), (400, 18), (419, 50), (400, 51)):
            with self.assertRaises(ValueError):
                diagnostic.replay_target(saved, records, limit)

    def test_metrics_detect_drift_missing_fields_and_integer_changes(self):
        expected = dict(step=418, feasible=True, loss=0.001, gradients=dict(encoder=0.02))
        diagnostic.compare(expected, dict(expected, loss=0.00100000001))
        for changed in (dict(expected, step=419), dict(expected, feasible=1),
                        dict(expected, loss=0.002), dict(step=418)):
            with self.assertRaises(diagnostic.TraceMismatch):
                diagnostic.compare(expected, changed)

    def test_trace_rejects_gaps_and_duplicate_steps(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value)/'metrics.jsonl'
            for steps in ((1, 1), (1, 3), ()):
                path.write_text(''.join(json.dumps(dict(step=x))+'\n' for x in steps))
                with self.assertRaises(ValueError):
                    diagnostic.read_trace(path)
            path.write_text('\n'.join(json.dumps(dict(step=x)) for x in (1, 2, 3)))
            self.assertEqual(list(diagnostic.read_trace(path)), [1, 2, 3])

    def test_output_rejects_input_tree_and_existing_directories(self):
        with tempfile.TemporaryDirectory() as value:
            base = Path(value); source = base/'input'; source.mkdir()
            for output in (source, source/'new'):
                with self.assertRaises(ValueError):
                    diagnostic.ensure_separate(output.resolve(), [source])
            with self.assertRaises(FileExistsError):
                diagnostic.ensure_separate(base.resolve(), [source])
            diagnostic.ensure_separate((base/'new').resolve(), [source])

    def test_paths_are_parsed_as_data_with_no_shell_evaluation(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value)/'paths.txt'
            path.write_text('E1_DIR=a=b\nHISTORY_DIR=$(false)\n')
            self.assertEqual(diagnostic.read_paths(path)['HISTORY_DIR'], '$(false)')
            self.assertEqual(diagnostic.read_paths(path)['E1_DIR'], 'a=b')
            path.write_text('E1_DIR=a\nE1_DIR=b\n')
            with self.assertRaises(ValueError):
                diagnostic.read_paths(path)


@unittest.skipUnless(HAS_TORCH, 'Torch unavailable; real replay not tested locally')
class RealReplayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('round12_capacity_fixture',
            ROOT/'tests/incremental/test_historical_update_round12.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.fixture = module.Checks
        with contextlib.redirect_stdout(io.StringIO()):
            cls.fixture.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def setUp(self):
        from gensid import historical_incremental_content as engine
        self.engine = engine
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        trained = self.root/'original'; history = self.root/'history'; history.mkdir()
        original_save = engine.save

        def capture(path, *args):
            original_save(path, *args)
            if args[3]['step'] == 3:
                shutil.copy2(path, history/'last.pth')

        args = engine.parser().parse_args(['--e1-dir', str(self.fixture.e1_dir),
            '--output', str(trained), '--device', 'cpu', '--max-steps', '12',
            '--anchor-batch-size', '3', '--save-every', '3', '--log-every', '100',
            '--history-input-condition', 'new_full_sid'])
        engine.save = capture
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                engine.run(args)
        finally:
            engine.save = original_save
        shutil.copy2(trained/'config.json', history/'config.json')
        rows = (trained/'metrics.jsonl').read_text().splitlines()[:6]
        (history/'metrics.jsonl').write_text('\n'.join(rows)+'\n')
        self.run_dir = self.root/'failed-run'; self.run_dir.mkdir()
        (self.run_dir/'paths.txt').write_text(f'E1_DIR={self.fixture.e1_dir}\nHISTORY_DIR={history}\n')
        (self.run_dir/'commit.txt').write_text('synthetic-fixture\n')
        self.args = diagnostic.parser().parse_args(['--run-dir', str(self.run_dir),
            '--output', str(self.root/'diagnostic'), '--device', 'cpu'])
        self.history = history

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_restore_matches_logged_steps_and_preserves_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = diagnostic.run(self.args)
        self.assertEqual(report['outcome'], 'overflow_not_reproduced')
        self.assertEqual(report['compared_steps'], [4, 5, 6])
        self.assertEqual(report['current_attempt']['attempted_update_step'], 7)
        self.assertTrue(report['observed_inputs_unchanged'])
        self.assertFalse(report['checkpoint_written'])
        self.assertFalse((Path(self.args.output)/'last.pth').exists())

    def test_real_restore_stops_at_first_trace_mismatch(self):
        path = self.history/'metrics.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[3]['total_loss'] += 0.1
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with contextlib.redirect_stdout(io.StringIO()):
            report = diagnostic.run(self.args)
        self.assertEqual(report['outcome'], 'trace_mismatch')
        self.assertEqual(report['compared_steps'], [])
        self.assertEqual(report['current_attempt']['attempted_update_step'], 4)
        self.assertTrue(report['observed_inputs_unchanged'])


if __name__ == '__main__':
    unittest.main()
