"""Check that smoke inference cannot masquerade as a completed feature export."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import check_incremental_llama as smoke

spec = importlib.util.spec_from_file_location('llama_fixture', Path(__file__).with_name('test_llama_content.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class Checks(unittest.TestCase):
    def setUp(self):
        self.f = fixture.Checks()
        self.addCleanup(self.f.doCleanups)
        self.f.setUp()
        self.args = self.f.args
        self.args.items = 2
        self.args.seed = 2026
        self.args.output = str(self.f.root / 'smoke')

    def test_small_output_is_separate_and_rejected_as_training_artifact(self):
        result = smoke.run(self.args, encoder_factory=fixture.FakeEncoder)
        self.assertEqual(len(fixture.FakeEncoder.calls), 2)
        self.assertEqual(result['stage'], 'llama_content_smoke_v1')
        self.assertEqual(result['shape'], [2, 3])
        self.assertFalse(result['usable_as_training_features'])
        self.assertFalse((Path(self.args.output) / 'base_train').exists())
        self.assertFalse(Path(self.args.output + '.incomplete').exists())
        self.assertEqual(result['training_status'], 'not_run')

    def test_completed_smoke_is_not_overwritten(self):
        smoke.run(self.args, encoder_factory=fixture.FakeEncoder)
        previous = (Path(self.args.output) / 'summary.json').read_bytes()
        with self.assertRaises(FileExistsError):
            smoke.run(self.args, encoder_factory=fixture.FakeEncoder)
        self.assertEqual((Path(self.args.output) / 'summary.json').read_bytes(), previous)

    def test_failure_preserves_diagnostic_without_claiming_full_features(self):
        fixture.FakeEncoder.fail_at = 1
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            smoke.run(self.args, encoder_factory=fixture.FakeEncoder)
        result = smoke.llama._json(Path(self.args.output) / 'summary.json')
        self.assertEqual(result['execution_status'], 'failed')
        self.assertFalse(result['usable_as_training_features'])
        self.assertFalse((Path(self.args.output) / 'sample_embeddings.npy').exists())

    def test_existing_partial_encoding_is_preserved(self):
        pending = Path(self.args.output + '.incomplete')
        pending.mkdir()
        (pending / 'sentinel.txt').write_text('retain original partial encoding')
        with self.assertRaises(FileExistsError):
            smoke.run(self.args, encoder_factory=fixture.FakeEncoder)
        self.assertFalse(Path(self.args.output).exists())
        self.assertEqual((pending / 'sentinel.txt').read_text(), 'retain original partial encoding')
        self.assertFalse(fixture.FakeEncoder.calls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
