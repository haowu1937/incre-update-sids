"""Synthetic gzip-to-artifact integration. No Beauty data or model download."""
import argparse
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genrec.temporal_artifacts import prepare_artifacts, file_sha256

class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reviews = self.root / 'reviews.json.gz'
        with gzip.open(self.reviews, 'wt', encoding='utf-8') as handle:
            for i in range(1, 101):
                handle.write(json.dumps({'reviewerID':('u' if i<91 else 'v'), 'asin':f'i{i%4}',
                                         'unixReviewTime':i, 'overall':4.5}) + '\n')
    def run_prepare(self, name='out'):
        return prepare_artifacts(self.reviews, self.root/name, 5, 'synthetic')
    def test_files_hashes_and_rows(self):
        result = self.run_prepare()
        self.assertEqual(result['reviews_file_sha256'], file_sha256(self.reviews))
        self.assertEqual(result['preprocessing']['input_events'], 100)
        for name, expected in result['artifact_files_sha256'].items():
            self.assertEqual(file_sha256(self.root/'out'/name), expected)
        for stage, stats in result['windows'].items():
            rows = [json.loads(line) for line in (self.root/'out'/f'{stage}.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), stats['example_count'])
            for row in rows:
                self.assertTrue(all(t < row['target_time'] for t in row['time_seq'][:-1]))
                self.assertLessEqual(len(row['item_seq'])-1,5)
                self.assertEqual(row['rating_seq'][-1],4.5)
    def test_deterministic_protocol(self):
        a=self.run_prepare('a'); b=self.run_prepare('b')
        self.assertEqual(a['protocol_sha256'],b['protocol_sha256'])
        self.assertEqual(a['artifact_files_sha256'],b['artifact_files_sha256'])
    def test_no_overwrite(self):
        self.run_prepare()
        with self.assertRaises(FileExistsError): self.run_prepare()
        self.assertTrue((self.root/'out'/'summary.json').is_file())
    def test_bad_input_fails_before_output(self):
        with gzip.open(self.reviews,'wt') as handle: handle.write('{"asin":"x"}\n')
        with self.assertRaises(ValueError): self.run_prepare()
        self.assertFalse((self.root/'out').exists())
    def test_user_cohorts_and_unrun_status(self):
        result=self.run_prepare()
        self.assertEqual(result['user_cohorts']['base_and_future_users'],1)
        self.assertEqual(result['user_cohorts']['future_only_users'],1)
        self.assertEqual(result['E0_full_status'],'not_run')
        self.assertFalse(result['dataset_identity_verified'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (output/'checks.json').write_text(json.dumps({'round':'round3','inputs':'synthetic_only',
        'tests_run':result.testsRun,'execution_status':'passed' if result.wasSuccessful() else 'failed',
        'E0_full_status':'not_run','E1_status':'not_run'},indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
