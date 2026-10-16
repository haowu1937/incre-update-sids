"""Safe handling of the one known untracked verifier; no ML imports or GPU."""
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/run_llama_base_pilot.sh'
VERIFIER = 'verify_incremental_embeddings.py'


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git('init', '-q', '-b', 'main')
        (self.repo / '.gitignore').write_text('results/\n')
        self.base = self.commit()
        (self.repo / VERIFIER).write_text('print("canonical verifier")\n')
        self.target = self.commit()
        self.git('update-ref', 'refs/remotes/research/vibe-dev', self.target)
        self.git('switch', '-qc', 'run/llama7b-embeddings', self.base)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.STDOUT).decode().strip()

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', 'commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def run_script(self):
        return subprocess.run(['bash', str(SCRIPT), self.target, '--sync-only'],
                              cwd=self.repo, text=True, capture_output=True)

    def test_identical_untracked_verifier_is_backed_up_and_canonical_restored(self):
        text = 'print("canonical verifier")\n'
        (self.repo / VERIFIER).write_text(text)
        self.git('branch', 'vibe-dev', self.base)
        p = self.run_script()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.git('rev-parse', 'HEAD'), self.target)
        self.assertEqual(self.git('branch', '--show-current'), 'vibe-dev')
        backups = list(self.repo.glob('results/incremental/Beauty/local_verifier_backup_*/' + VERIFIER))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), text)
        self.assertEqual((self.repo / VERIFIER).read_text(), text)
        self.assertEqual(self.git('rev-parse', 'run/llama7b-embeddings'), self.base)
        self.assertEqual(self.git('status', '--porcelain'), '')

    def test_different_verifier_is_not_moved_or_overwritten(self):
        text = 'print("local changes")\n'
        (self.repo / VERIFIER).write_text(text)
        p = self.run_script()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('Verifier differs', p.stdout)
        self.assertEqual((self.repo / VERIFIER).read_text(), text)
        self.assertEqual(self.git('rev-parse', 'HEAD'), self.base)

    def test_other_changes_stop_before_any_file_move(self):
        (self.repo / VERIFIER).write_text('print("canonical verifier")\n')
        (self.repo / 'other.py').write_text('retain\n')
        p = self.run_script()
        self.assertNotEqual(p.returncode, 0)
        self.assertTrue((self.repo / VERIFIER).is_file())
        self.assertEqual((self.repo / 'other.py').read_text(), 'retain\n')
        self.assertFalse((self.repo / 'results').exists())

    def test_diverged_vibe_dev_stops_before_file_move(self):
        self.git('switch', '-qc', 'vibe-dev')
        (self.repo / 'local.txt').write_text('retain\n')
        diverged = self.commit()
        self.git('switch', '-q', 'run/llama7b-embeddings')
        (self.repo / VERIFIER).write_text('print("canonical verifier")\n')
        p = self.run_script()
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(self.git('rev-parse', 'vibe-dev'), diverged)
        self.assertEqual(self.git('rev-parse', 'HEAD'), self.base)
        self.assertTrue((self.repo / VERIFIER).is_file())

    def test_clean_tree_switches_without_backup(self):
        p = self.run_script()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.git('rev-parse', 'HEAD'), self.target)
        self.assertTrue((self.repo / VERIFIER).is_file())
        self.assertFalse((self.repo / 'results').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
