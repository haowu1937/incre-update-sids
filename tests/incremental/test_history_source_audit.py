"""Synthetic source-lineage and safe server synchronization checks; no ML deps."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('source_audit', ROOT / 'scripts/audit_history_sources.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        self.git('init', '-q', '-b', 'main')
        self.put('.gitignore', 'results/\n')
        for name in a.ROOTS:
            self.put(name, '# baseline\n')
        self.put(a.ROOTS[1], 'from gensid.models import rqvae\n')
        self.put('gensid/models/__init__.py', '')
        self.put('gensid/models/rqvae.py', 'class RQVAE: pass\n')
        self.base = self.commit()
        self.sweep = self.repo / a.SWEEP
        self.plan = dict(seeds=[2026], previous_root='results/previous',
            sources={n: a.sha((self.repo / n).read_bytes()) for n in a.ROOTS})
        self.dump('results/previous/seed2026_evaluation/summary.json', {'stage': 'reference'})
        self.plan['reference_sha256'] = [a.sha((self.repo / 'results/previous/seed2026_evaluation/summary.json').read_bytes())]
        self.dump(a.SWEEP + '/predeclared_plan.json', self.plan)
        self.dump(a.SWEEP + '/predeclared_plan.before_cuda.json', {'original': True})
        self.dump(a.SWEEP + '/selection.json', dict(selected_rho=None,
            plan_sha256=a.sha((self.sweep / 'predeclared_plan.json').read_bytes())))
        self.dump(a.SWEEP + '/cuda_startup_fix_fixture.json', {'simulated': True})
        self.dump(a.SWEEP + '/summary.json', dict(execution_status='completed', selected_rho=None,
            levels=[dict(rho=.1, seeds=[dict(seed=2026, plain_H=.1, history_H=.2, difference=.1, RQ_MSE=.01)])]))
        self.put(a.SWEEP + '/compact.txt', 'fixture\n')
        self.dump(a.SWEEP + '/training_evaluations/rho_0p1/seed2026/summary.json',
            dict(stage='historical_weight_independent_comparison_v1',
                 groups=dict(anchor_train=dict(plain=dict(positive_damage=.1),
                     history_aware=dict(positive_damage=.2),
                     paired=dict(history_minus_plain_positive_damage=.1))), rq_mse=dict(history=.01),
                 evaluation_sources=dict(multiseed=self.plan['sources'][a.ROOTS[1]])))
        self.put(a.SWEEP + '/checkpoints/rho_0p3/seed2026/attempt_000/last.pth', 'DO NOT OPEN')
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.STDOUT).decode().strip()

    def put(self, name, text):
        p = self.repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def dump(self, name, value):
        self.put(name, json.dumps(value))

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', 'commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def collect(self, **kwargs):
        self.n += 1
        args = SimpleNamespace(repo=str(self.repo), baseline_sha=self.base,
            sweep_root=a.SWEEP, output=str(self.repo / ('results/audit_' + str(self.n))), delivery_sha=None)
        vars(args).update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            report = a.collect(args)
        return report, Path(args.output)

    def test_exact_sources_nested_claims_static_imports_and_manifest(self):
        report, out = self.collect()
        self.assertEqual(report['issues'], [])
        self.assertEqual(report['changed_source_paths'], [])
        self.assertIn('gensid/models/rqvae.py', report['sources'])
        self.assertTrue(all(c['matches_workspace'] for c in report['source_claims']))
        self.assertTrue(report['endpoints'][0]['summary_matches'])
        for name, digest in json.loads((out / 'files_sha256.json').read_text()).items():
            self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
        self.assertTrue((out / 'evidence' / a.SWEEP / 'predeclared_plan.before_cuda.json').is_file())

    def test_changed_source_preserved_without_touching_old_run_or_checkpoint(self):
        name = a.ROOTS[1]
        self.put(name, '# local CUDA fix\nfrom gensid.models import rqvae\n')
        original = {p: p.read_bytes() for p in self.sweep.rglob('*') if p.is_file()}
        read_bytes = Path.read_bytes

        def no_weights(p):
            self.assertNotEqual(p.suffix, '.pth', 'collector opened a checkpoint')
            return read_bytes(p)

        with mock.patch.object(Path, 'read_bytes', no_weights):
            report, out = self.collect()
        self.assertIn(name, report['changed_source_paths'])
        self.assertTrue(any(not c['matches_workspace'] and c['matches_baseline'] for c in report['source_claims']))
        self.assertEqual((out / 'sources/workspace' / name).read_bytes(), (self.repo / name).read_bytes())
        self.assertIn('local CUDA fix', (out / 'diffs' / (name + '.diff')).read_text())
        self.assertTrue(report['checkpoint_attempt_inventory'][0]['has_checkpoint'])
        self.assertFalse(report['checkpoint_attempt_inventory'][0]['has_summary'])
        self.assertEqual(original, {p: p.read_bytes() for p in original})

    def test_missing_evidence_is_explicit_and_existing_output_is_refused(self):
        report, out = self.collect(sweep_root='results/missing')
        self.assertEqual(report['execution_status'], 'collected')
        self.assertTrue(any('missing:' in x for x in report['issues']))
        self.assertIsNone(report['historical_execution_status'])
        with self.assertRaises(FileExistsError):
            self.collect(output=str(out))

    def test_wrong_plan_and_endpoint_are_reported(self):
        self.dump(a.SWEEP + '/selection.json', dict(selected_rho=1, plan_sha256='0' * 64))
        summary = json.loads((self.sweep / 'summary.json').read_text())
        summary['levels'][0]['seeds'][0]['history_H'] = .9
        self.dump(a.SWEEP + '/summary.json', summary)
        report, _ = self.collect()
        self.assertEqual(len(report['issues']), 3)
        self.assertFalse(report['endpoints'][0]['summary_matches'])

    def test_output_cannot_be_inside_historical_run_and_paths_are_confined(self):
        with self.assertRaises(ValueError):
            self.collect(output=str(self.sweep / 'audit'))
        with self.assertRaises(ValueError):
            self.collect(sweep_root='../outside')

    def server_fixture(self):
        source = (ROOT / 'scripts/audit_history_sources.py').read_text().replace(a.BASELINE, self.base)
        self.put('scripts/audit_history_sources.py', source)
        self.put('scripts/collect_history_source_audit.sh', (ROOT / 'scripts/collect_history_source_audit.sh').read_text())
        self.put('tests/incremental/test_history_source_audit.py', "print('fixture light check passed')\n")
        delivery = self.commit()
        self.git('remote', 'add', 'research', str(self.repo))
        self.git('update-ref', 'refs/remotes/research/vibe-dev', delivery)
        fakebin = Path(self.tmp.name) / 'bin'
        fakebin.mkdir()
        conda = fakebin / 'conda'
        conda.write_text('#!/bin/sh\nexit 0\n')
        conda.chmod(0o755)
        env = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ['PATH'])
        return delivery, env

    def run_server_script(self, delivery, env):
        p = subprocess.run(['bash', str(self.repo / 'scripts/collect_history_source_audit.sh'), delivery],
            cwd=self.repo, env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        archives = list((self.repo / 'results/incremental/Beauty').glob('source_audit_*.tar.gz'))
        self.assertEqual(len(archives), 1)
        return p.stdout

    def test_server_dirty_tree_collects_before_switch_and_preserves_changes(self):
        delivery, env = self.server_fixture()
        self.put(a.ROOTS[1], '# uncommitted CUDA fix\n')
        result = self.run_server_script(delivery, env)
        self.assertIn('blocked_dirty', result)
        self.assertEqual(self.git('branch', '--show-current'), 'main')
        self.assertEqual((self.repo / a.ROOTS[1]).read_text(), '# uncommitted CUDA fix\n')

    def test_server_clean_tree_switches_to_exact_delivery_and_checks(self):
        delivery, env = self.server_fixture()
        result = self.run_server_script(delivery, env)
        self.assertIn('workspace_sync = completed', result)
        self.assertEqual(self.git('branch', '--show-current'), 'vibe-dev')
        self.assertEqual(self.git('rev-parse', 'HEAD'), delivery)
        self.assertIn('fixture light check passed', result)

    def test_server_diverged_local_branch_is_preserved(self):
        delivery, env = self.server_fixture()
        self.git('switch', '-qc', 'vibe-dev')
        self.put('local.txt', 'local work\n')
        diverged = self.commit()
        self.git('switch', '-q', 'main')
        result = self.run_server_script(delivery, env)
        self.assertIn('blocked_diverged', result)
        self.assertEqual(self.git('branch', '--show-current'), 'main')
        self.assertEqual(self.git('rev-parse', 'vibe-dev'), diverged)


if __name__ == '__main__':
    unittest.main(verbosity=2)
