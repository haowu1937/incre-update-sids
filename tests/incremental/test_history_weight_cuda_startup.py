"""CUDA launcher and archived-fix checks using fake torch; no GPU/ML imports."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cuda_startup_check', ROOT / 'scripts/check_history_cuda_startup.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

FAKE_TORCH = '''import json, os
from pathlib import Path
__version__ = 'fake'
class version: cuda = 'fake'
def event(name):
    with open(os.environ['FAKE_CUDA_EVENTS'], 'a') as handle: handle.write(name + '\\n')
class device:
    def __init__(self, value): self.value=value; self.type=value.split(':')[0]
    def __str__(self): return self.value
class CUDA:
    initialized=False
    def set_device(self, value): event('set_device:' + str(value))
    def init(self): event('init'); self.initialized=True
    def is_initialized(self): return self.initialized
    def manual_seed_all(self, value): event('cuda_seed:' + str(value))
    def reset_peak_memory_stats(self, value):
        if not self.initialized: raise RuntimeError('CUDA not initialized')
        event('reset')
    def synchronize(self, value): event('sync')
    def get_device_name(self, value): return 'fake GPU'
cuda=CUDA()
def manual_seed(value): event('seed:' + str(value))
class Tensor:
    def item(self): return 1.
def ones(count, device):
    if os.environ.get('FAKE_CUDA_FAIL'): raise RuntimeError('probe failure fixture')
    event('ones:' + str(device)); return Tensor()
'''


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fake = self.root / 'fake'
        self.fake.mkdir()
        (self.fake / 'torch.py').write_text(FAKE_TORCH)
        self.events = self.root / 'events.txt'
        self.env = mock.patch.dict(os.environ, PYTHONPATH=str(self.fake), FAKE_CUDA_EVENTS=str(self.events))
        self.env.start()
        c.controller.ABORT.clear()

    def tearDown(self):
        self.env.stop()
        c.controller.ABORT.clear()
        self.tmp.cleanup()

    def test_runtime_sources_equal_the_successful_startup_fix(self):
        self.assertEqual(c.digest(ROOT / 'run_history_weight_sweep.py'), c.FIXED_CONTROLLER)
        self.assertEqual(c.digest(ROOT / 'history_weight_update.py'), c.UNCHANGED_ENGINE)

    def test_cuda_child_initializes_before_seed_reset_and_allocation(self):
        out = self.root / 'probe'
        out.mkdir()
        result = c.probe(out, 'cuda:1', 2027)
        self.assertTrue(result['initialized_at_engine_entry'])
        self.assertEqual(result['device'], 'cuda:1')
        self.assertEqual(result['seed'], 2027)
        self.assertEqual(self.events.read_text().splitlines(), [
            'set_device:cuda:1', 'init', 'seed:2027', 'cuda_seed:2027', 'reset', 'ones:cuda:1', 'sync'])
        self.assertFalse(result['training_started'])

    def test_cpu_child_uses_direct_launch_without_cuda_initialization(self):
        out = self.root / 'cpu'
        out.mkdir()
        result = c.probe(out, 'cpu', 2026)
        self.assertFalse(result['initialized_at_engine_entry'])
        self.assertEqual(self.events.read_text().splitlines(), ['seed:2026', 'ones:cpu'])

    def test_failed_child_keeps_traceback_and_restores_engine_pointer(self):
        out = self.root / 'failure'
        out.mkdir()
        original = c.controller.ENGINE
        with mock.patch.dict(os.environ, FAKE_CUDA_FAIL='1'):
            with self.assertRaisesRegex(RuntimeError, 'Job exited'):
                c.probe(out, 'cuda:0', 2026)
        self.assertEqual(c.controller.ENGINE, original)
        self.assertFalse(c.controller.ACTIVE)
        self.assertIn('probe failure fixture', (out / 'child.log').read_text())

    def evidence_fixture(self):
        archive = self.root / 'archive'
        archive.mkdir()
        backup = self.root / 'cuda_startup_fix_backups/stamp'
        backup.mkdir(parents=True)
        before = dict(seeds=[2026, 2027, 2028], steps=100,
                      sources={'run_history_weight_sweep.py': c.OLD_CONTROLLER})
        after = copy.deepcopy(before)
        after['sources']['run_history_weight_sweep.py'] = c.FIXED_CONTROLLER

        def save(name, data):
            p = archive / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return p

        original = backup / 'predeclared_plan.json'
        original.write_text(json.dumps(before))
        plan = save('evidence/results/sweep/predeclared_plan.json', json.dumps(after).encode())
        record = dict(backup_dir=str(backup), controller_before_sha256=c.OLD_CONTROLLER,
            controller_after_sha256=c.FIXED_CONTROLLER, unchanged_engine_sha256=c.UNCHANGED_ENGINE,
            training_objective_changed=False, seed_or_step_budget_changed=False,
            existing_preflight_and_checkpoints_preserved=True,
            plan_before_sha256=c.digest(original), plan_after_sha256=c.digest(plan))
        record_path = save('evidence/results/sweep/cuda_startup_fix_fixture.json', json.dumps(record).encode())
        for name in ('run_history_weight_sweep.py', 'history_weight_update.py'):
            save('sources/workspace/' + name, (ROOT / name).read_bytes())
        report = dict(issues=[], source_claims=[], observed_head='2fb3426615ec0033dd6b65734593075f7e8380e0',
            historical_execution_status='completed', selected_rho=None,
            evidence_files=[dict(path='results/sweep/cuda_startup_fix_fixture.json')],
            endpoints=[dict(summary_matches=True, difference=.01 if i else -.01) for i in range(15)])
        reportfile = save('report.json', json.dumps(report).encode())
        manifest = {p.relative_to(archive).as_posix(): c.digest(p) for p in archive.rglob('*') if p.is_file()}
        save('files_sha256.json', json.dumps(manifest).encode())
        return reportfile, original, record_path

    def test_fix_chain_accepts_only_controller_hash_change(self):
        reportfile, original, _ = self.evidence_fixture()
        result = c.check_evidence(reportfile, self.root)
        self.assertEqual(result['status'], 'metadata_consistent')
        self.assertEqual(result['history_worse_than_plain'], 14)
        before = c.read(original)
        before['steps'] = 101
        original.write_text(json.dumps(before))
        with self.assertRaisesRegex(ValueError, 'record differs'):
            c.check_evidence(reportfile, self.root)

    def test_modified_archived_fix_record_fails_checksum(self):
        reportfile, _, record_path = self.evidence_fixture()
        record_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            c.check_evidence(reportfile, self.root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
