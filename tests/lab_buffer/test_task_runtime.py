"""Real local Git/process roundtrips; zero ML/GPU/Beauty updates or cron changes."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/lab_buffer'))
import runner
import main_control
import task_runtime as runtime
import install_task_runtime as installer
import test_runner as fixtures

git = fixtures.git


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        fixtures.MailboxTest.setUp(self)
        git(self.author, 'checkout', 'vibe-dev')
        for name in ('main_control.py', 'main_service.py', 'task_runtime.py'):
            p = self.author/'scripts/lab_buffer'/name
            p.write_bytes((ROOT/'scripts/lab_buffer'/name).read_bytes())
        (self.author/'docs/lab_buffer').mkdir(parents=True)
        (self.author/'docs/lab_buffer/main_protocol.json').write_text(json.dumps({'protocol_id': 'sid-main-v1'}))
        (self.author/'experiment_report.md').write_text('# Existing downstream results\nMust remain.\n')
        self.scripts('first.py', 'first')
        self.commit()
        config = json.loads((self.root/'config.json').read_text())
        config.update(allowed_kinds=['smoke', 'experiment'], main_integration=True, task_runtime_version=1)
        runner.atomic_json(self.root/'config.json', config)
        runner.atomic_json(self.root/'task-runtime-policy.json', dict(protocol_version=1, max_parallel=2,
            read_roots=[str(self.base/'input-data')]))
        (self.root/'bin').mkdir()
        for name in installer.FILES:
            (self.root/'bin'/name).write_bytes((ROOT/'scripts/lab_buffer'/name).read_bytes())
        (self.root/'runner-code-sha.txt').write_text('f'*40+'\n')
        self.worker = runner.Runner(self.root)
        self.worker.fetch()
        self.c = main_control.Control(self.root)
        state, parent = self.c.read()
        state.update(mode='plan_completed')
        self.c.write(state, parent)

    def scripts(self, entry, label):
        (self.author/'case.json').write_text(json.dumps({'case': label}))
        (self.author/'check.py').write_text("import os; assert os.environ['CUDA_VISIBLE_DEVICES']==''; print('real subprocess check passed')\n")
        (self.author/entry).write_text("import json,os\nfrom pathlib import Path\np=Path('outputs/value.json');p.parent.mkdir(exist_ok=True)\np.write_text(json.dumps({'label':"+repr(label)+",'code_sha':os.environ['LAB_BUFFER_CODE_SHA']}))\n")
        (self.author/'verify.py').write_text('''import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--context');p.add_argument('--output');a=p.parse_args()
c=json.loads(Path(a.context).read_text());v=json.loads((Path(c['workspace'])/'outputs/value.json').read_text())
assert v['code_sha']==c['request']['code_sha']
raw=(json.dumps(c['contract'],ensure_ascii=False,indent=2,allow_nan=False)+'\\n').encode()
Path(a.output).write_text(json.dumps(dict(status='verified',job_id=c['request']['job_id'],
contract_sha256=c['contract_sha256'],actual_optimizer_steps=0,
report_markdown='### Transport fixture '+v['label']+'\\nSynthetic transport data; no ML result.')))
''')

    def commit(self):
        git(self.author, 'add', '.')
        git(self.author, 'commit', '-qm', 'new task code')
        self.sha = git(self.author, 'rev-parse', 'HEAD')
        git(self.author, 'push', '-q', 'origin', 'vibe-dev')

    def submission(self, job='arbitrary-research-917', entry='first.py'):
        task = dict(schema_version=1, job_id=job, worker_id='server2', kind='experiment', code_sha=self.sha,
            created_at=runner.now(), timeout_seconds=30, argv=['python', '-u', '-B', entry],
            artifacts=[runtime.RUNTIME_ARTIFACT, 'outputs/value.json'])
        scope = dict(protocol_version=1, scope_id='approved-fixture-scope', max_optimizer_steps=0,
            max_parallel=2, authorization='Explicit transport-only test fixture, not a research approval')
        contract = dict(protocol_version=1, scope_id=scope['scope_id'], job_id=job, code_sha=self.sha,
            manifest=dict(path='case.json', sha256=runtime.file_hash(self.author/'case.json')),
            preflight=[['python', '-u', '-B', 'check.py']], verifier='verify.py',
            optimizer_step_reservation=0, input_sha256={})
        return task, contract, scope

    def execute(self):
        original = subprocess.run
        def routed(args, **kw):
            if args[0] == 'nvidia-smi':
                return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
            return original(args, **kw)
        with patch.object(subprocess, 'run', side_effect=routed):
            return self.worker.tick()

    def test_two_unrelated_entries_and_shas_without_runner_changes(self):
        before = {p.name:p.read_bytes() for p in (self.root/'bin').iterdir()}
        state, parent = self.c.read()
        state['extension'] = {'scope_id':'never-installed-in-the-host', 'status':'completed'}
        self.c.write(state, parent)
        runtime.queue(self.c, *self.submission())
        result = self.execute()
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(self.c.process()['status'], 'plan_completed')
        report_head = self.c.read()[0]['receipts'][result['job_id']]['report_commit_sha']
        self.assertEqual(self.c.process()['status'], 'already_processed')
        self.assertEqual(self.execute()['status'], 'idle')
        self.worker.fetch()
        report = self.worker.read(report_head, 'experiment_report.md').decode()
        self.assertEqual(report.count('<!-- lab-task:'), 1)
        self.assertIn('Must remain.', report)
        # Second different immutable code/entry/manifest, same installed files.
        git(self.author, 'pull', '--ff-only', 'origin', 'vibe-dev')
        self.scripts('different_analysis.py', 'second')
        self.commit()
        runtime.queue(self.c, *self.submission('new-job-repair-27', 'different_analysis.py'))
        result = self.execute()
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(self.c.process()['status'], 'plan_completed')
        self.assertEqual(before, {p.name:p.read_bytes() for p in (self.root/'bin').iterdir()})
        self.assertEqual((self.root/'runner-code-sha.txt').read_text(), 'f'*40+'\n')
        self.assertEqual(len(self.c.read()[0]['receipts']), 2)

    def test_external_git_writer_keeps_exact_noncanonical_json_hashes(self):
        task, contract, scope = self.submission()
        raw = json.dumps(task, separators=(',', ':')).encode()
        scope_raw = json.dumps(scope).encode()
        contract.update(request_sha256=runner.digest(raw), scope_path='task-scopes/'+scope['scope_id']+'.json',
            scope_sha256=runner.digest(scope_raw))
        contract_raw = json.dumps(contract).encode()
        state, parent = self.c.read()
        path = 'task-contracts/'+task['job_id']+'.json'
        state.update(mode='queued', active_job_id=task['job_id'], code_sha=task['code_sha'],
            request_sha256=runner.digest(raw), task_dispatch=dict(protocol_version=1,
                contract_path=path, contract_sha256=runner.digest(contract_raw)),
            task_scope_ledger={scope['scope_id']:dict(scope_sha256=runner.digest(scope_raw),
                reservations={task['job_id']:0}, actual_steps={})})
        self.c.write(state, parent, {'jobs/'+task['job_id']+'.json':raw, path:contract_raw,
            contract['scope_path']:scope_raw})
        self.assertEqual(self.execute()['status'], 'completed')
        self.assertEqual(self.c.process()['status'], 'plan_completed')

    def test_preflight_failure_preserves_partial_evidence_and_never_repeats(self):
        (self.author/'check.py').write_text("raise SystemExit('isolated preflight failure')\n")
        self.commit()
        runtime.queue(self.c, *self.submission())
        result = self.execute()
        self.assertEqual(result['status'], 'failed')
        local = Path(result['workspace'])/runtime.RUNTIME_ARTIFACT
        self.assertFalse(json.loads(local.read_text())['command_started'])
        self.assertFalse((Path(result['workspace'])/'outputs/value.json').exists())
        self.assertIn('artifacts/'+result['job_id']+'/'+runtime.RUNTIME_ARTIFACT, result['artifacts'])
        with self.assertRaisesRegex(ValueError, 'failed/interrupted'):
            self.c.process()
        self.assertTrue(self.c.read()[0]['pause_requested'])
        self.assertEqual(self.c.process()['status'], 'already_processed')
        self.assertEqual(self.execute()['status'], 'idle')

    def test_queue_reserves_cumulative_budget_and_refuses_expansion(self):
        task, contract, scope = self.submission()
        scope['max_optimizer_steps'] = 3
        contract['optimizer_step_reservation'] = 3
        runtime.queue(self.c, task, contract, scope)
        self.execute(); self.c.process()
        task, contract, scope = self.submission('fresh-id')
        scope['max_optimizer_steps'] = 3
        contract['optimizer_step_reservation'] = 1
        with self.assertRaisesRegex(ValueError, 'Cumulative reservations'):
            runtime.queue(self.c, task, contract, scope)
        scope['max_optimizer_steps'] = 4
        with self.assertRaisesRegex(ValueError, 'immutable'):
            runtime.queue(self.c, task, contract, scope)

    def test_contract_mutation_and_pinned_manifest_mismatch_are_refused(self):
        task, contract, scope = self.submission()
        contract['manifest']['sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            runtime.queue(self.c, task, contract, scope)
        runtime.queue(self.c, *self.submission())
        self.worker.publish({'task-contracts/'+task['job_id']+'.json': b'{}\n'}, 'requests_branch')
        with self.assertRaisesRegex(ValueError, 'contract checksum'):
            self.execute()

    def test_gpu_busy_and_pause_do_not_launch_or_claim(self):
        task, contract, scope = self.submission()
        runtime.queue(self.c, task, contract, scope)
        original = subprocess.run
        def busy(args, **kw):
            return subprocess.CompletedProcess(args, 0, stdout='123\n', stderr='') if args[0]=='nvidia-smi' else original(args, **kw)
        with patch.object(subprocess, 'run', side_effect=busy):
            self.assertEqual(self.worker.tick()['status'], 'gpu_busy')
        self.assertFalse(self.worker.state_path(task['job_id']).exists())
        self.c.pause()
        self.assertEqual(self.execute()['status'], 'control_wait')
        self.assertFalse(self.worker.state_path(task['job_id']).exists())

    def test_input_change_stops_before_command(self):
        p = self.base/'input-data/feature.json';p.parent.mkdir();p.write_text('original')
        task, contract, scope = self.submission()
        contract['input_sha256'] = {str(p):runtime.file_hash(p)}
        runtime.queue(self.c, task, contract, scope)
        p.write_text('changed')
        result = self.execute()
        self.assertEqual(result['status'], 'failed')
        value = json.loads((Path(result['workspace'])/runtime.RUNTIME_ARTIFACT).read_text())
        self.assertFalse(value['command_started'])
        self.assertIn('checksum changed', value['error'])

    def test_shell_entry_unknown_protocol_and_resource_expansion_refused(self):
        for change in ('shell', 'protocol', 'parallel'):
            task, contract, scope = self.submission()
            if change == 'shell':task['argv'] = ['python', '-c', 'print(1)']
            if change == 'protocol':contract['protocol_version'] = 2
            if change == 'parallel':scope['max_parallel'] = 4
            with self.assertRaises(ValueError):runtime.queue(self.c, task, contract, scope)

    def test_active_lock_does_not_fetch_or_alter_live_work(self):
        before = (self.root/'config.json').read_bytes()
        with (self.root/'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(runner.Runner, 'fetch', side_effect=AssertionError('must not race active transport')):
                value = installer.activate(self.root, self.sha)
        self.assertEqual(value['status'], 'waiting_for_current_task')
        self.assertEqual((self.root/'config.json').read_bytes(), before)

    def test_uncertain_local_process_is_not_overwritten(self):
        self.worker.config.pop('task_runtime_version', None)
        config = dict(self.worker.config)
        runner.atomic_json(self.root/'config.json', config)
        runner.atomic_json(self.root/'state/uncertain.json', dict(result=dict(status='running'), pending_upload=False))
        with self.assertRaisesRegex(ValueError, 'Uncertain child'):
            installer.activate(self.root, self.sha)

    def prepare_install(self):
        config = json.loads((self.root/'config.json').read_text())
        config.pop('task_runtime_version')
        runner.atomic_json(self.root/'config.json', config)
        (self.root/'runner-code-sha.txt').write_text(self.sha+'\n')
        runner.atomic_json(self.root/'main-deployment.json', dict(project=str(self.base/'main-project')))
        self.worker.publish({'runtime-upgrades/task-runtime-v1.json': runtime.encode(dict(
            code_sha=self.sha, from_code_sha=self.sha, protocol_version=1, new_optimizer_steps=0))}, 'requests_branch')

    def test_idle_install_preserves_pause_records_control_and_is_idempotent(self):
        self.prepare_install()
        pause = b'{"reason":"user pause remains"}\n'
        (self.root/'PAUSED.json').write_bytes(pause)
        before = self.c.read()[0]
        with patch.object(installer, 'PREVIOUS', self.sha):
            value = installer.activate(self.root, self.sha)
            self.assertEqual(value['status'], 'installed')
            self.assertEqual(installer.activate(self.root, self.sha)['status'], 'already_installed')
        self.assertEqual((self.root/'PAUSED.json').read_bytes(), pause)
        self.assertEqual(self.c.read()[0], before)
        self.assertEqual(value['tasks_queued'], 0)
        self.assertFalse(value['cron_changed'])
        self.assertTrue((Path(value['backup'])/'config.json').is_file())

    def test_partial_install_keeps_backup_and_durable_pause(self):
        self.prepare_install()
        copyfile = installer.shutil.copyfile
        def fail_replacement(source, target, **kwargs):
            if Path(target).parent == self.root/'bin':
                raise OSError('injected replacement failure')
            return copyfile(source, target, **kwargs)
        with patch.object(installer, 'PREVIOUS', self.sha), patch.object(installer.shutil, 'copyfile', side_effect=fail_replacement):
            with self.assertRaisesRegex(OSError, 'replacement failure'):
                installer.activate(self.root, self.sha)
        self.assertIn('incomplete', json.loads((self.root/'PAUSED.json').read_text())['reason'])
        self.assertTrue((self.root/'runtime-upgrade-backups'/self.sha/'config.json').is_file())

    def test_interrupted_record_with_live_pid_blocks_installation(self):
        self.prepare_install()
        runner.atomic_json(self.root/'state/live.json', dict(result=dict(status='interrupted',
            pid=os.getpid(), exit_code=None), pending_upload=False))
        with self.assertRaisesRegex(ValueError, 'may still be live'):
            installer.activate(self.root, self.sha)


if __name__ == '__main__':
    unittest.main()
