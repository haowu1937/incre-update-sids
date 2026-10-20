"""Real local Git/subprocess tests with simulated acceptance data; no ML or GPU."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/lab_buffer'))
import main_control as control
import runner as current_runner
import attach_main
import accept_main
import test_runner as fixtures
git = fixtures.git


class MainTest(unittest.TestCase):
    task = fixtures.MailboxTest.task
    submit = fixtures.MailboxTest.submit

    def setUp(self):
        fixtures.MailboxTest.setUp(self)
        git(self.author, 'checkout', 'vibe-dev')
        for name in ('main_control.py', 'main_service.py'):
            p = self.author/'scripts/lab_buffer'/name
            p.write_bytes((REPO/'scripts/lab_buffer'/name).read_bytes())
        (self.author/'docs/lab_buffer').mkdir(parents=True)
        (self.author/'docs/lab_buffer/main_protocol.json').write_text(json.dumps({'protocol_id': 'sid-main-v1'}))
        (self.author/'experiment_report.md').write_text('# Report\nExisting results preserved.\n')
        # An actual child and local Git transport, but intentionally fake ML evidence.
        (self.author/'scripts/lab_buffer/accept_main.py').write_text('''import json,os,subprocess
from pathlib import Path
p=Path('outputs/main-acceptance.json');p.parent.mkdir()
metric={'recall@10':.01,'ndcg@10':.005}
audit=dict(stage='sid_main_existing_future_acceptance',execution_status='completed',
 job_id=os.environ['LAB_BUFFER_JOB_ID'],code_sha=os.environ['LAB_BUFFER_CODE_SHA'],
 training_started=False,original_inputs_unchanged=True,checks={'simulated_fixture':True},
 future_dir='/fixture/future',summary_sha256='a'*64,
 compact={'future':{'frozen':{'full_denominator':metric},'finetuned':{'full_denominator':metric}}})
p.write_text(json.dumps(audit)+'\\n')
''')
        git(self.author, 'add', '.')
        git(self.author, 'commit', '-qm', 'main fixture')
        self.sha = git(self.author, 'rev-parse', 'HEAD')
        git(self.author, 'push', '-q', 'origin', 'vibe-dev')
        config = json.loads((self.root/'config.json').read_text())
        config.update(allowed_kinds=['smoke', 'experiment'], main_integration=True)
        current_runner.atomic_json(self.root/'config.json', config)
        self.worker = current_runner.Runner(self.root)
        self.worker.fetch()
        self.dep = dict(runner_root=str(self.root), code_sha=self.sha)
        current_runner.atomic_json(self.root/'main-deployment.json', self.dep)
        self.c = control.Control(self.root)

    def queue(self):
        return self.c.queue(self.dep)

    def execute(self):
        original = subprocess.run
        def routed(args, **kw):
            if args[0] == 'nvidia-smi':
                return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
            return original(args, **kw)
        with patch.object(subprocess, 'run', side_effect=routed):
            return self.worker.tick()

    def test_full_round_and_deduplicated_report(self):
        self.assertEqual(self.queue()['status'], 'queued')
        result = self.execute()
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(result['training_started'])
        self.assertEqual(self.c.process()['status'], 'plan_completed')
        self.assertEqual(self.c.read()[0]['receipts'][result['job_id']]['status'], 'verified')
        before = self.worker.git('rev-parse', self.worker.ref('code_branch')).stdout
        self.assertEqual(self.c.process()['status'], 'plan_completed')
        self.assertEqual(self.worker.git('rev-parse', self.worker.ref('code_branch')).stdout, before)
        self.assertEqual(self.execute()['status'], 'idle')

    def test_raw_task_mutation_and_exact_allowlist(self):
        self.queue()
        state, _ = self.c.read()
        raw = self.worker.read(self.worker.ref('requests_branch'), 'jobs/history-future-accept-001.json')
        task = json.loads(raw)
        self.assertTrue(control.authorize_task(self.root, task, raw, state))
        task['argv'] = ['python', '-c', 'print(123)']
        raw = control.encoded(task)
        state['request_sha256'] = current_runner.digest(raw)
        with self.assertRaisesRegex(ValueError, 'outside'):
            control.authorize_task(self.root, task, raw, state)

    def test_concurrent_claim_and_stale_branch_are_rejected(self):
        self.queue()
        _, stale = self.c.read()
        token = self.c.claim('manual')
        self.assertIsNotNone(token)
        self.assertIsNone(control.Control(self.root).claim('scheduled'))
        with self.assertRaisesRegex(RuntimeError, 'Concurrent'):
            self.c.write(control.initial_state(), stale)
        self.c.release(token)

    def test_expired_lease_pauses_instead_of_relaunch(self):
        self.queue()
        state, parent = self.c.read()
        state['lease'] = dict(token='old', owner='lost', expires_at_unix=time.time()-1)
        self.c.write(state, parent)
        self.assertIsNone(self.c.claim('next'))
        state, _ = self.c.read()
        self.assertTrue(state['pause_requested'])
        self.assertEqual(state['mode'], 'needs_reconciliation')
        self.assertEqual(self.execute()['status'], 'control_wait')

    def test_pause_prevents_claiming_a_new_server_job(self):
        self.queue()
        self.c.pause()
        self.assertEqual(self.execute()['status'], 'control_wait')
        self.assertFalse(self.worker.state_path('history-future-accept-001').exists())

    def test_tampered_result_artifact_stops_acceptance(self):
        self.queue()
        result = self.execute()
        self.worker.publish({result['artifacts'][0]: b'{}\n'})
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.c.process()
        self.assertEqual(self.c.read()[0]['mode'], 'queued')

    def test_failure_evidence_and_budgeted_diagnosed_repair(self):
        self.queue()
        state, _ = self.c.read()
        raw = self.worker.read(self.worker.ref('requests_branch'), 'jobs/history-future-accept-001.json')
        task = json.loads(raw)
        record = self.worker.new_state(task, raw)
        record['result'].update(status='failed', exit_code=1, reason='simulated fixture failure', training_started=False)
        self.worker.save(record)
        self.worker.deliver(record)
        self.assertEqual(self.c.process()['status'], 'needs_diagnosis')
        self.assertEqual(self.c.read()[0]['receipts'][task['job_id']]['status'], 'needs_attention')
        with self.assertRaisesRegex(ValueError, 'different'):
            self.c.repair(self.sha, 'Concrete diagnosed cause: fixture has a known error')
        git(self.author, 'checkout', 'vibe-dev')
        (self.author/'repair.txt').write_text('Evidence-backed fixture-only repair')
        git(self.author, 'add', '.')
        git(self.author, 'commit', '-qm', 'repair')
        new_sha = git(self.author, 'rev-parse', 'HEAD')
        git(self.author, 'push', '-q', 'origin', 'vibe-dev')
        self.assertEqual(self.c.repair(new_sha, 'Concrete diagnosed cause: fixture has a known error')['job_id'], 'history-future-accept-002')
        self.assertEqual(self.execute()['status'], 'completed')
        self.assertEqual(self.c.process()['status'], 'plan_completed')
        with self.assertRaises(ValueError):
            self.c.repair(new_sha, 'Cannot run an additional repair experiment')

    def test_cron_merge_preserves_other_entries_and_reuses_marker(self):
        before = 'MAILTO=owner\n3 * * * * echo keep\n* * * * * old # git-lab-buffer-server2\n* * * * * collector # git-lab-retuned-collector\n'
        after = attach_main.merge_cron(before, '/fixed/python', Path('/fixed/root'))
        self.assertIn('MAILTO=owner\n3 * * * * echo keep\n', after)
        self.assertEqual(after.count('# git-lab-buffer-server2'), 1)
        self.assertNotIn('# git-lab-retuned-collector', after)
        self.assertEqual(attach_main.merge_cron(after, '/fixed/python', Path('/fixed/root')), after)

    def test_future_discovery_requires_exact_unique_identity(self):
        p = self.base/'project'
        root = p/'results/incremental/Beauty'
        root.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'found 0'):
            attach_main.discover(p)
        example = dict(stage='retuned_History_aware_Future_evaluation', execution_status='completed',
                       retuned_dir=str(p/attach_main.RETUNED), future={'finetuned': {'model_sha256':
                       'bca77fef0b0bb3a107f40d8868f0b805885a21152289a29e97baa4ad6b26177f'}})
        for name in ('one', 'two'):
            run = root/('llama_historical_future_'+name)/'evaluation'
            run.mkdir(parents=True)
            (run/'summary.json').write_text(json.dumps(example))
        with self.assertRaisesRegex(ValueError, 'found 2'):
            attach_main.discover(p)

    def test_install_reuses_existing_root_and_preserves_project_and_records(self):
        (self.root/'main-deployment.json').unlink()  # fixture's opt-in placeholder; legacy install has none
        project = self.base/'main-project'
        project.mkdir()
        (project/'sentinel.txt').write_text('running main experiment; do not change')
        future = project/'results/incremental/Beauty/llama_historical_future_fixture/evaluation'
        future.mkdir(parents=True)
        expected = json.loads((REPO/'docs/lab_buffer/history_future_expected.json').read_text())
        (self.author/'docs/lab_buffer/history_future_expected.json').write_text(json.dumps(expected))
        (self.author/'scripts/lab_buffer/attach_main.py').write_bytes((REPO/'scripts/lab_buffer/attach_main.py').read_bytes())
        git(self.author, 'add', '.')
        git(self.author, 'commit', '-qm', 'attach fixture')
        git(self.author, 'push', '-q', 'origin', 'vibe-dev')
        for name in ('summary.json', 'compact.json', 'config.json', 'selection_locked.json'):
            value = copy.deepcopy(expected) if name in ('summary.json', 'compact.json') else {}
            if name == 'summary.json':
                value.update(retuned_dir=str(project/attach_main.RETUNED), artifact_files_sha256={})
            (future/name).write_text(json.dumps(value))
        e1 = project/attach_main.E1
        retuned = project/attach_main.RETUNED
        e1.mkdir(parents=True)
        retuned.mkdir(parents=True)
        (retuned/'summary.json').write_text('{}')
        paths = {name: str(project/name) for name in ('protocol_dir','update_rqvae_dir','tiger_dir','sid_dir')}
        for value in paths.values():
            Path(value).mkdir()
            (Path(value)/'summary.json').write_text('{}')
        (e1/'summary.json').write_text(json.dumps(paths))
        base = project/'base_rqvae'
        base.mkdir()
        features = project/'features'
        features.mkdir()
        (features/'summary.json').write_text('{}')
        (Path(paths['update_rqvae_dir'])/'summary.json').write_text(json.dumps(dict(features_dir=str(features), base_rqvae_dir=str(base))))
        (base/'summary.json').write_text('{}')
        (Path(paths['tiger_dir'])/'summary.json').write_text('{}')
        for arm in ('plain', 'historical'):
            (retuned/arm).mkdir()
            (retuned/arm/'summary.json').write_text('{}')
        config = json.loads((self.root/'config.json').read_text())
        config.update(allowed_kinds=['smoke'], main_integration=False)
        current_runner.atomic_json(self.root/'config.json', config)
        (self.root/'bin').mkdir(exist_ok=True)
        (self.root/'bin/runner.py').write_text('# installed original version')
        before = {str(p): p.read_bytes() for p in project.rglob('*') if p.is_file()}
        cron = 'MAILTO=owner\n2 * * * * echo keep\n* * * * * old # git-lab-buffer-server2\n'
        installed = []
        original = subprocess.run
        def routed(args, **kw):
            if args[:2] == ['crontab', '-']:
                installed.append(kw['input'])
                return subprocess.CompletedProcess(args, 0)
            return original(args, **kw)
        with patch.object(attach_main, 'PROJECT', project), patch.object(attach_main, 'PYTHON', sys.executable), patch.object(attach_main, 'cron_read', return_value=cron), patch.object(subprocess, 'run', side_effect=routed):
            attach_main.attach(self.root, self.author/'scripts/lab_buffer')
        self.assertEqual(before, {str(p): p.read_bytes() for p in project.rglob('*') if p.is_file()})
        self.assertIn('2 * * * * echo keep', installed[0])
        self.assertEqual(installed[0].count('# git-lab-buffer-server2'), 1)
        self.assertEqual(self.c.read()[0]['mode'], 'queued')
        self.assertEqual(json.loads((self.root/'config.json').read_text())['allowed_kinds'], ['smoke','experiment'])
        self.assertTrue(list((self.root/'main-attach-backups').glob('*/runner.py')))

    def test_main_deadline_reports_without_killing_child_and_returns_artifact(self):
        git(self.author, 'checkout', 'vibe-dev')
        script = self.author/'scripts/lab_buffer/accept_main.py'
        script.write_text('import time; time.sleep(1.3)\n'+script.read_text())
        git(self.author, 'add', '.')
        git(self.author, 'commit', '-qm', 'slow fixture')
        git(self.author, 'push', '-q', 'origin', 'vibe-dev')
        self.dep['code_sha'] = git(self.author, 'rev-parse', 'HEAD')
        current_runner.atomic_json(self.root/'main-deployment.json', self.dep)
        original_job = control.job
        def short_job(*args):
            task = original_job(*args)
            task['timeout_seconds'] = 1
            return task
        with patch.object(control, 'job', side_effect=short_job), patch.object(current_runner.os, 'killpg') as kill:
            self.queue()
            result = self.execute()
        self.assertEqual(result['status'], 'timed_out')
        self.assertTrue(result['deadline_exceeded'])
        self.assertEqual(len(result['artifacts']), 1)
        kill.assert_not_called()
        self.assertTrue((self.root/'PAUSED.json').exists())

    def test_expected_observation_roundoff_and_path_boundaries(self):
        accept_main.compare({'x': .01+1e-14}, {'x': .01})
        with self.assertRaises(ValueError):
            accept_main.compare({'x': .011}, {'x': .01})
        with self.assertRaises(ValueError):
            accept_main.contained_file(self.base, '../escape.json')
        outside = self.base/'outside.json'
        outside.write_text('{}')
        directory = self.base/'artifacts'
        directory.mkdir()
        (directory/'link.json').symlink_to(outside)
        with self.assertRaises(ValueError):
            accept_main.contained_file(directory, 'link.json')


if __name__ == '__main__':
    unittest.main()
