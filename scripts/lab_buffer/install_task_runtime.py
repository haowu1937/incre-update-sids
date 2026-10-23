"""One-time stable-boundary installation, deferred until the existing task is idle.

Reuses root/interpreter/cron and preserves legacy experiment adapters. Future task
code/manifests travel through Git; this installer never queues a research task.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from runner import Runner, FINAL, atomic_json, digest, now

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = '1bc0b437973cea3b6e9293611eac2e39f7771983'
FILES = ('runner.py', 'main_control.py', 'main_service.py', 'task_runtime.py')


def activate(root, code):
    """Return waiting without Git access when the running task owns the lock."""
    root = Path(root).resolve()
    with (root/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='waiting_for_current_task', installed=False)
        r = Runner(root)
        if r.config.get('task_runtime_version') == 1:
            installed = (root/'runner-code-sha.txt').read_text().strip()
            if installed != code:
                raise ValueError('A different runtime is already installed; no overwrite')
            for name in FILES:
                if (root/'bin'/name).read_bytes() != (ROOT/'scripts/lab_buffer'/name).read_bytes():
                    raise ValueError('Installed runtime differs from its code SHA')
            return dict(status='already_installed', installed=True, code_sha=code)
        for path in (root/'state').glob('*.json'):
            state = json.loads(path.read_text())
            if state['result']['status'] not in FINAL:
                raise ValueError('Uncertain child lifetime: preserve state and diagnose, no installation')
            result = state['result']
            if result.get('pid') and result.get('exit_code') is None:
                try:
                    os.kill(result['pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError('A recorded child PID may still be live; no installation')
            if state.get('pending_upload'):
                return dict(status='waiting_for_result_delivery', installed=False)
        r.fetch()
        state = json.loads(r.read(r.ref('requests_branch'), 'control/main.json'))
        if state.get('lease'):
            return dict(status='waiting_for_controller_receipt', installed=False)
        if state['mode'] not in ('plan_completed', 'needs_diagnosis', 'needs_reconciliation'):
            return dict(status='waiting_for_current_plan_receipt', installed=False)
        authorization_raw = r.read(r.ref('requests_branch'), 'runtime-upgrades/task-runtime-v1.json')
        authorization = json.loads(authorization_raw or b'{}')
        if (authorization.get('code_sha') != code or authorization.get('from_code_sha') != PREVIOUS
                or authorization.get('protocol_version') != 1
                or authorization.get('new_optimizer_steps') != 0):
            raise ValueError('Missing exact authorized core-boundary upgrade')
        if (root/'runner-code-sha.txt').read_text().strip() != PREVIOUS:
            raise ValueError('Unexpected installed version; no overwrite')
        before = {}
        for name in FILES[:-1]:
            expected = r.read(PREVIOUS, 'scripts/lab_buffer/'+name)
            actual = (root/'bin'/name).read_bytes()
            before[name] = digest(actual)
            if expected is None or expected != actual:
                raise ValueError('Installed helper changed: '+name)
        backup = root/'runtime-upgrade-backups'/code
        backup.mkdir(parents=True, exist_ok=False)
        for name in FILES[:-1]:
            shutil.copy2(root/'bin'/name, backup/name)
        for name in ('config.json', 'runner-code-sha.txt'):
            shutil.copy2(root/name, backup/name)
        # A crash during this small group of replacements leaves a durable pause.
        # Successful installation restores the exact pre-existing pause, if any.
        pause = root/'PAUSED.json'
        old_pause = pause.read_bytes() if pause.exists() else None
        if old_pause is not None:
            (backup/'PAUSED.json').write_bytes(old_pause)
        atomic_json(pause, dict(reason='Core runtime installation incomplete; inspect backup', backup=str(backup), time=now()))
        installation_pause = pause.read_bytes()
        try:
            for name in FILES:
                target = root/'bin'/name
                temporary = target.with_suffix('.runtime-new')
                shutil.copyfile(ROOT/'scripts/lab_buffer'/name, temporary)
                os.replace(temporary, target)
            deployment = json.loads((root/'main-deployment.json').read_text())
            project = deployment.get('project', '/home/hao/frec/graph-informed-sids-web')
            # The local policy describes host capabilities, never experiment IDs,
            # recipes or optimizer budgets. Scope authorization stays in Git.
            atomic_json(root/'task-runtime-policy.json', dict(protocol_version=1, max_parallel=2,
                read_roots=[project, str(root/'main-deployment.json'), str(root/'convergence-deployment.json')]))
            config = dict(r.config, task_runtime_version=1)
            atomic_json(root/'config.json', config)
            (root/'runner-code-sha.txt').write_text(code+'\n')
            if pause.read_bytes() != installation_pause:
                raise ValueError('Pause changed during installation; preserve it and reconcile')
            if old_pause is None:
                pause.unlink()
            else:
                temporary = pause.with_suffix('.restore')
                temporary.write_bytes(old_pause)
                os.replace(temporary, pause)
        except BaseException:
            # Do not clear this pause or silently roll back a partial deployment.
            raise
        evidence = dict(status='installed', protocol_version=1, code_sha=code, installed_at=now(),
            installed_before=before, backup=str(backup), cron_changed=False, tasks_queued=0,
            optimizer_steps=0, main_workspace_written=False, processes_terminated=False,
            control_state_modified=False, existing_pause_preserved=old_pause is not None,
            capabilities=dict(task_protocol_versions=[1], max_parallel=2,
                read_roots=[project, str(root/'main-deployment.json'), str(root/'convergence-deployment.json')]))
        atomic_json(root/'task-runtime-install.json', evidence)
        r.publish({'workers/server2-task-runtime.json': (json.dumps(evidence, indent=2)+'\n').encode()})
        return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--expected-code-sha', required=True)
    p.add_argument('--wait-seconds', type=int, default=172800)
    a = p.parse_args()
    code = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if code != a.expected_code_sha:
        raise ValueError('Installer checkout differs from the delivered full SHA')
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    root = Path(a.root).resolve()
    with (root/'task-runtime-install.lock').open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='installer_already_waiting')), flush=True)
            return
        # Standard-library transport/protocol tests only; no GPU or Beauty work.
        checks = root/('task-runtime-checks-'+code+'.log')
        with checks.open('ab') as handle:
            subprocess.run([str(root_python(root)), '-B', '-m', 'unittest', 'discover', '-s',
                'tests/lab_buffer', '-p', 'test_task_runtime.py'], cwd=ROOT, stdout=handle,
                stderr=subprocess.STDOUT, check=True, timeout=300)
        deadline = time.monotonic()+a.wait_seconds
        previous = None
        while True:
            value = activate(root, code)
            if value['status'] != previous:
                print(json.dumps(value), flush=True)
                previous = value['status']
                atomic_json(root/'task-runtime-install-status.json', value)
            if value['installed']:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('No idle installation window; current task preserved, no retry scheduled')
            time.sleep(5)


def root_python(root):
    return json.loads((root/'config.json').read_text())['python']


if __name__ == '__main__':
    main()
