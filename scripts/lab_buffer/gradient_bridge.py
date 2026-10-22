"""Explicit 2026-10-22 scope extension; migrate the legacy gate, then bounded diagnostics.

Job 002 is an UNUSED legacy transport slot, not a repair or a Future rerun.
It upgrades only the installed mailbox helpers under its parent Runner's lock.
The previous acceptance receipts are retained. New scientific work has new IDs.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runner import Runner, atomic_json, digest, now
from main_control import encoded, job as legacy_job

SCOPE = 'history-gradient-v1'
MIGRATION = 'history-future-accept-002'
CONTROLLED = 'history-gradient-controlled-001'
PLAN = 'docs/lab_buffer/history_gradient_plan.json'


def migration_requested(deployment_path):
    d = json.loads(Path(deployment_path).read_text())
    runner = Runner(d['runner_root'])
    runner.fetch(('requests_branch',))
    raw = runner.read(runner.ref('requests_branch'), 'control/main.json')
    return bool(raw and json.loads(raw).get('extension', {}).get('scope_id') == SCOPE)


def task(deployment, code_sha, created_at=None):
    value = dict(schema_version=1, job_id=CONTROLLED, worker_id='server2', kind='experiment',
        code_sha=code_sha, created_at=created_at or now(), timeout_seconds=1800,
        argv=['python', '-u', '-B', 'diagnose_history_gradients.py', '--deployment',
              str(Path(deployment['runner_root'])/'main-deployment.json'), '--output', 'outputs/controlled'],
        artifacts=['outputs/controlled/compact.json'])
    return value


def authorize(root, request, raw, state):
    ex = state.get('extension', {})
    if ex.get('scope_id') != SCOPE or state['pause_requested'] or state['mode'] != 'queued':
        return False
    if state['active_job_id'] != request['job_id'] or digest(raw) != state['request_sha256']:
        return False
    deployment = json.loads((Path(root)/'main-deployment.json').read_text())
    if ex['status'] == 'migration_queued':
        expected = legacy_job(deployment, state['code_sha'], MIGRATION)
        expected['created_at'] = request['created_at']
    elif ex['status'] == 'controlled_queued':
        if not ex.get('logs_analysis_receipt'):
            raise ValueError('Existing gradient observations must be analyzed before any optimization')
        expected = task(deployment, state['code_sha'], request['created_at'])
    else:
        return False
    if request != expected:
        raise ValueError('Task exceeds the declared diagnostic command/SHA/budget')
    return True


def migrate(repo, deployment_path, output):
    from gradient_logs import collect
    repo, output = Path(repo), Path(output)
    d = json.loads(Path(deployment_path).read_text())
    root = Path(d['runner_root'])
    runner = Runner(root)
    runner.fetch(('requests_branch', 'code_branch'))
    state = json.loads(runner.read(runner.ref('requests_branch'), 'control/main.json'))
    raw = runner.read(runner.ref('requests_branch'), 'jobs/'+MIGRATION+'.json')
    request = json.loads(raw)
    code = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if (os.environ.get('LAB_BUFFER_JOB_ID') != MIGRATION or code != os.environ.get('LAB_BUFFER_CODE_SHA')
            or code != request['code_sha'] or not authorize(root, request, raw, state)):
        raise ValueError('Scope migration identity/authorization mismatch')
    deadline = time.monotonic()+3
    while True:
        running = json.loads((root/'state'/(MIGRATION+'.json')).read_text())['result']
        if running.get('pid') is not None or time.monotonic() >= deadline:
            break
        time.sleep(.01)
    if running['pid'] != os.getpid() or running['actual_code_sha'] != code:
        raise ValueError('Migration must be owned by this active parent Runner job')
    plan = json.loads((repo/PLAN).read_text())
    if plan['scope_id'] != SCOPE or plan['max_diagnostic_optimizer_steps'] != 60:
        raise ValueError('Unknown diagnostic scope/budget')
    installed = {}
    for name, sha in plan['legacy_installed_source_sha256'].items():
        p = root/'bin'/name
        installed[name] = digest(p.read_bytes())
        if installed[name] != sha:
            raise ValueError('Installed Runner differs from the accepted delivery: '+name)
    logs = collect(d['retuned_dir'], repo/'train_retuned_historical.py')
    backup = root/'scope-upgrade-backups'/code
    backup.mkdir(parents=True, exist_ok=False)
    atomic_json(backup/'config.json', runner.config)
    before = {}
    for name in ('main_control.py', 'main_service.py', 'gradient_bridge.py', 'gradient_logs.py', 'runner.py'):
        target = root/'bin'/name
        if target.exists():
            before[name] = digest(target.read_bytes())
            shutil.copyfile(target, backup/name)
    if (root/'runner-code-sha.txt').exists():
        shutil.copyfile(root/'runner-code-sha.txt', backup/'runner-code-sha.txt')
    # The parent Runner owns runner.lock for this entire child lifetime. No main
    # checkout, cron, input, or other process is changed.
    for name in ('gradient_logs.py', 'gradient_bridge.py', 'main_control.py', 'main_service.py', 'runner.py'):
        target = root/'bin'/name
        pending = target.with_suffix('.new')
        shutil.copyfile(repo/'scripts/lab_buffer'/name, pending)
        os.replace(pending, target)
    atomic_json(root/'config.json', dict(runner.config, main_scope_revision=2))
    (root/'runner-code-sha.txt').write_text(code+'\n')
    # The still-running old service understands plan_completed as "do not
    # interpret this job as a Future acceptance". The extension is explicitly
    # awaiting its terminal result; the new service observes it on the next tick.
    runner.fetch(('requests_branch',))
    parent = runner.git('rev-parse', runner.ref('requests_branch')).stdout.decode().strip()
    latest = json.loads(runner.read(runner.ref('requests_branch'), 'control/main.json'))
    if latest['active_job_id'] != MIGRATION or latest.get('lease') or latest['request_sha256'] != digest(raw):
        raise ValueError('Concurrent controller change during scope handoff')
    latest['mode'] = 'plan_completed'
    latest['extension'].update(status='awaiting_migration_result', installed_code_sha=code,
        legacy_slot=dict(job_id=MIGRATION, purpose='explicit scope upgrade + read-only log extraction',
                         not_a_repair=True, previous_repairs_used=0))
    runner.publish({'control/main.json': encoded(latest)}, 'requests_branch', expected_parent=parent)
    if any(digest(Path(f['path']).read_bytes()) != f['sha256'] for f in logs['files']):
        raise ValueError('Original trace changed during migration')
    return dict(stage='history_gradient_scope_migration_v1', execution_status='completed',
        job_id=MIGRATION, code_sha=code, scope_id=SCOPE, training_started=False,
        original_trace_inputs_unchanged=True, main_workspace_written=False, optimizer_steps_this_task=0,
        checks=dict(installed_legacy_sources_verified=True, trace_source_verified=True,
                    no_main_checkout_switch=True, no_cron_change=True, no_training=True),
        runner_upgrade=dict(backup=str(backup), previous_source_sha256=before,
                            installed_source_sha256={n: digest((root/'bin'/n).read_bytes()) for n in
                                ('runner.py','main_control.py','main_service.py','gradient_bridge.py','gradient_logs.py')}),
        gradient_logs=logs)


def observe(control, state):
    ex = state['extension']
    if state['pause_requested']:
        return dict(status='paused')
    if ex['status'] in ('logs_pending_analysis', 'controlled_pending_analysis', 'completed'):
        return dict(status=ex['status'])
    name = state['active_job_id']
    r = control.runner
    raw_request = r.read(r.ref('requests_branch'), 'jobs/'+name+'.json')
    result_raw = r.read(r.ref('results_branch'), 'jobs/'+name+'.json')
    if result_raw is None:
        return dict(status='waiting', job_id=name)
    request, result = json.loads(raw_request), json.loads(result_raw)
    if (digest(raw_request) != state['request_sha256'] or result['request_sha256'] != digest(raw_request)
            or any(request[k] != result[k] for k in ('job_id','worker_id','kind','code_sha'))):
        raise ValueError('Diagnostic request/result identity mismatch')
    if result['status'] in ('starting','running'):
        return dict(status=result['status'], job_id=name)
    token = control.claim('server-diagnostic-observer')
    if token is None:
        return dict(status='controller_busy')
    state, parent = control.read()
    if result['status'] != 'completed' or result['exit_code'] != 0 or result['actual_code_sha'] != request['code_sha']:
        state.update(mode='needs_diagnosis', pause_requested=True, lease=None)
        state['extension'].update(status='failed', failure_job_id=name, result=result)
        control.write(state, parent)
        return dict(status='needs_diagnosis')
    relative = ('outputs/main-acceptance.json' if name == MIGRATION else 'outputs/controlled/compact.json')
    path = 'artifacts/'+name+'/'+relative
    if result['artifacts'] != [path]:
        raise ValueError('Unexpected diagnostic artifact')
    raw = r.read(r.ref('results_branch'), path)
    if raw is None or digest(raw) != result['artifacts_sha256'][path]:
        raise ValueError('Diagnostic artifact checksum mismatch')
    artifact = json.loads(raw)
    expected = 'history_gradient_scope_migration_v1' if name == MIGRATION else 'history_gradient_controlled_v1'
    if (artifact['stage'] != expected or artifact['execution_status'] != 'completed'
            or artifact['job_id'] != name or artifact['code_sha'] != request['code_sha']):
        raise ValueError('Diagnostic artifact stage/identity mismatch')
    receipt = dict(schema_version=1, status='captured_for_web_analysis', job_id=name,
        request_sha256=digest(raw_request), artifact_sha256=digest(raw), code_sha=request['code_sha'],
        captured_at=now(), artifact_path=path)
    state['extension'].update(status='logs_pending_analysis' if name == MIGRATION else 'controlled_pending_analysis',
                              result_receipt=receipt)
    state.update(mode='awaiting_analysis', lease=None)
    # Restore the scientific repair counter: slot 002 was transport migration.
    if name == MIGRATION:
        state['repairs_used'] = state['extension']['legacy_slot']['previous_repairs_used']
    control.write(state, parent, {'phase-receipts/'+SCOPE+'/'+name+'.json': encoded(receipt)})
    return dict(status=state['extension']['status'], receipt=receipt)
