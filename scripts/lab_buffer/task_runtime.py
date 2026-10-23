"""Stable v1 task boundary. Research policy and verifiers are immutable Git data/code.

This is a trusted-repository executor, not an OS sandbox. It never upgrades itself,
grants a research scope, changes cron, retries a task or terminates a live child.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from runner import Runner, SHA, NAME, FINAL, MAX_MESSAGE, atomic_json, digest, now, validate_task

RUNTIME_ARTIFACT = 'outputs/task-runtime.json'


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def relative(value):
    p = Path(value)
    if not value or p.is_absolute() or '..' in p.parts or str(p) != value:
        raise ValueError('Expected a normalized repository-relative path')
    return p


def command(argv):
    if (not isinstance(argv, list) or len(argv) < 4 or argv[:3] != ['python', '-u', '-B']
            or not all(isinstance(v, str) and '\0' not in v for v in argv)):
        raise ValueError('Task/check requires python -u -B and a repository script; no shell/-c/-m')
    if relative(argv[3]).suffix != '.py':
        raise ValueError('Expected a Python script')


def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(r, task, raw, contract, scope):
    policy = json.loads((r.root/'task-runtime-policy.json').read_text())
    if r.config.get('task_runtime_version') != 1 or policy['protocol_version'] != 1:
        raise ValueError('Task runtime v1 is not installed')
    validate_task(raw, 'jobs/'+task['job_id']+'.json')
    if set(contract) != {'protocol_version', 'scope_id', 'job_id', 'code_sha', 'request_sha256',
                         'scope_path', 'scope_sha256', 'manifest', 'preflight', 'verifier',
                         'optimizer_step_reservation', 'input_sha256'}:
        raise ValueError('Unknown task contract fields')
    if (contract['protocol_version'] != 1 or scope['protocol_version'] != 1
            or not NAME.fullmatch(contract['scope_id']) or contract['scope_id'] != scope['scope_id']
            or any(contract[k] != task[k] for k in ('job_id', 'code_sha'))
            or contract['request_sha256'] != digest(raw) or task['kind'] != 'experiment'
            or task['worker_id'] != r.config['worker_id']
            or RUNTIME_ARTIFACT not in task['artifacts']):
        raise ValueError('Contract identity/protocol differs from the request')
    for value in (scope['max_optimizer_steps'], contract['optimizer_step_reservation']):
        if type(value) is not int or value < 0:
            raise ValueError('Optimizer budget must be a nonnegative integer')
    if (contract['optimizer_step_reservation'] > scope['max_optimizer_steps']
            or type(scope['max_parallel']) is not int
            or not 1 <= scope['max_parallel'] <= policy['max_parallel']):
        raise ValueError('Task exceeds the declared scope/local resource limit')
    command(task['argv'])
    if not isinstance(contract['preflight'], list) or not 1 <= len(contract['preflight']) <= 8:
        raise ValueError('Require explicit bounded preflight commands')
    for argv in contract['preflight']:
        command(argv)
    manifest = contract['manifest']
    relative(manifest['path'])
    source = r.read(task['code_sha'], manifest['path'])
    if source is None or digest(source) != manifest['sha256']:
        raise ValueError('Pinned scientific manifest hash differs')
    for script in [task['argv'][3], contract['verifier'], *[v[3] for v in contract['preflight']]]:
        relative(script)
        if Path(script).suffix != '.py' or r.read(task['code_sha'], script) is None:
            raise ValueError('Missing SHA-pinned entry/check/verifier')
    roots = [Path(v).resolve() for v in policy['read_roots']]
    if not isinstance(contract['input_sha256'], dict):
        raise ValueError('Input bindings must be explicit hashes')
    for path, sha in contract['input_sha256'].items():
        p = Path(path)
        if (not p.is_absolute() or not any(p.resolve().is_relative_to(base) for base in roots)
                or not isinstance(sha, str) or len(sha) != 64
                or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('Input outside local read roots or invalid checksum')


def load_contract(r, state, task, raw):
    dispatch = state['task_dispatch']
    path = 'task-contracts/'+task['job_id']+'.json'
    if (dispatch['contract_path'] != path or state['active_job_id'] != task['job_id']
            or state['code_sha'] != task['code_sha'] or state['request_sha256'] != digest(raw)):
        raise ValueError('Active dispatch differs from request')
    data = r.read(r.ref('requests_branch'), path)
    if data is None or digest(data) != dispatch['contract_sha256']:
        raise ValueError('Immutable contract checksum mismatch')
    contract = json.loads(data)
    scope_path = 'task-scopes/'+contract['scope_id']+'.json'
    if contract['scope_path'] != scope_path:
        raise ValueError('Unexpected scope authorization path')
    scope_raw = r.read(r.ref('requests_branch'), scope_path)
    if scope_raw is None or digest(scope_raw) != contract['scope_sha256']:
        raise ValueError('Scope authorization changed/missing')
    scope = json.loads(scope_raw)
    validate(r, task, raw, contract, scope)
    ledger = state['task_scope_ledger'][contract['scope_id']]
    if (ledger['scope_sha256'] != digest(scope_raw)
            or ledger['reservations'].get(task['job_id']) != contract['optimizer_step_reservation']
            or sum(ledger['reservations'].values()) > scope['max_optimizer_steps']):
        raise ValueError('Scope reservation/accounting mismatch')
    return contract, data, scope


def authorize(r, task, raw, state):
    if state['pause_requested'] or state['mode'] != 'queued' or state.get('lease'):
        return False
    contract, data, scope = load_contract(r, state, task, raw)
    snapshot = r.root/'task-contracts'/(task['job_id']+'.json')
    value = dict(contract=contract, scope=scope, request=task, request_raw=raw.decode(), contract_sha256=digest(data))
    if snapshot.exists() and json.loads(snapshot.read_text()) != value:
        raise ValueError('Local immutable task snapshot changed')
    if not snapshot.exists():
        atomic_json(snapshot, value)
    return True


def queue(control, task, contract, scope):
    """One CAS publishes scope, contract, request and reservation. No server install."""
    state, parent = control.read()
    r = control.runner
    if (state['mode'] != 'plan_completed' or state['pause_requested'] or state.get('lease')
            or (r.root/'PAUSED.json').exists()):
        raise ValueError('Prior plan must be completed and reconciled before new dispatch')
    raw = encode(task)
    contract = dict(contract, request_sha256=digest(raw), scope_sha256=digest(encode(scope)),
                    scope_path='task-scopes/'+scope['scope_id']+'.json')
    validate(r, task, raw, contract, scope)
    if r.git('merge-base', '--is-ancestor', task['code_sha'], r.ref('code_branch'), check=False).returncode:
        raise ValueError('Task code is outside the reviewed code branch')
    files = {'jobs/'+task['job_id']+'.json': raw,
             'task-contracts/'+task['job_id']+'.json': encode(contract)}
    for path in files:
        if r.read(r.ref('requests_branch'), path) is not None or r.read(r.ref('results_branch'), path) is not None:
            raise ValueError('Job/contract already exists; never overwrite or reuse an ID')
    path = contract['scope_path']
    previous = r.read(r.ref('requests_branch'), path)
    if previous is not None and previous != encode(scope):
        raise ValueError('Scope authorization is immutable; changes require a newly approved scope')
    files[path] = encode(scope)
    ledger = state.setdefault('task_scope_ledger', {}).setdefault(scope['scope_id'],
        dict(scope_sha256=contract['scope_sha256'], reservations={}, actual_steps={}))
    if ledger['scope_sha256'] != contract['scope_sha256']:
        raise ValueError('Scope hash differs from its ledger')
    ledger['reservations'][task['job_id']] = contract['optimizer_step_reservation']
    if sum(ledger['reservations'].values()) > scope['max_optimizer_steps']:
        raise ValueError('Cumulative reservations exceed the approved budget; no unused-budget recycling')
    state.update(mode='queued', active_job_id=task['job_id'], code_sha=task['code_sha'],
        request_sha256=digest(raw), task_dispatch=dict(protocol_version=1,
        contract_path='task-contracts/'+task['job_id']+'.json', contract_sha256=digest(encode(contract))))
    control.write(state, parent, files)
    return dict(status='queued', job_id=task['job_id'])


def check_inputs(contract):
    for path, sha in contract['input_sha256'].items():
        if file_hash(path) != sha:
            raise ValueError('Bound input checksum changed: '+path)


def source_file(r, workspace, code, name):
    path = workspace/relative(name)
    source = r.read(code, name)
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(workspace)
            or source is None or digest(path.read_bytes()) != digest(source)):
        raise ValueError('Local source differs from pinned Git object: '+name)
    return path


def clean_source(r, workspace, code):
    actual = subprocess.check_output(['git', '-C', str(workspace), 'rev-parse', 'HEAD'], env=r.env).decode().strip()
    if actual != code or subprocess.run(['git', '-C', str(workspace), 'diff', '--quiet', 'HEAD', '--'], env=r.env).returncode:
        raise ValueError('Pinned task checkout has changed tracked source')


def run(root, job_id, workspace):
    r = Runner(root)
    value = json.loads((r.root/'task-contracts'/(job_id+'.json')).read_text())
    contract, task = value['contract'], value['request']
    workspace = Path(workspace).resolve()
    if workspace != (r.root/'workspaces'/job_id).resolve():
        raise ValueError('Unexpected task workspace')
    record = dict(protocol_version=1, job_id=job_id, code_sha=task['code_sha'],
        contract_sha256=value['contract_sha256'], status='preflight', checks_completed=0,
        command_started=False, started_at=now())
    output = workspace/RUNTIME_ARTIFACT
    atomic_json(output, record)
    try:
        clean_source(r, workspace, task['code_sha'])
        check_inputs(contract)
        env = dict(os.environ, LAB_TASK_CONTRACT=str(r.root/'task-contracts'/(job_id+'.json')),
            LAB_SCOPE_STEP_LIMIT=str(contract['optimizer_step_reservation']),
            LAB_SCOPE_MAX_PARALLEL=str(value['scope']['max_parallel']))
        for argv in [*contract['preflight'], task['argv']]:
            source_file(r, workspace, task['code_sha'], argv[3])
            is_command = record['checks_completed'] == len(contract['preflight'])
            if is_command:
                # Refresh before dispatch; a pause during checks must not start training.
                r.fetch(('requests_branch',))
                state = json.loads(r.read(r.ref('requests_branch'), 'control/main.json'))
                load_contract(r, state, task, value['request_raw'].encode())
                if state['pause_requested'] or (r.root/'PAUSED.json').exists():
                    raise RuntimeError('Paused before task command; no experiment started')
                record.update(status='running', command_started=True)
                atomic_json(output, record)
            child_env = dict(env)
            if not is_command:
                child_env['CUDA_VISIBLE_DEVICES'] = ''
            # No shell evaluation, termination or hidden restart.
            subprocess.run([r.config['python'], *argv[1:]], cwd=workspace, env=child_env, check=True)
            if not is_command:
                record['checks_completed'] += 1
                atomic_json(output, record)
        check_inputs(contract)
        clean_source(r, workspace, task['code_sha'])
        record.update(status='completed', original_inputs_unchanged=True)
        return 0
    except BaseException as error:
        record.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        record['finished_at'] = now()
        atomic_json(output, record)


def observe(control, state):
    r = control.runner
    job_id = state['active_job_id']
    if job_id in state['receipts']:
        return dict(status='already_processed', job_id=job_id)
    raw = r.read(r.ref('requests_branch'), 'jobs/'+job_id+'.json')
    task = json.loads(raw)
    contract, data, scope = load_contract(r, state, task, raw)
    result_raw = r.read(r.ref('results_branch'), 'jobs/'+job_id+'.json')
    if result_raw is None:
        return dict(status='waiting', job_id=job_id)
    result = json.loads(result_raw)
    if (result['request_sha256'] != digest(raw)
            or any(result[k] != task[k] for k in ('job_id', 'code_sha', 'worker_id', 'kind'))):
        raise ValueError('Result identity differs from the exact request')
    if result['status'] not in FINAL:
        return dict(status=result['status'], job_id=job_id)
    token = control.claim('versioned-task-observer')
    if token is None:
        return dict(status='controller_busy_or_uncertain')
    try:
        if result['status'] != 'completed' or result['exit_code'] != 0 or result['actual_code_sha'] != task['code_sha']:
            raise ValueError('Task failed/interrupted; preserve evidence and diagnose before any new job')
        workspace = (r.root/'workspaces'/job_id).resolve()
        if Path(result['workspace']).resolve() != workspace:
            raise ValueError('Unexpected returned workspace')
        expected = ['artifacts/'+job_id+'/'+p for p in task['artifacts']]
        if result['artifacts'] != expected:
            raise ValueError('Expected artifact set differs')
        for name, path in zip(task['artifacts'], expected):
            artifact = r.read(r.ref('results_branch'), path)
            if artifact is None or digest(artifact) != result['artifacts_sha256'][path]:
                raise ValueError('Returned artifact hash differs')
            local = workspace/relative(name)
            if local.is_symlink() or not local.resolve().is_relative_to(workspace) or file_hash(local) != digest(artifact):
                raise ValueError('Local artifact differs from returned bytes')
        runtime = json.loads((workspace/RUNTIME_ARTIFACT).read_text())
        if (runtime['status'] != 'completed' or runtime['contract_sha256'] != digest(data)
                or runtime['checks_completed'] != len(contract['preflight'])
                or runtime['original_inputs_unchanged'] is not True):
            raise ValueError('Task/check/input acceptance failed')
        check_inputs(contract)
        clean_source(r, workspace, task['code_sha'])
        verifier = source_file(r, workspace, task['code_sha'], contract['verifier'])
        directory = r.root/'task-verifications'/job_id
        directory.mkdir(parents=True, exist_ok=True)
        context = directory/'context.json'
        out = directory/'result.json'
        atomic_json(context, dict(contract=contract, contract_sha256=digest(data), request=task, result=result,
            scope=scope, workspace=str(workspace), output=str(out)))
        log = directory/'verifier.log'
        # Verifier is task code from this job's full SHA, not an installed plugin.
        with log.open('ab') as handle:
            subprocess.run([r.config['python'], '-u', '-B', str(verifier), '--context', str(context),
                            '--output', str(out)], cwd=workspace, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, env=dict(r.env, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        if out.stat().st_size > MAX_MESSAGE:
            raise ValueError('Verifier result too large')
        verdict = json.loads(out.read_text())
        if (verdict['status'] != 'verified' or verdict['job_id'] != job_id
                or verdict['contract_sha256'] != digest(data)
                or type(verdict['actual_optimizer_steps']) is not int
                or not 0 <= verdict['actual_optimizer_steps'] <= contract['optimizer_step_reservation']
                or not isinstance(verdict['report_markdown'], str)):
            raise ValueError('Scientific verifier failed or exceeded the reserved budget')
        marker = '<!-- lab-task:'+job_id+':'+digest(data)+' -->'
        r.fetch(('code_branch',))
        head = r.git('rev-parse', r.ref('code_branch')).stdout.decode().strip()
        report = r.read(head, 'experiment_report.md').decode()
        if marker not in report:
            r.publish({'experiment_report.md': (report.rstrip()+'\n\n'+marker+'\n\n'+verdict['report_markdown']+'\n').encode()},
                      'code_branch', expected_parent=head)
            r.fetch(('code_branch',))
            head = r.git('rev-parse', r.ref('code_branch')).stdout.decode().strip()
        state, parent = control.read()
        if (state.get('lease') or {}).get('token') != token:
            raise ValueError('Lost shared report/receipt lease; reconcile marker')
        receipt = dict(status='verified_and_reported', job_id=job_id, code_sha=task['code_sha'],
            contract_sha256=digest(data), request_sha256=digest(raw), result_sha256=digest(result_raw),
            actual_optimizer_steps=verdict['actual_optimizer_steps'], report_commit_sha=head,
            artifacts_sha256=result['artifacts_sha256'], processed_at=now(), next_job_id=None)
        state['receipts'][job_id] = receipt
        state['task_scope_ledger'][scope['scope_id']]['actual_steps'][job_id] = verdict['actual_optimizer_steps']
        state.update(mode='plan_completed', lease=None)
        control.write(state, parent, {'receipts/'+job_id+'.json': encode(receipt)})
        return dict(status='plan_completed', receipt=receipt)
    except BaseException as error:
        state, parent = control.read()
        if (state.get('lease') or {}).get('token') == token:
            receipt = dict(status='needs_attention', job_id=job_id, request_sha256=digest(raw),
                result_sha256=digest(result_raw), reason=repr(error), processed_at=now(), next_job_id=None)
            log = r.root/'task-verifications'/job_id/'verifier.log'
            if log.is_file():
                with log.open('rb') as handle:
                    handle.seek(max(0, log.stat().st_size-8192))
                    receipt['verifier_log_tail'] = handle.read().decode('utf-8', 'replace')
            state['receipts'][job_id] = receipt
            state.update(mode='needs_diagnosis', pause_requested=True, lease=None)
            control.write(state, parent, {'receipts/'+job_id+'.json': encode(receipt)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'queue'])
    p.add_argument('--root', required=True)
    p.add_argument('--job-id')
    p.add_argument('--workspace')
    p.add_argument('--submission', help='JSON containing request, contract and explicitly approved scope')
    a = p.parse_args()
    if a.action == 'run':
        sys.exit(run(a.root, a.job_id, a.workspace))
    else:
        from main_control import Control
        v = json.loads(Path(a.submission).read_text())
        print(json.dumps(queue(Control(a.root), v['request'], v['contract'], v['scope']), indent=2))
