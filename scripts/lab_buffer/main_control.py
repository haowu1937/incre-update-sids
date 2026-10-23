#!/usr/bin/env python3
"""Git-backed state, leases and receipts; v1 versioned tasks plus legacy adapters.

Research authorization is read from persistent state, never inferred from this
module's installed version. CLI, server and Web share the same state.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import uuid

from runner import Runner, atomic_json, digest, now

STATE = 'control/main.json'
PROTOCOL = 'docs/lab_buffer/main_protocol.json'
AUTOMATION_ID = '6ad746543320819187a85beaf87af5bd'


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def initial_state():
    return dict(schema_version=1, protocol_id='sid-main-v1', mode='awaiting_deployment',
                pause_requested=False, active_job_id=None, lease=None, receipts={},
                plan=['accept_existing_history_future'], repairs_used=0,
                automation_id=AUTOMATION_ID, automation_acceptance='pending')


def job(deployment, code_sha, job_id):
    return dict(schema_version=1, job_id=job_id, worker_id='server2', kind='experiment',
                code_sha=code_sha, created_at=now(), timeout_seconds=1800,
                argv=['python', '-u', '-B', 'scripts/lab_buffer/accept_main.py',
                      '--deployment', str(Path(deployment['runner_root'])/'main-deployment.json'),
                      '--output', 'outputs/main-acceptance.json'],
                artifacts=['outputs/main-acceptance.json'])


def authorize_task(root, task, raw, state):
    if state and state.get('task_dispatch'):
        from task_runtime import authorize
        return authorize(Runner(root), task, raw, state)
    if state and state.get('extension', {}).get('scope_id') == 'history-joint-convergence-v1':
        from convergence_bridge import authorize
        return authorize(root, task, raw, state)
    if state and state.get('extension', {}).get('scope_id') == 'history-gradient-v1':
        from gradient_bridge import authorize
        return authorize(root, task, raw, state)
    if not state or state['pause_requested'] or state['mode'] != 'queued':
        return False
    if state['active_job_id'] != task['job_id'] or state['request_sha256'] != digest(raw):
        return False
    dep = json.loads((Path(root)/'main-deployment.json').read_text())
    expected = job(dep, state['code_sha'], task['job_id'])
    expected['created_at'] = task['created_at']
    permitted_id = ('history-future-accept-001' if state.get('repairs_used', 0) == 0
                    else 'history-future-accept-002')
    if task != expected or task['job_id'] != permitted_id:
        raise ValueError('Main job is outside the exact deployed acceptance command/SHA/budget')
    return True


class Control:
    def __init__(self, root):
        self.runner = Runner(root)
        self.root = Path(root)

    def read(self):
        self.runner.fetch()
        raw = self.runner.read(self.runner.ref('requests_branch'), STATE)
        state = json.loads(raw) if raw else initial_state()
        if state.get('schema_version') != 1 or state.get('protocol_id') != 'sid-main-v1':
            raise ValueError('Unknown control protocol/state')
        protocol = json.loads(self.runner.read(self.runner.ref('code_branch'), PROTOCOL))
        if protocol['protocol_id'] != state['protocol_id']:
            raise ValueError('Code/control protocol mismatch')
        parent = self.runner.git('rev-parse', self.runner.ref('requests_branch')).stdout.decode().strip()
        return state, parent

    def write(self, state, parent, extra=None):
        files = {STATE: encoded(state)}
        files.update(extra or {})
        self.runner.publish(files, 'requests_branch', expected_parent=parent)

    def queue(self, deployment):
        state, parent = self.read()
        if state['mode'] != 'awaiting_deployment' or state['pause_requested']:
            return dict(status=state['mode'], active_job_id=state['active_job_id'])
        task = job(deployment, deployment['code_sha'], 'history-future-accept-001')
        path = 'jobs/'+task['job_id']+'.json'
        if self.runner.read(self.runner.ref('requests_branch'), path) is not None:
            raise ValueError('Job already exists outside control state; reconcile, do not overwrite')
        raw = encoded(task)
        state.update(mode='queued', active_job_id=task['job_id'], code_sha=task['code_sha'],
                     request_sha256=digest(raw), deployment_sha256=digest(encoded(deployment)))
        self.write(state, parent, {path: raw})
        return dict(status='queued', job_id=task['job_id'])

    def claim(self, owner):
        state, parent = self.read()
        if state['lease']:
            if state['lease']['expires_at_unix'] < time.time():
                # An expired lease may have committed a report on the other branch.
                # Never guess whether a side effect happened or dispatch another job.
                state.update(mode='needs_reconciliation', pause_requested=True)
                self.write(state, parent)
            return None
        token = uuid.uuid4().hex
        state['lease'] = dict(token=token, owner=owner, acquired_at=now(), expires_at_unix=time.time()+900)
        self.write(state, parent)
        return token

    def repair(self, code_sha, diagnosis):
        """One evidence-backed software repair; unchanged inputs/zero SGD budget."""
        state, parent = self.read()
        if (state['mode'] != 'needs_diagnosis' or state['lease'] or state['repairs_used'] != 0
                or not isinstance(diagnosis, str) or len(diagnosis.strip()) < 20):
            raise ValueError('Repair requires a diagnosed terminal fault and unused repair allowance')
        _, result, _ = self.evidence(state)
        if result['status'] not in ('failed', 'blocked') or result.get('process_started') and result.get('exit_code') is None:
            raise ValueError('Unknown process lifetime: no repair may be dispatched')
        if (self.root/'PAUSED.json').exists():
            raise ValueError('Local uncertainty pause must be reconciled first')
        if code_sha == state['code_sha'] or len(code_sha) != 40 or self.runner.git(
                'merge-base', '--is-ancestor', code_sha, self.runner.ref('code_branch'), check=False).returncode:
            raise ValueError('Repair must pin a different, available, reviewed code SHA')
        deployment = json.loads((self.root/'main-deployment.json').read_text())
        task = job(deployment, code_sha, 'history-future-accept-002')
        path = 'jobs/'+task['job_id']+'.json'
        if self.runner.read(self.runner.ref('requests_branch'), path) is not None:
            raise ValueError('Repair job already exists; never overwrite it')
        raw = encoded(task)
        state.update(mode='queued', pause_requested=False, active_job_id=task['job_id'], code_sha=code_sha,
                     request_sha256=digest(raw), repairs_used=1,
                     repair=dict(parent_job_id=result['job_id'], diagnosis=diagnosis,
                                 fault_fingerprint=state['failure']['fingerprint']))
        self.write(state, parent, {path: raw})
        return dict(status='queued', job_id=task['job_id'])

    def release(self, token, **updates):
        state, parent = self.read()
        if not state['lease'] or state['lease']['token'] != token:
            raise ValueError('Lost controller lease; no further writes')
        state.update(updates)
        state['lease'] = None
        self.write(state, parent)

    def evidence(self, state):
        name = state['active_job_id']
        path = 'jobs/'+name+'.json'
        raw = self.runner.read(self.runner.ref('requests_branch'), path)
        task = json.loads(raw)
        result_raw = self.runner.read(self.runner.ref('results_branch'), path)
        if result_raw is None:
            return task, None, None
        result = json.loads(result_raw)
        if (digest(raw) != state['request_sha256'] or result['request_sha256'] != digest(raw)
                or any(result[k] != task[k] for k in ('job_id', 'worker_id', 'code_sha', 'kind'))):
            raise ValueError('Request/result identity mismatch')
        if result['status'] != 'completed':
            return task, result, None
        if result['actual_code_sha'] != task['code_sha'] or result['exit_code'] != 0 or result['training_started'] is not False:
            raise ValueError('Code/exit mismatch')
        artifact = 'artifacts/'+name+'/outputs/main-acceptance.json'
        if result['artifacts'] != [artifact]:
            raise ValueError('Unexpected acceptance artifacts')
        raw_artifact = self.runner.read(self.runner.ref('results_branch'), artifact)
        if raw_artifact is None or digest(raw_artifact) != result['artifacts_sha256'][artifact]:
            raise ValueError('Acceptance artifact SHA256 mismatch')
        audit = json.loads(raw_artifact)
        if (audit['stage'] != 'sid_main_existing_future_acceptance' or audit['execution_status'] != 'completed'
                or audit['job_id'] != name or audit['code_sha'] != task['code_sha']
                or audit['training_started'] is not False or audit['original_inputs_unchanged'] is not True
                or not audit['checks'] or not all(v is True for v in audit['checks'].values())):
            raise ValueError('Real-input acceptance failed')
        return task, result, (audit, digest(raw_artifact))

    def process(self, owner='server-after-acceptance'):
        state, _ = self.read()
        if state.get('task_dispatch'):
            from task_runtime import observe
            return observe(self, state)
        if state.get('extension', {}).get('scope_id') == 'history-joint-convergence-v1':
            from convergence_bridge import observe
            return observe(self, state)
        if state.get('extension', {}).get('scope_id') == 'history-gradient-v1':
            from gradient_bridge import observe
            return observe(self, state)
        if state['mode'] in ('awaiting_deployment', 'plan_completed'):
            return dict(status=state['mode'])
        if state['pause_requested']:
            return dict(status='paused')
        _, result, evidence = self.evidence(state)
        if result is None or result['status'] in ('starting', 'running'):
            return dict(status='waiting' if result is None else result['status'], job_id=state['active_job_id'])
        if state['active_job_id'] in state['receipts']:
            return dict(status='already_processed')
        token = self.claim(owner)
        if token is None:
            return dict(status='controller_busy_or_uncertain')
        try:
            state, _ = self.read()
            _, result, evidence = self.evidence(state)
            if evidence is None:
                fingerprint = digest(encoded({k: result.get(k) for k in ('status', 'reason', 'exit_code', 'request_sha256')}))
                receipt = dict(schema_version=1, job_id=state['active_job_id'], status='needs_attention',
                               result_sha256=digest(encoded(result)), request_sha256=result['request_sha256'],
                               fingerprint=fingerprint, processed_at=now(), next_job_id=None)
                state, parent = self.read()
                state['receipts'][state['active_job_id']] = receipt
                state.update(lease=None, mode='needs_diagnosis', pause_requested=True,
                             failure=dict(job_id=state['active_job_id'], fingerprint=fingerprint,
                                          status=result['status'], reason=result.get('reason'), log_tail=result.get('log_tail')))
                self.write(state, parent, {'receipts/'+state['active_job_id']+'.json': encoded(receipt)})
                return dict(status='needs_diagnosis', result=result)
            audit, sha = evidence
            marker = '<!-- sid-main-acceptance:'+sha+' -->'
            self.runner.fetch(('code_branch',))
            parent = self.runner.git('rev-parse', self.runner.ref('code_branch')).stdout.decode().strip()
            report = self.runner.read(self.runner.ref('code_branch'), 'experiment_report.md').decode()
            if marker not in report:
                compact = audit['compact']
                f, t = [compact['future'][k]['full_denominator'] for k in ('frozen', 'finetuned')]
                appendix = '\n\n### Git Runner acceptance of the completed History-aware Future experiment\n\n'+marker+'\n\n'
                appendix += ('Job `'+audit['job_id']+'`; code `'+audit['code_sha']+'`; original evaluation `'+audit['future_dir']+'`. '
                    'The independent Runner loaded the real checkpoints, reproduced the full SID ledgers, '
                    'verified source/config/artifact bindings and selection, recomputed both aggregate metrics from saved query records, '
                    'and left the input bytes unchanged. No optimizer steps or new Future predictions were performed. '
                    'This is artifact and reload acceptance, not an independent retraining or new generalization trial.\n\n')
                appendix += ('Full-denominator Future Recall@10: frozen '+format(f['recall@10']*100,'.6f')+'%, fine-tuned '+format(t['recall@10']*100,'.6f')+
                    '%; NDCG@10: '+format(f['ndcg@10'],'.12f')+' / '+format(t['ndcg@10'],'.12f')+'. '
                    'TIGER trained 2000 Update steps and selected step 2000. '
                    'The retuned paired anchor result remains negative for the historical penalty. '
                    'The older Base/Plain Future controls use a different RQ recipe; no isolated penalty benefit is established. '
                    'The explicitly requested two-condition plan is complete; no further training jobs are queued.\n')
                self.runner.publish({'experiment_report.md': (report+appendix).encode()}, 'code_branch', expected_parent=parent)
                self.runner.fetch(('code_branch',))
                parent = self.runner.git('rev-parse', self.runner.ref('code_branch')).stdout.decode().strip()
            receipt = dict(schema_version=1, job_id=audit['job_id'], status='verified',
                           request_sha256=result['request_sha256'], artifact_sha256=sha,
                           summary_sha256=audit['summary_sha256'], report_commit_sha=parent,
                           code_sha=audit['code_sha'], processed_at=now(), next_job_id=None)
            state, branch_parent = self.read()
            if not state['lease'] or state['lease']['token'] != token:
                raise ValueError('Lost lease after report publication; reconcile report marker')
            state['receipts'][audit['job_id']] = receipt
            state.update(lease=None, mode='plan_completed', automation_acceptance='passed')
            self.write(state, branch_parent, {'receipts/'+audit['job_id']+'.json': encoded(receipt)})
            return dict(status='plan_completed', receipt=receipt)
        except BaseException:
            # Preserve the claim. Recovery must inspect the cross-branch marker.
            raise

    def pause(self):
        state, parent = self.read()
        state.update(pause_requested=True, pause_reason='user paused automatic iteration', paused_at=now())
        self.write(state, parent)
        return dict(status='paused_new_dispatch_only')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['status', 'process', 'pause', 'repair'])
    p.add_argument('--root', required=True)
    p.add_argument('--code-sha')
    p.add_argument('--diagnosis')
    a = p.parse_args()
    c = Control(a.root)
    answer = (c.read()[0] if a.action == 'status' else c.process('manual-or-timed') if a.action == 'process'
              else c.repair(a.code_sha, a.diagnosis) if a.action == 'repair' else c.pause())
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
