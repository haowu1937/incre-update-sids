#!/usr/bin/env python3
"""Upgrade the existing mailbox installation under its lock; never alter the project."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from runner import Runner, atomic_json, digest, now
from main_control import Control, encoded

PROJECT = Path('/home/hao/frec/graph-informed-sids-web')
PYTHON = '/home/hao/miniconda3/envs/frec-gpu/bin/python'
E1 = 'results/incremental/Beauty/llama_plain_history_damage_ranking_20261019T111607Z_zCkKTi/evaluation'
RETUNED = 'results/incremental/Beauty/llama_history_retuned_20261020T123132Z_Edpqdn/experiment'
MARKERS = ('# git-lab-buffer-server2', '# git-lab-retuned-collector')


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def cron_read():
    r = subprocess.run(['crontab', '-l'], capture_output=True, text=True, env=dict(os.environ, LC_ALL='C'))
    if r.returncode and 'no crontab for' not in r.stderr.lower():
        raise RuntimeError('Cannot read existing crontab; no schedule modified')
    return r.stdout if r.returncode == 0 else ''


def merge_cron(text, python, root):
    if any(c in str(root)+python for c in (' ', '\n', '\t', '%', "'")):
        raise ValueError('Unexpected cron path characters')
    lines = [line for line in text.splitlines(keepends=True) if not any(m in line for m in MARKERS)]
    preserved = ''.join(lines)
    return preserved+('' if not preserved or preserved.endswith('\n') else '\n')+(
        '* * * * * '+python+' -B '+str(root/'bin/main_service.py')+' --root '+str(root)+
        ' >> '+str(root/'runner.log')+' 2>&1 # git-lab-buffer-server2\n')


def discover(project):
    """Resolve exactly one completed run by known inputs and returned model identity."""
    candidates = []
    expected_sha = 'bca77fef0b0bb3a107f40d8868f0b805885a21152289a29e97baa4ad6b26177f'
    for run in sorted((project/'results/incremental/Beauty').glob('llama_historical_future_*')):
        p = run/'evaluation/summary.json'
        if not p.is_file() or run.is_symlink() or project not in p.resolve().parents:
            continue
        value = json.loads(p.read_text())
        if (value.get('stage') == 'retuned_History_aware_Future_evaluation'
                and value.get('execution_status') == 'completed'
                and value.get('retuned_dir') == str(project/RETUNED)
                and value.get('future', {}).get('finetuned', {}).get('model_sha256') == expected_sha):
            candidates.append(p.parent)
    if len(candidates) != 1:
        raise ValueError('Require exactly one matching completed Future run; found '+str(len(candidates))+'. No latest-run guess.')
    return candidates[0]


def attach(root, source=None):
    from accept_main import compare
    root = Path(root).resolve()
    project = PROJECT.resolve()
    source = Path(source) if source is not None else Path(__file__).resolve().parent
    checkout = source.parents[1]
    if root == project or root in project.parents or project in root.parents:
        raise ValueError('Runner and project must be independent')
    old = json.loads((root/'config.json').read_text())
    if (old['worker_id'] != 'server2' or old['python'] != PYTHON
            or any(old[k] != v for k, v in [('requests_branch', 'lab-requests'), ('results_branch', 'lab-results'), ('code_branch', 'vibe-dev')])):
        raise ValueError('Existing Runner identity does not match the authorized installation')
    with (root/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Existing Runner/collector is active; preserved installation, retry the SAME attach later')
        if (root/'PAUSED.json').exists():
            raise RuntimeError('Existing PAUSED.json requires reconciliation; preserved it and stopped')
        for p in (root/'state').glob('*.json'):
            if json.loads(p.read_text())['result']['status'] in ('starting', 'running'):
                raise RuntimeError('Uncertain existing task '+p.stem+'; no upgrade or process termination')
        cron = cron_read()
        code_sha = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        runner = Runner(root)
        runner.fetch()
        if runner.git('merge-base', '--is-ancestor', code_sha, runner.ref('code_branch'), check=False).returncode:
            raise ValueError('Attach version is not reachable from vibe-dev')
        future = discover(project)
        compact = json.loads((future/'compact.json').read_text())
        expected = json.loads((checkout/'docs/lab_buffer/history_future_expected.json').read_text())
        compare(compact, expected)
        paths = [future/'summary.json', future/'compact.json', future/'config.json', future/'selection_locked.json',
                 project/E1/'summary.json', project/RETUNED/'summary.json']
        report = json.loads((future/'summary.json').read_text())
        # Full paths, not broad symlinks: the existing reports bind original absolute
        # paths. Detached jobs read those inputs directly and write only their workspace.
        for name in report['artifact_files_sha256']:
            from accept_main import contained_file
            paths.append(contained_file(future, name))
        e1 = json.loads((project/E1/'summary.json').read_text())
        plain = json.loads((Path(e1['update_rqvae_dir'])/'summary.json').read_text())
        bound = dict(e1_dir=str(project/E1), retuned_dir=str(project/RETUNED), future_dir=str(future),
                     protocol_dir=e1['protocol_dir'], features_dir=plain['features_dir'],
                     base_rqvae_dir=plain['base_rqvae_dir'], old_plain_dir=e1['update_rqvae_dir'],
                     base_tiger_dir=e1['tiger_dir'], sid_dir=e1['sid_dir'])
        for key, value in bound.items():
            p = Path(value)
            if not p.is_absolute() or not p.is_dir() or project not in p.resolve().parents:
                raise ValueError('Missing or external exact input '+key)
        for directory in [project/RETUNED/'plain', project/RETUNED/'historical',
                          Path(plain['base_rqvae_dir']), Path(e1['update_rqvae_dir']), Path(e1['tiger_dir'])]:
            paths += [directory/'summary.json']
            paths += [directory/name for name in ('last.pth', 'best.pth') if (directory/name).exists()]
        # Include source reports' explicit data/feature/SID/cohort artifacts too.
        # This binds bytes before/after the real reload without mounting a writable
        # alias of the project or relying on a broad directory modification time.
        directories = [Path(v) for v in bound.values()]+[project/RETUNED/arm for arm in ('plain', 'historical')]
        for directory in directories:
            summary = directory/'summary.json'
            paths.append(summary)
            metadata = json.loads(summary.read_text())
            for name in metadata.get('artifact_files_sha256', {}):
                paths.append(contained_file(directory, name))
        deployment = dict(schema_version=1, project=str(project), runner_root=str(root), code_sha=code_sha,
                          **bound, input_sha256={str(p): sha(p) for p in sorted(set(paths))})
        target = root/'main-deployment.json'
        if target.exists() and json.loads(target.read_text()) != deployment:
            raise ValueError('Existing deployment differs; preserve and reconcile rather than silently rebind')
        backup = root/'main-attach-backups'/now().replace(':', '').replace('+', '_')
        backup.mkdir(parents=True)
        atomic_json(backup/'config.json', old)
        (backup/'crontab.txt').write_text(cron)
        copied = ['runner.py', 'main_control.py', 'main_service.py']
        for name in copied:
            if (root/'bin'/name).exists():
                shutil.copyfile(root/'bin'/name, backup/name)
        if (root/'runner-code-sha.txt').is_file():
            shutil.copyfile(root/'runner-code-sha.txt', backup/'runner-code-sha.txt')
        atomic_json(target, deployment)
        # Copy helper first, switch the installed Runner last. The lock excludes
        # old cron invocations throughout the migration.
        for name in ['main_control.py', 'main_service.py', 'runner.py']:
            pending = root/'bin'/(name+'.new')
            shutil.copyfile(source/name, pending)
            os.replace(pending, root/'bin'/name)
        (root/'runner-code-sha.txt').write_text(code_sha+'\n')
        config = dict(old, allowed_kinds=['smoke', 'experiment'], main_integration=True)
        atomic_json(root/'config.json', config)
        evidence = dict(schema_version=1, worker_id='server2', status='main_attached', code_sha=code_sha,
                        project=str(project), root=str(root), python=config['python'],
                        cron_before=dict(runner=any(MARKERS[0] in l and not l.lstrip().startswith('#') for l in cron.splitlines()),
                                         collector=any(MARKERS[1] in l and not l.lstrip().startswith('#') for l in cron.splitlines())),
                        old_allowed_kinds=old['allowed_kinds'],
                        collector_installed=(root/'bin/collect_retuned.py').is_file(),
                        existing_state_files=sorted(p.name for p in (root/'state').glob('*.json')),
                        bound_inputs=bound, deployment_sha256=digest(encoded(deployment)),
                        input_access='absolute bound paths; no project symlinks; detached output workspace',
                        observed_at=now(), training_started=False)
        atomic_json(root/'main-deployment-evidence.json', evidence)
        runner.publish({'workers/server2-main.json': encoded(evidence)})
        # No request is exposed to the old smoke-only executable. The exact job
        # and its durable queue state are created atomically after the opt-in.
        queued = Control(root).queue(deployment)
        schedule = merge_cron(cron, config['python'], root)
        subprocess.run(['crontab', '-'], input=schedule, text=True, check=True)
        print(json.dumps(dict(status='attached', task=queued, evidence=evidence), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    try:
        attach(a.root)
    except Exception as error:
        failure = dict(status='main_attach_failed', error=type(error).__name__+': '+str(error),
                       traceback=traceback.format_exc(), training_started=False, observed_at=now())
        root = Path(a.root)
        if root.is_dir():
            atomic_json(root/'main-attach-error.json', failure)
            try:
                Runner(root).publish({'workers/server2-main-install-error.json': encoded(failure)})
            except Exception:
                pass
        raise


if __name__ == '__main__':
    main()
