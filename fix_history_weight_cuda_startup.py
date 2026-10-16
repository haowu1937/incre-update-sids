"""Initialize each CUDA child before running the unchanged weight-sweep engine.

Run from the project root. Only the controller and its recorded source hash
change. Every checkpoint, training engine, evaluation and experimental setting
is preserved. All changes are validated first and backed up before replacement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

PATCH = {'old_sha256': '1afee70be91aca93e27cc607708bb1de9a0c0ba14db6d98b377d456b612a1b47', 'new_sha256': '1cc112fcab6e2c582ac219c71305e3844f3b69d888e15a1342cb0a17a9c4e7fa', 'old': "            process = subprocess.Popen([sys.executable, '-u', '-B', str(ENGINE), *map(str,args)],\n                stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)", 'new': '            launch = [sys.executable, \'-u\', \'-B\', str(ENGINE), *map(str,args)]\n            # reset_peak_memory_stats does not initialize CUDA for an explicit\n            # device index. Initialize in this child before loading the unchanged\n            # engine; its seed_all still runs before any model construction.\n            if \'--device\' in args and str(args[args.index(\'--device\')+1]).startswith(\'cuda\'):\n                bootstrap = "import os,runpy,sys\\nos.environ.setdefault(\'CUBLAS_WORKSPACE_CONFIG\', \':4096:8\')\\nimport torch\\nengine=sys.argv.pop(1)\\nvalue=sys.argv[sys.argv.index(\'--device\')+1]\\ntorch.cuda.set_device(torch.device(value))\\ntorch.cuda.init()\\nrunpy.run_path(engine,run_name=\'__main__\')\\n"\n                launch = [sys.executable, \'-u\', \'-B\', \'-c\', bootstrap, str(ENGINE), *map(str,args)]\n            process = subprocess.Popen(launch,\n                stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)'}
ENGINE_SHA256 = '096b8b500446ea2057aa6d75e2f59b80ca55c85842c73c4f6260d77c9d80879a'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def atomic(path, value):
    temporary=path.with_name(path.name+'.cuda-fix.tmp')
    with temporary.open('wb') as handle:
        handle.write(value);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def apply(repo, run_dir):
    repo=Path(repo).resolve();run_dir=Path(run_dir).resolve()
    target=repo/'run_history_weight_sweep.py';engine=repo/'history_weight_update.py'
    planfile=run_dir/'predeclared_plan.json'
    if not target.is_file() or not planfile.is_file():
        raise ValueError('Controller or existing sweep plan is missing; no files changed')
    if not engine.is_file() or digest(engine.read_bytes())!=ENGINE_SHA256:
        raise ValueError('Training engine differs from the delivered weight-sweep version; no files changed')
    original=target.read_bytes();old_hash=digest(original)
    if old_hash not in (PATCH['old_sha256'],PATCH['new_sha256']):
        raise ValueError('Controller differs from the expected source; no files changed')
    old_plan=planfile.read_bytes();plan=json.loads(old_plan)
    if plan.get('stage')!='bounded_history_weight_sweep_v1':
        raise ValueError('Not a compatible existing weight-sweep plan; no files changed')
    if plan['sources']['run_history_weight_sweep.py'] not in (PATCH['old_sha256'],PATCH['new_sha256']):
        raise ValueError('Recorded controller version is unexpected; no files changed')
    for name,value in plan['sources'].items():
        if name=='run_history_weight_sweep.py':continue
        if digest((repo/name).read_bytes())!=value:
            raise ValueError('Recorded source differs: '+name+'; no files changed')
    if old_hash==PATCH['old_sha256']:
        source=original.decode()
        if source.count(PATCH['old'])!=1:raise ValueError('Patch location differs; no files changed')
        revised=source.replace(PATCH['old'],PATCH['new']).encode()
    else:revised=original
    if digest(revised)!=PATCH['new_sha256']:
        raise ValueError('Revised controller checksum differs; no files changed')
    plan['sources']['run_history_weight_sweep.py']=PATCH['new_sha256']
    if old_hash==PATCH['new_sha256'] and json.loads(old_plan)==plan:
        print('CUDA startup fix already applied; existing run can resume.')
        return dict(status='already_applied')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=repo/'cuda_startup_fix_backups'/stamp;backup.mkdir(parents=True)
    shutil.copy2(target,backup/target.name);shutil.copy2(planfile,backup/'predeclared_plan.json')
    record=dict(fix='explicit CUDA initialization before unchanged training engine',
        controller_before_sha256=old_hash,controller_after_sha256=PATCH['new_sha256'],
        unchanged_engine_sha256=ENGINE_SHA256,plan_before_sha256=digest(old_plan),
        plan_after_sha256=digest(encoded(plan)),backup_dir=str(backup),
        training_objective_changed=False,seed_or_step_budget_changed=False,
        existing_preflight_and_checkpoints_preserved=True)
    # Matching old/new controller hashes permit safely re-running this installer
    # if interrupted between the two atomic replacements.
    atomic(target,revised);atomic(planfile,encoded(plan))
    atomic(run_dir/('cuda_startup_fix_'+stamp+'.json'),encoded(record))
    print(json.dumps(dict(status='applied',backup_dir=str(backup),run_dir=str(run_dir),
        training_engine_unchanged=True,existing_preflight_reusable=True),indent=2))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.');parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();apply(args.repo,args.run_dir)


if __name__=='__main__':main()
