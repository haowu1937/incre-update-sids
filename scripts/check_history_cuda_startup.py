"""Check archived CUDA-fix metadata and launch one tensor through the controller.

No model, dataset, checkpoint or historical manifest is changed. The CUDA child
uses the same bootstrap as the sweep, with a temporary probe in place of its
training engine. This is a startup check, not a recommendation experiment.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_history_weight_sweep as controller

OLD_CONTROLLER = '1afee70be91aca93e27cc607708bb1de9a0c0ba14db6d98b377d456b612a1b47'
FIXED_CONTROLLER = '1cc112fcab6e2c582ac219c71305e3844f3b69d888e15a1342cb0a17a9c4e7fa'
UNCHANGED_ENGINE = '096b8b500446ea2057aa6d75e2f59b80ca55c85842c73c4f6260d77c9d80879a'

PROBE = '''import json, sys
from pathlib import Path
import torch
device = torch.device(sys.argv[sys.argv.index('--device') + 1])
output = Path(sys.argv[sys.argv.index('--result') + 1])
seed = int(sys.argv[sys.argv.index('--seed') + 1])
initialized_at_entry = torch.cuda.is_initialized()
if device.type == 'cuda' and not initialized_at_entry:
    raise RuntimeError('Controller did not initialize CUDA before engine entry')
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
value = torch.ones(1, device=device)
if device.type == 'cuda':
    torch.cuda.synchronize(device)
report = dict(execution_status='passed', device=str(device), seed=seed,
    initialized_at_engine_entry=initialized_at_entry, tensor_value=float(value.item()),
    torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
    training_started=False, checkpoint_loaded=False)
output.write_text(json.dumps(report, indent=2) + '\\n')
print(json.dumps(report), flush=True)
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_evidence(reportfile, repo=ROOT):
    reportfile = Path(reportfile).resolve()
    audit = read(reportfile)
    manifest = read(reportfile.parent / 'files_sha256.json')

    def checked(relative):
        path = (reportfile.parent / relative).resolve()
        if not path.is_relative_to(reportfile.parent) or manifest.get(relative) != digest(path):
            raise ValueError('Archived evidence checksum mismatch: ' + relative)
        return path

    checked('report.json')
    if audit['issues'] or any(not c['matches_workspace'] for c in audit['source_claims']):
        raise ValueError('Original audit reported unresolved issues or source mismatches')
    if audit['historical_execution_status'] != 'completed' or audit['selected_rho'] is not None:
        raise ValueError('Expected the completed sweep without a selected rho')
    records = [r['path'] for r in audit['evidence_files']
               if Path(r['path']).name.startswith('cuda_startup_fix_') and r['path'].endswith('.json')]
    if len(records) != 1:
        raise ValueError('Expected the single recorded CUDA fix from this audit')
    record_path = checked('evidence/' + records[0])
    record = read(record_path)
    plan_path = checked('evidence/' + str(Path(records[0]).parent / 'predeclared_plan.json'))
    backup_path = Path(record['backup_dir']).resolve() / 'predeclared_plan.json'
    if not backup_path.is_relative_to(Path(repo).resolve() / 'cuda_startup_fix_backups'):
        raise ValueError('Fix backup is outside the recorded backup tree')
    expected = dict(controller_before_sha256=OLD_CONTROLLER,
        controller_after_sha256=FIXED_CONTROLLER, unchanged_engine_sha256=UNCHANGED_ENGINE,
        training_objective_changed=False, seed_or_step_budget_changed=False,
        existing_preflight_and_checkpoints_preserved=True,
        plan_before_sha256=digest(backup_path), plan_after_sha256=digest(plan_path))
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('CUDA fix record differs from source/plan evidence')
    before, after = read(backup_path), read(plan_path)
    predicted = copy.deepcopy(before)
    predicted['sources']['run_history_weight_sweep.py'] = FIXED_CONTROLLER
    if before['sources']['run_history_weight_sweep.py'] != OLD_CONTROLLER or predicted != after:
        raise ValueError('Historical plan differs beyond the controller source hash')
    for name, expected_hash in (('run_history_weight_sweep.py', FIXED_CONTROLLER),
                                ('history_weight_update.py', UNCHANGED_ENGINE)):
        if digest(checked('sources/workspace/' + name)) != expected_hash:
            raise ValueError('Archived workspace source differs: ' + name)
    endpoints = audit['endpoints']
    if (len(endpoints) != 15 or not all(r['summary_matches'] for r in endpoints)
            or sum(r['difference'] > 0 for r in endpoints) != 14):
        raise ValueError('Unexpected archived endpoint consistency')
    return dict(status='metadata_consistent', observed_head=audit['observed_head'],
        fix_record_sha256=digest(record_path), plan_change='controller source hash only',
        engine_unchanged=True, comparisons=15, history_worse_than_plain=14,
        limitation='File/metadata consistency; checkpoints and historical execution not independently reverified')


def probe(output, device, seed):
    output = Path(output)
    probe_path = output / 'startup_probe.py'
    probe_path.write_text(PROBE)
    original = controller.ENGINE
    try:
        controller.ENGINE = probe_path
        controller.command(['--device', device, '--seed', str(seed),
                            '--result', str(output / 'child_result.json')], output / 'child.log')
    finally:
        controller.ENGINE = original
    return read(output / 'child_result.json')


def main(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), commit=subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        controller_sha256=digest(ROOT / 'run_history_weight_sweep.py'),
        engine_sha256=digest(ROOT / 'history_weight_update.py'))
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    try:
        if config['controller_sha256'] != FIXED_CONTROLLER or config['engine_sha256'] != UNCHANGED_ENGINE:
            raise ValueError('Expected fixed controller and unchanged engine')
        evidence = check_evidence(args.audit_report)
        child = probe(out, args.device, args.seed)
        result = dict(execution_status='passed', provenance=evidence, startup=child, training_started=False)
    except Exception as exc:
        result = dict(execution_status='failed', error=str(exc), training_started=False)
        raise
    finally:
        if 'result' in locals():
            (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
            text = json.dumps(result, indent=2) + '\n'
            (out / 'compact.txt').write_text(text)
            print(text, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-report', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=2026)
    main(p.parse_args())
