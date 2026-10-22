#!/usr/bin/env python3
"""Accept the completed real Beauty experiment without training or new predictions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runner import atomic_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def contained_file(base, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Artifact path escapes input directory')
    result = base/p
    if result.is_symlink() or not result.is_file() or base not in result.resolve().parents:
        raise ValueError('Unsafe or missing artifact '+name)
    return result


def compare(actual, expected):
    """Require all declared observations; floats allow only aggregation roundoff."""
    if isinstance(expected, dict):
        for k, v in expected.items():
            compare(actual[k], v)
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Observed value mismatch')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError('Observed value mismatch')


def accept(deployment, output):
    import torch
    import evaluate_historical_future as entry
    from genrec import future_evaluation as f
    from genrec.future_protocol import summarize_records
    d = read(deployment)
    expected = read(ROOT/'docs/lab_buffer/history_future_expected.json')
    future = Path(d['future_dir'])
    original = Path(d['project'])
    if output.exists() or original == output or original in output.parents:
        raise ValueError('Require fresh independent output')
    actual_sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_sha != os.environ['LAB_BUFFER_CODE_SHA']:
        raise ValueError('Deployment/code SHA mismatch')
    before = {path: sha(path) for path in d['input_sha256']}
    if before != d['input_sha256']:
        raise ValueError('Bound inputs changed after deployment')
    report = read(future/'summary.json')
    compact = read(future/'compact.json')
    if compact != entry.compact_report(report):
        raise ValueError('Summary/compact mismatch')
    compare(compact, expected)
    for name, value in report['binding']['sources'].items():
        if sha(contained_file(ROOT, name)) != value:
            raise ValueError('Original source binding mismatch: '+name)
    for name, value in report['artifact_files_sha256'].items():
        if sha(contained_file(future, name)) != value:
            raise ValueError('Original output artifact checksum mismatch: '+name)
    config = read(future/'config.json')
    if config['config'] != report['config'] or any(report['binding'][k] != v for k, v in config['binding'].items()):
        raise ValueError('Future provenance/config mismatch')
    lock = read(future/'selection_locked.json')
    if (lock['binding'] != report['binding'] or lock['future_used_for_selection'] is not False
            or lock['selection_split'] != 'update_val' or lock['selection_metric'] != 'ndcg@10'
            or lock['update_optimizer_step'] != 2000
            or lock['checkpoint_sha256'] != sha(future/'historical/best.pth')
            or lock['checkpoint_sha256'] != compact['future']['finetuned']['model_sha256']):
        raise ValueError('Selected checkpoint/lock mismatch')
    if not torch.cuda.is_available():
        raise ValueError('Real acceptance requires the configured CUDA device')
    torch.cuda.init()
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    ex, retuned, _ = entry.load_context(SimpleNamespace(e1_dir=d['e1_dir'], retuned_dir=d['retuned_dir']))
    mapping, _, _ = entry.build_mapping(ex, retuned, device)
    for name in ('train_ledger', 'deployment_ledger'):
        if mapping[name] != read(future/(name+'.json')):
            raise ValueError('Real checkpoint/full SID ledger mismatch')
    checkpoint = torch.load(future/'historical/best.pth', map_location='cpu', weights_only=False)
    if (checkpoint['format'] != 'paired_update_tiger_selected_v1' or checkpoint['optimizer_step'] != 2000
            or checkpoint['binding'] != dict(report['binding'], arm='historical') or checkpoint['config'] != report['config']):
        raise ValueError('Actual selected TIGER checkpoint binding mismatch')
    models = (('frozen', ex.teacher), ('finetuned', checkpoint))
    for state, saved in models:
        model = f.TIGER(report['config'], None, mapping['tokenizers']['deployment']).to(device)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        if not all(torch.equal(v.cpu(), saved['state_dict'][k]) for k, v in model.state_dict().items()):
            raise ValueError('Actual TIGER reload mismatch')
        del model
    training = report['training']
    history = training['validation_history']
    last = torch.load(future/'historical/last.pth', map_location='cpu', weights_only=False)
    if (last['format'] != f.FORMAT or last['sampler']['optimizer_steps'] != 2000
            or last['sampler']['processed_examples'] != 255919
            or last['best_step'] != 2000 or last['validation_history'] != history
            or last['binding'] != checkpoint['binding'] or last['config'] != checkpoint['config']):
        raise ValueError('Actual final TIGER budget/selection history mismatch')
    if [r['step'] for r in history] != [0, 100, 250, 500, 1000, 2000]:
        raise ValueError('Unexpected selection checkpoints')
    chosen = max(history, key=lambda x: (x['metrics']['full_denominator']['ndcg@10'], -x['step']))
    if chosen['step'] != training['best_update_optimizer_step'] or chosen['metrics'] != training['best_update_validation']:
        raise ValueError('Selection is not earliest maximum Update probe NDCG@10')
    base_ids = [r['sample_id'] for rows in ex.rows.values() for r in rows]
    update_train = f.load_stage(ex, 'update_train', base_ids)
    update_val = f.load_stage(ex, 'update_val', base_ids+[r['sample_id'] for r in update_train])
    raw_future = f.load_stage(ex, 'future_test', base_ids+[r['sample_id'] for r in update_train+update_val])
    expected_ids = [(i, r['sample_id'], r['user'], r['item_seq'][-1], r['target_servable'])
                    for i, r in enumerate(raw_future)]
    identities = None
    for state in ('frozen', 'finetuned'):
        p = future/('historical_'+state+'_queries.jsonl')
        with p.open() as stream:
            rows = [json.loads(line) for line in stream]
        ids = [(r['source_row'], r['sample_id'], r['user'], r['target_raw_item_id'], r['target_servable']) for r in rows]
        if (ids != expected_ids or len({r['sample_id'] for r in rows}) != len(rows)
                or (identities is not None and ids != identities)):
            raise ValueError('Future query identities differ or repeat')
        identities = ids
        compare(report['future'][state], summarize_records(rows, report['future'][state]['full_denominator'].keys()))
    after = {path: sha(path) for path in before}
    if before != after:
        raise ValueError('An original input changed during acceptance')
    return dict(stage='sid_main_existing_future_acceptance', execution_status='completed',
                job_id=os.environ['LAB_BUFFER_JOB_ID'], code_sha=actual_sha,
                future_dir=str(future), retuned_dir=str(retuned), summary_sha256=sha(future/'summary.json'),
                training_started=False, optimizer_steps_this_task=0, new_future_predictions=False,
                original_inputs_unchanged=True, protected_input_count=len(before),
                input_manifest_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                checks=dict(source_and_artifact_bindings=True, real_rqvae_mapping_reload=True,
                            real_tiger_reload=True, selection_recomputed=True, query_aggregates_recomputed=True,
                            actual_tiger_budget=True, same_future_query_identities=True,
                            exact_returned_observations=True), compact=compact)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output).resolve()
    deployment = read(a.deployment)
    workspace = Path(deployment['runner_root'])/'workspaces'/os.environ['LAB_BUFFER_JOB_ID']
    if ROOT != workspace.resolve() or ROOT not in output.parents or output.exists():
        raise FileExistsError('Require fresh output inside this exact detached job workspace')
    # Always write failure evidence to this job's independent output.
    started = time.monotonic()
    migration = False
    try:
        if os.environ['LAB_BUFFER_JOB_ID'] == 'history-future-accept-002':
            # Explicit user-authorized scope migration, NOT a Future rerun or
            # a claim that the accepted job 001 failed. The legacy executable
            # permits this one unused argv/ID slot; all new jobs get new IDs.
            from gradient_bridge import migrate, migration_requested
            migration = migration_requested(a.deployment)
        if migration:
            result = migrate(ROOT, a.deployment, output)
        else:
            result = accept(a.deployment, output)
    except BaseException as error:
        atomic_json(output, dict(stage='history_gradient_scope_migration_v1' if migration else 'sid_main_existing_future_acceptance', execution_status='failed',
                    job_id=os.environ.get('LAB_BUFFER_JOB_ID'), training_started=False,
                    error=type(error).__name__+': '+str(error), traceback=traceback.format_exc(),
                    runtime_seconds=time.monotonic()-started))
        raise
    result['runtime_seconds'] = time.monotonic()-started
    if migration:
        raw = json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n'
        if len(raw.encode()) > 64*1024:
            raise ValueError('Diagnostic text exceeds the explicit mailbox limit')
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_suffix('.tmp')
        pending.write_text(raw)
        os.replace(pending, output)
    else:
        atomic_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('compact', 'input_sha256', 'gradient_logs')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
