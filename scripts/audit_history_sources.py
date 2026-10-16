"""Read-only source/evidence collection for an existing history-weight sweep.

Standard library only. Never imports training code, opens checkpoints, executes
path files, edits old manifests, or starts CUDA. Hash claims are evidence to
review, not proof that the claimed source bytes actually executed.
"""
import argparse
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

BASELINE = 'e1a4cf42f40a67fd750e80ae7c3d0e9549eb0188'
SWEEP = 'results/incremental/Beauty/weight_sweep_20261015T161912Z'
ROOTS = ('run_history_weight_sweep.py', 'history_weight_update.py',
         'history_update_multiseed.py', 'audit_history_pair.py',
         'history_aware_frozen_encoder.py', 'history_aware_update_v2.py')
SOURCE_KEYS = dict(weight_source_sha256=ROOTS[1],
    previous_multiseed_source_sha256=ROOTS[2], multiseed_source_sha256=ROOTS[2],
    independent_auditor_sha256=ROOTS[3], audit_source_sha256=ROOTS[3],
    frozen_source_sha256=ROOTS[4], v2_source_sha256=ROOTS[5],
    joint_source_sha256=ROOTS[5])
MAX_METADATA_BYTES = 8 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args, required=True):
    p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True)
    if required and p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace').strip())
    return p.stdout if p.returncode == 0 else None


def confined(repo, path):
    path = Path(path)
    path = (repo / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(repo):
        raise ValueError('Path outside repository: ' + str(path))
    return path


def dependencies(name, data, exists):
    """Resolve static repository imports without importing modules."""
    result = set()
    for node in ast.walk(ast.parse(data, filename=name)):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                parent = name.split('/')[:-node.level]
                base = '.'.join(parent + ([base] if base else []))
            modules = [base] + [base + '.' + a.name for a in node.names]
        for module in modules:
            parts = module.split('.')
            candidates = ['/'.join(parts) + '.py', '/'.join(parts) + '/__init__.py']
            candidates += ['/'.join(parts[:i]) + '/__init__.py' for i in range(1, len(parts))]
            result.update(p for p in candidates if exists(p))
    return result


def source_claims(value, location='', stage=''):
    """Extract explicit source bindings, including nested experiment bindings."""
    if not isinstance(value, dict):
        return
    stage = value.get('stage', stage)
    for key, item in value.items():
        here = location + '/' + key
        if key in SOURCE_KEYS and isinstance(item, str):
            yield here, SOURCE_KEYS[key], item
        if key in ('sources', 'source_files_sha256', 'evaluation_sources') and isinstance(item, dict):
            for name, digest in item.items():
                target = name
                if key == 'evaluation_sources':
                    target = dict(auditor=ROOTS[3], legacy=ROOTS[4], multiseed=(
                        ROOTS[1] if stage == 'historical_weight_independent_comparison_v1' else ROOTS[2])).get(name, name)
                if target.endswith('.py') and isinstance(digest, str):
                    yield here + '/' + name, target, digest
        yield from source_claims(item, here, stage)


def collect(args):
    repo = Path(args.repo).resolve()
    baseline = git(repo, 'rev-parse', '--verify', args.baseline_sha + '^{commit}').decode().strip()
    sweep = confined(repo, args.sweep_root)
    out = Path(args.output).resolve()
    # Never put collection files inside the historical run being inspected.
    if out.is_relative_to(sweep) or sweep.is_relative_to(out):
        raise ValueError('Audit output and historical sweep must be separate')
    out.mkdir(parents=True, exist_ok=False)
    issues, documents, evidence, claims = [], {}, [], []
    status = git(repo, 'status', '--porcelain=v1', '--untracked-files=all').decode()
    head = git(repo, 'rev-parse', 'HEAD').decode().strip()
    branch = git(repo, 'branch', '--show-current').decode().strip()

    def save(name, data):
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def metadata(path, required=False):
        try:
            path = confined(repo, path)
            label = path.relative_to(repo).as_posix()
            if label in documents:
                return documents[label]
            if not path.is_file():
                if required:
                    issues.append('missing: ' + label)
                return None
            if path.stat().st_size > MAX_METADATA_BYTES:
                issues.append('metadata too large; skipped: ' + label)
                return None
            data = path.read_bytes()
            save('evidence/' + label, data)
            evidence.append(dict(path=label, sha256=sha(data), bytes=len(data)))
            value = json.loads(data) if path.suffix == '.json' else data.decode(errors='replace')
            documents[label] = value
            return value
        except (OSError, ValueError) as exc:
            issues.append('metadata unreadable: ' + str(path) + ': ' + str(exc))
            return None

    plan = metadata(sweep / 'predeclared_plan.json', True) or {}
    summary = metadata(sweep / 'summary.json', True) or {}
    selection = metadata(sweep / 'selection.json', True) or {}
    metadata(sweep / 'compact.txt', True)
    if not all(isinstance(v, dict) for v in (plan, summary, selection)):
        raise ValueError('Root plan/summary/selection must be JSON objects')
    planpath = sweep / 'predeclared_plan.json'
    if selection.get('plan_sha256') and planpath.is_file():
        if selection['plan_sha256'] != sha(planpath.read_bytes()):
            issues.append('selection plan_sha256 differs from current plan bytes')
    if summary.get('selected_rho') != selection.get('selected_rho'):
        issues.append('summary/selection selected_rho differs')
    for pattern in ('training_evaluations/*/seed*/summary.json',
                    'training_evaluations/*/seed*.request.json',
                    'checkpoints/rho_*/seed*/attempt_*/summary.json',
                    'checkpoints/rho_*/seed*/attempt_*/config.json', '*plan*.json'):
        for path in sorted(sweep.glob(pattern)):
            metadata(path)
    previous = plan.get('previous_root')
    if previous:
        for i, seed in enumerate(plan.get('seeds', [])):
            p = Path(previous) / ('seed' + str(seed) + '_evaluation') / 'summary.json'
            ref = metadata(p, True)
            hashes = plan.get('reference_sha256', [])
            if ref and i < len(hashes) and sha(confined(repo, p).read_bytes()) != hashes[i]:
                issues.append('reference summary hash differs: seed ' + str(seed))
    if plan.get('e1_dir'):
        metadata(Path(plan['e1_dir']) / 'summary.json', True)
    fixes = sorted(set(repo.glob('cuda_startup_fix_*.json')) |
                   set(sweep.rglob('cuda_startup_fix_*.json')))
    for p in fixes:
        metadata(p)
    if not fixes:
        issues.append('CUDA startup fix record not found in repository root or sweep tree')
    attempts = [dict(path=p.relative_to(repo).as_posix(),
                     has_summary=(p / 'summary.json').is_file(),
                     has_checkpoint=(p / 'last.pth').is_file())
                for p in sorted(sweep.glob('checkpoints/rho_*/seed*/attempt_*')) if p.is_dir()]
    # Keep bounded log tails, including the failed launch. No per-query files.
    logfiles = set(repo.glob(sweep.name + '*.log')) | set(sweep.glob('checkpoints/rho_*/seed*/attempt_*.log'))
    for p in sorted(logfiles):
        p = confined(repo, p)
        with p.open('rb') as handle:
            handle.seek(max(0, p.stat().st_size - 65536))
            save('log_tails/' + p.relative_to(repo).as_posix() + '.tail', handle.read())
    # Keep exact original source bytes and baseline bytes, including local fixes.
    baseline_files = set(git(repo, 'ls-tree', '-r', '--name-only', baseline).decode().splitlines())
    cache = {}

    def old(name):
        if name not in cache:
            cache[name] = git(repo, 'show', baseline + ':' + name, required=False) if name in baseline_files else None
        return cache[name]

    pending = set(ROOTS)
    pending.update(p.relative_to(repo).as_posix() for p in repo.glob('fix_history_weight_cuda_startup*.py'))
    for label, doc in documents.items():
        for field, name, digest in source_claims(doc):
            claims.append(dict(document=label, field=field, source=name, claimed_sha256=digest))
            pending.add(name)
    sources = {}
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        try:
            path = confined(repo, name)
            current = path.read_bytes() if path.is_file() else None
            before = old(name)
            row = dict(workspace_sha256=sha(current) if current is not None else None,
                       baseline_sha256=sha(before) if before is not None else None)
            sources[name] = row
            for version, data in (('workspace', current), ('baseline', before)):
                if data is None:
                    continue
                save('sources/' + version + '/' + name, data)
                try:
                    deps = dependencies(name, data, lambda p: (repo / p).is_file() or p in baseline_files)
                    row[version + '_static_imports'] = sorted(deps)
                    pending.update(deps - sources.keys())
                except (SyntaxError, UnicodeError) as exc:
                    issues.append('source parse failed: ' + name + ': ' + str(exc))
            if current != before:
                diff = ''.join(difflib.unified_diff(
                    (before or b'').decode(errors='replace').splitlines(True),
                    (current or b'').decode(errors='replace').splitlines(True),
                    fromfile='baseline/' + name, tofile='workspace/' + name))
                save('diffs/' + name + '.diff', diff.encode())
        except (OSError, ValueError) as exc:
            issues.append('source unreadable: ' + name + ': ' + str(exc))
    for row in claims:
        pair = sources.get(row['source'], {})
        row['matches_workspace'] = row['claimed_sha256'] == pair.get('workspace_sha256')
        row['matches_baseline'] = row['claimed_sha256'] == pair.get('baseline_sha256')
    endpoints = []
    for level in summary.get('levels', []):
        rho = level['rho']
        tag = ('%g' % rho).replace('.', 'p')
        for row in level.get('seeds', []):
            seed = row['seed']
            label = (sweep / 'training_evaluations' / ('rho_' + tag) /
                     ('seed' + str(seed)) / 'summary.json').relative_to(repo).as_posix()
            report = documents.get(label)
            if not report:
                issues.append('missing training comparison: rho=%s seed=%s' % (rho, seed))
                continue
            try:
                group = report['groups']['anchor_train']
                metrics = dict(plain_H=group['plain']['positive_damage'],
                    history_H=group['history_aware']['positive_damage'],
                    difference=group['paired']['history_minus_plain_positive_damage'],
                    RQ_MSE=report['rq_mse']['history'])
                consistent = all(abs(metrics[k] - row[k]) <= 1e-10 for k in metrics)
                endpoints.append(dict(rho=rho, seed=seed, summary_matches=consistent, **metrics))
                if not consistent:
                    issues.append('endpoint metrics differ: rho=%s seed=%s' % (rho, seed))
            except (KeyError, TypeError) as exc:
                issues.append('incomplete comparison: ' + label + ': ' + str(exc))
    packages = {}
    for name in ('torch', 'numpy', 'transformers', 'scikit-learn', 'sentence-transformers'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    changed = sorted(k for k, v in sources.items() if v['workspace_sha256'] != v['baseline_sha256'])
    mismatches = [r for r in claims if not r['matches_workspace']]
    report = dict(stage='history_source_evidence_collection_v1', execution_status='collected',
        collected_at_utc=datetime.now(timezone.utc).isoformat(),
        repo=str(repo), baseline_sha=baseline, observed_head=head, observed_branch=branch,
        delivery_sha=args.delivery_sha, collector_sha256=sha(Path(__file__).read_bytes()),
        workspace_status=status, sweep_root=str(sweep), seed=None,
        training_started=False, checkpoints_opened=False, historical_artifacts_modified=False,
        observed_environment=dict(python=platform.python_version(), packages=packages),
        historical_execution_status=summary.get('execution_status'), selected_rho=summary.get('selected_rho'),
        changed_source_paths=changed, sources=sources, source_claims=claims,
        endpoints=endpoints, checkpoint_attempt_inventory=attempts,
        evidence_files=evidence, issues=issues,
        limits=['Static repository imports plus explicit source bindings; dynamic imports may be absent.',
               'No checkpoint/data/per-query checksum verification or metric recomputation.',
               'Current files and saved hash claims cannot prove which bytes executed historically.',
               'collected does not mean provenance is verified; inspect diffs and fix records.'])
    save('report.json', (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())
    lines = ['execution_status = collected | training_started = False',
        'observed_branch = ' + branch + ' | observed_head = ' + head,
        'baseline = ' + baseline, 'workspace_dirty = ' + str(bool(status)),
        'historical_execution_status = ' + str(report['historical_execution_status']),
        'selected_rho = ' + str(report['selected_rho']),
        'source_files = %s | changed_vs_baseline = %s | source_claim_mismatches_vs_workspace = %s' % (
            len(sources), len(changed), len(mismatches)),
        'training_comparisons_found = %s | history_worse_than_plain = %s' % (
            len(endpoints), sum(r['difference'] > 0 for r in endpoints)),
        'CUDA_fix_records = ' + str(len(fixes)), 'issues = ' + str(len(issues))]
    lines += ['changed: ' + p for p in changed]
    lines += ['issue: ' + x for x in issues]
    lines += ['Source claims are recorded, not historical execution proof.', 'report = ' + str(out / 'report.json')]
    save('compact.txt', ('\n'.join(lines) + '\n').encode())
    manifest = {p.relative_to(out).as_posix(): sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
    save('files_sha256.json', (json.dumps(manifest, indent=2) + '\n').encode())
    print('\n'.join(lines))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.')
    p.add_argument('--sweep-root', default=SWEEP)
    p.add_argument('--baseline-sha', default=BASELINE)
    p.add_argument('--delivery-sha', default=None)
    p.add_argument('--output', required=True)
    return p


if __name__ == '__main__':
    collect(parser().parse_args())
