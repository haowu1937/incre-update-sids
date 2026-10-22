"""Bounded, read-only analysis of the existing retuned training trace (stdlib)."""
import hashlib
import json
import math
from pathlib import Path
import statistics

MAX_LOG_BYTES = 32*1024*1024


def distribution(values):
    v = sorted(values)
    if not v:
        return dict(count=0, mean=None, min=None, median=None, p90=None, max=None)
    def q(p):
        x = (len(v)-1)*p
        a, b = int(x), min(int(x)+1, len(v)-1)
        return v[a]+(v[b]-v[a])*(x-a)
    return dict(count=len(v), mean=statistics.fmean(v), min=v[0], median=q(.5), p90=q(.9), max=v[-1])


def components(rows):
    result = {}
    for group in ('encoder', 'decoder', 'codebooks'):
        entries = [r['gradient_components'][group] for r in rows if 'gradient_components' in r]
        n = len(entries)
        cos = [r['cosine'] for r in entries if r['cosine'] is not None]
        result[group] = dict(probed_steps=n,
            rq_norm=distribution([r['rq_gradient_norm'] for r in entries]),
            history_norm=distribution([r['history_gradient_norm'] for r in entries]),
            weighted_history_over_rq=distribution([r['weighted_history_over_rq'] for r in entries
                                                  if r['weighted_history_over_rq'] is not None]),
            cosine=distribution(cos), cosine_defined_steps=len(cos),
            negative_cosine_fraction=sum(v < 0 for v in cos)/len(cos) if cos else None,
            zero_history_gradient_steps=sum(r['history_gradient_norm'] == 0 for r in entries),
            zero_history_gradient_fraction=sum(r['history_gradient_norm'] == 0 for r in entries)/n if n else None,
            near_zero_history_gradient_fraction=sum(r['history_gradient_norm'] <= 1e-12 for r in entries)/n if n else None)
    return result


def validate_rows(rows, steps, weight):
    if [r['step'] for r in rows] != list(range(1, steps+1)):
        raise ValueError('Training trace must cover each completed step exactly once')
    for r in rows:
        for key in ('history_penalty', 'anchor_mean_delta_sequence_nll', 'global_gradient_norm_before_clip'):
            if not isinstance(r.get(key), (float, int)) or not math.isfinite(r[key]):
                raise ValueError('Missing/nonfinite training metric '+key)
        if type(r.get('gradient_clipped')) is not bool:
            raise ValueError('Missing clipping observation')
        if 'gradient_components' not in r:
            continue
        for group in ('encoder', 'decoder', 'codebooks'):
            g = r['gradient_components'][group]
            for key in ('rq_gradient_norm', 'history_gradient_norm', 'weighted_history_gradient_norm'):
                if not math.isfinite(g[key]) or g[key] < 0:
                    raise ValueError('Invalid component norm')
            if not math.isclose(g['weighted_history_gradient_norm'], weight*g['history_gradient_norm'], rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError('Logged history coefficient differs from bound recipe')
            rn = g['rq_gradient_norm']
            if rn > 0 and not math.isclose(g['weighted_history_over_rq'], weight*g['history_gradient_norm']/rn, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError('Logged gradient ratio does not match norms')
            if g['cosine'] is not None and (not math.isfinite(g['cosine']) or not -1 <= g['cosine'] <= 1):
                raise ValueError('Invalid gradient cosine')
    return rows


def summarize(rows, steps, weight):
    validate_rows(rows, steps, weight)
    windows = []
    for start in range(1, steps+1, 100):
        selected = rows[start-1:min(start+99, steps)]
        windows.append(dict(first_step=start, last_step=selected[-1]['step'], records=len(selected),
            mean_H=statistics.fmean(r['history_penalty'] for r in selected),
            mean_delta_NLL=statistics.fmean(r['anchor_mean_delta_sequence_nll'] for r in selected)))
    return dict(steps=steps, history_weight=weight, gradient_probe_steps=[r['step'] for r in rows if 'gradient_components' in r],
        gradient_components=components(rows), component_windows=[dict(first_step=s, last_step=min(s+249,steps),
            components=components(rows[s-1:min(s+249,steps)])) for s in range(1,steps+1,250)],
        training_batch_windows=windows,
        clipping=dict(records=len(rows), clipped_steps=sum(r['gradient_clipped'] for r in rows),
            clipped_fraction=sum(r['gradient_clipped'] for r in rows)/len(rows),
            global_norm_before_clip=distribution([r['global_gradient_norm_before_clip'] for r in rows])),
        gradient_probes=[dict(step=r['step'], gradient_components=r['gradient_components'],
            global_gradient_norm_before_clip=r['global_gradient_norm_before_clip'], gradient_clipped=r['gradient_clipped'])
            for r in rows if 'gradient_components' in r],
        scope='Pre-update observations on changing anchor batches; gradient probes are a sampled subset of steps. No fixed-checkpoint H curve or causal update claim.')


def collect(directory, source):
    directory = Path(directory).resolve()
    source = Path(source)
    request = json.loads((directory/'request.json').read_text())
    report = json.loads((directory/'summary.json').read_text())
    if (report['execution_status'] != 'completed' or request['binding'] != report['binding']
            or request['settings'] != report['settings'] or request['settings']['max_steps'] != 1000
            or request['binding']['retuned_source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest()):
        raise ValueError('Original completed training/source binding mismatch')
    files, by_step = [], {}
    for path in sorted((directory/'historical').glob('attempt_*/metrics.jsonl')):
        if path.is_symlink() or directory not in path.resolve().parents:
            raise ValueError('Unsafe metrics path')
        before = path.stat()
        if before.st_size > MAX_LOG_BYTES:
            raise ValueError('Trace exceeds the declared read limit')
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Training trace changed during read')
        if raw and not raw.endswith(b'\n'):
            raise ValueError('Incomplete final training record')
        for line in raw.splitlines():
            row = json.loads(line)
            step = row['step']
            if step in by_step and by_step[step] != row:
                raise ValueError('Conflicting observations for the same optimizer step')
            by_step[step] = row
        files.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), stable_during_read=True))
    rows = [by_step[k] for k in sorted(by_step)]
    result = summarize(rows, 1000, request['settings']['history_weight'])
    result.update(stage='existing_historical_gradient_logs', execution_status='completed',
        retuned_dir=str(directory), files=files,
        original_training_source_sha256=request['binding']['retuned_source_sha256'],
        training_config=request['settings'], checkpoint_sha256=report['training']['historical']['artifact_files_sha256']['last.pth'],
        holdout_evaluated=False, base_validation_evaluated=False, future_evaluated=False, optimizer_steps_this_task=0)
    return result
