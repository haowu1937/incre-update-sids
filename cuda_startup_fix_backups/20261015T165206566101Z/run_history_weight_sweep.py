"""Bounded, training-selected rho sweep. Seeds run in separate subprocesses.

No holdout/validation metric is read by the selection function. Higher weights
restart from Base, preserving the matched 100-step Plain control at each seed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

SEEDS = [2026, 2027, 2028]
RHOS = [0.1, 0.3, 1.0, 3.0, 10.0]
STEPS = 100
ENGINE = Path(__file__).resolve().with_name('history_weight_update.py')
ACTIVE = set()
PROCESS_LOCK = threading.Lock()
ABORT = threading.Event()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def tag(rho):
    return ('%g' % rho).replace('.', 'p')


def stats(values):
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))


def level_result(rho, reports):
    """Only anchor_train and reconstruction are inputs; never held-out metrics."""
    if sorted(r['update_seed'] for r in reports) != SEEDS:
        raise ValueError('Every level requires exactly all three seeds')
    details = []
    for r in sorted(reports, key=lambda v: v['update_seed']):
        if not r.get('eligible', True):
            details.append(dict(seed=r['update_seed'], eligible=False, reason=r['reason']))
            continue
        g = r['groups']['anchor_train']
        p, q = g['plain']['positive_damage'], g['history_aware']['positive_damage']
        delta = g['paired']['history_minus_plain_positive_damage']
        ci = g['paired']['positive_damage_cluster_ci95']
        mse_p, mse_q = r['rq_mse']['plain'], r['rq_mse']['history']
        values = [p, q, delta, *ci, mse_p, mse_q]
        if not all(math.isfinite(x) for x in values) or p <= 0 or mse_p <= 0:
            raise ValueError('Invalid training comparison')
        # These are exploratory, training-set criteria, not a significance claim
        # corrected for all weights tested.
        quality = mse_q <= mse_p * 1.01
        passed = quality and delta <= -0.01*p and ci[1] < 0
        details.append(dict(seed=r['update_seed'], eligible=True, plain_H=p,
            history_H=q, difference=delta, relative_reduction=(p-q)/p,
            training_user_cluster_ci95=ci, RQ_MSE=mse_q, Plain_RQ_MSE=mse_p,
            reconstruction_guard_passed=quality, training_criterion_passed=passed))
    valid = [r for r in details if r['eligible']]
    return dict(rho=rho, seeds=details, all_seeds_eligible=len(valid)==3,
        training_level_passed=len(valid)==3 and all(r['training_criterion_passed'] for r in valid),
        paired_training_H_difference=stats([r['difference'] for r in valid]) if len(valid)==3 else None,
        holdout_used_for_selection=False)


def select(levels):
    # Two adjacent predeclared weights must both pass, with all three seeds.
    for left, right in zip(levels, levels[1:]):
        if left['training_level_passed'] and right['training_level_passed']:
            return right['rho']
    return None


def checked_summary(directory):
    directory = Path(directory)
    r = read(directory/'summary.json')
    for name, expected in r['artifact_files_sha256'].items():
        if digest(directory/name) != expected:
            raise ValueError('Artifact checksum mismatch: '+str(directory/name))
    return r


def command(args, log):
    """File-backed output avoids pipe deadlocks and keeps long logs off screen."""
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', USE_TF='0', USE_FLAX='0')
    env.setdefault('OMP_NUM_THREADS', '1');env.setdefault('OPENBLAS_NUM_THREADS', '1')
    with log.open('w') as handle:
        with PROCESS_LOCK:
            if ABORT.is_set():raise RuntimeError('Another seed failed; remaining jobs cancelled')
            process = subprocess.Popen([sys.executable, '-u', '-B', str(ENGINE), *map(str,args)],
                stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            ACTIVE.add(process)
        try:
            code = process.wait()
        except BaseException:
            import signal
            os.killpg(process.pid, signal.SIGTERM);process.wait()
            raise
        finally:
            with PROCESS_LOCK:ACTIVE.discard(process)
    if code:
        raise RuntimeError('Job exited '+str(code)+'; inspect '+str(log))


def parallel(jobs, workers):
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(fn):label for label,fn in jobs}
        heartbeat=time.monotonic()
        try:
            while pending:
                ready=[future for future in pending if future.done()]
                if time.monotonic()-heartbeat>=30:
                    print('Running: '+', '.join(pending.values()),flush=True);heartbeat=time.monotonic()
                if not ready:
                    time.sleep(1)
                    continue
                for future in ready:
                    label=pending.pop(future)
                    results.append(future.result())
                    print('Finished '+label,flush=True)
        except BaseException:
            import signal
            ABORT.set()
            with PROCESS_LOCK:
                for process in ACTIVE:
                    if process.poll() is None:
                        try:os.killpg(process.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
            for future in pending:future.cancel()
            raise
    return results


def evaluate(e1, plain, history, out, seed, device, train_only):
    expected=dict(e1=str(e1),plain=str(plain),history=str(history),seed=seed,
                  train_only=train_only,engine_sha256=digest(ENGINE))
    marker=Path(out).with_suffix('.request.json')
    if Path(out,'summary.json').exists():
        if not marker.is_file() or read(marker)!=expected:
            raise ValueError('Existing evaluation request differs')
        return checked_summary(out)
    if Path(out).exists():
        # Independent evaluation writes only after every group succeeds. An
        # interrupted write must be inspected, never silently overwritten.
        raise RuntimeError('Incomplete evaluation directory; choose a new run output: '+str(out))
    write(marker,expected)
    argv=['evaluate','--e1-dir',e1,'--plain-dir',plain,'--history-dir',history,
          '--output',out,'--update-seed',seed,'--expected-steps',STEPS,
          '--device',device,'--batch-size',32]
    if train_only:argv+=['--train-only']
    command(argv,Path(out).with_suffix('.log'))
    return checked_summary(out)


def train_seed(root, e1, seed, rho, device):
    """Resume interrupted runs in a new attempt directory; retain every attempt."""
    base=Path(root)/('rho_'+tag(rho))/('seed'+str(seed))
    attempts=sorted(p for p in base.glob('attempt_*') if p.is_dir()) if base.exists() else []
    resume=None
    for attempt in attempts:
        if (attempt/'summary.json').is_file():
            r=checked_summary(attempt)
            c=r['config']
            if (r['stage']!='historical_weight_update_v1' or r['update_seed']!=seed
                    or c['historical_v2']['history_gradient_ratio']!=rho
                    or r['requested_update_optimizer_steps']!=STEPS
                    or r['binding']['weight_source_sha256']!=digest(ENGINE)):
                raise ValueError('Existing training request/source differs')
            return attempt,r
        if (attempt/'last.pth').is_file():resume=attempt/'last.pth'
    out=base/('attempt_%03d'%len(attempts))
    argv=['train','--e1-dir',e1,'--output',out,'--mode','history-aware',
          '--update-seed',seed,'--device',device,'--max-steps',STEPS,
          '--history-gradient-ratio',rho,'--anchor-batch-size',32,'--temperature',0.1,
          '--log-every',10,'--save-every',10,'--history-progress-every',0]
    if resume:argv+=['--resume-from',resume]
    print('Start rho='+str(rho)+' seed='+str(seed)+' device='+device,flush=True)
    command(argv,out.with_suffix('.log'))
    return out,checked_summary(out)


def brief(result):
    lines=['execution_status = '+result['execution_status'],
        'seeds = [2026,2027,2028] | steps = 100 | encoder = frozen | gate = disabled',
        'rho seed Plain_H History_H difference RQ_MSE training_pass']
    for level in result['levels']:
        for r in level['seeds']:
            if not r['eligible']:
                lines.append(f"{level['rho']:g} {r['seed']} INELIGIBLE {r['reason']}")
            else:
                lines.append(f"{level['rho']:g} {r['seed']} {r['plain_H']:.9g} {r['history_H']:.9g} {r['difference']:+.9g} {r['RQ_MSE']:.9g} {r['training_criterion_passed']}")
    lines += ['selected_rho = '+str(result['selected_rho']),
        'stop_reason = '+result['stop_reason'],
        'training_stability_observed = '+str(result['selected_rho'] is not None),
        'holdout_used_for_weight_selection = False | ranking_evaluated = False']
    if result.get('selected_evaluation'):
        for group, values in result['selected_evaluation'].items():
            lines.append(group+' differences = '+str(values['differences'])+
                         ' | mean +/- sample_SD = '+str(values['statistics']))
        lines.append('heldout_all_seed_improvement_supported = '+str(result['heldout_all_seed_improvement_supported']))
    lines.append('Fixed Base/teacher/cohorts; Update sampling seeds only. Training-selected exploratory sweep.')
    return '\n'.join(lines)+'\n'


def run(args):
    if not 1 <= args.workers <= 3:raise ValueError('workers must be 1, 2 or 3')
    previous=Path(args.previous_root).resolve();root=Path(args.output).resolve()
    references=[]
    for seed in SEEDS:
        r=checked_summary(previous/('seed'+str(seed)+'_evaluation'))
        if (r['stage']!='multiseed_independent_history_comparison_v1' or r['update_seed']!=seed
                or r['update_optimizer_steps']!=STEPS
                or r['design']['historical_v2']['history_gradient_ratio']!=0.1):
            raise ValueError('Expected completed three-seed rho=0.1 experiment')
        references.append(r)
    for r in references[1:]:
        for key in ('common_binding','design','cohort_hashes','evaluation_sources'):
            if r[key]!=references[0][key]:raise ValueError('Fixed reference design differs: '+key)
    if len({r['content_batch_trace_sha256'] for r in references})!=3:
        raise ValueError('Reference seeds must have distinct batch traces')
    e1=Path(args.e1_dir).resolve()
    plan=dict(stage='bounded_history_weight_sweep_v1',seeds=SEEDS,rhos=RHOS,steps=STEPS,
        previous_root=str(previous),e1_dir=str(e1),
        reference_sha256=[digest(previous/('seed'+str(s)+'_evaluation')/'summary.json') for s in SEEDS],
        sources={name:digest(ENGINE.with_name(name)) for name in (
            ENGINE.name,Path(__file__).name,'history_update_multiseed.py','audit_history_pair.py',
            'history_aware_frozen_encoder.py','history_aware_update_v2.py')},
        selection='two adjacent rho levels; all 3 seeds training H reduction >=1% and paired user CI upper<0; per-seed RQ MSE<=1.01 Plain',
        holdout='evaluate only the selected weight after weight selection, never escalate based on holdout',
        maximum_rho=10,concurrency_not_part_of_numerical_design=True)
    planfile=root/'predeclared_plan.json'
    if planfile.exists():
        if read(planfile)!=plan:raise ValueError('Existing sweep plan/data/source differs')
    elif root.exists():raise FileExistsError('Output exists without a matching plan')
    else:write(planfile,plan)
    devices=args.devices
    def device_for(seed):
        value=devices[SEEDS.index(seed)%len(devices)]
        return 'cpu' if value=='cpu' else 'cuda:'+value
    # Read-only recheck of all reused endpoints using the independent train-only
    # oracle. This does not use the previous held-out results in selection.
    def preflight(r):
        seed=r['update_seed']
        return evaluate(e1,r['plain_dir'],r['history_dir'],
                        root/'training_evaluations'/'rho_0p1'/('seed'+str(seed)),seed,device_for(seed),True)
    refs=parallel([(f'preflight seed {r["update_seed"]}',lambda r=r:preflight(r)) for r in references],args.workers)
    levels=[level_result(0.1,refs)];selected=None
    result=dict(execution_status='running',levels=levels,selected_rho=None,stop_reason='in_progress',
                holdout_used_for_selection=False,ranking_evaluated=False)
    write(root/'summary.json',result)
    trained={}
    for rho in RHOS[1:]:
        def job(r):
            seed=r['update_seed'];directory,train=train_seed(root/'checkpoints',e1,seed,rho,device_for(seed))
            trained[(rho,seed)]=directory
            if train['execution_status']!='completed' or train['full_sid_export_status']!='completed':
                return dict(update_seed=seed,eligible=False,reason=train['stop_reason'])
            return evaluate(e1,r['plain_dir'],directory,
                root/'training_evaluations'/('rho_'+tag(rho))/('seed'+str(seed)),seed,device_for(seed),True)
        reports=parallel([(f'rho {rho:g} seed {r["update_seed"]}',lambda r=r:job(r)) for r in references],args.workers)
        level=level_result(rho,reports);levels.append(level)
        selected=select(levels)
        print('rho '+str(rho)+' all-seed training pass='+str(level['training_level_passed']),flush=True)
        result.update(selected_rho=selected)
        write(root/'summary.json',result);(root/'compact.txt').write_text(brief(result))
        if selected is not None:break
    result.update(execution_status='completed',selected_rho=selected,
                  stop_reason='two_adjacent_training_levels_passed' if selected is not None else 'rho_limit_reached_without_stable_training_improvement')
    # Freeze the selection before running even one held-out forward pass.
    selection=dict(selected_rho=selected,selected_by='anchor_train_only',levels=levels,
                   plan_sha256=digest(planfile),heldout_evaluation_cannot_change_selection=True)
    selectionfile=root/'selection.json'
    if selectionfile.exists() and read(selectionfile)!=selection:
        raise ValueError('Previously frozen weight selection differs')
    write(selectionfile,selection)
    if selected is not None:
        def held(r):
            seed=r['update_seed']
            return evaluate(e1,r['plain_dir'],trained[(selected,seed)],
                root/'selected_evaluation'/('seed'+str(seed)),seed,device_for(seed),False)
        held_reports=parallel([(f'selected evaluation seed {r["update_seed"]}',lambda r=r:held(r)) for r in references],args.workers)
        result['selected_evaluation']={}
        for group in ('anchor_train','anchor_holdout','base_validation'):
            ordered=sorted(held_reports,key=lambda r:r['update_seed'])
            values=[r['groups'][group]['paired']['history_minus_plain_positive_damage'] for r in ordered]
            result['selected_evaluation'][group]=dict(differences=values,statistics=stats(values),
                user_cluster_CIs=[r['groups'][group]['paired']['positive_damage_cluster_ci95'] for r in ordered])
        hold=result['selected_evaluation']['anchor_holdout']
        result['heldout_all_seed_improvement_supported']=all(x<0 for x in hold['differences']) and all(ci[1]<0 for ci in hold['user_cluster_CIs'])
    write(root/'summary.json',result);(root/'compact.txt').write_text(brief(result))
    print(brief(result),flush=True)
    print('Share only: '+str(root/'compact.txt'),flush=True)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-root',required=True);p.add_argument('--e1-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--devices',nargs='+',default=['0'],help='Physical CUDA indices, e.g. 0 or 0 1 2; default: 3 jobs sharing GPU 0')
    return p


if __name__=='__main__':run(parser().parse_args())
