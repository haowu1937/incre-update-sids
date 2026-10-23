"""Frozen, standard-library protocol and train-only selection for the joint study."""
import hashlib
import json
import math
from pathlib import Path
import statistics

SCOPE = 'history-joint-convergence-v1'
JOB = 'history-joint-convergence-001'
PLAN = 'docs/lab_buffer/joint_convergence_plan.json'


def identifier(lr, weight, seed=2026):
    return ('lr'+format(lr, '.0e')+'_h'+format(weight, '.0e')+'_s'+str(seed)).replace('-', 'm').replace('+', 'p')


def matrix(plan, seed=2026):
    return [dict(arm_id=identifier(lr, w, seed), lr=lr, history_weight=w, seed=seed)
            for lr in plan['learning_rates'] for w in plan['history_weights']]


def validate_plan(p):
    expected = dict(schema_version=1, scope_id=SCOPE, learning_rates=[3e-5, 1e-4, 3e-4],
        history_weights=[0., 1e-4, 1e-3, 1e-2], seed=2026, confirmation_seed=2027,
        screen_steps=1500, final_steps=3000, warmup_steps=50, decay_end_step=2500,
        final_lr_ratio=.1, calibration_steps=100, max_optimizer_steps=30400,
        temperature=.5, max_grad_norm=.1, anchor_batch_size=128, batch_size=256,
        max_parallel=2, gpu_memory_limit_mib=24576, min_throughput_gain=1.1,
        holdout_evaluation=False, base_validation_evaluation=False, future_evaluation=False,
        tiger_training_steps=0, fixed_anchor_queries=128, gradient_every=25,
        save_every=50, eval_batch_size=32, deadline_seconds=86400)
    if p != expected:
        raise ValueError('Joint study manifest differs from the fixed reviewed protocol')
    return p


def read_plan(root):
    return validate_plan(json.loads((Path(root)/PLAN).read_text()))


def lr_at(step, peak, plan):
    if not 1 <= step <= plan['final_steps']:
        raise ValueError('Step outside declared LR schedule')
    warm, end, floor = plan['warmup_steps'], plan['decay_end_step'], plan['final_lr_ratio']
    if step <= warm:
        return peak*step/warm
    if step >= end:
        return peak*floor
    return peak*(floor+(1-floor)*(1+math.cos(math.pi*(step-warm)/(end-warm)))/2)


def mean(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Missing/nonfinite training observations')
    return statistics.fmean(values)


def window(summary, tail=False):
    rows = [r for r in summary['curve'] if r['epoch_complete'] and
            (not tail or 2500 < r['step'] <= 3000)]
    if len(rows) < 8:
        raise ValueError('Need eight completed-epoch observations')
    rows = rows[-8:]
    return dict(rq=mean([r['rq']['total_loss'] for r in rows]),
        H=mean([r['anchor_train']['H'] for r in rows]),
        new_mse=mean([r['groups']['update_train_first_seen']['reconstruction_loss'] for r in rows]),
        steps=[r['step'] for r in rows])


def quality(candidate, plain, tail=False):
    a, b = window(candidate, tail), window(plain, tail)
    if a['steps'] != b['steps']:
        raise ValueError('Paired monitoring steps differ')
    initial = candidate['initial']
    base_rq = initial['rq']['total_loss']
    base_new = initial['groups']['update_train_first_seen']['reconstruction_loss']
    benefit = a['H'] <= .95*b['H'] and b['H'] > 1e-8
    checks = dict(rq_below_initial=a['rq'] < base_rq,
        rq_within_plain=a['rq'] <= 1.02*b['rq'], H_lower_than_plain=benefit,
        new_item_adaptation=a['new_mse'] <= .9*base_new)
    violation = (max(a['rq']/base_rq-1, 0)+max(a['rq']/(1.02*b['rq'])-1, 0)
        +max((a['H']-.95*b['H'])/max(.95*b['H'], .001), 0)
        +max(a['new_mse']/(.9*base_new)-1, 0))
    return dict(candidate=a, plain=b, checks=checks, passed=all(checks.values()), violation=violation)


def stability(summary):
    rows = [r for r in summary['curve'] if r['epoch_complete'] and 2500 < r['step'] <= 3000][-8:]
    if len(rows) != 8:
        return dict(passed=False, reason='insufficient_constant_lr_tail')
    rq, hs = [[r['rq']['total_loss'] for r in rows], [r['anchor_train']['H'] for r in rows]]
    rq_drift = abs(mean(rq[4:])-mean(rq[:4]))/mean(rq)
    h_drift = abs(mean(hs[4:])-mean(hs[:4]))
    cv = statistics.pstdev(rq)/mean(rq)
    span = max(hs)-min(hs)
    checks = dict(rq_drift=rq_drift <= .02, rq_cv=cv <= .03,
        H_drift=h_drift <= max(.001, .02*mean(hs)), H_span=span <= max(.005, .1*mean(hs)))
    return dict(passed=all(checks.values()), checks=checks, rq_relative_drift=rq_drift,
        rq_cv=cv, H_absolute_drift=h_drift, H_span=span, steps=[r['step'] for r in rows])


def select_screen(summaries):
    controls = {(s['case']['lr'], s['case']['seed']): s for s in summaries
                if s['status']=='completed' and s['case']['history_weight']==0}
    candidates = []
    for s in summaries:
        key = (s['case']['lr'], s['case']['seed'])
        if s['status'] != 'completed' or not s['case']['history_weight'] or key not in controls:
            continue
        q = quality(s, controls[key])
        candidates.append(dict(arm_id=s['case']['arm_id'], control=controls[key]['case']['arm_id'],
            lr=key[0], weight=s['case']['history_weight'], **q))
    good = sorted([r for r in candidates if r['passed']], key=lambda r:(r['candidate']['H'],r['candidate']['rq'],r['lr'],r['weight']))[:2]
    frontier = [r for r in candidates if not any(
        other['candidate']['H'] <= r['candidate']['H'] and other['candidate']['rq'] <= r['candidate']['rq']
        and (other['candidate']['H'] < r['candidate']['H'] or other['candidate']['rq'] < r['candidate']['rq'])
        for other in candidates)]
    rest = sorted([r for r in frontier if r not in good],
        key=lambda r:(r['violation'],r['candidate']['H'],r['candidate']['rq'],r['lr'],r['weight']))
    selected = good+rest[:2-len(good)]
    return dict(ranking=candidates, selected=selected,
        arm_ids=sorted({v for r in selected for v in (r['arm_id'], r['control'])}))


def select_final(summaries):
    by_id = {s['case']['arm_id']: s for s in summaries}
    rows = []
    for s in summaries:
        c=s['case']
        if s['status']!='completed' or not c['history_weight']:
            continue
        p=by_id.get(identifier(c['lr'],0.,c['seed']))
        if p is None or p['status']!='completed':
            continue
        q=quality(s,p,True); stable=stability(s); control_stable=stability(p)
        rows.append(dict(arm_id=c['arm_id'],control=p['case']['arm_id'],quality=q,
            stability=stable,control_stability=control_stable,
            passed=q['passed'] and stable['passed']))
    good=sorted([r for r in rows if r['passed']],key=lambda r:(r['quality']['candidate']['H'],r['quality']['candidate']['rq'],r['arm_id']))
    return dict(candidates=rows, selected=good[0] if good else None)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def save(path, value):
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    os.replace(pending,path)
