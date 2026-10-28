"""Read-only scientific acceptance of three training-only loss trajectories."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_refresh_protocol import load_data,damage,versioned_mapping
from genrec.cf_letter_history_long_protocol import JOB,SCOPE,cases,plan,indices_sha,tail_stability,paired_assessment,load_anchors
ROOT=Path(__file__).resolve().parent


def verify(context):
    c=read(context);output=Path(c['workspace'])/'outputs/cf-history-long';p=plan(ROOT);r=read(output/'compact.json');audit=read(output/'data_audit.json')
    forbidden=('gpt_training_performed','sasrec_training_performed','downstream_training_performed','validation_evaluated','holdout_evaluated','future_evaluated')
    if (r['stage']!=SCOPE or r['job_id']!=JOB or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed'
            or not r['original_inputs_unchanged'] or r['uncertain_step_records'] or not r['diagnostic_only'] or any(r[k] is not False for k in forbidden)):
        raise ValueError('Diagnostic identity/scope differs')
    for name,value in r['returned_artifact_sha256'].items():
        if sha(output/(name+'.json'))!=value:raise ValueError('Returned curve changed')
    if sha(output/'local_artifact_index.json')!=r['local_artifact_index_sha256']:raise ValueError('Local artifact index changed')
    for name,value in read(output/'local_artifact_index.json').items():
        path=output/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(output.resolve()) or sha(path)!=value:raise ValueError('Local query/checkpoint artifact changed')
    for path,value in audit['input_sha256'].items():
        if sha(path)!=value:raise ValueError('Read-only source changed')
    deployment=next(k for k in c['contract']['input_sha256'] if k.endswith('/main-deployment.json'));data,binding=load_data(deployment,p)
    if audit['allowed_query_splits']!=['base_train','update_train'] or audit['anchor_train_queries']!=p['anchor_size'] or audit['training_items']!=p['expected_training_items']:raise ValueError('Training-only cohort differs')
    prepared=load_anchors(output/'prepared.npz')
    for key,value in data['anchors'].items():
        if not np.array_equal(prepared['anchors'][key],value):raise ValueError('Prepared anchors differ')
    for key in ('anchor_users','anchor_source_rows'):
        if not np.array_equal(prepared[key],data[key]):raise ValueError('Anchor identities differ')
    import torch
    from genrec.cf_letter_base_training import Sampler
    from genrec.cf_letter_history_long import public_curves
    from genrec.cf_letter_refresh_training import teacher,paired_nll,frozen_quantizer
    from genrec.cf_letter_models import CollaborativeRQ
    torch.set_num_threads(1)
    order=torch.randperm(p['anchor_size'],generator=torch.Generator().manual_seed(p['seed']+77)).numpy();fixed=order[:p['fixed_anchor_queries']]
    sampling=read(output/'sampling.json');sampler=Sampler(p['expected_training_items'],p['content_batch_size'],p['seed'])
    if (sampling['fixed_anchor_indices']!=fixed.tolist() or sampling['gradient_anchor_order']!=order[:p['gradient_probe_batches']*p['fixed_anchor_queries']].tolist()
            or sampling['content_seed']!=p['seed'] or sampling['anchor_seed']!=p['seed']+78
            or sampling['calibration_content_sha256']!=[indices_sha(sampler.take()[0]) for _ in range(p['gradient_probe_batches'])]):raise ValueError('Shared sampling differs')
    def check_score(path,indices,expected=None):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=len(indices):raise ValueError('Incomplete hard-H query records')
        for row,i in zip(rows,indices):
            if row['sample_id']!=str(data['anchors']['samples'][i]) or row['source_row']!=int(data['anchor_source_rows'][i]) or row['user']!=str(data['anchor_users'][i]):raise ValueError('Hard-H query identity differs')
        score=damage(rows,p['bootstrap_replicates'],p['seed'])
        if expected is not None and score!=expected:raise ValueError('Hard-H/bootstrap reaggregation differs')
        return rows,score
    initial,_=check_score(output/'initial/full_anchor.jsonl',np.arange(p['anchor_size']),r['initial_full_anchor_train'])
    check_score(output/'initial/fixed_anchor.jsonl',fixed,r['initial_fixed_anchor'])
    reference=np.load(output/'inputs/refresh/baseline/old_anchor_nll.npy',allow_pickle=False)
    if not np.array_equal(reference,np.asarray([v['old_sequence_nll'] for v in initial])) or abs(r['initial_full_anchor_train']['H']-p['refresh_predecessor']['expected_initial_full_H'])>1e-6:raise ValueError('Original positive-H reference differs')
    calibration=read(output/'gradient_calibration.json');ratios=[]
    if len(calibration['probes'])!=p['gradient_probe_batches']:raise ValueError('Calibration probe count differs')
    for probe in calibration['probes']:
        g=probe['gradients']['all']
        if probe['pre_update_damage_H']>0 and g['history_gradient_norm']>0 and g['rq_gradient_norm']>0:ratios.append(g['rq_gradient_norm']/g['history_gradient_norm'])
    weight=.1*float(np.median(ratios)) if len(ratios)==p['gradient_probe_batches'] else None
    if calibration['history_weight']!=weight or r['history_weight']!=weight:raise ValueError('Fixed training-only lambda differs')
    expected_cases=cases(p);branches=r['branches'];count=0;full_results=[]
    base=frozen_quantizer(output,p,'cpu');features=np.load(output/'inputs/refresh/refreshed_cf_embeddings.npy',allow_pickle=False)
    previous=np.load(output/'inputs/base/item_sids.npy',allow_pickle=False);initial_codes=np.load(output/'inputs/refresh/feature_only/semantic_codes.npy',allow_pickle=False)
    for result,case in zip(branches,expected_cases):
        if result['case']!=case:raise ValueError('Arm order/recipe differs')
        directory=output/case['name'];counter=read(directory/'optimizer_steps.json');trajectory=read(directory/'trajectory.json');full_result=read(directory/'summary.json');full_results.append(full_result)
        if {k:v for k,v in full_result.items() if k!='monitors'}!=result:raise ValueError('Compact arm summary differs')
        if counter['optimizer_step_intent'] is not None or counter['optimizer_steps_completed']!=result['optimizer_steps'] or result['optimizer_steps']!=len(trajectory) or not 1<=result['optimizer_steps']<=p['steps_per_branch']:raise ValueError('Arm step accounting differs')
        count+=result['optimizer_steps']
        if read(output/(case['objective']+'_curves.json'))!=public_curves(full_result,trajectory):raise ValueError('Published curve differs')
        saved=torch.load(directory/'initial.pth',map_location='cpu',weights_only=False)
        if saved['step']!=0 or saved['optimizer']['state'] or saved['code_sha']!=r['code_sha'] or not np.array_equal(saved['model']['features'].numpy(),features):raise ValueError('Arm initial optimizer/feature state differs')
        if any(not torch.equal(saved['model']['books.'+str(i)],book) for i,book in enumerate(base.books)):raise ValueError('Arm did not start from exact C0')
        group=saved['optimizer']['param_groups'][0]
        if group['lr']!=1e-4 or group['weight_decay']!=0. or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8:raise ValueError('Reset AdamW recipe differs')
        content=Sampler(p['expected_training_items'],p['content_batch_size'],p['seed']);anchors=Sampler(p['anchor_size'],p['anchor_batch_size'],p['seed']+78);expected_monitors=[]
        for index,entry in enumerate(trajectory,1):
            ci,done=content.take();ai,_=anchors.take()
            if (entry['step']!=index or entry['content_indices_sha256']!=indices_sha(ci) or entry['anchor_indices_sha256']!=indices_sha(ai)
                    or entry['epoch']!=content.epochs or entry['anchor_epoch']!=anchors.epochs
                    or entry['processed_content_examples']!=content.processed or entry['processed_anchor_examples']!=anchors.processed):raise ValueError('Paired per-step sampling differs')
            new=np.asarray(entry['train_anchor_new_nll']);H=float(np.maximum(new-reference[ai],0).mean())
            if new.shape!=(len(ai),) or not np.isfinite(new).all() or abs(entry['pre_update_damage_H']-H)>1e-5:raise ValueError('Actual training-batch hard H differs')
            objective=H if case['objective']=='h_only' else entry['rq_loss'] if case['objective']=='rq_only' else entry['rq_loss']+weight*H
            if abs(objective-entry['objective_loss'])>1e-5:raise ValueError('Actual minimized objective differs')
            norm=entry['gradient_norm_before_clip'];clip=min(1.,.1/(norm+1e-6))
            if not math.isfinite(norm) or not math.isclose(entry['clip_coefficient'],clip,rel_tol=1e-10):raise ValueError('Gradient clipping differs')
            for g in entry['gradient_components'].values():
                if g['rq_gradient_norm'] and not math.isclose(g['weighted_history_over_rq'],weight*g['history_gradient_norm']/g['rq_gradient_norm'],rel_tol=1e-10):raise ValueError('Weighted gradient arithmetic differs')
            codes=np.load(directory/f'semantic_codes_{index}.npy',allow_pickle=False);mapping,capacity=versioned_mapping(previous,codes,p['quantizer']['capacity'])
            if capacity!=entry['capacity']:raise ValueError('Per-update complete suffix feasibility differs')
            if mapping is None:
                if index!=len(trajectory) or result['capacity_feasible'] or not (directory/'rejected.pth').exists():raise ValueError('Capacity failure was not preserved')
                continue
            if index%p['fixed_monitor_every_steps']==0 or index==p['steps_per_branch']:
                _,score=check_score(directory/f'fixed_{index}.jsonl',fixed)
                if score['H']!=entry.get('post_update_fixed_H'):raise ValueError('Fixed-query hard H differs')
            scheduled=done and content.epochs%p['full_monitor_every_epochs']==0
            if scheduled or index==p['steps_per_branch']:expected_monitors.append((index,content.epochs,scheduled))
        actual_monitors=[(v['step'],v['epoch'],v['scheduled_epoch_monitor']) for v in full_result['monitors']]
        if expected_monitors!=actual_monitors:raise ValueError('Completed-epoch monitor schedule differs')
        for monitor in full_result['monitors']:
            step=monitor['step'];check_score(directory/f'full_{step}.jsonl',np.arange(p['anchor_size']),monitor['anchor_train'])
            codes=np.load(directory/f'semantic_codes_{step}.npy',allow_pickle=False);mapping,capacity=versioned_mapping(previous,codes,p['quantizer']['capacity'])
            if capacity!=monitor['capacity'] or not np.array_equal(mapping,np.load(directory/f'item_sids_{step}.npy',allow_pickle=False)):raise ValueError('Full-monitor mapping differs')
            saved_monitor=torch.load(directory/f'monitor_{step}.pth',map_location='cpu',weights_only=False)
            q=CollaborativeRQ(torch.tensor(features),p['quantizer']);q.load_state_dict(saved_monitor['model']);metrics,canonical=q.audit()
            if not np.array_equal(canonical.numpy(),np.load(directory/f'catalog_codes_{step}.npy',allow_pickle=False)):raise ValueError('CPU full-monitor canonical codes differ')
            if not np.array_equal(canonical[:len(previous)].numpy(),codes):raise ValueError('Old-item monitor codes differ')
            for key in ('total_loss','quantization_loss','reconstruction_mse'):
                if not math.isclose(metrics[key],monitor['rq'][key],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU full-monitor loss differs')
            restored=sum(q.books[i][canonical[:len(previous),i]] for i in range(3))
            old_mse=float((restored-q.features[:len(previous)]).square().mean())
            if not math.isclose(old_mse,monitor['old_item_reconstruction_mse'],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU old-item reconstruction differs')
            if monitor['gradients']!=trajectory[step-1]['gradient_components']:raise ValueError('Full-monitor gradients differ from actual update trace')
        if tail_stability(full_result['monitors'],p['stability'])!=result['stability']:raise ValueError('Finite-window stability arithmetic differs')
        if result['capacity_feasible'] and result['optimizer_steps']!=p['steps_per_branch']:raise ValueError('Feasible arm incomplete')
        last=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
        if last['step']!=result['optimizer_steps'] or not np.array_equal(last['model']['features'].numpy(),features) or last['sampler']['processed']!=content.processed or last['anchor_sampler']['processed']!=anchors.processed:raise ValueError('Endpoint checkpoint/samplers differ')
        if result['capacity_feasible']:
            q=CollaborativeRQ(torch.tensor(features),p['quantizer']);q.load_state_dict(last['model']);metrics,codes=q.audit()
            final=full_result['monitors'][-1]
            if not np.array_equal(codes.numpy(),np.load(directory/f"catalog_codes_{p['steps_per_branch']}.npy",allow_pickle=False)):raise ValueError('Canonical CPU final assignments differ')
            for key in ('total_loss','quantization_loss','reconstruction_mse'):
                if not math.isclose(metrics[key],final['rq'][key],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU final native loss/reconstruction differs')
    if count!=r['optimizer_steps_spent'] or count>p['optimizer_step_cap'] or len(branches)>3:raise ValueError('Global budget differs')
    if weight is None and (branches or r['research_outcome']!='no_usable_history_gradient'):raise ValueError('Zero-gradient stop bypassed')
    if weight is not None and r['research_outcome']!='resource_stopped' and len(branches)!=3:raise ValueError('Missing declared arms')
    if weight is not None:
        expected='resource_stopped' if len(branches)!=3 else 'three_arm_diagnostic_completed'
        if any(not b['capacity_feasible'] for b in branches):expected='capacity_infeasible'
        if r['research_outcome']!=expected:raise ValueError('Diagnostic execution outcome differs')
    if paired_assessment(full_results,p['stability'])!=r['paired_assessment']:raise ValueError('Paired tail acceptance differs')
    model,state=teacher(output,p,'cpu');small={k:v[:16] for k,v in data['anchors'].items()}
    mapping=np.load(output/'inputs/refresh/feature_only/item_sids.npy',allow_pickle=False);old,new,_=paired_nll(model,small,previous,mapping,p,'cpu')
    if not np.allclose(old,reference[:16],rtol=2e-5,atol=2e-4) or not np.allclose(new,[v['new_sequence_nll'] for v in initial[:16]],rtol=2e-5,atol=2e-4):raise ValueError('CPU initial likelihood differs')
    for b in full_results:
        if not b['capacity_feasible']:continue
        directory=output/b['case']['name'];mapping=np.load(directory/f"item_sids_{p['steps_per_branch']}.npy",allow_pickle=False);_,new,_=paired_nll(model,small,previous,mapping,p,'cpu')
        rows,_=check_score(directory/f"full_{p['steps_per_branch']}.jsonl",np.arange(p['anchor_size']))
        if not np.allclose(new,[v['new_sequence_nll'] for v in rows[:16]],rtol=2e-5,atol=2e-4):raise ValueError('CPU endpoint hard-history likelihood differs')
    lines=['### CF codebook H-only / RQ-only / Mixed long diagnostic','',f'Job `{JOB}`; code `{r["code_sha"]}`. Actual codebook updates: {count} / 3000. Outcome: `{r["research_outcome"]}`.',
        f'Identical C0 and fixed E1; frozen selected Base GPT. Initial full anchor_train H={r["initial_full_anchor_train"]["H"]:.9g}; fixed training-calibrated lambda={weight}. AdamW LR=1e-4, weight decay=0, clipping=0.1, temperature=0.5.',
        '', '| Arm | Steps | Last measured full H (step) | Native RQ loss | Hard MSE | Tail stable | Capacity feasible |','| --- | ---: | --- | ---: | ---: | --- | --- |']
    for b in full_results:
        end=b['monitors'][-1] if b['monitors'] else None
        lines.append('| '+b['case']['objective']+' | '+str(b['optimizer_steps'])+' | '+str(b.get('final_full_H'))+' ('+str(b.get('final_full_monitor_step'))+') | '+str(end['rq']['total_loss'] if end else None)+' | '+str(end['rq']['reconstruction_mse'] if end else None)+' | '+str(b['stability']['passed'])+' | '+str(b['capacity_feasible'])+' |')
    lines+=['', 'Paired tail assessment: `'+json.dumps(r['paired_assessment'],sort_keys=True)+'`.',
        '', 'Training minibatch H is separate from fixed-128 and full-4096 hard-H curves. Full monitors run every ten completed content epochs and at the endpoint; stability uses the last eight scheduled epoch monitors, excluding an incidental final partial epoch. These are finite-window diagnostics, not guaranteed asymptotic convergence or held-out generalization. No lambda/STE scan, SASRec/GPT fitting, ranking validation, anchor holdout or Future scoring ran. Existing downstream tables remain unchanged. The scope ends here; no automatic extension or next task.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=count,report_markdown='\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();save(a.output,verify(a.context))
