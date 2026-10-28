"""Read-only scientific acceptance for the bounded direct-codebook direction task."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_refresh_protocol import load_data,damage,versioned_mapping
from genrec.cf_letter_history_protocol import JOB,SCOPE,cases,plan

ROOT=Path(__file__).resolve().parent


def verify(context):
    c=read(context);output=Path(c['workspace'])/'outputs/cf-direction';p=plan(ROOT);r=read(output/'compact.json');audit=read(output/'data_audit.json')
    if (r['stage']!=SCOPE or r['job_id']!=JOB or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed' or not r['original_inputs_unchanged'] or r['uncertain_step_records']
            or not r['diagnostic_only'] or any(r[k] is not False for k in ('gpt_training_performed','sasrec_training_performed','full_training_performed','validation_evaluated','holdout_evaluated','future_evaluated'))):raise ValueError('Direction scope/identity differs')
    for name,value in r['returned_artifact_sha256'].items():
        if sha(output/(name+'.json'))!=value:raise ValueError('Bounded direction artifact changed')
    if sha(output/'local_artifact_index.json')!=r['local_artifact_index_sha256']:raise ValueError('Local artifact index changed')
    for name,value in read(output/'local_artifact_index.json').items():
        path=output/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(output.resolve()) or sha(path)!=value:raise ValueError('Local checkpoint/query artifact changed')
    for path,value in audit['input_sha256'].items():
        if sha(path)!=value:raise ValueError('Scientific source changed')
    deployment=next(k for k in c['contract']['input_sha256'] if k.endswith('/main-deployment.json'));data,binding=load_data(deployment,p)
    if audit['allowed_query_splits']!=['base_train','update_train'] or audit['anchor_train_queries']!=4096:raise ValueError('Unauthorized diagnostic cohort')
    with np.load(output/'prepared.npz',allow_pickle=False) as file:
        for key,value in data['anchors'].items():
            if not np.array_equal(file['anchor_'+key],value):raise ValueError('Prepared training anchors changed')
        for key in ('anchor_users','anchor_source_rows'):
            if not np.array_equal(file[key],data[key]):raise ValueError('Anchor identities changed')
    import torch
    from genrec.cf_letter_base_training import Sampler
    from genrec.cf_letter_history_direction import public_trajectory
    from genrec.cf_letter_refresh_training import teacher,paired_nll,frozen_quantizer
    torch.set_num_threads(1)
    sampling=read(output/'sampling.json');order=torch.randperm(4096,generator=torch.Generator().manual_seed(p['seed']+77)).numpy()
    if sampling['fixed_anchor_indices']!=order[:128].tolist() or sampling['gradient_anchor_order']!=order[:512].tolist():raise ValueError('Fixed training sampling differs')
    sampler=Sampler(len(binding['items']),p['content_batch_size'],p['seed'])
    if sampling['content_indices']!=[sampler.take()[0].tolist() for _ in range(5)]:raise ValueError('Fixed content sampling differs')
    def check_score(path,indices,expected=None):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=len(indices):raise ValueError('Incomplete hard-H query records')
        for row,i in zip(rows,indices):
            if row['sample_id']!=str(data['anchors']['samples'][i]) or row['source_row']!=int(data['anchor_source_rows'][i]) or row['user']!=str(data['anchor_users'][i]):raise ValueError('Hard-H query identity differs')
        if expected is not None and damage(rows,p['bootstrap_replicates'],p['seed'])!=expected:raise ValueError('Hard-H aggregation/bootstrap differs')
        return rows
    initial=check_score(output/'initial/full_anchor.jsonl',np.arange(4096),r['initial_full_anchor_train'])
    check_score(output/'initial/fixed_anchor.jsonl',order[:128],r['initial_fixed_anchor'])
    if abs(r['initial_full_anchor_train']['H']-p['refresh_predecessor']['expected_initial_full_H'])>1e-6 or r['initial_fixed_anchor']['H']<=0:raise ValueError('Initial positive H differs')
    calibration=read(output/'gradient_calibration.json');ratios=[]
    for probe in calibration['probes']:
        g=probe['gradients']['all']
        if probe['pre_update_damage_H']>0 and g['history_gradient_norm']>0 and g['rq_gradient_norm']>0:ratios.append(g['rq_gradient_norm']/g['history_gradient_norm'])
    weight=.1*float(np.median(ratios)) if ratios else None
    if calibration['history_weight']!=weight or r['history_weight']!=weight:raise ValueError('Training-only scalar calibration differs')
    expected_cases=cases(p);branches=r['branches'];count=0;groups={name:read(output/(name+'_curves.json')) for name in ('sgd','adamw')}
    base=frozen_quantizer(output,p,'cpu');features=np.load(output/'inputs/refresh/refreshed_cf_embeddings.npy',allow_pickle=False)
    previous=np.load(output/'inputs/base/item_sids.npy',allow_pickle=False)
    for result,case in zip(branches,expected_cases):
        if result['case']!=case:raise ValueError('Branch order/recipe differs')
        directory=output/case['name'];counter=read(directory/'optimizer_steps.json');trajectory=read(directory/'trajectory.json')
        if counter['optimizer_step_intent'] is not None or counter['optimizer_steps_completed']!=result['optimizer_steps'] or result['optimizer_steps']!=len(trajectory) or not 1<=result['optimizer_steps']<=5:raise ValueError('Branch optimizer accounting differs')
        count+=result['optimizer_steps']
        if groups[case['optimizer']][case['name']]!=public_trajectory(trajectory):raise ValueError('Published direction curve differs')
        saved=torch.load(directory/'initial.pth',map_location='cpu',weights_only=False)
        if saved['step']!=0 or saved['optimizer']['state'] or saved['code_sha']!=r['code_sha'] or not np.array_equal(saved['model']['features'].numpy(),features):raise ValueError('Branch initial state/optimizer differs')
        if any(not torch.equal(saved['model']['books.'+str(i)],book) for i,book in enumerate(base.books)):raise ValueError('Branches did not start from C0')
        group=saved['optimizer']['param_groups'][0]
        if group['lr']!=case['lr'] or group['weight_decay']!=(0. if case['optimizer']=='sgd' else .01):raise ValueError('Reset optimizer recipe differs')
        if case['optimizer']=='sgd' and group['momentum']!=0:raise ValueError('SGD momentum was not removed')
        if case['optimizer']=='adamw' and (tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8):raise ValueError('AdamW recipe differs')
        preceding_H=r['initial_fixed_anchor']['H']
        for entry in trajectory:
            if entry['step'] not in range(1,6) or not 0<=entry['clip_coefficient']<=1 or not math.isfinite(entry['pre_update_damage_H']):raise ValueError('Invalid actual loss/clip trace')
            for g in entry['gradient_components'].values():
                if g['rq_gradient_norm'] and g['weighted_history_over_rq'] is not None and not math.isclose(g['weighted_history_over_rq'],weight*g['history_gradient_norm']/g['rq_gradient_norm'],rel_tol=1e-10):raise ValueError('Weighted gradient arithmetic differs')
            if abs(entry['pre_update_damage_H']-preceding_H)>1e-5:raise ValueError('Logged pre-update damage differs from fixed hard score')
            codes=np.load(directory/f"semantic_codes_{entry['step']}.npy",allow_pickle=False);mapping,capacity=versioned_mapping(previous,codes,p['quantizer']['capacity'])
            if capacity!=entry['capacity']:raise ValueError('Per-update suffix capacity differs')
            if mapping is not None:
                if not np.array_equal(mapping,np.load(directory/f"item_sids_{entry['step']}.npy",allow_pickle=False)):raise ValueError('Per-update full SID mapping differs')
                fixed=check_score(directory/f"fixed_{entry['step']}.jsonl",order[:128]);preceding_H=damage(fixed,p['bootstrap_replicates'],p['seed'])['H']
                if preceding_H!=entry['post_update_fixed_H']:raise ValueError('Logged actual post-update hard H differs')
        for monitor in result['monitors']:
            step=monitor['step'];check_score(directory/f'full_{step}.jsonl',np.arange(4096),monitor['anchor_train'])
        if result['capacity_feasible'] and result['optimizer_steps']!=5:raise ValueError('Feasible branch incomplete')
        if result['initial_H']!=r['initial_full_anchor_train']['H']:raise ValueError('Branch H starting point differs')
        if result['changed_any_full_sid']!=any(v.get('full_sid_changed_fraction',0)>0 for v in trajectory):raise ValueError('Hard-SID plateau classification differs')
        if result['capacity_feasible'] and ([v['step'] for v in result['monitors']]!=[1,5] or result['final_full_H']!=result['monitors'][-1]['anchor_train']['H']):raise ValueError('Full anchor monitor endpoint differs')
    if len(branches)>12 or count!=r['optimizer_steps_spent'] or count>60:raise ValueError('Global micro-update budget differs')
    if weight is None and (branches or r['research_outcome']!='no_usable_history_gradient'):raise ValueError('Zero-gradient guard was bypassed')
    if weight is not None and r['research_outcome']!='resource_stopped' and len(branches)!=12:raise ValueError('Missing declared diagnostic branches')
    if weight is not None and r['research_outcome']!='resource_stopped':
        h_only=[b for b in branches if b['case']['objective']=='h_only']
        useful=any(b['capacity_feasible'] and b.get('final_full_H',math.inf)<r['initial_full_anchor_train']['H']-1e-6 for b in h_only)
        expected='hard_H_descent_observed' if useful else 'discrete_plateau_inconclusive' if not any(b['changed_any_full_sid'] for b in h_only) else 'no_hard_H_descent_in_declared_steps'
        if expected!=r['research_outcome']:raise ValueError('Hard-H direction outcome differs')
    # CPU reload checks canonical final full-catalog MSE and old/new likelihoods.
    previous=np.load(output/'inputs/base/item_sids.npy',allow_pickle=False);model,state=teacher(output,p,'cpu');small={k:v[:16] for k,v in data['anchors'].items()}
    mapping=np.load(output/'inputs/refresh/feature_only/item_sids.npy',allow_pickle=False)
    a,z,_=paired_nll(model,small,previous,mapping,p,'cpu')
    if not np.allclose(a,[v['old_sequence_nll'] for v in initial[:16]],rtol=2e-5,atol=2e-4) or not np.allclose(z,[v['new_sequence_nll'] for v in initial[:16]],rtol=2e-5,atol=2e-4):raise ValueError('CPU initial hard likelihood differs')
    for result in branches:
        if not result['capacity_feasible'] or not result['monitors']:continue
        directory=output/result['case']['name'];checkpoint=torch.load(directory/'last.pth',map_location='cpu',weights_only=False);codes=np.load(directory/'catalog_codes_5.npy',allow_pickle=False)
        restored=sum(checkpoint['model']['books.'+str(i)][torch.tensor(codes[:,i])] for i in range(3))
        mse=float((restored-checkpoint['model']['features']).square().mean())
        if not math.isclose(mse,result['monitors'][-1]['rq']['reconstruction_mse'],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU final hard reconstruction differs')
        final_mapping=np.load(directory/'item_sids_5.npy',allow_pickle=False);_,new_nll,_=paired_nll(model,small,previous,final_mapping,p,'cpu')
        final_rows=[json.loads(line) for line in (directory/'full_5.jsonl').read_text().splitlines()][:16]
        if not np.allclose(new_nll,[v['new_sequence_nll'] for v in final_rows],rtol=2e-5,atol=2e-4):raise ValueError('CPU final hard-history likelihood differs')
    lines=['### CF codebook historical-gradient direction diagnostic','',
        'Job `'+JOB+'`; code `'+r['code_sha']+'`. Actual micro-updates: '+str(count)+' / 60. Outcome: `'+r['research_outcome']+'`.',
        'Same frozen E1/Base GPT and original four-digit target/reference. Initial full anchor_train H='+format(r['initial_full_anchor_train']['H'],'.9g')+'; fixed calibrated lambda='+str(weight)+'.',
        '', '| Optimizer | LR | Objective | Steps | Final full anchor_train H | Capacity feasible |', '| --- | ---: | --- | ---: | ---: | --- |']
    for b in branches:lines.append('| '+b['case']['optimizer']+' | '+str(b['case']['lr'])+' | '+b['case']['objective']+' | '+str(b['optimizer_steps'])+' | '+str(b.get('final_full_H'))+' | '+str(b['capacity_feasible'])+' |')
    lines+=['','This is a fixed-sample five-step direction diagnostic, not convergence evidence or a lambda/STE sweep. Hard H is reaggregated from query records; raw norm/cosine and actual-update diagnostics do not prove hard-ID descent. No ranking validation, holdout/Future scoring, GPT/SASRec fitting, long quantizer training or automatic next task. Existing downstream tables remain unchanged.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=count,report_markdown='\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();save(a.output,verify(a.context))
