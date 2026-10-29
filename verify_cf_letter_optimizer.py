"""Read-only optimizer-study scientific acceptance; no installation changes."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_refresh_protocol import load_data,damage
from genrec.cf_letter_optimizer_protocol import JOB,SCOPE,cases,plan,pair_cases,selection,counters,paired_assessment,curve_groups
from genrec.cf_letter_history_long_protocol import indices_sha,tail_stability,load_anchors
from genrec.cf_letter_suffix_audit import SuffixAudit
ROOT=Path(__file__).resolve().parent


def verify_arm(output,data,binding,p,case,code_sha,*,assignment_device="cpu",gpu_guard=None):
    output=Path(output);r=read(output/'arm_result.json')
    if r['case']!=case or r['code_sha']!=code_sha or not r['teacher_unchanged'] or not r['cf_embeddings_unchanged']:raise ValueError('Arm identity/freeze differs')
    prepared=load_anchors(output/'prepared.npz')
    for k,v in data['anchors'].items():
        if not np.array_equal(prepared['anchors'][k],v):raise ValueError('Prepared anchor encoding differs')
    for k in ('anchor_users','anchor_source_rows'):
        if not np.array_equal(prepared[k],data[k]):raise ValueError('Prepared raw anchor identity differs')
    import torch
    from genrec.cf_letter_base_training import Sampler
    from genrec.cf_letter_history_long import public_curves
    from genrec.cf_letter_refresh_training import teacher,paired_nll,frozen_quantizer,quantize
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
    weight=p['fixed_history_weight'];branches=[r['result']];expected_cases=[case];count=0;full_results=[]
    base=frozen_quantizer(output,p,'cpu');features=np.load(output/'inputs/refresh/refreshed_cf_embeddings.npy',allow_pickle=False)
    previous=np.load(output/'inputs/base/item_sids.npy',allow_pickle=False);suffix_audit=SuffixAudit(previous,p['quantizer']['capacity']);initial_codes=np.load(output/'inputs/refresh/feature_only/semantic_codes.npy',allow_pickle=False)
    for result,case in zip(branches,expected_cases):
        if result['case']!=case:raise ValueError('Arm order/recipe differs')
        directory=output/case['name'];counter=read(directory/'optimizer_steps.json');trajectory=read(directory/'trajectory.json');full_result=read(directory/'summary.json');full_results.append(full_result)
        if {k:v for k,v in full_result.items() if k!='monitors'}!=result:raise ValueError('Compact arm summary differs')
        if counter['optimizer_step_intent'] is not None or counter['optimizer_steps_completed']!=result['optimizer_steps'] or result['optimizer_steps']!=len(trajectory) or not 1<=result['optimizer_steps']<=p['steps_per_branch']:raise ValueError('Arm step accounting differs')
        count+=result['optimizer_steps']
        if read(output/(case['name']+'_curves.json'))!=public_curves(full_result,trajectory):raise ValueError('Published curve differs')
        saved=torch.load(directory/'initial.pth',map_location='cpu',weights_only=False)
        if saved['step']!=0 or saved['optimizer']['state'] or saved['code_sha']!=r['code_sha'] or not np.array_equal(saved['model']['features'].numpy(),features):raise ValueError('Arm initial optimizer/feature state differs')
        if any(not torch.equal(saved['model']['books.'+str(i)],book) for i,book in enumerate(base.books)):raise ValueError('Arm did not start from exact C0')
        group=saved['optimizer']['param_groups'][0]
        if group['lr']!=case['lr'] or group['weight_decay']!=0. or tuple(group['betas'])!=(case['beta1'],p['beta2']) or group['eps']!=1e-8:raise ValueError('Reset AdamW recipe differs')
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
            codes=np.load(directory/f'semantic_codes_{index}.npy',allow_pickle=False);mapping,capacity=suffix_audit(codes)
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
            codes=np.load(directory/f'semantic_codes_{step}.npy',allow_pickle=False);mapping,capacity=suffix_audit(codes)
            if capacity!=monitor['capacity'] or not np.array_equal(mapping,np.load(directory/f'item_sids_{step}.npy',allow_pickle=False)):raise ValueError('Full-monitor mapping differs')
            saved_monitor=torch.load(directory/f'monitor_{step}.pth',map_location='cpu',weights_only=False)
            if saved_monitor['step']!=step or saved_monitor['code_sha']!=code_sha or not np.array_equal(saved_monitor['model']['features'].numpy(),features):raise ValueError('Monitoring checkpoint identity/features differ')
            q=CollaborativeRQ(torch.tensor(features),p['quantizer']);q.load_state_dict(saved_monitor['model']);q=q.to(assignment_device)
            if gpu_guard:gpu_guard()
            metrics,canonical=q.audit();canonical=canonical.cpu()
            if not np.array_equal(canonical.numpy(),np.load(directory/f'catalog_codes_{step}.npy',allow_pickle=False)):raise ValueError(assignment_device+' full-monitor canonical codes differ')
            old_canonical=canonical[:len(previous)].numpy() if assignment_device=='cpu' else quantize(q,q.features[:len(previous)],assignment_device)[0]
            if not np.array_equal(old_canonical,codes):raise ValueError('Old-item monitor codes differ')
            for key in ('total_loss','quantization_loss','reconstruction_mse'):
                if not math.isclose(metrics[key],monitor['rq'][key],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU full-monitor loss differs')
            restored=sum(q.books[i][torch.tensor(old_canonical[:,i],device=assignment_device)] for i in range(3))
            old_mse=float((restored-q.features[:len(previous)]).square().mean())
            if not math.isclose(old_mse,monitor['old_item_reconstruction_mse'],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU old-item reconstruction differs')
            if monitor['gradients']!=trajectory[step-1]['gradient_components']:raise ValueError('Full-monitor gradients differ from actual update trace')
        if tail_stability(full_result['monitors'],p['stability'])!=result['stability']:raise ValueError('Finite-window stability arithmetic differs')
        if result['capacity_feasible'] and result['optimizer_steps']!=p['steps_per_branch']:raise ValueError('Feasible arm incomplete')
        last=torch.load(directory/'last.pth',map_location='cpu',weights_only=False)
        if last['step']!=result['optimizer_steps'] or not np.array_equal(last['model']['features'].numpy(),features) or last['sampler']['processed']!=content.processed or last['anchor_sampler']['processed']!=anchors.processed:raise ValueError('Endpoint checkpoint/samplers differ')
        if result['capacity_feasible']:
            final_checkpoint=torch.load(directory/f"monitor_{p['steps_per_branch']}.pth",map_location='cpu',weights_only=False)
            if any(not torch.equal(v,final_checkpoint['model'][k]) for k,v in last['model'].items()):raise ValueError('Final model differs from final monitoring checkpoint')
            q=CollaborativeRQ(torch.tensor(features),p['quantizer']);q.load_state_dict(last['model']);q=q.to(assignment_device)
            if gpu_guard:gpu_guard()
            metrics,codes=q.audit();codes=codes.cpu()
            final=full_result['monitors'][-1]
            if not np.array_equal(codes.numpy(),np.load(directory/f"catalog_codes_{p['steps_per_branch']}.npy",allow_pickle=False)):raise ValueError(assignment_device+' final assignments differ')
            for key in ('total_loss','quantization_loss','reconstruction_mse'):
                if not math.isclose(metrics[key],final['rq'][key],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU final native loss/reconstruction differs')
    model,state=teacher(output,p,'cpu');small={k:v[:16] for k,v in data['anchors'].items()}
    mapping=np.load(output/'inputs/refresh/feature_only/item_sids.npy',allow_pickle=False);old,new,_=paired_nll(model,small,previous,mapping,p,'cpu')
    if not np.allclose(old,reference[:16],rtol=2e-5,atol=2e-4) or not np.allclose(new,[v['new_sequence_nll'] for v in initial[:16]],rtol=2e-5,atol=2e-4):raise ValueError('CPU initial likelihood differs')
    for b in full_results:
        if not b['capacity_feasible']:continue
        directory=output/b['case']['name'];mapping=np.load(directory/f"item_sids_{p['steps_per_branch']}.npy",allow_pickle=False);_,new,_=paired_nll(model,small,previous,mapping,p,'cpu')
        rows,_=check_score(directory/f"full_{p['steps_per_branch']}.jsonl",np.arange(p['anchor_size']))
        if not np.allclose(new,[v['new_sequence_nll'] for v in rows[:16]],rtol=2e-5,atol=2e-4):raise ValueError('CPU endpoint hard-history likelihood differs')
    return full_results[0]


def verify(context):
    c=read(context);output=Path(c['workspace'])/'outputs/cf-optimizer';p=plan(ROOT);r=read(output/'compact.json');audit=read(output/'data_audit.json')
    if (r['stage']!=SCOPE or r['job_id']!=JOB or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed'
            or not r['original_inputs_unchanged'] or r['uncertain_step_records'] or r['fixed_history_weight']!=p['fixed_history_weight']
            or any(r[k] is not False for k in ('gpt_training_performed','sasrec_training_performed','validation_evaluated','holdout_evaluated','future_evaluated'))):raise ValueError('Study scope/identity differs')
    for name,value in r['returned_artifact_sha256'].items():
        if sha(output/(name+'.json'))!=value:raise ValueError('Returned study artifact changed')
    if sha(output/'local_artifact_index.json')!=r['local_artifact_index_sha256']:raise ValueError('Local artifact index changed')
    cache={}
    for name,value in read(output/'local_artifact_index.json').items():
        path=output/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):raise ValueError('Unexpected local artifact path')
        stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in cache:cache[key]=sha(path)
        if cache[key]!=value:raise ValueError('Bound local artifact changed')
    for path,value in audit['input_sha256'].items():
        if sha(path)!=value:raise ValueError('Read-only original dependency changed')
    for name,value in p['accepted_control']['files_sha256'].items():
        if sha(output/'inputs/control'/name)!=value:raise ValueError('Reused control changed')
    prior=read(output/'inputs/control/compact.json')
    if prior['code_sha']!=p['accepted_control']['code_sha'] or prior['history_weight']!=p['fixed_history_weight']:raise ValueError('Wrong reused control lineage/lambda')
    before=next(v for v in prior['branches'] if v['case']['objective']=='h_only')
    before['case']=dict(name=p['accepted_control']['control_name'],optimizer='adamw',objective='h_only',steps=p['steps_per_branch'],phase='reused_control',lr=1e-4,beta1=.9)
    initial=dict(H=prior['initial_full_anchor_train']['H'],rq=prior['initial_rq']['total_loss'],mse=prior['initial_rq']['reconstruction_mse'])
    if before!=r['reused_control'] or initial!=r['initial']:raise ValueError('Reused control summary differs')
    deployment=next(path for path in c['contract']['input_sha256'] if path.endswith('/main-deployment.json'));data,binding=load_data(deployment,p)
    if audit['allowed_query_splits']!=p['allowed_query_splits'] or audit['training_items']!=11691 or audit['anchor_train_queries']!=4096:raise ValueError('Wrong training-only cohort')
    prepared=load_anchors(output/'prepared.npz')
    for k,v in data['anchors'].items():
        if not np.array_equal(prepared['anchors'][k],v):raise ValueError('Prepared cohort differs')
    screen=r['screen_results'];pair=r['pair_results'];full=[]
    screen_expected=cases(p)
    if len(screen)>5 or any(v['case']!=screen_expected[i] for i,v in enumerate(screen)):raise ValueError('Screen matrix/order differs')
    for result in screen:
        verified=verify_arm(output/'arms'/result['case']['name'],data,binding,p,result['case'],r['code_sha']);full.append(verified)
        if {k:v for k,v in verified.items() if k!='monitors'}!=result:raise ValueError('Root screen summary differs')
        if read(output/(result['case']['name']+'_curves.json'))!=read(output/'arms'/result['case']['name']/(result['case']['name']+'_curves.json')):raise ValueError('Returned screen curve differs')
    expected_selection=selection([before,*screen],initial,p) if len(screen)==5 else dict(candidates=[],selected_name=None,selected_case=None,pair_authorized=False,reason='incomplete_screen')
    if expected_selection!=r['selection'] or expected_selection!=read(output/'selection.json'):raise ValueError('Training-only selection/gates differ')
    selected=next((v for v in [before,*screen] if v['case']['name']==expected_selection['selected_name']),None)
    pair_expected=pair_cases(selected,p)
    if len(pair)>len(pair_expected) or any(v['case']!=pair_expected[i] for i,v in enumerate(pair)):raise ValueError('Conditional pair bypassed gate or recipe differs')
    for result in pair:
        verified=verify_arm(output/'arms'/result['case']['name'],data,binding,p,result['case'],r['code_sha'])
        if {k:v for k,v in verified.items() if k!='monitors'}!=result:raise ValueError('Root paired summary differs')
        if read(output/(result['case']['name']+'_curves.json'))!=read(output/'arms'/result['case']['name']/(result['case']['name']+'_curves.json')):raise ValueError('Returned paired curve differs')
    for name,arms in curve_groups(p).items():
        expected={arm:read(output/(arm+'_curves.json')) for arm in arms if (output/(arm+'_curves.json')).exists()}
        if read(output/(name+'.json'))!=expected:raise ValueError('Lossless grouped curve differs')
    expected_pair=paired_assessment([selected,*pair],p['stability']) if selected and len(pair)==2 else dict(eligible=False,reason='no_eligible_screen_or_pair_incomplete')
    if expected_pair!=r['paired_assessment']:raise ValueError('Paired efficacy arithmetic differs')
    journal=read(output/'launch_journal.json');executed=screen+pair
    if journal!=[v['case'] for v in executed] or len({v['name'] for v in journal})!=len(journal) or sum(v['steps'] for v in journal)>7000:raise ValueError('Prospective arm reservation/order differs')
    total,steps,intents=counters(output)
    if intents or total!=r['optimizer_steps_spent'] or total!=sum(v['optimizer_steps'] for v in executed) or total>7000 or steps!=r['stage_step_counts']:raise ValueError('Actual study budget differs')
    expected_outcome='resource_stopped' if r['resource_stop_reason'] else 'no_eligible_H_only_setting' if not expected_selection['pair_authorized'] else 'paired_verification_completed'
    if r['research_outcome']!=expected_outcome or (not r['resource_stop_reason'] and (len(screen)!=5 or (selected is not None and len(pair)!=2))):raise ValueError('Unexpected stage completion')
    lines=['### CF codebook learning-rate / first-moment ablation','',f'Job `{JOB}`; code `{r["code_sha"]}`. New optimizer steps: {total} / 7000; accepted control steps are reused, not rerun. Outcome: `{r["research_outcome"]}`.',
        f'Fixed E1/Base C0 and Base GPT; lambda={p["fixed_history_weight"]}, temperature=0.5, clipping=0.1, weight decay=0. Selection uses only full anchor_train and reconstruction.',
        '', '| Arm | LR | beta1 | Steps | Final full H | Tail H mean | Joint tail stable | Screen eligible |','| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    gates={v['name']:v for v in expected_selection['candidates']}
    for b in [before,*screen,*pair]:
        s=b['stability'];lines.append('| '+b['case']['name']+' | '+str(b['case']['lr'])+' | '+str(b['case']['beta1'])+' | '+str(b['optimizer_steps'])+' | '+str(b.get('final_full_H'))+' | '+str(s.get('H',{}).get('mean'))+' | '+str(s['passed'])+' | '+str(gates.get(b['case']['name'],{}).get('eligible','n/a'))+' |')
    lines+=['', 'Training-side selection: `'+json.dumps(expected_selection,sort_keys=True)+'`.', '', 'Paired assessment: `'+json.dumps(expected_pair,sort_keys=True)+'`.',
        '', 'Each new arm starts independently from C0(E1), with identical content/anchor order. The selector uses last-eight scheduled full-monitor means, not an intermediate low point. Full SID/suffix feasibility is checked after every update. CPU acceptance reaggregates hard H, checks canonical RQ/MSE and initial/final GPT likelihoods. These finite-window training diagnostics do not establish asymptotic convergence or held-out recommendation gains. No lambda/STE changes, SASRec/GPT fitting, ranking validation, holdout/Future scoring, automatic retry or next task. Existing Base/Future tables are unchanged.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=total,report_markdown='\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();save(a.output,verify(a.context))
