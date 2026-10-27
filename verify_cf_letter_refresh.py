"""Read-only scientific verifier for accepted-Base CF refresh diagnostics."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from genrec.cf_letter_protocol import encoded,gpt_arrays,metrics,read,save,sha
from genrec.cf_letter_refresh_protocol import JOB,SCOPE,damage,load_data,plan,versioned_mapping

ROOT=Path(__file__).resolve().parent


def verify(context):
    c=read(context);output=Path(c['workspace'])/'outputs/cf-refresh';p=plan(ROOT);r=read(output/'compact.json');audit=read(output/'data_audit.json')
    if (r['job_id']!=JOB or r['stage']!=SCOPE or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed'
            or not r['original_inputs_unchanged'] or r['uncertain_step_records']
            or any(r[k] is not False for k in ('quantizer_training_performed','gpt_training_performed','historical_training_performed','holdout_evaluated','future_evaluated'))):
        raise ValueError('Refresh diagnostic identity/scope failed')
    if sha(output/'data_audit.json')!=r['data_audit_sha256'] or sha(output/'curves.json')!=r['curves_sha256']:raise ValueError('Returned diagnostic artifacts changed')
    for name,value in r['local_artifact_sha256'].items():
        path=output/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(output.resolve()) or sha(path)!=value:raise ValueError('Local diagnostic artifact differs')
    for path,value in audit['input_sha256'].items():
        if sha(path)!=value:raise ValueError('Scientific source input changed')
    for name,value in p['predecessor']['files_sha256'].items():
        if sha(output/'inputs/base'/name)!=value:raise ValueError('Accepted dependency copy changed')
    deployment=next(k for k in c['contract']['input_sha256'] if k.endswith('/main-deployment.json'))
    data,binding=load_data(deployment,p)
    if audit['probe_sample_ids']!=binding['probe_sample_ids']:raise ValueError('Update probe changed')
    if sha(output/'prepared.npz')!=audit['prepared_sha256']:raise ValueError('Prepared arrays changed')
    with np.load(output/'prepared.npz',allow_pickle=False) as file:
        for split in ('train','val','anchors'):
            for key,value in data[split].items():
                if not np.array_equal(file[split+'_'+key],value):raise ValueError('Prepared causal data differs')
        for key in ('probe','anchor_users','anchor_source_rows'):
            if not np.array_equal(file[key],data[key]):raise ValueError('Anchor/probe identity differs')
    if list(r['stages'])!=['baseline','sasrec','feature_only']:raise ValueError('Missing ordered diagnostic stage')
    b=r['stages']['baseline'];s=r['stages']['sasrec'];f=r['stages']['feature_only']
    counter=read(output/'sasrec/optimizer_steps.json')
    if counter['optimizer_step_intent'] is not None or counter['optimizer_steps_completed']!=s['optimizer_steps'] or s['optimizer_steps']!=12240 or r['optimizer_steps_spent']!=12240 or s['epochs']!=30:
        raise ValueError('Refresh optimizer/epoch accounting differs')
    if b['optimizer_steps']!=0 or f['optimizer_steps']!=0 or not b['same_map_H_zero']:raise ValueError('Frozen diagnostic step accounting differs')
    curves=read(output/'curves.json')['sasrec'];selected=max(curves,key=lambda row:row['validation']['full_denominator']['ndcg@10'])
    if selected['step']!=s['selected_step'] or selected['epoch']!=s['selected_epoch']:raise ValueError('Update probe selection rule differs')
    for filename,key,indices in [('selected_probe_queries.jsonl','selected_probe',data['probe']),('full_update_validation_queries.jsonl','full_update_validation',np.arange(len(data['val']['targets'])))]:
        rows=[json.loads(line) for line in (output/'sasrec'/filename).read_text().splitlines()]
        if len(rows)!=len(indices):raise ValueError('Missing Update validation query')
        for row,i in zip(rows,indices):
            if row['sample_id']!=str(data['val']['samples'][i]) or row['source_row']!=int(i) or row['target']!=int(data['val']['targets'][i]) or any(not 1<=item<=p['old_catalog_items'] for item in row['predictions']):raise ValueError('Update query/catalog identity differs')
        result=metrics(rows,p['topk'])
        if any(result[k]!=s[key][k] for k in result):raise ValueError('Update metric arithmetic differs')
    def checked_anchor(filename):
        rows=[json.loads(line) for line in (output/filename).read_text().splitlines()]
        if len(rows)!=p['anchor_size']:raise ValueError('Anchor cohort incomplete')
        for row,source,sample,user in zip(rows,data['anchor_source_rows'],data['anchors']['samples'],data['anchor_users']):
            if row['source_row']!=int(source) or row['sample_id']!=str(sample) or row['user']!=str(user):raise ValueError('Anchor identity changed')
        return rows
    reference=checked_anchor('baseline/anchor_reference.jsonl')
    if damage(reference,p['bootstrap_replicates'],p['seed'])!=b['anchor_train'] or b['anchor_train']['H']>1e-6:raise ValueError('Same-map H arithmetic differs')
    previous=np.load(output/'inputs/base/item_sids.npy',allow_pickle=False);codes=np.load(output/'feature_only/semantic_codes.npy',allow_pickle=False)
    mapping,capacity=versioned_mapping(previous,codes,p['quantizer']['capacity'])
    if capacity!=f['capacity']:raise ValueError('Versioned suffix capacity differs')
    paired=None
    if mapping is not None:
        if not np.array_equal(mapping,np.load(output/'feature_only/item_sids.npy',allow_pickle=False)):raise ValueError('Refreshed mapping differs')
        paired=checked_anchor('feature_only/anchor_train_paired_losses.jsonl')
        if damage(paired,p['bootstrap_replicates'],p['seed'])!=f['anchor_train']:raise ValueError('Hard H arithmetic differs')
        if any(abs(a['old_sequence_nll']-b['old_sequence_nll'])>1e-6 for a,b in zip(paired,reference)):raise ValueError('Old reference changed')
        if r['research_outcome']!='feature_refresh_measured':raise ValueError('Measured outcome differs')
    elif f['anchor_train'] is not None or r['research_outcome']!='feature_refresh_capacity_infeasible':raise ValueError('Capacity failure was hidden')
    # Independent CPU reloads audit likelihoods, CF export and canonical hard MSE.
    import torch
    from genrec.cf_letter_models import SASRec
    from genrec.cf_letter_refresh_training import teacher,frozen_quantizer,paired_nll
    torch.set_num_threads(1)
    saved=torch.load(output/'sasrec/selected.pth',map_location='cpu',weights_only=False)
    initial=torch.load(output/'sasrec/initial.pth',map_location='cpu',weights_only=False)
    base=torch.load(output/'inputs/base/sasrec/selected.pth',map_location='cpu',weights_only=False)
    if saved['step']!=s['selected_step'] or saved['code_sha']!=r['code_sha'] or initial['step']!=0 or initial['optimizer']['state']:raise ValueError('Warm-start/selection lineage differs')
    for key,value in base['model'].items():
        actual=initial['model'][key][:len(value)] if key=='items.weight' else initial['model'][key]
        if not torch.equal(actual,value):raise ValueError('Original SASRec weights changed before refresh')
    features=np.load(output/'refreshed_cf_embeddings.npy',allow_pickle=False)
    if not np.array_equal(features,saved['model']['items.weight'][1:].numpy()):raise ValueError('Refreshed CF table is not selected weights')
    sas=SASRec(len(binding['items']),p['history'],p['sasrec']);sas.load_state_dict(saved['model']);sas.eval()
    samples=[json.loads(line) for line in (output/'sasrec/selected_probe_queries.jsonl').read_text().splitlines() if json.loads(line)['target']>0][:16]
    indices=[row['source_row'] for row in samples]
    with torch.no_grad():
        logits=sas(torch.tensor(data['val']['history'][indices]))@sas.items.weight[1:p['old_catalog_items']+1].T
        actual=torch.nn.functional.cross_entropy(logits,torch.tensor(data['val']['targets'][indices]-1),reduction='none').numpy()
    if not np.allclose(actual,[row['nll'] for row in samples],rtol=2e-5,atol=2e-4):raise ValueError('CPU refreshed SASRec checkpoint NLL differs')
    q=frozen_quantizer(output,p,'cpu')
    with torch.no_grad():
        restored=sum(book[torch.as_tensor(codes[:,i])] for i,book in enumerate(q.books));mse=float((restored-torch.from_numpy(features[:len(previous)])).square().mean())
    if not math.isclose(mse,f['old_item_quantizer']['reconstruction_mse'],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU hard CF reconstruction differs')
    model,state=teacher(output,p,'cpu');small={k:v[:16] for k,v in data['anchors'].items()}
    a,z,_=paired_nll(model,small,previous,previous if mapping is None else mapping,p,'cpu')
    if not np.allclose(a,[v['old_sequence_nll'] for v in reference[:16]],rtol=2e-5,atol=2e-4):raise ValueError('CPU Base reference NLL differs')
    if paired is not None and not np.allclose(z,[v['new_sequence_nll'] for v in paired[:16]],rtol=2e-5,atol=2e-4):raise ValueError('CPU refreshed-history NLL differs')
    lines=['### Shared CF refresh and feature-only damage diagnostic','',
        'Job `'+JOB+'`; code `'+r['code_sha']+'`. SASRec refresh: '+str(s['optimizer_steps'])+' steps / 30 cumulative epochs; selected epoch '+str(s['selected_epoch'])+'.',
        'Frozen Base GPT and codebooks; exact old anchor_train queries, old target/mask/catalog, four-digit NLL. Same-map H='+format(b['anchor_train']['H'],'.9g')+'.',
        'Outcome: `'+r['research_outcome']+'`. Old-item semantic churn='+format(100*f['semantic_changed_fraction'],'.6f')+'%; maximum required suffix capacity='+str(capacity['maximum_required_suffix_capacity'])+' / 512.']
    if paired is not None:lines+=['Feature-only anchor_train H='+format(f['anchor_train']['H'],'.9g')+'; mean delta sequence NLL='+format(f['anchor_train']['mean_delta_sequence_nll'],'.9g')+'.']
    lines+=['Old-item hard reconstruction MSE='+format(f['old_item_quantizer']['reconstruction_mse'],'.9g')+'.',
        'Server verification reaggregated query metrics/H, checked exact warm start and CPU likelihood/CF reconstruction. This is training-cohort diagnostic evidence, not historical-penalty benefit or Future ranking. No quantizer/GPT training, holdout/Future scoring or next task. Previous Base and Future tables remain unchanged.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=s['optimizer_steps'],report_markdown='\n\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();save(a.output,verify(a.context))
