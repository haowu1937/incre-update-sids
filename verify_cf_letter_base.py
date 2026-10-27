"""Read-only task-SHA scientific verifier for completed Base CF-LETTER artifacts."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from genrec.cf_letter_protocol import JOB,SCOPE,MANIFEST,allocate,load_base,metrics,plan,read,save,sha

ROOT=Path(__file__).resolve().parent


def verify(context):
    c=read(context);output=Path(c['workspace'])/'outputs/cf-base';p=plan(ROOT)
    report=read(output/'compact.json');audit=read(output/'data_audit.json')
    if (report['stage']!=SCOPE or report['job_id']!=JOB or report['execution_status']!='completed'
            or report['code_sha']!=c['request']['code_sha'] or not report['original_inputs_unchanged']
            or report['uncertain_step_records'] or report['optimizer_steps_spent']>54640
            or any(report[k] is not False for k in ('update_training_performed','historical_training_performed','future_evaluated'))):
        raise ValueError('Base-only scientific scope failed')
    if sha(output/'data_audit.json')!=report['data_audit_sha256'] or sha(output/'curves.json')!=report['curves_sha256']:
        raise ValueError('Bounded artifact checksums differ')
    roles=list(report['stages'])
    if not roles or roles!=['sasrec','quantizer','gpt'][:len(roles)]:raise ValueError('Stage order differs')
    gates=[s['quality_gate'] for s in report['stages'].values()]
    if report['research_outcome']=='base_chain_accepted':
        if len(roles)!=3 or not all(gates) or report.get('resource_stop_reason'):raise ValueError('Base acceptance gate inconsistent')
    elif report['research_outcome'].endswith('_quality_gate_failed'):
        if not all(gates[:-1]) or gates[-1] or not report['research_outcome'].startswith(roles[-1]):raise ValueError('Invalid scientific early stop')
    elif report['research_outcome']!='base_chain_completed_with_resource_guard':raise ValueError('Unknown Base outcome')
    for path,value in report['local_artifact_sha256'].items():
        local=output/path
        if Path(path).is_absolute() or '..' in Path(path).parts or local.is_symlink() or not local.resolve().is_relative_to(output.resolve()):
            raise ValueError('Unsafe artifact path')
        if sha(local)!=value:raise ValueError('Checkpoint/table digest differs: '+path)
    deployment=next(k for k in c['contract']['input_sha256'] if k.endswith('/main-deployment.json'))
    _,data,binding=load_base(deployment,p)
    if audit['protocol_sha256']!=binding['protocol_sha256'] or audit['probe_sample_ids']!=binding['probe_sample_ids']:
        raise ValueError('Base input/probe lineage differs')
    if sha(output/'prepared.npz')!=audit['prepared_sha256']:raise ValueError('Prepared arrays changed')
    with np.load(output/'prepared.npz',allow_pickle=False) as prepared:
        for split in ('base_train','base_val'):
            for key,value in data[split].items():
                if not np.array_equal(prepared[split+'_'+key],value):raise ValueError('Saved causal arrays differ')
    total=0
    for role,stage in report['stages'].items():
        count=read(output/role/'optimizer_steps.json')
        if count['optimizer_step_intent'] is not None or count['optimizer_steps_completed']!=stage['optimizer_steps']:
            raise ValueError('Durable stage counters differ')
        total+=stage['optimizer_steps']
        cap=p['quantizer']['steps'] if role=='quantizer' else p[role]['step_cap']
        if not 0<stage['optimizer_steps']<=cap:raise ValueError('Stage optimizer budget exceeded')
        if role!='quantizer' and not 0<=stage['selected_step']<=stage['optimizer_steps']:raise ValueError('Selected step outside training')
        if role in ('sasrec','gpt'):
            curve=read(output/'curves.json')[role]
            selected=max(curve,key=lambda row:row['validation']['full_denominator']['ndcg@10'])
            if selected['step']!=stage['selected_step']:raise ValueError('Fixed-probe earliest-tie selection differs')
            expected_gate=(stage['selected_step']>0 and stage['selected_training_probe_loss']<stage['initial_training_probe_loss']
                and stage['full_base_validation']['full_denominator']['recall@10']>0)
            if stage['quality_gate']!=expected_gate:raise ValueError('Predictor quality arithmetic differs')
            for filename,key,source in [('selected_probe_queries.jsonl','selected_probe',data['probe']),
                                        ('full_validation_queries.jsonl','full_base_validation',np.arange(len(data['base_val']['targets'])))]:
                rows=[json.loads(line) for line in (output/role/filename).read_text().splitlines()]
                if len(rows)!=len(source):raise ValueError('Missing validation query records')
                for row,i in zip(rows,source):
                    if row['source_row']!=int(i) or row['sample_id']!=str(data['base_val']['samples'][i]) or row['target']!=int(data['base_val']['targets'][i]):
                        raise ValueError('Validation query identity mismatch')
                    if any(not 1<=v<=p['expected_catalog_items'] for v in row['predictions']):raise ValueError('Prediction outside Base catalog')
                recomputed=metrics(rows,p['topk'])
                for field in ('all_queries','servable_queries','unavailable_targets','full_denominator','conditional_on_servable','mean_sequence_nll_servable'):
                    if recomputed[field]!=stage[key][field]:raise ValueError('Validation metric arithmetic differs')
    if total!=report['optimizer_steps_spent']:raise ValueError('Global optimizer counters differ')
    # Independent CPU reload: selected SASRec exports and fixed GPT likelihoods.
    import torch
    from genrec.cf_letter_models import SASRec,CollaborativeRQ,GenerativeRetriever
    from genrec.cf_letter_protocol import gpt_arrays
    torch.set_num_threads(1)
    sas=report['stages']['sasrec'];saved=torch.load(output/'sasrec/selected.pth',map_location='cpu',weights_only=False)
    if saved['step']!=sas['selected_step'] or saved['code_sha']!=report['code_sha']:raise ValueError('SASRec checkpoint lineage differs')
    s0=SASRec(p['expected_catalog_items'],20,p['sasrec']);s0.load_state_dict(saved['model']);s0.eval()
    sas_rows=[json.loads(line) for line in (output/'sasrec/selected_probe_queries.jsonl').read_text().splitlines() if json.loads(line)['target']>0][:16]
    indices=[r['source_row'] for r in sas_rows]
    with torch.no_grad():
        scores=s0(torch.tensor(data['base_val']['history'][indices]))@s0.items.weight[1:].T
        actual=torch.nn.functional.cross_entropy(scores,torch.tensor(data['base_val']['targets'][indices]-1),reduction='none').numpy()
    if not np.allclose(actual,[r['nll'] for r in sas_rows],rtol=2e-5,atol=2e-4):raise ValueError('CPU SASRec checkpoint NLL differs')
    if 'quantizer' in report['stages']:
        features=np.load(output/'cf_embeddings.npy',allow_pickle=False)
        if not np.array_equal(features,saved['model']['items.weight'][1:].numpy()):raise ValueError('CF export differs from selected SASRec')
        qstate=torch.load(output/'quantizer/selected.pth',map_location='cpu',weights_only=False)
        q=CollaborativeRQ(torch.from_numpy(features),p['quantizer']);q.load_state_dict(qstate['model'])
        codes=np.load(output/'quantizer/semantic_codes.npy',allow_pickle=False)
        mapping,collision=allocate(codes,512)
        if collision!=report['stages']['quantizer']['collision']:raise ValueError('Reported full-SID capacity differs')
        if not np.array_equal(mapping,np.load(output/'item_sids.npy',allow_pickle=False)):raise ValueError('SID export/suffix ledger differs')
        with torch.no_grad():
            # Audit canonical saved assignments, allowing ordinary CPU/CUDA rounding.
            restored=sum(book[torch.as_tensor(codes[:,i])] for i,book in enumerate(q.books))
            mse=float((restored-q.features).square().mean())
        if not math.isclose(mse,report['stages']['quantizer']['final']['reconstruction_mse'],rel_tol=2e-5,abs_tol=1e-8):raise ValueError('CPU hard reconstruction does not reproduce')
        mean_mse=float(np.square(features.astype('float64')-features.astype('float64').mean(0)).mean())
        if not math.isclose(mean_mse,report['stages']['quantizer']['mean_predictor_mse'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('CF baseline arithmetic differs')
        if report['stages']['quantizer']['quality_gate']!=(report['stages']['quantizer']['final']['reconstruction_mse']<mean_mse):raise ValueError('CF quality gate differs')
        if 'gpt' in report['stages']:
            sample=read(output/'gpt/reload_samples.json');indices=sample['source_rows']
            g=GenerativeRetriever(20,512,p['gpt']);g.load_state_dict(torch.load(output/'gpt/selected.pth',map_location='cpu',weights_only=False)['model']);g.eval()
            x,m,y=gpt_arrays(data['base_val'],mapping,512,p['gpt']['user_buckets'])
            with torch.no_grad():actual=g.nll(torch.tensor(x[indices]),torch.tensor(m[indices]),torch.tensor(y[indices])).numpy()
            if not np.allclose(actual,sample['nll'],rtol=2e-5,atol=2e-4):raise ValueError('CPU GPT checkpoint likelihoods differ')
    lines=['### CF-LETTER Base chain acceptance','',
        'Job `'+JOB+'`; code `'+report['code_sha']+'`. Base train/validation only; no Update, Historical or Future execution.',
        'Outcome: `'+report['research_outcome']+'`. Actual optimizer steps: '+str(total)+' / 54,640.',
        'All query masks, selected checkpoints and scores refer to the new 20-event, four-digit Base-catalog protocol.',
        '', '| Component | Actual steps | Selected step | Base validation Recall@10 | Base validation NDCG@10 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for role in ('sasrec','gpt'):
        if role in report['stages']:
            s=report['stages'][role];v=s['full_base_validation']['full_denominator']
            lines.append('| '+role+' | '+str(s['optimizer_steps'])+' | '+str(s['selected_step'])+' | '+format(100*v['recall@10'],'.6f')+'% | '+format(v['ndcg@10'],'.9f')+' |')
    if 'quantizer' in report['stages']:
        q=report['stages']['quantizer'];lines+=['','CF quantizer hard reconstruction MSE: '+format(q['final']['reconstruction_mse'],'.9g')+
            '; mean-vector baseline: '+format(q['mean_predictor_mse'],'.9g')+'; maximum semantic bucket: '+str(q['collision']['maximum_collision_bucket'])+'.']
    lines+=['','The server verifier recomputed validation aggregates from saved query records, reproduced the CF export and hard reconstruction, '
        'and checked fixed-query selected-GPT NLL after an independent CPU reload when that stage completed. '
        'GPU reload parity and historical-input readiness remain bounded implementation checks; no historical optimization has run. '
        'The previous Future table is unchanged. Base scope stops here; no Update/Future job is queued.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=total,report_markdown='\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();save(a.output,verify(a.context))
