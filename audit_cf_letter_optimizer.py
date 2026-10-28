"""Read-only CPU/GPU checkpoint-index diagnosis; zero optimizer updates."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,encoded

JOB='cf-letter-optimizer-audit-001'
SCOPE='cf-letter-optimizer-audit-v1'
MANIFEST='docs/experiments/cf_letter_optimizer_audit.json'
ROOT=Path(__file__).resolve().parent


def evidence(features,books,saved,cpu,rows,limit=8,comparison='cpu'):
    values=[]
    for row,level in rows[:limit]:
        residual=np.asarray(features[row],dtype=np.float64).copy()
        for j in range(level):residual-=np.asarray(books[j][saved[row,j]],dtype=np.float64)
        distances=np.sum((np.asarray(books[level],dtype=np.float64)-residual)**2,axis=1)
        best=int(np.argmin(distances));a=int(saved[row,level]);b=int(cpu[row,level])
        values.append(dict(row=int(row),level=int(level),saved_index=a,comparison_index=b,comparison=comparison,float64_nearest_on_saved_prefix=best,
            saved_distance_float64=float(distances[a]),nearest_distance_float64=float(distances[best]),saved_minus_nearest_float64=float(distances[a]-distances[best]),
            cpu_distance_on_saved_prefix_float64=float(distances[b])))
    return values


def inspect_checkpoint(path,saved_path,old_path,reported_mse,gpu=False,old_n=None):
    import torch
    from genrec.cf_letter_models import CollaborativeRQ
    from genrec.cf_letter_refresh_training import quantize
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);state=checkpoint['model'];config=checkpoint['settings']['quantizer']
    model=CollaborativeRQ(state['features'],config);model.load_state_dict(state);model.eval();model.requires_grad_(False)
    saved=np.load(saved_path,allow_pickle=False);old=np.load(old_path,allow_pickle=False);old_n=len(old) if old_n is None else old_n
    with torch.no_grad():
        cpu=model.assignments().cpu().numpy();features=model.features.cpu().numpy();books=[v.cpu().numpy() for v in model.books]
        restored=sum(np.asarray(book,dtype=np.float64)[saved[:,i]] for i,book in enumerate(books));mse=float(np.mean((restored-features.astype(np.float64))**2))
        positions=np.argwhere(cpu!=saved)
        result=dict(cpu_mismatch_elements=int(len(positions)),cpu_mismatch_items=int(np.count_nonzero(np.any(cpu!=saved,axis=1))),
            cpu_old_item_mismatch_items=int(np.count_nonzero(np.any(cpu[:old_n]!=saved[:old_n],axis=1))),
            saved_full_vs_old_chunk_mismatch_elements=int(np.count_nonzero(saved[:old_n]!=old)),reported_mse=float(reported_mse),cpu_saved_assignment_mse=mse,
            saved_mse_absolute_difference=abs(mse-reported_mse),saved_mse_relative_difference=abs(mse-reported_mse)/max(abs(reported_mse),1e-30),
            gpu_full_mismatch_elements=None,gpu_old_chunk_mismatch_elements=None)
        samples=evidence(features,books,saved,cpu,positions,limit=4)
        alternative=saved.copy();alternative[:old_n]=old
        samples+=evidence(features,books,saved,alternative,np.argwhere(alternative!=saved),limit=4,comparison='saved_old_item_chunk')
        if gpu:
            model=model.to('cuda:0');again=model.assignments().cpu().numpy();old_again,_=quantize(model,model.features[:old_n],'cuda:0')
            result.update(gpu_full_mismatch_elements=int(np.count_nonzero(again!=saved)),gpu_old_chunk_mismatch_elements=int(np.count_nonzero(old_again!=old)))
            del model;torch.cuda.empty_cache()
    return result,samples


def check_source(source,p):
    if source.resolve()!=source or sha(source/'compact.json')!=p['source_compact_sha256'] or sha(source/'local_artifact_index.json')!=p['source_index_sha256']:raise ValueError('Source evidence changed or aliased')
    if 'source_runtime_sha256' in p:
        runtime_path=source.parent/'task-runtime.json'
        if sha(runtime_path)!=p['source_runtime_sha256']:raise ValueError('Completed source runtime changed')
        runtime=read(runtime_path)
        if runtime['status']!='completed' or runtime['job_id']!=p['source_job_id'] or runtime['code_sha']!=p['source_code_sha']:raise ValueError('Source process completion is not established')
    index=read(source/'local_artifact_index.json');cache={}
    for name,value in index.items():
        path=source/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(source):raise ValueError('Invalid source-index path')
        stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in cache:cache[key]=sha(path)
        if cache[key]!=value:raise ValueError('Original source artifact changed: '+name)
    return len(index)


def run(deployment,output,p):
    if os.environ.get('LAB_BUFFER_JOB_ID')!=JOB or int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=0:raise ValueError('Read-only audit requires exact zero-step task')
    root=Path(read(deployment)['runner_root']).resolve();source=root/'workspaces'/p['source_job_id']/'outputs/cf-optimizer';output=Path(output).resolve()
    if output!=root/'workspaces'/JOB/'outputs/cf-optimizer-audit' or output.exists():raise ValueError('Require fresh independent audit output')
    output.mkdir(parents=True);began=time.monotonic();groups=[dict(monitors=[],examples=[]) for _ in range(3)]
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',optimizer_steps_spent=0,source_job_id=p['source_job_id'],original_acceptance='blocked',
        source_compact_sha256=p['source_compact_sha256'],source_index_sha256=p['source_index_sha256'],gpu_probe_performed=False,criteria_modified=False)
    try:
        if sha(ROOT/'genrec/cf_letter_models.py')!=p['model_source_sha256']:raise ValueError('Quantizer implementation differs from original task')
        report['source_files_checked']=check_source(source,p)
        original=read(source/'compact.json')
        if original['code_sha']!=p['source_code_sha'] or original['execution_status']!='completed' or original['optimizer_steps_spent']!=5000 or original['pair_results']:raise ValueError('Unexpected completed source state')
        from genrec.cf_letter_optimizer_protocol import counters
        actual,stage_steps,uncertain=counters(source)
        if actual!=5000 or uncertain or len(stage_steps)!=5 or any(v!=1000 for v in stage_steps.values()):raise ValueError('Original durable counters differ')
        report['source_counter_steps']=actual
        import torch
        torch.set_num_threads(1)
        records=[];pending=[]
        for number,arm in enumerate(original['screen_results']):
            name=arm['case']['name'];directory=source/'arms'/name/name;summary=read(directory/'summary.json')
            for monitor in summary['monitors']:
                step=monitor['step'];paths=(directory/f'monitor_{step}.pth',directory/f'catalog_codes_{step}.npy',directory/f'semantic_codes_{step}.npy')
                values,samples=inspect_checkpoint(*paths,monitor['rq']['reconstruction_mse'])
                values.update(arm=name,step=step);group=groups[min(number//2,2)];group['monitors'].append(values)
                remaining=max(0,16-len(group['examples']));group['examples'].extend([dict(v,arm=name,step=step) for v in samples[:remaining]])
                if values['cpu_mismatch_elements'] or values['saved_full_vs_old_chunk_mismatch_elements']:pending.append((values,paths,monitor['rq']['reconstruction_mse']))
                records.append(values)
            print(json.dumps(dict(arm=name,monitors=len(summary['monitors']),cpu_mismatch_monitors=sum(v['arm']==name and v['cpu_mismatch_elements']>0 for v in records))),flush=True)
        if pending:
            from run_cf_letter_base import gpu
            from genrec.cf_letter_base_training import seed_all
            while True:
                if time.monotonic()-began>p['wall_seconds']:raise RuntimeError('Audit deadline before GPU probe')
                visible,_=gpu()
                if not visible:break
                if time.monotonic()-began>p['wall_seconds']:raise RuntimeError('Foreign GPU remained occupied; preserve partial CPU diagnosis')
                time.sleep(5)
            seed_all(2026);torch.cuda.set_device(0)
            for previous,paths,mse in pending:
                repeated,_=inspect_checkpoint(*paths,mse,gpu=True)
                for key in ('gpu_full_mismatch_elements','gpu_old_chunk_mismatch_elements'):previous[key]=repeated[key]
            report['gpu_probe_performed']=True
        report.update(execution_status='completed',source_files_rechecked=check_source(source,p),source_files_unchanged=True,
            source_reported_optimizer_steps=5000,source_pair_ran=False,monitors_checked=len(records),cpu_mismatch_monitors=sum(v['cpu_mismatch_elements']>0 for v in records),
            cpu_mismatch_elements=sum(v['cpu_mismatch_elements'] for v in records),saved_chunk_mismatch_monitors=sum(v['saved_full_vs_old_chunk_mismatch_elements']>0 for v in records),
            gpu_reproduction_mismatch_monitors=sum((v['gpu_full_mismatch_elements'] or v['gpu_old_chunk_mismatch_elements'] or 0)>0 for v in records))
        report['gpu_probed_monitors']=len(pending)
        report['outcome']='assignment_differences_measured_original_acceptance_still_blocked'
    except BaseException as error:
        report.update(execution_status='failed',error=repr(error));raise
    finally:
        report['runtime_seconds']=time.monotonic()-began
        for i,value in enumerate(groups):
            raw=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode()
            if len(raw)>65536:raise ValueError('Diagnostic group exceeds mailbox limit')
            (output/f'details_{i+1}.json').write_bytes(raw)
        report['details_sha256']={f'details_{i+1}.json':sha(output/f'details_{i+1}.json') for i in range(3)}
        save(output/'compact.json',report);print(json.dumps(report),flush=True)


def verify(context):
    c=read(context);p=read(ROOT/MANIFEST);source=Path(read(next(iter(c['contract']['input_sha256'])))['runner_root'])/'workspaces'/p['source_job_id']/'outputs/cf-optimizer'
    output=Path(c['workspace'])/'outputs/cf-optimizer-audit';r=read(output/'compact.json')
    if r['stage']!=SCOPE or r['job_id']!=JOB or r['code_sha']!=c['request']['code_sha'] or r['execution_status']!='completed' or r['optimizer_steps_spent']!=0 or r['criteria_modified'] is not False or not r['source_files_unchanged']:raise ValueError('Read-only audit scope differs')
    if check_source(source,p)!=r['source_files_checked']:raise ValueError('Original source no longer matches')
    rows=[]
    for name,value in r['details_sha256'].items():
        if sha(output/name)!=value:raise ValueError('Diagnostic evidence changed')
        rows+=read(output/name)['monitors']
    if len(rows)!=170 or len(rows)!=r['monitors_checked'] or sum(v['cpu_mismatch_elements'] for v in rows)!=r['cpu_mismatch_elements']:raise ValueError('Diagnostic count arithmetic differs')
    lines=['### Read-only diagnosis of optimizer-study checkpoint indices','',f'Job `{JOB}`; code `{r["code_sha"]}`. Optimizer steps=0. Original `{p["source_job_id"]}` artifacts are unchanged; its scientific acceptance remains **blocked**.',
        f'CPU diagnosis checked {len(rows)} saved full monitors. CPU mismatch monitors={r["cpu_mismatch_monitors"]}, mismatch elements={r["cpu_mismatch_elements"]}; saved full-catalog versus old-item chunk mismatch monitors={r["saved_chunk_mismatch_monitors"]}. GPU probes cover {r["gpu_probed_monitors"]} implicated monitors; reproduction mismatch monitors={r["gpu_reproduction_mismatch_monitors"]}.',
        '', 'Conditional float64 distances are reported on each sampled saved-prefix residual; samples distinguish exact ties, very small gaps and clearly nonminimal stored choices. No numerical tolerance, original assignment, H selection threshold or training recipe was changed. This report validates collection of diagnostic evidence; it does not accept the original optimizer study or establish a floating-point cause without reviewing the measured gaps and backend reproduction. No model training, ranking/holdout/Future scoring or automatic new experiment.']
    return dict(status='verified',job_id=JOB,contract_sha256=c['contract_sha256'],actual_optimizer_steps=0,report_markdown='\n'.join(lines))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--deployment');a.add_argument('--context');a.add_argument('--output',required=True);v=a.parse_args()
    if v.context:save(v.output,verify(v.context))
    else:run(Path(v.deployment),Path(v.output),read(ROOT/MANIFEST))
