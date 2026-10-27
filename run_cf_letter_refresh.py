"""Bounded shared CF refresh and frozen-quantizer diagnostic; no other fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from genrec.cf_letter_protocol import encoded,read,save,sha
from genrec.cf_letter_refresh_protocol import JOB,SCOPE,load_data,plan,transfer_predecessor
from run_cf_letter_base import counts,gpu,paused

ROOT=Path(__file__).resolve().parent


def child(role,output,p):
    import torch
    from genrec.cf_letter_base_training import seed_all
    from genrec.cf_letter_refresh_training import baseline,refresh,feature_only
    torch.set_num_threads(1);seed_all(p['seed'])
    if not torch.cuda.is_available():raise RuntimeError('Real refresh task requires CUDA')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    with np.load(output/'prepared.npz',allow_pickle=False) as file:
        data={s:{k:file[s+'_'+k] for k in ('history','targets','users','samples')} for s in ('train','val','anchors')}
        for k in ('probe','anchor_users','anchor_source_rows'):data[k]=file[k]
    result={'baseline':baseline,'sasrec':refresh,'feature_only':feature_only}[role](output,data,p,read(output/'binding.json'),'cuda:0')
    result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    save(output/role/'summary.json',result)


def run(deployment,output,p):
    d=read(deployment);root=Path(d['runner_root']).resolve();output=Path(output).resolve()
    if output!=root/'workspaces'/JOB/'outputs/cf-refresh' or output.exists() or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB:
        raise ValueError('Require fresh exact-job independent output')
    if int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=p['optimizer_step_cap']:raise ValueError('Refresh scope budget differs')
    output.mkdir(parents=True);began=time.monotonic();next_control=0.;stop=None;binding=None
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',stages={},
        optimizer_step_cap=p['optimizer_step_cap'],optimizer_steps_spent=0,max_gpu_memory_mib=0,
        quantizer_training_performed=False,gpt_training_performed=False,historical_training_performed=False,
        holdout_evaluated=False,future_evaluated=False,research_outcome='not_finished')
    save(output/'compact.json',report);save(output/'curves.json',{});save(output/'data_audit.json',dict(status='initializing'))
    try:
        if shutil.disk_usage(output).free<6*1024**3:raise RuntimeError('Insufficient independent artifact space')
        predecessor=transfer_predecessor(root,output,p);data,binding=load_data(deployment,p)
        binding['input_sha256'].update(predecessor);save(output/'binding.json',binding)
        flat={s+'_'+k:v for s in ('train','val','anchors') for k,v in data[s].items()}
        flat.update({k:data[k] for k in ('probe','anchor_users','anchor_source_rows')});np.savez(output/'prepared.npz',**flat)
        save(output/'exposure.json',{k:binding[k] for k in ('positive_exposure','history_exposure')})
        audit=dict(status='verified',allowed_query_splits=p['allowed_query_splits'],protocol_sha256=binding['protocol_sha256'],
            input_sha256=binding['input_sha256'],old_catalog_items=len(binding['old_items']),training_items=len(binding['items']),
            cumulative_training_queries=len(data['train']['targets']),update_validation_queries=len(data['val']['targets']),
            old_catalog_servable_update_validation=int(sum(data['val']['targets']>0)),probe_sample_ids=binding['probe_sample_ids'],
            anchor_train_queries=len(data['anchors']['targets']),anchor_users=len(set(data['anchor_users'])),
            anchor_sample_ids_sha256=hashlib.sha256(encoded(binding['anchor_sample_ids'])).hexdigest(),
            prepared_sha256=sha(output/'prepared.npz'),training_items_sha256=hashlib.sha256(encoded(binding['items'])).hexdigest(),
            old_items_sha256=hashlib.sha256(encoded(binding['old_items'])).hexdigest(),
            anchor_source='existing E1 anchor_train; old identities, new Base GPT reference; no holdout scoring',
            historical_target='old full SID; four digits; no EOS',history=20,unknown_positions_preserved=True,
            items_with_zero_positive_exposure=int(sum(v==0 for v in binding['positive_exposure'][1:])),
            predecessor_read_only=True,predecessor_manifest=p['predecessor'],gpt_and_quantizer_frozen=True)
        save(output/'data_audit.json',audit)
        for role in ('baseline','sasrec','feature_only'):
            while True:
                if time.monotonic()-began>p['wall_seconds']:raise RuntimeError('Refresh deadline; no next stage')
                if time.monotonic()>=next_control:
                    if paused(root,JOB):raise RuntimeError('Refresh control paused; no next stage')
                    next_control=time.monotonic()+30
                observed,memory=gpu()
                if not observed:break
                print(json.dumps(dict(stage=role,status='waiting_for_external_GPU',pids=sorted(observed))),flush=True);time.sleep(5)
            command=[sys.executable,'-u','-B',str(ROOT/'run_cf_letter_refresh.py'),'--deployment',str(deployment),'--output',str(output),'--role',role]
            with (output/(role+'.log')).open('x') as handle:
                process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                save(output/'active_child.json',dict(role=role,pid=process.pid,argv=command));last=0.
                while process.poll() is None:
                    try:
                        observed,memory=gpu();report['max_gpu_memory_mib']=max(report['max_gpu_memory_mib'],memory)
                        if observed-{process.pid}:stop=stop or 'Foreign GPU process appeared; no subsequent stage'
                        if memory>p['gpu_memory_limit_mib']:stop=stop or 'GPU memory guard exceeded; no subsequent stage'
                    except Exception as error:stop=stop or repr(error)
                    if time.monotonic()-began>p['wall_seconds']:stop=stop or 'Deadline; current stage continues'
                    if time.monotonic()>=next_control:
                        try:
                            if paused(root,JOB):stop=stop or 'Pause; current stage continues'
                        except Exception as error:stop=stop or repr(error)
                        next_control=time.monotonic()+30
                    if time.monotonic()-last>=30:
                        total,steps,intents=counts(output);progress=read(output/role/'progress.json') if (output/role/'progress.json').exists() else {}
                        value=dict(stage=role,pid=process.pid,optimizer_steps=total,stage_steps=steps,progress=progress,stop_new_stages=stop)
                        save(output/'progress.json',value);print(json.dumps(value),flush=True);last=time.monotonic()
                    time.sleep(2)
            if process.returncode:raise RuntimeError('Refresh stage failed: '+role+'; see '+str(output/(role+'.log')))
            summary=read(output/role/'summary.json');report['stages'][role]={k:v for k,v in summary.items() if k!='curve'}
            if role=='sasrec':save(output/'curves.json',dict(sasrec=summary['curve']))
            save(output/'compact.json',report)
            if stop and role!='feature_only':raise RuntimeError(stop)
        report.update(execution_status='completed',research_outcome='feature_refresh_'+report['stages']['feature_only']['outcome'],resource_stop_reason=stop)
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error));(output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        total,steps,intents=counts(output);report.update(optimizer_steps_spent=total,stage_step_counts=steps,uncertain_step_records=intents,runtime_seconds=time.monotonic()-began)
        report['original_inputs_unchanged']=all(sha(path)==value for path,value in binding['input_sha256'].items()) if binding else None
        report['local_artifact_sha256']={str(path.relative_to(output)):sha(path) for path in output.rglob('*') if path.is_file() and path.suffix in ('.pth','.npy','.npz','.jsonl')}
        report['data_audit_sha256']=sha(output/'data_audit.json');report['curves_sha256']=sha(output/'curves.json');save(output/'compact.json',report)
        for name in ('compact.json','data_audit.json','curves.json'):
            if (output/name).stat().st_size>65536:raise ValueError('Refresh artifact exceeds mailbox limit: '+name)
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--role',choices=['baseline','sasrec','feature_only']);a=parser.parse_args();p=plan(ROOT)
    if a.role:child(a.role,Path(a.output),p)
    else:run(Path(a.deployment),Path(a.output),p)
