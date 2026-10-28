"""One 3000-step training-only CF history-gradient task; preserve all prior results."""
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
from genrec.cf_letter_refresh_protocol import load_data,transfer_predecessor
from genrec.cf_letter_history_long_protocol import JOB,SCOPE,plan,transfer_refresh,save_anchors,load_anchors,verify_zero_step_preflight
from run_cf_letter_base import counts,gpu,paused

ROOT=Path(__file__).resolve().parent


def child(output,p):
    if p.get('execution_authorized') is not True:raise ValueError('Long diagnostic has no explicit execution authorization')
    import torch
    from genrec.cf_letter_base_training import seed_all
    from genrec.cf_letter_history_long import Engine
    torch.set_num_threads(1);seed_all(p['seed'])
    if not torch.cuda.is_available():raise RuntimeError('Real CF direction diagnostic requires CUDA')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    data=load_anchors(output/'prepared.npz')
    engine=Engine(output,data,read(output/'binding.json'),p,'cuda:0');result=engine.run()
    result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    save(output/'diagnostic_summary.json',result)


def run(deployment,output,p):
    if p.get('execution_authorized') is not True:raise ValueError('Long diagnostic has no explicit execution authorization')
    d=read(deployment);root=Path(d['runner_root']).resolve();output=Path(output).resolve()
    if output!=root/'workspaces'/JOB/'outputs/cf-history-long' or output.exists() or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB:raise ValueError('Require fresh exact-job direction output')
    if int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=3000:raise ValueError('Direction scope budget differs')
    output.mkdir(parents=True);began=time.monotonic();binding=None;stop=None
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',optimizer_step_cap=3000,optimizer_steps_spent=0,
        diagnostic_only=True,gpt_training_performed=False,sasrec_training_performed=False,downstream_training_performed=False,validation_evaluated=False,holdout_evaluated=False,future_evaluated=False,max_gpu_memory_mib=0)
    for name in ('compact','data_audit','gradient_calibration','h_only_curves','rq_only_curves','mixed_curves'):save(output/(name+'.json'),report if name=='compact' else {})
    try:
        if shutil.disk_usage(output).free<6*1024**3:raise RuntimeError('Insufficient independent output space')
        proof=verify_zero_step_preflight(root,p)
        base=transfer_predecessor(root,output,p);refresh=transfer_refresh(root,output,p);data,binding=load_data(deployment,p)
        binding['input_sha256'].update(base);binding['input_sha256'].update(refresh);binding['input_sha256'].update(proof);save(output/'binding.json',binding)
        save_anchors(output/'prepared.npz',data)
        audit=dict(status='verified',allowed_query_splits=p['allowed_query_splits'],input_sha256=binding['input_sha256'],protocol_sha256=binding['protocol_sha256'],
            training_items=len(binding['items']),old_catalog_items=len(binding['old_items']),anchor_train_queries=len(data['anchor_users']),anchor_users=len(set(data['anchor_users'])),
            anchor_sample_ids_sha256=hashlib.sha256(encoded(binding['anchor_sample_ids'])).hexdigest(),training_items_sha256=hashlib.sha256(encoded(binding['items'])).hexdigest(),
            prepared_sha256=sha(output/'prepared.npz'),old_reference='original Base GPT, original E0 full target and masks; four digits; no EOS',
            predecessor_read_only=True,distance_scales='Base E0 median row-distance std; first 4096 rows; frozen',temperature=.5,incremental_dead_code_reset=False,trainable_components=['three_codebooks'])
        save(output/'data_audit.json',audit)
        while True:
            if time.monotonic()-began>p['wall_seconds'] or paused(root,JOB):raise RuntimeError('Paused/deadline before direction command')
            observed,_=gpu()
            if not observed:break
            print(json.dumps(dict(status='waiting_for_external_GPU',pids=sorted(observed))),flush=True);time.sleep(5)
        command=[sys.executable,'-u','-B',str(ROOT/'run_cf_letter_history_long.py'),'--deployment',str(deployment),'--output',str(output),'--child']
        with (output/'diagnose.log').open('x') as handle:
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            save(output/'active_child.json',dict(pid=process.pid,argv=command));last=0.;next_control=0.
            while process.poll() is None:
                try:
                    observed,memory=gpu();report['max_gpu_memory_mib']=max(report['max_gpu_memory_mib'],memory)
                    if observed-{process.pid}:stop=stop or 'Foreign GPU process; stop subsequent branches'
                    if memory>p['gpu_memory_limit_mib']:stop=stop or 'GPU memory guard; stop subsequent branches'
                except Exception as error:stop=stop or repr(error)
                if time.monotonic()-began>p['wall_seconds']:stop=stop or 'Deadline; current branch continues'
                if time.monotonic()>=next_control:
                    try:
                        if paused(root,JOB):stop=stop or 'Manual pause; current branch continues'
                    except Exception as error:stop=stop or repr(error)
                    next_control=time.monotonic()+30
                if stop and not (output/'stop-new-branches.json').exists():save(output/'stop-new-branches.json',dict(reason=stop))
                if time.monotonic()-last>=30:
                    total,steps,intents=counts(output);progress=read(output/'branch_progress.json') if (output/'branch_progress.json').exists() else {}
                    value=dict(pid=process.pid,optimizer_steps=total,branches=progress,stop_new_branches=stop);save(output/'progress.json',value);print(json.dumps(value),flush=True);last=time.monotonic()
                time.sleep(2)
        if process.returncode:raise RuntimeError('Direction child failed; see '+str(output/'diagnose.log'))
        report.update(read(output/'diagnostic_summary.json'));report.update(execution_status='completed',research_outcome=report['outcome'],resource_stop_reason=stop)
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error));(output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        total,steps,intents=counts(output);report.update(optimizer_steps_spent=total,stage_step_counts=steps,uncertain_step_records=intents,runtime_seconds=time.monotonic()-began)
        report['original_inputs_unchanged']=all(sha(path)==value for path,value in binding['input_sha256'].items()) if binding else None
        index={str(path.relative_to(output)):sha(path) for path in output.rglob('*') if path.is_file() and path.suffix in ('.pth','.npy','.npz','.jsonl','.json') and path.name not in ('compact.json','local_artifact_index.json')}
        save(output/'local_artifact_index.json',index);report['local_artifact_index_sha256']=sha(output/'local_artifact_index.json')
        report['returned_artifact_sha256']={name:sha(output/(name+'.json')) for name in ('data_audit','gradient_calibration','h_only_curves','rq_only_curves','mixed_curves')}
        save(output/'compact.json',report)
        for name in ('compact','data_audit','gradient_calibration','h_only_curves','rq_only_curves','mixed_curves'):
            if (output/(name+'.json')).stat().st_size>65536:raise ValueError('Direction artifact exceeds mailbox limit: '+name)
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True);parser.add_argument('--child',action='store_true')
    a=parser.parse_args();p=plan(ROOT)
    if a.child:child(Path(a.output),p)
    else:run(Path(a.deployment),Path(a.output),p)
