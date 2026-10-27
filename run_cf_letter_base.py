"""One immutable Base-only task: SASRec -> CF-RQ -> decoder-only retriever."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from genrec.cf_letter_protocol import JOB,SCOPE,MANIFEST,load_base,plan,read,save,sha

ROOT=Path(__file__).resolve().parent


def counts(output):
    values={};uncertain=[]
    for path in Path(output).glob('*/optimizer_steps.json'):
        value=read(path);values[path.parent.name]=max(value['optimizer_steps_completed'],value.get('optimizer_step_intent') or 0)
        if value.get('optimizer_step_intent'):uncertain.append(str(path))
    return sum(values.values()),values,uncertain


def gpu():
    a=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    b=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if a.returncode or b.returncode:raise RuntimeError('GPU observation unavailable; stop new stages')
    return set(int(s.strip()) for s in a.stdout.splitlines() if s.strip()),int(b.stdout.strip())


def paused(root,job_id=JOB):
    root=Path(root)
    if (root/'PAUSED.json').exists():return True
    config=read(root/'config.json');branch=config['requests_branch'];repo=str(root/'transport.git')
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_OPTIONAL_LOCKS='0')
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'):env.pop(key,None)
    subprocess.run(['git','--git-dir='+repo,'fetch','--quiet','--no-write-fetch-head','origin',
        '+refs/heads/'+branch+':refs/remotes/origin/'+branch],env=env,capture_output=True,check=True,timeout=45)
    state=json.loads(subprocess.check_output(['git','--git-dir='+repo,'show','refs/remotes/origin/'+branch+':control/main.json'],env=env,timeout=10))
    return state['pause_requested'] or state['active_job_id']!=job_id


def child(role,output,p):
    import torch
    from genrec.cf_letter_base_training import seed_all,train_predictor,train_quantizer
    torch.set_num_threads(1);seed_all(p['seed'])
    if not torch.cuda.is_available():raise RuntimeError('Real Base task requires CUDA')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    binding=read(output/'binding.json')
    with np.load(output/'prepared.npz',allow_pickle=False) as file:
        data={split:{key:file[split+'_'+key] for key in ('history','targets','users','samples')} for split in ('base_train','base_val')}
        data['probe']=file['probe']
    if role=='quantizer':result=train_quantizer(output,p,binding,'cuda:0')
    else:
        mapping=np.load(output/'item_sids.npy',allow_pickle=False) if role=='gpt' else None
        result=train_predictor(role,output,data,p,binding,'cuda:0',mapping)
    result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    save(output/role/'summary.json',result)


def projection(report):
    return {k:v for k,v in report.items() if k!='curve'}


def run(deployment,output,p):
    d=read(deployment);root=Path(d['runner_root']).resolve();output=Path(output).resolve()
    expected=root/'workspaces'/JOB/'outputs/cf-base'
    if output!=expected or output.exists() or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB:
        raise ValueError('Require fresh exact-job independent output')
    if int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=p['optimizer_step_cap']:
        raise ValueError('Scope budget differs from Base manifest')
    output.mkdir(parents=True);began=time.monotonic();next_control=0.;stop=None
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',
        optimizer_steps_spent=0,optimizer_step_cap=p['optimizer_step_cap'],stages={},
        update_training_performed=False,historical_training_performed=False,future_evaluated=False,
        original_inputs_unchanged=None,research_outcome='not_finished',max_gpu_memory_mib=0)
    curves={};binding=None
    save(output/'data_audit.json',dict(status='initializing'));save(output/'curves.json',curves);save(output/'compact.json',report)
    try:
        if shutil.disk_usage(output).free<6*1024**3:raise RuntimeError('Less than 6 GiB free for independent artifacts')
        _,data,binding=load_base(deployment,p)
        binding['input_sha256'][str(Path(deployment).resolve())]=sha(deployment)
        save(output/'binding.json',binding)
        arrays={split+'_'+key:value for split,values in data.items() if split!='probe' for key,value in values.items()}
        np.savez(output/'prepared.npz',**arrays,probe=data['probe'])
        binding['prepared_sha256']=sha(output/'prepared.npz');save(output/'binding.json',binding)
        audit={k:v for k,v in binding.items() if k!='items'}
        audit.update(status='verified',catalog_items=len(binding['items']),base_train_queries=len(data['base_train']['targets']),
            base_validation_queries=len(data['base_val']['targets']),base_validation_servable=int(sum(data['base_val']['targets']>0)),
            immutable_event_id_policy=True,history=20,target_sid_tokens=4,validation_refit=False,
            sasrec_bos_carrier=True,gpt_initialization='native GPT2 random weights; no download',
            cf_exposure_counts=dict(positive_targets=np.bincount(data['base_train']['targets'],minlength=len(binding['items'])+1)[1:].tolist()))
        # Large per-item exposure counts remain local; the bounded audit reports distribution.
        frequency=np.asarray(audit.pop('cf_exposure_counts')['positive_targets'])
        save(output/'item_exposure.json',dict(positive_targets=frequency.tolist()))
        audit['positive_target_exposure']={str(q):float(np.percentile(frequency,q)) for q in (0,25,50,95,100)}
        audit['items_with_zero_positive_target_exposure']=int(sum(frequency==0))
        save(output/'data_audit.json',audit)
        for role in ('sasrec','quantizer','gpt'):
            while True:
                if time.monotonic()-began>p['wall_seconds']:stop='Base bundle deadline; no next stage'
                if time.monotonic()>=next_control:
                    if paused(root):stop='Control pause/change; no next stage'
                    next_control=time.monotonic()+30
                if stop:raise RuntimeError(stop)
                running,memory=gpu()
                if not running:break
                print(json.dumps(dict(stage=role,status='waiting_for_external_GPU',pids=sorted(running))),flush=True);time.sleep(5)
            command=[sys.executable,'-u','-B',str(ROOT/'run_cf_letter_base.py'),'--deployment',str(deployment),'--output',str(output),'--role',role]
            log=output/(role+'.log');handle=log.open('x')
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            save(output/'active_child.json',dict(role=role,pid=process.pid,argv=command))
            last=0.
            try:
                while process.poll() is None:
                    try:
                        observed,memory=gpu();report['max_gpu_memory_mib']=max(report['max_gpu_memory_mib'],memory)
                        if observed-{process.pid}:stop=stop or 'Foreign GPU work appeared; stop subsequent stages'
                        if memory>p['gpu_memory_limit_mib']:stop=stop or 'GPU memory guard exceeded; stop subsequent stages'
                    except Exception as error:stop=stop or repr(error)
                    total,stages,intents=counts(output)
                    if total>p['optimizer_step_cap']:stop=stop or 'Unexpected global optimizer accounting excess'
                    if time.monotonic()-began>p['wall_seconds']:stop=stop or 'Deadline observed; current stage continues'
                    if time.monotonic()>=next_control:
                        try:
                            if paused(root):stop=stop or 'Pause observed; current stage continues'
                        except Exception as error:stop=stop or repr(error)
                        next_control=time.monotonic()+30
                    if time.monotonic()-last>=30:
                        progress=read(output/role/'progress.json') if (output/role/'progress.json').exists() else {}
                        value=dict(stage=role,pid=process.pid,optimizer_steps=total,stage_steps=stages,progress=progress,stop_new_stages=stop)
                        save(output/'progress.json',value);print(json.dumps(value),flush=True);last=time.monotonic()
                    time.sleep(2)
            finally:handle.close()  # No signals or process termination on errors.
            if process.returncode:
                report['failed_stage']=role;raise RuntimeError('Base stage failed: '+role+'; see '+str(log))
            s=read(output/role/'summary.json');report['stages'][role]=projection(s)
            curves[role]=s['curve'];save(output/'curves.json',curves)
            save(output/'compact.json',report)
            if not s['quality_gate']:
                report['research_outcome']=role+'_quality_gate_failed';break
            if stop and role!='gpt':raise RuntimeError(stop)
        else:report['research_outcome']='base_chain_accepted' if not stop else 'base_chain_completed_with_resource_guard'
        report['resource_stop_reason']=stop
        report['execution_status']='completed'
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error),research_outcome='execution_failed')
        (output/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        total,stages,intents=counts(output)
        report.update(optimizer_steps_spent=total,stage_step_counts=stages,uncertain_step_records=intents,
            runtime_seconds=time.monotonic()-began)
        if binding:
            report['original_inputs_unchanged']=all(sha(path)==value for path,value in binding['input_sha256'].items())
        report['local_artifact_sha256']={str(path.relative_to(output)):sha(path) for path in output.rglob('*')
            if path.is_file() and path.suffix in ('.pth','.npy','.npz')}
        report['curves_sha256']=sha(output/'curves.json');report['data_audit_sha256']=sha(output/'data_audit.json')
        save(output/'compact.json',report)
        if (output/'compact.json').stat().st_size>65536:raise ValueError('Base compact exceeds mailbox limit')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--role',choices=['sasrec','quantizer','gpt'])
    args=parser.parse_args();p=plan(ROOT)
    if args.role:child(args.role,Path(args.output),p)
    else:run(Path(args.deployment),Path(args.output),p)
