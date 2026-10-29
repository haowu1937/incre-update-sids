"""Five fixed-budget GPT updates, global selection lock, five serial Future passes."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_downstream_protocol import JOB,SCOPE,plan,cases,transfer,load_training,load_stage,save_arrays,load_arrays,valid_lock,paired,verify_zero_step_failure
from run_cf_letter_optimizer import Scheduler
from run_cf_letter_base import gpu,paused
ROOT=Path(__file__).resolve().parent
ARTIFACTS=['compact','data_audit','training','selection_locked','future','comparisons']


def counters(output):
    steps={};uncertain=[]
    for path in (Path(output)/'arms').glob('*/optimizer_steps.json'):
        r=read(path);steps[path.parent.name]=max(r['optimizer_steps_completed'],r.get('optimizer_step_intent') or 0)
        if r.get('optimizer_step_intent') is not None:uncertain.append(str(path))
    return sum(steps.values()),steps,uncertain


def publish_training(output,p):
    value={v['name']:read(Path(output)/(v['name']+'_curves.json')) for v in p['arms'] if (Path(output)/(v['name']+'_curves.json')).exists()}
    save(Path(output)/'training.json',value)
    if (Path(output)/'training.json').stat().st_size>65536:raise ValueError('Training artifact too large')


def prepared(output):
    values=load_arrays(Path(output)/'prepared.npz')
    return {k[6:]:v for k,v in values.items() if k.startswith('train_')},{k[4:]:v for k,v in values.items() if k.startswith('val_')},values['probe']


class Control:
    def __init__(self,root,p,began):self.root=root;self.p=p;self.began=began;self.reason=None;self.last=0.
    def observe(self):
        try:
            live,memory=gpu()
            if live-{os.getpid()}:self.reason=self.reason or 'Foreign GPU appeared; finish active stage only'
            if memory>self.p['gpu_memory_limit_mib']:self.reason=self.reason or 'GPU memory guard; finish active stage only'
            if time.monotonic()-self.began>self.p['wall_seconds']:self.reason=self.reason or 'Wall guard; finish active stage only'
            if time.monotonic()-self.last>30:
                if paused(self.root,JOB):self.reason=self.reason or 'Manual pause; finish active stage only'
                self.last=time.monotonic()
        except Exception as error:self.reason=self.reason or repr(error)
    def admit(self):
        while True:
            live,memory=gpu()
            # Busy before this stage: wait, preserving external work.
            if time.monotonic()-self.began>self.p['wall_seconds']:self.reason='Wall guard before next stage'
            if paused(self.root,JOB):self.reason='Manual pause before next stage'
            if self.reason:raise RuntimeError(self.reason)
            if not (live-{os.getpid()}) and memory<=self.p['gpu_memory_limit_mib']:return
            time.sleep(5)


def child(output,deployment,p,name):
    case=next((v for v in cases(p) if v['name']==name),None)
    if case is None or read(output/'arms'/name/'case.json')!=case or name not in [v['name'] for v in read(output/'launch_journal.json')]:raise ValueError('Unreserved child')
    import torch
    from genrec.cf_letter_downstream_training import train_arm
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('Beauty GPT requires CUDA')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();train,val,probe=prepared(output)
    # Parent scheduler owns cross-child GPU/pause admission. This child finishes
    # its already-launched arm; it never starts another arm or kills a process.
    value=train_arm(output,case,train,val,probe,p,'cuda:0',lambda:None)
    value.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    save(output/'arms'/name/'summary.json',value);save(output/'arms'/name/(name+'_curves.json'),value)
    save(output/'arms'/name/'arm_result.json',dict(result=value))


def run(deployment,output,p):
    root=Path(read(deployment)['runner_root']).resolve();output=Path(output).resolve()
    if (output!=root/'workspaces'/JOB/'outputs/cf-downstream' or output.exists() or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB
        or int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=10000):raise ValueError('Require exact fresh10000-step downstream workspace')
    output.mkdir(parents=True);began=time.monotonic();bound={}
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',optimizer_steps_cap=10000,
        quantizer_training_performed=False,sasrec_training_performed=False,holdout_evaluated=False,future_used_for_selection=False,
        training_names=[],future_names=[],max_gpu_memory_mib=0)
    for name in ARTIFACTS:save(output/(name+'.json'),report if name=='compact' else {})
    save(output/'launch_journal.json',[])
    try:
        import shutil
        if shutil.disk_usage(output).free<12*1024**3:raise RuntimeError('Need12GiB free for five GPT runs/checkpoints')
        bound.update(verify_zero_step_failure(root,p));bound.update(transfer(root,output,p));train,val,probe,binding=load_training(deployment,p);bound.update(binding['input_sha256']);binding['input_sha256']=dict(bound)
        arrays={**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()},'probe':probe};save_arrays(output/'prepared.npz',arrays)
        binding['prepared_sha256']=sha(output/'prepared.npz');save(output/'binding.json',binding);save(output/'data_audit.json',binding)
        scheduler=Scheduler(output,deployment,p,report,began,job_id=JOB,child_script='run_cf_letter_downstream.py',publisher=publish_training,counter=counters)
        training=scheduler.run(cases(p));report['training_names']=[v['case']['name'] for v in training]
        if scheduler.stop or len(training)!=5:raise RuntimeError(scheduler.stop or 'Incomplete five-arm adaptation')
        lock=dict(job_id=JOB,code_sha=report['code_sha'],created_at=time.time(),future_used_for_selection=False,
            probe_sample_ids=binding['probe_sample_ids'],arms={v['case']['name']:{k:v[k] for k in ('selected_update_step','selected_sha256','mapping_sha256')} for v in training})
        # Keep the manifest terminology distinct from total training updates.
        for value in lock['arms'].values():value['selected_step']=value.pop('selected_update_step')
        valid_lock(lock,p);save(output/'selection_locked.json',lock);lock_sha=sha(output/'selection_locked.json')
        control=Control(root,p,began);control.admit()
        future,new_bound=load_stage(deployment,p,'future_test',read(output/'selection_locked.json'));bound.update(new_bound)
        if set(future['samples'])&(set(train['samples'])|set(val['samples'])):raise ValueError('Future query identity leaks across stages')
        save_arrays(output/'future_prepared.npz',future);binding.update(input_sha256=bound,future_prepared_sha256=sha(output/'future_prepared.npz'))
        save(output/'data_audit.json',binding)
        save(output/'future_opened.json',dict(time=time.time(),selection_lock_sha256=lock_sha))
        import torch
        from genrec.cf_letter_base_training import seed_all
        from genrec.cf_letter_downstream_training import future_arm
        torch.set_num_threads(1);seed_all(p['seed']);torch.cuda.set_device(0)
        summaries={}
        for case in cases(p):
            control.admit();summaries[case['name']]=future_arm(output,case,future,p,'cuda:0',lock_sha,control.observe)
            report['future_names'].append(case['name']);save(output/'future.json',summaries)
        rows={name:[json.loads(line) for line in (output/'arms'/name/'future_queries.jsonl').read_text().splitlines()] for name in summaries}
        comparisons={}
        for before,after in [('plain_b09','historical_b09'),('plain_b0','historical_b0'),*[('ft_old',name) for name in summaries if name!='ft_old']]:
            comparisons[after+'_minus_'+before]=paired(rows[before],rows[after],p)
        save(output/'comparisons.json',comparisons);report.update(execution_status='completed',selection_locked_before_future=True,
            selection_lock_sha256=lock_sha,research_outcome='five_locked_GPT_Update_Future_conditions_completed',stop_new_stages_reason=control.reason)
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error));(output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        publish_training(output,p);total,steps,intents=counters(output)
        report.update(optimizer_steps_spent=total,stage_step_counts=steps,uncertain_step_records=intents,
            original_inputs_unchanged=all(sha(path)==value for path,value in bound.items()) if bound else None,runtime_seconds=time.monotonic()-began)
        index={};cache={}
        for path in output.rglob('*'):
            if not path.is_file() or path.suffix not in ('.json','.jsonl','.pth','.npy','.npz') or path in (output/'compact.json',output/'local_artifact_index.json'):continue
            stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if key not in cache:cache[key]=sha(path)
            index[str(path.relative_to(output))]=cache[key]
        save(output/'local_artifact_index.json',index);report['local_artifact_index_sha256']=sha(output/'local_artifact_index.json')
        report['returned_artifact_sha256']={name:sha(output/(name+'.json')) for name in ARTIFACTS if name!='compact'}
        save(output/'compact.json',report)
        if any((output/(n+'.json')).stat().st_size>65536 for n in ARTIFACTS):raise ValueError('Downstream mailbox exceeded64KiB')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case')
    a=parser.parse_args();p=plan(ROOT)
    if a.case:child(Path(a.output),Path(a.deployment),p,a.case)
    else:run(Path(a.deployment),Path(a.output),p)
