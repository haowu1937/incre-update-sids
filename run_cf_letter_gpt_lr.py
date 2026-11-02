"""Ten fresh GPT arms at3e-5; reuse1e-4 trajectories; Update-only comparison."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_gpt_lr_protocol import JOB,SCOPE,ARTIFACTS,plan,cases,settings,transfer,publications
from genrec.cf_letter_downstream_protocol import load_training,save_arrays
from genrec.lab_transport_gate import GitGate
from run_cf_letter_optimizer import Scheduler
from run_cf_letter_downstream import counters,prepared
from run_cf_letter_multiseed import seal
ROOT=Path(__file__).resolve().parent


def child(output,p,name):
    c=next((v for v in cases(p) if v['name']==name),None)
    if c is None or read(output/'arms'/name/'case.json')!=c or c not in read(output/'launch_journal.json'):raise ValueError('Unreserved LR child')
    import torch
    from genrec.cf_letter_downstream_training import train_arm
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('Beauty GPT LR requires CUDA')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();train,val,probe=prepared(output)
    value=train_arm(output,c,train,val,probe,settings(ROOT,p,c['seed']),'cuda:0',lambda:None)
    value.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    save(output/'arms'/name/'summary.json',value);save(output/'arms'/name/(name+'_curves.json'),value);save(output/'arms'/name/'arm_result.json',dict(result=value))


def run(deployment,output,p):
    root=Path(read(deployment)['runner_root']).resolve();output=Path(output).resolve();began=time.monotonic();bound={}
    if (output!=root/'workspaces'/JOB/'outputs/cf-gpt-lr' or output.exists() or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB or int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=10000):raise ValueError('Require fresh10000-step LR workspace')
    output.mkdir(parents=True)
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',optimizer_step_cap=10000,seeds=p['seeds'],
        quantizer_training_performed=False,sasrec_training_performed=False,base_training_performed=False,holdout_evaluated=False,future_evaluated=False,
        new_lr=3e-5,baseline_lr=1e-4,completed_case_names=[],max_gpu_memory_mib=0)
    for name in ARTIFACTS:save(output/(name+'.json'),{})
    try:
        bound=transfer(root,output,p);q=settings(ROOT,p);train,val,probe,binding=load_training(deployment,q);bound.update(binding['input_sha256']);binding['input_sha256']=bound
        save_arrays(output/'prepared.npz',{**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()},'probe':probe})
        binding['prepared_sha256']=sha(output/'prepared.npz');save(output/'binding.json',binding);save(output/'data_audit.json',binding)
        # Validate every reused prefix/selected checkpoint before any new update.
        publications(output,p)
        gate=GitGate(root,JOB,output);scheduler=Scheduler(output,deployment,q,report,began,job_id=JOB,child_script='run_cf_letter_gpt_lr.py',publisher=lambda *_:None,counter=counters,pause_check=gate.check)
        values=scheduler.run(cases(p))
        if scheduler.stop or len(values)!=len(cases(p)):raise RuntimeError(scheduler.stop or 'Incomplete LR matrix')
        report.update(execution_status='completed',stop_new_stages_reason=None)
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error));(output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        publications(output,p)
        total,counts,intents=counters(output);report.update(optimizer_steps_spent=total,stage_step_counts=counts,uncertain_step_records=intents,
            original_inputs_unchanged=all(sha(f)==v for f,v in bound.items()) if bound else None,runtime_seconds=time.monotonic()-began,
            local_artifact_index_sha256=seal(output),returned_artifact_sha256={n:sha(output/(n+'.json')) for n in ARTIFACTS if n!='compact'})
        save(output/'compact.json',report)
        if any((output/(n+'.json')).stat().st_size>65536 for n in ARTIFACTS):raise ValueError('LR result exceeds bounded mailbox')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case')
    a=parser.parse_args();p=plan(ROOT)
    if a.case:child(Path(a.output),p,a.case)
    else:run(Path(a.deployment),Path(a.output),p)
