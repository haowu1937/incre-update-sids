"""Two explicitly authorized RQ-only/Mixed pairs; reuse installed generic Runner."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_refresh_protocol import load_data,transfer_predecessor
from genrec.cf_letter_optimizer_pair_protocol import JOB,SCOPE,plan,cases,assessments,transfer_audit,curve_groups,publish_curves
from genrec.cf_letter_optimizer_protocol import transfer_refresh,save_anchors,load_anchors,counters
from run_cf_letter_optimizer import Scheduler
from run_cf_letter_base import gpu,paused
ROOT=Path(__file__).resolve().parent


def artifact_names(p):
    return ['compact','data_audit','paired_assessment',*curve_groups(p)]


def authorized_case(output,p,name):
    allowed=cases(p)
    case=next((v for v in allowed if v['name']==name),None)
    if case is None or case!=read(output/'arms'/name/'case.json'):raise ValueError('Unapproved child recipe')
    if name not in [v['name'] for v in read(output/'launch_journal.json')]:raise ValueError('Child has no durable budget claim')
    return case


def child(output,p,name):
    if not p['execution_authorized']:raise ValueError('No execution authorization')
    case=authorized_case(output,p,name)
    import torch
    from genrec.cf_letter_base_training import seed_all
    from genrec.cf_letter_optimizer_training import OptimizerEngine
    torch.set_num_threads(1);seed_all(p['seed'])
    if not torch.cuda.is_available():raise RuntimeError('Beauty optimizer study requires CUDA')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    directory=output/'arms'/name
    engine=OptimizerEngine(directory,load_anchors(directory/'prepared.npz'),read(directory/'binding.json'),p,'cuda:0')
    result=engine.run_case(case)
    result.update(code_sha=os.environ['LAB_BUFFER_CODE_SHA'],case=case,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    save(directory/'arm_result.json',result)


def run(deployment,output,p):
    if not p['execution_authorized']:raise ValueError('No execution authorization')
    root=Path(read(deployment)['runner_root']).resolve();output=Path(output).resolve()
    if output!=root/'workspaces'/JOB/'outputs/cf-optimizer-pair' or output.exists() or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB or int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=4000:raise ValueError('Require fresh exact-job 4000-step output')
    output.mkdir(parents=True);began=time.monotonic();binding=None
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',optimizer_step_cap=4000,max_gpu_memory_mib=0,
        diagnostic_only=True,gpt_training_performed=False,sasrec_training_performed=False,validation_evaluated=False,holdout_evaluated=False,future_evaluated=False,
        fixed_history_weight=p['fixed_history_weight'],pair_results=[],previous_screen_eligible=False,previous_scientific_acceptance="blocked",canonical_assignment_backend="cuda:0")
    for name in artifact_names(p):save(output/(name+'.json'),{} if name!='compact' else report)
    save(output/'launch_journal.json',[])
    try:
        if shutil.disk_usage(output).free<6*1024**3:raise RuntimeError('Need 6 GiB free private output space')
        base=transfer_predecessor(root,output,p);refresh=transfer_refresh(root,output,p);audit_evidence=transfer_audit(root,output,p)
        data,binding=load_data(deployment,p);binding['input_sha256'].update(base);binding['input_sha256'].update(refresh);binding['input_sha256'].update(audit_evidence)
        save(output/'binding.json',binding);save_anchors(output/'prepared.npz',data)
        save(output/'data_audit.json',dict(allowed_query_splits=p['allowed_query_splits'],input_sha256=binding['input_sha256'],training_items=len(binding['items']),old_catalog_items=len(binding['old_items']),anchor_train_queries=len(data['anchor_users']),prepared_sha256=sha(output/'prepared.npz'),trainable_components=['three_codebooks'],predecessor_read_only=True))
        scheduler=Scheduler(output,deployment,p,report,began,job_id=JOB,child_script='run_cf_letter_optimizer_pair.py',publisher=publish_curves)
        pairs=scheduler.run(cases(p));report['pair_results']=pairs
        report['paired_assessment']=assessments(pairs,p);save(output/'paired_assessment.json',report['paired_assessment'])
        report.update(execution_status='completed',resource_stop_reason=scheduler.stop,
            research_outcome='resource_stopped' if scheduler.stop else 'explicit_two_setting_pairs_completed')
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error));(output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        publish_curves(output,p)
        total,steps,intents=counters(output);report.update(optimizer_steps_spent=total,stage_step_counts=steps,uncertain_step_records=intents,runtime_seconds=time.monotonic()-began,
            original_inputs_unchanged=all(sha(path)==value for path,value in binding['input_sha256'].items()) if binding else None)
        # Cache identical private hardlinks while hashing every distinct local file.
        cache={};index={}
        for path in output.rglob('*'):
            if not path.is_file() or path.suffix not in ('.pth','.npy','.npz','.jsonl','.json') or path in (output/'compact.json',output/'local_artifact_index.json'):continue
            stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if key not in cache:cache[key]=sha(path)
            index[str(path.relative_to(output))]=cache[key]
        save(output/'local_artifact_index.json',index);report['local_artifact_index_sha256']=sha(output/'local_artifact_index.json')
        report['returned_artifact_sha256']={n:sha(output/(n+'.json')) for n in artifact_names(p) if n!='compact'}
        save(output/'compact.json',report)
        if any((output/(n+'.json')).stat().st_size>65536 for n in artifact_names(p)):raise ValueError('Bounded mailbox artifact exceeded 64 KiB')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case')
    a=parser.parse_args();p=plan(ROOT)
    if a.case:child(Path(a.output),p,a.case)
    else:run(Path(a.deployment),Path(a.output),p)
