"""Copy immutable partial evidence, train only three missing GPTs, then locked Future."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_multiseed_protocol import ARTIFACTS,settings,rq_cases,gpt_cases,comparison_pairs,summarize
from genrec.cf_letter_multiseed_resume_protocol import JOB,SCOPE,SOURCE_CODE,MISSING,plan,source_check,copy_indexed,check_reused,expected_counts
from genrec.cf_letter_downstream_protocol import load_stage,save_arrays,paired,valid_lock
from genrec.lab_transport_gate import GitGate
from run_cf_letter_optimizer import Scheduler
from run_cf_letter_downstream import counters,prepared
from run_cf_letter_multiseed import total_counters,seal,collect_training
from run_cf_letter_base import gpu
ROOT=Path(__file__).resolve().parent


class StageControl:
    def __init__(self,gate,p,began):self.gate=gate;self.p=p;self.began=began;self.reason=None;self.last_check=0.
    def observe(self):
        live,memory=gpu()
        if live-{os.getpid()}:self.reason=self.reason or 'Foreign GPU; finish active stage'
        if memory>self.p['gpu_memory_limit_mib']:self.reason=self.reason or 'GPU memory guard'
        if time.monotonic()-self.began>self.p['wall_seconds']:self.reason=self.reason or 'Wall guard'
        if time.monotonic()-self.last_check>=30:
            if self.gate.check() is True:self.reason=self.reason or 'Manual pause'
            self.last_check=time.monotonic()
    def admit(self):
        while True:
            if self.reason:raise RuntimeError(self.reason)
            if time.monotonic()-self.began>self.p['wall_seconds']:raise RuntimeError('Wall guard before next stage')
            decision=self.gate.check()
            if decision is True:raise RuntimeError('Manual pause before next stage')
            live,memory=gpu()
            if decision is False and not(live-{os.getpid()}) and memory<=self.p['gpu_memory_limit_mib']:return
            time.sleep(5)


def child(output,p,name):
    if name not in MISSING:raise ValueError('Only three missing GPT arms authorized')
    case=next(c for c in gpt_cases(p) if c['name']==name)
    if read(output/'arms'/name/'case.json')!=case or case not in read(output/'launch_journal.json'):raise ValueError('Unreserved continuation child')
    import torch
    from genrec.cf_letter_downstream_training import train_arm
    from genrec.cf_letter_base_training import seed_all
    torch.set_num_threads(1);seed_all(case['seed'])
    if not torch.cuda.is_available():raise RuntimeError('Continuation requires production CUDA')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();train,val,probe=prepared(output)
    value=train_arm(output,case,train,val,probe,settings(ROOT,p,'gpt',case['seed']),'cuda:0',lambda:None)
    value.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    save(output/'arms'/name/'summary.json',value);save(output/'arms'/name/(name+'_curves.json'),value)
    save(output/'arms'/name/'arm_result.json',dict(result=value))


def run(deployment,output,m,p):
    root=Path(read(deployment)['runner_root']).resolve();output=Path(output).resolve();began=time.monotonic()
    if (output!=root/'workspaces'/JOB/'outputs/cf-multiseed-resume' or output.exists()
        or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB or int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=6000):raise ValueError('Require fresh6000-step continuation workspace')
    output.mkdir(parents=True);rq=output/'rq';gpt=output/'gpt';bound={};source=None
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',seeds=p['seeds'],
        optimizer_steps_cap=40000,new_optimizer_step_cap=6000,inherited_optimizer_steps=34000,
        max_gpu_memory_mib=0,completed_case_names=[],future_names=[],sasrec_training_performed=False,base_training_performed=False,
        holdout_evaluated=False,future_used_for_selection=False)
    for name in ARTIFACTS:save(output/(name+'.json'),{})
    try:
        source,index=source_check(root,m,p)
        # Estimate once per old inode; never link to the original outputs.
        unique={}
        for name in index:
            s=(source/name).stat();unique[(s.st_dev,s.st_ino)]=s.st_size
        if shutil.disk_usage(output).free<sum(unique.values())+8*1024**3:raise RuntimeError('Insufficient disk for private copy plus three GPTs')
        copy_indexed(source,output,{n:v for n,v in index.items() if len(Path(n).parts)>1})
        save(output/'provenance/source_compact.json',read(source/'compact.json'))
        save(output/'provenance/source_index.json',index)
        bound=read(source/'data_audit.json')['input_sha256']
        for f in (source/'compact.json',source/'local_artifact_index.json',source.parent/'task-runtime.json'):bound[str(f)]=sha(f)
        # The copied failure marker is retained as provenance, not a current stop.
        if (gpt/'stop-new-arms.json').exists():
            shutil.move(str(gpt/'stop-new-arms.json'),str(output/'provenance/source_stop.json'))
        save(output/'provenance/recovery.json',dict(source_job_id=m['source_job_id'],source_code_sha=SOURCE_CODE,
            source_index_sha256=sha(source/'local_artifact_index.json'),inherited_steps=34000,missing_gpt_arms=MISSING,new_step_cap=6000))
        q=settings(ROOT,p,'gpt');gate=GitGate(root,JOB,output)
        scheduler=Scheduler(gpt,deployment,q,report,began,job_id=JOB,child_script='run_cf_letter_multiseed_resume.py',publisher=lambda *_:None,counter=counters,pause_check=gate.check)
        scheduler.journal=read(gpt/'launch_journal.json')
        inherited=[c for c in gpt_cases(p) if c['name'] not in MISSING]
        if scheduler.journal!=inherited:raise ValueError('Original GPT launch sequence differs')
        results=scheduler.run([c for c in gpt_cases(p) if c['name'] in MISSING])
        if scheduler.stop or len(results)!=3:raise RuntimeError(scheduler.stop or 'Incomplete continuation')
        check_reused(output);source_check(root,m,p)
        total,counts,intents=total_counters(output)
        if total!=40000 or counts!=expected_counts(p,True) or intents:raise ValueError('Family cap or durable counts differ')
        training=collect_training(gpt,p);save(gpt/'training.json',training)
        binding=read(gpt/'data_audit.json');bound.update(binding['input_sha256'])
        lock=dict(job_id=JOB,code_sha=report['code_sha'],created_at=time.time(),future_used_for_selection=False,probe_sample_ids=binding['probe_sample_ids'],
            arms={n:dict(selected_step=v['selected_update_step'],selected_sha256=v['selected_sha256'],mapping_sha256=v['mapping_sha256']) for n,v in training.items()})
        valid_lock(lock,q);save(gpt/'selection_locked.json',lock);lock_sha=sha(gpt/'selection_locked.json')
        control=StageControl(gate,q,began);control.admit()
        future,other=load_stage(deployment,q,'future_test',lock);bound.update(other);train,val,probe=prepared(gpt)
        if set(future['samples'])&(set(train['samples'])|set(val['samples'])):raise ValueError('Future overlaps fitting or selection')
        save_arrays(gpt/'future_prepared.npz',future);save(gpt/'future_opened.json',dict(time=time.time(),selection_lock_sha256=lock_sha))
        binding.update(input_sha256=bound,future_prepared_sha256=sha(gpt/'future_prepared.npz'));save(gpt/'data_audit.json',binding)
        import torch
        from genrec.cf_letter_base_training import seed_all
        from genrec.cf_letter_downstream_training import future_arm
        torch.set_num_threads(1);seed_all(p['cohort_seed']);torch.cuda.set_device(0);futures={}
        for case in gpt_cases(p):
            control.admit();futures[case['name']]=future_arm(gpt,case,future,settings(ROOT,p,'gpt',case['seed']),'cuda:0',lock_sha,control.observe)
            report['future_names'].append(case['name']);save(gpt/'future.json',futures);save(output/'future.json',futures)
        rows={n:[json.loads(v) for v in (gpt/'arms'/n/'future_queries.jsonl').read_text().splitlines()] for n in futures}
        comparisons={a+'_minus_'+b:paired(rows[b],rows[a],q) for b,a in comparison_pairs(p)};save(gpt/'comparisons.json',comparisons)
        stats=summarize(futures,p,q['topk']);stats['per_seed_user_cluster_comparisons']=comparisons;save(output/'statistics.json',stats)
        report.update(execution_status='completed',selection_locked_before_future=True,selection_lock_sha256=lock_sha,stop_new_stages_reason=control.reason)
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error));(output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        training=collect_training(gpt,p)
        save(output/'rq_results.json',{c['name']:read(rq/'arms'/c['name']/'arm_result.json') for c in rq_cases(p) if (rq/'arms'/c['name']/'arm_result.json').exists()})
        save(output/'runs.json',{n:{k:v for k,v in d.items() if k!='curve'} for n,d in training.items()})
        save(output/'selections.json',dict(arms={n:dict(selected_update_step=d['selected_update_step'],selected_sha256=d['selected_sha256'],mapping_sha256=d['mapping_sha256'],
            probe_curve=[dict(step=v['step'],ndcg10=v['metrics']['full_denominator']['ndcg@10']) for v in d['curve']]) for n,d in training.items()},lock_sha256=sha(gpt/'selection_locked.json') if (gpt/'selection_locked.json').exists() else None))
        save(output/'data_audit.json',dict(input_sha256=bound,cohort_seed=p['cohort_seed'],rq_prepared_sha256=sha(rq/'prepared.npz') if (rq/'prepared.npz').exists() else None,
            gpt_data_audit_sha256=sha(gpt/'data_audit.json') if (gpt/'data_audit.json').exists() else None))
        if report['execution_status']=='completed':
            n,counts,intents=counters(gpt)
            sub=dict(stage=SCOPE,job_id=JOB,code_sha=report['code_sha'],execution_status='completed',original_inputs_unchanged=True,selection_locked_before_future=True,
                future_used_for_selection=False,quantizer_training_performed=False,sasrec_training_performed=False,holdout_evaluated=False,returned_artifact_sha256={},
                selection_lock_sha256=report['selection_lock_sha256'],optimizer_steps_spent=n,stage_step_counts=counts,uncertain_step_records=intents,local_artifact_index_sha256=seal(gpt))
            save(gpt/'compact.json',sub)
        total,counts,intents=total_counters(output);new=sum(counts.get('gpt/'+n,0) for n in MISSING)
        report.update(optimizer_steps_spent=total,new_optimizer_steps_spent=new,stage_step_counts=counts,uncertain_step_records=intents,
            original_inputs_unchanged=all(sha(f)==v for f,v in bound.items()) if bound else None,runtime_seconds=time.monotonic()-began,
            local_artifact_index_sha256=seal(output),returned_artifact_sha256={n:sha(output/(n+'.json')) for n in ARTIFACTS if n!='compact'})
        save(output/'compact.json',report)
        if any((output/(n+'.json')).stat().st_size>65536 for n in ARTIFACTS):raise ValueError('Continuation artifact exceeds64KiB')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case')
    a=parser.parse_args();m,p=plan(ROOT)
    if a.case:child(Path(a.output),p,a.case)
    else:run(Path(a.deployment),Path(a.output),m,p)
