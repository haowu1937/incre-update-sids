"""Ten RQ updates, fifteen paired GPT adaptations, global lock, serial Future."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_multiseed_protocol import JOB,SCOPE,ARTIFACTS,plan,settings,rq_cases,gpt_cases,comparison_pairs,summarize
from genrec.cf_letter_refresh_protocol import transfer_predecessor,load_data
from genrec.cf_letter_optimizer_protocol import transfer_refresh,save_anchors,load_anchors,counters as rq_counters
from genrec.cf_letter_optimizer_pair_protocol import transfer_audit
from genrec.cf_letter_downstream_protocol import load_training,load_stage,save_arrays,paired,valid_lock,digest
from run_cf_letter_optimizer import Scheduler
from run_cf_letter_downstream import counters as gpt_counters,prepared,Control
ROOT=Path(__file__).resolve().parent


def total_counters(output):
    steps={};intents=[]
    for stage,fn in [('rq',rq_counters),('gpt',gpt_counters)]:
        _,counts,unknown=fn(Path(output)/stage)
        steps.update({stage+'/'+k:v for k,v in counts.items()});intents.extend(unknown)
    return sum(steps.values()),steps,intents


def seal(output,exclude=('compact.json','local_artifact_index.json')):
    output=Path(output);index={};cache={}
    for f in output.rglob('*'):
        if not f.is_file() or f.suffix not in ('.json','.jsonl','.npy','.npz','.pth') or f.parent==output and f.name in exclude:continue
        st=f.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha(f)
        index[str(f.relative_to(output))]=cache[key]
    save(output/'local_artifact_index.json',index)
    return sha(output/'local_artifact_index.json')


def collect_training(output,p):
    return {c['name']:read(Path(output)/'arms'/c['name']/'summary.json') for c in gpt_cases(p)
            if (Path(output)/'arms'/c['name']/'summary.json').exists()}


def child(output,p,name):
    stage=output.name;matrix=rq_cases(p) if stage=='rq' else gpt_cases(p) if stage=='gpt' else []
    case=next((c for c in matrix if c['name']==name),None)
    if case is None or read(output/'arms'/name/'case.json')!=case or case not in read(output/'launch_journal.json'):
        raise ValueError('Unreserved multi-seed child')
    q=settings(ROOT,p,stage,case['seed'])
    import torch
    from genrec.cf_letter_base_training import seed_all
    torch.set_num_threads(1);seed_all(case['seed'])
    if not torch.cuda.is_available():raise RuntimeError('Beauty multi-seed training requires CUDA')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    directory=output/'arms'/name
    if stage=='rq':
        from genrec.cf_letter_optimizer_training import OptimizerEngine
        engine=OptimizerEngine(directory,load_anchors(directory/'prepared.npz'),read(directory/'binding.json'),q,'cuda:0')
        value=engine.run_case(case);value.update(code_sha=os.environ['LAB_BUFFER_CODE_SHA'],case=case)
        if not value['result']['capacity_feasible'] or value['result']['optimizer_steps']!=p['rq_steps']:
            save(directory/'arm_result.json',value);raise ValueError('Preserved infeasible/incomplete seed; no replacement seed')
    else:
        from genrec.cf_letter_downstream_training import train_arm
        train,val,probe=prepared(output);value=train_arm(output,case,train,val,probe,q,'cuda:0',lambda:None)
        value.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        save(directory/'summary.json',value);save(directory/(name+'_curves.json'),value)
        value=dict(result=value)
    save(directory/'arm_result.json',value)


def run(deployment,output,p):
    root=Path(read(deployment)['runner_root']).resolve();output=Path(output).resolve()
    if (output!=root/'workspaces'/JOB/'outputs/cf-multiseed' or output.exists()
        or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB or int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=p['optimizer_step_cap']):
        raise ValueError('Require exact fresh40000-step multi-seed workspace')
    output.mkdir(parents=True);began=time.monotonic();bound={}
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',seeds=p['seeds'],
        optimizer_steps_cap=p['optimizer_step_cap'],max_gpu_memory_mib=0,completed_case_names=[],future_names=[],
        sasrec_training_performed=False,base_training_performed=False,holdout_evaluated=False,future_used_for_selection=False)
    for n in ARTIFACTS:save(output/(n+'.json'),report if n=='compact' else {})
    rq=output/'rq';gpt=output/'gpt';rq.mkdir();gpt.mkdir();control=None
    try:
        if shutil.disk_usage(output).free<40*1024**3:raise RuntimeError('Need40GiB free private output space for15 GPT checkpoints')
        q=settings(ROOT,p,'rq')
        bound.update(transfer_predecessor(root,rq,q));bound.update(transfer_refresh(root,rq,q));bound.update(transfer_audit(root,rq,q))
        data,binding=load_data(deployment,q);bound.update(binding['input_sha256']);binding['input_sha256']=dict(bound)
        save_anchors(rq/'prepared.npz',data);save(rq/'binding.json',binding)
        schedule=Scheduler(rq,deployment,q,report,began,job_id=JOB,child_script='run_cf_letter_multiseed.py',publisher=lambda *_:None,counter=rq_counters)
        results=schedule.run(rq_cases(p))
        if schedule.stop or len(results)!=len(rq_cases(p)):raise RuntimeError(schedule.stop or 'Incomplete RQ seeds')
        q=settings(ROOT,p,'gpt');shutil.copytree(rq/'inputs',gpt/'inputs',copy_function=os.link)
        train,val,probe,binding=load_training(deployment,q);bound.update(binding['input_sha256']);binding['input_sha256']=dict(bound)
        save_arrays(gpt/'prepared.npz',{**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()},'probe':probe})
        binding['prepared_sha256']=sha(gpt/'prepared.npz');save(gpt/'binding.json',binding);save(gpt/'data_audit.json',binding)
        (gpt/'mappings').mkdir()
        for case in gpt_cases(p):
            source=rq/'inputs/base/item_sids.npy' if case['source'] is None else rq/'arms'/case['source']/case['source']/f'item_sids_{p["rq_steps"]}.npy'
            shutil.copyfile(source,gpt/'mappings'/(case['name']+'.npy'))
        schedule=Scheduler(gpt,deployment,q,report,began,job_id=JOB,child_script='run_cf_letter_multiseed.py',publisher=lambda *_:None,counter=gpt_counters)
        trained=schedule.run(gpt_cases(p))
        if schedule.stop or len(trained)!=len(gpt_cases(p)):raise RuntimeError(schedule.stop or 'Incomplete GPT seeds')
        training=collect_training(gpt,p);save(gpt/'training.json',training)
        lock=dict(job_id=JOB,code_sha=report['code_sha'],created_at=time.time(),future_used_for_selection=False,probe_sample_ids=binding['probe_sample_ids'],
            arms={n:dict(selected_step=v['selected_update_step'],selected_sha256=v['selected_sha256'],mapping_sha256=v['mapping_sha256']) for n,v in training.items()})
        valid_lock(lock,q);save(gpt/'selection_locked.json',lock);lock_sha=sha(gpt/'selection_locked.json')
        control=Control(root,q,began,job_id=JOB);control.admit()
        future,other=load_stage(deployment,q,'future_test',lock);bound.update(other)
        if set(future['samples'])&(set(train['samples'])|set(val['samples'])):raise ValueError('Overlapping train/val/Future query identities')
        save_arrays(gpt/'future_prepared.npz',future);save(gpt/'future_opened.json',dict(time=time.time(),selection_lock_sha256=lock_sha))
        binding.update(input_sha256=dict(bound),future_prepared_sha256=sha(gpt/'future_prepared.npz'));save(gpt/'data_audit.json',binding)
        import torch
        from genrec.cf_letter_downstream_training import future_arm
        from genrec.cf_letter_base_training import seed_all
        torch.set_num_threads(1);seed_all(p['cohort_seed']);torch.cuda.set_device(0);futures={}
        for case in gpt_cases(p):
            control.admit();futures[case['name']]=future_arm(gpt,case,future,settings(ROOT,p,'gpt',case['seed']),'cuda:0',lock_sha,control.observe)
            report['future_names'].append(case['name']);save(gpt/'future.json',futures);save(output/'future.json',futures)
        rows={n:[json.loads(v) for v in (gpt/'arms'/n/'future_queries.jsonl').read_text().splitlines()] for n in futures}
        comparisons={after+'_minus_'+before:paired(rows[before],rows[after],q) for before,after in comparison_pairs(p)}
        save(gpt/'comparisons.json',comparisons)
        stats=summarize(futures,p,q['topk']);stats['per_seed_user_cluster_comparisons']=comparisons;save(output/'statistics.json',stats)
        report.update(execution_status='completed',selection_locked_before_future=True,selection_lock_sha256=lock_sha,stop_new_stages_reason=control.reason)
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error));(output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        rq_values={c['name']:read(rq/'arms'/c['name']/'arm_result.json') for c in rq_cases(p) if (rq/'arms'/c['name']/'arm_result.json').exists()}
        save(output/'rq_results.json',rq_values)
        training=collect_training(gpt,p);save(gpt/'training.json',training)
        save(output/'runs.json',{n:{k:v for k,v in d.items() if k!='curve'} for n,d in training.items()})
        save(output/'selections.json',dict(arms={n:dict(selected_update_step=d['selected_update_step'],selected_sha256=d['selected_sha256'],mapping_sha256=d['mapping_sha256'],
            probe_curve=[dict(step=v['step'],ndcg10=v['metrics']['full_denominator'][settings(ROOT,p,'gpt')['selection_metric']]) for v in d['curve']]) for n,d in training.items()},
            lock_sha256=sha(gpt/'selection_locked.json') if (gpt/'selection_locked.json').exists() else None))
        save(output/'data_audit.json',dict(input_sha256=bound,cohort_seed=p['cohort_seed'],rq_prepared_sha256=sha(rq/'prepared.npz') if (rq/'prepared.npz').exists() else None,
            gpt_data_audit_sha256=sha(gpt/'data_audit.json') if (gpt/'data_audit.json').exists() else None))
        if report['execution_status']=='completed':
            n,counts,intents=gpt_counters(gpt)
            sub=dict(stage=SCOPE,job_id=JOB,code_sha=report['code_sha'],execution_status='completed',original_inputs_unchanged=True,
                selection_locked_before_future=True,future_used_for_selection=False,quantizer_training_performed=False,sasrec_training_performed=False,holdout_evaluated=False,
                returned_artifact_sha256={},selection_lock_sha256=report['selection_lock_sha256'],optimizer_steps_spent=n,stage_step_counts=counts,uncertain_step_records=intents)
            sub['local_artifact_index_sha256']=seal(gpt);save(gpt/'compact.json',sub)
        total,counts,intents=total_counters(output)
        report.update(optimizer_steps_spent=total,stage_step_counts=counts,uncertain_step_records=intents,
            original_inputs_unchanged=all(sha(f)==v for f,v in bound.items()) if bound else None,runtime_seconds=time.monotonic()-began,
            local_artifact_index_sha256=seal(output),returned_artifact_sha256={n:sha(output/(n+'.json')) for n in ARTIFACTS if n!='compact'})
        save(output/'compact.json',report)
        if any((output/(n+'.json')).stat().st_size>65536 for n in ARTIFACTS):raise ValueError('Multi-seed mailbox exceeds64KiB')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case')
    a=parser.parse_args();p=plan(ROOT)
    if a.case:child(Path(a.output),p,a.case)
    else:run(Path(a.deployment),Path(a.output),p)
