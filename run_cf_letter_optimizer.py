"""Five bounded H-only ablations and a training-gated pair; no Runner upgrade."""
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
from genrec.cf_letter_optimizer_protocol import JOB,SCOPE,plan,cases,pair_cases,selection,transfer_control,transfer_refresh,save_anchors,load_anchors,counters,can_launch,paired_assessment,curve_groups,publish_curves
from run_cf_letter_base import gpu,paused
ROOT=Path(__file__).resolve().parent


def artifact_names(p):
    return ['compact','data_audit','selection',*curve_groups(p)]


def authorized_case(output,p,name):
    allowed=cases(p)
    if (output/'selection.json').exists():
        chosen=read(output/'selection.json')
        if chosen.get('pair_authorized') is True:
            selected=next((v for v in read(output/'screen_results.json') if v['case']['name']==chosen['selected_name']),None)
            allowed+=pair_cases(selected,p)
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


class Scheduler:
    def __init__(self,output,deployment,p,report,began):
        self.output=output;self.deployment=deployment;self.p=p;self.report=report;self.began=began
        self.root=Path(read(deployment)['runner_root']);self.stop=None;self.journal=[];self.last_control=0.;self.last_progress=0.

    def run(self,recipes):
        active={};finished={};pending=list(recipes);failure=None
        while pending or active:
            for pid,(process,case,handle) in list(active.items()):
                if process.poll() is None:continue
                handle.close();del active[pid]
                if process.returncode:
                    failure=failure or 'Arm '+case['name']+' failed; preserve '+str(self.output/'arms'/case['name']/'train.log');self.stop=failure
                else:
                    value=read(self.output/'arms'/case['name']/'arm_result.json');finished[case['name']]=value['result']
                    shutil.copyfile(self.output/'arms'/case['name']/(case['name']+'_curves.json'),self.output/(case['name']+'_curves.json'))
                    publish_curves(self.output,self.p)
                    self.report.setdefault('completed_case_names',[]).append(case['name'])
            try:
                observed,memory=gpu();self.report['max_gpu_memory_mib']=max(self.report['max_gpu_memory_mib'],memory)
                if active and observed-set(active):self.stop=self.stop or 'External GPU process appeared; active arms finish'
                if memory>self.p['gpu_memory_limit_mib']:self.stop=self.stop or 'GPU memory guard; active arms finish'
            except Exception as error:
                observed=set();memory=0;self.stop=self.stop or repr(error)
            if time.monotonic()-self.began>self.p['wall_seconds']:self.stop=self.stop or 'Four-hour guard; active arms finish'
            if time.monotonic()-self.last_control>=30:
                try:
                    if paused(self.root,JOB):self.stop=self.stop or 'Manual pause; active arms finish'
                except Exception as error:self.stop=self.stop or repr(error)
                self.last_control=time.monotonic()
            # A successful previous child may have exited between nvidia-smi and poll.
            # Its still-tracked PID is permitted; foreign PIDs never are.
            while pending and can_launch(active,observed,memory,self.p,bool(self.stop)):
                case=pending.pop(0)
                try:self.start(case,active)
                except Exception as error:
                    failure='Arm launch failed: '+repr(error);self.stop=failure;break
            if self.stop:
                save(self.output/'stop-new-arms.json',dict(reason=self.stop));pending=[]
            if time.monotonic()-self.last_progress>=30:
                total,steps,intents=counters(self.output);value=dict(active_pids=sorted(active),pending=[v['name'] for v in pending],optimizer_steps=total,stage_steps=steps,stop_new_arms=self.stop)
                save(self.output/'progress.json',value);print(json.dumps(value),flush=True);self.last_progress=time.monotonic()
            if pending or active:time.sleep(2)
        if failure:raise RuntimeError(failure)
        return [finished[c['name']] for c in recipes if c['name'] in finished]

    def start(self,case,active):
        if any(v['name']==case['name'] for v in self.journal) or sum(v['steps'] for v in self.journal)+case['steps']>self.p['optimizer_step_cap']:raise ValueError('Duplicate arm or prospective budget overflow')
        directory=self.output/'arms'/case['name'];directory.mkdir(parents=True)
        # Hardlinks only to this job's validated private copies, never to old sources.
        shutil.copytree(self.output/'inputs',directory/'inputs',copy_function=os.link)
        os.link(self.output/'prepared.npz',directory/'prepared.npz');os.link(self.output/'binding.json',directory/'binding.json')
        save(directory/'case.json',case);self.journal.append(case);save(self.output/'launch_journal.json',self.journal)
        command=[sys.executable,'-u','-B',str(ROOT/'run_cf_letter_optimizer.py'),'--deployment',str(self.deployment),'--output',str(self.output),'--case',case['name']]
        handle=(directory/'train.log').open('x')
        try:process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            handle.close();raise
        active[process.pid]=(process,case,handle);save(directory/'process.json',dict(pid=process.pid,argv=command))


def run(deployment,output,p):
    if not p['execution_authorized']:raise ValueError('No execution authorization')
    root=Path(read(deployment)['runner_root']).resolve();output=Path(output).resolve()
    if output!=root/'workspaces'/JOB/'outputs/cf-optimizer' or output.exists() or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB or int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=7000:raise ValueError('Require fresh exact-job 7000-step output')
    output.mkdir(parents=True);began=time.monotonic();binding=None
    report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',optimizer_step_cap=7000,max_gpu_memory_mib=0,
        diagnostic_only=True,gpt_training_performed=False,sasrec_training_performed=False,validation_evaluated=False,holdout_evaluated=False,future_evaluated=False,
        fixed_history_weight=p['fixed_history_weight'],screen_results=[],pair_results=[])
    for name in artifact_names(p):save(output/(name+'.json'),{} if name!='compact' else report)
    save(output/'launch_journal.json',[])
    try:
        if shutil.disk_usage(output).free<6*1024**3:raise RuntimeError('Need 6 GiB free private output space')
        base=transfer_predecessor(root,output,p);refresh=transfer_refresh(root,output,p);control,before,initial=transfer_control(root,output,p)
        data,binding=load_data(deployment,p);binding['input_sha256'].update(base);binding['input_sha256'].update(refresh);binding['input_sha256'].update(control)
        save(output/'binding.json',binding);save_anchors(output/'prepared.npz',data)
        save(output/'data_audit.json',dict(allowed_query_splits=p['allowed_query_splits'],input_sha256=binding['input_sha256'],training_items=len(binding['items']),old_catalog_items=len(binding['old_items']),anchor_train_queries=len(data['anchor_users']),prepared_sha256=sha(output/'prepared.npz'),trainable_components=['three_codebooks'],predecessor_read_only=True))
        report.update(initial=initial,reused_control=before)
        scheduler=Scheduler(output,deployment,p,report,began);screen=scheduler.run(cases(p));report['screen_results']=screen
        save(output/'screen_results.json',[before,*screen])
        complete=len(screen)==5
        chosen=selection([before,*screen],initial,p) if complete else dict(candidates=[],selected_name=None,selected_case=None,pair_authorized=False,reason='incomplete_screen')
        save(output/'selection.json',chosen);report['selection']=chosen
        selected=next((v for v in [before,*screen] if v['case']['name']==chosen['selected_name']),None)
        pair=[]
        if chosen['pair_authorized'] and not scheduler.stop:pair=scheduler.run(pair_cases(selected,p))
        report['pair_results']=pair
        report['paired_assessment']=paired_assessment([selected,*pair],p['stability']) if selected and len(pair)==2 else dict(eligible=False,reason='no_eligible_screen_or_pair_incomplete')
        report.update(execution_status='completed',resource_stop_reason=scheduler.stop,
            research_outcome='resource_stopped' if scheduler.stop else 'no_eligible_H_only_setting' if not chosen['pair_authorized'] else 'paired_verification_completed')
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
