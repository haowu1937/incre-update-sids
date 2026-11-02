"""Three new TIGER fits; four paired Hungarian/unmapped Future conditions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from genrec.cf_letter_protocol import read,save,sha
from genrec.tiger_hungarian_protocol import JOB,SCOPE,NAMES,plan,cases,bounded,projections
from genrec.lab_transport_gate import GitGate
from run_cf_letter_optimizer import Scheduler
from run_cf_letter_base import gpu
from run_cf_letter_downstream import counters
from run_cf_letter_multiseed import seal
ROOT=Path(__file__).resolve().parent


class TIGERScheduler(Scheduler):
    def start(self,case,active):
        if case in self.journal or sum(c['steps'] for c in self.journal)+case['steps']>self.p['optimizer_step_cap']:raise ValueError('Duplicate/overbudget TIGER claim')
        folder=self.output/'arms'/case['name'];folder.mkdir(parents=True,exist_ok=False);save(folder/'case.json',case)
        self.journal.append(case);save(self.output/'launch_journal.json',self.journal)
        command=[sys.executable,'-u','-B',str(ROOT/'run_tiger_hungarian.py'),'--deployment',str(self.deployment),'--output',str(self.output),'--case',case['name']]
        handle=(folder/'train.log').open('x')
        try:process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:handle.close();raise
        active[process.pid]=(process,case,handle);save(folder/'process.json',dict(pid=process.pid,argv=command))


def stage(deployment,output,p,gate,began,phase,name=None):
    # Only this job's owned process is started; never terminate another process.
    while True:
        if time.monotonic()-began>p['wall_seconds']:raise RuntimeError('20h stop-new-stage guard')
        decision=gate.check()
        if decision is True:raise RuntimeError('Paused; no next TIGER stage')
        live,memory=gpu()
        if decision is False and not live and memory<=p['gpu_memory_limit_mib']:break
        save(output/'stage_progress.json',dict(status='waiting_for_gpu_or_git',phase=phase,case=name));time.sleep(2)
    command=[sys.executable,'-u','-B',str(ROOT/'run_tiger_hungarian.py'),'--deployment',str(deployment),'--output',str(output),'--phase',phase]
    if name:command+=['--case',name]
    log=output/(phase+('_'+name if name else '')+'.log')
    with log.open('x') as handle:
        process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);paused=False;last=0.
        save(output/'stage_progress.json',dict(status='running',phase=phase,case=name,pid=process.pid))
        while process.poll() is None:
            if time.monotonic()-last>=30:
                paused=paused or gate.check() is True;last=time.monotonic()
                print(json.dumps(dict(phase=phase,case=name,pid=process.pid,stop_new_stages=paused)),flush=True)
            time.sleep(2)
    if process.returncode:raise RuntimeError('Stage failed; preserve '+str(log))
    if paused:raise RuntimeError('Paused during stage; active stage finished, no next launch')


def execute(deployment,output,p):
    root=Path(read(deployment)['runner_root']).resolve();output=Path(output).resolve();began=time.monotonic()
    if output!=root/'workspaces'/JOB/'outputs/tiger-hungarian' or output.exists() or os.environ.get('LAB_BUFFER_JOB_ID')!=JOB or int(os.environ.get('LAB_SCOPE_STEP_LIMIT','-1'))!=6000:raise ValueError('Fresh exact6000-step workspace required')
    output.mkdir(parents=True);report=dict(job_id=JOB,scope_id=SCOPE,code_sha=os.environ['LAB_BUFFER_CODE_SHA'],execution_status='running',completed_case_names=[],max_gpu_memory_mib=0,
        rqvae_training_performed=False,base_tiger_training_performed=False,holdout_evaluated=False,future_used_for_matching_or_selection=False,reused_control='historical_unmapped',future_passes_new=0)
    for name in NAMES:bounded(output/(name+'.json'),{})
    try:
        gate=GitGate(root,JOB,output);stage(deployment,output,p,gate,began,'prepare')
        scheduler=TIGERScheduler(output,deployment,p,report,began,job_id=JOB,child_script='run_tiger_hungarian.py',publisher=lambda out,recipe:projections(out,recipe),counter=counters,pause_check=gate.check)
        values=scheduler.run(cases(p))
        if scheduler.stop or len(values)!=3:raise RuntimeError(scheduler.stop or 'Incomplete TIGER arms')
        arms={}
        for name in p['arms']:
            folder=output/('reused' if name=='historical_unmapped' else 'arms')/name;s=read(folder/'summary.json')
            arms[name]=dict(selected_sha256=sha(folder/'best.pth'),selected_step=s['best_update_optimizer_step'],mapping_sha256=sha(output/'maps'/(name+'_deployment.json')))
        save(output/'selection_locked.json',dict(arms=arms,future_used_for_selection=False,selection_split='update_val',selection_metric='ndcg@10',created_at=time.time()))
        for name in p['new_training_arms']:
            stage(deployment,output,p,gate,began,'future',name);report['future_passes_new']+=1
        from genrec.tiger_hungarian_experiment import comparisons
        bounded(output/'comparisons.json',comparisons(output,p));report.update(execution_status='completed',selection_locked_before_new_future=True)
    except BaseException as error:
        report.update(execution_status='failed',error=type(error).__name__+': '+str(error));(output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        projections(output,p);total,steps,intents=counters(output)
        bound=read(output/'audit_full.json')['input_sha256'] if (output/'audit_full.json').exists() else {}
        report.update(optimizer_steps_spent=total,stage_step_counts=steps,uncertain_step_records=intents,original_inputs_unchanged=all(sha(f)==h for f,h in bound.items()) if bound else None,
            runtime_seconds=time.monotonic()-began,local_artifact_index_sha256=seal(output),returned_artifact_sha256={n:sha(output/(n+'.json')) for n in NAMES if n!='compact'})
        bounded(output/'compact.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case');parser.add_argument('--phase',choices=['prepare','train','future'],default='train')
    args=parser.parse_args();p=plan();output=Path(args.output)
    if not args.case and args.phase=='train':execute(Path(args.deployment),output,p)
    else:
        import torch
        from genrec import tiger_hungarian_experiment as experiment
        if not torch.cuda.is_available():raise RuntimeError('Real Beauty TIGER task requires CUDA')
        torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device('cuda:0')
        if args.phase=='prepare':experiment.prepare(args.deployment,output,p,device)
        elif args.phase=='future':experiment.evaluate_future(args.deployment,output,p,args.case,device)
        else:
            case=next((c for c in cases(p) if c['name']==args.case),None)
            if case is None or case not in read(output/'launch_journal.json') or case!=read(output/'arms'/args.case/'case.json'):raise ValueError('Unreserved TIGER arm')
            experiment.train(args.deployment,output,p,args.case,device)
