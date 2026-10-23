"""One Runner-owned train-only study bundle; at most two isolated GPU children."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from joint_convergence import (JOB, SCOPE, identifier, matrix, read_plan, save, sha,
    select_screen, select_final)

ROOT=Path(__file__).resolve().parent


def gpu_state():
    commands=[['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
              ['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']]
    answers=[subprocess.run(c,capture_output=True,text=True,timeout=10) for c in commands]
    if any(a.returncode for a in answers):raise RuntimeError('GPU observation failed; no new child dispatch')
    pids={int(x.strip()) for x in answers[0].stdout.splitlines() if x.strip()}
    memory,utilization=[int(x.strip()) for x in answers[1].stdout.strip().split(',')]
    return dict(pids=sorted(pids),memory_mib=memory,utilization_percent=utilization,cpu_load_1m=os.getloadavg()[0])


def dispatch_paused(deployment):
    root=Path(deployment['runner_root'])
    config=json.loads((root/'config.json').read_text())
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_OPTIONAL_LOCKS='0')
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'):env.pop(key,None)
    # Only this branch is refreshed. The parent heartbeat writes lab-results;
    # avoiding FETCH_HEAD also avoids a shared ref-file race.
    branch=config['requests_branch']
    fetched=subprocess.run(['git','--git-dir='+str(root/'transport.git'),'fetch','--quiet',
        '--no-write-fetch-head','--no-tags','origin','+refs/heads/'+branch+':refs/remotes/origin/'+branch],
        capture_output=True,env=env,timeout=45)
    if fetched.returncode:raise RuntimeError('Cannot refresh pause/control state; no new dispatch')
    r=subprocess.run(['git','--git-dir='+str(root/'transport.git'),'show',
        'refs/remotes/origin/'+config['requests_branch']+':control/main.json'],capture_output=True,text=True,timeout=10)
    if r.returncode:raise RuntimeError('Cannot verify cached control state')
    state=json.loads(r.stdout)
    return state['pause_requested'] or state.get('active_job_id')!=JOB or state.get('extension',{}).get('scope_id')!=SCOPE


def expected_steps(output):
    total=0;uncertain=[]
    for p in Path(output).glob('arms/*/optimizer_steps.json'):
        count=json.loads(p.read_text())
        total+=max(count['optimizer_steps_completed'],count.get('optimizer_step_intent') or 0)
        if count.get('optimizer_step_intent'):uncertain.append(str(p))
    return total,uncertain


def paired_checks(summaries):
    controls={(s['case']['lr'],s['case']['seed']):s for s in summaries if s['case']['history_weight']==0 and s['status']=='completed'}
    checks=[]
    for s in summaries:
        if not s['case']['history_weight'] or s['status']!='completed':continue
        p=controls.get((s['case']['lr'],s['case']['seed']))
        if p is None:continue
        fields=('initial','content_sampler_sha256','anchor_sampler_sha256','anchor_queries_processed','processed_examples','valid_step')
        if any(s[k]!=p[k] for k in fields):raise ValueError('Paired initialization/sampling/monitor contract differs')
        ca,cb=dict(s['config']),dict(p['config']);ca.pop('compatibility_weight');cb.pop('compatibility_weight')
        if ca!=cb or s['binding']!=p['binding'] or s['optimizer_groups']!=p['optimizer_groups']:
            raise ValueError('Paired config/optimizer/inputs differ')
        checks.append(dict(arm=s['case']['arm_id'],control=p['case']['arm_id'],passed=True))
    return checks


class Bundle:
    def __init__(self,deployment,output,plan):
        self.deployment=Path(deployment);self.d=json.loads(self.deployment.read_text())
        self.output=Path(output);self.plan=plan;self.began=time.monotonic()
        self.deadline=self.began+plan['deadline_seconds'];self.sequence=0
        self.report=dict(stage=SCOPE,execution_status='running',job_id=JOB,
            code_sha=os.environ['LAB_BUFFER_CODE_SHA'],max_optimizer_steps=plan['max_optimizer_steps'],
            holdout_evaluated=False,base_validation_evaluated=False,future_evaluated=False,new_tiger_training_steps=0,
            phases={},owned_children=[],foreign_gpu_observed=False)

    def progress(self):
        total,uncertain=expected_steps(self.output)
        self.report.update(optimizer_steps_spent=total,uncertain_step_records=uncertain,
            elapsed_seconds=time.monotonic()-self.began)
        if total>self.plan['max_optimizer_steps']:raise ValueError('Global optimizer-step budget exceeded')
        save(self.output/'progress.json',self.report)

    def batch(self,jobs,parallel,label):
        """A child failure never kills its siblings; no new work follows an unknown fault."""
        pending=list(jobs);active={};results=[];resources=[];fault=None;began=time.monotonic();next_progress=0.;next_control=0.
        while pending or active:
            try:gpu=gpu_state()
            except Exception as error:
                fault=fault or str(error)
                gpu=dict(pids=[],memory_mib=0,utilization_percent=0,observation_failed=True)
            owned=set(active)
            foreign=set(gpu['pids'])-owned
            # The bundle parent is CPU-only and never owns a CUDA context.
            if foreign:self.report['foreign_gpu_observed']=True
            resources.append(dict(seconds=round(time.monotonic()-began,3),**gpu,owned_pids=sorted(owned)))
            if active and gpu['memory_mib']>self.plan['gpu_memory_limit_mib']:
                fault=fault or 'GPU memory guard exceeded; preserve live children, stop new dispatch'
            if time.monotonic()>self.deadline:
                fault=fault or 'Bundle deadline exceeded; preserve live children, stop new dispatch'
            if (Path(self.d['runner_root'])/'PAUSED.json').exists():
                fault=fault or 'Runner pause observed; no further child dispatch'
            if (self.output/'STOP_NEW_TASKS').exists():fault=fault or 'Explicit study dispatch stop'
            if time.monotonic()>=next_control:
                try:
                    if dispatch_paused(self.d):fault=fault or 'Control state paused or changed; stop new arm dispatch'
                except Exception as error:fault=fault or str(error)
                next_control=time.monotonic()+30
            # A foreign process present before dispatch means wait. If it appears
            # during this study, finish owned children then require reconciliation.
            if foreign and active:fault=fault or 'Foreign GPU process appeared during study'
            if time.monotonic()>=next_progress:
                live=[]
                for pid,(_,_,spec,_) in active.items():
                    directory=self.output/'arms'/spec['output_id'];entry=dict(pid=pid,arm=spec['output_id'],target=spec['target'])
                    path=directory/'optimizer_steps.json'
                    if path.exists():entry.update(json.loads(path.read_text()))
                    curve=directory/'curve.json'
                    if curve.exists():
                        last=json.loads(curve.read_text())[-1]
                        entry.update(last_monitor_step=last['step'],RQ=last['rq']['total_loss'],H=last['anchor_train']['H'])
                    live.append(entry)
                self.report['current_progress']=dict(phase=label,active=live,pending=len(pending),gpu=gpu)
                self.progress()
                print(json.dumps(self.report['current_progress'],allow_nan=False),flush=True)
                next_progress=time.monotonic()+30
            if not fault and not foreign:
                while pending and len(active)<parallel:
                    spec=pending.pop(0);case=spec['case'];directory=self.output/'arms'/spec['output_id']
                    if directory.exists() and not spec.get('resume'):
                        raise ValueError('Fresh arm output already exists: '+str(directory))
                    cmd=[sys.executable,'-u','-B',str(ROOT/'train_joint_convergence.py'),
                        '--deployment',str(self.deployment),'--output',str(directory),
                        '--case',json.dumps(case,separators=(',',':')),'--target',str(spec['target'])]
                    if spec.get('resume'):cmd.append('--resume')
                    invocation_id=JOB+'--'+spec['output_id']+'--'+str(spec['target'])
                    self.sequence+=1
                    log=self.output/'logs'/(str(self.sequence)+'-'+spec['output_id']+'.log')
                    log.parent.mkdir(exist_ok=True);handle=log.open('xb')
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1',LAB_BUFFER_ARM_JOB_ID=invocation_id),stdout=handle,stderr=subprocess.STDOUT)
                    active[proc.pid]=(proc,handle,spec,log)
                    self.report['owned_children'].append(dict(pid=proc.pid,job_id=invocation_id,case=case,argv=cmd,
                        output=str(directory),log=str(log),phase=label))
                    save(self.output/'children.json',self.report['owned_children'])
                    self.progress()
            for pid,(proc,handle,spec,log) in list(active.items()):
                if proc.poll() is None:continue
                handle.close();directory=self.output/'arms'/spec['output_id']
                summary=directory/'summary.json'
                if summary.exists():value=json.loads(summary.read_text())
                else:value=dict(status='failed',case=spec['case'],target=spec['target'],
                    error=(directory/'launch-failure.json').read_text() if (directory/'launch-failure.json').exists() else 'No arm summary')
                value.update(exit_code=proc.returncode,output_id=spec['output_id'],log=str(log))
                results.append(value);del active[pid]
                for record in self.report['owned_children']:
                    if record['pid']==pid and record.get('exit_code') is None:record['exit_code']=proc.returncode
                save(self.output/'children.json',self.report['owned_children'])
                if proc.returncode not in (0,2) or value['status'] not in ('completed','stopped_capacity_infeasible'):
                    fault=fault or 'Child failed/interrupted; no automatic retry'
                self.progress()
            if fault and not active:
                save(self.output/(label+'-resources.json'),resources)
                self.report['phases'][label]=[short(s) for s in results]
                self.report['undispatched']=[x['output_id'] for x in pending]
                raise RuntimeError(fault)
            if pending or active:time.sleep(1)
        elapsed=time.monotonic()-began
        save(self.output/(label+'-resources.json'),resources)
        self.report['phases'][label]=[short(s) for s in results]
        self.progress()
        return results,dict(runtime_seconds=elapsed,peak_total_memory_mib=max((r['memory_mib'] for r in resources),default=0),
            aggregate_steps_per_second=sum(s['target']-s.get('first_step',0) for s in results)/elapsed,resource_sha256=sha(self.output/(label+'-resources.json')))

    def execute(self):
        plan=self.plan;case=matrix(plan)[1]
        calibration=lambda prefix:[dict(case=case,output_id=prefix+str(i),target=100) for i in range(2)]
        a,serial=self.batch(calibration('cal-serial-'),1,'calibration_serial')
        b,concurrent=self.batch(calibration('cal-parallel-'),2,'calibration_parallel')
        if any(s['status']!='completed' for s in a+b):raise ValueError('Calibration arm failed; no research dispatch')
        # Numerical endpoints must also match; concurrency cannot change sampling/results.
        if any(s['content_sampler_sha256']!=a[0]['content_sampler_sha256'] or s['curve']!=a[0]['curve'] for s in a+b):
            raise ValueError('Sequential/parallel calibration differs numerically')
        gain=concurrent['aggregate_steps_per_second']/serial['aggregate_steps_per_second']
        parallel=2 if gain>=plan['min_throughput_gain'] and concurrent['peak_total_memory_mib']<=plan['gpu_memory_limit_mib'] else 1
        self.report['calibration']=dict(serial=serial,parallel=concurrent,gain=gain,selected_parallelism=parallel)
        screen,_=self.batch([dict(case=c,output_id=c['arm_id'],target=1500) for c in matrix(plan)],parallel,'screen')
        self.report['screen_pair_checks']=paired_checks(screen)
        screen_selection=select_screen(screen);self.report['screen_selection']=screen_selection
        curves(self.output/'screen-curves.json',screen)
        selected={s['case']['arm_id']:s for s in screen}
        jobs=[dict(case=selected[k]['case'],output_id=k,target=3000,resume=True) for k in screen_selection['arm_ids']]
        extended,_=self.batch(jobs,parallel,'extended') if jobs else ([],{})
        self.report['extended_pair_checks']=paired_checks(extended)
        curves(self.output/'extended-curves.json',extended)
        selection=select_final(extended);self.report['final_selection']=selection
        confirmation=[]
        if selection['selected']:
            chosen=selected[selection['selected']['arm_id']]['case']
            cases=[dict(lr=chosen['lr'],history_weight=w,seed=2027,arm_id=identifier(chosen['lr'],w,2027)) for w in (0.,chosen['history_weight'])]
            confirmation,_=self.batch([dict(case=c,output_id=c['arm_id'],target=3000) for c in cases],parallel,'confirmation')
            self.report['confirmation_pair_checks']=paired_checks(confirmation)
            self.report['confirmation']=select_final(confirmation)
        curves(self.output/'confirmation-curves.json',confirmation)
        self.report.update(execution_status='completed',research_outcome='joint_candidate_confirmed' if
            self.report.get('confirmation',{}).get('selected') else 'no_confirmed_joint_candidate_within_declared_grid')
        plot_curves(self.output)
        self.progress()


def short(s):
    keys=('status','case','target','valid_step','processed_examples','anchor_queries_processed','content_sampler_sha256',
        'anchor_sampler_sha256','checkpoint_sha256','archived_checkpoint_sha256','teacher_unchanged','checkpoint_reload_assignments_equal',
        'peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','runtime_seconds','optimizer_step_count','exit_code','output_id')
    out={k:s[k] for k in keys if k in s}
    if s.get('curve'):
        row=s['curve'][-1]
        out['endpoint']=dict(step=row['step'],rq=row['rq'],H=row['anchor_train']['H'],
            new_mse=row['groups']['update_train_first_seen']['reconstruction_loss'],capacity=row['capacity'],sid_changes=row['sid_changes'])
    if s.get('failure'):out['failure']={k:v for k,v in s['failure'].items() if k!='traceback'}
    if s.get('error'):out['error']=s['error'][-2000:]
    return out


def curves(path,summaries):
    value=dict(columns=['step','epoch','epoch_complete','RQ','H','MSE','quantization','new_item_MSE','required_suffix_capacity'],arms={})
    for s in summaries:
        value['arms'][s['case']['arm_id']]=[[r['step'],r['epoch'],r['epoch_complete'],r['rq']['total_loss'],
            r['anchor_train']['H'],r['rq']['reconstruction_loss'],r['rq']['quantization_loss'],
            r['groups']['update_train_first_seen']['reconstruction_loss'],r['capacity']['maximum_required_suffix_capacity']]
            for r in s.get('curve',[])]
    save(path,value)


def plot_curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for phase in ('screen','extended','confirmation'):
        path=Path(output)/(phase+'-curves.json')
        data=json.loads(path.read_text())
        if not data['arms']:continue
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for name,rows in data['arms'].items():
            axes[0].plot([r[0] for r in rows],[r[3] for r in rows],label=name,linewidth=1)
            axes[1].plot([r[0] for r in rows],[r[4] for r in rows],label=name,linewidth=1)
        for ax,title in zip(axes,('Full catalog RQ loss','Full anchor_train hard H')):
            ax.set(xlabel='Optimizer step',ylabel=title);ax.grid(alpha=.2)
        axes[1].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1,1))
        fig.suptitle(phase+' (training monitors only)');fig.tight_layout()
        fig.savefig(Path(output)/(phase+'-curves.svg'),bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deployment',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output).resolve();plan=read_plan(ROOT);d=json.loads(Path(a.deployment).read_text())
    code=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if (os.environ.get('LAB_BUFFER_JOB_ID')!=JOB or code!=os.environ.get('LAB_BUFFER_CODE_SHA') or
        ROOT!=(Path(d['runner_root'])/'workspaces'/JOB).resolve() or ROOT not in output.parents or output.exists()):
        raise ValueError('Require fresh code-bound Runner bundle workspace')
    output.mkdir(parents=True)
    bundle=Bundle(a.deployment,output,plan)
    protected=d['protected_input_sha256']
    try:
        if any(sha(p)!=v for p,v in protected.items()):raise ValueError('Original study input changed')
        # Mandatory real RQ/TIGER tests are repeated in the actual task environment.
        checks=output/'real-model-checks.log'
        with checks.open('x') as log:
            subprocess.run([sys.executable,'-B',str(ROOT/'tests/incremental/test_joint_convergence.py'),'--require-torch'],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        bundle.report['real_model_checks_sha256']=sha(checks)
        bundle.execute()
        if any(sha(p)!=v for p,v in protected.items()):raise ValueError('Original study inputs changed during execution')
        bundle.report['original_inputs_unchanged']=True
    except BaseException as error:
        bundle.report.update(execution_status='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        bundle.progress()
    # Bound every returned artifact; full raw curves/checkpoints/logs stay local.
    for name in ('screen-curves.json','extended-curves.json','confirmation-curves.json'):
        if not (output/name).exists():curves(output/name,[])
    save(output/'summary.json',bundle.report)
    compact=dict(bundle.report);compact.pop('owned_children',None);compact.pop('traceback',None)
    compact['full_summary_sha256']=sha(output/'summary.json')
    for name in ('screen-curves.json','extended-curves.json','confirmation-curves.json'):
        compact.setdefault('curves_sha256',{})[name]=sha(output/name)
    save(output/'compact.json',compact)
    if any((output/name).stat().st_size>64*1024 for name in ('compact.json','screen-curves.json','extended-curves.json','confirmation-curves.json')):
        raise ValueError('Study return artifact exceeds mailbox size limit; preserve local summary')
    print(json.dumps(dict(stage=SCOPE,status=compact['execution_status'],optimizer_steps=compact.get('optimizer_steps_spent'),
        outcome=compact.get('research_outcome')),allow_nan=False),flush=True)
    raise SystemExit(0 if compact['execution_status']=='completed' else 1)


if __name__=='__main__':main()
