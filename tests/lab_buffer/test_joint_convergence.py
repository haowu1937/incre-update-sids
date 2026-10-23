"""Exact gate, bounded pool and existing-installation upgrade wiring; no GPU."""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/lab_buffer'))
import convergence_bridge as bridge
import attach_joint_convergence as attach
import joint_convergence as protocol
import run_joint_convergence as bundle
import main_service
from runner import digest


class Gate(unittest.TestCase):
    def test_busy_cron_does_not_race_the_active_parent_git_transport(self):
        with (patch.object(main_service,'Runner') as runner,
                patch.object(main_service,'Control',side_effect=AssertionError('Concurrent Git access')),
                patch.object(sys,'argv',['main_service.py','--root','/fixture'])):
            runner.return_value.tick.return_value={'status':'busy'}
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main_service.main(),0)

    def test_exact_job_command_sha_budget_and_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code='f'*40
            (root/'convergence-deployment.json').write_text(json.dumps(dict(code_sha=code)))
            task=bridge.task(root,code);raw=json.dumps(task).encode()
            state=dict(mode='queued',pause_requested=False,active_job_id=bridge.JOB,code_sha=code,
                request_sha256=digest(raw),extension=dict(scope_id=bridge.SCOPE,status='queued',max_optimizer_steps=30400,real_model_preflight_passed=True))
            self.assertTrue(bridge.authorize(root,task,raw,state))
            for key,value in [('argv',['python','-c','print(1)']),('timeout_seconds',86401),('code_sha','e'*40),('job_id','another-job')]:
                bad=copy.deepcopy(task);bad[key]=value;encoded=json.dumps(bad).encode()
                with self.assertRaises(ValueError):bridge.authorize(root,bad,encoded,{**state,'request_sha256':digest(encoded)})
            self.assertFalse(bridge.authorize(root,task,raw,{**state,'pause_requested':True}))
            bad=copy.deepcopy(state);bad['extension']['real_model_preflight_passed']=False
            with self.assertRaises(ValueError):bridge.authorize(root,task,raw,bad)

    def test_counter_counts_rejected_steps_and_uncertain_intents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'arms/a/optimizer_steps.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(optimizer_steps_completed=418,optimizer_step_intent=419)))
            count,uncertain=bundle.expected_steps(tmp)
            self.assertEqual(count,419);self.assertEqual(len(uncertain),1)

    def test_curve_artifact_sizes_fit_real_catalog_epoch_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            summaries=[]
            for case in protocol.matrix(protocol.read_plan(ROOT)):
                rows=[]
                for step in [0,*range(46,1501,46),1500]:
                    rows.append(dict(step=step,epoch=step//46,epoch_complete=step%46==0,
                        rq=dict(total_loss=.006498323446164355,reconstruction_loss=.002695704708016656,quantization_loss=.015210474677656924),
                        anchor_train={'H':.12129605753580108},groups={'update_train_first_seen':{'reconstruction_loss':.004371601029891859}},
                        capacity={'maximum_required_suffix_capacity':256}))
                summaries.append(dict(case=case,curve=rows))
            path=Path(tmp)/'curves.json';bundle.curves(path,summaries)
            self.assertLess(path.stat().st_size,64*1024)


class Pool(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.dep=self.root/'deployment.json';self.dep.write_text(json.dumps({'runner_root':str(self.root)}))
        self.output=self.root/'output';self.output.mkdir()
        self.plan=protocol.read_plan(ROOT)
        self.env=patch.dict('os.environ',{'LAB_BUFFER_CODE_SHA':'f'*40});self.env.start();self.addCleanup(self.env.stop)
        self.b=bundle.Bundle(self.dep,self.output,self.plan)
        self.started=[];self.running=[];self.foreign=False;self.fail=False;self.clock=0
        owner=self
        class Process:
            def __init__(self,cmd,**kw):
                owner.started.append(cmd);self.pid=100+len(owner.started);self.polls=0;self.returncode=None
                owner.running.append(self)
                self.directory=Path(cmd[cmd.index('--output')+1]);self.case=json.loads(cmd[cmd.index('--case')+1])
                self.target=int(cmd[cmd.index('--target')+1]);self.directory.mkdir(parents=True)
                self.max_active=sum(p.returncode is None for p in owner.running)
            def poll(self):
                self.polls+=1
                if self.polls<2:return None
                if self.returncode is None:
                    self.returncode=1 if owner.fail and self.pid==101 else 0
                    report=dict(status='failed' if self.returncode else 'completed',case=self.case,target=self.target,first_step=0)
                    (self.directory/'summary.json').write_text(json.dumps(report))
                return self.returncode
        def gpu():
            if owner.foreign:return dict(pids=[999],memory_mib=100,utilization_percent=0)
            return dict(pids=[p.pid for p in owner.running if p.returncode is None],memory_mib=100,utilization_percent=0)
        for p in (patch.object(bundle.subprocess,'Popen',Process),patch.object(bundle,'gpu_state',side_effect=gpu),
                  patch.object(bundle,'dispatch_paused',return_value=False),patch.object(bundle.time,'sleep',return_value=None)):
            p.start();self.addCleanup(p.stop)

    def jobs(self):
        case=protocol.matrix(self.plan)[1]
        return [dict(case=case,output_id='arm'+str(i),target=100) for i in range(3)]

    def test_pool_is_bounded_and_preserves_per_arm_argv_and_outputs(self):
        results,_=self.b.batch(self.jobs(),2,'test')
        self.assertEqual(len(results),3);self.assertLessEqual(max(p.max_active for p in self.running),2)
        self.assertTrue(all(p.returncode==0 for p in self.running))
        self.assertEqual(len({str(p.directory) for p in self.running}),3)
        self.assertTrue((self.output/'children.json').is_file())

    def test_unknown_failure_finishes_siblings_and_stops_pending_work(self):
        self.fail=True
        with self.assertRaisesRegex(RuntimeError,'Child failed'):self.b.batch(self.jobs(),2,'failed')
        self.assertEqual(len(self.started),2)
        self.assertTrue(all(p.returncode is not None for p in self.running))
        self.assertEqual(self.b.report['undispatched'],['arm2'])

    def test_foreign_gpu_waits_without_launch_and_deadline_stops_dispatch(self):
        self.foreign=True;self.b.deadline=time.monotonic()-1
        with self.assertRaisesRegex(RuntimeError,'deadline'):self.b.batch(self.jobs(),2,'wait')
        self.assertEqual(self.started,[])


class Acceptance(unittest.TestCase):
    def test_terminal_curve_selection_report_receipt_and_deduplication(self):
        """Synthetic losses, real protocol arithmetic and local artifact bytes; no ML."""
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'workspaces'/bridge.JOB;output=source/'outputs/convergence';output.mkdir(parents=True)
            for name in ('joint_convergence.py','run_joint_convergence.py'):(source/name).write_bytes((ROOT/name).read_bytes())
            (source/'docs/lab_buffer').mkdir(parents=True)
            (source/'docs/lab_buffer/joint_convergence_plan.json').write_bytes((ROOT/'docs/lab_buffer/joint_convergence_plan.json').read_bytes())
            plan=protocol.read_plan(source);cases=protocol.matrix(plan)
            def make(case,target):
                directory=output/'arms'/case['arm_id'];directory.mkdir(parents=True,exist_ok=True)
                cp=directory/('checkpoint-'+str(target)+'.pth');cp.write_text('Synthetic checkpoint bytes, not a Torch claim.')
                rows=[]
                for step in [0,*range(46,target+1,46),target]:
                    row=dict(step=step,epoch=step//46,epoch_complete=step>0 and step%46==0,
                        rq=dict(total_loss=.004,reconstruction_loss=.002,quantization_loss=.008),
                        anchor_train={'H':.1 if case['history_weight']==0 else .04},
                        groups={'update_train_first_seen':{'reconstruction_loss':.006}},
                        capacity={'maximum_required_suffix_capacity':20},sid_changes={})
                    rows.append(row)
                report=dict(case=case,target=target,status='completed',valid_step=target,output_id=case['arm_id'],
                    initial=dict(rq={'total_loss':.006},groups={'update_train_first_seen':{'reconstruction_loss':.01}}),
                    curve=rows,archived_checkpoint_sha256=protocol.sha(cp))
                protocol.save(directory/('summary-'+str(target)+'.json'),report)
                protocol.save(directory/'optimizer_steps.json',dict(optimizer_steps_completed=target,optimizer_step_intent=None))
                return report
            screen=[make(c,1500) for c in cases];selection=protocol.select_screen(screen)
            extended=[make(c,3000) for c in cases if c['arm_id'] in selection['arm_ids']]
            final=protocol.select_final(extended);chosen=next(c for c in cases if c['arm_id']==final['selected']['arm_id'])
            confirmation=[make(dict(lr=chosen['lr'],history_weight=w,seed=2027,arm_id=protocol.identifier(chosen['lr'],w,2027)),3000)
                          for w in (0.,chosen['history_weight'])]
            for i in range(4):
                directory=output/'arms'/('cal-'+str(i));directory.mkdir()
                protocol.save(directory/'optimizer_steps.json',dict(optimizer_steps_completed=100,optimizer_step_intent=None))
            total,_=bundle.expected_steps(output)
            report=dict(stage=bridge.SCOPE,execution_status='completed',job_id=bridge.JOB,code_sha='f'*40,
                max_optimizer_steps=30400,optimizer_steps_spent=total,uncertain_step_records=[],
                original_inputs_unchanged=True,holdout_evaluated=False,base_validation_evaluated=False,future_evaluated=False,
                new_tiger_training_steps=0,screen_selection=selection,final_selection=final,
                confirmation=protocol.select_final(confirmation),calibration={'selected_parallelism':2},
                research_outcome='joint_candidate_confirmed',phases={})
            for phase,rows in [('screen',screen),('extended',extended),('confirmation',confirmation)]:
                report['phases'][phase]=[bundle.short(r) for r in rows]
                bundle.curves(output/(phase+'-curves.json'),rows)
            protocol.save(output/'summary.json',report)
            compact={**report,'full_summary_sha256':protocol.sha(output/'summary.json'),
                'curves_sha256':{name:protocol.sha(output/name) for name in ('screen-curves.json','extended-curves.json','confirmation-curves.json')}}
            protocol.save(output/'compact.json',compact)
            request=bridge.task(source,'f'*40);raw=json.dumps(request).encode()
            result={**request,'status':'completed','exit_code':0,'actual_code_sha':'f'*40,'request_sha256':digest(raw),
                'workspace':str(source),'artifacts':['artifacts/'+bridge.JOB+'/'+name for name in bridge.ARTIFACTS]}
            result['artifacts_sha256']={p:digest((source/name).read_bytes()) for p,name in zip(result['artifacts'],bridge.ARTIFACTS)}
            files={'jobs/'+bridge.JOB+'.json':raw}
            files.update({p:(source/name).read_bytes() for p,name in zip(result['artifacts'],bridge.ARTIFACTS)})
            state=dict(mode='queued',pause_requested=False,active_job_id=bridge.JOB,request_sha256=digest(raw),
                extension=dict(scope_id=bridge.SCOPE,status='queued'),lease=None)
            writes=[];published=[]
            class Runner:
                def ref(self,key):return key
                def read(self,ref,path):
                    if path=='jobs/'+bridge.JOB+'.json' and ref=='results_branch':return json.dumps(result).encode()
                    if path=='experiment_report.md':return b'# Existing Future results\n'
                    return files.get(path)
                def fetch(self,*args):pass
                def git(self,*args,**kw):
                    raw=(ROOT/args[1].split(':',1)[1]).read_bytes() if args[0]=='show' else b'a'*40
                    return subprocess.CompletedProcess(args,0,stdout=raw)
                def publish(self,files,*args,**kw):published.append(files)
            class Control:
                runner=Runner()
                root=Path(tmp)
                def claim(self,who):state['lease']={'token':'lease'};return 'lease'
                def read(self):return copy.deepcopy(state),'parent'
                def write(self,new,parent,extra):state.clear();state.update(new);writes.append(extra)
            c=Control();accepted=bridge.observe(c,copy.deepcopy(state))
            self.assertEqual(accepted['status'],'plan_completed')
            self.assertEqual(state['extension']['optimizer_steps_spent'],total)
            self.assertEqual(len(published),1)
            self.assertIn(b'Existing Future results',published[0]['experiment_report.md'])
            self.assertIn(b'Train-only study',published[0]['experiment_report.md'])
            self.assertTrue(writes)
            self.assertEqual(bridge.observe(c,copy.deepcopy(state))['status'],'completed')
            self.assertEqual(len(published),1)


class Attachment(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'runner';self.root.mkdir();(self.root/'bin').mkdir();(self.root/'state').mkdir()
        self.source=Path(self.tmp.name)/'code';(self.source/'scripts/lab_buffer').mkdir(parents=True);(self.source/'docs/lab_buffer').mkdir(parents=True)
        self.manifest=(ROOT/'docs/lab_buffer/joint_convergence_plan.json').read_bytes()
        (self.source/'docs/lab_buffer/joint_convergence_plan.json').write_bytes(self.manifest)
        self.old={name:('legacy '+name).encode() for name in attach.HELPERS[:-1]}
        for name,content in self.old.items():(self.root/'bin'/name).write_bytes(content)
        for name in attach.HELPERS:(self.source/'scripts/lab_buffer'/name).write_bytes(('new '+name).encode())
        (self.root/'runner-code-sha.txt').write_text(attach.INSTALLED+'\n')
        (self.root/'config.json').write_text(json.dumps(dict(python=sys.executable,main_scope_revision=2)))
        (self.root/'main-deployment.json').write_text(json.dumps(dict(runner_root=str(self.root),future_dir='/future',input_sha256={},e1_dir='/e1',retuned_dir='/retuned')))
        self.code='f'*40;self.events=[];self.fail=False
        self.state=dict(mode='plan_completed',pause_requested=False,lease=None,extension={'scope_id':'history-gradient-v1','status':'completed'},
            pending_scope=dict(scope_id=bridge.SCOPE,code_sha=self.code,max_optimizer_steps=30400,manifest_sha256=digest(self.manifest)))
        owner=self
        class Runner:
            def __init__(self,root):self.config=json.loads((Path(root)/'config.json').read_text())
            def read(self,*args):return None
            def ref(self,key):return key
            def publish(self,files):owner.events.append(files)
            def pause(self,reason):(owner.root/'PAUSED.json').write_text(reason)
        class Control:
            def __init__(self,root):pass
            def read(self):return copy.deepcopy(owner.state),'parent'
            def claim(self,owner):return claim()
            def write(self,state,parent,extra=None):owner.state=copy.deepcopy(state);owner.events.append(extra or {})
        def claim():owner.state['lease']={'token':'lease'};return 'lease'
        def output(cmd,**kw):
            if cmd[-2:]==['rev-parse','HEAD']:return owner.code+'\n'
            if cmd[-2]=='show':return owner.old[cmd[-1].rsplit('/',1)[-1]]
            raise AssertionError(cmd)
        def run(cmd,**kw):
            if cmd[0]=='git':return subprocess.CompletedProcess(cmd,0)
            self.assertEqual(cmd[0],sys.executable);self.assertEqual(cmd[-1],'--require-torch')
            kw['stdout'].write('Fixture mocks ML only; mandatory real tests remain in production.\n')
            if owner.fail:raise subprocess.CalledProcessError(1,cmd)
            return subprocess.CompletedProcess(cmd,0)
        for p in (patch.object(attach,'ROOT',self.source),patch.object(attach,'Runner',Runner),patch.object(attach,'Control',Control),
                  patch.object(attach.subprocess,'check_output',side_effect=output),patch.object(attach.subprocess,'run',side_effect=run)):
            p.start();self.addCleanup(p.stop)

    def test_upgrade_preserves_old_scope_and_queues_only_exact_new_job(self):
        result=attach.attach(self.root,self.code)
        self.assertEqual(result['status'],'attached');self.assertFalse(result['cron_changed'])
        self.assertEqual(self.state['extension']['scope_id'],bridge.SCOPE)
        self.assertEqual(self.state['completed_scope_history'][0]['scope_id'],'history-gradient-v1')
        requests=[k for e in self.events for k in e if k.startswith('jobs/')]
        self.assertEqual(requests,['jobs/'+bridge.JOB+'.json'])
        backup=Path(result['backup'])
        self.assertEqual((backup/'runner.py').read_bytes(),self.old['runner.py'])
        self.assertEqual(json.loads((self.root/'config.json').read_text())['python'],sys.executable)

    def test_failed_real_preflight_leaves_installed_helpers_unchanged(self):
        self.fail=True
        with self.assertRaises(subprocess.CalledProcessError):attach.attach(self.root,self.code)
        self.assertTrue(self.state['pause_requested'])
        for name,content in self.old.items():self.assertEqual((self.root/'bin'/name).read_bytes(),content)
        self.assertFalse(any(k.startswith('jobs/') for e in self.events for k in e))


if __name__=='__main__':unittest.main()
