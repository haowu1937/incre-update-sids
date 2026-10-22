"""Existing-trace validation and real old-Runner→new-Runner Git handoff; no ML."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/lab_buffer'))
import gradient_logs as logs
import gradient_bridge as bridge
import main_control as control
import runner
import test_runner as fixtures

def row(step):
    r = dict(step=step, history_penalty=.1, anchor_mean_delta_sequence_nll=.02,
             global_gradient_norm_before_clip=.2, gradient_clipped=True)
    if step==1 or step%25==0:
        r['gradient_components'] = {}
        for name in ('encoder','decoder','codebooks'):
            hn = 0. if name=='decoder' else 2.
            r['gradient_components'][name] = dict(rq_gradient_norm=1.,history_gradient_norm=hn,
                weighted_history_gradient_norm=1e-4*hn,weighted_history_over_rq=1e-4*hn,
                cosine=-.5 if hn else None)
    return r


class Trace(unittest.TestCase):
    def test_probe_denominators_clipping_and_batch_windows(self):
        s=logs.summarize([row(i) for i in range(1,1001)],1000,1e-4)
        self.assertEqual(len(s['gradient_probe_steps']),41)
        self.assertEqual(s['gradient_components']['encoder']['negative_cosine_fraction'],1.)
        self.assertEqual(s['gradient_components']['decoder']['zero_history_gradient_fraction'],1.)
        self.assertIsNone(s['gradient_components']['decoder']['cosine']['mean'])
        self.assertEqual(s['clipping']['clipped_fraction'],1.)
        self.assertEqual([x['records'] for x in s['training_batch_windows']],[100]*10)

    def test_missing_steps_nonfinite_and_wrong_weight_rejected(self):
        rows=[row(i) for i in range(1,101)]
        for bad in [rows[:-1],rows+rows[:1]]:
            with self.assertRaises(ValueError): logs.summarize(bad,100,1e-4)
        for key,value in [('history_penalty',float('nan')),('gradient_clipped',None)]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError): logs.summarize(bad,100,1e-4)
        with self.assertRaises(ValueError): logs.summarize(rows,100,1e-3)

    def test_ratios_are_sampled_observations_not_update_evidence(self):
        s=logs.summarize([row(i) for i in range(1,26)],25,1e-4)
        self.assertEqual(s['gradient_components']['encoder']['probed_steps'],2)
        self.assertIn('No fixed-checkpoint H curve',s['scope'])
        self.assertLess(len(json.dumps(s)),64*1024)


class Handoff(unittest.TestCase):
    task=fixtures.MailboxTest.task
    submit=fixtures.MailboxTest.submit
    def setUp(self):
        fixtures.MailboxTest.setUp(self)
        self.project=self.base/'project';self.project.mkdir()
        self.retuned=self.project/'retuned';self.retuned.mkdir()
        self.bin=self.root/'bin';self.bin.mkdir()
        legacy={}
        for name in ('runner.py','main_control.py','main_service.py'):
            raw=subprocess.check_output(['git','-C',str(ROOT),'show','02eca7996ccddc2b0e35c47cf4cd00aafe200291:scripts/lab_buffer/'+name])
            (self.bin/name).write_bytes(raw);legacy[name]=hashlib.sha256(raw).hexdigest()
        config=json.loads((self.root/'config.json').read_text())
        config.update(main_integration=True,allowed_kinds=['smoke','experiment'])
        runner.atomic_json(self.root/'config.json',config)
        fake=self.base/'tools';fake.mkdir()
        (fake/'nvidia-smi').write_text('#!/bin/sh\nexit 0\n');(fake/'nvidia-smi').chmod(0o755)
        self.env=dict(os.environ,PATH=str(fake)+os.pathsep+os.environ['PATH'])
        fixtures.git(self.author,'checkout','vibe-dev')
        for name in ('runner.py','main_control.py','main_service.py','accept_main.py','gradient_bridge.py','gradient_logs.py'):
            (self.author/'scripts/lab_buffer'/name).write_bytes((ROOT/'scripts/lab_buffer'/name).read_bytes())
        (self.author/'docs/lab_buffer').mkdir(parents=True)
        (self.author/'docs/lab_buffer/main_protocol.json').write_text(json.dumps({'protocol_id':'sid-main-v1'}))
        (self.author/'docs/lab_buffer/history_gradient_plan.json').write_text(json.dumps(dict(scope_id=bridge.SCOPE,max_diagnostic_optimizer_steps=60,legacy_installed_source_sha256=legacy)))
        source=b'# immutable fixture training source\n';(self.author/'train_retuned_historical.py').write_bytes(source)
        binding=dict(retuned_source_sha256=hashlib.sha256(source).hexdigest())
        settings=dict(max_steps=1000,history_weight=1e-4)
        (self.retuned/'request.json').write_text(json.dumps(dict(binding=binding,settings=settings)))
        (self.retuned/'summary.json').write_text(json.dumps(dict(execution_status='completed',binding=binding,settings=settings,training={'historical':{'artifact_files_sha256':{'last.pth':'f'*64}}})))
        log=self.retuned/'historical/attempt_fixture/metrics.jsonl';log.parent.mkdir(parents=True)
        log.write_text(''.join(json.dumps(row(i))+'\n' for i in range(1,1001)))
        fixtures.git(self.author,'add','.');fixtures.git(self.author,'commit','-qm','scope extension fixture')
        self.sha=fixtures.git(self.author,'rev-parse','HEAD');fixtures.git(self.author,'push','-q','origin','vibe-dev')
        self.d=dict(runner_root=str(self.root),project=str(self.project),retuned_dir=str(self.retuned))
        runner.atomic_json(self.root/'main-deployment.json',self.d)
        self.c=control.Control(self.root)
        self.c.runner.fetch()
        request=control.job(self.d,self.sha,bridge.MIGRATION)
        raw=control.encoded(request)
        state=control.initial_state()
        state.update(mode='queued',active_job_id=bridge.MIGRATION,code_sha=self.sha,repairs_used=1,
            request_sha256=runner.digest(raw),extension=dict(scope_id=bridge.SCOPE,status='migration_queued',purpose='user-authorized diagnostic scope extension'))
        _,parent=self.c.read()
        self.c.write(state,parent,{'jobs/'+bridge.MIGRATION+'.json':raw})

    def test_real_legacy_parent_handoff_keeps_evidence_and_never_runs_future(self):
        before={str(p):p.read_bytes() for p in self.project.rglob('*') if p.is_file()}
        # Actual old service imports its old helpers from bin; the SHA-pinned child
        # runs the new migrator. This tests the live-parent compatibility boundary.
        command=[sys.executable,'-B',str(self.bin/'main_service.py'),'--root',str(self.root)]
        first=subprocess.run(command,env=self.env,capture_output=True,text=True,timeout=60)
        self.assertEqual(first.returncode,0,first.stdout+first.stderr)
        self.assertFalse((self.root/'PAUSED.json').exists(),first.stdout+first.stderr)
        self.assertEqual(json.loads((self.root/'config.json').read_text())['main_scope_revision'],2)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.project.rglob('*') if p.is_file()})
        second=subprocess.run(command,env=self.env,capture_output=True,text=True,timeout=60)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        state,_=self.c.read()
        self.assertEqual(state['extension']['status'],'logs_pending_analysis')
        self.assertEqual(state['repairs_used'],0)
        receipt=state['extension']['result_receipt']
        raw=self.c.runner.read(self.c.runner.ref('results_branch'),receipt['artifact_path'])
        self.assertEqual(runner.digest(raw),receipt['artifact_sha256'])
        artifact=json.loads(raw)
        self.assertEqual(artifact['gradient_logs']['gradient_probe_steps'][0],1)
        self.assertEqual(artifact['gradient_logs']['optimizer_steps_this_task'],0)
        self.assertLess(len(raw),64*1024)
        self.assertEqual(len(list((self.root/'workspaces').iterdir())),1)
        self.assertNotIn('compact',artifact)  # no relabelled Future acceptance

    def test_controlled_task_cannot_run_before_log_analysis(self):
        state,_=self.c.read()
        state['extension']['status']='controlled_queued'
        t=bridge.task(self.d,self.sha);raw=control.encoded(t)
        state.update(active_job_id=t['job_id'],request_sha256=runner.digest(raw))
        with self.assertRaisesRegex(ValueError,'analyzed'):
            bridge.authorize(self.root,t,raw,state)
        state['extension']['logs_analysis_receipt']={'report_commit_sha':'a'*40}
        self.assertTrue(bridge.authorize(self.root,t,raw,state))
        t['timeout_seconds']=9999;raw=control.encoded(t);state['request_sha256']=runner.digest(raw)
        with self.assertRaises(ValueError): bridge.authorize(self.root,t,raw,state)

    def test_preflight_repair_requires_exact_failure_and_zero_spent_budget(self):
        state,_=self.c.read()
        state['extension'].update(status='controlled_queued',logs_analysis_receipt={'report_commit_sha':'a'*40})
        t=bridge.task(self.d,self.sha,job_id=bridge.CONTROLLED_REPAIR);raw=control.encoded(t)
        state.update(active_job_id=t['job_id'],request_sha256=runner.digest(raw))
        with self.assertRaisesRegex(ValueError,'Repair lacks evidence'):bridge.authorize(self.root,t,raw,state)
        authorization=dict(failed_job_id=bridge.CONTROLLED,diagnostic_steps_spent=0,
            mandatory_real_model_checks_passed=True,code_sha=self.sha,
            failed_artifact_sha256='e87e19e408001f2d04bf6932e64d591d57f8927de8031038f779d2b157f2415f')
        state['extension']['controlled_repair_authorization']=authorization
        self.assertTrue(bridge.authorize(self.root,t,raw,state))
        for key,value in [('diagnostic_steps_spent',1),('mandatory_real_model_checks_passed',False),
                          ('code_sha','0'*40),('failed_artifact_sha256','0'*64)]:
            prior=authorization[key];authorization[key]=value
            with self.assertRaisesRegex(ValueError,'Repair lacks evidence'):bridge.authorize(self.root,t,raw,state)
            authorization[key]=prior
        with self.assertRaisesRegex(ValueError,'Unknown controlled job'):
            bridge.task(self.d,self.sha,job_id='arbitrary-repeat')


if __name__=='__main__':unittest.main()
