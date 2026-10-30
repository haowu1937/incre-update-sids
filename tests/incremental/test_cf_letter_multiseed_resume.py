"""Transport recovery, bounded continuation and actual mixed-SHA private-copy pipeline."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
from genrec.cf_letter_protocol import read,save,sha,encoded
from genrec.cf_letter_multiseed_resume_protocol import JOB,MANIFEST,MISSING,plan,submission,expected_counts,copy_indexed,check_reused
from genrec.lab_transport_gate import GitGate,redact
HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class Protocol(unittest.TestCase):
    def test_only_remaining6000_reserved_with_unchanged40000_family(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        m,p=plan(ROOT);self.assertEqual(sum(expected_counts(p).values()),34000);self.assertEqual(sum(expected_counts(p,True).values()),40000)
        self.assertEqual([c for c in expected_counts(p,True) if c not in expected_counts(p)],['gpt/'+n for n in MISSING])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            runner=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            b=submission('a'*40,'b'*64,root,ROOT);task_runtime.validate(runner,b['request'],encoded(b['request']),b['contract'],b['scope'])
            self.assertEqual(b['contract']['optimizer_step_reservation'],6000);self.assertEqual(b['scope']['family_actual_step_cap'],40000)
            self.assertEqual(len(b['request']['artifacts']),8);self.assertEqual(len(b['contract']['preflight']),5)

    def test_copy_is_private_and_existing_source_evidence_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';target=root/'target';(source/'gpt/arms/old').mkdir(parents=True);(source/'rq').mkdir()
            (source/'gpt/arms/old/selected.pth').write_bytes(b'checkpoint');(source/'rq/prepared.npz').write_bytes(b'data')
            os.link(source/'rq/prepared.npz',source/'rq/alias.npz')
            index={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()};copy_indexed(source,target,index)
            self.assertNotEqual((source/'rq/prepared.npz').stat().st_ino,(target/'rq/prepared.npz').stat().st_ino)
            self.assertEqual((target/'rq/prepared.npz').stat().st_ino,(target/'rq/alias.npz').stat().st_ino)
            check_reused(target);(target/'gpt/arms/old/selected.pth').write_bytes(b'changed')
            self.assertEqual((source/'gpt/arms/old/selected.pth').read_bytes(),b'checkpoint')
            with self.assertRaises(ValueError):check_reused(target)
            (source/'outside').symlink_to(root/'outside')
            with self.assertRaises(ValueError):copy_indexed(source,root/'other',{'outside':'x'})

    def test_production_source_check_rejects_hidden_updates_or_changed_index(self):
        import hashlib
        import genrec.cf_letter_multiseed_resume_protocol as mod
        m,p=plan(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'workspaces'/mod.SOURCE_JOB/'outputs/cf-multiseed';source.mkdir(parents=True)
            for name,count in expected_counts(p).items():
                stage,arm=name.split('/');folder=source/stage/'arms'/arm
                if stage=='rq':folder=folder/arm
                save(folder/'optimizer_steps.json',dict(optimizer_steps_completed=count,optimizer_step_intent=None))
            index={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()};save(source/'local_artifact_index.json',index)
            compact=dict(execution_status='failed',code_sha=mod.SOURCE_CODE,optimizer_steps_spent=34000,uncertain_step_records=[],original_inputs_unchanged=True,
                future_names=[],error="CalledProcessError(128, ['git', 'fetch'])",local_artifact_index_sha256=sha(source/'local_artifact_index.json'))
            save(source/'compact.json',compact);save(source/'future.json',{});save(source/'statistics.json',{});save(source/'data_audit.json',{'input_sha256':{}})
            save(source.parent/'task-runtime.json',dict(checks_completed=4,command_started=True,status='failed'))
            result=dict(status='failed',code_sha=mod.SOURCE_CODE,actual_code_sha=mod.SOURCE_CODE,exit_code=1)
            save(root/'state'/(mod.SOURCE_JOB+'.json'),dict(result=result,pending_upload=False))
            proof=dict(result_sha256=hashlib.sha256(encoded(result)).hexdigest(),artifacts_sha256={})
            original=mod.read
            def reading(path):return proof if Path(path).name=='cf_letter_multiseed_failure.json' else original(path)
            with patch.object(mod,'read',side_effect=reading):
                self.assertEqual(mod.source_check(root,m,p)[1],index)
                save(source/'gpt/arms/s2031_plain/optimizer_steps.json',dict(optimizer_steps_completed=1,optimizer_step_intent=None))
                with self.assertRaisesRegex(ValueError,'distribution'):mod.source_check(root,m,p)

    def test_failed_fetch_waits_then_rechecks_fresh_control_and_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'config.json',dict(requests_branch='lab-requests'));gate=GitGate(root,JOB,root)
            error=subprocess.CalledProcessError(128,['git','fetch'],stderr=b'fatal https://user:secret@github.com/repo ghp_secret123 Authorization: Bearer private')
            with patch('genrec.lab_transport_gate.subprocess.run',side_effect=error) as run:
                self.assertIsNone(gate.check());self.assertIsNone(gate.check());self.assertEqual(run.call_count,1)
            text=(root/'control-transport.jsonl').read_text();self.assertNotIn('secret',text);self.assertNotIn('private',text)
            gate.next_retry=0
            response=type('Response',(),{'stdout':encoded(dict(pause_requested=False,active_job_id=JOB,mode='queued'))})()
            with patch('genrec.lab_transport_gate.subprocess.run',return_value=response) as run:
                self.assertIs(gate.check(),False);self.assertEqual(run.call_count,2)
            self.assertEqual(gate.failures,0)
            save(root/'PAUSED.json',{'reason':'manual'})
            with patch('genrec.lab_transport_gate.subprocess.run') as run:self.assertIs(gate.check(),True);run.assert_not_called()

    def test_scheduler_preserves_pending_work_during_transport_wait(self):
        import run_cf_letter_optimizer as mod
        from run_cf_letter_downstream import counters
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();save(root/'deployment.json',dict(runner_root=str(root)))
            p=dict(gpu_memory_limit_mib=24576,wall_seconds=72000,parallel=2,optimizer_step_cap=6000)
            recipes=[dict(name=n,steps=2000) for n in MISSING];checks=iter([None,False]);calls=[];ticks=[0]
            def monotonic():ticks[0]+=31;return ticks[0]
            class Scheduler(mod.Scheduler):
                def start(self,case,active):
                    calls.append(case['name']);folder=out/'arms'/case['name'];save(folder/'optimizer_steps.json',dict(optimizer_steps_completed=2000,optimizer_step_intent=None))
                    save(folder/'arm_result.json',dict(result=case));save(folder/(case['name']+'_curves.json'),case)
                    class Process:
                        returncode=0
                        def poll(self):return 0
                    active[len(calls)]=(Process(),case,open(os.devnull,'w'))
            def check():
                decision=next(checks,False)
                if decision is None:self.assertEqual(calls,[])
                return decision
            with patch.object(mod,'gpu',return_value=(set(),0)),patch.object(mod.time,'monotonic',side_effect=monotonic),patch.object(mod.time,'sleep',return_value=None):
                s=Scheduler(out,root/'deployment.json',p,{'max_gpu_memory_mib':0},0,job_id=JOB,publisher=lambda *_:None,counter=counters,pause_check=check)
                self.assertEqual(s.run(recipes),recipes)
            self.assertEqual(calls,MISSING);self.assertIsNone(s.stop);self.assertEqual(counters(out)[0],6000)

    def test_unknown_control_is_not_silently_treated_as_transport_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'config.json',dict(requests_branch='lab-requests'));gate=GitGate(root,JOB,root)
            response=type('Response',(),{'stdout':b'{}'})()
            with patch('genrec.lab_transport_gate.subprocess.run',return_value=response):
                with self.assertRaises(ValueError):gate.check()

    def test_local_reconciliation_archives_only_matching_pause_without_install_or_launch(self):
        import hashlib
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location('reconcile_resume',ROOT/'scripts/lab_buffer/reconcile_multiseed_resume.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'bin').mkdir();(root/'runner.lock').touch()
            result=dict(job_id=module.SOURCE,status='failed',finished_at='2026-10-30T14:33:09+00:00')
            save(root/'state'/(module.SOURCE+'.json'),dict(result=result,pending_upload=False))
            save(root/'PAUSED.json',dict(reason='Different manual pause',time='2026-10-30T14:34:00+00:00'))
            save(root/'main-service-error.json',dict(error=module.REASON,new_jobs_paused=True))
            proof=dict(result_sha256=hashlib.sha256(encoded(result)).hexdigest(),artifacts_sha256={})
            raw=encoded(dict(code_sha='a'*40));state=dict(lease=None,pause_requested=False,mode='queued',active_job_id=JOB,code_sha='a'*40,
                request_sha256=hashlib.sha256(raw).hexdigest(),receipts={module.SOURCE:{'status':'needs_attention'}})
            calls=[]
            class Runner:
                def __init__(self,r):self.env={};self.config=dict(code_branch='vibe-dev',requests_branch='lab-requests',results_branch='lab-results')
                def fetch(self):calls.append('fetch')
                def ref(self,key):return key
                def git(self,*args,**kwargs):calls.append(args);return SimpleNamespace(returncode=0)
                def read(self,ref,path):
                    if path.endswith('cf_letter_multiseed_failure.json'):return encoded(proof)
                    if path=='control/main.json':return encoded(state)
                    return raw
            runner=SimpleNamespace(Runner=Runner,FINAL={'failed','completed'},atomic_json=save,now=lambda:'now')
            runtime=SimpleNamespace(load_contract=lambda *args:({'optimizer_step_reservation':6000},b'',{'family_actual_step_cap':40000}))
            with patch.dict(sys.modules,{'runner':runner,'task_runtime':runtime}):
                with self.assertRaisesRegex(ValueError,'Different pause'):module.reconcile(root,'a'*40)
                self.assertTrue((root/'PAUSED.json').exists());self.assertEqual(calls,[])
                save(root/'PAUSED.json',dict(reason=module.REASON,time='2026-10-30T14:34:00+00:00'))
                value=module.reconcile(root,'a'*40)
                self.assertFalse(value['task_launched']);self.assertFalse(value['runner_upgraded']);self.assertFalse((root/'PAUSED.json').exists())
                self.assertTrue((Path(value['archive'])/'PAUSED.json').exists())
                self.assertEqual(len(list((root/'resolved-pauses').iterdir())),1)

    def test_verifier_checks_whole_family_but_charges_only_new6000_and_binds_mixed_SHAs(self):
        import verify_cf_letter_multiseed_resume as verifier
        from genrec.cf_letter_multiseed_resume_protocol import SOURCE_CODE
        m,p=plan(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);workspace=root/'workspaces'/JOB;out=workspace/'outputs/cf-multiseed-resume';source=root/'old'
            save(source/'compact.json',{});save(out/'provenance/source_index.json',{});save(out/'provenance/source_compact.json',{});save(out/'provenance/reused_files.json',{})
            save(out/'compact.json',dict(new_optimizer_steps_spent=6000,inherited_optimizer_steps=34000,new_optimizer_step_cap=6000))
            context=root/'context.json';save(context,dict(workspace=str(workspace),request={'code_sha':'b'*40}))
            with patch.object(verifier,'source_check',return_value=(source,{})),patch.object(verifier,'check_reused'),patch.object(verifier,'total_counters',return_value=(40000,expected_counts(p,True),[])),patch('verify_cf_letter_multiseed.verify',return_value=dict(status='verified',actual_optimizer_steps=40000,report_markdown='family')) as check:
                verdict=verifier.verify(context)
            self.assertEqual(verdict['actual_optimizer_steps'],6000)
            args=check.call_args.kwargs;self.assertEqual(args['rq_code_sha'],SOURCE_CODE)
            self.assertEqual(sum(v=='b'*40 for v in args['gpt_code_shas'].values()),3)
            self.assertEqual(sum(v==SOURCE_CODE for v in args['gpt_code_shas'].values()),12)


@unittest.skipUnless(HAVE,'Actual private-copy mixed-code RQ/GPT acceptance mandatory on server')
class RealModels(unittest.TestCase):
    def test_actual_partial_copy_three_new_GPTs_lock_Future_and_full_verifier(self):
        spec=importlib.util.spec_from_file_location('resume_real',ROOT/'tests/incremental/test_cf_letter_multiseed.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        with patch.dict(os.environ,{'LAB_BUFFER_CODE_SHA':'a'*40}):
            fixture.RealModels().test_actual_two_seed_RQ_to_GPT_global_lock_future_and_scientific_acceptance(resume_fixture=True)


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual continuation private-copy/GPT/lock/Future/verifier checks required')
    unittest.main()
