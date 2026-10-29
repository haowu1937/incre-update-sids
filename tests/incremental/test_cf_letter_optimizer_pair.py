"""Exact scope, paired efficacy and new-entry scheduling; actual tiny full acceptance."""
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
from genrec.cf_letter_protocol import read,save,sha,encoded
from genrec.cf_letter_optimizer_pair_protocol import JOB,SCOPE,MANIFEST,plan,cases,assessments,publish_curves,submission
HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class Protocol(unittest.TestCase):
    def test_exact_four_arms_new_reservation_and_generic_contract(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        p=plan(ROOT);matrix=cases(p)
        self.assertEqual([(c['lr'],c['beta1'],c['objective']) for c in matrix],[(1e-5,.9,'rq_only'),(1e-5,.9,'mixed'),(1e-5,0.,'rq_only'),(1e-5,0.,'mixed')])
        self.assertEqual(sum(v['steps'] for v in matrix),4000)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            runner=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            bundle=submission('a'*40,'b'*64,root,ROOT);task_runtime.validate(runner,bundle['request'],encoded(bundle['request']),bundle['contract'],bundle['scope'])
            self.assertEqual(bundle['contract']['optimizer_step_reservation'],4000);self.assertEqual(len(bundle['request']['artifacts']),6)
            for k,v in [('optimizer_step_cap',4001),('fixed_history_weight',.001),('previous_screen_eligible',True),('canonical_assignment_backend','cpu'),('parallel',3)]:
                changed=copy.deepcopy(p);changed[k]=v;save(root/MANIFEST,changed)
                with self.assertRaises(ValueError):plan(root)

    def test_pair_gate_preserves_stability_and_no_h_only_dependency(self):
        p=plan(ROOT);rows=[]
        for c in cases(p):
            mixed=c['objective']=='mixed'
            rows.append(dict(case=c,optimizer_steps=1000,capacity_feasible=True,stability=dict(sufficient=True,passed=True,
                H={'mean':.09 if mixed else .1},rq={'mean':.0101 if mixed else .01},reconstruction_mse={'mean':.00201 if mixed else .002})))
        result=assessments(rows,p);self.assertTrue(all(v['passed'] for v in result.values()))
        rows[1]['stability']['passed']=False;self.assertFalse(assessments(rows,p)['lr1e5_b09']['passed'])
        self.assertFalse(assessments(rows[:1],p)['lr1e5_b09']['eligible'])
        rows[1]['stability']['passed']=True;rows[1]['stability']['rq']['mean']=.011
        self.assertFalse(assessments(rows,p)['lr1e5_b09']['passed'])

    def test_scheduler_uses_new_job_child_and_publisher_with_two_children(self):
        import run_cf_letter_optimizer as driver
        import run_cf_letter_optimizer_pair as paired
        p=plan(ROOT);matrix=cases(p);live=set();launched=[];peaks=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'output';(out/'inputs').mkdir(parents=True)
            (out/'inputs/x').write_bytes(b'new private file');(out/'prepared.npz').write_bytes(b'fixture');save(out/'binding.json',{});save(root/'deployment.json',dict(runner_root=str(root)))
            class Process:
                def __init__(self,argv,**kwargs):
                    self.pid=200+len(launched);self.name=argv[-1];self.returncode=None;self.polls=0
                    self_test.assertEqual(Path(argv[3]).name,'run_cf_letter_optimizer_pair.py')
                    self_test.assertEqual(paired.authorized_case(out,p,self.name),next(v for v in matrix if v['name']==self.name))
                    launched.append(self.name);live.add(self.pid);peaks.append(len(live))
                def poll(self):
                    self.polls+=1
                    if self.polls<3:return None
                    if self.returncode is None:
                        live.remove(self.pid);self.returncode=0
                        save(out/'arms'/self.name/'arm_result.json',dict(result=dict(case=next(v for v in matrix if v['name']==self.name))))
                        save(out/'arms'/self.name/(self.name+'_curves.json'),{})
                    return 0
            self_test=self
            with patch.object(driver,'gpu',side_effect=lambda:(set(live),1000)),patch.object(driver,'paused',return_value=False) as pause,patch.object(driver.subprocess,'Popen',Process),patch.object(driver.time,'sleep',return_value=None):
                s=driver.Scheduler(out,root/'deployment.json',p,{'max_gpu_memory_mib':0},time.monotonic(),job_id=JOB,child_script='run_cf_letter_optimizer_pair.py',publisher=publish_curves)
                self.assertEqual(len(s.run(matrix)),4);pause.assert_called_with(root,JOB)
            self.assertEqual(max(peaks),2);self.assertFalse(live);self.assertEqual(launched,[v['name'] for v in matrix])
            self.assertEqual(set(read(out/'curves_lr1e5_b09.json')),{v['name'] for v in matrix[:2]})
            self.assertFalse((out/'curves_screen_1.json').exists())


@unittest.skipUnless(HAVE,'Torch/transformers unavailable; actual tiny acceptance mandatory on server')
class RealModels(unittest.TestCase):
    def test_real_four_arm_training_and_full_read_only_acceptance(self):
        import torch
        from genrec.cf_letter_base_training import seed_all
        from genrec.cf_letter_optimizer_training import OptimizerEngine
        from genrec.cf_letter_optimizer_protocol import save_anchors,counters
        import verify_cf_letter_optimizer_pair as verifier
        spec=importlib.util.spec_from_file_location('long_pair_fixture',ROOT/'tests/incremental/test_cf_letter_history_long.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        torch.set_num_threads(1);seed_all(2026)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'outputs/cf-optimizer-pair';output.mkdir(parents=True)
            p,data,binding=fixture.RealModels().fixture(output);settings=plan(ROOT)
            p.update({k:settings[k] for k in ('pair_settings','fixed_history_weight','beta2','eps')});p.update(optimizer_step_cap=40,canonical_assignment_backend='cpu')
            # Bind all actual tiny dependency bytes, retaining the production check paths.
            save(output/'inputs/base/compact.json',{});save(output/'inputs/base/data_audit.json',{});save(output/'inputs/refresh/compact.json',{})
            p['predecessor']['files_sha256']={str(f.relative_to(output/'inputs/base')):sha(f) for f in (output/'inputs/base').rglob('*') if f.is_file() and f.name not in ('compact.json','data_audit.json')}
            p['predecessor']['compact_sha256']=sha(output/'inputs/base/compact.json');p['predecessor']['audit_sha256']=sha(output/'inputs/base/data_audit.json')
            p['refresh_predecessor']['files_sha256']={str(f.relative_to(output/'inputs/refresh')):sha(f) for f in (output/'inputs/refresh').rglob('*') if f.is_file() and f.name!='compact.json'}
            p['refresh_predecessor']['compact_sha256']=sha(output/'inputs/refresh/compact.json');p['audit_predecessor']={'files_sha256':{}}
            save_anchors(output/'prepared.npz',data);save(output/'binding.json',binding)
            results=[]
            for case in cases(p):
                seed_all(2026);directory=output/'arms'/case['name'];directory.mkdir(parents=True)
                shutil.copytree(output/'inputs',directory/'inputs');shutil.copyfile(output/'prepared.npz',directory/'prepared.npz')
                engine=OptimizerEngine(directory,data,binding,p,'cpu');value=engine.run_case(case)
                value.update(code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),case=case);save(directory/'arm_result.json',value)
                results.append(value['result']);shutil.copyfile(directory/(case['name']+'_curves.json'),output/(case['name']+'_curves.json'))
            publish_curves(output,p);assessment=assessments(results,p);save(output/'paired_assessment.json',assessment);save(output/'launch_journal.json',cases(p))
            save(output/'data_audit.json',dict(allowed_query_splits=p['allowed_query_splits'],training_items=52,anchor_train_queries=16,input_sha256={},prepared_sha256=sha(output/'prepared.npz')))
            index={str(f.relative_to(output)):sha(f) for f in output.rglob('*') if f.is_file()};save(output/'local_artifact_index.json',index)
            total,steps,intents=counters(output)
            save(output/'compact.json',dict(stage=SCOPE,job_id=JOB,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),execution_status='completed',optimizer_step_cap=40,
                optimizer_steps_spent=total,stage_step_counts=steps,uncertain_step_records=intents,original_inputs_unchanged=True,fixed_history_weight=p['fixed_history_weight'],
                previous_screen_eligible=False,previous_scientific_acceptance='blocked',canonical_assignment_backend='cpu',gpt_training_performed=False,sasrec_training_performed=False,
                validation_evaluated=False,holdout_evaluated=False,future_evaluated=False,returned_artifact_sha256={},local_artifact_index_sha256=sha(output/'local_artifact_index.json'),
                pair_results=results,paired_assessment=assessment,resource_stop_reason=None,research_outcome='explicit_two_setting_pairs_completed'))
            context=Path(tmp)/'context.json';save(context,dict(workspace=tmp,request={'code_sha':os.environ.get('LAB_BUFFER_CODE_SHA')},contract={'input_sha256':{'/fixture/main-deployment.json':'x'}},contract_sha256='fixture'))
            before={str(f):sha(f) for f in output.rglob('*') if f.is_file()}
            with patch.object(verifier,'plan',return_value=p),patch.object(verifier,'load_data',return_value=(data,binding)):
                verdict=verifier.verify(context,assignment_device='cpu')
            self.assertEqual(verdict['status'],'verified');self.assertEqual(verdict['actual_optimizer_steps'],40)
            self.assertTrue(all(sha(f)==value for f,value in before.items()))


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual four-arm CPU training/verifier preflight required before Beauty')
    unittest.main()
