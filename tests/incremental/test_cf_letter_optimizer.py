"""Scope/selection/concurrency regressions and real CPU optimizer-arm acceptance."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
import numpy as np
from genrec.cf_letter_protocol import save,read,encoded,sha
from genrec.cf_letter_optimizer_protocol import plan,cases,pair_cases,selection,screen_gate,submission,can_launch,counters
from genrec.cf_letter_suffix_audit import SuffixAudit
HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class Protocol(unittest.TestCase):
    def test_exact_matrix_scope_and_generic_runtime(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        p=plan(ROOT);matrix=cases(p)
        self.assertEqual(len(matrix),5);self.assertEqual(sum(v['steps'] for v in matrix),5000)
        self.assertNotIn((1e-4,.9),[(v['lr'],v['beta1']) for v in matrix])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            runner=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            b=submission('a'*40,'b'*64,root,ROOT);task_runtime.validate(runner,b['request'],encoded(b['request']),b['contract'],b['scope'])
            self.assertEqual(b['contract']['optimizer_step_reservation'],7000);self.assertEqual(b['scope']['max_parallel'],2)
            for field,value in [('optimizer_step_cap',7001),('parallel',3),('fixed_history_weight',.001),('allowed_query_splits',['base_train','update_train','update_val'])]:
                changed=copy.deepcopy(p);changed[field]=value;save(root/'docs/experiments/cf_letter_optimizer.json',changed)
                with self.assertRaises(ValueError):plan(root)

    def fixture_result(self,case,H=.09,rq=.0101,mse=.00201,stable=True):
        return dict(case=case,optimizer_steps=1000,capacity_feasible=True,stability=dict(sufficient=True,passed=stable,H={'mean':H},rq={'mean':rq},reconstruction_mse={'mean':mse}))

    def test_gate_uses_stable_tail_and_bounds_both_reconstruction_terms(self):
        p=plan(ROOT);initial=dict(H=.1,rq=.01,mse=.002);case=cases(p)[0]
        self.assertTrue(screen_gate(self.fixture_result(case),initial,p)['eligible'])
        for kw in [dict(H=.099),dict(rq=.0103),dict(mse=.0021),dict(stable=False)]:self.assertFalse(screen_gate(self.fixture_result(case,**kw),initial,p)['eligible'])
        rows=[self.fixture_result(c,H=.09) for c in cases(p)]
        chosen=selection(rows,initial,p);self.assertEqual(chosen['selected_name'],rows[0]['case']['name'])
        pairs=pair_cases(rows[0],p);self.assertEqual([v['objective'] for v in pairs],['rq_only','mixed'])
        self.assertTrue(all(v['lr']==rows[0]['case']['lr'] and v['beta1']==rows[0]['case']['beta1'] for v in pairs))
        self.assertFalse(selection([self.fixture_result(case,H=.1)],initial,p)['pair_authorized']);self.assertEqual(pair_cases(None,p),[])

    def test_vectorized_suffix_audit_exactly_matches_reference_including_overflow(self):
        from genrec.cf_letter_refresh_protocol import versioned_mapping
        rng=np.random.default_rng(2026)
        for capacity in (2,4,16):
            n=min(50,capacity**4//2);raw=np.column_stack(np.unravel_index(rng.choice(capacity**4,n,replace=False),(capacity,)*4))
            previous=raw+np.arange(4)*capacity;audit=SuffixAudit(previous,capacity)
            for _ in range(40):
                codes=rng.integers(capacity,size=(n,3));a,b=versioned_mapping(previous,codes,capacity);x,y=audit(codes)
                self.assertEqual(b,y)
                if a is None:self.assertIsNone(x)
                else:self.assertTrue(np.array_equal(a,x))
            a,b=versioned_mapping(previous,raw[:,:3],capacity);x,y=audit(raw[:,:3]);self.assertTrue(np.array_equal(a,x));self.assertEqual(b,y)

    def test_admission_waits_for_foreign_gpu_and_never_exceeds_two(self):
        p=plan(ROOT)
        self.assertTrue(can_launch({},set(),0,p));self.assertTrue(can_launch({12:None},{12},3000,p))
        self.assertFalse(can_launch({12:None,13:None},{12,13},6000,p));self.assertFalse(can_launch({}, {99},1000,p))
        self.assertFalse(can_launch({12:None},{12,99},5000,p));self.assertFalse(can_launch({},set(),30000,p));self.assertFalse(can_launch({},set(),0,p,True))

    def test_first_child_can_start_before_selection_file_has_results(self):
        from run_cf_letter_optimizer import authorized_case
        p=plan(ROOT);case=cases(p)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'selection.json',{});save(root/'launch_journal.json',[case]);save(root/'arms'/case['name']/'case.json',case)
            self.assertEqual(authorized_case(root,p,case['name']),case)
            with self.assertRaises(ValueError):authorized_case(root,p,'pair_mixed')
            save(root/'selection.json',{'pair_authorized':True,'selected_name':case['name']});save(root/'screen_results.json',[{'case':case}])
            paired=pair_cases({'case':case},p)[1];save(root/'arms'/paired['name']/'case.json',paired);save(root/'launch_journal.json',[case,paired])
            self.assertEqual(authorized_case(root,p,paired['name']),paired)

    def test_scheduler_reserves_once_and_waits_for_other_arm_after_failure(self):
        import run_cf_letter_optimizer as driver
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);output=root/'out';(output/'inputs').mkdir(parents=True);(output/'inputs/x').write_bytes(b'private copy')
                (output/'prepared.npz').write_bytes(b'fixture');save(output/'binding.json',{});save(root/'deployment.json',dict(runner_root=str(root)))
                p=plan(ROOT);matrix=cases(p)[:3];live=set();peak=[0];launched=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        self.name=command[-1];self.pid=100+len(launched);launched.append(self.name);live.add(self.pid);peak[0]=max(peak[0],len(live));self.polls=0;self.returncode=None
                    def poll(self):
                        self.polls+=1
                        if self.polls<3:return None
                        if self.returncode is None:
                            self.returncode=1 if fail and self.pid==100 else 0;live.remove(self.pid)
                            if not self.returncode:
                                save(output/'arms'/self.name/'arm_result.json',dict(result={'case':next(v for v in matrix if v['name']==self.name)}));save(output/'arms'/self.name/(self.name+'_curves.json'),{})
                        return self.returncode
                with patch.object(driver,'gpu',side_effect=lambda:(set(live),1000)),patch.object(driver,'paused',return_value=False),patch.object(driver.subprocess,'Popen',Process),patch.object(driver.time,'sleep',return_value=None):
                    scheduler=driver.Scheduler(output,root/'deployment.json',p,{'max_gpu_memory_mib':0},time.monotonic())
                    if fail:
                        with self.assertRaises(RuntimeError):scheduler.run(matrix)
                    else:self.assertEqual(len(scheduler.run(matrix)),3)
                self.assertLessEqual(peak[0],2);self.assertFalse(live);self.assertEqual(len(launched),2 if fail else 3)
                self.assertEqual(len(read(output/'launch_journal.json')),len(launched));self.assertEqual((output/'inputs/x').read_bytes(),b'private copy')


@unittest.skipUnless(HAVE,'Torch/transformers unavailable; actual model checks required on server')
class RealModels(unittest.TestCase):
    def setUp(self):
        import torch
        from genrec.cf_letter_base_training import seed_all
        torch.set_num_threads(1);seed_all(2026)

    def fixture(self,output):
        spec=importlib.util.spec_from_file_location('accepted_long_fixture',ROOT/'tests/incremental/test_cf_letter_history_long.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        p,data,binding=module.RealModels().fixture(output)
        settings=plan(ROOT)
        for key in ('fixed_history_weight','beta2','eps'):p[key]=settings[key]
        p['optimizer_step_cap']=70
        return p,data,binding

    def test_actual_optimizer_matrix_and_cpu_scientific_arm_acceptance(self):
        import torch
        from genrec.cf_letter_optimizer_training import OptimizerEngine
        from genrec.cf_letter_optimizer_protocol import save_anchors
        from verify_cf_letter_optimizer import verify_arm
        settings=plan(ROOT);variants=cases(settings)
        # Include both conditional objectives to verify the exact optimizer recipe.
        variants+=pair_cases({'case':variants[2]},settings)
        with tempfile.TemporaryDirectory() as tmp:
            sources={};traces=[]
            for case in variants:
                self.setUp()
                output=Path(tmp)/case['name'];output.mkdir();p,data,binding=self.fixture(output);case=dict(case,steps=10)
                save_anchors(output/'prepared.npz',data)
                engine=OptimizerEngine(output,data,binding,p,'cpu');value=engine.run_case(case)
                value.update(code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),case=case);save(output/'arm_result.json',value)
                verified=verify_arm(output,data,binding,p,case,os.environ.get('LAB_BUFFER_CODE_SHA'))
                self.assertEqual(verified['optimizer_steps'],10)
                saved=torch.load(output/case['name']/'initial.pth',map_location='cpu',weights_only=False)
                self.assertEqual(tuple(saved['optimizer']['param_groups'][0]['betas']),(case['beta1'],.999));self.assertFalse(saved['optimizer']['state'])
                rows=read(output/case['name']/'trajectory.json');traces.append([(v['content_indices_sha256'],v['anchor_indices_sha256']) for v in rows])
                self.assertEqual([v['step'] for v in verified['monitors']],[4,8,10]);self.assertLess((output/(case['name']+'_curves.json')).stat().st_size,65536)
            self.assertTrue(all(t==traces[0] for t in traces))

    def test_actual_H_gradient_routing_stays_identical(self):
        import torch
        from genrec.cf_letter_optimizer_training import OptimizerEngine
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);p,data,binding=self.fixture(output);engine=OptimizerEngine(output,data,binding,p,'cpu')
            loss,_,_,_=engine.historical(engine.fixed,engine.initial_codes,engine.initial_mapping)
            g=torch.autograd.grad(loss,list(engine.model.books));self.assertTrue(all(torch.isfinite(v).all() and v.norm()>0 for v in g))
            self.assertTrue(all(not v.requires_grad and v.grad is None for v in engine.teacher.parameters()));self.assertFalse(engine.model.features.requires_grad)


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual Torch/GPT/codebook optimizer checks required before Beauty updates')
    unittest.main()
