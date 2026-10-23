"""Train-only selection locally; mandatory real RQ/TIGER resume checks on server."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import joint_convergence as j
from train_joint_convergence import Arm,sampler_identity


def summary(lr=1e-4,weight=1e-3,H=.04,rq=.004,new=.006,seed=2026):
    rows=[dict(step=s,epoch=i+1,epoch_complete=True,rq={'total_loss':rq},anchor_train={'H':H},
        groups={'update_train_first_seen':{'reconstruction_loss':new}}) for i,s in enumerate(range(2600,3000,46))]
    return dict(status='completed',case=dict(lr=lr,history_weight=weight,seed=seed,arm_id=j.identifier(lr,weight,seed)),
        initial=dict(rq={'total_loss':.006},groups={'update_train_first_seen':{'reconstruction_loss':.01}}),curve=rows)


class Protocol(unittest.TestCase):
    def setUp(self):self.plan=j.read_plan(ROOT)

    def test_budget_matrix_and_schedule_are_fixed(self):
        self.assertEqual(4*100+12*1500+4*1500+2*3000,self.plan['max_optimizer_steps'])
        self.assertEqual(len({c['arm_id'] for c in j.matrix(self.plan)}),12)
        for key,value in [('future_evaluation',True),('max_optimizer_steps',40000),('max_parallel',4)]:
            p=copy.deepcopy(self.plan);p[key]=value
            with self.assertRaises(ValueError):j.validate_plan(p)
        self.assertAlmostEqual(j.lr_at(1,1.,self.plan),.02)
        self.assertEqual(j.lr_at(50,1.,self.plan),1.)
        self.assertEqual(j.lr_at(2500,1.,self.plan),.1)
        self.assertEqual(j.lr_at(3000,1.,self.plan),.1)
        self.assertGreater(j.lr_at(1500,1.,self.plan),j.lr_at(1501,1.,self.plan))

    def test_stability_quality_and_zero_h_tie(self):
        control=summary(weight=0,H=.1)
        candidate=summary()
        self.assertTrue(j.stability(candidate)['passed'])
        self.assertTrue(j.quality(candidate,control,True)['passed'])
        self.assertIsNotNone(j.select_final([control,candidate])['selected'])
        self.assertFalse(j.quality(summary(H=0),summary(weight=0,H=0),True)['passed'])
        bad=copy.deepcopy(candidate);bad['curve'][-1]['anchor_train']['H']=.2
        self.assertFalse(j.stability(bad)['passed'])
        # MSE-only progress cannot pass the full RQ requirement.
        self.assertFalse(j.quality(summary(rq=.007),control,True)['passed'])

    def test_selection_is_train_only_paired_and_bounded(self):
        rows=[]
        for lr in self.plan['learning_rates']:
            rows.extend([summary(lr=lr,weight=0,H=.1),summary(lr=lr,H=.06),summary(lr=lr,weight=.01,H=.07)])
        choice=j.select_screen(rows)
        self.assertLessEqual(len(choice['selected']),2);self.assertLessEqual(len(choice['arm_ids']),4)
        for row in choice['selected']:self.assertIn(row['control'],choice['arm_ids'])
        self.assertEqual(j.select_screen([summary()])['arm_ids'],[])
        bad=summary();bad['curve'][-1]['step']+=1
        with self.assertRaises(ValueError):j.quality(bad,summary(weight=0))

    def test_curve_export_and_plot_are_actual_epoch_values(self):
        from run_joint_convergence import curves,plot_curves
        with tempfile.TemporaryDirectory() as tmp:
            s=summary()
            for r in s['curve']:
                r['rq'].update(reconstruction_loss=.002,quantization_loss=.008)
                r['capacity']={'maximum_required_suffix_capacity':20}
            for name in ('screen','extended','confirmation'):curves(Path(tmp)/(name+'-curves.json'),[s])
            result=plot_curves(tmp)
            data=json.loads((Path(tmp)/'screen-curves.json').read_text())
            self.assertEqual(next(iter(data['arms'].values()))[-1][4],.04)
            self.assertIn(result['status'],('rendered','deferred'))
            if result['status']=='rendered':
                self.assertGreater((Path(tmp)/'screen-curves.svg').stat().st_size,1000)
            else:
                self.assertEqual(json.loads((Path(tmp)/'plot-status.json').read_text()),result)

    def test_missing_optional_plotter_preserves_all_numeric_curves(self):
        from run_joint_convergence import curves,plot_curves
        with tempfile.TemporaryDirectory() as tmp:
            for phase in ('screen','extended','confirmation'):curves(Path(tmp)/(phase+'-curves.json'),[])
            before={p.name:p.read_bytes() for p in Path(tmp).glob('*-curves.json')}
            with patch.dict(sys.modules,{'matplotlib':None}):result=plot_curves(tmp)
            self.assertEqual(result['status'],'deferred')
            self.assertEqual(before,{p.name:p.read_bytes() for p in Path(tmp).glob('*-curves.json')})


HAVE=importlib.util.find_spec('torch') is not None

@unittest.skipUnless(HAVE,'PyTorch absent; real checks are mandatory before server dispatch')
class Real(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        spec=importlib.util.spec_from_file_location('joint_real_fixture',ROOT/'tests/incremental/test_historical_update_round12.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        cls.fixture=mod.Checks;cls.fixture.setUpClass()
        from gensid import historical_incremental_content as h
        cls.h=h;cls.ex=h.experiment(type('Args',(),{'e1_dir':str(cls.fixture.e1_dir)})())

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)
        self.plan=j.read_plan(ROOT)
        self.plan.update(batch_size=3,anchor_batch_size=3,fixed_anchor_queries=3,final_steps=8,
            warmup_steps=2,decay_end_step=6,gradient_every=2,save_every=2)
        self.case=dict(lr=1e-5,history_weight=1e-4,seed=2026,arm_id='tiny')

    def arm(self,name,resume=False,weight=None):
        case=copy.deepcopy(self.case)
        if weight is not None:case['history_weight']=weight
        with contextlib.redirect_stdout(io.StringIO()):
            return Arm(self.ex,case,self.path/name,self.plan,{'fixture':True},device='cpu',resume=resume)

    def fit(self,arm,target):
        with contextlib.redirect_stdout(io.StringIO()):result=arm.fit(target)
        self.assertEqual(result['status'],'completed',result.get('failure'))
        return result

    def test_exact_optimizer_rng_content_anchor_and_scheduler_resume(self):
        import torch
        full=self.fit(self.arm('full'),8)
        self.fit(self.arm('split'),3)
        resumed=self.fit(self.arm('split',True),8)
        a=torch.load(self.path/'full/last.pth',weights_only=False)
        b=torch.load(self.path/'split/last.pth',weights_only=False)
        self.assertTrue(all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items()))
        for field in ('content_sampler_sha256','anchor_sampler_sha256','processed_examples','anchor_queries_processed'):
            self.assertEqual(full[field],resumed[field])
        self.assertEqual(full['curve'][-1],resumed['curve'][-1])
        self.assertEqual(a['scheduler'],b['scheduler'])
        self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
        for k in a['optimizer']['state']:
            for field,value in a['optimizer']['state'][k].items():
                other=b['optimizer']['state'][k][field]
                self.assertTrue(torch.equal(value,other) if torch.is_tensor(value) else value==other)
        self.assertTrue((self.path/'split/checkpoint-3.pth').is_file())

    def test_plain_measures_real_h_and_monitor_preserves_rng_sampler_mode(self):
        import torch
        arm=self.arm('plain',weight=0)
        arm.model.eval();rng=torch.get_rng_state().clone();identity=sampler_identity(arm.penalty.snapshot())
        with contextlib.redirect_stdout(io.StringIO()):arm.measure(False)
        self.assertFalse(arm.model.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(identity,sampler_identity(arm.penalty.snapshot()))
        report=self.fit(arm,3)
        rows=[json.loads(x) for x in (self.path/'plain/metrics.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['anchor_queries']>0 and r['H_before']==r['history_penalty'] for r in rows))
        self.assertTrue(all(r['weighted_H']==0 for r in rows))
        self.assertEqual(report['anchor_queries_processed'],9)
        self.h.check_teacher(arm.teacher,self.ex)

    def test_capacity_failure_preserves_rejected_and_exact_last_valid_state(self):
        import torch
        arm=self.arm('capacity');before=copy.deepcopy(arm.model.state_dict())
        original=self.h.suffix_capacity_audit
        def failed(*args,**kwargs):
            result=original(*args,**kwargs);result['full_sid_allocation_feasible']=False;return result
        with patch.object(self.h,'suffix_capacity_audit',side_effect=failed):
            with contextlib.redirect_stdout(io.StringIO()):report=arm.fit(3)
        self.assertEqual(report['status'],'stopped_capacity_infeasible')
        self.assertEqual(report['valid_step'],0)
        self.assertEqual(report['optimizer_step_count']['optimizer_steps_completed'],1)
        self.assertTrue((self.path/'capacity/rejected_candidate.pth').is_file())
        self.assertTrue(all(torch.equal(v,arm.model.state_dict()[k]) for k,v in before.items()))
        with self.assertRaises(ValueError):self.arm('capacity',True)


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('PyTorch required; no synthetic substitute for real model checks')
    unittest.main()
