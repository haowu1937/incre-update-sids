"""Budget checks and real RQ/TIGER fixed-anchor direction tests; no Beauty data."""
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
import diagnose_history_gradients as d


class Protocol(unittest.TestCase):
    def test_fixed_cases_budget_and_forbidden_evaluations(self):
        p=json.loads((ROOT/'docs/lab_buffer/history_gradient_plan.json').read_text())
        d.validate_plan(p)
        self.assertEqual(len(d.cases(p)),12)
        self.assertEqual(sum(c['steps'] for c in d.cases(p)),60)
        for key,value in [('history_weight',1e-3),('temperature',.1),('future_evaluation',True),
                          ('holdout_evaluation',True),('steps_per_branch',6),('carry_original_optimizer_momentum',True)]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):d.validate_plan(bad)

    def test_compact_retains_actual_update_sign_and_hard_observations(self):
        record=dict(step=1,H_before=.1,fixed_after={'H':.11},clip_coefficient=1.,
                    capacity={'full_sid_allocation_feasible':True},sid_changes={'full_sid_changed':2},
                    gradient_components={'all':{'update_norm':.01,'history_dot_actual_update':-.001,'rq_dot_actual_update':.1}})
        report=dict(stage=d.STAGE,branches=[dict(case={'objective':'h_only'},step_records=[record])])
        c=d.compact(report)
        self.assertLess(c['branches'][0]['steps'][0]['H_dot_update'],0)
        self.assertEqual(c['branches'][0]['steps'][0]['fixed_after']['H'],.11)
        self.assertNotIn('gradient_components',c['branches'][0]['steps'][0])


HAVE=importlib.util.find_spec('torch') is not None

@unittest.skipUnless(HAVE,'PyTorch absent; real checks are mandatory in the server job')
class Real(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        spec=importlib.util.spec_from_file_location('gradient_real_fixture',ROOT/'tests/incremental/test_historical_update_round12.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.fixture=module.Checks;cls.fixture.setUpClass()
        from gensid import historical_incremental_content as h
        cls.h=h;cls.ex=h.experiment(type('Args',(),{'e1_dir':str(cls.fixture.e1_dir)})())
        cls.cp=torch.load(Path(cls.ex.e1['update_rqvae_dir'])/'last.pth',weights_only=False)
        model=h.build_model(cls.ex.parent['config']);model.load_state_dict(cls.ex.parent['state_dict'])
        cls.cp['distance_scales']=h.distance_scales(model,cls.ex.features[:cls.ex.groups['base_train'][1]],cls.ex.old_batch)
        cls.cp['config'].update(anchor_batch_size=3,anchor_sampler_seed=2027,temperature=.5,
            compatibility_weight=1e-4,max_grad_norm=.1,batch_size=min(3,len(cls.ex.features)//2))

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)
        self.plan=json.loads((ROOT/'docs/lab_buffer/history_gradient_plan.json').read_text())
        self.plan.update(steps_per_branch=2,fixed_anchor_queries=3,full_anchor_eval_steps=[0,1,2])

    def engine(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return d.Engine(self.ex,copy.deepcopy(self.cp),self.plan,self.output,'cpu')

    def test_fixed_hard_forward_reset_branches_and_actual_sgd_direction(self):
        import torch
        e=self.engine()
        self.assertGreater(e.initial_full['H'],0)
        teacher={k:v.clone() for k,v in e.teacher.state_dict().items()}
        first=None
        for obj in ('h_only','rq_only','mixed'):
            with contextlib.redirect_stdout(io.StringIO()):r=e.execute(dict(optimizer='sgd_reset',lr=1e-4,objective=obj,steps=2))
            self.assertEqual(r['status'],'completed')
            self.assertEqual(r['optimizer_steps'],2)
            self.assertTrue(r['initial_weights_equal'] and r['optimizer_state_reset'])
            if first is None:first=r['sampling_sha256']
            self.assertEqual(r['sampling_sha256'],first)
            self.assertEqual(r['initial_full'],e.initial_full)
            for record in r['step_records']:
                self.assertTrue(record['capacity']['full_sid_allocation_feasible'])
                self.assertIn('H',record['fixed_after'])
                if obj=='h_only':
                    self.assertLessEqual(record['gradient_components']['all']['history_dot_actual_update'],1e-10)
                    self.assertEqual(record['gradient_components']['decoder']['update_norm'],0.)
            if obj=='rq_only':self.assertNotEqual(r['step_records'][0]['gradient_components']['decoder']['update_norm'],0.)
        self.assertEqual(e.total_completed_steps,6)
        self.assertTrue(all(torch.equal(v,teacher[k]) for k,v in e.teacher.state_dict().items()))
        with contextlib.redirect_stdout(io.StringIO()):adam=e.execute(dict(optimizer='adamw_reset',lr=1e-4,objective='mixed',steps=2))
        self.assertTrue(adam['optimizer_state_reset'])
        self.assertEqual(adam['sampling_sha256'],first)

    def test_infeasible_candidate_is_preserved_and_next_branch_resets(self):
        import torch
        e=self.engine();original=self.h.suffix_capacity_audit
        def failed(*a,**kw):
            r=original(*a,**kw);r.update(full_sid_allocation_feasible=False,maximum_required_suffix_capacity=999)
            return r
        with patch.object(self.h,'suffix_capacity_audit',side_effect=failed):
            with contextlib.redirect_stdout(io.StringIO()):r=e.execute(dict(optimizer='sgd_reset',lr=1e-5,objective='h_only',steps=2))
        self.assertEqual(r['status'],'stopped_capacity_infeasible')
        self.assertEqual(r['optimizer_steps'],1)
        path=self.output/'sgd_reset_1e-05_h_only'
        valid=torch.load(path/'last_valid.pth',weights_only=False)
        rejected=torch.load(path/'rejected_candidate.pth',weights_only=False)
        self.assertEqual(valid['step'],0);self.assertEqual(rejected['step'],1)
        self.assertFalse(valid['production_checkpoint'])
        self.assertTrue(all(torch.equal(v,e.initial_state[k]) for k,v in valid['state_dict'].items()))
        with contextlib.redirect_stdout(io.StringIO()):after=e.execute(dict(optimizer='sgd_reset',lr=1e-6,objective='rq_only',steps=2))
        self.assertTrue(after['initial_weights_equal'])


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('PyTorch is required; no simulated substitute')
    unittest.main()
