"""Synthetic CPU verification of transactional hard-H candidate selection."""
import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import history_aware_hard_gate as g

torch.set_num_threads(1)

def exact(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('frozen_fixture',ROOT/'tests/incremental/test_frozen_encoder_update.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        cls.fixture=m.Checks;cls.fixture.setUpClass()

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def setUp(self):
        self.helper=self.fixture();self.helper.setUp();self.root=self.helper.root
        self.helper.train('control',mode='plain')
        self.control=self.root/'control'

    def tearDown(self):self.helper.tearDown()

    def args(self,name='gated',steps=12,rho=.1,resume=None):
        values=['train','--e1-dir',str(self.fixture.previous.e1_dir),'--plain-dir',str(self.control),
            '--output',str(self.root/name),'--device','cpu','--max-steps',str(steps),
            '--history-gradient-ratio',str(rho),'--anchor-batch-size','3','--log-every','3','--save-every','3']
        if resume:values+=['--resume-from',str(resume)]
        return g.parser().parse_args(values)

    def train(self,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return g.train(self.args(*args,**kwargs))

    def cp(self,name):return torch.load(self.root/name/'last.pth',weights_only=False)

    def test_gate_strict_H_quality_capacity_and_content_tie(self):
        content=dict(feasible=True,history=dict(positive_damage=.1),rq_mse=1.)
        joint=dict(feasible=True,history=dict(positive_damage=.09),rq_mse=1.005)
        self.assertEqual(g.choose(content,joint,1e-6,1.01)[0],'historical')
        # No previous-step H is passed: H=0 at Base does not block updating.
        for delta,reason in [(dict(rq_mse=1.02),'historical_reconstruction_cap_exceeded'),
                             (dict(history=dict(positive_damage=.1)),'historical_hard_H_not_lower'),
                             (dict(feasible=False),'historical_candidate_infeasible')]:
            self.assertEqual(g.choose(content,dict(joint,**delta),1e-6,1.01),('content',reason))
        self.assertIsNone(g.choose(dict(content,feasible=False),joint,1e-6,1.01)[0])

    def test_both_proposals_use_same_Adam_moments_and_commit_selected_transaction(self):
        cp=self.cp('control');model=g.f.build_model(cp['config']);model.load_state_dict(cp['state_dict'])
        model.encoder.requires_grad_(False)
        opt=torch.optim.AdamW(model.parameters(),lr=cp['config']['lr'],weight_decay=cp['config']['weight_decay'])
        opt.load_state_dict(cp['optimizer']);base=g.transaction(model,opt)
        gradients={n:torch.ones_like(p)*.001 if p.requires_grad else None for n,p in model.named_parameters()}
        joint={n:v.clone() if v is not None else None for n,v in gradients.items()}
        for n in joint:
            if n.startswith('rq.') and joint[n] is not None:joint[n]*=-2
        a=g.proposal(model,opt,base,gradients)
        b=g.proposal(model,opt,base,joint)
        repeat=g.proposal(model,opt,base,gradients)
        self.assertTrue(exact(a,repeat))
        self.assertFalse(exact(a['optimizer'],b['optimizer']))
        # Independently replay exactly one AdamW step with the joint gradient.
        g.restore_transaction(model,opt,base)
        for n,p in model.named_parameters():p.grad=joint[n].clone() if joint[n] is not None else None
        opt.step();self.assertTrue(exact(g.transaction(model,opt),b))
        g.restore_transaction(model,opt,a)
        self.assertTrue(exact(g.transaction(model,opt),a))
        g.restore_transaction(model,opt,b)
        self.assertTrue(exact(g.transaction(model,opt),b))
        for k in a['optimizer']['state']:
            self.assertTrue(torch.equal(a['optimizer']['state'][k]['step'],base['optimizer']['state'][k]['step']+1))
        self.assertTrue(all(torch.equal(v,base['state_dict'][k]) for k,v in a['state_dict'].items() if k.startswith('encoder.')))

    def test_zero_rho_exactly_replays_completed_frozen_Plain(self):
        r=self.train('zero',rho=0)
        a,b=self.cp('zero'),self.cp('control')
        for key in ('state_dict','optimizer','sampler_state'):
            self.assertTrue(exact(a[key],b[key]),key)
        self.assertEqual(r['final_full_anchor_train'],g.f.read_json(self.control/'summary.json')['final_full_anchor_train'])
        self.assertEqual(r['work_counters']['committed_historical_steps'],0)
        self.assertEqual(r['work_counters']['optimizer_proposals'],12)
        self.assertEqual(r['work_counters']['history_gradient_queries'],0)
        self.assertTrue(r['encoder_unchanged_from_base'])

    def test_real_model_local_guard_and_exact_resume(self):
        before={str(p):g.f.sha256(p) for p in self.control.rglob('*') if p.is_file()}
        full=self.train('full');self.train('smoke',steps=9)
        resumed=self.train('resumed',resume=self.root/'smoke/last.pth')
        a,b=self.cp('full'),self.cp('resumed')
        for key in ('state_dict','optimizer','sampler_state','rng_state','work_counters'):
            self.assertTrue(exact(a[key],b[key]),key)
        self.assertEqual(full['final_full_anchor_train'],resumed['final_full_anchor_train'])
        self.assertEqual(full['work_counters']['history_gradient_queries'],144)
        self.assertEqual(full['work_counters']['local_guard_checks'],12)
        self.assertTrue(full['per_step_local_guard_passed'])
        self.assertFalse(full['independent_plain_endpoint_dominance_guaranteed'])
        rows=g.f.read_jsonl(self.root/'full/metrics.jsonl')
        for row in rows:
            self.assertLessEqual(row['local_H_change_vs_content'],0.)
            self.assertEqual(row['selected_history']['queries'],12)
            for s in row['codebook_gradients']:self.assertLessEqual(s['effective_gradient_ratio'],.100001)
        self.assertTrue(all(g.f.sha256(Path(p))==digest for p,digest in before.items()))

    def test_invalid_content_restores_previous_complete_checkpoint_and_sampler(self):
        self.train('one',steps=1)
        actual=g.candidate_metrics;calls=[0]
        def invalid(model,ex,history):
            result=actual(model,ex,history);calls[0]+=1
            if calls[0]==2:return dict(feasible=False,reason='full_sid_capacity_exhausted'),None
            return result
        with mock.patch.object(g,'candidate_metrics',side_effect=invalid):r=self.train('stop')
        self.assertEqual(r['execution_status'],'stopped');self.assertEqual(r['update_optimizer_steps'],1)
        self.assertEqual(r['work_counters']['rolled_back_rounds'],1)
        a,b=self.cp('one'),self.cp('stop')
        for key in ('state_dict','optimizer','sampler_state','rng_state'):self.assertTrue(exact(a[key],b[key]),key)
        self.assertEqual(r['full_sid_export_status'],'completed')

    def test_comparison_training_only_records_binding_and_budget(self):
        self.train('full')
        self.helper.train('ungated')
        args=g.parser().parse_args(['compare','--rqvae-dir',str(self.root/'full'),
            '--ungated-dir',str(self.root/'ungated'),'--output',str(self.root/'compare')])
        with contextlib.redirect_stdout(io.StringIO()):r=g.compare(args)
        self.assertEqual(r['queries'],12);self.assertTrue(r['same_content_optimizer_and_sampler'])
        self.assertFalse(r['holdout_used']);self.assertFalse(r['ranking_evaluated']);self.assertFalse(r['same_compute'])
        self.assertTrue(r['per_step_local_guard_passed']);self.assertIsNotNone(r['ungated'])
        self.assertEqual(len(r['hard_gated_minus_plain']['positive_damage_cluster_ci95']),2)
        for key,digest in g.f.read_json(self.root/'full/summary.json')['artifact_files_sha256'].items():
            self.assertEqual(g.f.sha256(self.root/'full'/key),digest)
        self.train('short',steps=3)
        args.rqvae_dir=str(self.root/'short');args.output=str(self.root/'bad_compare')
        with self.assertRaisesRegex(ValueError,'step budget'):g.compare(args)
        self.assertFalse(Path(args.output).exists())

    def test_wrong_resume_and_bad_config_fail_before_output(self):
        self.train('smoke',steps=3)
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            self.train('bad',rho=.2,resume=self.root/'smoke/last.pth')
        self.assertFalse((self.root/'bad').exists())
        with self.assertRaises(FileExistsError):self.train('smoke')
        args=self.args('bad_ratio');args.rq_mse_ratio_cap=.9
        with self.assertRaises(ValueError):g.train(args)
        self.assertFalse((self.root/'bad_ratio').exists())
        args=self.args('too_long',steps=13)
        with self.assertRaisesRegex(ValueError,'budget'):g.train(args)
        self.assertFalse((self.root/'too_long').exists())
        args=self.args('wrong_forward');args.anchor_batch_size=4
        with self.assertRaisesRegex(ValueError,'forward settings'):g.train(args)
        self.assertFalse((self.root/'wrong_forward').exists())


if __name__=='__main__':unittest.main(verbosity=2)
