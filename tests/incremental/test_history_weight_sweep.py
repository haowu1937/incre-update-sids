"""Synthetic CPU checks for weight extension, independent selection and subprocess isolation."""
import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import history_weight_update as w
import history_update_multiseed as m
import run_history_weight_sweep as s
torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('multiseed_fixture',ROOT/'tests/incremental/test_multiseed_history_update.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.fixture=module.Checks;cls.fixture.setUpClass()

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def setUp(self):
        self.f=self.fixture();self.f.setUp();self.root=self.f.root;s.ABORT.clear()

    def tearDown(self):self.f.tearDown();s.ABORT.clear()

    def train(self,name,rho=.1,seed=2027,steps=12,resume=None):
        a=self.f.args(name,seed=seed,steps=steps,resume=resume);a.history_gradient_ratio=rho
        with contextlib.redirect_stdout(io.StringIO()):return w.train(a)

    def test_rho_point_one_exact_legacy_parity_and_frozen_encoder(self):
        self.f.train('original',seed=2027);r=self.train('extended')
        a,b=self.f.cp('original'),self.f.cp('extended')
        for key in ('state_dict','optimizer','sampler_state','rng_state','old_anchor_nll'):
            self.assertTrue(m.audit.equal(a[key],b[key]),key)
        self.assertTrue(r['encoder_unchanged_from_base'])

    def test_larger_ratios_are_not_clipped_and_invalid_ratios_fail(self):
        for rho in (1.,3.,10.):
            rq=[torch.tensor([.1,.2]),torch.tensor([.3,.1])]
            hg=[torch.tensor([1.,-1.]),torch.tensor([2.,3.])]
            mixed,entries=w.normalized_mix(rq,hg,rho,1e-12)
            for a,b,c,entry in zip(rq,hg,mixed,entries):
                self.assertAlmostEqual(float((c-a).norm()/a.norm()),rho,places=5)
                self.assertAlmostEqual(entry['effective_gradient_ratio'],rho,places=5)
        r=self.train('rho3',rho=3.)
        self.assertEqual(r['config']['historical_v2']['history_gradient_ratio'],3.)
        self.assertTrue(r['encoder_unchanged_from_base'])
        for rho in (-1,float('nan'),10.01):
            with self.assertRaisesRegex(ValueError,'Invalid'):self.train('invalid',rho=rho)
        self.assertFalse((self.root/'invalid').exists())

    def test_large_ratio_exact_resume_and_configuration_guard(self):
        self.train('full',rho=3.);self.train('smoke',rho=3.,steps=9)
        self.train('resumed',rho=3.,resume=self.root/'smoke/last.pth')
        a,b=self.f.cp('full'),self.f.cp('resumed')
        for key in ('state_dict','optimizer','sampler_state','rng_state'):
            self.assertTrue(m.audit.equal(a[key],b[key]),key)
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            self.train('wrong',rho=1.,resume=self.root/'smoke/last.pth')
        self.assertFalse((self.root/'wrong').exists())
        # A crash after the last optimizer step can finalize exports without
        # taking another step or changing its model/Adam/sampler/RNG state.
        r=self.train('finalize',rho=3.,resume=self.root/'full/last.pth')
        self.assertEqual(r['steps_this_run'],0)
        c=self.f.cp('finalize')
        for key in ('state_dict','optimizer','sampler_state','rng_state'):
            self.assertTrue(m.audit.equal(a[key],c[key]),key)

    def test_train_only_independent_evaluation_reuses_legacy_plain(self):
        # Seed 2026 uses the earlier format without a recorded per-batch digest.
        self.f.h.train('2026_plain',mode='plain')
        self.train('2026_history',rho=3.,seed=2026)
        a=self.f.eval_args(seed=2026);a.train_only=True
        original=w.audit.independent_records;seen=[]
        def train_only(teacher,ex,split,chosen,*args,**kwargs):
            self.assertEqual(split,'base_train');self.assertEqual(list(chosen),ex.cohorts['anchor_train'])
            seen.append(split);return original(teacher,ex,split,chosen,*args,**kwargs)
        with mock.patch.object(w.audit,'independent_records',side_effect=train_only),contextlib.redirect_stdout(io.StringIO()):
            r=w.evaluate(a)
        self.assertEqual(len(seen),2);self.assertEqual(list(r['groups']),['anchor_train'])
        self.assertFalse(r['holdout_evaluated']);self.assertTrue(r['within_seed_same_batch_order'])
        a.output=str(self.root/'held');a.train_only=False
        with contextlib.redirect_stdout(io.StringIO()):r=w.evaluate(a)
        self.assertEqual(set(r['groups']),{'anchor_train','anchor_holdout','base_validation'})

    def report(self,seed,delta=-.02,ci=(-.03,-.01),mse=1.):
        return dict(update_seed=seed,groups=dict(anchor_train=dict(
            plain=dict(positive_damage=1.),history_aware=dict(positive_damage=1.+delta),
            paired=dict(history_minus_plain_positive_damage=delta,positive_damage_cluster_ci95=list(ci)))),
            rq_mse=dict(plain=1.,history=mse))

    def test_selection_requires_all_seeds_quality_and_adjacent_levels(self):
        good=[self.report(seed) for seed in s.SEEDS]
        passed=s.level_result(.3,good);self.assertTrue(passed['training_level_passed'])
        self.assertIsNone(s.select([passed]))
        next_pass=s.level_result(1.,good)
        self.assertEqual(s.select([passed,next_pass]),1.)
        for bad in (self.report(2028,delta=.01),self.report(2028,ci=(-.03,.001)),self.report(2028,mse=1.011),
                    dict(update_seed=2028,eligible=False,reason='capacity')):
            failed=s.level_result(.3,good[:2]+[bad]);self.assertFalse(failed['training_level_passed'])
            self.assertIsNone(s.select([passed,failed,next_pass]))
        with self.assertRaisesRegex(ValueError,'exactly'):s.level_result(.3,good[:2])
        # Held-out contents cannot influence selection, even contradictory ones.
        mutated=copy.deepcopy(good)
        for r in mutated:r['groups']['anchor_holdout']={'positive_damage':1e99}
        self.assertEqual(s.level_result(.3,good),s.level_result(.3,mutated))

    def test_three_seed_jobs_are_real_separate_concurrent_processes(self):
        engine=self.root/'fake_engine.py'
        engine.write_text("import json,os,sys,time\nfrom pathlib import Path\na=time.time();time.sleep(.4)\nPath(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),start=a,end=time.time())))\n")
        jobs=[]
        for seed in s.SEEDS:
            jobs.append((str(seed),lambda seed=seed:s.command([self.root/(str(seed)+'.json')],self.root/(str(seed)+'.log'))))
        with mock.patch.object(s,'ENGINE',engine),contextlib.redirect_stdout(io.StringIO()):s.parallel(jobs,3)
        rows=[s.read(self.root/(str(seed)+'.json')) for seed in s.SEEDS]
        self.assertEqual(len({r['pid'] for r in rows}),3)
        self.assertLess(max(r['start'] for r in rows),min(r['end'] for r in rows))

    def test_subprocess_failure_cancels_other_seed_and_no_false_completion(self):
        engine=self.root/'failure.py';marker=self.root/'should_not_exist'
        engine.write_text("import sys,time\nfrom pathlib import Path\nif sys.argv[1]=='bad':sys.exit(17)\ntime.sleep(10)\nPath(sys.argv[1]).write_text('wrong')\n")
        with mock.patch.object(s,'ENGINE',engine),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'exited 17'):
                s.parallel([('bad',lambda:s.command(['bad'],self.root/'bad.log')),
                            ('other',lambda:s.command([marker],self.root/'other.log'))],2)
        self.assertFalse(marker.exists());self.assertFalse(s.ACTIVE)

    def test_controller_freezes_selection_before_heldout_and_stops_at_limit(self):
        previous=self.root/'previous'
        for seed in s.SEEDS:
            s.write(previous/('seed'+str(seed)+'_evaluation')/'summary.json',dict(
                stage=m.EVALUATION,update_seed=seed,update_optimizer_steps=100,
                design=dict(historical_v2=dict(history_gradient_ratio=.1)),common_binding={},
                cohort_hashes={},evaluation_sources={},content_batch_trace_sha256=str(seed),
                plain_dir=str(self.root/('plain'+str(seed))),history_dir=str(self.root/('old_history'+str(seed))),
                artifact_files_sha256={}))
        for improvement in (True,False):
            output=self.root/('run_'+str(improvement));calls=[]
            args=s.parser().parse_args(['--previous-root',str(previous),'--e1-dir',str(self.fixture.fixture.previous.e1_dir),
                                       '--output',str(output),'--devices','cpu','--workers','3'])
            def training(root,e1,seed,rho,device):
                calls.append(('train',rho,seed))
                return Path(root)/('rho_'+s.tag(rho))/str(seed),dict(execution_status='completed',full_sid_export_status='completed')
            def evaluation(e1,plain,history,out,seed,device,train_only):
                rho=.1 if 'old_history' in str(history) else float(next(p[4:] for p in Path(history).parts if p.startswith('rho_')).replace('p','.'))
                calls.append(('eval',rho,seed,train_only))
                r=self.report(seed,delta=-.02 if improvement and rho>=.3 else .02)
                if not train_only:
                    self.assertEqual(s.read(output/'selection.json')['selected_rho'],1.)
                    for group in ('anchor_holdout','base_validation'):r['groups'][group]=copy.deepcopy(r['groups']['anchor_train'])
                return r
            with mock.patch.object(s,'train_seed',side_effect=training),mock.patch.object(s,'evaluate',side_effect=evaluation),contextlib.redirect_stdout(io.StringIO()):
                result=s.run(args)
            if improvement:
                self.assertEqual(result['selected_rho'],1.)
                self.assertEqual(len([c for c in calls if c[0]=='train']),6)
                self.assertEqual(len([c for c in calls if c[0]=='eval' and not c[3]]),3)
            else:
                self.assertIsNone(result['selected_rho'])
                self.assertEqual(len([c for c in calls if c[0]=='train']),12)
                self.assertFalse(any(c[0]=='eval' and not c[3] for c in calls))
                self.assertEqual(result['stop_reason'],'rho_limit_reached_without_stable_training_improvement')


if __name__=='__main__':unittest.main(verbosity=2)
