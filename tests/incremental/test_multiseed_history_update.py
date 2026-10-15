"""CPU checks of seed isolation, legacy parity, paired ordering, resume and aggregation."""
import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import history_update_multiseed as m

torch.set_num_threads(1)

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('frozen_fixture',ROOT/'tests/incremental/test_frozen_encoder_update.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.fixture=module.Checks;cls.fixture.setUpClass()

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def setUp(self):
        self.h=self.fixture();self.h.setUp();self.root=self.h.root

    def tearDown(self):self.h.tearDown()

    def args(self,name,seed=2027,mode='history-aware',steps=12,resume=None):
        values=['train','--e1-dir',str(self.fixture.previous.e1_dir),'--output',str(self.root/name),
            '--mode',mode,'--update-seed',str(seed),'--device','cpu','--max-steps',str(steps),
            '--anchor-batch-size','3','--temperature','0.1','--log-every','3','--save-every','3']
        if resume:values+=['--resume-from',str(resume)]
        return m.parser().parse_args(values)

    def train(self,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return m.train(self.args(*args,**kwargs))

    def cp(self,name):return torch.load(self.root/name/'last.pth',weights_only=False)

    def eval_args(self,seed=2027,prefix=None,name=None):
        prefix=prefix or str(seed);name=name or ('eval_'+str(seed))
        return m.parser().parse_args(['evaluate','--e1-dir',str(self.fixture.previous.e1_dir),
            '--plain-dir',str(self.root/(prefix+'_plain')),'--history-dir',str(self.root/(prefix+'_history')),
            '--output',str(self.root/name),'--update-seed',str(seed),'--expected-steps','12','--device','cpu','--batch-size','3'])

    def evaluate(self,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return m.evaluate(self.eval_args(*args,**kwargs))

    def test_new_engine_seed_2026_exactly_matches_both_legacy_arms(self):
        for mode in ('plain','history-aware'):
            self.h.train('legacy_'+mode,mode=mode)
            r=self.train('new_'+mode,seed=2026,mode=mode)
            a,b=self.cp('legacy_'+mode),self.cp('new_'+mode)
            for key in ('state_dict','optimizer','old_anchor_nll','distance_scales'):
                self.assertTrue(m.audit.equal(a[key],b[key]),key)
            self.assertTrue(m.audit.equal(m.sampler_core(a['sampler_state']),m.sampler_core(b['sampler_state'])))
            self.assertEqual(r['update_seed'],2026)
            self.assertTrue(r['encoder_unchanged_from_base'])

    def test_different_Update_seeds_keep_Base_teacher_and_anchors_fixed(self):
        cps={}
        for seed in (2027,2028):
            a=self.train(str(seed)+'_plain',seed=seed,mode='plain')
            b=self.train(str(seed)+'_history',seed=seed)
            ca,cb=self.cp(str(seed)+'_plain'),self.cp(str(seed)+'_history');cps[seed]=ca
            self.assertTrue(m.audit.equal(ca['sampler_state'],cb['sampler_state']))
            self.assertEqual(a['initial_train_catalog_metrics'],b['initial_train_catalog_metrics'])
            self.assertEqual(ca['distance_scales'],cb['distance_scales'])
            self.assertTrue(torch.equal(ca['old_anchor_nll'],cb['old_anchor_nll']))
            self.assertEqual(a['base_checkpoint_seed'],2026);self.assertEqual(a['anchor_cohort_seed'],2026)
            ex=m.h.experiment(self.args('unused'))
            expected=m.expected_sampler(seed,len(ex.features),a['config']['batch_size'],12)
            self.assertTrue(m.audit.equal(expected,ca['sampler_state']))
            self.assertEqual(m.read_json(self.root/(str(seed)+'_plain')/'anchor_train_tensor_rows.json'),ex.cohorts['anchor_train'])
        self.assertNotEqual(cps[2027]['sampler_state']['batch_trace_sha256'],cps[2028]['sampler_state']['batch_trace_sha256'])
        self.assertEqual(cps[2027]['initial_metrics'],cps[2028]['initial_metrics'])
        self.assertTrue(torch.equal(cps[2027]['old_anchor_nll'],cps[2028]['old_anchor_nll']))
        for key,value in cps[2027]['state_dict'].items():
            if key.startswith('encoder.'):self.assertTrue(torch.equal(value,cps[2028]['state_dict'][key]))

    def test_exact_resume_for_new_seed_both_arms_including_batch_digest(self):
        for mode in ('plain','history-aware'):
            self.train('full_'+mode,mode=mode);self.train('smoke_'+mode,mode=mode,steps=3)
            self.train('resume_'+mode,mode=mode,resume=self.root/('smoke_'+mode)/'last.pth')
            a,b=self.cp('full_'+mode),self.cp('resume_'+mode)
            for key in ('state_dict','optimizer','sampler_state','rng_state'):
                self.assertTrue(m.audit.equal(a[key],b[key]),key)
            # Resume deliberately recomputes the fixed old NLL cache once.
            # Its extra read-only work does not change training or batch order.
            for key,value in a['work_counters'].items():
                if key!='old_reference_queries':self.assertEqual(value,b['work_counters'][key])
            self.assertEqual(b['work_counters']['old_reference_queries'],a['work_counters']['old_reference_queries']+12)

    def test_independent_three_seed_evaluation_and_sample_SD(self):
        self.h.train('2026_plain',mode='plain');self.h.train('2026_history')
        reports=[]
        for seed in (2026,2027,2028):
            if seed!=2026:
                self.train(str(seed)+'_plain',seed=seed,mode='plain');self.train(str(seed)+'_history',seed=seed)
            reports.append(self.evaluate(seed=seed))
        self.assertTrue(reports[0]['legacy_2026_training_reused'])
        self.assertTrue(all(r['cohort_hashes']==reports[0]['cohort_hashes'] for r in reports))
        args=m.parser().parse_args(['aggregate','--evaluation-dirs',*[str(self.root/('eval_'+str(s))) for s in (2026,2027,2028)],'--output',str(self.root/'aggregate')])
        with contextlib.redirect_stdout(io.StringIO()):r=m.aggregate(args)
        self.assertEqual(r['seeds'],[2026,2027,2028]);self.assertFalse(r['base_or_teacher_seed_varied'])
        for group in r['groups']:
            delta=[x['groups'][group]['history_aware']['positive_damage']-x['groups'][group]['plain']['positive_damage'] for x in reports]
            result=r['groups'][group]['positive_damage']['history_minus_plain']
            self.assertAlmostEqual(result['mean'],float(np.mean(delta)),places=12)
            self.assertAlmostEqual(result['sample_std'],float(np.std(delta,ddof=1)),places=12)
        self.assertTrue(r['across_seed_different_orders']);self.assertFalse(r['holdout_used_for_training'])
        args.output=str(self.root/'duplicate');args.evaluation_dirs[2]=args.evaluation_dirs[1]
        with self.assertRaisesRegex(ValueError,'exactly'):m.aggregate(args)
        self.assertFalse(Path(args.output).exists())
        args.evaluation_dirs[2]=str(self.root/'eval_2028');args.output=str(self.root/'changed_cohort')
        d=self.root/'eval_2028/summary.json';saved=m.read_json(d);bad=copy.deepcopy(saved);bad['cohort_hashes']['anchor_holdout']='bad';m.write_json(d,bad)
        with self.assertRaisesRegex(ValueError,'cohort_hashes'):m.aggregate(args)
        self.assertFalse(Path(args.output).exists());m.write_json(d,saved)

    def test_wrong_seed_resume_budget_and_batch_digest_fail(self):
        self.train('2027_plain',mode='plain');self.train('2027_history')
        args=self.eval_args(seed=2028,prefix='2027')
        with self.assertRaisesRegex(ValueError,'seed differs'):m.evaluate(args)
        self.assertFalse(Path(args.output).exists())
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            self.train('bad_resume',seed=2028,resume=self.root/'2027_history/last.pth')
        self.assertFalse((self.root/'bad_resume').exists())
        args=self.args('invalid',seed=-1)
        with self.assertRaisesRegex(ValueError,'Update seed'):m.train(args)
        self.assertFalse(Path(args.output).exists())
        args=self.eval_args();args.expected_steps=100
        with self.assertRaisesRegex(ValueError,'step budget'):m.evaluate(args)
        args=self.eval_args();cp=self.cp('2027_plain');cp['sampler_state']['batch_trace_sha256']='f'*64
        torch.save(cp,self.root/'2027_plain/last.pth')
        r=m.read_json(self.root/'2027_plain/summary.json');r['artifact_files_sha256']['last.pth']=m.sha256(self.root/'2027_plain/last.pth');r['content_batch_trace_sha256']='f'*64;m.write_json(self.root/'2027_plain/summary.json',r)
        with self.assertRaisesRegex(ValueError,'batch digest'):m.evaluate(args)


if __name__=='__main__':unittest.main(verbosity=2)
