"""CPU synthetic checks: frozen encoder intervention, matched control, exact resume."""
import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import history_aware_frozen_encoder as f

torch.set_num_threads(1)


def exact(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('round12_fixture',ROOT/'tests/incremental/test_historical_update_round12.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.previous=module.Checks;cls.previous.setUpClass()

    @classmethod
    def tearDownClass(cls):cls.previous.tearDownClass()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)

    def tearDown(self):self.tmp.cleanup()

    def args(self,name='trial',mode='history-aware',steps=12,resume=None):
        v=['train','--mode',mode,'--e1-dir',str(self.previous.e1_dir),'--output',str(self.root/name),
           '--device','cpu','--max-steps',str(steps),'--anchor-batch-size','3','--log-every','3',
           '--save-every','3','--history-progress-every','0']
        if resume:v+=['--resume-from',str(resume)]
        return f.parser().parse_args(v)

    def train(self,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return f.train(self.args(*args,**kwargs))

    def checkpoint(self,name):return torch.load(self.root/name/'last.pth',weights_only=False)

    def test_first_step_arms_identical_and_encoder_frozen(self):
        a=self.train('plain',mode='plain',steps=1);b=self.train('ha',steps=1)
        ca,cb=self.checkpoint('plain'),self.checkpoint('ha')
        self.assertTrue(exact(ca['state_dict'],cb['state_dict']))
        self.assertTrue(exact(ca['optimizer'],cb['optimizer']))
        ex=f.h.experiment(self.args())
        for k,v in ca['state_dict'].items():
            if k.startswith('encoder.'):self.assertTrue(torch.equal(v,ex.parent['state_dict'][k]))
        for r in (a,b):
            self.assertTrue(r['encoder_unchanged_from_base'])
            self.assertEqual(r['trained_components'],['decoder','three_codebooks'])
            self.assertEqual(r['full_anchor_train_at_run_start']['positive_damage'],0.)
        self.assertEqual(a['work_counters']['history_gradient_queries'],0)
        self.assertEqual(b['work_counters']['history_gradient_queries'],12)
        self.assertFalse(b['anchor_holdout_used_for_training'])

    def test_plain_matches_independent_frozen_content_adam_reference(self):
        report=self.train('plain',mode='plain')
        cp=self.checkpoint('plain');ex=f.h.experiment(self.args())
        f.seed_all(cp['config']['seed'])
        model=f.build_model(cp['config']);model.load_state_dict(ex.parent['state_dict'])
        model.encoder.requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cp['config']['lr'],weight_decay=cp['config']['weight_decay'])
        generator=torch.Generator().manual_seed(cp['config']['seed'])
        cursor=len(ex.features);processed=0
        for step in range(12):
            if cursor==len(ex.features):
                order=torch.randperm(len(ex.features),generator=generator);cursor=0
            ids=order[cursor:cursor+cp['config']['batch_size']];cursor+=len(ids);processed+=len(ids)
            model.train();model.encoder.eval();optimizer.zero_grad(set_to_none=True)
            data=ex.features[ids];out,qloss,_,dense,books,residuals=model(data,None,use_sk=False)
            loss=model.compute_loss(out,qloss,ids,dense,dense_out_codebooks=books,x_res_list=residuals,xs=data)[0]
            loss.backward();optimizer.step()
        self.assertTrue(exact(model.state_dict(),cp['state_dict']))
        self.assertTrue(exact(optimizer.state_dict(),cp['optimizer']))
        self.assertEqual(processed,report['processed_examples'])
        self.assertTrue(any(not torch.equal(v,ex.parent['state_dict'][k]) for k,v in cp['state_dict'].items() if k.startswith('decoder.')))
        self.assertTrue(any(not torch.equal(v,ex.parent['state_dict'][k]) for k,v in cp['state_dict'].items() if k.startswith('rq.')))

    def test_exact_resume_both_arms_across_sampler_epochs(self):
        for mode in ('plain','history-aware'):
            with self.subTest(mode=mode):
                full=self.train(mode+'_full',mode=mode)
                self.train(mode+'_smoke',mode=mode,steps=3)
                resumed=self.train(mode+'_resume',mode=mode,resume=self.root/(mode+'_smoke')/'last.pth')
                ca,cb=self.checkpoint(mode+'_full'),self.checkpoint(mode+'_resume')
                for key in ('state_dict','optimizer','sampler_state','rng_state'):
                    self.assertTrue(exact(ca[key],cb[key]),key)
                self.assertEqual(full['final_full_anchor_train'],resumed['final_full_anchor_train'])
                self.assertEqual(resumed['steps_this_run'],9)
                self.assertTrue(full['encoder_unchanged_from_base'])
                if mode=='history-aware':
                    self.assertEqual(full['work_counters']['history_gradient_queries'],144)
                    self.assertGreater(full['work_counters']['nonzero_history_gradient_steps'],0)

    def test_history_forward_labels_full_coverage_and_gradient_routing(self):
        # The frozen-encoder toy run has H=0 for its first eight steps.
        # Use its real step-9 positive-damage endpoint, with the original old NLL unchanged.
        self.train('active',steps=9)
        cp=self.checkpoint('active');ex=f.h.experiment(self.args())
        model=f.build_model(cp['config']);model.load_state_dict(cp['state_dict']);model.encoder.requires_grad_(False)
        teacher=f.h.frozen_teacher(ex,'cpu')
        with contextlib.redirect_stdout(io.StringIO()):
            history=f.FullHistory(model,teacher,ex,cp['config'],cp['distance_scales'],0)
        codes,_,_,mapping=f.catalog(model,ex)
        hard,grads=history.measure(codes,mapping,True)
        self.assertEqual(history.selected.tolist(),ex.cohorts['anchor_train'])
        self.assertFalse(set(history.selected)&set(ex.cohorts['anchor_holdout']))
        self.assertEqual(hard['queries'],12)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        expected=f.v2.reencode_history(ex.arrays['base_train'],ex.rows['base_train'],mapping)
        for pack in history.packs:
            old,ids,embeds=history.inputs(pack,codes,mapping,True)
            self.assertTrue(np.array_equal(ids.numpy(),expected['input_ids'][pack['indices'],:ids.shape[1]]))
            self.assertTrue(torch.equal(embeds.detach(),teacher.t5.get_input_embeddings()(ids)))
            self.assertTrue(np.array_equal(old['labels'],ex.arrays['base_train']['labels'][pack['indices']]))
        rows=f.hard_records(history,codes,mapping)
        self.assertAlmostEqual(np.mean([r['positive_damage'] for r in rows]),hard['positive_damage'],places=6)
        f.h.check_teacher(teacher,ex)
        config=copy.deepcopy(cp['config']);config['historical_v2']['anchor_batch_size']=5
        with contextlib.redirect_stdout(io.StringIO()):
            partial=f.FullHistory(model,teacher,ex,config,cp['distance_scales'],0)
        self.assertEqual([len(p['indices']) for p in partial.packs],[5,5,2])
        hm,gg=partial.measure(codes,mapping,True)
        self.assertAlmostEqual(hm['positive_damage'],hard['positive_damage'],places=6)
        self.assertTrue(all(torch.allclose(a,b,atol=1e-6,rtol=1e-3) for a,b in zip(grads,gg)))

    def test_compare_matched_endpoints_and_train_only_statistics(self):
        self.train('plain',mode='plain');self.train('ha')
        args=f.parser().parse_args(['compare','--plain-dir',str(self.root/'plain'),'--history-dir',str(self.root/'ha'),'--output',str(self.root/'contrast')])
        with contextlib.redirect_stdout(io.StringIO()):r=f.compare(args)
        self.assertEqual(r['queries'],12);self.assertTrue(r['same_content_optimizer_and_sampler'])
        self.assertTrue(r['encoder_frozen_and_unchanged_both']);self.assertFalse(r['same_compute'])
        self.assertFalse(r['holdout_used']);self.assertFalse(r['ranking_evaluated'])
        a=f.read_jsonl(self.root/'plain/anchor_train_hard_losses.jsonl');b=f.read_jsonl(self.root/'ha/anchor_train_hard_losses.jsonl')
        self.assertAlmostEqual(r['history_aware_minus_plain']['positive_damage'],np.mean([y['positive_damage']-x['positive_damage'] for x,y in zip(a,b)]),places=12)
        for name in ('plain','ha'):
            summary=f.read_json(self.root/name/'summary.json')
            self.assertTrue(all(f.sha256(self.root/name/key)==value for key,value in summary['artifact_files_sha256'].items()))
        self.train('short',mode='plain',steps=3)
        args.plain_dir=str(self.root/'short');args.output=str(self.root/'bad_compare')
        with self.assertRaisesRegex(ValueError,'step budget'):f.compare(args)
        self.assertFalse(Path(args.output).exists())

    def test_overflow_rollback_and_source_checkpoint_immutability(self):
        ex=f.h.experiment(self.args());base=Path(ex.plain['base_rqvae_dir'])/'last.pth';digest=f.sha256(base)
        self.train('one',steps=1)
        original=f.catalog;calls=[0]
        def reject(model,ex):
            result=original(model,ex);calls[0]+=1
            if calls[0]==3:return result[0],dict(result[1],full_sid_allocation_feasible=False),None,None
            return result
        with mock.patch.object(f,'catalog',side_effect=reject):r=self.train('overflow')
        a,b=self.checkpoint('one'),self.checkpoint('overflow')
        self.assertEqual(r['execution_status'],'stopped');self.assertEqual(r['update_optimizer_steps'],1)
        for key in ('state_dict','optimizer','sampler_state','rng_state'):self.assertTrue(exact(a[key],b[key]),key)
        self.assertEqual(f.sha256(base),digest);self.assertTrue(r['encoder_unchanged_from_base'])
        self.assertEqual(r['full_sid_export_status'],'completed')

    def test_bad_resume_and_existing_output_fail_before_writes(self):
        self.train('plain',mode='plain',steps=3)
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            self.train('bad',resume=self.root/'plain/last.pth')
        self.assertFalse((self.root/'bad').exists())
        with self.assertRaises(FileExistsError):self.train('plain',mode='plain')
        args=self.args('invalid');args.history_gradient_ratio=float('nan')
        with self.assertRaises(ValueError):f.train(args)
        self.assertFalse((self.root/'invalid').exists())


if __name__=='__main__':unittest.main(verbosity=2)
