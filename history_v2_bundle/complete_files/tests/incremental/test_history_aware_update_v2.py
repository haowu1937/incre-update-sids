"""Synthetic CPU regression checks with actual repository RQ-VAE and TIGER."""
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

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import history_aware_update_v2 as v2
torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('previous_fixture',ROOT/'tests/incremental/test_historical_update_round12.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cls.previous = module.Checks; cls.previous.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.previous.tearDownClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self,name='joint',steps=12,rho=.1,resume=None):
        values = ['train','--e1-dir',str(self.previous.e1_dir),'--output',str(self.root/name),
            '--device','cpu','--max-steps',str(steps),'--history-gradient-ratio',str(rho),
            '--anchor-batch-size','3','--history-progress-every','0','--log-every','100','--save-every','3']
        if resume: values += ['--resume-from',str(resume)]
        return v2.parser().parse_args(values)

    def train(self,args):
        with contextlib.redirect_stdout(io.StringIO()): return v2.train(args)

    def context(self,plain=True):
        ex = v2.h.experiment(self.args())
        config = copy.deepcopy(ex.plain['config'])
        config['historical_v2'] = dict(anchor_batch_size=3,temperature=.1)
        model = v2.build_model(config)
        model.load_state_dict(ex.parent['state_dict'])
        scales = v2.h.distance_scales(model,ex.features[:ex.groups['base_train'][1]],ex.old_batch)
        if plain:
            cp = torch.load(Path(ex.e1['update_rqvae_dir'])/'last.pth',weights_only=False)
            model.load_state_dict(cp['state_dict'])
        teacher = v2.h.frozen_teacher(ex,'cpu')
        with contextlib.redirect_stdout(io.StringIO()):
            history = v2.FullHistory(model,teacher,ex,config,scales,progress_every=0)
        codes,capacity,ledger,mapping = v2.catalog(model,ex)
        return ex,model,teacher,history,codes,mapping

    def test_normalized_mix_bounds_and_zero_gradients(self):
        a = torch.tensor([[1e-4,0.]],dtype=torch.float64)
        b = torch.tensor([[0.,.01]],dtype=torch.float64)
        mixed,stats = v2.normalized_mix([a],[b],.1,1e-12)
        self.assertAlmostEqual(stats[0]['alpha'],.001,places=9)
        self.assertAlmostEqual(float((mixed[0]-a).norm()/a.norm()),.1,places=9)
        self.assertFalse(mixed[0].requires_grad)
        for rq,history in [(a,torch.zeros_like(b)),(torch.zeros_like(a),b)]:
            result,entry = v2.normalized_mix([rq],[history],.1,1e-12)
            self.assertEqual(entry[0]['alpha'],0.)
            self.assertTrue(torch.equal(result[0],rq))
        with self.assertRaises(FloatingPointError):
            v2.normalized_mix([a],[b*float('nan')],.1,1e-12)

    def test_complete_sid_hard_forward_original_labels_and_book_only_gradient(self):
        ex,model,teacher,history,codes,mapping = self.context()
        pack = history.packs[0]
        old,ids,embeds = history.inputs(pack,codes,mapping,True)
        expected = v2.reencode_history(ex.arrays['base_train'],ex.rows['base_train'],mapping)
        self.assertTrue(np.array_equal(ids.numpy(),expected['input_ids'][pack['indices'],:ids.shape[1]]))
        self.assertTrue(torch.equal(embeds.detach(),teacher.t5.get_input_embeddings()(ids)))
        self.assertTrue(np.array_equal(old['labels'].numpy(),ex.arrays['base_train']['labels'][pack['indices']]))
        self.assertTrue(np.array_equal(old['attention_mask'].numpy(),ex.arrays['base_train']['attention_mask'][pack['indices'],:ids.shape[1]]))
        logits = teacher(dict(inputs_embeds=embeds,attention_mask=old['attention_mask'],labels=old['labels'],use_cache=False)).logits
        parameters = list(model.named_parameters())
        values = torch.autograd.grad(torch.nn.functional.cross_entropy(logits.transpose(1,2),old['labels']),
            [p for _,p in parameters],allow_unused=True)
        for (name,_),gradient in zip(parameters,values):
            if name.startswith(('encoder.','decoder.')): self.assertIsNone(gradient)
        self.assertGreater(sum(float(g.abs().sum()) for (name,_),g in zip(parameters,values) if g is not None and name.startswith('rq.')),0)
        v2.h.check_teacher(teacher,ex)

    def test_full_anchor_accumulation_preserves_content_grads_and_identity_zero(self):
        # This toy Plain endpoint improves every selected query, so its ReLU H is zero.
        # Use an actual intermediate joint checkpoint with positive damage for the
        # active-gradient check; do not alter the frozen old NLL reference.
        self.train(self.args(name='gradient_point',steps=3))
        ex,model,teacher,history,codes,mapping = self.context()
        model.load_state_dict(torch.load(self.root/'gradient_point/last.pth',weights_only=False)['state_dict'])
        codes,_,_,mapping = v2.catalog(model,ex)
        data = ex.features[:3]
        values = model(data,None,use_sk=False)
        loss = model.compute_loss(values[0],values[1],torch.arange(3),values[3],
            dense_out_codebooks=values[4],x_res_list=values[5],xs=data)[0]
        loss.backward()
        before = {n:p.grad.clone() if p.grad is not None else None for n,p in model.named_parameters()}
        measured,gradients = history.measure(codes,mapping,backward=True,step=1)
        self.assertEqual(measured['queries'],12)
        self.assertEqual(history.selected.tolist(),ex.cohorts['anchor_train'])
        self.assertFalse(set(history.selected)&set(ex.cohorts['anchor_holdout']))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)
        for n,p in model.named_parameters():
            self.assertTrue((p.grad is None and before[n] is None) or torch.equal(p.grad,before[n]))
        hard,_ = history.measure(codes,mapping,backward=False)
        self.assertAlmostEqual(measured['positive_damage'],hard['positive_damage'],places=7)
        ex,model,teacher,history,codes,mapping = self.context(plain=False)
        measured,gradients = history.measure(codes,mapping,backward=True,step=1)
        self.assertEqual(measured['positive_damage'],0.)
        self.assertEqual(measured['changed_history_fraction'],0.)
        self.assertTrue(all(float(g.abs().sum())==0 for g in gradients))

    def test_zero_ratio_exactly_matches_plain_update_weights_and_sampler(self):
        report = self.train(self.args(rho=0))
        ex = v2.h.experiment(self.args())
        a = torch.load(self.root/'joint/last.pth',weights_only=False)
        b = torch.load(Path(ex.e1['update_rqvae_dir'])/'last.pth',weights_only=False)
        self.assertEqual(report['update_optimizer_steps'],12)
        self.assertTrue(all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items()))
        self.assertTrue(torch.equal(a['sampler_state']['order'],b['sampler_state']['order']))
        self.assertEqual(a['sampler_state']['cursor'],b['sampler_state']['cursor'])
        self.assertEqual(report['processed_examples'],ex.plain['processed_examples'])
        self.assertEqual(report['work_counters']['history_gradient_queries'],0)
        self.assertEqual(report['final_train_catalog_metrics'],ex.plain['final_train_catalog_metrics'])

    def test_nonzero_resume_matches_uninterrupted_all_components_and_history_coverage(self):
        full = self.train(self.args(name='full'))
        smoke = self.train(self.args(name='smoke',steps=3))
        continued = self.train(self.args(name='continued',resume=self.root/'smoke/last.pth'))
        a = torch.load(self.root/'full/last.pth',weights_only=False)
        b = torch.load(self.root/'continued/last.pth',weights_only=False)
        self.assertEqual(full['update_optimizer_steps'],12)
        self.assertEqual(continued['steps_this_run'],9)
        self.assertTrue(all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items()))
        self.assertTrue(torch.equal(a['sampler_state']['order'],b['sampler_state']['order']))
        self.assertEqual(a['sampler_state']['cursor'],b['sampler_state']['cursor'])
        self.assertEqual(full['work_counters']['history_gradient_queries'],12*12)
        self.assertEqual(continued['work_counters']['history_gradient_queries'],12*12)
        self.assertGreater(full['work_counters']['nonzero_history_gradient_steps'],0)
        ex = v2.h.experiment(self.args())
        for prefix in ('encoder.','decoder.','rq.'):
            self.assertTrue(any(not torch.equal(v,ex.parent['state_dict'][k]) for k,v in a['state_dict'].items() if k.startswith(prefix)))
        rows = v2.read_jsonl(self.root/'full/metrics.jsonl')
        self.assertTrue(all(row['history_before_optimizer_step']['queries']==12 for row in rows))
        for row in rows:
            for entry in row['codebook_gradients']:
                self.assertLessEqual(entry['effective_gradient_ratio'],.100001)

    def test_capacity_failure_restores_model_optimizer_sampler_and_saves_feasible_checkpoint(self):
        self.train(self.args(name='before_capacity',steps=1))
        actual = v2.catalog
        calls = [0]
        def fail_candidate(model,ex):
            result = actual(model,ex); calls[0] += 1
            if calls[0]==3:
                return result[0],dict(result[1],full_sid_allocation_feasible=False),None,None
            return result
        with mock.patch.object(v2,'catalog',side_effect=fail_candidate):
            report = self.train(self.args(name='capacity'))
        self.assertEqual(report['execution_status'],'stopped')
        self.assertEqual(report['update_optimizer_steps'],1)
        self.assertEqual(report['work_counters']['rolled_back_optimizer_attempts'],1)
        ex = v2.h.experiment(self.args())
        cp = torch.load(self.root/'capacity/last.pth',weights_only=False)
        before = torch.load(self.root/'before_capacity/last.pth',weights_only=False)
        self.assertTrue(all(torch.equal(v,before['state_dict'][k]) for k,v in cp['state_dict'].items()))
        for parameter,state in cp['optimizer']['state'].items():
            for key,value in state.items():
                self.assertTrue(torch.equal(value,before['optimizer']['state'][parameter][key]))
        self.assertEqual(cp['sampler_state']['cursor'],before['sampler_state']['cursor'])
        self.assertTrue(torch.equal(cp['sampler_state']['order'],before['sampler_state']['order']))
        self.assertTrue(torch.equal(cp['sampler_state']['generator_state'],before['sampler_state']['generator_state']))
        self.assertEqual(report['full_sid_export_status'],'completed')

    def test_partial_history_batches_accumulate_the_full_query_mean(self):
        self.train(self.args(name='partial_gradient_point',steps=3))
        ex,model,teacher,history,codes,mapping = self.context()
        model.load_state_dict(torch.load(self.root/'partial_gradient_point/last.pth',weights_only=False)['state_dict'])
        codes,_,_,mapping = v2.catalog(model,ex)
        config = copy.deepcopy(history.config)
        with contextlib.redirect_stdout(io.StringIO()):
            config['historical_v2']['anchor_batch_size'] = 5
            partial = v2.FullHistory(model,teacher,ex,config,history.scales,progress_every=0)
            config['historical_v2']['anchor_batch_size'] = 12
            complete = v2.FullHistory(model,teacher,ex,config,history.scales,progress_every=0)
        self.assertEqual([len(p['indices']) for p in partial.packs],[5,5,2])
        a,ga = partial.measure(codes,mapping,backward=True)
        b,gb = complete.measure(codes,mapping,backward=True)
        self.assertAlmostEqual(a['positive_damage'],b['positive_damage'],places=6)
        for x,y in zip(ga,gb):
            self.assertTrue(torch.allclose(x,y,rtol=1e-3,atol=1e-6))

    def test_matched_evaluation_ranking_cis_hashes_and_source_immutability(self):
        ex = v2.h.experiment(self.args())
        before = {str(p):v2.sha256(p) for directory in [ex.e1_dir,Path(self.previous.previous.fixture.name)]
            for p in directory.rglob('*') if p.is_file()}
        report = self.train(self.args())
        for name,digest in report['artifact_files_sha256'].items():
            self.assertEqual(v2.sha256(self.root/'joint'/name),digest)
        args = v2.parser().parse_args(['evaluate','--e1-dir',str(ex.e1_dir),'--rqvae-dir',str(self.root/'joint'),
            '--output',str(self.root/'evaluation'),'--device','cpu','--batch-size','3','--ranking'])
        with contextlib.redirect_stdout(io.StringIO()): result = v2.evaluate(args)
        self.assertEqual(result['groups']['anchor_holdout']['new_full_sid']['queries'],12)
        self.assertEqual(result['base_validation_ranking']['history_aware']['all_queries'],10)
        self.assertEqual(result['base_validation_ranking']['history_aware']['unavailable_targets'],5)
        self.assertEqual(result['base_validation_ranking']['history_aware']['invalid_predictions'],0)
        self.assertEqual(len(result['paired_history_aware_vs_plain']['anchor_holdout']['new_full_sid']['positive_damage_cluster_ci95']),2)
        self.assertTrue(result['same_optimizer_step_budget_as_plain'])
        self.assertFalse(result['same_compute_as_plain'])
        for name,digest in result['artifact_files_sha256'].items():
            self.assertEqual(v2.sha256(self.root/'evaluation'/name),digest)
        self.assertTrue(all(v2.sha256(Path(path))==digest for path,digest in before.items()))

    def test_bad_config_overwrite_and_short_budget_evaluation_fail_before_output(self):
        args = self.args(rho=float('nan'))
        with self.assertRaises(ValueError): v2.train(args)
        self.assertFalse(Path(args.output).exists())
        with self.assertRaisesRegex(ValueError,'budget'): v2.train(self.args(steps=13))
        self.root.joinpath('exists').mkdir()
        with self.assertRaises(FileExistsError): v2.train(self.args(name='exists'))
        self.train(self.args(name='smoke',steps=3))
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            v2.train(self.args(name='changed',rho=.2,resume=self.root/'smoke/last.pth'))
        self.assertFalse(self.root.joinpath('changed').exists())
        args = v2.parser().parse_args(['evaluate','--e1-dir',str(self.previous.e1_dir),'--rqvae-dir',str(self.root/'smoke'),
            '--output',str(self.root/'bad_eval'),'--device','cpu'])
        with self.assertRaisesRegex(ValueError,'budget'): v2.evaluate(args)
        self.assertFalse(self.root.joinpath('bad_eval').exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
