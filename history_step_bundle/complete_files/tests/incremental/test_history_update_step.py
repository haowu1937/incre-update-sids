"""Synthetic CPU checks for checkpoint-preserving single-step interventions."""
import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import diagnose_history_update_step as diag
torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('v2_fixture',ROOT/'tests/incremental/test_history_aware_update_v2.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.previous=module.Checks;cls.previous.setUpClass()
        cls.helper=cls.previous();cls.helper.setUp()
        cls.helper.train(cls.helper.args(name='early',steps=3))
        cls.helper.train(cls.helper.args(name='late',resume=cls.helper.root/'early/last.pth'))

    @classmethod
    def tearDownClass(cls):
        cls.helper.tearDown();cls.previous.tearDownClass()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)

    def tearDown(self):self.tmp.cleanup()

    def args(self,output='diagnostic'):
        return diag.parser().parse_args(['--e1-dir',str(self.previous.previous.e1_dir),
            '--rqvae-dirs',str(self.helper.root/'early'),str(self.helper.root/'late'),
            '--output',str(self.root/output),'--device','cpu','--progress-every','0'])

    def test_same_adamw_joint_proposal_exactly_replays_the_next_v2_training_step(self):
        ex=diag.v2.h.experiment(self.args())
        cp=torch.load(self.helper.root/'early/last.pth',weights_only=False)
        model=diag.v2.build_model(cp['config']);model.load_state_dict(cp['state_dict'])
        teacher=diag.v2.h.frozen_teacher(ex,'cpu')
        with contextlib.redirect_stdout(io.StringIO()):
            history=diag.v2.FullHistory(model,teacher,ex,cp['config'],cp['distance_scales'],progress_every=0)
        selected=diag.next_content_batch(cp,len(ex.features))
        before_cursor=cp['sampler_state']['cursor']
        if before_cursor<len(ex.features):
            expected_rows=cp['sampler_state']['order'][before_cursor:before_cursor+len(selected)]
        else:
            generator=torch.Generator(device='cpu').manual_seed(cp['config']['seed'])
            generator.set_state(cp['sampler_state']['generator_state'])
            expected_rows=torch.randperm(len(ex.features),generator=generator)[:len(selected)]
        self.assertTrue(torch.equal(selected,expected_rows))
        codes,_,_,mapping=diag.v2.catalog(model,ex)
        diag.v2.restore_rng(cp['rng_state'])
        grads,_=diag.content_gradients(model,ex,selected,'cpu')
        _,history_grads=history.measure(codes,mapping,backward=True)
        rq=[grads['rq.vq_layers.'+str(i)+'.embedding.weight'] for i in range(3)]
        mixed,_=diag.v2.normalized_mix(rq,history_grads,.1,1e-12)
        expected=diag.adam_proposal(model,cp,grads,mixed)
        self.helper.train(self.helper.args(name='replay',steps=4,resume=self.helper.root/'early/last.pth'))
        actual=torch.load(self.helper.root/'replay/last.pth',weights_only=False)['state_dict']
        self.assertTrue(all(torch.equal(v,actual[n]) for n,v in expected.items()))
        original=torch.load(self.helper.root/'early/last.pth',weights_only=False)
        self.assertEqual(cp['sampler_state']['cursor'],original['sampler_state']['cursor'])
        for n,state in cp['optimizer']['state'].items():
            for key,value in state.items():self.assertTrue(torch.equal(value,original['optimizer']['state'][n][key]))
        plain=diag.adam_proposal(model,cp,grads)
        self.assertTrue(all(torch.equal(v,expected[n]) for n,v in plain.items() if n.startswith(('encoder.','decoder.'))))

    def test_two_points_full_anchor_evaluation_no_holdout_no_checkpoints_and_source_hashes(self):
        before={str(p):diag.v2.sha256(p) for p in self.helper.root.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):result=diag.run(self.args())
        self.assertEqual([p['checkpoint_step'] for p in result['points']],[3,12])
        self.assertEqual(result['training_steps_committed'],0)
        self.assertFalse(result['holdout_used']);self.assertFalse(result['checkpoint_writes'])
        for point in result['points']:
            self.assertEqual(point['full_anchor_train_before']['queries'],12)
            rows={r['name']:r for r in point['candidates']}
            self.assertIn('history_direct_descent',rows)
            self.assertIn('history_direct_reverse',rows)
            self.assertEqual(len(rows),6)
            for row in rows.values():
                if row['hard'] is not None:self.assertEqual(row['hard']['queries'],12)
            self.assertAlmostEqual(point['joint_minus_content_hard_positive_damage'],
                rows['joint_adam']['hard']['positive_damage']-rows['content_adam']['hard']['positive_damage'])
            self.assertEqual(rows['joint_encoder_only']['relative_parameter_update']['codebooks'],0)
            self.assertEqual(rows['joint_codebooks_only']['relative_parameter_update']['encoder'],0)
            self.assertLess(rows['history_direct_descent']['proxy_linear_history_change_codebooks_only'],0)
            self.assertGreater(rows['history_direct_reverse']['proxy_linear_history_change_codebooks_only'],0)
            self.assertTrue(point['model_restored']);self.assertTrue(point['teacher_unchanged'])
        self.assertFalse(any((self.root/'diagnostic').rglob('*.pth')))
        self.assertTrue(all(diag.v2.sha256(Path(p))==value for p,value in before.items()))

    def test_sampler_rollover_reproducible_and_source_state_not_consumed(self):
        cp=torch.load(self.helper.root/'early/last.pth',weights_only=False)
        count=len(cp['sampler_state']['order'])
        cp['sampler_state']['cursor']=count
        original=cp['sampler_state']['generator_state'].clone()
        a=diag.next_content_batch(cp,count);b=diag.next_content_batch(cp,count)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(cp['sampler_state']['cursor'],count)
        self.assertTrue(torch.equal(cp['sampler_state']['generator_state'],original))

    def test_invalid_step_and_overwrite_fail_before_output(self):
        args=self.args();args.direct_relative_step=float('nan')
        with self.assertRaises(ValueError):diag.run(args)
        self.assertFalse(Path(args.output).exists())
        self.root.joinpath('exists').mkdir()
        with self.assertRaises(FileExistsError):diag.run(self.args(output='exists'))


if __name__=='__main__':unittest.main(verbosity=2)
