"""Real tiny TIGER train/decode/resume with synthetic upstream checkpoint context.

The upstream E1/RQ-VAE loader is stubbed; it already has separate tests. This
suite exercises the new orchestration and actual repository TIGER/Trainer.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import torch
import genrec.future_evaluation as future
from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
from genrec.models.TIGER.model import TIGER
from genrec.temporal import build_temporal_protocol
from genrec.temporal_sids import allocate_suffixes, maps_from_ledger, read_json, sha256, write_json
from genrec.temporal_tiger import cpu_state, seed_all
from genrec.trainer import Trainer

torch.set_num_threads(1)


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        reviews = [('u'+str(i%5),'item'+str(i%7),i+1,5.) for i in range(200)]
        bounds = build_temporal_protocol(reviews,max_history=3)['manifest']['boundaries']
        for edge, raw in (('base_train_end','base_val_new'),('base_end','update_new'),
                          ('update_train_end','update_val_new'),('update_end','future_only')):
            i = bounds[edge]
            user,_,ts,rating = reviews[i]
            reviews[i] = (user,raw,ts,rating)
        protocol = build_temporal_protocol(reviews,max_history=3)
        self.registries, self.manifest = protocol['registries'],protocol['manifest']
        directory = self.root/'protocol'
        directory.mkdir()
        write_json(directory/'registries.json',self.registries)
        write_json(directory/'protocol_manifest.json',self.manifest)
        for stage,rows in protocol['examples'].items():
            (directory/(stage+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        write_json(directory/'summary.json',dict(artifact_files_sha256={p.name:sha256(p) for p in directory.iterdir()}))
        self.maps = {}
        for arm in ('base','plain'):
            raw = self.registries['update']['id2item'][1:]
            codes = [[(i+int(arm=='plain'))%2,(i//2)%2,0] for i in range(len(raw))]
            n = len(self.registries['update_train']['id2item'])-1
            train = allocate_suffixes(raw[:n],codes[:n],[2]*3,32)
            deploy = allocate_suffixes(raw,codes,[2]*3,32,train)
            tokenizers = {}
            for key,ledger in (('train',train),('deployment',deploy)):
                _,mapping = maps_from_ledger(ledger)
                tokenizers[key] = FrozenSIDTokenizer(mapping,[2,2,2,32],3)
            self.maps[arm] = dict(train_ledger=train,deployment_ledger=deploy,tokenizers=tokenizers)
        config = dict(num_layers=1,num_decoder_layers=1,d_model=16,d_ff=32,num_heads=2,d_kv=8,
            dropout_rate=.1,activation_function='relu',feed_forward_proj='relu',seed=2026,
            lr=.003,weight_decay=.05,batch_size=4,warmup_steps=1,max_grad_norm=1.,
            topk=[1,3],num_beams=4,metrics=['recall','ndcg'],eval_batch_size=3,device='cpu')
        seed_all(2026)
        teacher = dict(config=config,state_dict=cpu_state(TIGER(config,None,self.maps['base']['tokenizers']['train'])))
        for name in ('e1','base_rqvae','plain_rqvae','tiger'):
            (self.root/name).mkdir()
        torch.save(teacher,self.root/'tiger/best.pth')
        for name in ('base_rqvae','plain_rqvae'):
            (self.root/name/'last.pth').write_bytes(name.encode())
        e1 = dict(protocol_dir=str(directory),update_rqvae_dir=str(self.root/'plain_rqvae'),
            tiger_dir=str(self.root/'tiger'),protocol_sha256=self.manifest['protocol_sha256'],
            base_rqvae_sha256=sha256(self.root/'base_rqvae/last.pth'),
            update_rqvae_sha256=sha256(self.root/'plain_rqvae/last.pth'),
            selected_tiger_sha256=sha256(self.root/'tiger/best.pth'))
        write_json(self.root/'e1/summary.json',e1)
        self.ex = SimpleNamespace(e1=e1,e1_dir=self.root/'e1',teacher=teacher,
            binding=dict(synthetic_fixture=True),plain=dict(base_rqvae_dir=str(self.root/'base_rqvae')),
            rows={stage:protocol['examples'][stage] for stage in ('base_train','base_val')})
        self.input_hashes = {p:sha256(p) for p in self.root.rglob('*') if p.is_file()}

    def tearDown(self):
        self.temp.cleanup()

    def args(self,name,resume=False):
        values = ['--e1-dir',str(self.root/'e1'),'--output',str(self.root/name),'--device','cpu',
            '--max-steps','2','--selection-steps','1','2','--topk','1','3','--num-beams','4',
            '--eval-batch-size','3','--probe-size','10','--warmup-steps','1','--save-every','1']
        if resume:
            values.append('--resume')
        return future.parser().parse_args(values)

    def run_pipeline(self,args):
        actual_loader = future.load_stage
        def checked_loader(ex,stage,seen=()):
            if stage == 'future_test':
                output = Path(args.output)
                self.assertTrue((output/'selection_locked.json').exists())
                self.assertTrue(all((output/arm/'summary.json').exists() for arm in ('base','plain')))
                self.assertEqual(read_json(output/'pair_checks.json')['execution_status'],'passed')
            return actual_loader(ex,stage,seen)
        with patch.object(future,'experiment',return_value=self.ex), \
             patch.object(future,'build_maps',return_value=(self.maps,self.manifest,self.registries)), \
             patch.object(future,'load_stage',side_effect=checked_loader), \
             contextlib.redirect_stdout(io.StringIO()):
            return future.run(args)

    def test_four_real_tiger_conditions_and_future_after_selection(self):
        report = self.run_pipeline(self.args('full'))
        self.assertFalse(report['future_used_for_selection'])
        self.assertTrue(report['pair_checks']['same_sampler_endpoint'])
        for arm in ('base','plain'):
            self.assertTrue(report['training'][arm]['exact_base_warm_start'])
            self.assertEqual(report['training'][arm]['update_optimizer_steps'],2)
            for state in ('frozen','finetuned'):
                result = report['future'][arm][state]
                self.assertEqual(result['all_queries'],self.manifest['windows']['future_test']['example_count'])
                self.assertEqual(result['unavailable_targets'],self.manifest['windows']['future_test']['unservable_targets'])
                self.assertEqual(result['invalid_predictions'],0)
                for key,value in result['conditional_on_servable'].items():
                    self.assertAlmostEqual(result['full_denominator'][key],value*result['coverage'])
                self.assertTrue(result['teacher_frozen_during_evaluation'])
        self.assertTrue(all(sha256(path)==value for path,value in self.input_hashes.items()))

    def test_exact_resume_with_dropout_and_no_future_before_interruption(self):
        self.run_pipeline(self.args('continuous'))
        original = Trainer.fit_steps
        def interrupted(trainer,arrays,sampler,generator,target_steps,make_batch,on_step):
            def callback(loss,norm):
                on_step(loss,norm)
                if sampler['optimizer_steps']==1:
                    raise RuntimeError('intentional interruption after checkpoint')
            return original(trainer,arrays,sampler,generator,target_steps,make_batch,callback)
        with patch.object(Trainer,'fit_steps',interrupted):
            with self.assertRaisesRegex(RuntimeError,'intentional interruption'):
                self.run_pipeline(self.args('resumed'))
        self.assertFalse((self.root/'resumed/selection_locked.json').exists())
        self.assertFalse((self.root/'resumed/base_frozen.json').exists())
        self.run_pipeline(self.args('resumed',resume=True))
        for arm in ('base','plain'):
            a,b = [torch.load(self.root/name/arm/'last.pth',weights_only=False) for name in ('continuous','resumed')]
            self.assertTrue(all(torch.equal(value,b['state_dict'][key]) for key,value in a['state_dict'].items()))
            self.assertTrue(torch.equal(a['generator'],b['generator']))
            self.assertEqual(a['validation_history'],b['validation_history'])

    def test_resume_binding_and_future_cache_tampering_rejected(self):
        self.run_pipeline(self.args('full'))
        changed = self.args('full',resume=True)
        changed.lr *= 2
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            self.run_pipeline(changed)
        path=self.root/'full/base_frozen.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            self.run_pipeline(self.args('full',resume=True))

    def test_future_interruption_reuses_locked_models_and_completed_predictions(self):
        original = future.future_metrics
        calls = 0
        def fail_second(*args,**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('interrupted Future evaluation')
            return original(*args,**kwargs)
        with patch.object(future,'future_metrics',side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError,'interrupted Future'):
                self.run_pipeline(self.args('partial_future'))
        output = self.root/'partial_future'
        retained = [output/'selection_locked.json',output/'base_frozen.json',
                    *(output/arm/name for arm in ('base','plain') for name in ('best.pth','last.pth'))]
        before = {path:sha256(path) for path in retained}
        report = self.run_pipeline(self.args('partial_future',resume=True))
        self.assertEqual(report['execution_status'],'completed')
        self.assertTrue(all(sha256(path)==value for path,value in before.items()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
