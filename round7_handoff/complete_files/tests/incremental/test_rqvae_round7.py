"""Synthetic-only CPU checks; no claim of completed Beauty experiments."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from gensid.incremental_content import (build_model, collision_stats, evaluate,
    load_base_features, load_checkpoint, parser, run, sha256, write_json)

torch.set_num_threads(1)


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.features = self.root / 'features'
        stage = self.features / 'base_train'
        stage.mkdir(parents=True)
        # Structured but nontrivial features make reconstruction improvements observable.
        rng = np.random.default_rng(19)
        values = rng.normal(0.25, 0.1, (33, 8)).astype(np.float32)
        np.save(stage / 'embeddings.npy', values)
        raw = ['[PAD]'] + ['item_' + str(i) for i in range(len(values))]
        write_json(stage / 'registry.json', dict(id2item=raw, item2id={x:i for i,x in enumerate(raw)}))
        write_json(stage / 'feature_rows.json', [dict(raw_item_id=x, gr_item_id=i+1,
            gensid_global_item_id=i, feature_row=i) for i,x in enumerate(raw[1:])])
        self.summary = dict(stage='E0_fixed_content_embeddings', execution_status='completed',
            graph_used=False, protocol_sha256='synthetic_protocol_only',
            stages=dict(base_train=dict(shape=[33,8])), artifact_files_sha256={
                'base_train/'+name: sha256(stage/name) for name in
                ('embeddings.npy','feature_rows.json','registry.json')})
        write_json(self.features / 'summary.json', self.summary)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, name, steps=8, resume=None):
        args = parser().parse_args(['--features-dir', str(self.features), '--output', str(self.root/name),
            '--device','cpu','--max-steps',str(steps),'--batch-size','8','--lr','0.01',
            '--num-emb-list','4','4','4','--e-dim','4','--layers','16','8',
            '--init-samples','33','--kmeans-iters','5','--eval-batch-size','8',
            '--save-every','4','--log-every','100'])
        args.resume = str(resume) if resume else None
        return args

    def execute(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(args)

    def test_no_graph_import_and_base_only(self):
        values, rows, _ = load_base_features(self.features)
        self.assertEqual(values.shape, (33,8))
        self.assertNotIn('gensid.models.graph_encoder', sys.modules)
        self.assertFalse((self.features/'update_train').exists())

    def test_bad_input_fails_before_output(self):
        with (self.features/'base_train/embeddings.npy').open('ab') as handle:
            handle.write(b'changed')
        args = self.args('invalid')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.execute(args)
        self.assertFalse(Path(args.output).exists())

    def test_invalid_id_even_with_updated_hash(self):
        path = self.features/'base_train/feature_rows.json'
        rows = json.loads(path.read_text())
        rows[2]['gensid_global_item_id'] = 0
        write_json(path, rows)
        self.summary['artifact_files_sha256']['base_train/feature_rows.json'] = sha256(path)
        write_json(self.features/'summary.json',self.summary)
        with self.assertRaisesRegex(ValueError, 'ID contract'):
            load_base_features(self.features)

    def test_exact_steps_export_and_reload(self):
        result = self.execute(self.args('exact',7))
        # One epoch: 8+8+8+8+1, next two batches: 16.
        self.assertEqual(result['optimizer_steps'],7)
        self.assertEqual(result['processed_examples'],49)
        self.assertTrue(result['checkpoint_reload_assignments_equal'])
        self.assertEqual(result['E0_full_status'],'not_run')
        self.assertEqual(result['collision_suffix_status'],'not_created')
        rows = [json.loads(x) for x in (self.root/'exact/semantic_assignments.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),33)
        self.assertTrue(all(len(x['semantic_code']) == 3 for x in rows))
        self.assertEqual(rows[3]['raw_item_id'],'item_3')
        self.assertTrue(all(0 <= code < 4 for row in rows for code in row['semantic_code']))
        model = build_model(result['config'])
        model.load_state_dict(load_checkpoint(self.root/'exact/last.pth')['state_dict'])
        with patch.object(model.rq.vq_layers[0], 'init_emb', side_effect=AssertionError('reinitialized')):
            values,_,_ = load_base_features(self.features)
            model.train()
            model(values,None,use_sk=False)

    def test_resume_matches_uninterrupted(self):
        full = self.execute(self.args('full',12))
        self.execute(self.args('part',4))
        resumed = self.execute(self.args('resume',12,self.root/'part/last.pth'))
        a,b = (load_checkpoint(self.root/name/'last.pth') for name in ('full','resume'))
        for key,value in a['state_dict'].items():
            self.assertTrue(torch.equal(value,b['state_dict'][key]), key)
        for key,state in a['optimizer']['state'].items():
            for field,value in state.items():
                other = b['optimizer']['state'][key][field]
                self.assertTrue(torch.equal(value,other) if torch.is_tensor(value) else value == other)
        self.assertEqual(full['processed_examples'],resumed['processed_examples'])
        self.assertEqual((self.root/'full/semantic_assignments.jsonl').read_bytes(),
                         (self.root/'resume/semantic_assignments.jsonl').read_bytes())

    def test_reconstruction_improves_and_gradients_finite(self):
        result = self.execute(self.args('learn',100))
        self.assertLess(result['final_train_catalog_metrics']['reconstruction_loss'],
                        result['initial_train_catalog_metrics']['reconstruction_loss'])
        model = build_model(result['config'])
        model.load_state_dict(load_checkpoint(self.root/'learn/last.pth')['state_dict'])
        x,_,_ = load_base_features(self.features)
        model.train()
        out,q,idx,dense,codes,residuals = model(x,None,use_sk=False)
        loss,*_ = model.compute_loss(out,q,None,dense,dense_out_codebooks=codes,x_res_list=residuals,xs=x)
        loss.backward()
        for parameter in model.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for level in model.rq.vq_layers:
            self.assertIsNotNone(level.embedding.weight.grad)
            self.assertTrue(torch.isfinite(level.embedding.weight.grad).all())

    def test_legacy_checkpoint_initialization_flags(self):
        result = self.execute(self.args('legacy',4))
        saved = load_checkpoint(self.root/'legacy/last.pth')['state_dict']
        legacy = {k:v for k,v in saved.items() if not k.endswith('_initialized')}
        model = build_model(result['config'])
        model.load_state_dict(legacy,strict=True)
        self.assertTrue(all(layer.initted for layer in model.rq.vq_layers))

    def test_no_overwrite_resume_binding_and_budget(self):
        self.execute(self.args('part',4))
        with self.assertRaises(FileExistsError):
            self.execute(self.args('part',4))
        args = self.args('changed',8,self.root/'part/last.pth')
        args.batch_size = 9
        with self.assertRaisesRegex(ValueError, 'identical'):
            self.execute(args)
        self.assertFalse(Path(args.output).exists())
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.execute(self.args('short',2,self.root/'part/last.pth'))

    def test_collisions_are_retained_and_bounded(self):
        result = collision_stats(np.array([[0,1,2],[0,1,2],[1,1,2]]),[4,4,4])
        self.assertEqual(result['unique_semantic_codes'],2)
        self.assertEqual(result['maximum_collision_bucket'],2)
        self.assertEqual(result['items_in_collision_buckets'],2)
        with self.assertRaisesRegex(ValueError,'bounds'):
            collision_stats(np.array([[0,1,4]]),[4,4,4])


if __name__ == '__main__':
    result = unittest.main(verbosity=2,exit=False).result
    print(json.dumps(dict(round='round7', inputs='synthetic_only', tests_run=result.testsRun,
        execution_status='passed' if result.wasSuccessful() else 'failed',
        E0_full_status='not_run', E1_status='not_run'),indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
