"""Synthetic-only correctness checks with the actual repository Base RQ-VAE."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from genrec.temporal import build_temporal_protocol
from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
from genrec.temporal_sids import (allocate_suffixes,code_to_tokens,export_base_sids,
    load_protocol,maps_from_ledger,read_json,sha256,write_json)
from gensid.incremental_content import parser as rq_parser,run as train_rq

torch.set_num_threads(1)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        root = Path(cls.fixture.name)
        reviews = [('u'+str(i%5),'item_'+str(i%9),i+1,5.0) for i in range(160)]
        boundaries = build_temporal_protocol(reviews,max_history=4)['manifest']['boundaries']
        start = boundaries['base_train_end']
        for i in range(start,start+5):
            u,_,ts,rating = reviews[i]
            reviews[i] = (u,'validation_only_'+str(i%2),ts,rating)
        u,_,ts,rating = reviews[100]
        reviews[100] = (u,'update_only',ts,rating)
        protocol = build_temporal_protocol(reviews,max_history=4)
        data = root/'protocol';data.mkdir()
        write_json(data/'protocol_manifest.json',protocol['manifest'])
        write_json(data/'registries.json',protocol['registries'])
        for stage in ('base_train','base_val'):
            with (data/(stage+'.jsonl')).open('w') as handle:
                for row in protocol['examples'][stage]:
                    handle.write(json.dumps(row)+'\n')
        report = dict(stage='E0_data_preparation',execution_status='completed',
            protocol_sha256=protocol['manifest']['protocol_sha256'],
            artifact_files_sha256={p.name:sha256(p) for p in data.iterdir()})
        write_json(data/'summary.json',report)
        features = root/'features';features.mkdir()
        global_rows = len(protocol['registries']['update_train']['id2item'])-1
        values = np.random.default_rng(31).normal(0.1,0.3,(global_rows,8)).astype(np.float32)
        stats = {}
        for stage in ('base_train','update_train'):
            directory = features/stage;directory.mkdir()
            registry = protocol['registries'][stage]
            n = len(registry['id2item'])-1
            np.save(directory/'embeddings.npy',values[:n])
            write_json(directory/'registry.json',registry)
            write_json(directory/'feature_rows.json',[dict(raw_item_id=raw,gr_item_id=i+1,
                gensid_global_item_id=i,feature_row=i) for i,raw in enumerate(registry['id2item'][1:])])
            stats[stage] = dict(shape=[n,8])
        report = dict(stage='E0_fixed_content_embeddings',execution_status='completed',graph_used=False,
            protocol_sha256=protocol['manifest']['protocol_sha256'],stages=stats,
            artifact_files_sha256={p.relative_to(features).as_posix():sha256(p)
                for p in features.rglob('*') if p.is_file()})
        write_json(features/'summary.json',report)
        args = rq_parser().parse_args(['--features-dir',str(features),'--output',str(root/'rqvae'),
            '--device','cpu','--max-steps','8','--batch-size','4','--e-dim','4',
            '--num-emb-list','2','2','2','--layers','16','8',
            '--init-samples','9','--kmeans-iters','5','--eval-batch-size','4'])
        with contextlib.redirect_stdout(io.StringIO()):
            train_rq(args)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ('protocol','features','rqvae'):
            shutil.copytree(Path(self.fixture.name)/name,self.root/name)

    def tearDown(self):
        self.tmp.cleanup()

    def export(self,name='output',suffix_capacity=16,expected=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return export_base_sids(self.root/'protocol',self.root/'features',self.root/'rqvae',
                self.root/name,device='cpu',suffix_capacity=suffix_capacity,
                expected_checkpoint_sha256=expected)

    def rehash(self,directory,name):
        path = self.root/directory
        summary = read_json(path/'summary.json')
        summary['artifact_files_sha256'][name] = sha256(path/name)
        write_json(path/'summary.json',summary)

    def test_zero_based_capacity_and_persistent_append(self):
        a = allocate_suffixes(['a','b'],[[0,0,0],[0,0,0]],[2,2,2],2)
        b = allocate_suffixes(['a','b','c'],[[0,0,0],[0,0,0],[1,1,1]],[2,2,2],2,a)
        self.assertEqual(a['entries'],b['entries'][:2])
        self.assertEqual([x['collision_suffix'] for x in a['entries']],[0,1])
        self.assertEqual(code_to_tokens([1,1,1,1],[2,2,2,2]),[2,4,6,8])
        with self.assertRaisesRegex(ValueError,'capacity exhausted'):
            allocate_suffixes(['a','b','c'],[[0,0,0]]*3,[2,2,2],2,a)

    def test_allocator_rejects_code_change_reorder_and_invalid_index(self):
        a = allocate_suffixes(['a','b'],[[0,0,0],[1,0,0]],[2,2,2],4)
        for raw,codes in ((['b','a'],[[1,0,0],[0,0,0]]),(['a','b'],[[1,0,0],[1,0,0]])):
            with self.assertRaisesRegex(ValueError,'changed'):
                allocate_suffixes(raw,codes,[2,2,2],4,a)
        with self.assertRaisesRegex(ValueError,'capacity'):
            allocate_suffixes(['a'],[[2,0,0]],[2,2,2],4)

    def test_frozen_tokenizer_mask_target_and_catalog_prefixes(self):
        ledger = allocate_suffixes(['a','b'],[[0,0,0],[0,0,0]],[2,2,2],4)
        _,mapping = maps_from_ledger(ledger)
        tokenizer = FrozenSIDTokenizer(mapping,[2,2,2,4],2)
        ids,mask,labels = tokenizer._tokenize_once({'item_seq':['outside','a','b']})
        self.assertEqual(ids[:4],[0]*4)
        self.assertEqual(mask[:4],[0]*4)
        self.assertEqual(mask[-1],1)
        self.assertEqual(ids[4:8],mapping['a'])
        self.assertEqual(labels,[*mapping['b'],tokenizer.eos_token])
        prefix = [0]
        for token in labels:
            self.assertIn(token,tokenizer.allowed_next_tokens(prefix))
            prefix.append(token)
        with self.assertRaisesRegex(ValueError,'outside'):
            tokenizer.allowed_next_tokens([0,2])
        with self.assertRaisesRegex(ValueError,'unavailable target'):
            tokenizer._tokenize_once({'item_seq':['a','outside']})

    def test_fully_masked_history_keeps_eos(self):
        t = FrozenSIDTokenizer({'a':[1,3,5,7]},[2,2,2,2],2)
        ids,mask,labels = t._tokenize_once({'item_seq':['outside','a']})
        self.assertEqual(len(mask),len(ids))
        self.assertEqual(len(mask),t.max_token_seq_len)
        self.assertEqual(sum(mask),1)
        self.assertEqual(ids[4],t.eos_token)
        self.assertEqual(mask[4],1)
        self.assertEqual(labels[-1],t.eos_token)

    def test_export_integrity_counts_and_no_training(self):
        source_checkpoint = sha256(self.root/'rqvae/last.pth')
        with patch('gensid.trainer.Trainer.fit_steps',side_effect=AssertionError('training during export')):
            result = self.export()
        self.assertEqual(sha256(self.root/'rqvae/last.pth'),source_checkpoint)
        self.assertTrue(result['original_base_assignments_equal'])
        self.assertFalse(result['update_or_future_item_assignments_created'])
        self.assertEqual(result['recommendation_training_status'],'not_run')
        self.assertEqual(result['E0_full_status'],'not_run')
        _,manifest,registries,rows = load_protocol(self.root/'protocol')
        stats = result['tiger_splits']['base_val']
        self.assertEqual(stats['all_examples'],len(rows['base_val']))
        self.assertEqual(stats['targets_outside_frozen_catalog'],5)
        self.assertEqual(stats['encoded_examples'],len(rows['base_val'])-5)
        contract = result['tokenizer_contract']
        self.assertEqual(contract['vocab_size'],sum([2,2,2,16])+2)
        self.assertEqual(contract['n_digit'],4)
        self.assertEqual(contract['n_user_tokens'],0)
        for relative,digest in result['artifact_files_sha256'].items():
            self.assertEqual(sha256(self.root/'output'/relative),digest)
        tensors = np.load(self.root/'output/tiger/base_val.npz',allow_pickle=False)
        self.assertEqual(tensors['input_ids'].shape[1],17)
        self.assertEqual(tensors['labels'].shape[1],5)
        self.assertTrue(np.all(tensors['labels']>0))
        self.assertTrue(np.any(tensors['attention_mask']==0))

    def test_catalog_gating_and_old_full_sid_preservation(self):
        self.export()
        train = read_json(self.root/'output/base_train/item2tokens.json')
        deploy = read_json(self.root/'output/base/item2tokens.json')
        self.assertTrue(all(deploy[raw]==tokens for raw,tokens in train.items()))
        self.assertFalse(any(raw.startswith('validation_only') for raw in train))
        self.assertTrue(any(raw.startswith('validation_only') for raw in deploy))
        self.assertNotIn('update_only',deploy)
        self.assertEqual(len(set(map(tuple,deploy.values()))),len(deploy))

    def test_deterministic_export_and_no_overwrite(self):
        self.export('first');self.export('second')
        for name in ('base_train/allocation_registry.json','base/item2tokens.json','tiger/base_val.npz'):
            self.assertEqual((self.root/'first'/name).read_bytes(),(self.root/'second'/name).read_bytes())
        with self.assertRaises(FileExistsError):
            self.export('first')

    def test_input_tamper_and_expected_checkpoint_fail_before_output(self):
        with self.assertRaisesRegex(ValueError,'explicitly selected'):
            self.export(expected='0'*64)
        self.assertFalse((self.root/'output').exists())
        with (self.root/'protocol/base_val.jsonl').open('a') as handle:
            handle.write('{}\n')
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.export()
        self.assertFalse((self.root/'output').exists())

    def test_availability_flags_and_same_time_fail_before_output(self):
        path = self.root/'protocol/base_val.jsonl'
        original = path.read_text()
        rows = [json.loads(x) for x in original.splitlines()]
        rows[0]['target_servable'] = not rows[0]['target_servable']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows));self.rehash('protocol','base_val.jsonl')
        with self.assertRaisesRegex(ValueError,'flags'):
            self.export()
        self.assertFalse((self.root/'output').exists())
        rows = [json.loads(x) for x in original.splitlines()]
        rows[0]['time_seq'][0] = rows[0]['target_time']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows));self.rehash('protocol','base_val.jsonl')
        with self.assertRaisesRegex(ValueError,'causal'):
            self.export()

    def test_shared_base_prefix_cannot_change(self):
        path = self.root/'features/update_train/embeddings.npy'
        x = np.load(path,allow_pickle=False);x[0,0]+=0.1;np.save(path,x)
        self.rehash('features','update_train/embeddings.npy')
        # Changed feature-summary identity already invalidates the RQ-VAE binding.
        with self.assertRaisesRegex(ValueError,'binding'):
            self.export()
        self.assertFalse((self.root/'output').exists())

    def test_canonical_protocol_identity_is_verified(self):
        path = self.root/'protocol/protocol_manifest.json'
        manifest = read_json(path);manifest['max_history']+=1;write_json(path,manifest)
        self.rehash('protocol','protocol_manifest.json')
        with self.assertRaisesRegex(ValueError,'Canonical'):
            load_protocol(self.root/'protocol')


if __name__ == '__main__':
    result = unittest.main(verbosity=2,exit=False).result
    print(json.dumps(dict(round='round8',inputs='synthetic_only',tests_run=result.testsRun,
        execution_status='passed' if result.wasSuccessful() else 'failed',
        E0_full_status='not_run',E1_status='not_run'),indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
