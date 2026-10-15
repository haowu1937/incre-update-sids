"""CPU synthetic checks using real dataset/tokenizer modules; no model download."""
import argparse
import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genrec.dataset import AbstractDataset
from genrec.tokenizer import EncoderDecoderTokenizerMixin

class FixtureTokenizer(EncoderDecoderTokenizerMixin):
    n_digit = 2
    eos_token = 99
    padding_token = 0
    item2tokens = {"a": (1, 11), "b": (2, 12), "c": (3, 13), "d": (4, 14)}
    def __init__(self, explicit=True, users=0):
        self.config = {"max_item_seq_len": 2, "explicit_causal_examples": explicit}
        self.n_user_tokens = users
    def _token_single_user(self, user):
        return 80

class Checks(unittest.TestCase):
    def batch(self):
        return {"user": ["u"], "item_seq": [["a", "b", "c", "d"]]}
    def test_explicit_one_target(self):
        out=FixtureTokenizer().tokenize_function(self.batch(), "train")
        self.assertEqual(out['input_ids'], [[2,12,3,13,99]])
        self.assertEqual(out['labels'], [[4,14,99]])
    def test_legacy_prefixes(self):
        out=FixtureTokenizer(False).tokenize_function(self.batch(), "train")
        self.assertEqual(len(out['labels']),3)
        self.assertEqual(out['labels'],[[2,12,99],[3,13,99],[4,14,99]])
    def test_masks_with_and_without_user(self):
        for users in (0, 4):
            tok=FixtureTokenizer(users=users)
            ids,mask,_=tok._tokenize_once({'user':'u','item_seq':['a','b']})
            self.assertEqual(len(ids),tok.max_token_seq_len)
            self.assertEqual(len(mask),len(ids))
            self.assertEqual(mask,[int(x!=0) for x in ids])
    def test_eval_same_single_example(self):
        tok=FixtureTokenizer()
        self.assertEqual(tok.tokenize_function(self.batch(),'train'),tok.tokenize_function(self.batch(),'val'))
    def test_empty_history_rejected(self):
        with self.assertRaises(ValueError):
            FixtureTokenizer()._tokenize_once({'user':'u','item_seq':['a']})
    def reviews(self):
        return [('u', 'abcd'[i%4], i, None) for i in range(1,101)]
    def dataset(self):
        return AbstractDataset({'max_item_seq_len':2,'explicit_causal_examples':True,'n_user_tokens':0})
    def test_stage_adapter_and_metadata(self):
        ds=self.dataset(); splits=ds.prepare_temporal_stage(self.reviews(),'base')
        self.assertEqual(set(splits),{'train','val'})
        self.assertEqual(set(splits['train'].column_names),{'user','item_seq'})
        for split,data in splits.items():
            self.assertEqual(len(data),len(ds.temporal_metadata[split]))
            batch=data[:]; out=FixtureTokenizer().tokenize_function(batch,split)
            self.assertEqual(len(out['labels']),len(data))
        self.assertIs(ds.split(),splits)
    def test_update_future_and_identity(self):
        ds=self.dataset(); ds.prepare_temporal_stage(self.reviews(),'base'); ids=ds.item2id.copy()
        out=ds.prepare_temporal_stage(self.reviews(),'update')
        self.assertEqual(set(out),{'train','val','test'})
        self.assertEqual(ds.item2id,ids)
        self.assertTrue(all(x['target_servable'] for x in ds.temporal_metadata['test']))
    def test_adapter_guards(self):
        ds=self.dataset()
        with self.assertRaises(ValueError): ds.prepare_temporal_stage(self.reviews(),'future')
        ds.config['explicit_causal_examples']=False
        with self.assertRaises(ValueError): ds.prepare_temporal_stage(self.reviews())
        ds.config.update(explicit_causal_examples=True,n_user_tokens=8)
        with self.assertRaises(ValueError): ds.prepare_temporal_stage(self.reviews())

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    dest=Path(args.output); dest.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    payload={'round':'round2','inputs':'synthetic_only','execution_status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,'python':sys.version,'source_files_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('genrec/dataset.py','genrec/tokenizer.py','genrec/temporal.py')},'E0_full_status':'not_run','E1_status':'not_run'}
    (dest/'checks.json').write_text(json.dumps(payload,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
