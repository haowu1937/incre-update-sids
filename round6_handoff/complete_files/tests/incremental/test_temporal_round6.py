"""Content contracts with a deterministic fake encoder; no models downloaded."""
import argparse
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from genrec.temporal import build_temporal_protocol
from genrec.temporal_content import prepare_content, encode_content, content_text, file_sha256, resolve_encoder
from prepare_dataset import encode_sent_emb


class FakeEncoder:
    calls = []
    bad = False
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        FakeEncoder.calls.append(self)
    def to(self, device):
        self.device = device
        return self
    def eval(self):
        return self
    def encode(self, sentences, **kwargs):
        self.sentences = sentences
        self.encode_kwargs = kwargs
        rows = [[1.0, float(len(s)), float(hashlib.sha256(s.encode()).digest()[0]+1)]
                for s in sentences]
        out = np.array(rows, dtype=np.float64)
        if self.bad:
            out[0, 0] = np.nan
        return out


def fake_resolver(model, revision, offline):
    return model, 'a'*40, {'kind': 'synthetic_test_only', 'commit': 'a'*40}


class Checks(unittest.TestCase):
    def setUp(self):
        FakeEncoder.calls.clear()
        FakeEncoder.bad = False
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        reviews = [('u', f'i{i%4}', i, 4.0) for i in range(1, 101)]
        reviews[0] = ('u', 'old', 1, 4.0)
        reviews[65] = ('u', 'new', 66, 4.0)
        reviews[78] = ('u', 'valonly', 79, 4.0)
        reviews[90] = ('u', 'future', 91, 4.0)
        protocol = build_temporal_protocol(reviews, max_history=5)
        self.protocol = protocol
        self.data = self.root / 'data'
        self.data.mkdir()
        for name, value in (('registries.json', protocol['registries']),
                            ('protocol_manifest.json', protocol['manifest'])):
            (self.data/name).write_text(json.dumps(value))
        for stage, rows in protocol['examples'].items():
            (self.data/(stage+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (self.data/'summary.json').write_text(json.dumps(dict(stage='E0_data_preparation',
            execution_status='completed', protocol_sha256=protocol['manifest']['protocol_sha256'],
            reviews_file_sha256='synthetic', artifact_files_sha256={p.name:file_sha256(p) for p in self.data.iterdir()})))
        self.meta = self.root/'meta.gz'
        self.records = [dict(asin=raw, title=f'<b>Product {raw}</b>', brand='A &amp; B',
                             categories=[['Beauty', 'Skin']], description=['Nice', 'Useful'],
                             salesRank={'Beauty': 1}, related={'also_bought': ['future']}, overall=5)
                        for raw in ('i0','i1','i2','i3','old','new','valonly','future')]
        self.write_meta()

    def write_meta(self):
        with gzip.open(self.meta, 'wt', encoding='utf-8') as handle:
            for r in self.records:
                handle.write(json.dumps(r)+'\n')

    def metadata(self, name='content', policy='error'):
        return prepare_content(self.meta, self.data, self.root/name, policy)

    def encode(self, name='features'):
        return encode_content(self.root/'content', self.root/name, expected_dim=3,
                              model_factory=FakeEncoder, encoder_resolver=fake_resolver)

    def test_whitelist_and_html_cleaning(self):
        text=content_text(self.records[0])
        self.assertIn('Brand: A & B', text)
        self.assertNotIn('<b>', text)
        self.assertNotIn('future', text)
        other=dict(self.records[0], salesRank=999, related='changed', reviewText='LEAK')
        self.assertEqual(text,content_text(other))

    def test_stage_catalogs_exclude_future_and_preserve_ids(self):
        result=self.metadata()
        for stage in ('base_train','update_train','update'):
            rows=[json.loads(s) for s in (self.root/'content'/stage/'content.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),result['stages'][stage]['items'])
            self.assertEqual([r['raw_item_id'] for r in rows],self.protocol['registries'][stage]['id2item'][1:])
            self.assertNotIn('future',[r['raw_item_id'] for r in rows])
        base=(self.root/'content/base_train/content.jsonl').read_text()
        self.assertNotIn('valonly',base)
        self.assertNotIn('new',base)

    def test_missing_error_and_explicit_placeholder(self):
        self.records=[r for r in self.records if r['asin']!='old']; self.write_meta()
        with self.assertRaisesRegex(ValueError,'Missing metadata'):
            self.metadata()
        self.assertFalse((self.root/'content').exists())
        result=self.metadata(policy='placeholder')
        self.assertEqual(result['stages']['base_train']['missing_records'],1)
        self.assertIn('Product information unavailable.',(self.root/'content/base_train/content.jsonl').read_text())

    def test_conflicting_duplicate_allowed_fields_rejected(self):
        self.records.append(dict(self.records[0],title='Different')); self.write_meta()
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            self.metadata()

    def test_duplicate_behavioural_fields_do_not_change_content(self):
        self.records.append(dict(self.records[0],salesRank=9999)); self.write_meta()
        result=self.metadata()
        self.assertEqual(result['duplicate_eligible_metadata_records'],1)

    def test_frozen_stage_prefixes_and_float32(self):
        self.metadata(); result=self.encode()
        all_values=np.load(self.root/'features/update/embeddings.npy')
        for stage in ('base_train','update_train'):
            values=np.load(self.root/'features'/stage/'embeddings.npy')
            self.assertEqual(values.dtype,np.float32)
            np.testing.assert_array_equal(values,all_values[:len(values)])
        self.assertEqual(len(FakeEncoder.calls),1)
        self.assertEqual(FakeEncoder.calls[0].kwargs['revision'],'a'*40)
        self.assertEqual(FakeEncoder.calls[0].max_seq_length,256)
        self.assertEqual(result['training_status'],'not_run')
        self.assertFalse(result['graph_used'])

    def test_feature_rows_raw_id_round_trip_and_hashes(self):
        self.metadata(); result=self.encode()
        for name,sha in result['artifact_files_sha256'].items():
            self.assertEqual(file_sha256(self.root/'features'/name),sha)
        rows=json.loads((self.root/'features/update/feature_rows.json').read_text())
        for i,r in enumerate(rows):
            self.assertEqual((r['feature_row'],r['gensid_global_item_id'],r['gr_item_id']),(i,i,i+1))

    def test_modified_metadata_artifact_rejected(self):
        self.metadata()
        with (self.root/'content/update/content.jsonl').open('a') as handle:handle.write('{}\n')
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.encode()
        self.assertFalse((self.root/'features').exists())

    def test_nonfinite_encoder_output_is_not_published(self):
        self.metadata(); FakeEncoder.bad=True
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            self.encode()
        self.assertFalse((self.root/'features').exists())

    def test_no_overwrite(self):
        self.metadata()
        with self.assertRaises(FileExistsError):self.metadata()
        self.encode()
        with self.assertRaises(FileExistsError):self.encode()

    def test_legacy_encoder_call_without_revision(self):
        ds=SimpleNamespace(n_items=2,id_mapping={'id2item':['[PAD]','x']},item2meta={'x':'Hello'})
        values=encode_sent_emb(ds,str(self.root/'raw'),dict(metadata='sentence',
            sent_emb_model='fake',device='cpu',sent_emb_batch_size=1),model_factory=FakeEncoder)
        self.assertEqual(values.shape,(1,3))
        self.assertEqual(FakeEncoder.calls[-1].kwargs,{})
        self.assertEqual(values.dtype,np.float32)

    def test_offline_revision_and_local_encoder_identity(self):
        with self.assertRaisesRegex(ValueError,'40-character'):
            resolve_encoder('not/a/local/path',None,True)
        model,rev,identity=resolve_encoder('test/model','b'*40,True)
        self.assertEqual(rev,'b'*40)
        local=self.root/'encoder';local.mkdir();(local/'weights').write_bytes(b'one')
        first=resolve_encoder(str(local),None,True)[2]
        (local/'weights').write_bytes(b'two')
        second=resolve_encoder(str(local),None,True)[2]
        self.assertNotEqual(first['sha256'],second['sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (output/'checks.json').write_text(json.dumps(dict(round='round6',inputs='synthetic_only_fake_encoder',
        tests_run=result.testsRun,execution_status='passed' if result.wasSuccessful() else 'failed',
        real_encoder_status='not_run',E0_full_status='not_run',E1_status='not_run'),indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
