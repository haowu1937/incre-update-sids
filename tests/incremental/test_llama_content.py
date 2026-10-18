"""Synthetic temporal contracts; no network or pretrained weights required."""
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from genrec.temporal import build_temporal_protocol
from genrec.temporal_content import prepare_content, file_sha256
import prepare_incremental_llama as llama


class FakeEncoder:
    calls = []
    fail_at = None
    info = dict(kind='synthetic_test_encoder')

    def __init__(self, args):
        self.count = 0

    def encode(self, texts):
        self.count += 1
        if self.count == self.fail_at:
            raise RuntimeError('Simulated interruption')
        self.calls.extend(texts)
        return np.asarray([[len(t), hashlib.sha256(t.encode()).digest()[0]+1, 1]
                           for t in texts], dtype=np.float32)

    def memory(self):
        return dict(peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None)


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        FakeEncoder.calls = []
        FakeEncoder.fail_at = None
        reviews = [('u', f'i{i%4}', i, 4.0) for i in range(1, 101)]
        for index, item in ((0, 'old'), (65, 'new'), (78, 'valonly'), (90, 'future')):
            reviews[index] = ('u', item, index+1, 4.0)
        self.raw = self.root / 'reviews.gz'
        with gzip.open(self.raw, 'wt') as handle:
            for u, item, t, rating in reviews:
                handle.write(json.dumps(dict(reviewerID=u, asin=item,
                    unixReviewTime=t, overall=rating)) + '\n')
        protocol = build_temporal_protocol(reviews, max_history=5)
        self.data = self.root / 'protocol'
        self.data.mkdir()
        for name, value in (('registries.json', protocol['registries']),
                            ('protocol_manifest.json', protocol['manifest'])):
            llama.write_json(self.data/name, value)
        for stage, rows in protocol['examples'].items():
            (self.data/(stage+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        llama.write_json(self.data/'summary.json', dict(stage='E0_data_preparation',
            execution_status='completed', protocol_sha256=protocol['manifest']['protocol_sha256'],
            reviews_file_sha256=file_sha256(self.raw),
            artifact_files_sha256={p.name: file_sha256(p) for p in self.data.iterdir()}))
        self.meta = self.root/'metadata.gz'
        with gzip.open(self.meta, 'wt') as handle:
            for raw in ('i0', 'i1', 'i2', 'i3', 'old', 'new', 'valonly', 'future'):
                handle.write(json.dumps(dict(asin=raw, title='Product '+raw,
                    categories=[['Beauty', 'Skin'], ['Ignored']], description=['Nice', 'Useful'],
                    brand='DO NOT ENCODE', related={'also_bought': ['future']}, overall=5))+'\n')
        self.content = self.root/'content'
        prepare_content(self.meta, self.data, self.content)
        self.model = self.root/'model'
        self.model.mkdir()
        llama.write_json(self.model/'config.json', dict(model_type='llama', hidden_size=3))
        llama.write_json(self.model/'model.safetensors.index.json',
                         dict(weight_map={'x': 'model-00001-of-00001.safetensors'}))
        (self.model/'model-00001-of-00001.safetensors').write_bytes(b'synthetic only')
        (self.model/'tokenizer.model').write_bytes(b'synthetic only')
        self.args = llama.parser().parse_args([
            '--protocol-dir', str(self.data), '--content-dir', str(self.content),
            '--reviews', str(self.raw), '--model-dir', str(self.model),
            '--output', str(self.root/'features'), '--expected-dim', '3',
            '--device', 'cpu', '--dtype', 'float32'])

    def run_encoding(self):
        return llama.run(self.args, encoder_factory=FakeEncoder)

    def test_pooling_uses_fixed_denominator_and_ignores_padding(self):
        hidden = np.asarray([[[2, 4], [4, 8], [999, 999], [999, 999]]], np.float32)
        mask = np.asarray([[1, 1, 0, 0]])
        np.testing.assert_array_equal(llama.fixed_length_pool(hidden, mask), [[1.5, 3]])
        with self.assertRaises(ValueError):
            llama.fixed_length_pool(hidden, mask[:, :2])

    def test_text_matches_notebook_and_excludes_behavior_and_brand(self):
        record = dict(title='A', categories=[['Beauty', 'Skin'], ['Ignore']],
                      description=['Nice', 'Useful'], brand='SECRET', reviewText='LEAK')
        self.assertEqual(llama.mitigating_text(record),
            "Title: A. Categories: Beauty, Skin. Description: ['Nice', 'Useful'].")
        self.assertEqual(llama.mitigating_text({}),
                         'Title: None. Categories: None. Description: None.')

    def test_exports_match_protocol_shared_rows_and_hashes(self):
        result = self.run_encoding()
        out = Path(self.args.output)
        shared = np.load(out/'update/embeddings.npy')
        for stage in llama.STAGES:
            rows = llama._json(out/stage/'feature_rows.json')
            np.testing.assert_array_equal(np.load(out/stage/'embeddings.npy'), shared[:len(rows)])
            self.assertEqual(llama._json(out/stage/'registry.json'),
                             llama._json(self.data/'registries.json')[stage])
        for name, digest in result['artifact_files_sha256'].items():
            self.assertEqual(file_sha256(out/name), digest, name)
        self.assertEqual(len(FakeEncoder.calls), len(shared))
        self.assertFalse(any('future' in text or 'DO NOT ENCODE' in text for text in FakeEncoder.calls))
        self.assertTrue(result['shared_features_equal'])
        self.assertFalse(result['future_only_items_encoded'])
        self.assertEqual(result['training_status'], 'not_run')

    def test_resume_continues_after_last_flushed_batch(self):
        FakeEncoder.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.run_encoding()
        pending = Path(self.args.output+'.incomplete')
        self.assertEqual(llama._json(pending/'progress.json')['completed_rows'], 1)
        FakeEncoder.fail_at = None
        self.args.resume = True
        result = self.run_encoding()
        self.assertEqual(result['encoded_items_this_invocation'], len(FakeEncoder.calls)-1)
        resumed = np.load(Path(self.args.output)/'shared_embeddings.npy')
        self.args.resume = False
        self.args.output = str(self.root/'fresh')
        self.run_encoding()
        np.testing.assert_array_equal(resumed, np.load(self.root/'fresh/shared_embeddings.npy'))

    def test_resume_rejects_changed_config(self):
        FakeEncoder.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.run_encoding()
        self.args.resume = True
        self.args.max_length = 256
        with self.assertRaisesRegex(ValueError, 'Resume inputs'):
            self.run_encoding()

    def test_resume_rejects_changed_weights(self):
        FakeEncoder.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.run_encoding()
        (self.model/'model-00001-of-00001.safetensors').write_bytes(b'changed')
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, 'Resume inputs'):
            self.run_encoding()

    def test_no_overwrite(self):
        self.run_encoding()
        old = file_sha256(Path(self.args.output)/'summary.json')
        with self.assertRaises(FileExistsError):
            self.run_encoding()
        self.assertEqual(old, file_sha256(Path(self.args.output)/'summary.json'))

    def test_wrong_reviews_fail_before_output(self):
        self.raw.write_bytes(b'wrong data')
        with self.assertRaisesRegex(ValueError, 'Raw reviews'):
            self.run_encoding()
        self.assertFalse(Path(self.args.output+'.incomplete').exists())

    def test_tampered_registry_fails_before_output(self):
        (self.content/'base_train/registry.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_encoding()
        self.assertFalse(Path(self.args.output+'.incomplete').exists())

    def test_missing_model_shard_fails_before_output(self):
        (self.model/'model-00001-of-00001.safetensors').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_encoding()
        self.assertFalse(Path(self.args.output+'.incomplete').exists())


class RealBackendCheck(unittest.TestCase):
    def test_tiny_local_llama_cpu_no_download(self):
        try:
            import torch
            from transformers import LlamaConfig, LlamaModel
        except ImportError:
            self.skipTest('torch/transformers not installed in this environment')

        class TinyTokenizer:
            pad_token_id, eos_token_id, bos_token_id = 0, 2, 1
            add_bos_token, add_eos_token = True, False

            def __call__(self, texts, **kwargs):
                return dict(input_ids=torch.tensor([[1, 3, 0, 0], [1, 4, 5, 0]]),
                            attention_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]))

        with tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(2026)
            config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=8)
            model = LlamaModel(config).eval()
            model.save_pretrained(temp, safe_serialization=True)
            args = type('Args', (), dict(model_dir=temp, device='cpu',
                dtype='float32', attention='eager', max_length=4))()
            with patch('transformers.LlamaTokenizer.from_pretrained', return_value=TinyTokenizer()):
                encoder = llama.LocalLlamaEncoder(args)
            actual = encoder.encode(['A', 'B'])
            inputs = TinyTokenizer()(['A', 'B'])
            with torch.inference_mode():
                hidden = model(**inputs, use_cache=False).last_hidden_state
                expected = (hidden * inputs['attention_mask'][..., None]).sum(1) / 4
            np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)
            self.assertFalse(encoder.model.training)
            self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
