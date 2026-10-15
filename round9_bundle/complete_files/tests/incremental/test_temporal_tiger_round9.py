"""Synthetic checks with the actual repository RQ-VAE and TIGER/T5."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
from genrec.models.TIGER.model import TIGER
from genrec.temporal_sids import export_base_sids, read_json, sha256, write_json
from genrec.temporal_tiger import (MODEL_DEFAULTS, catalog_prefix, evaluate, load_sids,
    make_batch, parser, probe_rows, rng_state, run, seed_all)
from genrec.trainer import Trainer

torch.set_num_threads(1)
TINY = dict(num_layers=1, num_decoder_layers=1, d_model=16, d_ff=32,
            num_heads=2, d_kv=8, dropout_rate=0.2)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('round8_fixture',
            ROOT/'tests/incremental/test_temporal_sids_round8.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.previous = module.Checks
        cls.previous.setUpClass()
        cls.fixture = tempfile.TemporaryDirectory()
        cls.sid_dir = Path(cls.fixture.name)/'sids'
        root = Path(cls.previous.fixture.name)
        with contextlib.redirect_stdout(io.StringIO()):
            export_base_sids(root/'protocol', root/'features', root/'rqvae', cls.sid_dir,
                             device='cpu', suffix_capacity=16)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()
        cls.previous.tearDownClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sid = self.root/'sids'
        shutil.copytree(self.sid_dir, self.sid)
        write_json(self.root/'tiny.json', TINY)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, name, steps=4, resume=None, smoke=True):
        values = ['--sid-dir', str(self.sid), '--output', str(self.root/name),
            '--device', 'cpu', '--batch-size', '8', '--eval-batch-size', '3',
            '--max-steps', str(steps), '--warmup-steps', '2', '--probe-size', '10',
            '--eval-every', '100', '--save-every', '2', '--log-every', '2',
            '--num-beams', '4', '--topk', '1', '3', '--model-config', str(self.root/'tiny.json')]
        if resume:
            values += ['--resume-from', str(resume)]
        if smoke:
            values += ['--smoke']
        return parser().parse_args(values)

    def train(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.args(*args, **kwargs))

    def trainer(self, tokenizer):
        config = dict(MODEL_DEFAULTS, **TINY, lr=0.003, weight_decay=0.05,
                      batch_size=8, warmup_steps=2, max_grad_norm=1.0,
                      num_beams=4, topk=[1, 3], metrics=['recall', 'ndcg'])
        return Trainer.for_temporal(config, TIGER(config, None, tokenizer), tokenizer)

    def rehash(self, name):
        summary = read_json(self.sid/'summary.json')
        summary['artifact_files_sha256'][name] = sha256(self.sid/name)
        write_json(self.sid/'summary.json', summary)

    def test_verified_row_targets_masks_and_cold_denominator(self):
        summary, tokenizer, arrays, audits = load_sids(self.sid)
        self.assertEqual(len(arrays['base_train']['labels']), 81)
        self.assertEqual(len(audits['base_val']), 10)
        self.assertEqual(sum(not r['target_servable'] for r in audits['base_val']), 5)
        batch = make_batch(arrays['base_val'], np.arange(5), 'cpu')
        self.assertEqual(batch['input_ids'].dtype, torch.long)
        self.assertTrue(torch.equal(batch['attention_mask'].bool(), batch['input_ids'] != 0))
        self.assertEqual(tokenizer.n_user_tokens, 0)

    def test_actual_t5_training_resume_is_exact_with_dropout(self):
        self.train('continuous', 8)
        self.train('partial', 3)
        resumed = self.train('resumed', 8, self.root/'partial/last.pth')
        a = torch.load(self.root/'continuous/last.pth', weights_only=False)
        b = torch.load(self.root/'resumed/last.pth', weights_only=False)
        self.assertEqual(a['sampler']['processed_examples'], b['sampler']['processed_examples'])
        self.assertEqual(a['sampler']['cursor'], b['sampler']['cursor'])
        self.assertTrue(torch.equal(a['sampler']['order'], b['sampler']['order']))
        self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))
        self.assertEqual(a['scheduler'], b['scheduler'])
        for key in a['state_dict']:
            self.assertTrue(torch.equal(a['state_dict'][key], b['state_dict'][key]), key)
        for key, state in a['optimizer']['state'].items():
            for field, value in state.items():
                self.assertTrue(torch.equal(value, b['optimizer']['state'][key][field]))
        self.assertEqual(resumed['steps_this_run'], 5)

    def test_real_constrained_beams_remain_legal_through_single_branch_levels(self):
        # Only one first/second/third token: inactive beams must never escape the catalog.
        tokenizer = FrozenSIDTokenizer({str(i): [1, 3, 5, 7+i] for i in range(8)}, [2, 2, 2, 8], 2)
        trainer = self.trainer(tokenizer)
        ids, mask, label = tokenizer._tokenize_once(dict(item_seq=['0', '1']))
        batch = dict(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask]),
                     labels=torch.tensor([label]))
        trainer.model.eval()
        preds = trainer.model.generate(batch, 3, True, catalog_prefix(tokenizer))
        self.assertTrue(torch.isfinite(preds['scores']).all())
        values = [tuple(v) for v in preds['preds'][0].tolist()]
        self.assertEqual(len(set(values)), 3)
        self.assertTrue(set(values).issubset(set(tokenizer.item2tokens.values())))
        self.assertEqual(preds['sequences'].shape, (1, 3, 6))
        self.assertTrue(torch.all(preds['sequences'][:, :, -1] == tokenizer.eos_token))

    def test_production_architecture_vocab_and_masked_history_positions(self):
        tokenizer = FrozenSIDTokenizer({str(i): [1, 257, 513, 769+i] for i in range(26)},
                                      [256]*4, 50)
        config = dict(MODEL_DEFAULTS, num_beams=20)
        model = TIGER(config, None, tokenizer)
        ids, mask, label = tokenizer._tokenize_once(dict(item_seq=['0', 'outside', '1', '2']))
        arrays = dict(input_ids=np.asarray([ids], dtype=np.int32),
                      attention_mask=np.asarray([mask], dtype=np.uint8),
                      labels=np.asarray([label], dtype=np.int32))
        trimmed = make_batch(arrays, np.arange(1), 'cpu')
        self.assertEqual(trimmed['input_ids'].shape, (1, 13))
        self.assertTrue(torch.all(trimmed['input_ids'][:, 4:8] == 0))
        self.assertTrue(torch.all(trimmed['attention_mask'][:, 4:8] == 0))
        self.assertEqual(model.t5.config.vocab_size, 1026)
        model.eval()
        with torch.no_grad():
            compact = model(trimmed).logits
            full = model(dict(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask]),
                              labels=torch.tensor([label]))).logits
            self.assertTrue(torch.allclose(compact, full, atol=2e-6, rtol=2e-6))
            preds = model.generate(trimmed, 10, True, catalog_prefix(tokenizer))
        self.assertEqual(len({tuple(x) for x in preds['preds'][0].tolist()}), 10)
        self.assertTrue(torch.isfinite(preds['scores']).all())
        model.train()
        loss = model(trimmed).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in model.parameters()))

    def test_repository_metrics_full_and_conditional_and_rng_preserved(self):
        _, tok, arrays, audit = load_sids(self.sid)
        seed_all(40)
        trainer = self.trainer(tok)
        before = rng_state()
        results = evaluate(trainer, tok, arrays['base_val'], audit['base_val'],
                           list(range(10)), 'cpu', 3, progress=False)
        self.assertTrue(torch.equal(before['torch'], rng_state()['torch']))
        self.assertTrue(trainer.model.training)
        self.assertEqual(results['unavailable_targets'], 5)
        self.assertEqual(results['decoded_queries'], 5)
        for key in results['full_denominator']:
            self.assertAlmostEqual(results['full_denominator'][key],
                                   results['conditional_on_servable'][key]*0.5)
        self.assertEqual(results['invalid_predictions'], 0)

    def test_full_validation_selected_reload_hashes_and_status(self):
        report = self.train('full', 4, smoke=False)
        self.assertTrue(report['selected_checkpoint_reload_equal'])
        self.assertEqual(report['full_base_validation']['all_queries'], 10)
        self.assertEqual(report['E0_base_end_to_end_status'], 'completed')
        self.assertEqual(report['E0_full_status'], 'not_run')
        self.assertEqual(report['E1_status'], 'not_run')
        for name, digest in report['artifact_files_sha256'].items():
            self.assertEqual(sha256(self.root/'full'/name), digest)
        selected = torch.load(self.root/'full/best.pth', weights_only=False)
        self.assertEqual(selected['optimizer_step'], report['best_optimizer_step'])
        _, tok, arrays, _ = load_sids(self.sid)
        model = self.trainer(tok).model
        model.load_state_dict(selected['state_dict'])
        model.eval()
        batch = make_batch(arrays['base_val'], np.arange(2), 'cpu')
        with torch.no_grad():
            first = model(batch).logits.clone()
            model.load_state_dict(torch.load(self.root/'full/best.pth', weights_only=False)['state_dict'])
            self.assertTrue(torch.equal(first, model(batch).logits))

    def test_corruption_and_bad_config_fail_before_output(self):
        with (self.sid/'tiger/base_train.npz').open('ab') as handle:
            handle.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.train('bad')
        self.assertFalse((self.root/'bad').exists())

    def test_rehashed_wrong_labels_still_fail(self):
        with np.load(self.sid/'tiger/base_train.npz') as file:
            values = {key: file[key] for key in file.files}
        values['labels'][0, 0] = 999
        np.savez_compressed(self.sid/'tiger/base_train.npz', **values)
        self.rehash('tiger/base_train.npz')
        with self.assertRaisesRegex(ValueError, 'label row mismatch'):
            self.train('bad')
        self.assertFalse((self.root/'bad').exists())

    def test_resume_rejects_changed_config_and_no_overwrite(self):
        self.train('partial', 2)
        args = self.args('changed', 4, self.root/'partial/last.pth')
        args.lr = 0.002
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            run(args)
        self.assertFalse((self.root/'changed').exists())
        with self.assertRaises(FileExistsError):
            self.train('partial', 4)

    def test_probe_selection_is_deterministic_and_includes_cold_queries(self):
        _, _, _, audit = load_sids(self.sid)
        selected = probe_rows(audit['base_val'], 10, 2026)
        self.assertEqual(selected, list(range(10)))
        self.assertEqual(probe_rows(audit['base_val'], 6, 7), probe_rows(audit['base_val'], 6, 7))
        self.assertNotEqual(probe_rows(audit['base_val'], 6, 7), probe_rows(audit['base_val'], 6, 8))

    def test_actual_t5_learns_synthetic_train_examples(self):
        self.train('learn', 60)
        values = [json.loads(line)['loss'] for line in (self.root/'learn/train_metrics.jsonl').read_text().splitlines()]
        self.assertLess(np.mean(values[-10:]), np.mean(values[:10])*0.75)


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False).result
    print(json.dumps(dict(round='round9', inputs='synthetic_only', tests_run=result.testsRun,
        execution_status='passed' if result.wasSuccessful() else 'failed',
        E0_full_status='not_run', E1_status='not_run'), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
