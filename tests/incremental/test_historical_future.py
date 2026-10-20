"""History-aware Future provenance checks and real RQ-VAE/TIGER integration."""
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name, value in [('USE_TF', '0'), ('USE_FLAX', '0'), ('HF_HUB_OFFLINE', '1'), ('TRANSFORMERS_OFFLINE', '1')]:
    os.environ.setdefault(name, value)
import evaluate_historical_future as driver


class Protocol(unittest.TestCase):
    def args(self):
        return driver.parser().parse_args(['--e1-dir', 'e1', '--retuned-dir', 'retuned', '--output', 'new'])

    def test_existing_future_recipe_defaults_and_invalid_values(self):
        args = self.args()
        driver.validate_args(args)
        self.assertEqual((args.max_steps, args.lr, args.warmup_steps, args.probe_size,
                          args.topk, args.num_beams), (2000, .003, 200, 512, [5, 10, 20], 20))
        for key, value in [('max_steps', 0), ('lr', math.nan), ('lr', -1), ('warmup_steps', -1),
                           ('num_beams', 10), ('topk', [5, 5]), ('topk', []), ('eval_batch_size', 0)]:
            changed = copy.deepcopy(args)
            setattr(changed, key, value)
            with self.assertRaises(ValueError):
                driver.validate_args(changed)

    @staticmethod
    def metadata():
        request = dict(format='retuned_scalar_history_v1', settings=dict(max_steps=1000), binding=dict(source='fixed'))
        pair = dict(execution_status='passed', same_update_steps=1000, processed_examples=254551,
            same_base=True, same_content_recipe=True, same_optimizer_and_clipping=True, same_content_sampler_endpoint=True)
        training = {arm: dict(execution_status='completed', update_optimizer_steps=1000,
            processed_examples=254551, teacher_unchanged=True, checkpoint_reload_assignments_equal=True,
            binding=request['binding'], artifact_files_sha256={'last.pth':'hash'}) for arm in ('plain', 'historical')}
        report = dict(stage='paired_retuned_scalar_history', execution_status='completed',
            original_inputs_unchanged=True, settings=request['settings'], binding=request['binding'],
            pair_checks=pair, training=training, evaluation=dict(execution_status='completed',
                binding=request['binding'], artifact_files_sha256={}), protected_input_sha256={})
        return request, report, driver.original_compact(report), pair

    def test_completed_budget_checks_and_exact_compact_projection(self):
        values = self.metadata()
        driver.validate_retuned_metadata(*values)
        for key, value in [('execution_status', 'failed'), ('original_inputs_unchanged', False)]:
            request, report, compact, pair = copy.deepcopy(values)
            report[key] = value
            with self.assertRaises(ValueError):
                driver.validate_retuned_metadata(request, report, compact, pair)
        request, report, compact, pair = copy.deepcopy(values)
        compact['training']['historical']['update_optimizer_steps'] = 999
        with self.assertRaises(ValueError):
            driver.validate_retuned_metadata(request, report, compact, pair)
        request, report, compact, pair = copy.deepcopy(values)
        report['training']['historical']['update_optimizer_steps'] = 999
        with self.assertRaises(ValueError):
            driver.validate_retuned_metadata(request, report, driver.original_compact(report), pair)


HAVE_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAVE_TORCH, 'Server launcher requires real RQ-VAE/TIGER checks')
class RealModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from genrec.temporal import build_temporal_protocol
        from genrec.temporal_sids import export_base_sids, sha256, write_json
        from genrec.temporal_tiger import parser as tiger_parser, run as train_tiger
        from gensid.incremental_content import parser as rq_parser, run as train_rq
        from gensid.plain_incremental_content import parser as plain_parser, run as train_plain
        from genrec.historical_damage import parser as e1_parser, run as measure_e1
        import train_retuned_historical as retuned
        torch.set_num_threads(1)
        cls.fixture = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.fixture.name)
        reviews = [('u'+str(i%5), 'item'+str(i%9), i+1, 5.) for i in range(200)]
        bounds = build_temporal_protocol(reviews, max_history=4)['manifest']['boundaries']
        for key, raw in [('base_train_end', 'base_val_new'), ('base_end', 'update_new'),
                         ('update_train_end', 'update_val_new'), ('update_end', 'future_only')]:
            i = bounds[key]
            u, _, ts, rating = reviews[i]
            reviews[i] = (u, raw, ts, rating)
        protocol = build_temporal_protocol(reviews, max_history=4)
        data = root/'protocol'; data.mkdir()
        write_json(data/'protocol_manifest.json', protocol['manifest'])
        write_json(data/'registries.json', protocol['registries'])
        for stage, rows in protocol['examples'].items():
            (data/(stage+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        write_json(data/'summary.json', dict(stage='E0_data_preparation', execution_status='completed',
            protocol_sha256=protocol['manifest']['protocol_sha256'],
            artifact_files_sha256={p.name:sha256(p) for p in data.iterdir()}))
        features = root/'features'; features.mkdir()
        n = len(protocol['registries']['update']['id2item'])-1
        values = np.random.default_rng(31).normal(.1, .3, (n, 8)).astype(np.float32)
        stats = {}
        for stage in ('base_train', 'update_train', 'update'):
            directory = features/stage; directory.mkdir()
            registry = protocol['registries'][stage]
            size = len(registry['id2item'])-1
            np.save(directory/'embeddings.npy', values[:size])
            write_json(directory/'registry.json', registry)
            write_json(directory/'feature_rows.json', [dict(raw_item_id=raw, gr_item_id=i+1,
                gensid_global_item_id=i, feature_row=i) for i, raw in enumerate(registry['id2item'][1:])])
            stats[stage] = dict(shape=[size, 8])
        write_json(features/'summary.json', dict(stage='E0_fixed_content_embeddings', execution_status='completed',
            graph_used=False, protocol_sha256=protocol['manifest']['protocol_sha256'], stages=stats,
            artifact_files_sha256={p.relative_to(features).as_posix():sha256(p)
                                  for p in features.rglob('*') if p.is_file()}))
        write_json(root/'tiny.json', dict(num_layers=1, num_decoder_layers=1, d_model=16,
            d_ff=32, num_heads=2, d_kv=8, dropout_rate=.2))
        with contextlib.redirect_stdout(io.StringIO()):
            train_rq(rq_parser().parse_args(['--features-dir', str(features), '--output', str(root/'base_rq'),
                '--device', 'cpu', '--max-steps', '8', '--batch-size', '4', '--e-dim', '4',
                '--num-emb-list', '2', '2', '2', '--layers', '16', '8', '--init-samples', '9',
                '--kmeans-iters', '5', '--eval-batch-size', '4']))
            export_base_sids(data, features, root/'base_rq', root/'sids', device='cpu', suffix_capacity=32)
            train_tiger(tiger_parser().parse_args(['--sid-dir', str(root/'sids'), '--output', str(root/'tiger'),
                '--device', 'cpu', '--max-steps', '20', '--batch-size', '8', '--eval-batch-size', '3',
                '--warmup-steps', '2', '--probe-size', '10', '--eval-every', '100',
                '--num-beams', '4', '--topk', '1', '3', '--model-config', str(root/'tiny.json')]))
            train_plain(plain_parser().parse_args(['--protocol-dir', str(data), '--features-dir', str(features),
                '--base-rqvae-dir', str(root/'base_rq'), '--output', str(root/'plain'), '--device', 'cpu',
                '--max-steps', '12', '--batch-size', '5', '--eval-batch-size', '4', '--lr', '.005']))
            measure_e1(e1_parser().parse_args(['--protocol-dir', str(data), '--tiger-dir', str(root/'tiger'),
                '--update-rqvae-dir', str(root/'plain'), '--sid-dir', str(root/'sids'), '--output', str(root/'e1'),
                '--expected-tiger-sha256', sha256(root/'tiger/best.pth'), '--device', 'cpu',
                '--batch-size', '3', '--anchor-size', '12', '--bootstrap-reps', '40', '--ranking']))
            report = retuned.run(retuned.parser().parse_args(['--e1-dir', str(root/'e1'),
                '--output', str(root/'retuned'), '--device', 'cpu', '--max-steps', '3',
                '--anchor-batch-size', '3', '--eval-batch-size', '3', '--save-every', '1']))
        if report['execution_status'] != 'completed':
            raise RuntimeError('Real retuned fixture failed: '+str(report))
        cls.original_hashes = {p:sha256(p) for p in root.rglob('*') if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.output = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.output.cleanup()

    def args(self, name='future', resume=False):
        values = ['--e1-dir', str(self.root/'e1'), '--retuned-dir', str(self.root/'retuned'),
            '--output', str(Path(self.output.name)/name), '--device', 'cpu', '--max-steps', '2',
            '--selection-steps', '1', '2', '--topk', '1', '3', '--num-beams', '4',
            '--eval-batch-size', '3', '--probe-size', '10', '--warmup-steps', '1', '--save-every', '1']
        return driver.parser().parse_args(values+(['--resume'] if resume else []))

    def run_pipeline(self, args):
        from genrec import future_evaluation as f
        original = f.load_stage
        def checked(ex, stage, seen=()):
            if stage == 'future_test':
                directory = Path(args.output)
                self.assertTrue((directory/'selection_locked.json').exists())
                self.assertTrue((directory/'historical/summary.json').exists())
            return original(ex, stage, seen)
        with patch.object(f, 'load_stage', checked), contextlib.redirect_stdout(io.StringIO()):
            return driver.run(args)

    def test_real_pipeline_keeps_rq_sids_and_evaluates_both_tiger_states(self):
        from genrec.temporal_sids import read_json, sha256
        report = self.run_pipeline(self.args())
        self.assertEqual(report['execution_status'], 'completed')
        self.assertEqual(set(report['future']), {'frozen', 'finetuned'})
        self.assertEqual(report['training']['update_optimizer_steps'], 2)
        self.assertTrue(report['training']['exact_base_warm_start'])
        self.assertFalse(report['rqvae_training_performed'])
        self.assertFalse(report['future_used_for_selection'])
        output = Path(self.args().output)
        ledger = read_json(output/'train_ledger.json')
        deployment = read_json(output/'deployment_ledger.json')
        self.assertEqual(ledger, read_json(self.root/'retuned/historical/allocation_registry.json'))
        self.assertEqual(deployment['entries'][:len(ledger['entries'])], ledger['entries'])
        for state, result in report['future'].items():
            self.assertEqual(result['invalid_predictions'], 0)
            self.assertGreater(result['unavailable_targets'], 0)
            self.assertTrue(result['teacher_frozen_during_evaluation'])
            for key, value in result['conditional_on_servable'].items():
                self.assertAlmostEqual(result['full_denominator'][key], value*result['coverage'])
        self.assertTrue(all(sha256(p)==value for p, value in self.original_hashes.items()))

    def test_exact_resume_with_dropout_and_future_cache(self):
        import torch
        from genrec import future_evaluation as f
        from genrec.trainer import Trainer
        self.run_pipeline(self.args('continuous'))
        original = Trainer.fit_steps
        def interrupted(trainer, arrays, sampler, generator, target, make_batch, callback):
            def stop(loss, norm):
                callback(loss, norm)
                if sampler['optimizer_steps'] == 1:
                    raise RuntimeError('intentional training interruption')
            return original(trainer, arrays, sampler, generator, target, make_batch, stop)
        with patch.object(Trainer, 'fit_steps', interrupted):
            with self.assertRaisesRegex(RuntimeError, 'intentional training'):
                self.run_pipeline(self.args('resumed'))
        self.assertFalse((Path(self.output.name)/'resumed/selection_locked.json').exists())
        self.run_pipeline(self.args('resumed', resume=True))
        for name in ('last.pth', 'best.pth'):
            a, b = [torch.load(Path(self.output.name)/run/'historical'/name, weights_only=False)
                    for run in ('continuous', 'resumed')]
            self.assertTrue(all(torch.equal(value, b['state_dict'][key]) for key, value in a['state_dict'].items()))
        # No training or Future recomputation is allowed for a completed cached run.
        with patch.object(f, 'fit_arm', side_effect=AssertionError('unexpected retraining')), \
             patch.object(f, 'future_metrics', side_effect=AssertionError('unexpected Future recomputation')):
            self.assertEqual(self.run_pipeline(self.args('resumed', resume=True))['execution_status'], 'completed')

    def test_partial_future_resume_preserves_locked_model_and_completed_predictions(self):
        from genrec import future_evaluation as f
        original = f.future_metrics
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('intentional Future interruption')
            return original(*args, **kwargs)
        with patch.object(f, 'future_metrics', interrupted):
            with self.assertRaisesRegex(RuntimeError, 'intentional Future'):
                self.run_pipeline(self.args())
        directory = Path(self.args().output)
        names = ['selection_locked.json', 'historical/best.pth', 'historical/last.pth', 'historical_frozen.json',
                 'historical_frozen_queries.jsonl']
        before = {name:driver.digest(directory/name) for name in names}
        report = self.run_pipeline(self.args(resume=True))
        self.assertEqual(report['execution_status'], 'completed')
        self.assertTrue(all(driver.digest(directory/name)==value for name, value in before.items()))
        (directory/'historical_frozen.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.run_pipeline(self.args(resume=True))

    def test_changed_input_mapping_and_resume_settings_are_rejected(self):
        args = self.args()
        self.run_pipeline(args)
        changed = self.args(resume=True); changed.lr *= 2
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            self.run_pipeline(changed)
        path = self.root/'retuned/historical/item2tokens.json'
        before = path.read_bytes()
        try:
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                self.run_pipeline(self.args(resume=True))
        finally:
            path.write_bytes(before)


if __name__ == '__main__':
    if '--require-torch' in sys.argv:
        sys.argv.remove('--require-torch')
        if not HAVE_TORCH:
            raise SystemExit('PyTorch required for real RQ-VAE/TIGER integration checks')
    unittest.main(verbosity=2)
