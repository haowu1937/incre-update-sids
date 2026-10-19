"""Future causality, fixed catalogs, reserved slots and denominator checks; no Torch."""
import ast
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genrec.future_protocol import (check_registry_prefixes, sliced_metrics,
                                    summarize_records, target_group, validate_stage)
from genrec.frozen_sid_tokenizer import FrozenSIDTokenizer
from genrec.temporal import build_temporal_protocol
from genrec.temporal_sids import allocate_suffixes, code_to_tokens, maps_from_ledger, prepare_split


class Checks(unittest.TestCase):
    def setUp(self):
        reviews = [('u'+str(i%5), 'item'+str(i%8), i+1, 5.) for i in range(200)]
        bounds = build_temporal_protocol(reviews, max_history=4)['manifest']['boundaries']
        for key, raw in (('base_train_end','base_val_new'), ('base_end','update_new'),
                         ('update_train_end','update_val_new'), ('update_end','future_only')):
            index = bounds[key]
            user, _, ts, rating = reviews[index]
            reviews[index] = (user, raw, ts, rating)
        self.protocol = build_temporal_protocol(reviews, max_history=4)
        self.manifest, self.registries = self.protocol['manifest'], self.protocol['registries']

    def validate(self, stage='future_test', rows=None, seen=()):
        return validate_stage(rows or self.protocol['examples'][stage], stage,
                              self.manifest, self.registries, seen)

    def test_stage_protocol_counts_and_global_id_prefixes(self):
        check_registry_prefixes(self.registries, self.manifest)
        seen = []
        for stage in ('update_train', 'update_val', 'future_test'):
            self.validate(stage, seen=seen)
            seen += [row['sample_id'] for row in self.protocol['examples'][stage]]

    def test_simultaneous_or_future_history_rejected(self):
        for stamp in ('same', 'later'):
            rows = copy.deepcopy(self.protocol['examples']['future_test'])
            rows[0]['time_seq'][0] = rows[0]['target_time']+(stamp == 'later')
            with self.assertRaisesRegex(ValueError, 'Noncausal'):
                self.validate(rows=rows)

    def test_wrong_availability_and_catalog_rejected(self):
        for key, value in (('catalog_version','base_train'), ('target_servable',False)):
            rows = copy.deepcopy(self.protocol['examples']['update_train'])
            rows[0][key] = value
            with self.assertRaisesRegex(ValueError, 'availability'):
                self.validate('update_train', rows)

    def test_future_item_cannot_be_added_to_catalog(self):
        bad = copy.deepcopy(self.registries)
        bad['update']['id2item'].append('future_only')
        bad['update']['item2id']['future_only'] = len(bad['update']['id2item'])-1
        with self.assertRaisesRegex(ValueError, 'registry'):
            check_registry_prefixes(bad, self.manifest)
        self.assertEqual(target_group('future_only',self.registries),'future_only')
        self.assertEqual(target_group('update_val_new',self.registries),'update_validation_first_seen')

    def test_duplicate_samples_across_stages_rejected(self):
        sample = self.protocol['examples']['future_test'][0]['sample_id']
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            self.validate(seen=[sample])

    def test_future_unavailable_history_blocks_keep_position_and_old_training_slots(self):
        raw = self.registries['update']['id2item'][1:]
        codes = [[0,0,0] for _ in raw]
        ledger = allocate_suffixes(raw, codes, [2]*3, 32)
        _, mapping = maps_from_ledger(ledger)
        tokenizer = FrozenSIDTokenizer(mapping, [2,2,2,32], 4)
        rows = self.protocol['examples']['future_test']
        arrays, audit, _ = prepare_split(rows,tokenizer,{item:i for i,item in enumerate(raw)})
        self.assertEqual(len(audit),len(rows))
        for i, source in enumerate(arrays['source_rows']):
            for j, available in enumerate(rows[source]['history_available']):
                if not available:
                    self.assertEqual(arrays['input_ids'][i,j*4:j*4+4].tolist(),[0]*4)
                    self.assertEqual(arrays['attention_mask'][i,j*4:j*4+4].tolist(),[0]*4)
        n = len(self.registries['update_train']['id2item'])-1
        train = allocate_suffixes(raw[:n],codes[:n],[2]*3,32)
        extended = allocate_suffixes(raw,codes,[2]*3,32,train)
        self.assertEqual(train['entries'],extended['entries'][:n])

    def test_versioned_extension_preserves_departed_old_addresses(self):
        tree = ast.parse((ROOT/'genrec/historical_damage.py').read_text())
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='versioned_suffixes')
        namespace = dict(code_to_tokens=code_to_tokens)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<real_versioned_allocator>','exec'),namespace)
        allocate = namespace['versioned_suffixes']
        old = allocate_suffixes(['a','b'],[[0,0,0],[1,0,0]],[2]*3,4)
        train = allocate(old,['a','b'],[[1,0,0],[1,0,0]])
        deployment = allocate(old,['a','b','c'],[[1,0,0],[1,0,0],[0,0,0]])
        self.assertEqual(deployment['entries'][:2],train['entries'])
        self.assertEqual(deployment['entries'][2]['collision_suffix'],1)
        self.assertEqual(old['entries'][0]['collision_suffix'],0)

    def test_full_denominator_and_conditional_metrics_are_separate(self):
        records = [dict(target_servable=True,sequence_nll=2.,metrics={'recall@10':1.},
                        target_group='base_train',update_train_frequency=3),
                   dict(target_servable=False,sequence_nll=None,metrics={'recall@10':0.},
                        target_group='future_only',update_train_frequency=0)]
        report = summarize_records(records,['recall@10'])
        self.assertEqual(report['full_denominator']['recall@10'],.5)
        self.assertEqual(report['conditional_on_servable']['recall@10'],1.)
        self.assertEqual(report['coverage'],.5)
        self.assertIsNone(sliced_metrics(records,['recall@10'])['target_group:future_only']['mean_sequence_nll_servable'])
        records[1]['metrics']['recall@10'] = 1.
        with self.assertRaisesRegex(ValueError,'unavailable'):
            summarize_records(records,['recall@10'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
