"""Pure temporal and metric checks for paired Future evaluation."""
from collections import defaultdict
import math


def validate_stage(rows, stage, manifest, registries, earlier_sample_ids=()):
    reference = 'update' if stage == 'future_test' else 'update_train'
    if manifest['catalog_reference'][stage] != reference:
        raise ValueError('Unexpected evaluation catalog reference')
    catalog = set(registries[reference]['id2item'][1:])
    window = manifest['windows'][stage]
    seen = set(earlier_sample_ids)
    if len(rows) != window['example_count']:
        raise ValueError('Evaluation example count mismatch')
    missing_targets = missing_histories = 0
    for row in rows:
        items, times = row['item_seq'], row['time_seq']
        if (len(items) < 2 or len(items)-1 > manifest['max_history'] or len(items) != len(times)
                or any(type(t) is not int for t in times)
                or times[-1] != row['target_time']
                or any(t >= times[-1] for t in times[:-1])
                or times[:-1] != sorted(times[:-1])
                or not window['first_timestamp'] <= times[-1] <= window['last_timestamp']):
            raise ValueError('Noncausal history or target outside temporal window')
        expected = [raw in catalog for raw in items[:-1]]
        available = items[-1] in catalog
        if (row['catalog_version'] != reference or row['history_available'] != expected
                or any(type(v) is not bool for v in row['history_available'])
                or type(row['target_servable']) is not bool or row['target_servable'] != available):
            raise ValueError('Stage availability flags disagree with frozen catalog')
        if row['sample_id'] in seen:
            raise ValueError('Repeated sample ID across temporal stages')
        seen.add(row['sample_id'])
        missing_targets += not available
        missing_histories += not all(expected)
    if (missing_targets != window['unservable_targets']
            or missing_histories != window['examples_with_unavailable_history']):
        raise ValueError('Stage availability counts disagree with manifest')
    if stage == 'update_train' and (missing_targets or missing_histories):
        raise ValueError('Update training contains unavailable items')
    return seen


def check_registry_prefixes(registries, manifest):
    previous = []
    for stage in ('base_train', 'base', 'update_train', 'update'):
        registry = registries[stage]
        ids, mapping = registry['id2item'], registry['item2id']
        if (not ids or ids[0] != '[PAD]' or len(ids) != len(mapping)
                or any(mapping.get(raw) != i for i, raw in enumerate(ids))
                or ids[:len(previous)] != previous
                or len(ids)-1 != manifest['catalog_sizes'][stage]):
            raise ValueError('Non-prefix temporal registry or invalid global IDs')
        previous = ids


def target_group(raw, registries):
    for stage, label in (('base_train', 'base_train'), ('base', 'base_validation_first_seen'),
                         ('update_train', 'update_train_first_seen'), ('update', 'update_validation_first_seen')):
        if raw in registries[stage]['item2id']:
            return label
    return 'future_only'


def summarize_records(records, metric_keys):
    if not records:
        raise ValueError('Empty Future evaluation cohort')
    available = [row for row in records if row['target_servable']]
    for row in records:
        if (set(row['metrics']) != set(metric_keys)
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in row['metrics'].values())
                or (not row['target_servable'] and any(row['metrics'].values()))):
            raise ValueError('Invalid metric or nonzero score for unavailable target')
        if row['target_servable'] and (row['sequence_nll'] is None or not math.isfinite(row['sequence_nll'])):
            raise ValueError('Missing/nonfinite servable target NLL')
        if not row['target_servable'] and row['sequence_nll'] is not None:
            raise ValueError('Unavailable target has an invented NLL')
    sums = {key: sum(row['metrics'][key] for row in records) for key in metric_keys}
    return dict(all_queries=len(records), servable_queries=len(available),
        unavailable_targets=len(records)-len(available), coverage=len(available)/len(records),
        full_denominator={key: value/len(records) for key, value in sums.items()},
        conditional_on_servable={key: value/len(available) if available else None for key, value in sums.items()},
        mean_sequence_nll_servable=sum(row['sequence_nll'] for row in available)/len(available) if available else None)


def sliced_metrics(records, metric_keys):
    groups = defaultdict(list)
    for row in records:
        groups['target_group:'+row['target_group']].append(row)
        groups['update_train_frequency_le_5' if row['update_train_frequency'] <= 5
               else 'update_train_frequency_gt_5'].append(row)
    return {name: summarize_records(group, metric_keys) for name, group in sorted(groups.items())}
