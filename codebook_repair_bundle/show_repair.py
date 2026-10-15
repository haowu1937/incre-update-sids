"""Print compact repair and optional evaluation results (about 25 lines)."""
import json
from pathlib import Path
import sys

if len(sys.argv) not in (2, 3):
    raise SystemExit('Usage: python show_repair.py REPAIR_DIR [EVAL_DIR]')
s = json.loads((Path(sys.argv[1])/'summary.json').read_text())
print('accepted_repair_steps =', s['accepted_repair_steps'])
print('proposals_checked =', s['proposals_checked'], '| stop_reason =', s['stop_reason'])
print('full_training_anchor_queries =', s['hard_acceptance_queries'], '| gradient_queries =', s['gradient_subset_queries'])
print('training_hard_positive_damage: plain = %.9g repair = %.9g' % (
    s['initial_full_anchor_train']['positive_damage'], s['final_full_anchor_train']['positive_damage']))
print('RQ_MSE: plain = %.9g repair = %.9g fixed_cap = %.9g' % (
    s['fixed_initial_reconstruction_mse'], s['final_train_catalog_metrics']['reconstruction_loss'], s['fixed_reconstruction_mse_cap']))
print('unique_codes =', s['collision_statistics']['unique_semantic_codes'], '| max_bucket =', s['collision_statistics']['maximum_collision_bucket'])
print('full_sid_export_status =', s['full_sid_export_status'])
print('encoder_unchanged =', s['encoder_unchanged'], '| decoder_unchanged =', s['decoder_unchanged'])
print('extra_repair_compute = True | holdout_used_for_selection =', s['anchor_holdout_used_for_selection'])
if len(sys.argv) == 3:
    e = json.loads((Path(sys.argv[2])/'summary.json').read_text())
    for group, comparisons in e['paired_repair_vs_plain'].items():
        print(group)
        for condition, comparison in comparisons.items():
            damage = e['groups'][group][condition]['mean_positive_part_delta']
            print('  %s: positive_damage=%.9g difference=%.9g CI95=%s' % (
                condition, damage, comparison['mean_repair_minus_plain_positive_damage'], comparison['positive_damage_cluster_ci95']))
            if group == 'anchor_holdout':
                print('  paired_Delta_NLL=%.9g CI95=%s' % (
                    comparison['mean_repair_minus_plain_sequence_nll'], comparison['sequence_nll_cluster_ci95']))
    ranking = e['base_validation_ranking']
    if ranking:
        print('metric old plain repair repair-minus-plain (full denominator)')
        for metric in ('recall@5', 'recall@10', 'ndcg@5', 'ndcg@10'):
            print(metric, *['%.9g' % ranking[key]['full_denominator'][metric] for key in ('old', 'plain', 'repaired')],
                '%.9g' % ranking['repair_minus_plain_full_denominator'][metric])
        print('invalid_predictions =', ranking['repaired']['invalid_predictions'])
