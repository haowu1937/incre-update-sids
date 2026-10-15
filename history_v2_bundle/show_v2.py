"""Compact training/evaluation output; optional fixed-budget readiness check."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('training_dir');p.add_argument('evaluation_dir',nargs='?')
p.add_argument('--require-steps',type=int)
a=p.parse_args()
s=json.loads((Path(a.training_dir)/'summary.json').read_text())
print('execution_status =',s['execution_status'],'| stop_reason =',s['stop_reason'])
print('update_optimizer_steps =',s['update_optimizer_steps'],'| steps_this_run =',s['steps_this_run'])
print('rho =',s['config']['historical_v2']['history_gradient_ratio'],'| full_SID_forward = True | history_gradient = codebooks_only')
print('anchor_train_queries_per_step =',s['anchor_train_queries'],'| holdout_used =',s['anchor_holdout_used_for_training'])
print('history_gradient_queries_processed =',s['work_counters']['history_gradient_queries'])
print('peak_effective_gradient_ratio_by_level =',s['work_counters']['peak_effective_gradient_ratio_by_level'])
print('peak_relative_parameter_update_by_level =',s['work_counters']['peak_relative_parameter_update_by_level'])
print('final_train_full_SID_positive_damage = %.9g' % s['final_full_anchor_train']['positive_damage'])
print('RQ_MSE = %.9g | Plain_1000step_RQ_MSE = %.9g' % (
    s['final_train_catalog_metrics']['reconstruction_loss'],s['plain_train_catalog_metrics']['reconstruction_loss']))
print('base_semantic_churn = %.9g' % s['base_semantic_churn']['changed_item_fraction'])
print('unique_codes =',s['collision_statistics']['unique_semantic_codes'],'| max_bucket =',s['collision_statistics']['maximum_collision_bucket'])
print('full_sid_export_status =',s['full_sid_export_status'],'| reload_equal =',s['checkpoint_reload_assignments_equal'])
print('same_step_budget_as_plain =',s['same_optimizer_step_budget_as_plain'],'| same_compute_as_plain = False')
median=s.get('median_optimizer_step_seconds_this_run')
print('median_optimizer_step_seconds_this_run =',median)
if median is not None and s['execution_status']=='completed' and s['update_optimizer_steps']<s['plain_optimizer_step_budget']:
    print('rough_remaining_training_hours_to_1000 = %.3f (excludes startup/final evaluation)' % (
        median*(s['plain_optimizer_step_budget']-s['update_optimizer_steps'])/3600))
if a.require_steps is not None and (s['execution_status']!='completed' or s['update_optimizer_steps']!=a.require_steps):
    raise SystemExit('Requested step budget not completed; do not start formal evaluation.')
if a.evaluation_dir:
    e=json.loads((Path(a.evaluation_dir)/'summary.json').read_text())
    for group,conditions in e['paired_history_aware_vs_plain'].items():
        print(group)
        for condition,c in conditions.items():
            print('  %s: positive_damage=%.9g difference=%.9g CI95=%s' % (
                condition,e['groups'][group][condition]['mean_positive_part_delta'],
                c['mean_history_aware_minus_plain_positive_damage'],c['positive_damage_cluster_ci95']))
            if group=='anchor_holdout':
                print('  paired_Delta_NLL=%.9g CI95=%s' % (
                    c['mean_history_aware_minus_plain_sequence_nll'],c['sequence_nll_cluster_ci95']))
    r=e['base_validation_ranking']
    if r:
        print('metric old plain history_aware history_aware-minus-plain (full denominator)')
        for metric in ('recall@5','recall@10','ndcg@5','ndcg@10'):
            print(metric,*['%.9g' % r[k]['full_denominator'][metric] for k in ('old','plain','history_aware')],
                '%.9g' % r['history_aware_minus_plain_full_denominator'][metric])
        print('invalid_predictions =',r['history_aware']['invalid_predictions'])
