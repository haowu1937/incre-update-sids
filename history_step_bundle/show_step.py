"""Print about 25 lines for two checkpoints."""
import json
from pathlib import Path
import sys

if len(sys.argv)!=2:raise SystemExit('Usage: python show_step.py DIAGNOSTIC_DIR')
s=json.loads((Path(sys.argv[1])/'summary.json').read_text())
print('execution_status =',s['execution_status'],'| training_steps_committed =',s['training_steps_committed'])
print('holdout_used =',s['holdout_used'],'| checkpoint_writes =',s['checkpoint_writes'])
for p in s['points']:
    print('checkpoint_step =',p['checkpoint_step'],'| full_train_queries =',p['full_anchor_train_before']['queries'])
    print('H_before = %.9g | next_content_batch_size = %d' % (p['full_anchor_train_before']['positive_damage'],p['next_content_batch_size']))
    print('candidate hard_delta proxy_book_delta RQ_MSE semantic_change suffix_change feasible')
    for r in p['candidates']:
        fmt=lambda v:'-' if v is None else '%.7g' % v
        print(r['name'],fmt(r['hard_positive_damage_change']),fmt(r['proxy_linear_history_change_codebooks_only']),
            fmt(r['full_catalog_rq_metrics']['reconstruction_loss']),fmt(r['changed_catalog_semantic_fraction']),
            fmt(r['changed_catalog_suffix_fraction']),r['full_sid_allocation_feasible'])
    print('joint_minus_content_hard_damage =',p['joint_minus_content_hard_positive_damage'])
    print('extra_Adam_update/content_Adam_update_by_level =',
        [r['extra_update_relative_to_content_update'] for r in p['extra_history_adam_update_by_level']])
    print('extra_Adam_update_cosine_with_negative_history_gradient =',
        [r['extra_update_cosine_with_negative_history_gradient'] for r in p['extra_history_adam_update_by_level']])
