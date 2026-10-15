"""Print the diagnostic evidence needed to choose the next training change."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = json.loads((root/'summary.json').read_text())
failed = report['failed_checkpoint']
if failed:
    print(json.dumps({'failed_checkpoint': {
        'optimizer_steps': failed['optimizer_steps'],
        'unique_semantic_codes': failed['observation']['unique_semantic_codes'],
        'maximum_collision_bucket': failed['observation']['maximum_collision_bucket'],
        'codebooks': failed['observation']['codebooks'],
        'latent_variance_mean': failed['observation']['latent_variance_mean'],
        'full_update_catalog_metrics': failed['full_update_catalog_metrics'],
        'mean_embedding_baseline_mse': failed['mean_embedding_baseline_mse']
    }}, indent=2))
for name, lane in report['lanes'].items():
    traces = [json.loads(x) for x in (root/name/'trace.jsonl').read_text().splitlines()]
    selected = {1,2,3,lane['optimizer_steps']}
    if lane['first_complete_collapse_post_step'] is not None:
        selected.add(lane['first_complete_collapse_post_step'])
    details = []
    for row in traces:
        if row['step'] in selected:
            details.append(dict(step=row['step'],
                rq_loss=row['rq_loss'], history_penalty=row['history_penalty'],
                unique_codes=row['post_update']['unique_semantic_codes'],
                max_bucket=row['post_update']['maximum_collision_bucket'],
                codebooks_used=[x['used'] for x in row['post_update']['codebooks']],
                latent_variance_mean=row['post_update']['latent_variance_mean'],
                gradient_contributions=row['gradient_contributions'],
                soft_assignment=row['soft_assignment'], parameter_updates=row['parameter_updates']))
    print(json.dumps({name: dict(first_complete_collapse_post_step=lane['first_complete_collapse_post_step'],
        all_complete_collapse_post_steps=lane['all_complete_collapse_post_steps'],
        final_rq_metrics=lane['final_rq_metrics'], selected_steps=details)}, indent=2))
