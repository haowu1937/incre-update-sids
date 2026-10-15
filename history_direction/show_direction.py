"""Print a compact local direction table for manual sharing."""
import json
from pathlib import Path
import sys

r = json.loads((Path(sys.argv[1])/'summary.json').read_text())
print('probe_queries =', r['probe_queries'])
print('base_hard_positive_damage =', f"{r['hard_reference']['positive_damage']:.8f}")
print('STE_tangent_relative_gradient_difference =', r['gradient_match']['relative_gradient_difference'])
print('zero_surrogate_gradient =', r['zero_surrogate_gradient'])
print('scope relative_step direction hard_delta tangent_delta changed_history unique_codes max_bucket RQ_MSE')
for x in r['candidates']:
    print(x['scope'], f"{x['relative_step']:.1e}", 'descent' if x['sign'] == -1 else 'reverse',
          f"{x['hard_positive_damage_change']:+.7f}",
          f"{x['tangent_positive_damage_change']:+.7f}",
          f"{x['hard']['changed_history_fraction_from_probe_base']:.4f}",
          x['hard']['unique_semantic_codes'], x['hard']['maximum_collision_bucket'],
          f"{x['rq_metrics']['reconstruction_loss']:.7g}")
