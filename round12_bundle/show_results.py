"""Print the compact matched compatibility evaluation for manual sharing."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = json.loads((root/'summary.json').read_text())
print(json.dumps({key: report[key] for key in (
    'execution_status', 'update_optimizer_steps', 'compatibility_weight',
    'paired_compatible_vs_plain', 'rqvae_reconstruction', 'churn')}, indent=2))
print(json.dumps({'anchor_holdout': report['groups']['anchor_holdout'],
                  'base_validation': report['groups']['base_validation']}, indent=2))
if report['base_validation_ranking'] is not None:
    print(json.dumps(report['base_validation_ranking'], indent=2))
