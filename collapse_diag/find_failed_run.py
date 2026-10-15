"""Locate the latest completed 1000-step fixed-suffix Beauty run for diagnosis."""
import json
from pathlib import Path

matches = []
for path in Path('ckpt/incremental/Beauty').glob('*/summary.json'):
    try:
        row = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if (row.get('stage') == 'historical_aware_update_content_rqvae'
            and row.get('execution_status') == 'completed'
            and row.get('update_optimizer_steps') == 1000
            and row.get('protocol_sha256') == '97ecf43d164f74e9b13f2c814d3edd9cace9130e8f9a9d2581421b7b2b6e2e66'
            and row.get('config', {}).get('history_input_condition') == 'new_semantics_old_suffix'
            and row.get('full_sid_export_status') == 'capacity_exhausted'):
        matches.append(path)
if not matches:
    raise SystemExit('No completed 1000-step capacity-exhausted fixed-suffix Beauty run found.')
print(max(matches, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent.as_posix())
