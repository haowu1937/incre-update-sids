"""Locate the completed Beauty E1 bound to the reported teacher and plain Update."""
import json
from pathlib import Path

PROTOCOL = '97ecf43d164f74e9b13f2c814d3edd9cace9130e8f9a9d2581421b7b2b6e2e66'
TEACHER = '4538d5d8bcab84e6f6919cdd66a041c257d1fa62f8519bd787918f00a68d8451'
UPDATE_NAME = 'plain_update_rqvae_1000_20261009T124124Z'
matches = []
for path in Path('results/incremental/Beauty').glob('*/summary.json'):
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    if (report.get('stage') == 'E1_historical_input_reencoding_damage'
            and report.get('E1_status') == 'completed'
            and report.get('protocol_sha256') == PROTOCOL
            and report.get('selected_tiger_sha256') == TEACHER
            and Path(report.get('update_rqvae_dir', '')).name == UPDATE_NAME):
        matches.append(path)
if not matches:
    raise SystemExit('No completed E1 matches the reported Beauty experiment. Set E1_DIR manually.')
print(max(matches, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent.as_posix())
