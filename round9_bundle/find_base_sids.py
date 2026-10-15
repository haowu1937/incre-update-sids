"""Print the newest completed SID directory for the selected Beauty Base checkpoint."""
import json
from pathlib import Path

PROTOCOL_SHA = '97ecf43d164f74e9b13f2c814d3edd9cace9130e8f9a9d2581421b7b2b6e2e66'
RQ_SHA = 'b1d696a8bd086b8c76f01afafd4380bfba14c05413abe17923459b73ac0b7b8d'
matches = []
for path in Path('cache/incremental/Beauty').glob('*/summary.json'):
    try:
        summary = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if (summary.get('stage') == 'E0_base_frozen_sid_preparation'
            and summary.get('execution_status') == 'completed'
            and summary.get('protocol_sha256') == PROTOCOL_SHA
            and summary.get('rqvae_checkpoint_sha256') == RQ_SHA):
        matches.append(path)
if not matches:
    raise SystemExit('No completed frozen SID directory matches the selected Beauty protocol/RQ-VAE.')
print(max(matches, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent.as_posix())
