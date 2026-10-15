"""Locate the completed Beauty Base TIGER whose selected checkpoint was reported."""
import json
from pathlib import Path

PROTOCOL = '97ecf43d164f74e9b13f2c814d3edd9cace9130e8f9a9d2581421b7b2b6e2e66'
SELECTED = '4538d5d8bcab84e6f6919cdd66a041c257d1fa62f8519bd787918f00a68d8451'
matches = []
for path in Path('ckpt/incremental/Beauty').glob('*/summary.json'):
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    if (report.get('stage') == 'E0_base_content_TIGER'
            and report.get('execution_status') == 'completed'
            and report.get('E0_base_end_to_end_status') == 'completed'
            and report.get('protocol_sha256') == PROTOCOL
            and report.get('artifact_files_sha256', {}).get('best.pth') == SELECTED):
        matches.append(path)
if not matches:
    raise SystemExit('No completed Base TIGER matches the reported selected best.pth SHA.')
print(max(matches, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent.as_posix())
