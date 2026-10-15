"""Install History-aware joint Update v2 complete files after checking their source versions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    repo = Path(args.repo).resolve()
    manifest = json.loads((bundle/'manifest.json').read_text())
    for name, value in manifest['dependencies'].items():
        if not (repo/name).is_file() or digest(repo/name) != value:
            raise ValueError('Required prior source differs; no files changed: '+name)
    pending = []
    for name, record in manifest['files'].items():
        source, target = bundle/'complete_files'/name, repo/name
        if digest(source) != record['new_sha256']:
            raise ValueError('Bundle checksum mismatch: '+name)
        if target.exists():
            existing = digest(target)
            if existing == record['new_sha256']:
                continue
            if existing != record['previous_sha256']:
                raise ValueError('Local file differs from expected source; no files changed: '+name)
        elif record['previous_sha256'] is not None:
            raise ValueError('Required source file missing; no files changed: '+name)
        pending.append((name, source, target))
    backup = repo/'historical_joint_v2_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for name, source, target in pending:
        if target.exists():
            old = backup/name
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(json.dumps(dict(installed_files=len(pending), repo=str(repo),
        backup_dir=str(backup) if backup.exists() else None), indent=2))


if __name__ == '__main__':
    main()
