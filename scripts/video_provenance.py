"""Source fingerprints for evaluation video annotations and composites."""
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    entry = {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if path.suffix != '.mp4':
        entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entry


def current(target, manifest, expected):
    if not target.exists() or not manifest.exists():
        return False
    return json.loads(manifest.read_text()) == expected


def write_manifest(path, record):
    Path(path).write_text(json.dumps(record, indent=2)+'\n')


def harness(row, hybrid=False):
    if row.get('harness'):
        return row['harness']
    return 'v9 (inferred legacy)' if hybrid else 'original SO101 port (unversioned)'
