"""Preview/copy updates to existing project imports; never deletes or pushes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'feature_memory', 'thu_preprocessing', 'dailydvs200_preprocessing', 'audit_gcstr'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    for entry in json.loads((ROOT/'provenance/import_manifest.json').read_text())['files']:
        rel = Path(entry['path'])
        if rel.parts[0] not in ALLOWED or '..' in rel.parts:
            continue
        src, dst = args.source/rel, ROOT/rel
        if not src.is_file() or src.is_symlink():
            print('MISSING/SYMLINK (not removed):', rel)
            continue
        content = src.read_bytes()
        if dst.exists() and hashlib.sha256(content).digest() == hashlib.sha256(dst.read_bytes()).digest():
            continue
        print('UPDATE' if args.apply else 'WOULD UPDATE', rel)
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

if __name__ == '__main__':
    main()
