"""Check tracked files for accidentally committed data, binaries or credentials."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = {'.pt','.pth','.ckpt','.npy','.npz','.h5','.hdf5','.aedat','.aedat4','.mat','.whl','.so','.pdf','.zip'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')

def main():
    files = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).split(b'\0')
    failed = []
    for raw in files:
        if not raw:
            continue
        path = ROOT/raw.decode()
        if path.is_symlink() or path.suffix.lower() in BANNED or path.name == '.env':
            failed.append((str(path.relative_to(ROOT)), 'forbidden file type'))
            continue
        data = path.read_bytes()
        if len(data)>10*1024**2 or b'\0' in data:
            failed.append((str(path.relative_to(ROOT)), 'binary/oversized'))
        if SECRET.search(data):
            failed.append((str(path.relative_to(ROOT)), 'possible credential'))
    if failed:
        for item in failed: print(*item, sep=': ')
        raise SystemExit(1)
    print('PASS: tracked files contain no forbidden data/binaries or recognized credential patterns.')

if __name__ == '__main__':
    main()
