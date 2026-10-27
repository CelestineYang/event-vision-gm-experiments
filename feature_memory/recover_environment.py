"""Recover the interrupted dependency installation, then continue validation."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/remote-home/datasets/UCF101-feature-memory-runs')
python = str(root / 'venv/bin/python')
env = os.environ.copy()
env.update(PIP_CACHE_DIR=str(root / 'pip_cache'), TMPDIR=str(root / 'tmp'))
(root / 'tmp').mkdir(exist_ok=True)
status = root / 'environment_status.json'
try:
    status.write_text(json.dumps({'stage': 'recovering_pip_and_torch'}))
    subprocess.run([python, '-m', 'pip', 'install', '--upgrade', 'pip==24.3.1',
                    '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple'], env=env, check=True)
    subprocess.run([python, '-m', 'pip', 'install',
                    'https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl',
                    'https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl',
                    'numpy==1.26.3', '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple'],
                   env=env, check=True)
except BaseException as exc:
    status.write_text(json.dumps({'stage': 'failed', 'error': repr(exc)}))
    raise
# PID 0 has no /proc entry, so the supervisor proceeds directly to validation.
os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name('prepare_environment.py')), '0'])
