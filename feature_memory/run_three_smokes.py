"""Run independent real-data fresh/resume smoke checks; never formal training."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess

WORK = Path(__file__).resolve().parent
ROOT = Path('/remote-home/datasets/UCF101-feature-memory-runs')
PYTHON = str(ROOT / 'venv/bin/python')

def run(mode, gpu, root):
    output = root / (mode + '_smoke')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    status = root / (mode + '_status.json')
    command = [PYTHON, '-u', 'train_gcstr_feature_memory.py', '--config',
               str(WORK / 'configs/feature_memory_target' / (mode + '.json')),
               '--output', str(output), '--smoke']
    for stage in ('fresh', 'resume'):
        status.write_text(json.dumps(dict(mode=mode, gpu=gpu, stage=stage, status='running')))
        with (root / (mode + '_' + stage + '.log')).open('w') as log:
            result = subprocess.run(command + (['--resume'] if stage == 'resume' else []),
                                    cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.write_text(json.dumps(dict(mode=mode, gpu=gpu, stage=stage, status='failed', returncode=result.returncode)))
            return False
    status.write_text(json.dumps(dict(mode=mode, gpu=gpu, status='passed', stages=['fresh', 'resume'])))
    return True

if __name__ == '__main__':
    root = ROOT / ('smoke_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir()
    print(str(root), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, mode, gpu, root) for mode, gpu in [('B', 1), ('G', 2), ('GM', 3)]]
        results = [f.result() for f in futures]
    print('All smoke checks passed:', all(results), flush=True)
    raise SystemExit(0 if all(results) else 1)
