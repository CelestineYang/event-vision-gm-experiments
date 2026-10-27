"""Benchmark three/four ranks, then restore the existing formal three-rank job."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch

ROOT=Path('/remote-home/datasets/UCF101-feature-memory-runs')
RUN=ROOT/'GM_ddp3_batch48_20261024T144614Z'
OUT=ROOT/'GM_three_four_comparison_v1'
WORK=Path(__file__).resolve().parent

def main():
    OUT.mkdir(exist_ok=False)
    saved=torch.load(RUN/'training/last.pt',map_location='cpu')
    assert saved['epoch']>=18 and saved['optimizer_state']['state'] and len(saved['rng_states'])==3
    epoch=saved['epoch'];del saved
    pid=int((RUN/'pid').read_text())
    assert str(RUN).encode() in Path(f'/proc/{pid}/cmdline').read_bytes() and os.getpgid(pid)==pid
    os.killpg(pid,signal.SIGTERM)
    codes={}
    try:
        time.sleep(5)
        for world in [3,4]:
            out=OUT/f'gpu{world}';out.mkdir()
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with (out/'benchmark.log').open('w') as log:
                p=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
                    'benchmark_three_gpu.py','--worker','--output',str(out)],cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:codes[str(world)]=p.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM);p.wait();raise
    finally:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with (RUN/'train.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',
                'train_feature_memory_ddp.py','--config',str(RUN/'train_config.json'),'--output',str(RUN/'training'),'--resume'],
                cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (RUN/'pid').write_text(str(p.pid))
        (OUT/'status.json').write_text(json.dumps(dict(benchmark_exit_codes=codes,restored_pid=p.pid,resume_epoch=epoch+1),indent=2))
        print('Restored formal training',p.pid,'epoch',epoch+1,flush=True)

if __name__=='__main__':main()
