"""Wait for committed epoch 18, then resume the same three ranks on GPUs 1/2/3."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

RUN = Path('/remote-home/datasets/UCF101-feature-memory-runs/GM_ddp3_batch48_20261024T144614Z')
WORK = Path(__file__).resolve().parent
STATUS = RUN/'gpu_switch_status.json'

def mark(stage, **extra):
    payload = dict(stage=stage, target_gpus=[1,2,3], **extra)
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2)+'\n')
    tmp.replace(STATUS)
    print(json.dumps(payload), flush=True)

def main():
    lock = (RUN/'gpu_switch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = int((RUN/'pid').read_text())
    cmdpath = Path(f'/proc/{pid}/cmdline')
    assert cmdpath.exists() and str(RUN).encode() in cmdpath.read_bytes()
    assert os.getpgid(pid) == pid
    mark('waiting_for_epoch18_checkpoint', old_pid=pid)
    while True:
        state = json.loads((RUN/'training/status.json').read_text())
        if state['completed_epoch'] >= 18:
            break
        if not cmdpath.exists() or not cmdpath.read_bytes():
            raise RuntimeError('Original trainer exited before epoch 18 was committed')
        time.sleep(5)
    # status.json is written after checkpoint and metrics are committed.
    import torch
    checkpoint = torch.load(RUN/'training/last.pt', map_location='cpu')
    epoch = checkpoint['epoch']
    assert epoch >= 18 and checkpoint['data_epoch'] == epoch
    assert len(checkpoint['rng_states']) == 3 and checkpoint['optimizer_state']['state']
    assert not checkpoint['smoke']
    del checkpoint
    assert str(RUN).encode() in cmdpath.read_bytes() and os.getpgid(pid) == pid
    mark('stopping_old_gpu_assignment', saved_epoch=epoch, old_pid=pid)
    os.killpg(pid, signal.SIGTERM)
    # Wait until the trainer's exclusive output lock is released.
    runlock = (RUN/'training/run.lock').open('a')
    for _ in range(60):
        try:
            fcntl.flock(runlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(runlock, fcntl.LOCK_UN)
            break
        except BlockingIOError:
            time.sleep(1)
    else:
        raise RuntimeError('Old trainer did not release its output lock')
    time.sleep(5)
    mark('checking_target_capacity', saved_epoch=epoch)
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--id=1,2,3', '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        free = [int(v.strip()) for v in raw.splitlines()]
        if len(free) == 3 and min(free) >= 11000:
            break
        mark('waiting_for_target_capacity', saved_epoch=epoch, free_mib=free)
        time.sleep(15)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1,2,3', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
               'train_feature_memory_ddp.py', '--config', str(RUN/'train_config.json'),
               '--output', str(RUN/'training'), '--resume']
    with (RUN/'train.log').open('a') as log:
        log.write('\nGPU switch: committed epoch %d; resume on physical GPUs 1,2,3\n' % epoch)
        log.flush()
        process = subprocess.Popen(command, cwd=WORK, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (RUN/'pid').write_text(str(process.pid))
    mark('restarted', pid=process.pid, saved_epoch=epoch, resume_epoch=epoch+1, command=command)
    time.sleep(30)
    if process.poll() is not None:
        raise RuntimeError('Resumed launcher exited; inspect train.log')
    mark('launcher_alive', pid=process.pid, saved_epoch=epoch, resume_epoch=epoch+1)

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        mark('failed', error=repr(error))
        raise
