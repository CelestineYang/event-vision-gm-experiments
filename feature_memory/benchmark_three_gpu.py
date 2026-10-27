"""Compare three-card GM throughput; always restore the existing single-card job."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/remote-home/datasets/UCF101-feature-memory-runs')
WORK = Path(__file__).resolve().parent
SINGLE = ROOT/'GM_single_batch24_20261024T143833Z'


def worker(output):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from train_feature_memory_ddp import ChunkModel, run_pass
    from train_gcstr_feature_memory import configuration, seed_all
    from gcstr_feature_memory_fusion import load_model
    from gcstr_video_sequence_dataset import VideoSequenceDataset, make_schedule
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    world = dist.get_world_size()
    cfg = configuration(SINGLE/'train_config.json')
    cfg['video_batch_size_per_gpu'] = 16
    global_cfg = dict(cfg, video_batch_size_per_gpu=16*world)
    data = VideoSequenceDataset(global_cfg, 'train', 1)
    chosen = sorted(np.random.default_rng(0).choice(sorted(data.groups), size=96, replace=False).tolist())
    reference = json.loads((ROOT/'GM_single_batch_benchmark_v2/GM_batch24.json').read_text())
    assert chosen == reference['videos']
    data.groups = {v:data.groups[v] for v in chosen}
    data.windows = sum(map(len,data.groups.values()))
    assert data.windows == reference['windows']
    data.position = {i:(j,indices) for indices in data.groups.values() for j,i in enumerate(indices)}
    data.schedule = make_schedule(data.groups,16*world,4,cfg['seed'],1,True)
    seed_all(cfg['seed'])
    model = load_model(cfg).cuda()
    groups,_ = model.parameter_groups(cfg)
    optimizer = torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
    checkpoint = torch.load(ROOT/'formal_batch16_20261024T105557Z/GM/last.pt',map_location='cpu')
    model.load_state_dict(checkpoint['model_state'],strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    del checkpoint
    ddp = DDP(ChunkModel(model,cfg['point_aux_weight']),device_ids=[rank],
              broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    seed_all(cfg['seed']+rank)
    reports=[]
    for repeat in range(2):
        dist.barrier();torch.cuda.reset_peak_memory_stats()
        metrics,_=run_pass(ddp,data,torch.device('cuda',rank),cfg,optimizer)
        peaks=[None]*world
        dist.all_gather_object(peaks,dict(rank=rank,allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                        reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        if rank==0:
            reports.append(dict(repeat=repeat,metrics=metrics,memory=peaks))
            (output/'results.json').write_text(json.dumps(dict(per_gpu_batch=16,global_batch=16*world,
                videos=chosen,windows=data.windows,chunks=len(data),reports=reports),indent=2))
    dist.destroy_process_group()


def supervise(output):
    output.mkdir(exist_ok=False)
    import torch
    checkpoint=torch.load(SINGLE/'last.pt',map_location='cpu')
    assert checkpoint['model_state'] and checkpoint['optimizer_state']['state']
    saved_epoch=checkpoint['epoch'];del checkpoint
    pid=int((SINGLE/'pid').read_text())
    cmd=Path(f'/proc/{pid}/cmdline')
    assert cmd.exists() and str(SINGLE).encode() in cmd.read_bytes() and os.getpgid(pid)==pid
    os.killpg(pid,signal.SIGTERM)
    result=None
    try:
        time.sleep(5)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (output/'benchmark.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',
                str(Path(__file__)),'--worker','--output',str(output)],cwd=WORK,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                result=proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait();raise
    finally:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with (SINGLE/'train.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u','train_gcstr_feature_memory.py','--config',str(SINGLE/'train_config.json'),
                '--output',str(SINGLE),'--resume'],cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (SINGLE/'pid').write_text(str(p.pid))
        (output/'status.json').write_text(json.dumps(dict(benchmark_exit_code=result,single_gpu_restarted_pid=p.pid,
            resume_epoch=saved_epoch+1),indent=2))
        print('Single-card GM resumed:',p.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    (worker if args.worker else supervise)(args.output)
