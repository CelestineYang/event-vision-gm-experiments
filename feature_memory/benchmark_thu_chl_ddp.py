"""Train-only bounded THU benchmark; never stops or modifies existing jobs."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from thu_chl_fusion import load_model
from thu_chl_sequence_dataset import THUVideoSequenceDataset
from train_thu_chl_gm import configuration,seed_all
from train_feature_memory_ddp import ChunkModel,run_pass
from gcstr_video_sequence_dataset import make_schedule

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');world=dist.get_world_size()
free,total=torch.cuda.mem_get_info();budget=min(free-2*1024**3,int(total*.46))
if budget<8*1024**3:raise RuntimeError('Less than 8 GiB safe allocator budget')
torch.cuda.set_per_process_memory_fraction(budget/total,rank)
cfg=configuration(Path(__file__).parent/'configs/thu_chl_gm.json');cfg.update(video_batch_size_per_gpu=16,ddp_world_size=world)
global_cfg=dict(cfg,video_batch_size_per_gpu=16*world)
data=THUVideoSequenceDataset(global_cfg,'train',1)
candidates=sorted(v for v in data.groups if len(data.groups[v])>=5)
chosen=sorted(np.random.default_rng(0).choice(candidates,size=49,replace=False).tolist())
data.groups={v:data.groups[v] for v in chosen};data.windows=sum(map(len,data.groups.values()))
data.position={i:(j,indices) for indices in data.groups.values() for j,i in enumerate(indices)}
data.schedule=make_schedule(data.groups,16*world,4,cfg['seed'],1,True)
if rank==0:
 args.output.mkdir(parents=True,exist_ok=False)
 (args.output/'config.json').write_text(json.dumps(dict(config=cfg,videos=chosen,windows=data.windows,chunks=len(data),global_batch=16*world),indent=2)+'\n')
dist.barrier();seed_all(0)
model=load_model(cfg).cuda(rank);groups,_=model.parameter_groups(cfg);opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
ddp=DDP(ChunkModel(model,cfg['point_aux_weight']),device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
seed_all(rank);reports=[]
for repeat in range(2):
 dist.barrier();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 result,_=run_pass(ddp,data,torch.device('cuda',rank),cfg,opt)
 torch.cuda.synchronize();elapsed=time.perf_counter()-started
 payload=dict(rank=rank,seconds=elapsed,allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20,budget_mib=budget/2**20)
 memory=[None]*world;dist.all_gather_object(memory,payload)
 if rank==0:
  reports.append(dict(repeat=repeat,metrics=result,memory=memory,wall_seconds=max(r['seconds'] for r in memory),windows_per_second=data.windows/max(r['seconds'] for r in memory)))
  (args.output/'results.json').write_text(json.dumps(dict(status='complete' if repeat==1 else 'running',world=world,per_gpu_batch=16,global_batch=16*world,windows=data.windows,videos=49,reports=reports),indent=2)+'\n')
  print(json.dumps(dict(event='benchmark',world=world,repeat=repeat,windows=data.windows,memory=memory)),flush=True)
dist.destroy_process_group()
