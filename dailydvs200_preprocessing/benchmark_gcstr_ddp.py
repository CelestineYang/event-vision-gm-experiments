"""Bounded same-data/same-checkpoint single vs four GPU benchmark."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,'/remote-home/datasets/DailyDVS-200-prepared/gcstr_run1_code')
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from train_gcstr import Windows,GlobalBatchSampler,VIEWS
from gcstr_models import GCSTRBranchClassifier

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
 world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank)
 if world>1:dist.init_process_group('nccl')
 torch.manual_seed(0);np.random.seed(0)
 ds=Windows('/remote-home/datasets/DailyDVS-200-prepared/prepared_v1','train');ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(0))[:2048].tolist();ds.rows=[ds.rows[i] for i in ids];ds.epoch=5
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 models={};opts={}
 for v in VIEWS:
  m=GCSTRBranchClassifier(v,200,False).cuda(rank)
  if world>1:m=torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
  m.load_state_dict(saved['models'][v],strict=True);opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(saved['optimizers'][v])
  models[v]=DDP(m,device_ids=[rank]) if world>1 else m;opts[v]=opt
 del saved
 workers=4 if world==1 else 2
 loader=DataLoader(ds,batch_sampler=GlobalBatchSampler(len(ds),128,rank,world,5),num_workers=workers,pin_memory=True,multiprocessing_context='spawn')
 torch.cuda.reset_peak_memory_stats();times=[];last=time.perf_counter();measured=None
 for step,(images,labels,_) in enumerate(loader,1):
  labels=labels.cuda(rank)
  for v in VIEWS:
   opts[v].zero_grad(set_to_none=True);out=models[v](images[v].cuda(rank,non_blocking=True));loss=torch.nn.functional.cross_entropy(out,labels)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward();opts[v].step()
  torch.cuda.synchronize();now=time.perf_counter();times.append(now-last);last=now
  if step==2:measured=now
  if rank==0:print(json.dumps({'step':step,'steps':len(loader),'seconds':times[-1]}),flush=True)
 elapsed=last-measured
 payload=dict(rank=rank,steady_seconds=elapsed,allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20)
 results=[None]*world
 if world>1:dist.all_gather_object(results,payload)
 else:results=[payload]
 if rank==0:
  result=dict(world=world,global_batch=128,workers_per_rank=workers,total_windows=2048,warmup_batches=2,measured_windows=1792,seconds=max(r['steady_seconds'] for r in results),ranks=results,batch_seconds_rank0=times)
  result['windows_per_second']=1792/result['seconds'];a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
