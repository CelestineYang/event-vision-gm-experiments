"""Authorized epoch-boundary switch, preserving source run and fallback resume."""
import json,os,signal,subprocess,time
from pathlib import Path
BASE=Path('/remote-home/datasets/DailyDVS-200-prepared')
STATE=BASE/'ddp4_migration/switch_status.json'
OLD=BASE/'gcstr_run1';NEW=BASE/'gcstr_run1_ddp4'
PYTHON='/root/anaconda3/envs/torch201/bin/python'

def status(state,**extra):
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,**extra),indent=2)+'\n');tmp.replace(STATE)
 print(json.dumps(dict(state=state,**extra)),flush=True)

def running(pid):
 p=Path(f'/proc/{pid}/stat')
 return p.exists() and p.read_text().split()[2]!='Z'

def main():
 status('waiting_for_four_gpu_benchmark')
 deadline=time.monotonic()+1800
 while not (BASE/'ddp4_migration/four.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Four-GPU benchmark did not finish; original untouched')
  time.sleep(2)
 single=json.loads((BASE/'ddp4_migration/single.json').read_text());four=json.loads((BASE/'ddp4_migration/four.json').read_text())
 speedup=single['seconds']/four['seconds']
 if speedup<1.1:status('not_switched',reason='less than 10 percent measured speedup',speedup=speedup);return
 if NEW.exists():raise RuntimeError('Target run already exists')
 launch=json.loads((BASE/'gcstr_run1_launch.json').read_text());pid=launch['pid']
 status('waiting_for_epoch_save',target_epoch=5,source_pid=pid,speedup=speedup)
 while True:
  s=json.loads((OLD/'status.json').read_text())
  if s['epoch']>=5:break
  if not running(pid):raise RuntimeError('Original process exited before epoch5 save')
  time.sleep(2)
 cmdline=Path(f'/proc/{pid}/cmdline').read_bytes()
 if str(OLD).encode() not in cmdline or os.getpgid(pid)!=pid:raise RuntimeError('Source process identity mismatch')
 status('stopping_source_at_saved_epoch',epoch=s['epoch'],pid=pid)
 os.killpg(pid,signal.SIGTERM)
 for _ in range(60):
  if not running(pid):break
  time.sleep(1)
 else:raise RuntimeError('Original did not stop; not launching duplicate')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 # Explicit free rendezvous endpoint avoids sharing the existing UCF store.
 cmd=[PYTHON,'-u','-m','torch.distributed.run','--rdzv-backend=c10d','--rdzv-endpoint=localhost:0','--rdzv-id=dailydvs200-ddp4','--nproc-per-node=4',str(BASE/'gcstr_ddp4_code/train_gcstr.py'),'--store',str(BASE/'prepared_v1'),'--output',str(NEW),'--epochs','50','--batch-size','128','--workers','2','--lr','0.0001','--seed','0','--device','cuda:0','--test-as-val','--migrate-from',str(OLD)]
 with (BASE/'gcstr_run1_ddp4.log').open('x') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 (BASE/'gcstr_run1_ddp4_launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd,'source_epoch':s['epoch'],'physical_gpus':[0,1,2,3]},indent=2)+'\n')
 status('starting_four_gpu',pid=p.pid,source_epoch=s['epoch'],speedup=speedup)
 for _ in range(600):
  log=(BASE/'gcstr_run1_ddp4.log').read_text(errors='replace')
  if '"phase": "train", "step": 1' in log:
   status('switched',pid=p.pid,source_epoch=s['epoch'],speedup=speedup,output=str(NEW));return
  if p.poll() is not None:break
  time.sleep(1)
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=60)
 # Original files/weights are intact, so recovery uses the exact old configuration.
 env['CUDA_VISIBLE_DEVICES']='3';env['TORCH_HOME']=launch['torch_home']
 with (BASE/'gcstr_run1.log').open('a') as log:
  restored=subprocess.Popen(launch['command']+['--resume'],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 launch['pid']=restored.pid;(BASE/'gcstr_run1_launch.json').write_text(json.dumps(launch,indent=2)+'\n');(BASE/'gcstr_run1.pid').write_text(str(restored.pid)+'\n')
 status('fallback_single_restarted',pid=restored.pid,failed_ddp_pid=p.pid,source_epoch=s['epoch'])

if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=repr(e));raise
