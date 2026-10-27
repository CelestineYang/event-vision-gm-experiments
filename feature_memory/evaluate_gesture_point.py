"""Full local Test evaluation of the unmodified uploaded point classifier."""
import json,time
from pathlib import Path
import numpy as np
import torch
from gesture_sequence_dataset import GestureVideoSequenceDataset,point_input
from gesture_eventmamba_pretrained import PretrainedEventMambaV2
from gcstr_video_sequence_dataset import keyed_rng
from train_gesture_gm import PredictionMetrics,configuration,seed_all
cfg=configuration(Path(__file__).parent/'configs/gesture_gm_smoke.json');seed_all(100000);torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(.20)
m=PretrainedEventMambaV2.from_checkpoint(cfg['point_checkpoint']).cuda().eval()
ds=GestureVideoSequenceDataset(cfg,'test');metrics=PredictionMetrics();opened={};started=time.monotonic()
with torch.inference_mode():
 for offset in range(0,len(ds.rows),32):
  rows=ds.rows[offset:offset+32];points=[]
  for r in rows:
   name=r['event_file']
   if name not in opened:
    opened={name:np.load(ds.store/name,mmap_mode='r')}
   a=opened[name][r['left']:r['right']]
   p,_=point_input((a[:,2],a[:,0],a[:,1],a[:,3]),keyed_rng('points',cfg['seed'],0,r['video_id'],r['window_id']),1024);points.append(p)
  logits=m(torch.from_numpy(np.stack(points)).cuda());metrics.update(logits,torch.tensor([r['label'] for r in rows]),[r['video_id'] for r in rows])
  if offset%640==0:print(json.dumps({'windows':offset+len(rows),'total':len(ds.rows),'seconds':time.monotonic()-started}),flush=True)
result,predictions=metrics.finish();result.update(seconds=time.monotonic()-started,checkpoint=cfg['point_checkpoint'],sha256=cfg['point_sha256'],protocol='local Test, deterministic online 1024-point sampling; probability-mean segment vote; eval seed100000',load_report=m.load_report)
out=Path(cfg['store']).parent/'gesture_point_test_audit.json';out.write_text(json.dumps({'metrics':result,'predictions':predictions},indent=2)+'\n');print(json.dumps(result),flush=True)
