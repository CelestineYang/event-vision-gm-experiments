"""Train-only fixed input diagnostic: README feature KNN vs upstream module coordinate KNN."""
from pathlib import Path
import json
import numpy as np
import torch
from gesture_eventmamba_pretrained import PretrainedEventMambaV2
from gesture_sequence_dataset import GestureVideoSequenceDataset,point_input
from gcstr_video_sequence_dataset import keyed_rng
from eventmamba_v1_layers import LocalGrouper
cfg=json.loads((Path(__file__).parent/'configs/gesture_gm_smoke.json').read_text());ds=GestureVideoSequenceDataset(cfg,'train',0)
chosen=[]
for label in range(10):
 ids=[i for i,r in enumerate(ds.rows) if r['label']==label]
 chosen.extend(ids[j] for j in np.linspace(0,len(ids)-1,8,dtype=int))
points=[];labels=[];opened={}
for i in chosen:
 r=ds.rows[i];name=r['event_file']
 if name not in opened:opened={name:np.load(ds.store/name,mmap_mode='r')}
 a=opened[name][r['left']:r['right']];p,_=point_input((a[:,2],a[:,0],a[:,1],a[:,3]),keyed_rng('audit',i));points.append(p);labels.append(r['label'])
x=torch.from_numpy(np.stack(points)).cuda();y=torch.tensor(labels,device='cuda');torch.set_num_threads(1);results={}
for kind in ['feature','coordinate']:
 m=PretrainedEventMambaV2.from_checkpoint(cfg['point_checkpoint']).eval()
 if kind=='coordinate':
  for i in range(3):
   old=m.local_grouper_list[i];g=LocalGrouper(m.feature_list[i],m.group_number[i],24,False,'anchor');g.load_state_dict(old.state_dict());m.local_grouper_list[i]=g
 m=m.cuda();logits=[]
 with torch.inference_mode():
  torch.manual_seed(17)
  for offset in range(0,len(x),16):logits.append(m(x[offset:offset+16]))
 z=torch.cat(logits);results[kind]={'train_windows':len(x),'correct':int((z.argmax(-1)==y).sum()),'ce':float(torch.nn.functional.cross_entropy(z,y))};print(json.dumps(results[kind]|{'knn':kind}),flush=True)
 del m
out=Path(cfg['store']).parent/'gesture_checkpoint_knn_diagnostic.json';out.write_text(json.dumps({'split':'train only, no augmentation, 8 windows/class','results':results},indent=2)+'\n')
