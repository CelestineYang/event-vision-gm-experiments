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
for kind in ['feature']:
 m=PretrainedEventMambaV2.from_checkpoint(cfg['point_checkpoint']).eval()
 if kind=='coordinate':
  for i in range(3):
   old=m.local_grouper_list[i];g=LocalGrouper(m.feature_list[i],m.group_number[i],24,False,'anchor');g.load_state_dict(old.state_dict());m.local_grouper_list[i]=g
 m=m.cuda();logits=[]
 with torch.inference_mode():
  torch.manual_seed(17)
  for offset in range(0,len(x),16):logits.append(m(x[offset:offset+16]))
 z=torch.cat(logits);results[kind]={'train_windows':len(x),'correct':int((z.argmax(-1)==y).sum()),'ce':float(torch.nn.functional.cross_entropy(z,y))};print(json.dumps(results[kind]|{'knn':kind}),flush=True)
 mapping={0:0,1:1,2:3,3:4,4:5,5:6,6:7,7:8,8:9}
 target=torch.tensor([mapping.get(v,-1) for v in labels],device='cuda');mask=target>=0
 results[kind]['known_class_remap_correct']=int((z.argmax(-1)[mask]==target[mask]).sum())
 results[kind]['known_class_windows']=int(mask.sum())
 results[kind]['confusion']={str(label):torch.bincount(z.argmax(-1)[y==label],minlength=10).tolist() for label in range(10)}
 # Verify the putative output class 2 on independent raw Train 'other gesture' segments.
 import csv
 rawroot=Path('/remote-home/datasets/Joind-dataset for Action Recognition/DVSGesture')
 files=sorted({r['event_file'] for r in ds.rows});extra=[]
 for name in files[::12][:8]:
  a=np.load(ds.store/name,mmap_mode='r');ann=rawroot/'annotations'/(Path(name).stem+'_labels.csv')
  with ann.open() as f:r=next(r for r in csv.DictReader(f) if int(r['class'])==11)
  seg=a[(a[:,2]>=int(r['startTime_usec']))&(a[:,2]<int(r['endTime_usec']))]
  start=int(seg[0,2]);window=seg[(seg[:,2]>=start)&(seg[:,2]<start+500000)]
  pts,_=point_input((window[:,2],window[:,0],window[:,1],window[:,3]),keyed_rng('other',name));extra.append(pts)
 with torch.inference_mode():other=m(torch.from_numpy(np.stack(extra)).cuda()).argmax(-1).tolist()
 results[kind]['official_other_train_predictions']=other
 print(json.dumps(results[kind]),flush=True)
 del m
out=Path(cfg['store']).parent/'gesture_checkpoint_label_diagnostic.json';out.write_text(json.dumps({'split':'train only, no augmentation, 8 windows/class','results':results},indent=2)+'\n')
