"""Audit every smoke window and run a single GPU optimization step per view."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'thu_preprocessing'))
import h5py
import numpy as np
import torch
from train_gcstr import Windows, VIEWS
from prepare_dailydvs200 import load_events
from gcstr_models import GCSTRBranchClassifier

store=Path(sys.argv[1]); results={}
for split in ('train','test'):
    ds=Windows(store,split); seen={}
    with h5py.File(store/f'{split}.h5') as h:
        assert h.attrs['complete']
        assert len(h['data'])==len(ds)
        for i,row in enumerate(ds.rows):
            if row['sample_id'] not in seen:
                e,_=load_events(row['event_file']);seen[row['sample_id']]=np.zeros(len(e),bool)
            seen[row['sample_id']][row['left']:row['right']]=True
            assert int(h['label'][row['point_row']])==row['label']
            pts=h['data'][row['point_row']]
            assert pts.shape==(2048,3) and np.isfinite(pts).all() and (pts>=0).all() and (pts<1).all()
            images,label,sid=ds[i]
            assert sid==row['sample_id'] and label==row['label']
            assert all(tuple(im.shape)==(3,224,224) and torch.isfinite(im).all() for im in images.values())
    assert all(mask.all() for mask in seen.values()), 'uncovered events'
    results[split]={'windows':len(ds),'recordings':len(seen),'all_events_covered':True,'all_images_finite':True}
train=Windows(store,'train'); batch=[train[i] for i in (0,len(train)-1)]
assert torch.cuda.is_available()
results['cuda_visible_devices']=__import__('os').environ.get('CUDA_VISIBLE_DEVICES')
results['gpu_steps']={}
for view in VIEWS:
    model=GCSTRBranchClassifier(view,200,pretrained=False).cuda().train()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    x=torch.stack([b[0][view] for b in batch]).cuda();y=torch.tensor([b[1] for b in batch],device='cuda')
    logits=model(x);assert logits.shape==(2,200)
    loss=torch.nn.functional.cross_entropy(logits,y);assert torch.isfinite(loss)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step();results['gpu_steps'][view]={'loss':loss.item(),'finite_gradients':True}
    del model,opt,x,y,logits,loss;torch.cuda.empty_cache()
(store/'smoke_check.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
