"""Read-only targeted checks of actual trained branches and official numerics."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path('/remote-home/yanglansong')
sys.path.insert(0,str(ROOT/'thu_preprocessing'))
from train_gcstr import Windows, Metrics, load_frozen_branch
from gcstr_representations import build_gcstr_views, _mean_timestamp_xy
from render_frames import render

torch.set_num_threads(2)
report={}
cases=[('THU','THU-EACT-50-CHL-prepared','chl_prepared','ty_p',8),
       ('Gesture','DVSGesture-prepared','gesture_prepared_v2','xt_p',10),
       ('Daily','DailyAction-DVS-prepared','daily_prepared','ty_p',12)]
for name,folder,store,view,n in cases:
    base=Path('/remote-home/datasets')/folder
    data=Windows(base/store,'test')
    samples={r['sample_id']:r for r in json.loads((base/store/'samples.json').read_text())}
    checks={}
    for v in ['cstr3','xt_p','ty_p']:
        rows=json.loads((base/'gcstr_run1'/v/'predictions_best.json').read_text())
        assert len(rows)==len({r['sample_id'] for r in rows})
        assert all(r['label']==samples[r['sample_id']]['label'] and samples[r['sample_id']]['split']=='test' for r in rows)
        metrics=[json.loads(s) for s in (base/'gcstr_run1'/v/'metrics.jsonl').read_text().splitlines()]
        selected=min(metrics,key=lambda x:x['test']['video_nll'])
        accuracy=sum(x['prediction']==x['label'] for x in rows)/len(rows)
        nll=np.mean([x['nll'] for x in rows])
        assert abs(accuracy-selected['test']['video_top1'])<1e-12
        assert abs(nll-selected['test']['video_nll'])<1e-10
        checks[v]=dict(epoch=selected['epoch'],videos=len(rows),top1=accuracy,metrics_match=True)
    chosen=[];labels=set()
    for row in data.rows:
        if row['label'] not in labels:
            chosen.append(row['sample_id']);labels.add(row['label'])
        if len(chosen)==n:break
    data.rows=[r for r in data.rows if r['sample_id'] in chosen]
    model=load_frozen_branch(base/'gcstr_run1'/view/'best_loss.pt')
    before={k:v.clone() for k,v in model.state_dict().items()}
    sums={};count={};metric=Metrics()
    with torch.no_grad():
        for images,labels,ids in torch.utils.data.DataLoader(data,batch_size=16,num_workers=0):
            logits=model(images[view]);metric.update(logits,labels,ids)
            for p,i in zip(logits.double().softmax(-1).numpy(),ids):
                sums[i]=sums.get(i,0)+p;count[i]=count.get(i,0)+1
    actual,rows=metric.finish()
    expected={r['sample_id']:r for r in json.loads((base/'gcstr_run1'/view/'predictions_best.json').read_text())}
    mismatch=[r['sample_id'] for r in rows if r['prediction']!=expected[r['sample_id']]['prediction']]
    assert not mismatch,mismatch
    assert all(r['prediction']==int((sums[r['sample_id']]/count[r['sample_id']]).argmax()) for r in rows)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    # Check that changing a row label cannot change the rendered model input.
    x=data[0][0];old=data.rows[0]['label'];data.rows[0]['label']=(old+1)%data.classes
    y=data[0][0];data.rows[0]['label']=old
    assert all(torch.equal(x[v],y[v]) for v in x)
    cfg=json.loads((base/'gcstr_run1/config.json').read_text());hashes={}
    for f,h in cfg['code'].items():
        candidates=[ROOT/'thu_preprocessing'/f,ROOT/'thu_preprocessing/legacy_chl_run1'/f]
        hashes[f]=[str(p) for p in candidates if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==h]
    # Official timestamp normalization vs the implementation actually used by training.
    row=data.rows[0];events=np.load(base/store/row['event_file'],mmap_mode='r')[row['left']:row['right']]
    views=render(events,row['start_us'],row['end_us'],data.sensor)
    original=build_gcstr_views(events[:,2],events[:,0],events[:,1],events[:,3],views=('cstr3','xt_p','ty_p'),config=data.sensor)
    delta={v:dict(max_abs=float(np.abs(views[v]-original[v]).max()),mean_abs=float(np.abs(views[v]-original[v]).mean())) for v in views}
    report[name]=dict(saved_metrics=checks,real_inference=dict(view=view,videos=len(rows),windows=len(data),predictions_match=True,
        model_buffers_unchanged=True,independent_soft_vote_match=True,label_does_not_change_input=True),trained_source_matches=hashes,
        fixed_window_vs_event_extent_projection_difference=delta)
    print(name,json.dumps(report[name]),flush=True)

# Execute the official mean-timestamp function itself, without importing its CLI/cache code.
tree=ast.parse((ROOT/'gcstr_official_audit/frame_dataset.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_mean_timestamps')
scope={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'official_mean','exec'),scope)
rng=np.random.default_rng(7);a=np.zeros(1000,dtype=[('x','i8'),('y','i8'),('t','i8')]);a['x']=rng.integers(0,128,1000);a['y']=rng.integers(0,128,1000);a['t']=np.sort(rng.integers(0,1000000,1000))
expected=scope['get_mean_timestamps'](a,(128,128))[0];actual=_mean_timestamp_xy(a['t'],a['x'],a['y'],128,128)
np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
report['official_mean_function_match']=True
(ROOT/'audit_gcstr/runtime_audit.json').write_text(json.dumps(report,indent=2))
