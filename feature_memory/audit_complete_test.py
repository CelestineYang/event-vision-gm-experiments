"""Read-only audit of omitted UCF test records against original archive payloads."""
import csv,json,io
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import numpy as np
from scipy.io import loadmat
B=Path('/remote-home/datasets/Joind-dataset for Action Recognition/ucf101_feature_memory_migration_20261024_v1')
ROOT=B.parent
report=json.loads((B/'data/experiment/dataset_audit.json').read_text())
split={r['video_id']:r for r in csv.DictReader((B/'data/experiment/video_splits.csv').open()) if r['split']=='test'}
archives={};result=[]
try:
 for r in report['excluded_videos']:
  if r['split']!='test':continue
  row=dict(video_id=r['video_id'],label=r['label_id'],old_reason=r['filter_reason'])
  name=r['source_archive']
  if name not in archives:
   z=ZipFile(ROOT/name);archives[name]=(z,z.namelist())
  z,names=archives[name];relative=split[r['video_id']]['event_relpath']
  matches=[n for n in names if n==relative or n.endswith('/'+relative)]
  if len(matches)!=1:raise ValueError((relative,matches))
  raw=loadmat(io.BytesIO(z.read(matches[0])))
  fields={k:np.asarray(raw[k]).reshape(-1) for k in ['ts','x','y','pol']}
  assert len({len(v) for v in fields.values()})==1
  t=fields['ts'].astype('i8');row['raw_events']=len(t)
  if len(t):
   row.update(span_us=int(t.max()-t.min()),backward_pairs=int((np.diff(t)<0).sum()),large_backward_pairs=int((np.diff(t)<-2**31).sum()))
   ts=np.sort(t,kind='stable');count=max(0,(int(ts[-1])-int(ts[0])-1000000)//500000+1)
   starts=int(ts[0])+np.arange(count,dtype='i8')*500000
   sizes=np.searchsorted(ts,starts+1000000)-np.searchsorted(ts,starts)
   row.update(candidate_windows=count,nonempty_windows=int((sizes>0).sum()),low_nonempty_windows=int(((sizes>0)&(sizes<=10000)).sum()),empty_windows=int((sizes==0).sum()))
  result.append(row)
  if len(result)%25==0:print('audited',len(result),flush=True)
finally:
 for z,_ in archives.values():z.close()
out=dict(expected_test=len(split),missing_from_old_manifest=len(result),records=result,confirmed_empty=sum(r['raw_events']==0 for r in result))
Path('/remote-home/yanglansong/feature_memory/complete_test_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='records'}),flush=True)
