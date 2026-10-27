"""Read-only audit of official Gesture membership, annotations and window bounds."""
import csv,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
root=Path('/remote-home/datasets/Joind-dataset for Action Recognition/DVSGesture')
store=Path('/remote-home/datasets/DVSGesture-prepared/gesture_prepared_v2')
config=json.loads((store/'config.json').read_text());samples=json.loads((store/'samples.json').read_text());report={}
for split in ('train','test'):
 p=root/'splits'/f'trials_to_{split}.txt';names=p.read_text().split()
 assert hashlib.sha256(p.read_bytes()).hexdigest()==config[split+'_list_sha256']
 records=[r for r in samples if r['split']==split];assert {r['filename'] for r in records}==set(names)
 expected=set()
 for name in names:
  with (root/'annotations'/(Path(name).stem+'_labels.csv')).open() as f:
   for j,row in enumerate(csv.DictReader(f)):
    label=int(row['class']);start=int(row['startTime_usec']);end=int(row['endTime_usec'])
    if label==11:continue
    expected.add((name,j,label-1,start,end))
 assert {(r['filename'],r['annotation_row'],r['label'],r['segment_start_us'],r['segment_end_us']) for r in records}==expected
 rows=[json.loads(l) for l in (store/f'{split}_windows.jsonl').read_text().splitlines()];byvideo={r['sample_id']:[] for r in records}
 for row in rows:byvideo[row['sample_id']].append(row)
 current=None;events=None
 for record in records:
  if current!=record['event_file']:
   current=record['event_file'];events=np.load(store/current,mmap_mode='r')
  # A recording may reset timestamps between annotations; mask exactly like preparation.
  t=events[:,2];indices=np.flatnonzero((t>=record['segment_start_us'])&(t<record['segment_end_us']))
  assert len(indices) and np.all(np.diff(indices)==1)
  base=int(indices[0]);segment=t[indices];assert np.all(np.diff(segment)>=0)
  bounds=[(s,s+500000,int(np.searchsorted(segment,s,'left'))+base,int(np.searchsorted(segment,s+500000,'right'))-1+base) for s in range(int(segment[0]),int(segment[-1])-500000,250000)]
  actual=byvideo[record['sample_id']]
  assert [(r['start_us'],r['end_us'],r['left'],r['right']) for r in actual]==bounds
  assert all(r['label']==record['label'] and r['right']>r['left'] and r['raw_events']==r['right']-r['left'] for r in actual)
 report[split]={'recordings':len(names),'segments':len(records),'windows':len(rows),'invalid_windows':sum(not r['valid'] for r in rows),'labels':dict(Counter(r['label'] for r in records)),'subjects':sorted({r['subject'] for r in records})}
assert not set(report['train']['subjects'])&set(report['test']['subjects'])
report['result']='all official membership, labels, raw segment boundaries and EventMamba window indices match'
out=store.parent/'gesture_gm_protocol_audit.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
