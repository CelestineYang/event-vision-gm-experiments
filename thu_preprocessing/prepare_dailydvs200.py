"""DailyDVS-200 v1: official Train/Test, full-cover 1 s / 0.5 s windows."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def bounds(t, window=1000000, step=500000):
    """Half-open windows; final event included; end-aligned tail, no duplicates."""
    if not len(t):
        raise ValueError('Empty recording requires explicit evaluation policy')
    duration = int(t[-1]) + 1
    starts = list(range(0, max(0, duration-window)+1, step))
    if duration > window and starts[-1] + window < duration:
        starts.append(duration-window)
    for start in starts:
        end = start + window
        yield start, end, int(np.searchsorted(t,start)), int(np.searchsorted(t,end))


def load_events(path):
    with np.load(path, allow_pickle=False) as z:
        t = z['t'].astype(np.int64)
        if not len(t): raise ValueError('Empty recording: '+str(path))
        origin = int(t[0]); t -= origin
        x,y,p = (z[k].astype(np.int64) for k in ('x','y','p'))
    if any(len(a)!=len(t) for a in (x,y,p)) or (np.diff(t)<0).any():
        raise ValueError('Invalid array lengths or timestamp order')
    if (x<0).any() or (x>=320).any() or (y<0).any() or (y>=240).any() or not np.isin(p,[0,1]).all():
        raise ValueError('Invalid coordinates/polarity')
    return np.column_stack([x,y,t,p]), origin


def sample_points(events, start, end, seed, count=2048):
    """t,x,y; same fixed-window coordinates for base cache and future online use."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(events), size=count, replace=len(events)<count))
    e = events[idx]
    return np.column_stack([(e[:,2]-start)/(end-start),e[:,0]/320.,e[:,1]/240.]).astype('f4')


def official_entries(path):
    rows=[]
    for line in path.read_text().splitlines():
        rel,label=line.rsplit(None,1); folder,name=rel.split('/')
        rows.append((f'action_{int(folder.split("_")[1]):03d}/{name}',int(label)))
    return rows


def prepare(args):
    root,out=args.root.resolve(),args.output.resolve()
    if out==root or root in out.parents: raise ValueError('Output must be outside source')
    records={}
    hashes={}
    for split in ('train','test'):
        src=root/'splits'/f'{split}_npz.csv'
        with src.open() as f: rows=list(csv.DictReader(f))
        official=args.official_splits/f'{split}.txt'
        if [(r['event_relpath'],int(r['label_id'])) for r in rows]!=official_entries(official):
            raise ValueError('Local split differs from official: '+split)
        hashes[split]={'csv':sha(src),'official':sha(official)}
        records[split]=rows
    if {r['event_relpath'] for r in records['train']} & {r['event_relpath'] for r in records['test']}:
        raise ValueError('Train/Test overlap')
    if args.smoke_per_split:
        # Spread across the split instead of taking a single class prefix.
        records={s:[rs[i] for i in np.linspace(0,len(rs)-1,args.smoke_per_split,dtype=int)] for s,rs in records.items()}
    out.mkdir(parents=True,exist_ok=False)
    config=dict(dataset='DailyDVS-200',classes=200,width=320,height=240,window_us=1000000,step_us=500000,
        points=2048,seed=args.seed,event_backend='npz_reference',timestamp_unit='us',
        timestamp_origin='first event; subtract in int64 before any floating conversion',
        point_coordinates='t=(t-start)/1000000, x=x/320, y=y/240; time-sorted; replacement only if fewer than 2048',
        window_boundary='[start,end); short recording retained; end-aligned tail covers final event',
        minimum_events=1,radius_filter=False,offline_point_augmentation=False,
        fusion_policy='online shared event augmentation BEFORE point sampling and frame rendering; cached points are unaugmented base only',
        model_selection='official Test NLL',independent_test=False,val_usage='unused',
        source_root=str(root),source_manifest_hashes=hashes,code_sha256=sha(__file__),numpy=np.__version__,
        scope='smoke' if args.smoke_per_split else 'full')
    save(out/'config.json',config)
    samples=[]
    for split,rs in records.items():
        for r in rs:
            samples.append(dict(sample_id=r['sample_id'],split=split,label=int(r['label_id']),
                filename=r['event_relpath'],event_file=str(root/r['npz_relpath']),event_backend='npz_reference'))
    save(out/'samples.json',samples)
    coverage={}; begun=time.monotonic()
    try:
        for split,rs in records.items():
            n=0; empty=[]; complete_records=0
            with h5py.File(out/f'{split}.h5','x') as h, (out/f'{split}_windows.jsonl').open('w') as f, (out/f'{split}_sources.jsonl').open('w') as sources:
                h.attrs['complete']=False
                h.attrs['config_sha256']=sha(out/'config.json')
                data=h.create_dataset('data',(0,2048,3),maxshape=(None,2048,3),dtype='f4',chunks=(1,2048,3))
                labels=h.create_dataset('label',(0,),maxshape=(None,),dtype='i2')
                marks=h.create_dataset('mark',(0,),maxshape=(None,),dtype='i4')
                for mark,r in enumerate(rs):
                    path=root/r['npz_relpath']; events,origin=load_events(path)
                    if len(events)!=int(r['event_count']) or origin!=int(r['start_time_usec']) or int(events[-1,2])!=int(r['duration_usec']):
                        raise ValueError('Source metadata mismatch: '+str(path))
                    sources.write(json.dumps(dict(sample_id=r['sample_id'],path=str(path),sha256=sha(path),events=len(events),timestamp_origin_us=origin))+'\n')
                    base=dict(sample_id=r['sample_id'],split=split,label=int(r['label_id']),filename=r['event_relpath'],event_file=str(path),event_backend='npz_reference',mark=mark)
                    rows=[]; points=[]
                    for wid,(start,end,left,right) in enumerate(bounds(events[:,2])):
                        valid=right>left
                        row=dict(**base,window_id=wid,start_us=start,end_us=end,left=left,right=right,raw_events=right-left,valid=valid,point_row=n+len(points) if valid else None)
                        rows.append(row)
                        if valid:
                            seed=int.from_bytes(hashlib.sha256(f'{args.seed}:{r["sample_id"]}:{wid}'.encode()).digest()[:8],'little')
                            points.append(sample_points(events[left:right],start,end,seed))
                        else: empty.append(dict(sample_id=r['sample_id'],window_id=wid))
                    if not points: raise ValueError('Recording has no usable input: '+r['sample_id'])
                    k=len(points)
                    for ds in (data,labels,marks): ds.resize(n+k,axis=0)
                    data[n:n+k]=np.stack(points);labels[n:n+k]=int(r['label_id']);marks[n:n+k]=mark
                    for row in rows:f.write(json.dumps(row)+'\n')
                    n+=k; complete_records+=1
                    progress=dict(state='running',scope=config['scope'],split=split,recordings=mark+1,total=len(rs),windows=n,elapsed_seconds=round(time.monotonic()-begun,1))
                    save(out/'status.json',progress)
                    if mark%25==0 or mark+1==len(rs): print(json.dumps(progress),flush=True)
                h.attrs['complete']=True
            coverage[split]=dict(recordings=len(rs),eligible_recordings=complete_records,valid_windows=n,empty_windows=empty,excluded_recordings=[])
            save(out/'coverage.json',coverage)
        save(out/'status.json',dict(state='complete',scope=config['scope'],elapsed_seconds=round(time.monotonic()-begun,1)))
    except BaseException as e:
        save(out/'status.json',dict(state='failed',error=repr(e)))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--official-splits',type=Path,required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--smoke-per-split',type=int,default=0)
    args=p.parse_args()
    if args.smoke_per_split<0:p.error('negative smoke size')
    prepare(args)
