"""Evaluate frozen checkpoints on every test recording, retaining existing windows."""
import argparse
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from train_gcstr import Windows, VIEWS, load_frozen_branch, epoch
from prepare_chl import sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    ds=Windows(args.store,'test')
    records=[r for r in json.loads((args.store/'split_records.json').read_text()) if r['split']=='test']
    expected={r['sample_id']:r['label'] for r in records}
    original={r['sample_id'] for r in ds.rows}
    assert original <= expected.keys()
    added=[]
    for r in records:
        if r['sample_id'] in original:continue
        e=np.load(args.store/r['event_file'],mmap_mode='r')
        if not len(e) or e[-1,2]<=e[0,2]:raise ValueError('Empty/zero-duration recording: '+r['sample_id'])
        if e[-1,2]-e[0,2]>1500000:raise ValueError('Missing recording is not short: '+r['sample_id'])
        row=dict(**r,window_id=0,start_us=int(e[0,2]),end_us=int(e[-1,2]),left=0,right=len(e),
            raw_events=len(e),valid=True,point_row=None,fallback='entire short recording; actual time span; no event-count exclusion')
        added.append(row)
    ds.rows.extend(added)
    assert {r['sample_id'] for r in ds.rows}==expected.keys()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'status.json',dict(state='running'))
    write_json(args.output/'protocol.json',dict(original_store=str(args.store),
        split_sha256=sha(args.store/'split_records.json'),original_windows_sha256=sha(args.store/'test_windows.jsonl'),
        weights={v:sha(args.checkpoints/v/'best_loss.pt') for v in VIEWS},
        original_recordings=len(original),added_recordings=len(added),test_recordings=len(expected),
        fallback='all cached events; [first,last] actual duration mapped to time bins; no padding',
        selection='existing subset-test-selected best_loss checkpoints, no reselection',independent_test=False))
    write_json(args.output/'added_windows.json',added)
    with (args.output/'test_windows.jsonl').open('w') as f:
        for r in ds.rows:f.write(json.dumps(r)+'\n')
    models={v:load_frozen_branch(args.checkpoints/v/'best_loss.pt','cuda:0') for v in VIEWS}
    loader=DataLoader(ds,batch_size=32,num_workers=1,pin_memory=True,multiprocessing_context='spawn')
    results=epoch(models,loader,'cuda:0',epoch_number=0)
    summary={}
    for v,(metrics,predictions) in results.items():
        assert len(predictions)==len(expected)
        assert {r['sample_id']:r['label'] for r in predictions}==expected
        previous={r['sample_id']:r for r in json.loads((args.checkpoints/v/'predictions_best.json').read_text())}
        unchanged=[r for r in predictions if r['sample_id'] in original]
        if any(r['prediction']!=previous[r['sample_id']]['prediction'] or abs(r['nll']-previous[r['sample_id']]['nll'])>1e-4 for r in unchanged):
            raise ValueError('Original-subset predictions changed beyond numerical tolerance: '+v)
        extra=[r for r in predictions if r['sample_id'] not in original]
        summary[v]=dict(metrics=metrics,correct=sum(r['prediction']==r['label'] for r in predictions),
            added_correct=sum(r['prediction']==r['label'] for r in extra),added_total=len(extra),
            original_predictions_verified=True)
        write_json(args.output/(v+'_predictions.json'),predictions)
    write_json(args.output/'results.json',summary)
    write_json(args.output/'status.json',dict(state='complete'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
