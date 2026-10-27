"""Re-evaluate frozen THU branches with raw-event test windows restored."""
import argparse,json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from train_gcstr import Windows,VIEWS,load_frozen_branch,epoch
from prepare_chl import sha,write_json


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--store',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();ds=Windows(args.store,'test')
 rows=[json.loads(l) for l in (args.store/'test_windows.jsonl').read_text().splitlines()]
 expected={r['sample_id']:r['label'] for r in json.loads((args.store/'samples.json').read_text()) if r['split']=='test'}
 added=[]
 for r in rows:
  if r['valid']:continue
  e=np.load(args.store/r['event_file'],mmap_mode='r')[r['left']:r['right']]
  if len(e)==0 or r['end_us']<=r['start_us'] or (e[:,2]<r['start_us']).any() or (e[:,2]>r['end_us']).any():
   raise ValueError('Non-renderable window: '+str(r))
  added.append(dict(r,valid=True,original_point_valid=False))
 ds.rows.extend(added)
 assert len(ds.rows)==len(rows)
 assert {r['sample_id'] for r in ds.rows}==expected.keys()
 affected={r['sample_id'] for r in added}
 args.output.mkdir(parents=True,exist_ok=False)
 write_json(args.output/'status.json',dict(state='running'))
 write_json(args.output/'protocol.json',dict(source_store=str(args.store),windows_sha256=sha(args.store/'test_windows.jsonl'),
  weights={v:sha(args.checkpoints/v/'best_loss.pt') for v in VIEWS},added_windows=len(added),affected_videos=len(affected),
  window_rule='all existing test windows; original raw slices and time boundaries; no point eligibility filter',
  selection='existing best_loss checkpoints; no retraining or reselection',independent_test=False))
 write_json(args.output/'added_windows.json',added)
 with (args.output/'test_windows.jsonl').open('w') as f:
  for r in ds.rows:f.write(json.dumps(r)+'\n')
 models={v:load_frozen_branch(args.checkpoints/v/'best_loss.pt','cuda:0') for v in VIEWS}
 loader=DataLoader(ds,batch_size=32,num_workers=2,pin_memory=True,multiprocessing_context='spawn')
 results=epoch(models,loader,'cuda:0',epoch_number=0);summary={}
 for v,(metrics,predictions) in results.items():
  assert len(predictions)==len(expected) and {r['sample_id']:r['label'] for r in predictions}==expected
  previous={r['sample_id']:r for r in json.loads((args.checkpoints/v/'predictions_best.json').read_text())}
  changes=[]
  discrepancies=[]
  for r in predictions:
   old=previous[r['sample_id']]
   if r['sample_id'] not in affected and (r['prediction']!=old['prediction'] or abs(r['nll']-old['nll'])>1e-4):
    discrepancies.append(dict(sample_id=r['sample_id'],before=old['prediction'],after=r['prediction'],nll_delta=r['nll']-old['nll']))
   if r['prediction']!=old['prediction']:changes.append(dict(sample_id=r['sample_id'],label=r['label'],before=old['prediction'],after=r['prediction'],before_windows=old['windows'],after_windows=r['windows']))
  summary[v]=dict(metrics=metrics,before_correct=sum(r['prediction']==r['label'] for r in previous.values()),
   after_correct=sum(r['prediction']==r['label'] for r in predictions),changed_predictions=len(changes),
   wrong_to_right=sum(r['before']!=r['label'] and r['after']==r['label'] for r in changes),
   right_to_wrong=sum(r['before']==r['label'] and r['after']!=r['label'] for r in changes),unaffected_verified=not discrepancies,unaffected_discrepancies=discrepancies)
  write_json(args.output/(v+'_predictions.json'),predictions);write_json(args.output/(v+'_changes.json'),changes)
 write_json(args.output/'results.json',summary);write_json(args.output/'status.json',dict(state='complete'))
 print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
