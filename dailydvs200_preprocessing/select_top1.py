"""Preserve exact historical video-Top1 winners without modifying live trainers."""
import argparse,fcntl,json,os,shutil,time
from pathlib import Path
import torch

BASE=Path('/remote-home/datasets')
DAILY=BASE/'DailyDVS-200-prepared'
RUNS=[('UCF101 GM',BASE/'UCF101-feature-memory-runs/GM_ddp3_batch48_20261024T144614Z/training',[],True),
 ('THU-CHL GM',BASE/'THU-EACT-50-CHL-prepared/gm_ddp3_batch48_v1/training',[],True)]
RUNS += [('DailyDVS-200 '+v,DAILY/'gcstr_run1_ddp4'/v,[DAILY/'gcstr_run1'/v],False) for v in ('cstr3','xt_p','ty_p')]

def atomic_json(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def score(row,gm):return row['evaluation']['main'] if gm else row['test']

def reconcile(name,folder,archives,gm):
 rows=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines() if l.strip()]
 maximum=max(score(r,gm)['video_top1'] for r in rows)
 winners=[r for r in rows if score(r,gm)['video_top1']==maximum]
 target=folder/'best_top1.pt';selected=None
 candidates=[target,folder/'best_video_nll.pt',folder/'best_loss.pt',folder/'last.pt']
 for a in archives:candidates.extend([a/'best_loss.pt',a/'last.pt'])
 for p in candidates:
  if not p.exists():continue
  saved=torch.load(p,map_location='cpu',weights_only=False)
  metrics=saved['metrics'];m=score(metrics,gm)
  if m['video_top1']==maximum and saved['epoch'] in [r['epoch'] for r in winners]:
   selected={'epoch':saved['epoch'],'source':str(p)}
   if p!=target:
    # Copy the already read snapshot, avoiding a race with atomic last.pt replacement.
    temp=target.with_suffix('.tmp');torch.save(saved,temp);temp.replace(target)
   del saved;break
  del saved
 row=winners[0];m=score(row,gm)
 result=dict(experiment=name,selection_metric='video_top1',historical_best_epoch=row['epoch'],
   tied_epochs=[r['epoch'] for r in winners],video_top1=maximum,correct=round(maximum*m['videos']),videos=m['videos'],
   video_nll_at_first_top1_winner=m['video_nll'],last_logged_epoch=rows[-1]['epoch'],
   checkpoint_available=selected is not None,checkpoint=str(target) if selected else None,
   checkpoint_epoch=selected['epoch'] if selected else None,
   note='Historical maximum from logs; not a claim of checkpoint availability' if not selected else 'Exact winning weights preserved; legacy NLL checkpoints retained separately')
 atomic_json(folder/'selection_top1.json',result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
 lock=(DAILY/'top1_selector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signatures={};results={}
 while True:
  for name,folder,archives,gm in RUNS:
   p=folder/'metrics.jsonl';sig=(p.stat().st_mtime_ns,p.stat().st_size)
   if signatures.get(name)==sig:continue
   # The metrics append follows atomic model writes. A partial final line is retried.
   try:results[name]=reconcile(name,folder,archives,gm)
   except (json.JSONDecodeError,EOFError):continue
   signatures[name]=sig;print(json.dumps(results[name]),flush=True)
  atomic_json(DAILY/'historical_top1_summary.json',list(results.values()))
  if not args.watch:break
  status=json.loads((DAILY/'gcstr_run1_ddp4/status.json').read_text())
  if status['state']=='complete' and all(r['last_logged_epoch']==status['epoch'] for n,r in results.items() if n.startswith('Daily')):break
  time.sleep(1)

if __name__=='__main__':main()
