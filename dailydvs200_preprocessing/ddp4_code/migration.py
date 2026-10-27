"""Explicit epoch-boundary single-rank to four-rank migration checks."""
import json,math,shutil
from pathlib import Path
import torch

def validate(old,new):
    if old['world_size']!=1 or new['world_size']!=4:raise ValueError('Only 1->4 migration supported')
    if old['batch_norm']!='local' or new['batch_norm']!='sync':raise ValueError('Unexpected BN transition')
    allowed={'world_size','batch_norm','workers','code'}
    if set(old)!=set(new):raise ValueError('Configuration schema mismatch')
    for k in old:
        if k not in allowed and old[k]!=new[k]:raise ValueError('Migration changed '+k)
    for k in old['code']:
        if k!='train_gcstr.py' and old['code'][k]!=new['code'].get(k):raise ValueError('Representation/model changed: '+k)

def copy_history(source,target,saved):
    for view in saved['models']:
        src=source/view;dst=target/view;dst.mkdir()
        best=torch.load(src/'best_loss.pt',map_location='cpu',weights_only=False)
        if best['epoch']>saved['epoch'] or not math.isclose(best['metrics']['test']['video_nll'],saved['best'][view],rel_tol=1e-12):raise ValueError('Inconsistent best checkpoint')
        for name in ('best_loss.pt','predictions_best.json','last.pt'):shutil.copy2(src/name,dst/name)
        rows=[json.loads(l) for l in (src/'metrics.jsonl').read_text().splitlines() if l]
        (dst/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<=saved['epoch']))
