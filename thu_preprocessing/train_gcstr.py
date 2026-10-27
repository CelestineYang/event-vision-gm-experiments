"""Three independent event-image ResNet-18 branches sharing full-event windows."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from prepare_chl import sha, write_json
from render_frames import render
from gcstr_representations import GCSTRConfig
from gcstr_augmentation import augment_events
from gcstr_models import GCSTRBranchClassifier

VIEWS = ('cstr3', 'xt_p', 'ty_p')


def distributed():
    return dist.is_available() and dist.is_initialized()


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


class GlobalBatchSampler:
    """Shard each global batch without padding or dropping real windows."""
    def __init__(self, size, batch_size, rank, world_size, seed):
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        if size < world_size:
            raise ValueError('Training needs at least one window per rank')

    def __iter__(self):
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed)).tolist()
        batches = [indices[i:i+self.batch_size] for i in range(0, self.size, self.batch_size)]
        # A tiny final global batch is merged so every rank participates in backward.
        if len(batches) > 1 and len(batches[-1]) < self.world_size:
            batches[-2].extend(batches.pop())
        for batch in batches:
            yield batch[self.rank::self.world_size]

    def __len__(self):
        n = (self.size + self.batch_size - 1) // self.batch_size
        return n - int(n > 1 and 0 < self.size % self.batch_size < self.world_size)


def rng_state():
    return dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def gather(value):
    if not distributed(): return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


class Windows(Dataset):
    def __init__(self, store, split, seed=0):
        self.store, self.split, self.seed, self.epoch = Path(store), split, seed, 0
        if json.loads((self.store/'status.json').read_text())['state'] != 'complete':
            raise ValueError('Preprocessing not complete')
        self.rows = [r for line in (self.store/(split+'_windows.jsonl')).read_text().splitlines()
                     if (r := json.loads(line))['valid']]
        if not self.rows: raise ValueError('Empty window split')
        metadata=json.loads((self.store/'config.json').read_text())
        self.sensor=GCSTRConfig(sensor_width=metadata.get('width',346),sensor_height=metadata.get('height',260))
        self.classes=metadata.get('classes',50)
        if any(not 0<=r['label']<self.classes for r in self.rows):raise ValueError('Invalid class label')
        self.opened = OrderedDict()

    def __len__(self): return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy(); state['opened'] = OrderedDict(); return state

    def __getitem__(self, index):
        import hashlib
        row = self.rows[index]
        name = row['event_file']
        if name not in self.opened:
            if row.get('event_backend') == 'npz_reference':
                from prepare_dailydvs200 import load_events
                # Keep decompressed recording memory bounded per data worker.
                self.opened.clear()
                self.opened[name], _ = load_events(self.store/name)
            else:
                self.opened[name] = np.load(self.store/name, mmap_mode='r', allow_pickle=False)
                if len(self.opened)>8: self.opened.popitem(last=False)
        self.opened.move_to_end(name)
        events = self.opened[name][row['left']:row['right']]
        if self.split == 'train':
            seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["sample_id"]}:{row["window_id"]}'.encode()).digest()[:8], 'little')
            t,x,y,p = augment_events(events[:,2], events[:,0], events[:,1], events[:,3],
                                    mode='spatial', rng=np.random.default_rng(seed),
                                    config=self.sensor)
            events = np.column_stack([x,y,t,p])
        views = render(events,row['start_us'],row['end_us'],self.sensor)
        mean=np.array([.485,.456,.406],dtype='f4')[:,None,None]
        std=np.array([.229,.224,.225],dtype='f4')[:,None,None]
        return {v:torch.from_numpy(np.asarray((views[v]-mean)/std,dtype='f4')) for v in VIEWS}, row['label'], row['sample_id']


class Metrics:
    def __init__(self):
        self.n=0;self.nll=0.;self.top1=0;self.top5=0;self.videos={}

    def update(self, logits, labels, ids):
        lp=logits.detach().double().log_softmax(-1).cpu();labels=labels.cpu()
        self.n+=len(labels);self.nll+=float(-lp.gather(1,labels[:,None]).sum())
        self.top1+=int((lp.argmax(-1)==labels).sum())
        self.top5+=int((lp.topk(min(5,lp.shape[1]),dim=-1).indices==labels[:,None]).any(-1).sum())
        for value,label,name in zip(lp,labels,ids):
            label=int(label)
            if name not in self.videos:self.videos[name]=[value.clone(),label,1]
            else:
                total,old,count=self.videos[name]
                if old!=label:raise ValueError('Video label changed')
                self.videos[name]=[torch.logaddexp(total,value),label,count+1]

    def merge(self, other):
        self.n += other.n; self.nll += other.nll
        self.top1 += other.top1; self.top5 += other.top5
        for name, (total, label, count) in other.videos.items():
            if name not in self.videos: self.videos[name] = [total.clone(), label, count]
            else:
                old, old_label, old_count = self.videos[name]
                if old_label != label: raise ValueError('Video label changed across ranks')
                self.videos[name] = [torch.logaddexp(old, total), label, old_count + count]

    def finish(self):
        predictions=[]
        for name,(total,label,count) in sorted(self.videos.items()):
            lp=total-np.log(count)
            predictions.append(dict(sample_id=name,label=label,prediction=int(lp.argmax()),
                                    nll=float(-lp[label]),top5=bool((lp.topk(min(5,len(lp))).indices==label).any()),windows=count))
        return dict(windows=self.n,videos=len(predictions),window_nll=self.nll/self.n,
                    window_top1=self.top1/self.n,window_top5=self.top5/self.n,
                    video_nll=float(np.mean([r['nll'] for r in predictions])),
                    video_top1=float(np.mean([r['prediction']==r['label'] for r in predictions])),
                    video_top5=float(np.mean([r['top5'] for r in predictions]))),predictions


def epoch(models, loader, device, optimizers=None, epoch_number=None):
    metrics={v:Metrics() for v in VIEWS}
    for model in models.values():model.train(optimizers is not None)
    with torch.set_grad_enabled(optimizers is not None):
        for step,(images,labels,ids) in enumerate(loader,1):
            labels=labels.to(device)
            scale = 1.0
            if optimizers and distributed():
                count = torch.tensor(float(len(labels)), device=device)
                dist.all_reduce(count)
                scale = len(labels) * dist.get_world_size() / count.item()
            for view in VIEWS:
                if optimizers:optimizers[view].zero_grad(set_to_none=True)
                logits=models[view](images[view].to(device,non_blocking=True))
                loss=torch.nn.functional.cross_entropy(logits,labels)
                if not torch.isfinite(loss):raise FloatingPointError(view)
                if optimizers:(loss * scale).backward();optimizers[view].step()
                metrics[view].update(logits,labels,ids)
            if epoch_number is not None and (not distributed() or dist.get_rank()==0) and (step==1 or step%10==0 or step==len(loader)):
                print(json.dumps(dict(epoch=epoch_number,phase='train' if optimizers else 'test',step=step,steps=len(loader))),flush=True)
    combined = {v:Metrics() for v in VIEWS}
    for rank_metrics in gather(metrics):
        for v in VIEWS: combined[v].merge(rank_metrics[v])
    return {v:m.finish() for v,m in combined.items()}


def save(path, value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def load_frozen_branch(path, device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=GCSTRBranchClassifier(saved['view'],num_classes=saved['config'].get('classes',50),pretrained=False)
    model.load_state_dict(saved['model_state'],strict=True)
    return model.to(device).requires_grad_(False).eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=128,help='Global batch size across all ranks')
    p.add_argument('--workers',type=int,default=4,help='DataLoader workers per rank')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--test-as-val',action='store_true',required=True,
                   help='Explicitly acknowledge original Test selects video-NLL checkpoint')
    p.add_argument('--smoke',action='store_true',help='Two complete videos per split, one epoch, random initialization')
    args=p.parse_args()
    world = int(os.environ.get('WORLD_SIZE','1')); rank = int(os.environ.get('RANK','0'))
    if world > 1:
        if args.batch_size % world or args.batch_size < 2 * world:
            p.error('Global batch size must be divisible by world size and at least 2 per rank')
        if args.device.startswith('cuda'):
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank); args.device = f'cuda:{local_rank}'
        dist.init_process_group(backend='nccl' if args.device.startswith('cuda') else 'gloo')
    if args.batch_size<2 or args.epochs<1 or args.workers<0 or args.lr<=0:p.error('Invalid budget')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    status=json.loads((args.store/'status.json').read_text())
    if status.get('scope')!='full' and not args.smoke:raise ValueError('Full training requires full preprocessing')
    data={s:Windows(args.store,s,args.seed) for s in ('train','test')}
    if {r['sample_id'] for r in data['train'].rows}&{r['sample_id'] for r in data['test'].rows}:
        raise ValueError('Cross-split video leakage')
    if args.smoke:
        args.epochs=1
        for ds in data.values():
            selected=sorted({r['sample_id'] for r in ds.rows})[:2]
            ds.rows=[r for r in ds.rows if r['sample_id'] in selected]
    identity={f:sha(args.store/f) for f in ('config.json','samples.json','train_windows.jsonl','test_windows.jsonl')}
    code={f:sha(Path(__file__).parent/f) for f in ('train_gcstr.py','render_frames.py','gcstr_models.py','gcstr_augmentation.py','gcstr_representations.py')}
    config=dict(epochs=args.epochs,batch_size=args.batch_size,workers=args.workers,lr=args.lr,seed=args.seed,
                world_size=world,batch_norm='sync' if world>1 and args.device.startswith('cuda') else 'local',
                classes=50,optimizer='Adam',weight_decay=0.,scheduler=None,early_stopping=False,
                augmentation='spatial',pretrained=not args.smoke,smoke=args.smoke,identity=identity,code=code,
                selection_protocol='original_test_as_val',independent_test=False,selection_metric='video_nll',
                video_vote='arithmetic mean of window probabilities',window_selection='valid official Point windows')
    config.update(classes=data['train'].classes,width=data['train'].sensor.sensor_width,height=data['train'].sensor.sensor_height)
    if data['test'].classes!=config['classes']:raise ValueError('Split class count mismatch')
    checkpoint=args.output/'resume.pt'
    restored=torch.load(checkpoint,map_location='cpu',weights_only=False) if args.resume else None
    if restored and restored['config']!=config:raise ValueError('Resume config/data/code mismatch')
    if not args.resume and rank == 0:
        args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'config.json',config)
    if distributed(): dist.barrier()
    # Only rank zero downloads ImageNet weights; DDP broadcasts initial parameters.
    models={v:GCSTRBranchClassifier(v,config['classes'],pretrained=not args.smoke and not args.resume and rank==0).to(args.device) for v in VIEWS}
    if world > 1 and args.device.startswith('cuda'):
        models={v:torch.nn.SyncBatchNorm.convert_sync_batchnorm(m) for v,m in models.items()}
    optimizers={v:torch.optim.Adam(m.parameters(),lr=args.lr) for v,m in models.items()}
    best={v:float('inf') for v in VIEWS};start=1
    if restored:
        for v in VIEWS:
            models[v].load_state_dict(restored['models'][v]);optimizers[v].load_state_dict(restored['optimizers'][v])
        best=restored['best'];start=restored['epoch']+1
    if distributed():
        models={v:DDP(m,device_ids=[torch.cuda.current_device()] if args.device.startswith('cuda') else None) for v,m in models.items()}
    if restored:
        state=restored['rng_states'][rank]
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if torch.cuda.is_available() and state['cuda_rng'] is not None:torch.cuda.set_rng_state(state['cuda_rng'])
    for number in range(start,args.epochs+1):
        begin=time.perf_counter();data['train'].epoch=number
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        def loader(split):
            options=dict(num_workers=args.workers,pin_memory=args.device.startswith('cuda'),
                         generator=torch.Generator().manual_seed(args.seed+number),
                         **({'multiprocessing_context':'spawn'} if args.workers else {}))
            if split=='train':
                return DataLoader(data[split],batch_sampler=GlobalBatchSampler(
                    len(data[split]),args.batch_size,rank,world,args.seed+number),**options)
            return DataLoader(data[split],batch_size=args.batch_size//world,
                              sampler=list(range(rank,len(data[split]),world)),**options)
        trained=epoch(models,loader('train'),args.device,optimizers,epoch_number=number)
        # Eval shards may have unequal batch counts; bypass DDP forward collectives.
        if distributed():
            for model in models.values():
                for buffer in unwrap(model).buffers(): dist.broadcast(buffer,src=0)
        tested=epoch({v:unwrap(m) for v,m in models.items()},loader('test'),args.device,epoch_number=number)
        states=gather(rng_state())
        peak=torch.tensor(torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,device=args.device)
        if distributed(): dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if rank != 0:
            dist.barrier()
            continue
        for v in VIEWS:
            folder=args.output/v;folder.mkdir(exist_ok=True)
            record=dict(epoch=number,train=trained[v][0],test=tested[v][0],seconds=time.perf_counter()-begin,
                        peak_cuda_bytes=int(peak.item()))
            saved=dict(epoch=number,view=v,model_state=unwrap(models[v]).state_dict(),config=config,metrics=record)
            save(folder/'last.pt',saved)
            if tested[v][0]['video_nll']<best[v]:
                best[v]=tested[v][0]['video_nll'];save(folder/'best_loss.pt',saved)
                write_json(folder/'predictions_best.json',tested[v][1])
            with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        save(checkpoint,dict(epoch=number,config=config,best=best,models={v:unwrap(m).state_dict() for v,m in models.items()},
              optimizers={v:o.state_dict() for v,o in optimizers.items()},rng_states=states))
        write_json(args.output/'status.json',dict(state='complete' if number==args.epochs else 'running',epoch=number,best_video_nll=best))
        print(json.dumps(dict(epoch=number,test={v:tested[v][0] for v in VIEWS},seconds=time.perf_counter()-begin)),flush=True)
        if distributed(): dist.barrier()
    if distributed(): dist.destroy_process_group()


if __name__=='__main__':main()
