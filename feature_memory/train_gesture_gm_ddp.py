"""Video-lane DDP: one synchronized forward/backward per TBPTT chunk.

Global schedule is partitioned by video lanes, never by windows. Empty ranks
participate in reduction without encoding fake examples. Evaluation merges
probability sums and checks exact manifest coverage across ranks.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from gcstr_feature_memory_fusion import detach_state, sha256
from gesture_fusion import load_model
from gcstr_video_sequence_dataset import make_schedule
from gesture_sequence_dataset import GestureVideoSequenceDataset as VideoSequenceDataset
from train_gesture_gm import (configuration, identity, seed_all,
    atomic_save, write_json, emit, objective_sum, PredictionMetrics)


def rng_state():
    # A rank must not initialize CUDA contexts on the other three devices.
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


class LaneDataset(Dataset):
    def __init__(self, base, rank, world):
        self.base, self.rank, self.world = base, rank, world
        self.width = base.config['video_batch_size_per_gpu'] // world
        assert self.width * world == base.config['video_batch_size_per_gpu']

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        entry = self.base.schedule[index]
        videos = entry['videos'][self.rank*self.width:(self.rank+1)*self.width]
        batches = []
        for t in range(entry['start'], entry['stop']):
            samples = [self.base.window(self.base.groups[v][t]) if t < len(self.base.groups[v]) else None
                       for v in videos]
            template = next((x for x in samples if x is not None), None)
            if template is None:
                continue
            for i, sample in enumerate(samples):
                if sample is None:
                    samples[i] = {k: torch.zeros_like(v) if torch.is_tensor(v) else
                                  '' if isinstance(v, str) else False if isinstance(v, bool) else 0
                                  for k, v in template.items()}
            batches.append(default_collate(samples))
        return dict(batches=batches, group_start=entry['group_start'], group_end=entry['group_end'])


class ChunkModel(nn.Module):
    def __init__(self, model, auxiliary):
        super().__init__()
        self.model, self.auxiliary = model, auxiliary

    def forward(self, batches, state):
        # Anchor only a normally-used parameter for ranks with no real windows.
        loss = next(p for p in self.model.parameters() if p.requires_grad).reshape(-1)[0] * 0
        records = []
        for batch in batches:
            output, state = self.model.forward_window(batch, state)
            valid = batch['valid']
            local = {k: output[k][valid] for k in ('logits', 'point_logits')}
            labels = batch['label'][valid]
            loss = loss + objective_sum(local, labels, self.auxiliary)
            ids = [v for v, keep in zip(batch['video_id'], valid.tolist()) if keep]
            wins = [v for v, keep in zip(batch['window_id'], valid.tolist()) if keep]
            records.append((local['logits'].detach(), local['point_logits'].detach(), labels, ids, wins))
            ended = batch['video_end'] & valid
            state = {k: v.masked_fill(ended.reshape(-1, *([1]*(v.ndim-1))), 0) for k, v in state.items()}
        return loss, detach_state(state) if state is not None else None, records


def merge_metrics(payloads, expected):
    merged = {k: PredictionMetrics() for k in ('main', 'point')}
    seen = set()
    for payload in payloads:
        if seen.intersection(payload['seen']):
            raise RuntimeError('Duplicate target across DDP ranks')
        seen.update(payload['seen'])
        for key, source in payload['metrics'].items():
            target = merged[key]
            for attr in ('windows', 'ce', 'correct', 'top5'):
                setattr(target, attr, getattr(target, attr)+getattr(source, attr))
            if set(target.videos).intersection(source.videos):
                raise RuntimeError('A video crossed rank boundaries')
            target.videos.update(source.videos)
    if seen != expected:
        raise RuntimeError('DDP global coverage mismatch')
    results, predictions = {}, {}
    for key, metric in merged.items():
        results[key], predictions[key] = metric.finish()
    return results, predictions


def run_pass(ddp, base, device, cfg, optimizer=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    training = optimizer is not None
    ddp.train(training)
    loader = DataLoader(LaneDataset(base, rank, world), batch_size=None, num_workers=cfg['workers'],
                        pin_memory=True, generator=torch.Generator().manual_seed(cfg['seed']+base.epoch+rank))
    metrics = {k: PredictionMetrics() for k in ('main', 'point')}
    seen, state = set(), None
    started = time.perf_counter()
    for step, chunk in enumerate(loader):
        if chunk['group_start']:
            state = None
        batches = [{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
                   for b in chunk['batches']]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, state, records = (ddp if training else ddp.module)(batches, state)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite DDP loss')
            if training:
                # DDP averages gradients. Recover global sum with the same
                # epoch-wide equal-window objective as the single-GPU trainer.
                (loss * (world * len(base)/base.windows)).backward()
                nn.utils.clip_grad_norm_(ddp.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
        for logits, point, labels, ids, wins in records:
            pairs = set(zip(ids, wins))
            if len(pairs) != len(ids) or seen.intersection(pairs):
                raise RuntimeError('Repeated local target')
            seen.update(pairs)
            metrics['main'].update(logits, labels, ids)
            metrics['point'].update(point, labels, ids)
        if chunk['group_end']:
            state = None
        if rank == 0 and (step == 0 or (step+1) % 100 == 0 or step+1 == len(base)):
            emit('progress', phase='train_online_ddp' if training else 'fixed_eval_ddp', epoch=base.epoch,
                 chunks=step+1, total_chunks=len(base), rank0_windows=len(seen), seconds=time.perf_counter()-started)
    payloads = [None]*world if rank == 0 else None
    dist.gather_object(dict(metrics=metrics, seen=seen), payloads, dst=0)
    result, predictions = None, None
    if rank == 0:
        expected = {(base.rows[i]['video_id'], str(base.rows[i]['window_id'])) for indices in base.groups.values() for i in indices}
        result, predictions = merge_metrics(payloads, expected)
        elapsed = time.perf_counter()-started
        result.update(seconds=elapsed, windows_per_second=base.windows/elapsed, steps=len(base) if training else 0)
    return result, predictions


def data_for(cfg, split, epoch, smoke):
    data = VideoSequenceDataset(cfg, split, epoch)
    if smoke:
        # One full global batch plus a singleton; full recordings remain intact.
        count=cfg['video_batch_size_per_gpu']+1
        videos=[v for v in sorted(data.groups) if len(data.groups[v])>=5][:count]
        if len(videos)!=count:raise ValueError('Not enough long Train videos for smoke')
        data.groups={v:data.groups[v] for v in videos}
        data.windows=sum(map(len,data.groups.values()))
        data.position={i:(j,indices) for indices in data.groups.values() for j,i in enumerate(indices)}
        data.schedule=make_schedule(data.groups,cfg['video_batch_size_per_gpu'],cfg['tbptt_steps'],cfg['seed'],epoch,split=='train')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--migrate', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    device = torch.device('cuda', local)
    cfg = configuration(args.config)
    free,total=torch.cuda.mem_get_info()
    budget=min(free-2*1024**3,int(total*cfg['allocator_fraction']))
    if budget<8*1024**3:raise RuntimeError('Insufficient safe shared-GPU memory budget')
    torch.cuda.set_per_process_memory_fraction(budget/total,local)
    assert cfg['mode'] == 'GM' and world == cfg['ddp_world_size']
    global_cfg = dict(cfg, video_batch_size_per_gpu=cfg['video_batch_size_per_gpu']*world)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        lock = (args.output/'run.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume and any(p.name != 'run.lock' for p in args.output.iterdir()):
            raise FileExistsError('Refusing nonempty output')
    dist.barrier()
    seed_all(cfg['seed'])
    model = load_model(cfg).to(device)
    groups, _ = model.parameter_groups(cfg)
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    train_data = data_for(global_cfg, 'train', 1, args.smoke)
    run_identity = identity(cfg, train_data, model)
    run_identity['ddp_code'] = sha256(Path(__file__))
    start, total_steps, best = 1, 0, dict(top1=-1., epoch=0)
    if args.migrate and args.resume:
        raise ValueError('Migration and resume are mutually exclusive')
    if args.migrate or args.resume:
        path = args.output/'last.pt' if args.resume else args.migrate
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if args.resume:
            assert saved['identity'] == run_identity and saved['smoke'] == args.smoke
        else:
            assert not saved['smoke'] and saved['identity']['config']['mode'] == 'GM'
            assert saved['identity'] == identity(saved['identity']['config'], train_data, model)
            assert saved['data_epoch'] == saved['epoch']
            for k, v in saved['identity']['config'].items():
                # Explicit migration may change batching; all other original
                # model, data and optimization settings must remain identical.
                if k != 'video_batch_size_per_gpu':
                    assert cfg[k] == v, k
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        start, total_steps, best = saved['epoch']+1, saved['total_steps'], saved['best']
        if args.resume:
            restore_rng(saved['rng_states'][rank])
        else:
            seed_all(cfg['seed']+start*1000+rank)
        del saved
    else:
        seed_all(cfg['seed']+rank)
    wrapper = ChunkModel(model, cfg['point_aux_weight'])
    ddp = DDP(wrapper, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True,
              gradient_as_bucket_view=True)
    if rank == 0 and not args.resume:
        write_json(args.output/'config.json', dict(identity=run_identity, global_video_batch=global_cfg['video_batch_size_per_gpu'],
            smoke=args.smoke, migration=str(args.migrate) if args.migrate else None,
            migration_rng='independent rank seeds; not bitwise single-GPU continuation',
            historical_best_directory=str(args.migrate.parent) if args.migrate else None))
    limit = start if args.smoke else cfg['epochs']
    for epoch in range(start, limit+1):
        train_data = data_for(global_cfg, 'train', epoch, args.smoke)
        torch.cuda.reset_peak_memory_stats()
        train, _ = run_pass(ddp, train_data, device, cfg, optimizer)
        total_steps += len(train_data)
        previous = rng_state()
        seed_all(cfg['seed']+100000+rank)
        eval_data = data_for(global_cfg, 'train' if args.smoke else 'test', 0, args.smoke)
        evaluation, predictions = run_pass(ddp, eval_data, device, cfg)
        restore_rng(previous)
        rngs, peaks = [None]*world, [None]*world
        dist.all_gather_object(rngs, rng_state())
        dist.all_gather_object(peaks, dict(rank=rank, allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                          reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        if rank == 0:
            improved = evaluation['main']['video_top1'] > best['top1']
            if improved:
                best = dict(top1=evaluation['main']['video_top1'], epoch=epoch)
            item = dict(epoch=epoch, train=train, evaluation=evaluation, total_steps=total_steps, best=best, memory=peaks)
            checkpoint = dict(identity=run_identity, smoke=args.smoke, epoch=epoch, data_epoch=epoch,
                model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), rng_states=rngs,
                best=best, total_steps=total_steps, metrics=item)
            atomic_save(args.output/'last.pt', checkpoint)
            if improved:
                atomic_save(args.output/'best_top1.pt', checkpoint)
                write_json(args.output/'predictions_best.json', predictions)
            write_json(args.output/'predictions_last.json', predictions)
            with (args.output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(item, allow_nan=False)+'\n')
            write_json(args.output/'status.json', dict(completed_epoch=epoch, total_epochs=cfg['epochs'],
                status='smoke_passed' if args.smoke else 'complete' if epoch == limit else 'running', best=best))
            emit('epoch', **item)
        dist.barrier()
    if rank == 0:
        emit('finished', epoch=limit, smoke=args.smoke)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
