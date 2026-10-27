"""Four-process CPU numerical check: ragged lanes, empty rank, global gradients."""
import copy
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from test_gcstr_feature_memory import TinyFusion, TinySequence
from gcstr_feature_memory_fusion import FeatureMemoryFusion
from gcstr_video_sequence_dataset import make_schedule
from train_gesture_gm_ddp import ChunkModel, LaneDataset, merge_metrics
from train_gesture_gm import seed_all, PredictionMetrics

def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    seed_all(7)
    model = FeatureMemoryFusion(TinyFusion(), 'GM', seed=3)
    reference = ChunkModel(copy.deepcopy(model), .5)
    wrapped = DDP(ChunkModel(model, .5), find_unused_parameters=True, broadcast_buffers=False)
    data = TinySequence(1)
    data.config = {'video_batch_size_per_gpu':4}
    data.schedule = make_schedule(data.groups, 4, 2, 0, 1, True)
    lane = LaneDataset(data, rank, world)
    state, full_state = None, None
    seen = set()
    metrics = {'main':PredictionMetrics(), 'point':PredictionMetrics()}
    for step in range(len(data)):
        wrapped.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
        local = lane[step]
        loss, state, records = wrapped(local['batches'], state)
        (loss*world).backward()
        full_loss, full_state, _ = reference(data[step]['batches'], full_state)
        full_loss.backward()
        for (name,p),(_,q) in zip(wrapped.module.named_parameters(),reference.named_parameters()):
            if not p.requires_grad:
                continue
            if q.grad is None:
                assert p.grad is None or torch.count_nonzero(p.grad)==0, name
            else:
                assert p.grad is not None, name
                torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-5,msg=name)
        for logits,point,labels,ids,wins in records:
            seen.update(zip(ids,wins)); metrics['main'].update(logits,labels,ids); metrics['point'].update(point,labels,ids)
    payloads=[None]*world
    dist.all_gather_object(payloads,dict(seen=seen,metrics=metrics))
    expected={(r['video_id'],r['window_id']) for r in data.rows}
    result,_=merge_metrics(payloads,expected)
    assert result['main']['windows']==11 and result['main']['videos']==3
    if rank==0: print('PASS: four-rank gradients match single-process global loss; empty ranks, tails, cross-chunk state, exact 11-window coverage',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
