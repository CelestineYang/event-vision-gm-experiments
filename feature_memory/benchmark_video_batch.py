"""Isolated real Train window throughput/memory benchmark, no checkpoint writes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/remote-home/datasets/UCF101-feature-memory-runs')
WORK = Path(__file__).resolve().parent

def trial(mode, batch, output, realistic=False):
    import torch
    from gcstr_video_sequence_dataset import VideoSequenceDataset, make_schedule
    from gcstr_feature_memory_fusion import load_model
    from train_gcstr_feature_memory import configuration, seed_all, run_pass
    config = configuration(WORK / 'configs/feature_memory_target' / (mode + '.json'))
    config['video_batch_size_per_gpu'] = batch
    seed_all(config['seed'])
    data = VideoSequenceDataset(config, 'train', 1)
    if realistic:
        import numpy as np
        videos = sorted(data.groups)
        chosen = sorted(np.random.default_rng(0).choice(videos, size=96, replace=False).tolist())
        data.groups = {v: data.groups[v] for v in chosen}
    else:
        chosen = [v for v in sorted(data.groups) if len(data.groups[v]) >= 8][:16]
        assert len(chosen) == 16
        data.groups = {v: data.groups[v][:8] for v in chosen}
    data.windows = sum(map(len, data.groups.values()))
    data.schedule = make_schedule(data.groups, batch, 4, config['seed'], 1, True)
    data.position = {i: (j, indices) for indices in data.groups.values() for j, i in enumerate(indices)}
    model = load_model(config).cuda()
    groups, _ = model.parameter_groups(config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    if realistic:
        checkpoint = torch.load(ROOT/'formal_batch16_20261024T105557Z/GM/last.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        del checkpoint
    reports = []
    for repeat in range(2):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        metrics, _ = run_pass(model, data, torch.device('cuda:0'), config, optimizer)
        torch.cuda.synchronize()
        reports.append(dict(repeat=repeat, allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                            reserved_mib=torch.cuda.max_memory_reserved()/2**20, metrics=metrics))
    result = dict(mode=mode, batch=batch, tbptt=4, videos=chosen, windows=data.windows,
                  full_videos=realistic, chunks=len(data),
                  lane_utilization=data.windows/(len(data)*batch*4), reports=reports)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--realistic', action='store_true')
    args = parser.parse_args()
    if args.mode:
        trial(args.mode, args.batch, args.output, args.realistic)
    else:
        args.output.mkdir(exist_ok=False)
        trials = 0
        for batch in ((16, 24, 32) if args.realistic else (8, 16)):
            for mode in (('GM',) if args.realistic else ('B', 'G', 'GM')):
                name = f'{mode}_batch{batch}'
                (args.output/'status.json').write_text(json.dumps(dict(status='running', trial=name)))
                with (args.output/(name+'.log')).open('w') as log:
                    result = subprocess.run([sys.executable, str(Path(__file__)), '--mode', mode,
                        '--batch', str(batch), '--output', str(args.output/(name+'.json'))] +
                        (['--realistic'] if args.realistic else []),
                        cwd=WORK, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    (args.output/'status.json').write_text(json.dumps(dict(status='failed', trial=name)))
                    sys.exit(result.returncode)
                trials += 1
        (args.output/'status.json').write_text(json.dumps(dict(status='passed', trials=trials)))
