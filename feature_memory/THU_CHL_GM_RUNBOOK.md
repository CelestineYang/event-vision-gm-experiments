# THU-CHL GM adaptation

## Provenance and initialization

Official EventMamba source pinned to `7f61fd397cf69f10f0f92a182109d36cd7b6fdbb` in `eventmamba_v1_upstream/`. README configuration row says THU-CHA (download row says THU-CHL): V1, channels 64/128/256, groups 1024/512/256. The generic V1 source defaults to smaller groups; our separate model explicitly applies the THU README groups. Group counts are not encoded in the state dict, so strict loading alone cannot recover them. Use coordinate-distance KNN for V1 as explicitly required by the upstream README. V1 architecture contains BiMamba type v2 internally; these two version names refer to different things.

`thu_best_checkpoint.pth`: SHA-256 `5a7f82e8f2e1d0e855d1a84690d41b035ff0677a9b97943ee2ecac83a94b87e7`. Strictly restores all 168 state tensors, including the 50-class point classifier. Three exposed stages are [B,1024,64], [B,512,128], [B,256,256], with chronological (t,x,y) coordinates. The first embedding's 6 input channels come from grouping/concatenation; external input remains [B,3,2048].

Initialize gCSTR from THU `gcstr_run1/{cstr3,xt_p,ty_p}/best_loss.pt` (epochs 21/18/20). Strictly check source checkpoint hashes and dataset identity. Freeze image backbones, fine-tune EventMamba including its classifier, freeze point BN running statistics. Fusion layers initialize randomly; Gate residual projections and Memory readout projections initialize to zero. This is **not** UCF's migration from a pretrained epoch40 Fusion A model. No pretrained THU Fusion A weights are claimed.

The uploaded checkpoint contains no recording split/class-name manifest. Its THU provenance is user-supplied; architecture compatibility and matching log metrics do not independently prove its historical split. No claim to reproduce the historical 59.479554% sequence score has been made.

## Data

Reuse `THU-EACT-50-CHL-prepared/chl_prepared`, 346x260, original 1 s / 0.5 s EventMamba window boundaries. Train: 1,786 recordings, 7,129 originally eligible windows. Test: **544 recordings, 2,392 windows**, including the previously omitted 58 point-invalid windows. Train/Test IDs are disjoint. Test sparse windows are never dropped; zero-event windows cause an explicit error.

Each Train window: raw events -> video-consistent spatial augmentation -> both full-event gCSTR rendering and point preprocessing. Point preprocessing normalizes t/x/y for radius filtering (.01, nb_points=5), then samples 2048 points and sorts them chronologically. Final point normalization uses sampled time min/max, x/346, y/260. If fewer than 100 points survive the filter, sample the unfiltered **augmented** events and count a fallback. Test uses the same rule without augmentation. Zero sampled time span maps to zero time, rather than NaN. Point cache HDF5 is not reused during online fusion training.

Images use the exact THU historical `render_frames` implementation, copied into the separate `thu_chl_render.py`; do not substitute the other gCSTR whole-event normalization. Frames use all augmented events; the point radius filter only affects point sampling. Class label never enters preprocessing.

## Training entry and starting configuration

`train_thu_chl_gm.py` is an independent single-GPU entry derived from the existing trainer, using `thu_chl_fusion.py`, `thu_chl_sequence_dataset.py`, and separate `eventmamba_v1_*` files. Existing UCF and DailyDVS code/weights/processes were not changed.

Config: `configs/thu_chl_gm.json`: 50 epochs, FP32, AdamW, point LR 1e-5, random base fusion LR 5e-5, new Gate/Memory LR 1e-4, weight decay 1e-4, gradient clip 1, auxiliary point loss .5, TBPTT 4, 2 workers. Video batch 2 is a **conservative smoke starting value**, not a finalized throughput choice. Test video NLL selects checkpoints; video probabilities are averaged across all its windows. Test is not independent of model selection, per the user's established protocol.

Epoch-boundary checkpoints contain model, optimizer, RNG, configuration/data/code/source hashes. Smoke reads three complete Train videos only for model inputs; Test manifest identity may still be checked. Smoke verifies singleton tails, cross-chunk memory, original-base equality at initialization and saved weights. The copied trainer also supports `--resume` and `--stop-after-epoch`.

When a GPU becomes available, from `/remote-home/yanglansong`:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/remote-home/datasets/UCF101-feature-memory-runs/venv/bin/python -u \
feature_memory/train_thu_chl_gm.py \
--config feature_memory/configs/thu_chl_gm.json \
--output /remote-home/datasets/THU-EACT-50-CHL-prepared/gm_v1_smoke \
--device cuda:0 --smoke
```

The same command has now passed on physical GPU 2 (CUDA_VISIBLE_DEVICES=2), while existing jobs remained running. Check live GPU usage before rerunning it in a new output directory. Formal training should follow successful real CUDA smoke and a batch/throughput check, using a new output directory and without `--smoke`.

## Completed validation

- `thu_chl_cpu_tests.log`: four THU integration tests passed: strict real checkpoint loading; real input feature/logit parity with the retained official V1 forward; exact Train/Test scheduling; augmentation shared before both representations; all 58 restored Test windows usable; frozen/trainable parameter groups; source identity.
- CPU logit parity uses an explicit PyTorch conv plus selective_scan_ref substitution for the CUDA fused operator. It validates wiring and checkpoint use, **not CUDA kernel numerics or CUDA backward**. Production retains the real BiMamba implementation.
- `thu_chl_gm_regression.log`: 12 existing CPU Gate/Memory tests passed, including memory masking/reset and synthetic training/resume equivalence.
- `thu_chl_radius_audit.json`: scipy radius filter indices exactly matched Open3D 0.18 on three real windows (one sparse); no claim of exhaustive filter equivalence at all floating-point boundary cases.
- `thu_chl_identity_audit.json`: source/data/code fingerprints. `thu_checkpoint_audit.json` records the initial checkpoint inspection.

THU formal training has not been launched. After CPU adaptation, the user authorized use of spare GPU 0–2 memory. A real GPU 2 smoke completed: three Train recordings, 24 windows, four optimization steps, two continuing TBPTT chunks, eight singleton windows; peak allocated 1304.32 MiB and reserved 1592 MiB. Initial CUDA regression, finite losses/gradients and saved-checkpoint tensor checks passed. This is not a held-out accuracy measurement or a CUDA resume test. Existing UCF and DailyDVS jobs were not stopped; transient compute contention was possible. Artifacts: `/remote-home/datasets/THU-EACT-50-CHL-prepared/gm_v1_smoke`, log in sibling `gm_v1_smoke.log`.

## Shared-GPU batch16 benchmark

User requested UCF settings: video batch 16 per GPU, TBPTT 4, same AdamW parameter groups. `benchmark_thu_chl_ddp.py` ran on Train only (49 complete recordings / 270 windows), single GPU2 then DDP GPU0/1/2, two passes each. Second passes: single 47.218 s (5.718 windows/s); DDP 24.566 s (10.991 windows/s), 1.922x speedup. DDP peak allocated 8640.05 MiB/rank, reserved 9204 MiB/rank. Exact global window coverage checked, singleton/empty-rank tails completed. Shared GPUs retained the existing UCF job; this is not dedicated-resource throughput. Allocator cap left 2 GiB free at launch. No benchmark weights were saved and no formal THU run was started. Full results: `thu_chl_batch16_benchmark.json`.

## Formal THU DDP launch

User authorized formal training after the shared-GPU benchmark. Started `gm_ddp3_batch48_v1` on GPU 0/1/2 with `configs/thu_chl_gm_ddp3.json`, via independent `train_thu_chl_gm_ddp.py`. Fresh initialization from THU branch checkpoints; benchmark/smoke weights are not reused. World size 3, per-rank video batch 16, global 48, TBPTT 4, 50 epochs. Source files copied into run `code/` before launch. Outputs under `/remote-home/datasets/THU-EACT-50-CHL-prepared/gm_ddp3_batch48_v1/training`, log `train.log`, launcher PID in `pid`. This supersedes the earlier statement that formal training was not started. Existing UCF and DailyDVS processes continue; sharing compute can reduce their throughput.
