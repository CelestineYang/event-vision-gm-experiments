# DailyDVS-200 preprocessing v1

This is a new dataset protocol, not an EventMamba reproduction. Official source: https://github.com/QiWang233/DailyDVS-200/tree/main/train_val_test . The downloaded list hashes and exact comparison are in `split_audit.json`.

- Official Train: 14,773 recordings. Official Test: 4,093 recordings. Labels 0–199. Never reassign recordings or split windows across sets.
- User-selected model selection: Test video NLL, with arithmetic mean of window probabilities per recording. No independent held-out Test claim. Official Val is unused; its current list contains 3,735 entries and overlaps Test by 555 filenames.
- Window 1,000,000 us, step 500,000 us, half-open `[start,end)`. Subtract the first event timestamp in int64 before conversion to floating point. Use regular windows and an additional end-aligned window if needed to cover the last event; short recordings use one nominal 1 s window. No maximum-window truncation.
- Retain all nonempty windows regardless of event count, for both splits. Empty temporal windows remain in the manifest with `valid=false` and in coverage reports. Empty recordings or invalid source arrays fail preprocessing for explicit investigation; never silently remove a Test recording.
- Sensor coordinates: width 320, height 240. No spatial filtering, rotation or Open3D removal. Full-event gCSTR rendering uses every event in the exact indexed slice.
- Base point cache: 2,048 points, uniform random sample without replacement where possible; with replacement for short windows. Seed is derived from global seed 0, sample ID and window ID. Sort selected indices in timestamp order. Coordinates `(t-start)/1e6, x/320, y/240`, float32; polarity is retained in event sources but omitted from the three point coordinates.
- Future fusion training must augment the shared event window first, then sample points and render gCSTR from that same augmented window. The saved HDF5 points are unaugmented base inputs, not an implementation of the future fusion augmentation pipeline.

Source NPZ files are referenced in place, with SHA-256 saved per recording. Do not move or mutate them after preparation. Output has config, samples, Train/Test window JSONL, Train/Test point HDF5, source checksums, coverage and atomic status files. No redundant full-event or rendered-frame cache is generated. HDF5 `mark` values index recordings within each split, not class IDs.

Implementation: `../thu_preprocessing/prepare_dailydvs200.py`. Existing `train_gcstr.Windows` supports the explicit `npz_reference` backend and holds only one decompressed recording per worker. Random-window training will decompress NPZ on cache misses; training throughput should be benchmarked before choosing a loader/cache strategy. No formal training is launched by preprocessing.

Environment: `/root/anaconda3/envs/torch201/bin/python`. Preprocessing uses CPU; CUDA_VISIBLE_DEVICES=3 reserves GPU visibility for checks only. The smoke checks all windows for coverage, label/point alignment and finite gCSTR frames, then executes one optimization step for each view on GPU 3 (random model initialization, not an accuracy experiment).

Outputs: `/remote-home/datasets/DailyDVS-200-prepared/{smoke_v1,prepared_v1}`. Full log: `/remote-home/datasets/DailyDVS-200-prepared/preprocess_v1.log`.

## Four GPU migration

User authorized migration at the saved epoch5 boundary. Controller: `switch_gcstr_ddp4.py`; live status: `/remote-home/datasets/DailyDVS-200-prepared/ddp4_migration/switch_status.json`. Source `gcstr_run1` is preserved. Target `gcstr_run1_ddp4`, log `gcstr_run1_ddp4.log`. Frozen trainer code `gcstr_ddp4_code`. Global batch remains 128 (32/rank), LR 1e-4, 50 total epochs; workers change 4 total to 2/rank, BN local to SyncBatchNorm. Adam states, three best weights, predictions and historical metrics are carried forward. Rank0 RNG is restored and new ranks receive deterministic seeds; not bitwise continuation. Failed startup triggers original single-GPU resume. Controller reports switched only after first new training step. Benchmark: single6.73 vs four31.32 windows/s, 1792 measured windows after two warmup batches; short shared-resource measurement, not a full-epoch ETA. Four training regression and three migration validation tests passed. See `ddp4_migration_report.json`.
