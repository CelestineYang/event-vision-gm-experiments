# Cross-server migration and ablations

All server paths below are references, not downloadable Git assets. Copy or mount
these assets separately; verify identity hashes instead of editing them away.

| Dataset | External data under `/remote-home/datasets/` | Initialization / evaluation requirements |
| --- | --- | --- |
| THU-CHL | `THU-EACT-50-CHL-prepared/chl_prepared/` | 3 `gcstr_run1/*/best_loss.pt` files plus `/remote-home/yanglansong/feature_memory/thu_best_checkpoint.pth`; all544 Test segments/2392 windows |
| DVS Gesture | `DVSGesture-prepared/gesture_prepared_v2/` | 3 `gcstr_run1/*/best_loss.pt`; `eventmamba_pretrained/uploaded_20261027/best_checkpoint(1).pth`; reset final point classifier; all264 Test segments/6959 windows |
| DailyAction | `DailyAction-DVS-prepared/daily_prepared/` and `splits_seed0/` | 3 gCSTR branches; `gcstr_run1_test288/test_windows.jsonl` and its protocol/added-window records preserve all288 Test recordings |
| DailyDVS-200 | `DailyDVS-200-prepared/prepared_v1/` PLUS referenced `Joind-dataset for Action Recognition/DailyDvs-200/processed/` | Window manifests refer to absolute external NPZ paths; copying prepared_v1 alone is insufficient |
| UCF101-DVS | `Joind-dataset for Action Recognition/ucf101_feature_memory_migration_20261024_v1/data/` | Same bundle's `weights/epoch_040.pt`; use bundle checksum inventory and migration tool; bundle's old handoff status is historical |

Preserve config.json, samples.json, Train/Test window manifests and status.json
alongside prepared event files. Gesture/THU use mmap NPY events in train/ and test/;
GM regenerates aligned views/point inputs from those events. Point H5 files alone
cannot support this shared-augmentation pipeline. Raw recordings and annotations
are additionally required to rerun preprocessing or the raw protocol audits.

THU, Gesture and UCF101 have GM adapters. DailyAction and DailyDVS-200 currently
have gCSTR experiments; this does not imply complete GM adapters or compatible
pretrained point encoders for them. HMDB was inspected but no formal experiment
was launched here.

## Fair ablations

For each dataset preserve the same base initialization, exact sample lists,
window protocol, class mapping, augmentations, training budget, global batch,
optimizer and selection rule across B/G/GM. Do not initialize an ablation from
an already-trained GM result. UCF uses Fusion A, whereas THU/Gesture start with
random fusion; do not call these identical initialization protocols.

Current run settings: UCF/THU three GPUs x16 =48 videos; Gesture two GPUs x32 =64;
TBPTT4,50 epochs, seed0, AdamW, FP32, weight decay1e-4, gradient clip1,
point auxiliary weight0.5, LR point1e-5/fusion5e-5/new modules1e-4. The fresh
Gesture final point layer belongs to new_modules. Image backbones and point BN
running statistics are frozen. Existing image source weights were NLL-selected;
historical maximum-Top1 image checkpoints were not retained.

The current protocol uses Test as validation for model selection, not an
independent held-out final selection set. New selection uses historical maximum
video Top1 (mean probabilities across windows); keep the earlier epoch on ties.
Older scripts' NLL checkpoint logic must be changed explicitly and tested.

Single-GPU configs accept B/G/GM. DDP training entrypoints restrict to GM;
remove that restriction only after validating each requested mode, empty-rank
behavior, global loss normalization, state resets and resumption. Memory-only
requires its own implementation. Keep such changes separate from the historical
snapshot, and never modify the files used by live jobs.

## Environment

The validated GM runtime uses Python3.10, torch2.0.1+cu118,
torchvision0.15.2+cu118, numpy1.26.3, scipy1.13.1, einops0.7.0,
custom BiMamba mamba-ssm1.1.1, causal-conv1d1.4.0. The gCSTR-only torch201
Conda environment is different (torch2.1.0+cu118). Do not conflate them.

Use environment/versions.json and the supplied custom Mamba source. The prior
build targets sm_86 for RTX3090; compile/verify for the target hardware and
matching PyTorch/CUDA ABI. An upstream package with the same version string
is not sufficient evidence of compatibility. Check strict checkpoint loading,
CUDA forward/backward, dataset coverage and DDP smoke before launching.

Open3D is not needed for the current Gesture GM runtime. Historical preprocessing
and raw-format tools may require additional dependencies; consult each dataset's
README rather than installing every preprocessing tool in the training runtime.
