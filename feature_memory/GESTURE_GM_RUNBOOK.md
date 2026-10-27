# DVS Gesture GM preparation and checkpoint audit (2026-10-27)

## Decision and important class mismatch

Use existing official Train/Test and actions 1..10 (exclude official class 11 Other).
Transfer the uploaded EventMamba V2 encoder, **reset only classifier.4**, and train
that fresh 10-class output layer at the new-module LR (1e-4). Keep all preceding
point parameters exactly pretrained, with LR 1e-5. This is transfer learning,
not an exact resumption of the uploaded classifier. Its optimizer is not loaded.

The upload is internally consistent: epoch=69 (human epoch70), sequence accuracy
0.9962121212, window accuracy97.09%, 168 finite state tensors. Text and TensorBoard
agree. Strict loading requires V2 embedding24, stage channels48/96/192,
groups512/256/128, K24, feature-space neighbors. THU V1 is incompatible.
Reference: https://github.com/rhwxmx/EventMamba/tree/7f61fd397cf69f10f0f92a182109d36cd7b6fdbb

**The ten source classifier outputs are not our ten target classes.** Evidence:
SpikingJelly 0.0.0.0.14 creates folders0..10 and inherits torchvision DatasetFolder
without overriding class enumeration; alphabetical ordering is0,1,10,2,..9.
EventMamba's script skips loader label10, thus excluding original label9 (Air
guitar), while retaining original label10 (Other) as output2. Uploaded training
package/version is not provided, so this mapping is a strong inference from
source code AND actual Train predictions, not claimed historical metadata.
Train diagnostics: 72/72 windows from the nine shared classes correct under this
mapping; 8/8 raw Train Other examples produce output2. Coordinate KNN did not fix
the mismatch; preserve documented V2 feature KNN. No axis transforms or class
selection were tuned on Test.

Raw uploaded classifier on our full target Test scored63/264 (23.8636%), revealing
the mismatch. This diagnostic is not GM performance. Its reported99.6212% is not
a comparable baseline for our current ten-class task. A class permutation alone
cannot recover the missing Air-guitar class. The target protocol is unchanged.

## Data and alignment

Store: /remote-home/datasets/DVSGesture-prepared/gesture_prepared_v2
Official Train98 recordings/1078 annotated segments/26816 windows; Test24
recordings/264 segments/6959 windows. Train subjects01..23, Test24..29.
All source lists, CSV labels, segment bounds and every stored window index have
been independently checked against raw cached events. No invalid windows or
missing target segments. Sequence unit is one annotated gesture occurrence,
not an entire multi-label recording. Memory resets at each sequence boundary.

Windows500ms, stride250ms, matching EventMamba's strict end<last and excluded
last <=end event index. No new tail windows or class changes. Shared per-sequence
rotation/crop plus per-window jitter acts on full events; frames use all these
events and points are then sampled from the SAME augmented events. Sample1024,
sort indices, float32 sampled t minmax with epsilon1e-6, x/128,y/128. No Open3D
filter. Test has no spatial augmentation. Deterministic per-window sampling;
original pretrained H5 points and exact original training membership not uploaded.

## Model and optimization

Stagewise Gate+Memory, same fusion/memory core as UCF and THU. Three pretrained
Gesture gCSTR ResNet18 encoders frozen, point BN statistics frozen but affine
trainable. Fusion starts random and gate/memory residuals start zero, matching
THU initialization. UCF originally started from trained Fusion A, so initial
fusion provenance is not identical.

Image sources: gcstr_run1/{cstr3,xt_p,ty_p}/best_loss.pt, epochs11/32/39, exactly
as the earlier THU branch-loading policy. These are legacy NLL-selected files.
Historical Top1 winners27/5/7 (some ties) were not retained; last.pt does not
recover them. We do not claim these source weights are historical Top1 winners.

Formal config: configs/gesture_gm_ddp3.json. GPUs0,1,2; per-GPU video batch16,
global48; TBPTT4;50 epochs;seed0;AdamW;weight decay1e-4;clip1;FP32;no scheduler,
no early stopping;point aux0.5. Pretrained point LR1e-5, fusion5e-5, new modules
(including reset output head)1e-4. Two loader workers/rank.

**Selection: historical highest video-level Test Top1**, soft-vote probabilities
across all windows in each segment; ties retain earliest. Native best_top1.pt,
last.pt, predictions and resume identities are saved. NLL is only reported.
This retains the user's Test-as-Val policy; there is no independent held-out
selection set.

## Validation artifacts

All artifacts under /remote-home/datasets/DVSGesture-prepared:
- gesture_gm_protocol_audit.json: complete raw-boundary/list/label audit.
- eventmamba_pretrained/uploaded_20261027/{checkpoint_audit,tensorboard_audit}.json
- gesture_checkpoint_label_diagnostic.json and gesture_label_alignment_audit.json
- gesture_point_test_audit.json: unadapted-head diagnostic only.
- gesture_gm_cpu_tests_v2.log: strict load/feature parity, full coverage, exact
  preparation sampling, shared augmentation, frozen modules, preserved backbone
  versus reset head, unique optimizer parameter coverage.
- gm_core_regression.log:12 existing GM tests.
- gm_ddp_numerical_test.log:four-rank gradients match global loss, including empty
  rank, ragged tails, state across chunks and exact11-window coverage.
- gm_smoke_label_corrected_v2:single-GPU Train-only69-window/3-segment smoke.
- gm_ddp3_smoke_label_corrected_v2:three-GPU per16 Train-only1288-window/49-segment
  smoke; epoch-boundary resume through epoch2 also passed. Peak allocated memory
  approximately3725MiB/rank (reserved4064MiB). Frozen image weights/buffers remained
  bit-exact; all saved model tensors finite; best_top1.pt matched the logged maximum.
  Smoke results are not scientific Test scores.

Earlier gm_smoke_v1 and gm_ddp3_smoke_v1 are pre-label-fix diagnostics and must not
be resumed into a formal run. No existing UCF/THU/Daily training files modified.
Concurrent smoke uses extra compute briefly; it does not stop ongoing jobs.

## Prepared formal run (not launched)

Frozen code and config:
/remote-home/datasets/DVSGesture-prepared/gm_ddp3_batch48_v1/code
Manifest: ../code_sha256.json. State: ../planned_run.json.

Launch after confirming the audit results:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/remote-home/datasets/UCF101-feature-memory-runs/venv/bin/python -u \
-m torch.distributed.run --standalone --nproc_per_node=3 \
/remote-home/datasets/DVSGesture-prepared/gm_ddp3_batch48_v1/code/train_gesture_gm_ddp.py \
--config /remote-home/datasets/DVSGesture-prepared/gm_ddp3_batch48_v1/code/configs/gesture_gm_ddp3.json \
--output /remote-home/datasets/DVSGesture-prepared/gm_ddp3_batch48_v1/training
```
Use a persistent session or detached subprocess and redirect logs to datasets.
For resumption use identical snapshot/config plus --resume. Smoke and formal
outputs are separate; do not migrate smoke weights.

## Authorized two-GPU run

User updated the plan to GPUs0,1, per-GPU video batch32/global64. All other
settings stay as audited, including the reset point output head and Top1 selection.
Run root: /remote-home/datasets/DVSGesture-prepared/gm_ddp2_batch64_v1.
Frozen config: code/configs/gesture_gm_ddp2.json.
The separate smoke uses65 complete Train segments; formal training starts from
pretrained encoders and a fresh output head, never from smoke weights.
Formal log: train.log. Launch PID/command: launch.json.
