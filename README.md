# Event Vision GM Experiments

Research code for full-event gCSTR representations, EventMamba feature fusion,
and Gate/Memory ablations. Imported from the experiment workspace on 2026-10-27.

## Contents

- `feature_memory/`: fusion, Gate/Memory, sequence scheduling, dataset adapters,
  training/evaluation, tests and runbooks.
- `thu_preprocessing/`: THU-CHL, DVS Gesture, DailyAction and DailyDVS-200
  preprocessing and gCSTR branch training/evaluation.
- `dailydvs200_preprocessing/`: official split copies, audit and DDP migration code.
- `audit_gcstr/`: prior split/runtime audit code and reports.
- `experiments/`: immutable copies of the actual UCF101/THU/Gesture GM run configs.
- `environment/mamba_build/`: the actual custom BiMamba source used by the runs,
  including the causal-conv1d interface adaptation. Preserve its upstream license.
- `provenance/import_manifest.json`: source paths and SHA-256 for imported files.
- `docs/MIGRATION.md`: external assets, protocol differences and ablation readiness.

Dataset events, point caches, checkpoints, virtual environments and generated
training outputs are intentionally external. Their paths in the configs are
server-specific and must be mapped explicitly on another machine. The import
manifest describes the initial snapshot, not the hash of every future revision.

## Important protocol differences

- UCF101 GM initializes from the pretrained Fusion A `epoch_040.pt`.
- THU GM uses a pretrained EventMamba V1 plus independent gCSTR branches, with
  random fusion initialization.
- Gesture GM uses EventMamba V2 (24 -> 48/96/192). The uploaded classifier's class
  semantics differ from the target ten actions. Preserve its encoder, reset the
  final classifier, and retain the target labels. See `feature_memory/GESTURE_GM_RUNBOOK.md`.
- Newly configured experiments select the highest video-level Test Top-1. Older
  THU/UCF training entrypoints still save NLL winners and need explicit adaptation
  before using them for new Top-1-selected ablations. Historical config copies
  remain unchanged for provenance.
- Model modes B/G/GM exist, but current DDP entrypoints still assert GM. A pure
  Memory-only mode is not implemented. Do not assume all ablations are runnable
  by changing one config field.

## Git workflow

Use this checkout for future development, one branch per change or ablation.
Keep long-running experiments on their frozen code snapshots outside the checkout.
Review `git diff` and run relevant tests before committing; push through an
appropriately authorized Git client or the connected GitHub plugin. No GitHub
credentials are stored here. Local command-line authentication is separate from
GitHub plugin authorization.

```bash
git switch -c experiment/your-ablation
# edit code/configs and perform the relevant checks
python tools/check_repository.py
git diff --stat
git add <reviewed-files>
git commit -m "Describe the experiment or change"
git push -u origin HEAD
```

The original live workspace is separate. Changes made there are not automatically
tracked in this checkout. `tools/sync_workspace.py --source /path/to/workspace`
previews updates to existing imported project files; add `--apply` to copy them.
It does not delete files, import new files, overwrite environment/run snapshots,
or push commits. Add new source files deliberately and review the diff.

## Licensing

Third-party licenses are retained under their corresponding source directories,
including EventMamba and the custom Mamba build. No blanket license is asserted
for the original research code in this initial import.
