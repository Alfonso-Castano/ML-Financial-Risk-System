---
phase: 02-ml-model-training-pipeline
plan: 03
subsystem: ml
tags: [pytorch, scikit-learn, StandardScaler, DataLoader, tqdm, early-stopping, training-loop]

# Dependency graph
requires:
  - phase: 02-01
    provides: build_feature_matrix returning (X, y, profile_ids) numpy arrays
  - phase: 02-02
    provides: FinancialRiskModel and FinancialDataset classes
provides:
  - backend/ml/train.py - runnable training script (python -m backend.ml.train)
  - models/latest_model.pth - trained model weights (best val-loss checkpoint)
  - models/scaler_stats.json - StandardScaler mean/scale for 9 features (Phase 3 inference)
  - models/training_history.json - train/val loss history for evaluate.py
affects:
  - 02-04-evaluate (reads training_history.json and latest_model.pth)
  - 03-api-layer (loads latest_model.pth and scaler_stats.json for inference)

# Tech tracking
tech-stack:
  added: [matplotlib>=3.0, tqdm>=4.0, sklearn.model_selection.train_test_split, sklearn.preprocessing.StandardScaler]
  patterns:
    - Profile-level stratified splitting via index arrays (not row splitting)
    - Scaler fit on train only, transform applied to all splits
    - Best-val-loss checkpoint saving with early stopping patience counter
    - tqdm context manager for progress bar with set_postfix for live loss display
    - squeeze(-1) instead of squeeze() to handle single-sample last-batch edge case

key-files:
  created:
    - backend/ml/train.py
    - models/latest_model.pth
    - models/scaler_stats.json
    - models/training_history.json
  modified:
    - backend/config/settings.py (added training hyperparameters and file paths)
    - requirements.txt (added all 6 dependencies)

key-decisions:
  - "squeeze(-1) used instead of squeeze() to prevent shape mismatch on last batch when batch size=1"
  - "training_history.json saved so evaluate.py can generate loss plots without re-running training"
  - "models/ directory created at runtime via Path.mkdir(parents=True, exist_ok=True)"

patterns-established:
  - "Split by profile index with stratify=y preserves class balance across all three splits"
  - "scaler_stats.json format: {feature_names, mean, scale} - used by Phase 3 predictor"

# Metrics
duration: 2min
completed: 2026-02-23
---

# Phase 2 Plan 03: Training Pipeline Summary

**End-to-end training pipeline using stratified profile splitting, StandardScaler fitted on train only, tqdm progress bar, early stopping at epoch 62/75, best val loss 0.0329**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-23T07:35:19Z
- **Completed:** 2026-02-23T07:37:37Z
- **Tasks:** 2
- **Files modified:** 4 (settings.py, requirements.txt, train.py created, models/ artifacts)

## Accomplishments

- Full training pipeline runs end-to-end: `python -m backend.ml.train` loads CSV, engineers 9 features, splits 3000 profiles into train=2101/val=449/test=450, scales, trains, and saves best model
- StandardScaler fit on train data only (no data leakage), scaler statistics persisted to `models/scaler_stats.json` with all 9 feature means and scales for Phase 3 API inference
- Training converged well: val loss dropped from ~0.49 to 0.0329 over 62 epochs before early stopping triggered, model checkpoint saved at best validation performance

## Task Commits

Each task was committed atomically:

1. **Task 1: Add training config and install matplotlib** - `c345949` (chore)
2. **Task 2: Implement training pipeline in train.py** - `4dcfeae` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `backend/ml/train.py` - Complete training pipeline with train_epoch, validate_epoch, save_scaler_stats, main()
- `backend/config/settings.py` - Added training hyperparameters (NUM_EPOCHS=75, BATCH_SIZE=64, LEARNING_RATE=0.001, EARLY_STOPPING_PATIENCE=10) and file paths
- `requirements.txt` - All 6 dependencies: numpy, pandas, torch, scikit-learn, matplotlib, tqdm
- `models/latest_model.pth` - Trained model weights (best val-loss checkpoint, epoch ~49 of 62)
- `models/scaler_stats.json` - StandardScaler mean/scale arrays of length 9 for each feature
- `models/training_history.json` - train/val loss history for standalone evaluate.py use

## Decisions Made

- `squeeze(-1)` used instead of `squeeze()` to prevent shape mismatch when last batch contains exactly 1 sample (PyTorch squeeze() collapses all size-1 dims, turning shape (1,) to scalar)
- `training_history.json` written from `__main__` block so evaluate.py can generate loss curve plots independently without re-running training (per locked decision)
- `num_workers=0` in DataLoaders for Windows compatibility (multiprocessing issues on Windows with PyTorch)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed squeeze(-1) instead of squeeze() for last-batch shape mismatch**
- **Found during:** Task 2 (first training run)
- **Issue:** When the last batch has exactly 1 sample, `model(features).squeeze()` collapses shape from `(1,)` to `()` (scalar), which mismatches label shape `(1,)`, causing `ValueError: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([]))`
- **Fix:** Changed `.squeeze()` to `.squeeze(-1)` in both `train_epoch` and `validate_epoch` - only removes the last dimension, preserving correct shape for all batch sizes including 1
- **Files modified:** `backend/ml/train.py`
- **Verification:** Training ran end-to-end to completion without errors
- **Committed in:** `4dcfeae` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential correctness fix. No scope creep. Same file, same implementation intent.

## Issues Encountered

- PyTorch squeeze() edge case on final batch with 1 sample - resolved via squeeze(-1) (see Deviations above)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `models/latest_model.pth` exists and is loadable (confirmed by load_state_dict call at end of main)
- `models/scaler_stats.json` has 9 feature means and scales, ready for Phase 3 predictor
- `models/training_history.json` has train/val loss arrays, ready for 02-04 evaluate.py
- Training script runnable via `python -m backend.ml.train` from project root
- Ready to execute 02-04 (evaluate.py - metrics, confusion matrix, loss curves)

---
*Phase: 02-ml-model-training-pipeline*
*Completed: 2026-02-23*
