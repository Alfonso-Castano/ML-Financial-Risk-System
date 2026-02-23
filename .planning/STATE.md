# Project State

## Current Status

**Phase**: 02 - ML Model & Training
**Current Plan**: Phase 2 complete (4/4 plans done)
**Last Updated**: 2026-02-23

## Progress

| Phase | Status | Progress |
|-------|--------|----------|
| 1: Foundation & Synthetic Data | ✓ Complete | 100% (1/1 plans) |
| 2: ML Model & Training | ✓ Complete | 100% (4/4 plans complete) |
| 3: API Layer & Orchestration | ○ Pending | 0% |
| 4: Frontend Dashboard | ○ Pending | 0% |
| 5: Deployment & Documentation | ○ Pending | 0% |

**Legend**: ○ Pending | ◆ In Progress | ✓ Complete

## Project Context

**Core Value**: Learn how to integrate ML models into applications with clean architecture
**Current Focus**: Phase 2 complete - all ML artifacts ready. Begin Phase 3 (API layer).

See `.planning/PROJECT.md` for full architecture and constraints.

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files | Completed |
|-------|------|----------|-------|-------|-----------|
| 01 | 01 | 3min 22sec | 2 | 12 | 2026-02-17 |
| 02 | 01 | 1min | 1 | 1 | 2026-02-23 |
| 02 | 02 | 4min | 1 | 2 | 2026-02-23 |
| 02 | 03 | 2min | 2 | 4 | 2026-02-23 |
| 02 | 04 | 5min | 2 | 5 | 2026-02-23 |

## Recent Decisions

- 2026-02-23: evaluate.py runs standalone (python -m backend.ml.evaluate), not called by train.py - architectural isolation decision
- 2026-02-23: matplotlib.use('Agg') placed before pyplot import for headless rendering on all platforms
- 2026-02-23: All sklearn metrics wrapped with float() to prevent numpy scalar JSON serialization errors
- 2026-02-23: Test set reconstructed deterministically in evaluate.py using same seed/ratios as train.py
- 2026-02-23: squeeze(-1) used instead of squeeze() to handle single-sample last-batch shape mismatch in BCELoss
- 2026-02-23: training_history.json saved at end of __main__ so evaluate.py can generate loss plots independently
- 2026-02-23: num_workers=0 in DataLoaders for Windows compatibility with PyTorch multiprocessing
- 2026-02-23: Feature set locked to 9: avg_income, avg_expenses, final_savings, debt_payment, credit_score, debt_ratio, liquidity_ratio, net_cash_flow, consec_negative_months
- 2026-02-23: debt_ratio denominator is free cash flow (income - expenses), not gross income
- 2026-02-23: Named layer attributes (fc1, relu1, dropout1...) over nn.Sequential for educational clarity
- 2026-02-23: Scaling is external to FinancialDataset - StandardScaler fits in train.py to prevent data leakage
- 2026-02-23: input_size parameterized (default=9) so model is not hard-coded to feature count
- 2026-02-17: Used 4 financial archetypes (struggling, getting_by, stable, comfortable) matching user's salary bands
- 2026-02-17: Calibrated archetype parameters iteratively to achieve 36.2% stress ratio (target: ~35%)
- 2026-02-17: Applied long format (one row per person-month) for time series compatibility
- 2026-02-17: Fixed debt payment per profile, variable income/expenses with 15% monthly variance
- 2026-02-13: Project initialized with minimal GSD structure
- 2026-02-13: Architecture locked (no service layer, no database, vanilla JS)
- 2026-02-13: Complexity guardrails added to CLAUDE.md

## Next Action

**Begin Phase 3**: API Layer & Orchestration

**Phase 2 artifacts ready for Phase 3:**
- `models/latest_model.pth` - trained model weights (62 epochs, recall=0.9448)
- `models/scaler_stats.json` - feature scaling statistics (mean/scale for 9 features)
- `models/metrics.json` - evaluation results (recall=0.9448, ROC-AUC=0.9983)
- `backend/ml/model.py` - FinancialRiskModel class
- `backend/ml/evaluate.py` - compute_metrics, run_evaluation (reusable in predictor)
- `backend/data/feature_engineering.py` - build_feature_matrix, FEATURE_NAMES

**Architecture for Phase 3:**
- `backend/api/routes.py` - thin API routes (FastAPI)
- `backend/api/schemas.py` - Pydantic request/response models
- `backend/ml/predictor.py` - orchestration: load model, scale input, predict

## Last Session

**Session timestamp:** 2026-02-23T07:41:26Z
**Stopped at:** Completed 02-ml-model-training-pipeline/02-04-PLAN.md
**Status:** Phase 2 fully complete - evaluate.py built, recall=0.9448, all plots generated

---

*Updated: 2026-02-23 after Phase 2 Plan 04 completion*
